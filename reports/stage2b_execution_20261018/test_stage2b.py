"""CPU software fixtures only, never physical/model-accuracy evidence."""
import contextlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import run_stage2b as m
from adapter import PostSmoothingAdapter,gain_at


class Boundary:
    velocityType='fixed'
    def __init__(self):self.velocity=np.zeros((1,2,1,2),np.float32)
    def setVelocity(self,x):self.velocity=x.copy()
    def getVelocity(self,_):return self.velocity.copy()
    def GetFluxes(self):return self.velocity[:,1:2].copy()
    def hasTransform(self):return False


class Raw:
    def __init__(self):
        self._top_block_idx=0;self._bottom_block_idx=1;self.episode_length=80;self.mode=SimpleNamespace(value='val');self.audit=None
    def reset(self,**kw):
        self._n_steps=0;self._CylinderEnvBase__last_control=np.zeros(1,np.float32)
        self._top_boundary=Boundary();self._bottom_boundary=Boundary()
        self._top_velocity=np.array([[[[0.,0.]],[[1.,1.]]]],np.float32);self._bottom_velocity=self._top_velocity.copy()
        blocks=[SimpleNamespace(getBoundary=lambda face:self._top_boundary),SimpleNamespace(getBoundary=lambda face:self._bottom_boundary)]
        self._domain=SimpleNamespace(getBlock=lambda i:blocks[i],GetBoundaryFluxBalance=lambda:np.array([0.]))
        self._sim=SimpleNamespace(total_step=1,total_time=1.)
        self._apply_action(np.zeros(1,np.float32))
    def _apply_action(self,u):
        self._top_boundary.setVelocity(self._top_velocity*u);self._bottom_boundary.setVelocity(self._bottom_velocity*u)
    def step(self,a):
        for j in range(25):
            self._CylinderEnvBase__last_control+=np.float32(.1)*(a-self._CylinderEnvBase__last_control)
            self._apply_action(self._CylinderEnvBase__last_control)
            before=m.s1.clock(self);self._sim.total_step+=1;self._sim.total_time+=.01
            if self.audit:self.audit.observe(j,before)
        self._n_steps+=1


class AdapterTests(unittest.TestCase):
    def test_eighty_steps_and_transition40_no_memory_reset(self):
        for condition in ('S0','S1','S2'):
            raw,bare=Raw(),Raw();adapter=PostSmoothingAdapter(raw,condition);raw.reset();bare.reset()
            self.assertEqual(adapter.gain,1.)
            for t in range(80):
                action=np.array([.2 if t%3 else -.2],np.float32);raw.step(action);bare.step(action)
                np.testing.assert_array_equal(m.s1.memory(raw),m.s1.memory(bare))
                g=.5 if condition=='S1' or (condition=='S2' and t>=40) else 1.
                np.testing.assert_array_equal(raw._top_boundary.velocity,bare._top_boundary.velocity*g)
                self.assertEqual(adapter.arguments[0]['gain'],g)
            with self.assertRaises(RuntimeError):raw.step(action)
            raw.reset();self.assertEqual(adapter.t,0);self.assertEqual(adapter.gain,1.)
            adapter.close();self.assertNotIn('_apply_action',vars(raw))
    def test_no_extra_schedule(self):
        for c,t in [('J',0),('S2',80),('S0',-1)]:
            with self.assertRaises(RuntimeError):gain_at(c,t)


class ValidationTests(unittest.TestCase):
    def test_prefix_tolerance_and_relative_reporting(self):
        with tempfile.TemporaryDirectory() as d:
            p=m.Prefix(Path(d));p.value('small',[0.],[2e-7]);self.assertEqual(p.rows[0]['max_relative'],2e5)
            p.value('large',[0.],[.01])
            self.assertFalse(json.loads((Path(d)/'prefix_differences.json').read_text())[-1]['allclose'])
    def test_rng_and_warm_state_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=m.Prefix(Path(d))
            p.identity('planner_rng',{'torch':'A'},{'torch':'B'})
            p.value('warm',[.2],[.3])
    def test_segment_reward_and_absolute_interval_lift(self):
        rows=[dict(t=t,reward=float(t),drag=3.,lift=(-1)**t*2.,requested_action=[1.2 if t==0 else 1.],clipped_action=[1.]) for t in range(80)]
        s=m.summaries(rows)
        self.assertEqual(s['whole']['mean_signed_lift'],0.);self.assertEqual(s['whole']['mean_abs_control_interval_lift'],2.)
        self.assertEqual(s['pre40']['reward_sum'],sum(range(40)));self.assertEqual(s['post40']['reward_sum'],sum(range(40,80)))
        self.assertEqual(s['whole']['clipping_frequency'],1/80);self.assertEqual(s['whole']['saturation_frequency'],1.)
    def test_counter_offset_excluded(self):
        o=object.__new__(m.CounterObserver);o.control_calls=o.reset_calls=o.control_accepted=o.reset_accepted=0;o.initial_offset=None
        with patch.object(m.s1.NumericalObserver,'emit',lambda *a:None):
            o.emit(dict(status='entered',clock_before={'solver_steps':7}))
            o.emit(dict(status='success',phase='reset',clock_before={'solver_steps':7},clock_after={'solver_steps':10}))
            o.emit(dict(status='success',phase='control',clock_before={'solver_steps':10},clock_after={'solver_steps':12}))
        self.assertEqual(o.initial_offset,7);self.assertEqual(o.reset_accepted,3);self.assertEqual(o.control_accepted,2)


class FakeAudit:
    def __init__(self,raw,adapter):
        self.raw,self.adapter=raw,adapter;raw.audit=self
        self.calls=self.control_calls=self.reset_calls=self.control_accepted=self.reset_accepted=self.failed=0;self.initial_offset=1;self.active=False;self.outer=[]
    def arm_profiles(self):self.profiles={'top':self.raw._top_velocity.copy(),'bottom':self.raw._bottom_velocity.copy()}
    def begin(self,a):self.active=True;self.outer=[]
    def observe(self,j,before):
        self.calls+=1;self.control_calls+=1;self.control_accepted+=1
        boundaries={side:{'velocity_measured':getattr(self.raw,'_'+side+'_boundary').velocity.copy(),'flux_measured':getattr(self.raw,'_'+side+'_boundary').GetFluxes()} for side in ('top','bottom')}
        self.outer.append(dict(native_success=True,native_fields_finite=True,status='success',gain=self.adapter.gain,transition=self.raw._n_steps,outer=j,
                               smoothed_memory_measured=m.s1.memory(self.raw),boundary=boundaries,clock_before=before,clock_after=m.s1.clock(self.raw)))
    def close(self):pass


class RunnerTests(unittest.TestCase):
    def run_fixture(self,root,fail_prefix=False,fail_weights=False):
        created=[];current={}
        for name in ('STAGE2B_EXPERIMENT_CONTRACT.md','package_hashes.json'):(root/name).write_text('{}')
        fake_torch=SimpleNamespace(manual_seed=lambda _:None,cuda=SimpleNamespace(manual_seed_all=lambda _:None))
        def create_agent():
            agent=SimpleNamespace();policy=SimpleNamespace(reset=lambda seed:current.update(planner_seed=seed));return agent,policy,'frozen'
        @contextlib.contextmanager
        def factory(repo):yield create_agent,{}
        def environment(c,d):
            raw=Raw();adapter=PostSmoothingAdapter(raw,c);audit=FakeAudit(raw,adapter);created.append(c);current['condition']=c
            def obs():
                v=np.full((151,2),float(raw._top_boundary.velocity.sum()),np.float32)
                if fail_prefix and c=='S2' and raw._n_steps==1:v+=.1
                return v.ravel().copy(),dict(original_velocity=v,original_pressure=np.zeros(151,np.float32))
            def reset(**kw):raw.reset(**kw);return obs()
            def step(a):
                raw.step(np.clip(a,-1,1));o,info=obs();info.update(drag=3.,lift=float(o[0]));return o,-abs(float(o[0])),False,raw._n_steps==80,info
            return SimpleNamespace(reset=reset,step=step,close=lambda:None),raw,SimpleNamespace(identity=lambda:dict(split='val',index=8)),adapter,audit
        def snapshot(raw,path):
            np.savez(path.with_suffix('.npz'),memory=m.s1.memory(raw),boundary=raw._top_boundary.velocity)
            m.s1.write(path.with_suffix('.json'),{'clock':m.s1.clock(raw),'rng':{'planner_seed':current['planner_seed']}})
        def planner(agent,policy,observation,t,directory):
            a=np.array([.1+.01*np.sin(t)+.01*observation[0]],np.float32)
            np.savez(directory/f'planner_{t:02d}.npz',warm_before=np.array([t]),warm_after=np.array([t+1]),requested_action=a,clipped_action=a)
            rec={'t0':t==0,'rng_before':{'seed':current['planner_seed'],'t':t},'rng_after':{'seed':current['planner_seed'],'t':t+1},'warm_before_present':t>0}
            m.s1.write(directory/f'planner_{t:02d}.json',rec);current['t']=t;return a,rec
        def invariant(*a):
            if fail_weights and current.get('t')==2:raise RuntimeError('frozen online/target weights changed')
        with patch.dict(sys.modules,torch=fake_torch),patch.object(m,'prerequisite',lambda *a:None),\
             patch.object(m.s1,'provenance',lambda *a:{}),patch.object(m.frozen,'factory',factory),\
             patch.object(m,'environment',environment),patch.object(m.s1,'snapshot',snapshot),\
             patch.object(m.p1,'effective',lambda raw:dict(smoothing=.1,lift_penalty=1.)),\
             patch.object(m.p1,'seed_policy',lambda policy,seed:policy.reset(seed)),\
             patch.object(m.p1,'rng',lambda:{}),patch.object(m.p1,'rng_hash',lambda x:{}),\
             patch.object(m.frozen,'planner_capture',planner),patch.object(m.frozen,'invariant',invariant),\
             patch.object(m.frozen,'weight_hash',lambda a:'frozen'),patch.object(m.p1,'protect',lambda *a:None):
            if fail_weights:
                with self.assertRaises(RuntimeError):m.run(root,root,root)
            else:m.run(root,root,root)
        return created
    def test_exact_960_and_four_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);created=self.run_fixture(p);cost=json.loads((p/'cost.json').read_text())
            self.assertEqual(len(created),12);self.assertEqual(cost['completed_controls'],960);self.assertEqual(cost['attempted_controls'],960)
            self.assertEqual(cost['control_outer_calls'],24000);self.assertEqual(cost['control_accepted_increments'],24000)
            self.assertEqual(len(json.loads((p/'paired_outcomes.json').read_text())),4)
    def test_prefix_disagreement_does_not_stop_new_pilot(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);created=self.run_fixture(p,fail_prefix=True);cost=json.loads((p/'cost.json').read_text())
            self.assertEqual(len(created),12);self.assertEqual(cost['completed_controls'],960)
            self.assertTrue((p/'e0_p0_S2/shift_gate.json').exists())
    def test_weight_failure_before_physical_action(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);created=self.run_fixture(p,fail_weights=True);cost=json.loads((p/'cost.json').read_text())
            self.assertEqual(created,['S0']);self.assertEqual(cost['attempted_controls'],2)


class FrozenAndWatchdogTests(unittest.TestCase):
    def test_online_and_target_hash_and_config_guards(self):
        import copy
        online=np.array([1.],np.float32);target=np.array([2.],np.float32)
        a=SimpleNamespace(model=SimpleNamespace(state_dict=lambda:{'x':online},training=False),
            model_target=SimpleNamespace(state_dict=lambda:{'x':target},training=False),
            cfg=SimpleNamespace(**copy.deepcopy(m.frozen.REFERENCE['parameters'])),std=m.frozen.REFERENCE['planner_std'],
            optim=SimpleNamespace(state={}),pi_optim=SimpleNamespace(state={}))
        baseline=m.frozen.weight_hash(a);m.frozen.invariant(a,baseline)
        target[0]=3
        with self.assertRaisesRegex(RuntimeError,'weights changed'):m.frozen.invariant(a,baseline)
        target[0]=2;a.cfg.num_samples+=1
        with self.assertRaisesRegex(RuntimeError,'configuration'):m.frozen.invariant(a,baseline)
        a.cfg.num_samples-=1;a.optim.state['step']=1
        with self.assertRaisesRegex(RuntimeError,'optimizer'):m.frozen.invariant(a,baseline)
    def test_watchdog_exact_2700_no_retry(self):
        import launch,subprocess
        now=[0.]
        class Child:
            pid=123
            def wait(self,timeout=None):
                if timeout is None:return -9
                now[0]+=timeout;raise subprocess.TimeoutExpired('fixture',timeout)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'package_hashes.json').write_text('{}');(root/'STAGE2B_EXPERIMENT_CONTRACT.md').write_text('fixture')
            with patch.object(launch,'HERE',root),patch.object(launch.time,'monotonic',lambda:now[0]),patch.object(launch.subprocess,'Popen',return_value=Child()) as spawn,patch.object(launch.os,'killpg') as kill,patch('builtins.print'),patch.object(sys,'argv',['launch.py','--repo',str(root),'--stage1',str(root),'--execute']):
                with self.assertRaises(SystemExit):launch.main()
            self.assertEqual(now[0],2700);self.assertEqual(spawn.call_count,1);self.assertEqual(kill.call_count,1)
            status=json.loads(next((root/'reports').glob('*/launcher_status.json')).read_text());self.assertEqual(status['status'],'watchdog_timeout')


if __name__=='__main__':unittest.main(verbosity=2)
