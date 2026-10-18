"""CPU test doubles verify software gates only. No fluid/physical evidence."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from adapter import PostSmoothingAdapter, gain_at
import pilot


class Boundary:
    def __init__(self):
        self.velocity=np.zeros((1,2,1,2),np.float32)
        self.velocityType='fixed'
    def setVelocity(self,x): self.velocity=x.copy()
    def getVelocity(self, computational):return self.velocity.copy()
    def GetFluxes(self):return self.velocity[:,1:2].copy()
    def hasTransform(self):return False


class Block:
    def __init__(self,b):self.b=b;self.velocity=np.zeros(2,np.float32);self.pressure=np.zeros(2,np.float32)
    def getBoundary(self,face):return self.b


class Domain:
    def __init__(self,top,bottom):self.blocks=[Block(top),Block(bottom)]
    def getBlock(self,i):return self.blocks[i]
    def getBlocks(self):return self.blocks
    def GetBoundaryFluxBalance(self):return np.array([0.],np.float32)


class Sim:
    def __init__(self):self.total_time=0.;self.total_step=0;self.flux_balance_tol=np.array([1e-5])
    def single_step(self):self.total_time+=.01;self.total_step+=1;return True


class Raw:
    def __init__(self):self._n_steps=0;self.sent=[];self._top_block_idx=0;self._bottom_block_idx=1
    def _get_simulation(self,*a):return Sim()
    def reset(self,*a,**kw):
        self._CylinderEnvBase__last_control=np.zeros(1,np.float32);self._n_steps=0
        self._top_boundary,self._bottom_boundary=Boundary(),Boundary()
        self._domain=Domain(self._top_boundary,self._bottom_boundary)
        self._top_velocity=np.array([[[[0.,0.]],[[1.,1.]]]],np.float32)
        self._bottom_velocity=self._top_velocity.copy()
        self._sim=self._get_simulation()
        self._apply_action(np.zeros(1,np.float32))
        return 'reset',{}
    def _apply_action(self,action):
        self.sent.append(action.copy())
        self._top_boundary.setVelocity(self._top_velocity*action)
        self._bottom_boundary.setVelocity(self._bottom_velocity*action)
    def step(self,action):
        self.last_command=action
        for i in range(25):
            self._CylinderEnvBase__last_control=self._CylinderEnvBase__last_control+np.float32(.1)*(action-self._CylinderEnvBase__last_control)
            self._apply_action(self._CylinderEnvBase__last_control)
            self._sim.single_step()
        self._n_steps+=1
        return None


class AdapterTests(unittest.TestCase):
    def test_gain_one_equals_bare_and_detach_restores_namespace(self):
        bare,hook=Raw(),Raw();adapter=PostSmoothingAdapter(hook,'N-hook')
        bare.reset();hook.reset()
        for a in pilot.ACTIONS:
            command=np.array([a],np.float32);bare.step(command);hook.step(command)
            np.testing.assert_array_equal(bare._top_boundary.velocity,hook._top_boundary.velocity)
            np.testing.assert_array_equal(pilot.memory(bare),pilot.memory(hook))
            self.assertIs(hook.last_command,command)
        adapter.close()
        self.assertNotIn('_apply_action',vars(hook));self.assertNotIn('reset',vars(hook))
        self.assertNotIn('_stage1_adapter_attached',vars(hook))

    def test_jump_preserves_memory_and_scales_after_smoothing(self):
        bare,jump=Raw(),Raw();adapter=PostSmoothingAdapter(jump,'J');bare.reset();jump.reset()
        for t,a in enumerate(pilot.ACTIONS):
            command=np.array([a],np.float32);bare.step(command);jump.step(command)
            np.testing.assert_array_equal(pilot.memory(bare),pilot.memory(jump))
            g=1 if t<6 else .5
            np.testing.assert_array_equal(jump._top_boundary.velocity,bare._top_boundary.velocity*g)
            self.assertEqual(adapter.arguments[0]['gain'],g)
        self.assertNotEqual(float(jump.sent[6*25+1][0]),0.)
        adapter.close()

    def test_reset_gain_one_zero_memory_and_fresh_boundaries(self):
        raw=Raw();adapter=PostSmoothingAdapter(raw,'F');raw.reset();old=raw._top_boundary
        raw.step(np.array([.2],np.float32));raw.reset()
        self.assertIsNot(old,raw._top_boundary);self.assertEqual(adapter.gain,1.);self.assertEqual(adapter.t,0)
        np.testing.assert_array_equal(pilot.memory(raw),[0.]);np.testing.assert_array_equal(old.velocity,old.velocity.copy())
        raw.step(np.array([.2],np.float32));self.assertEqual(adapter.gain,.5);adapter.close()

    def test_isolation_double_attach_and_schedule_limits(self):
        one,two=Raw(),Raw();adapter=PostSmoothingAdapter(one,'J')
        self.assertNotIn('_apply_action',vars(two))
        with self.assertRaisesRegex(RuntimeError,'already'):PostSmoothingAdapter(one,'J')
        with self.assertRaises(RuntimeError):gain_at('J',12)
        with self.assertRaises(RuntimeError):gain_at('other',0)
        adapter.close()

    def test_native_failure_latches_no_retry(self):
        raw=Raw();adapter=PostSmoothingAdapter(raw,'F');raw.reset()
        def fail():raise RuntimeError('native failed')
        raw._sim.single_step=fail
        with self.assertRaisesRegex(RuntimeError,'native failed'):raw.step(np.array([.2],np.float32))
        with self.assertRaisesRegex(RuntimeError,'no retry'):raw.step(np.array([.2],np.float32))
        adapter.close()

    def test_illegal_memory_mutation_rejected(self):
        raw=Raw();orig=raw._apply_action
        def bad(a):orig(a);raw._CylinderEnvBase__last_control+=1
        raw._apply_action=bad;adapter=PostSmoothingAdapter(raw,'F')
        with self.assertRaisesRegex(RuntimeError,'memory'):raw.reset()
        adapter.close();self.assertIs(raw._apply_action,bad)


class ObserverTests(unittest.TestCase):
    def setup_audit(self,path,arm='J'):
        raw=Raw();adapter=PostSmoothingAdapter(raw,arm);audit=pilot.NumericalObserver(raw,path,adapter)
        raw.reset();audit.arm_profiles();return raw,adapter,audit

    def test_actual_boundary_and_flux_readback_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            raw,adapter,audit=self.setup_audit(Path(d))
            for t,a in enumerate(pilot.ACTIONS):
                command=np.array([a],np.float32);audit.begin(command);raw.step(command);audit.active=False
                self.assertEqual(len(audit.outer),25)
                self.assertEqual(audit.outer[0]['gain'],1 if t<6 else .5)
                self.assertIn('velocity_measured',audit.outer[0]['boundary']['top'])
            self.assertEqual(audit.calls,300);audit.close();adapter.close()

    def test_false_solver_return_stops_and_preserves_record(self):
        with tempfile.TemporaryDirectory() as d:
            raw=Raw();original=raw._get_simulation
            def create():
                sim=original();sim.single_step=lambda:False;return sim
            raw._get_simulation=create
            audit=pilot.NumericalObserver(raw,Path(d));raw.reset();audit.arm_profiles();audit.begin(np.array([0.],np.float32))
            with self.assertRaisesRegex(RuntimeError,'returned failure'):raw.step(np.array([0.],np.float32))
            self.assertEqual(audit.calls,1);self.assertEqual(audit.failed,1);audit.close()
            self.assertEqual(json.loads((Path(d)/'solver.jsonl').read_text().splitlines()[-1])['status'],'failed')

    def test_readback_disagreement_is_not_replaced_by_expected(self):
        with tempfile.TemporaryDirectory() as d:
            raw,adapter,audit=self.setup_audit(Path(d));raw._top_boundary.getVelocity=lambda _:np.ones((1,2,1,2),np.float32)
            audit.begin(np.array([.2],np.float32))
            with self.assertRaisesRegex(RuntimeError,'getter disagreement'):raw.step(np.array([.2],np.float32))
            self.assertEqual(raw._sim.total_step,0);audit.close();adapter.close()

    def test_profile_mutation_and_stale_handle_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            raw,adapter,audit=self.setup_audit(Path(d));raw._top_velocity*=2;audit.begin(np.array([.2],np.float32))
            with self.assertRaisesRegex(RuntimeError,'profile tensor mutated'):raw.step(np.array([.2],np.float32))
            audit.close();adapter.close()
        raw=Raw();raw.reset();raw._top_boundary=Boundary()
        with self.assertRaisesRegex(RuntimeError,'stale'):pilot.boundary_readback(raw)

    def test_nonfinite_native_and_flux_fail(self):
        raw=Raw();raw.reset();raw._domain.blocks[0].pressure[0]=np.nan
        with self.assertRaisesRegex(RuntimeError,'nonfinite'):pilot.finite_native(raw)
        raw=Raw();raw.reset();raw._top_boundary.GetFluxes=lambda:np.ones((1,1,1,2),np.float32)
        with self.assertRaisesRegex(RuntimeError,'flux'):pilot.boundary_readback(raw)

    def test_repeat_tolerance_is_fixed_and_differences_saved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);np.savez(p/'a.npz',x=np.array([0.]));np.savez(p/'b.npz',x=np.array([1.]))
            with self.assertRaisesRegex(RuntimeError,'numerical mismatch'):pilot.compare_files(p/'a.npz',p/'b.npz',p,'bad')
            self.assertFalse(json.loads((p/'bad_differences.json').read_text())[0]['allclose'])




# End-to-end runner fixtures deliberately use fake observations, never accuracy evidence.
class RunnerTests(unittest.TestCase):
    def execute(self, root, broken_nominal=False):
        import sys
        fake_torch=SimpleNamespace(manual_seed=lambda _:None,cuda=SimpleNamespace(manual_seed_all=lambda _:None))
        created=[]
        def factory(arm,directory):
            raw=Raw();raw.episode_length=80;raw.mode=SimpleNamespace(value='val')
            adapter=None if arm=='N-bare' else PostSmoothingAdapter(raw,arm)
            audit=pilot.NumericalObserver(raw,directory,adapter)
            def obs():
                v=np.full((151,2),float(raw._top_boundary.velocity.sum()),np.float32)
                if broken_nominal and arm=='N-hook':v+=1
                return v.ravel(),dict(original_velocity=v,original_pressure=np.zeros(151,np.float32))
            def reset(**kw):raw.reset(**kw);return obs()
            def step(a):
                raw.step(a);o,info=obs();info.update(drag=np.array([1.]),lift=np.array([0.]));return o,0.,False,False,info
            env=SimpleNamespace(reset=reset,step=step,close=lambda:None)
            identity=SimpleNamespace(identity=lambda:dict(split='val',index=8,domain_id='CPU fixture'))
            created.append(arm)
            return env,raw,identity,adapter,audit
        def snapshot(raw,path):
            np.savez(path.with_suffix('.npz'),memory=pilot.memory(raw),velocity=raw._top_boundary.velocity)
            pilot.write(path.with_suffix('.json'),dict(rng={},inventory={}))
        with patch.dict(sys.modules,torch=fake_torch),patch.object(pilot,'environment',factory),\
             patch.object(pilot,'provenance',lambda *a:None),patch.object(pilot,'snapshot',snapshot),\
             patch.object(pilot.p1,'effective',lambda raw:dict(smoothing=.1,lift_penalty=1.)),\
             patch.object(pilot.p1,'protect',lambda *a:None):
            if broken_nominal:
                with self.assertRaisesRegex(RuntimeError,'numerical mismatch'):pilot.run(root,root)
            else:pilot.run(root,root)
        return created

    def test_exact_96_budget_and_all_arms(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);created=self.execute(root)
            cost=json.loads((root/'cost.json').read_text())
            self.assertEqual(created,['N-bare','N-bare','N-hook','N-hook','F','F','J','J'])
            self.assertEqual(cost['attempted_controls'],96);self.assertEqual(cost['completed_controls'],96)
            self.assertEqual(cost['outer_calls_including_resets'],2400)
            self.assertEqual(json.loads((root/'worker_status.json').read_text())['status'],'passed')

    def test_nominal_failure_blocks_all_interventions(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);created=self.execute(root,True)
            self.assertEqual(created,['N-bare','N-bare','N-hook'])
            self.assertEqual(json.loads((root/'cost.json').read_text())['attempted_controls'],24)
            self.assertFalse((root/'F_0').exists())

    def test_external_watchdog_kills_once_at_600_without_retry(self):
        import launch,sys,subprocess
        class Child:
            pid=1234
            def wait(self,timeout=None):
                if timeout is None:return -9
                now[0]+=timeout
                raise subprocess.TimeoutExpired('fixture',timeout)
        now=[0.]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'package_hashes.json').write_text('{}')
            with patch.object(launch,'HERE',root),patch.object(launch.time,'monotonic',lambda:now[0]),\
                 patch.object(launch.subprocess,'Popen',return_value=Child()) as popen,\
                 patch.object(launch.os,'killpg') as kill,patch('builtins.print'),\
                 patch.object(sys,'argv',['launch.py','--repo',str(root),'--execute']):
                with self.assertRaises(SystemExit):launch.main()
            self.assertEqual(now[0],600);self.assertEqual(popen.call_count,1);self.assertEqual(kill.call_count,1)
            status=json.loads(next((root/'reports').glob('*/launcher_status.json')).read_text())
            self.assertEqual(status['status'],'watchdog_timeout')

if __name__=='__main__':unittest.main(verbosity=2)
