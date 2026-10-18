"""Stage 2B outcome-focused closed-loop worker; must be supervised by launch.py."""
import gc
import ast
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import p1_support as p1
import stage1_support as s1
from adapter import PostSmoothingAdapter,require,array,gain_at
import frozen

HERE=Path(__file__).resolve().parent
MATRIX=[(e,p,c) for e in (0,1) for p in (0,1) for c in ('S0','S1','S2')]


def prerequisite(path,out):
    expected=p1.read(HERE/'stage1_expected_output_hashes.json')
    package=p1.read(path/'package_hashes.json')
    require(p1.sha(HERE/'stage1_adapter_reference.py')==package['adapter.py'],'verified adapter reference changed')
    require(p1.sha(HERE/'stage1_support.py')==package['pilot.py'],'verified numerical observer source changed')
    def bodies(file):
        tree=ast.parse(file.read_text())
        return [ast.dump(node,include_attributes=False) for node in tree.body if not (isinstance(node,ast.FunctionDef) and node.name=='gain_at')]
    require(bodies(HERE/'adapter.py')==bodies(HERE/'stage1_adapter_reference.py'),'adapter differs outside approved schedule function')
    require(p1.read(path/'output_hashes.json')==expected,'Stage 1 output manifest identity mismatch')
    for name,h in expected.items():require((path/name).is_file() and p1.sha(path/name)==h,'Stage 1 evidence differs: '+name)
    require(p1.read(path/'launcher_status.json')['status']=='passed' and p1.read(path/'worker_status.json')['status']=='passed','Stage 1 did not pass')
    require(p1.read(path/'cost.json')['completed_controls']==96,'Stage 1 incomplete')
    s1.write(out/'stage1_gate.json',{'status':'VERIFIED','path':str(path),'files_verified':len(expected),
                                  'manifest_sha256':p1.sha(path/'output_hashes.json')})


class CounterObserver(s1.NumericalObserver):
    """Count actual per-call differences, not final-counter sums."""
    def __init__(self,*args,**kwargs):
        self.control_calls=self.reset_calls=self.control_accepted=self.reset_accepted=0
        self.initial_offset=None
        super().__init__(*args,**kwargs)
    def emit(self,event):
        super().emit(event)
        if event['status']=='entered' and self.initial_offset is None:self.initial_offset=event['clock_before']['solver_steps']
        if event['status']=='success':
            delta=event['clock_after']['solver_steps']-event['clock_before']['solver_steps']
            require(delta>0,'accepted counter did not advance')
            if event['phase']=='control':self.control_calls+=1;self.control_accepted+=delta
            else:self.reset_calls+=1;self.reset_accepted+=delta


class Prefix:
    def __init__(self,out):self.out=out;self.rows=[]
    def value(self,label,a,b,atol=1e-5,rtol=1e-5):
        a,b=np.asarray(a),np.asarray(b);same=a.shape==b.shape
        finite=same and np.isfinite(a).all() and np.isfinite(b).all()
        diff=np.abs(a.astype(np.float64)-b.astype(np.float64)) if finite else None
        row={'field':label,'shape_equal':same,'allclose':bool(finite and np.allclose(a,b,atol=atol,rtol=rtol)),
             'bitwise_equal':bool(same and np.array_equal(a,b)),
             'max_abs':float(np.max(diff)) if finite and diff.size else (0. if finite else None),
             'max_relative':float(np.max(diff/np.maximum(np.abs(a),1e-12))) if finite and diff.size else (0. if finite else None),
             'relative_denominator_floor':1e-12,'atol':atol,'rtol':rtol}
        self.rows.append(row)
        require(finite, 'invalid/nonfinite diagnostic arrays: '+label)
        # Cross-run numerical disagreement is diagnostic, never an acceptance gate.
        if not row['allclose']:self.save()
    def identity(self,label,a,b):
        self.rows.append({'field':label,'exact_identity':a==b})
        if a!=b:self.save()
    def save(self):s1.write(self.out/'prefix_differences.json',self.rows)
    def npz(self,label,a,b):
        with np.load(a) as x,np.load(b) as y:
            self.identity(label+'_keys',sorted(x.files),sorted(y.files))
            for key in x.files:self.value(label+'/'+key,x[key],y[key])
    def state(self,ref,cur,i):
        self.npz(f'o{i}',ref/f'obs_{i:02d}.npz',cur/f'obs_{i:02d}.npz')
        self.npz(f'native_state{i}',ref/f'state_{i:02d}.npz',cur/f'state_{i:02d}.npz')
        a,b=p1.read(ref/f'state_{i:02d}.json'),p1.read(cur/f'state_{i:02d}.json')
        self.identity(f'RNG_at_o{i}',a['rng'],b['rng'])
        for k in ('solver_time','solver_steps','control_steps'):
            self.value(f'o{i}/{k}',a['clock'][k],b['clock'][k],1e-6 if k=='solver_time' else 0.,0.)
    def planner(self,ref,cur,t):
        self.npz(f'planner{t}',ref/f'planner_{t:02d}.npz',cur/f'planner_{t:02d}.npz')
        a,b=p1.read(ref/f'planner_{t:02d}.json'),p1.read(cur/f'planner_{t:02d}.json')
        for key in ('t0','rng_before','rng_after','warm_before_present'):self.identity(f'planner{t}/{key}',a[key],b[key])
    def transition(self,ref,cur,t,a,b):
        for key in ('reward','drag','lift','abs_lift','requested_action','clipped_action','memory','gain'):
            self.value(f't{t}/{key}',a[key],b[key])
        aa,bb=p1.read(ref/f'actuator_{t:02d}.json'),p1.read(cur/f'actuator_{t:02d}.json')
        self.identity(f't{t}/outer_count',len(aa),len(bb))
        for j,(x,y) in enumerate(zip(aa,bb)):
            for key in ('native_success','native_fields_finite','status','gain','transition','outer'):
                self.identity(f't{t}/outer{j}/{key}',x[key],y[key])
            self.value(f't{t}/outer{j}/memory',x['smoothed_memory_measured'],y['smoothed_memory_measured'])
            for side in ('top','bottom'):
                for key in ('velocity_measured','flux_measured'):
                    self.value(f't{t}/outer{j}/{side}/{key}',x['boundary'][side][key],y['boundary'][side][key])
            for clock in ('clock_before','clock_after'):
                for key in ('solver_time','solver_steps','control_steps'):
                    self.value(f't{t}/outer{j}/{clock}/{key}',x[clock][key],y[clock][key],1e-6 if key=='solver_time' else 0.,0.)
        self.state(ref,cur,t+1);self.save()


def summaries(rows):
    def subset(xs):
        if not xs:return {'controls':0}
        requested=np.asarray([r['requested_action'] for r in xs]);clipped=np.asarray([r['clipped_action'] for r in xs])
        return {'controls':len(xs),'reward_sum':sum(r['reward'] for r in xs),'reward_mean':float(np.mean([r['reward'] for r in xs])),
                'mean_drag':float(np.mean([r['drag'] for r in xs])),'mean_signed_lift':float(np.mean([r['lift'] for r in xs])),
                'mean_abs_control_interval_lift':float(np.mean([abs(r['lift']) for r in xs])),
                'mean_requested_action':float(requested.mean()),'mean_abs_requested_action':float(np.abs(requested).mean()),
                'mean_abs_clipped_action':float(np.abs(clipped).mean()),
                'clipping_frequency':float(np.mean(np.any(requested!=clipped,axis=1))),
                'saturation_frequency':float(np.mean(np.any(np.abs(clipped)>=1.,axis=1)))}
    return {'whole':subset(rows),'pre40':subset([r for r in rows if r['t']<40]),'post40':subset([r for r in rows if r['t']>=40])}


def environment(condition,directory):
    # Same Stage 1 factory semantics; no class-global hooks.
    import torch
    from types import SimpleNamespace
    from fluidgym_rl.envs import make_fluid_env
    from fluidgym.wrappers import FlattenObservation
    from fluidgym_rl.comparison.evaluate import NominalEnv
    raw=make_fluid_env('CylinderJet2D-easy-v0',flatten=False,mode='val',cuda_device=torch.device('cuda:0'))
    require(type(raw).__name__=='CylinderJetEnv2D' and not raw._differentiable and not raw._use_marl and raw._enable_actions,'native class/flags differ')
    identity=p1.EnvObserver(raw,trace_actuator=False);adapter=PostSmoothingAdapter(raw,condition);audit=CounterObserver(raw,directory,adapter)
    cfg=SimpleNamespace(obs_shape=[302],action_dim=1,episode_length=80,device='cuda:0')
    return NominalEnv(FlattenObservation(raw),cfg),raw,identity,adapter,audit


def run(repo,stage1,out):
    import torch
    start=time.monotonic();prerequisite(stage1,out)
    prov=s1.provenance(repo,out)
    prov['protocol']={'environment':'CylinderJet2D-easy-v0','split':'val','randomize':True,'matrix':MATRIX,'episode_length':80,
                      'S0':{'gain':1.},'S1':{'gain':.5,'onset':0},'S2':{'gain_before':1.,'gain_after':.5,'onset':40},
                      'max_controls':960,'max_runs':12,'watchdog_seconds':2700,'model_updates':0,
                      'experiment':'Stage 2B new exploratory pilot','prefix_is_diagnostic_only':True,'prefix_atol':1e-5,'prefix_rtol':1e-5,'clock_atol':1e-6,'clock_rtol':0}
    s1.write(out/'provenance.json',prov)
    cost={'attempted_controls':0,'completed_controls':0,'runs_started':0,'runs_completed':0,'control_outer_calls':0,'reset_outer_calls':0,
          'control_accepted_increments':0,'reset_accepted_increments':0,'initial_counter_offsets':0,'observed_solver_entries':0,'solver_failures':0,'optimizer_updates':0}
    s1.write(out/'cost.json',cost);episodes=[];paired=[]
    with frozen.factory(repo) as (create_agent,source):
        s1.write(out/'frozen_provenance.json',source)
        # Completed before the first environment construction/reset or CFD.
        s1.write(out/'preregistration_ready.json',{'status':'VERIFIED pre-CFD provenance gates',
            'contract_sha256':p1.sha(out/'STAGE2B_EXPERIMENT_CONTRACT.md'),
            'provenance_sha256':p1.sha(out/'provenance.json'),
            'frozen_provenance_sha256':p1.sha(out/'frozen_provenance.json'),
            'package_manifest_sha256':p1.sha(out/'package_hashes.json')})
        for es,ps,condition in MATRIX:
            require(cost['runs_started']<12 and cost['attempted_controls']<960,'run/CFD budget exhausted before reset')
            name=f'e{es}_p{ps}_{condition}';directory=out/name;directory.mkdir()
            agent,policy,weights=create_agent()  # restoration gate before this run's CFD reset
            random.seed(es);np.random.seed(es);torch.manual_seed(es);torch.cuda.manual_seed_all(es)
            env,raw,identity,adapter,audit=environment(condition,directory)
            prefix=Prefix(directory);ref=out/f'e{es}_p{ps}_S0';records=[];cost['runs_started']+=1
            s1.write(out/'cost.json',cost)
            try:
                flat,info=env.reset(seed=es,randomize=True)
                effective=p1.effective(raw);require(effective['smoothing']==.1 and effective['lift_penalty']==1. and raw.episode_length==80,'physics/interface mismatch')
                require(raw.mode.value=='val' and adapter.t==0 and adapter.gain==1.,'reset schedule/split mismatch')
                s1.close(s1.memory(raw),np.zeros(1,np.float32),'reset memory',0.,0.)
                audit.arm_profiles();b=s1.boundary_readback(raw)
                for side in ('top','bottom'):s1.close(b[side]['velocity_measured'],np.zeros_like(b[side]['velocity_measured']),'reset boundary',0.,0.)
                np.savez(directory/'profiles.npz',**audit.profiles);s1.write(directory/'reset_boundary.json',b)
                # Environment uses private RNG; planner seeded only after native reset.
                p1.seed_policy(policy,ps);require(not hasattr(agent,'_prev_mean'),'planner reset did not clear warm state')
                meta={'environment_seed':es,'planner_seed':ps,'condition':condition,'identity':identity.identity(),'effective':effective,
                      'reset_clock':s1.clock(raw),'reset_calls':audit.reset_calls,'weights_before':weights,
                      'rng_after_planner_reset':p1.rng_hash(p1.rng())}
                s1.write(directory/'initialization.json',meta)
                s1.save_obs(directory,0,flat,info);s1.snapshot(raw,directory/'state_00')
                if condition!='S0':
                    reference=p1.read(ref/'initialization.json')
                    require(reference['identity']==meta['identity'],'physical initialization mismatch')
                    require(reference['effective']==meta['effective'],'effective configuration mismatch')
                    prefix.identity('initial_domain',reference['identity'],meta['identity']);prefix.identity('effective_config',reference['effective'],meta['effective'])
                    prefix.state(ref,directory,0);prefix.save()
                for t in range(80):
                    require(cost['attempted_controls']<960 and cost['runs_started']<=12,'hard execution budget exceeded')
                    before=s1.clock(raw)
                    action,planner=frozen.planner_capture(agent,policy,flat,t,directory)
                    frozen.invariant(agent,weights)  # before physical control, no changed-weight action can be applied
                    if condition=='S2' and t<40:prefix.planner(ref,directory,t);prefix.save()
                    if condition=='S2' and t==40:
                        # Nominal prefix differences are retained, not required to pass; no reset at onset.
                        s1.write(directory/'shift_gate.json',{'status':'scheduled onset; prefix diagnostic only','prefix_transitions':40,'first_changed_transition':40,'planner_reset_at_shift':False})
                    command=np.clip(action,-1,1).astype(np.float32)
                    s1.write(directory/f'transition_{t:02d}_begin.json',{'t':t,'requested_action':action,'clipped_action':command,
                              'gain':gain_at(condition,t),'clock_before':before,'memory_before':s1.memory(raw)})
                    calls=audit.calls;audit.begin(command);cost['attempted_controls']+=1;s1.write(out/'cost.json',cost)
                    try:
                        flat,reward,term,trunc,info=env.step(action)  # original CheckedEnv clips the original request
                        cost['completed_controls']+=1
                    finally:
                        audit.active=False;s1.write(directory/f'actuator_{t:02d}.json',audit.outer)
                    s1.save_obs(directory,t+1,flat,info);now=s1.clock(raw)
                    require(not term and bool(trunc)==(t==79),'unexpected horizon termination')
                    require(audit.calls-calls==25 and len(audit.outer)==25,'native outer count')
                    require(now['control_steps']==t+1 and now['solver_time']>before['solver_time'],'clock/control counter')
                    record={'t':t,'gain':gain_at(condition,t),'requested_action':action,'clipped_action':command,'reward':reward,
                            'drag':p1.scalar(info['drag']),'lift':p1.scalar(info['lift']),'abs_lift':abs(p1.scalar(info['lift'])),
                            'memory':s1.memory(raw),'native_solver_success':all(x['native_success'] for x in audit.outer),'outer_calls':len(audit.outer),'clipped':bool(np.any(action!=command)),'saturated':bool(np.any(np.abs(command)>=1.)),'terminated':term,'truncated':trunc,**now}
                    require(np.isfinite([record['reward'],record['drag'],record['lift'],record['abs_lift']]).all(),'nonfinite outcome')
                    records.append(record);s1.write(directory/'steps.json',records)
                    s1.snapshot(raw,directory/f'state_{t+1:02d}')
                    if condition=='S2' and t<40:
                        reference=p1.read(ref/'steps.json')[t];prefix.transition(ref,directory,t,reference,record)
                    s1.write(out/'cost.json',cost)
                frozen.invariant(agent,weights)
                episode={'run':name,**meta,'weights_after':frozen.weight_hash(agent),'metrics':summaries(records),'status':'passed'}
                episodes.append(episode);s1.write(directory/'episode.json',episode);s1.write(out/'episodes.json',episodes)
                cost['runs_completed']+=1
                if condition=='S2':
                    nominal=p1.read(ref/'episode.json')['metrics']['post40'];shifted=episode['metrics']['post40']
                    paired.append({'environment_seed':es,'planner_seed':ps,'Delta_post':shifted['reward_sum']-nominal['reward_sum'],
                                   'post40_metric_differences_S2_minus_S0':{k:shifted[k]-nominal[k] for k in nominal if k!='controls'}})
                    s1.write(out/'paired_outcomes.json',paired)
            finally:
                s1.write(directory/'steps.json',records)
                for key,value in [('control_outer_calls',audit.control_calls),('reset_outer_calls',audit.reset_calls),
                                  ('control_accepted_increments',audit.control_accepted),('reset_accepted_increments',audit.reset_accepted),
                                  ('initial_counter_offsets',audit.initial_offset or 0),('observed_solver_entries',audit.calls),('solver_failures',audit.failed)]:cost[key]+=value
                cost['worker_wall_seconds']=time.monotonic()-start;s1.write(out/'cost.json',cost)
                weights_ok=False
                if 'agent' in locals():
                    try:
                        actual=frozen.weight_hash(agent);weights_ok=actual==weights
                        s1.write(directory/'weights_final.json',{'expected':weights,'actual':actual})
                    except BaseException as exc:s1.write(directory/'weights_final.json',{'expected':weights,'error':repr(exc)})
                audit.close();adapter.close();env.close();del env,raw,identity,adapter,audit,agent,policy;gc.collect()
                require(weights_ok,'frozen weight verification failed during finalization')
    require(cost['completed_controls']==960 and cost['runs_completed']==12,'incomplete pilot')
    p1.protect(repo);prerequisite(stage1,out)
    s1.write(out/'worker_status.json',{'status':'passed','diagnostic_prefix_pairs_recorded':4,'training_updates':0,
             'scope':'NEW Stage 2B outcome-focused exploratory pilot; original Stage 2 remains failed; no model-error/history claim'})


if __name__=='__main__':
    repo,stage1,out=map(Path,sys.argv[1:4]);sys.path.insert(0,str(repo/'src'))
    try:
        require(os.environ.get('FGRL_STAGE2B_WATCHDOG')=='2700','use external launch.py watchdog')
        run(repo,stage1,out)
    except BaseException as exc:
        s1.write(out/'worker_status.json',{'status':'failed','error':repr(exc),'traceback':traceback.format_exc()})
        traceback.print_exc();sys.exit(1)
