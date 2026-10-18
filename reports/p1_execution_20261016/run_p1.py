#!/usr/bin/env python3
"""Bounded P1 only. No training, source replacement, or broad dataset collection."""
import argparse, contextlib, copy, csv, datetime, hashlib, importlib.metadata as md
import json, os, pathlib, pickle, random, subprocess, sys, tarfile, tempfile, time, traceback, uuid
sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
EXPECTED = json.loads((HERE/'expected.json').read_text())
B_SEQUENCE = {'zero': [0., 0., 0., 0.], 'bounded': [0., .2, -.2, 0.]}
ATOL, RTOL = 1e-5, 1e-5  # numerical repeat gate; NOT a physical-accuracy tolerance

def read(p): return json.loads(pathlib.Path(p).read_text())
def write(p, obj):
 p=pathlib.Path(p); tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def need(ok, msg):
 if not ok: raise RuntimeError(msg)
def npv(x):
 import numpy as np
 return x.detach().cpu().numpy().copy() if hasattr(x,'detach') else np.asarray(x).copy()
def scalar(x): return float(npv(x).reshape(-1)[0])
def dumpcsv(path, rows):
 if not rows:return
 with pathlib.Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def compare(a,b,atol=ATOL,rtol=RTOL):
 import numpy as np
 a,b=np.asarray(a),np.asarray(b)
 return {'shape_equal':a.shape==b.shape,'max_abs':float(np.max(np.abs(a-b))) if a.shape==b.shape and a.size else None,'allclose':bool(a.shape==b.shape and np.allclose(a,b,atol=atol,rtol=rtol)),'bitwise_equal':bool(np.array_equal(a,b))}
def checkpoint_files(repo):
 return {k:sha(repo/k) for k in EXPECTED['checkpoints']}
def protect(repo):
 paths={**EXPECTED['required_artifacts'],**EXPECTED['checkpoints']}
 for k,v in paths.items():need((repo/k).is_file() and sha(repo/k)==v,'protected artifact differs: '+k)
 return paths

def provenance(repo, out):
 import torch, fluidgym
 from fluidgym.config import config
 from fluidgym.simulation.extensions import PISOtorch
 package=pathlib.Path(fluidgym.__file__).resolve().parent
 sources={str(p.relative_to(package)):sha(p) for p in package.rglob('*.py')}
 pinned_head=subprocess.check_output(['git','-C',str(package),'rev-parse','HEAD'],text=True).strip()
 need(pinned_head=='50071140b73a2a917fb96d33522a769f903f8c27','FluidGym checkout commit differs')
 need(sources==EXPECTED['sources'],'current FluidGym Python source differs from reviewed pinned source')
 need(sha(PISOtorch.__file__)==EXPECTED['backend_hash'],'CFD binary differs from reviewed preflight')
 versions={k:md.version(k) for k in EXPECTED['versions']}
 need(versions==EXPECTED['versions'],'runtime dependency drift from preflight')
 need(torch.cuda.is_available(),'CUDA required')
 domain=pathlib.Path(config.initial_domains_path)/'cylinder_2D_Re100_Res24'
 for k,v in EXPECTED['domains'].items():need((domain/k).is_file() and sha(domain/k)==v,'dataset mismatch/missing: '+k)
 protect(repo)
 pipeline={str(p.relative_to(repo)):sha(p) for p in (repo/'src/fluidgym_rl').rglob('*.py')}
 # Refuse changes in executable inference/evaluation code since the historical run.
 core=['src/fluidgym_rl/comparison/evaluate.py','src/fluidgym_rl/evalcontract.py','src/fluidgym_rl/tdmpc/evaluation.py','src/fluidgym_rl/tdmpc2_runtime.py','src/fluidgym_rl/envs.py']
 for k in core:need(pipeline.get(k)==EXPECTED['historical_pipeline_hashes'].get(k),'core inference/evaluation source drift: '+k)
 result={'scope':'current-pinned-environment replication','fluidgym_file':str(package/'__init__.py'),'source_hashes':sources,'pinned_git_head':pinned_head,'backend':{'path':PISOtorch.__file__,'sha256':sha(PISOtorch.__file__)},'versions':versions,'python':sys.version,'executable':sys.executable,'domain_root':str(domain),'domain_hashes':EXPECTED['domains'],'pipeline_hashes':pipeline,'checkpoint_hashes':checkpoint_files(repo),'script_sha256':sha(__file__),'expected_sha256':sha(HERE/'expected.json'),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'torch_flags':flags(),'git_status':subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)}
 write(out/'provenance.json',result);return result

def flags():
 import torch
 return {'matmul_precision':torch.get_float32_matmul_precision(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'tf32':torch.backends.cuda.matmul.allow_tf32}
def rng():
 import torch,numpy as np
 return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state().clone(),'torch_cuda':[x.clone() for x in torch.cuda.get_rng_state_all()]}
def rng_hash(s):
 result={k:hashlib.sha256(pickle.dumps(v)).hexdigest() for k,v in s.items() if k in ('python','numpy')}
 result['torch_cpu']=hashlib.sha256(s['torch_cpu'].numpy().tobytes()).hexdigest()
 result['torch_cuda']=[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in s['torch_cuda']];return result

def seed_policy(policy, seed):
 import numpy as np
 random.seed(seed);np.random.seed(seed);policy.reset(seed)

class SolverObserver:
 def __init__(self):self.calls=0;self.failed=0
 def __enter__(self):
  from fluidgym.simulation import Simulation
  from unittest.mock import patch
  original=Simulation.single_step
  def checked(sim,*a,**kw):
   self.calls+=1
   try:result=original(sim,*a,**kw)
   except BaseException:self.failed+=1;raise
   if not result:self.failed+=1;raise RuntimeError('native Simulation.single_step returned failure')
   return result
  self.patch=patch.object(Simulation,'single_step',checked);self.patch.start();return self
 def __exit__(self,*a):self.patch.stop()

class EnvObserver:
 """Observe official calls and read memory; never substitute state or control."""
 def __init__(self,raw,trace_actuator=True):
  self.raw=raw;self.identities=[];self.applied=[]
  self.load_original=raw._load_initial_domain;self.apply_original=raw._apply_action
  def load(*a,**kw):
   import inspect
   bound=inspect.signature(self.load_original).bind(*a,**kw);bound.apply_defaults()
   mode=bound.arguments['mode'];idx=int(bound.arguments['idx'])
   self.identities.append({'split':mode.value,'index':idx,'domain_id':raw.initial_domain_id})
   return self.load_original(*a,**kw)
  def apply(action,*a,**kw):
   self.applied.append(npv(action).reshape(-1).tolist());return self.apply_original(action,*a,**kw)
  raw._load_initial_domain=load
  if trace_actuator:raw._apply_action=apply
 def memory(self):
  value=getattr(self.raw,'_CylinderEnvBase__last_control',None)
  need(value is not None,'runtime actuator memory inaccessible in reviewed source')
  return npv(value).reshape(-1)
 def clock(self):return {'solver_time':scalar(self.raw._sim.total_time),'solver_steps':int(self.raw._sim.total_step),'control_steps':int(self.raw._n_steps)}
 def identity(self):
  need(len(self.identities)>0,'actual initial-domain load not observed')
  item=dict(self.identities[-1]);prefix=str(item['index'])+'/'+item['split']
  item['files']={k:v for k,v in EXPECTED['domains'].items() if k in [prefix+'.json',prefix+'.npz','domain_statistics.json']}
  need(len(item['files'])==3,'initial-domain file provenance incomplete');return item

def environment(trace_actuator=True):
 import torch
 from types import SimpleNamespace
 from fluidgym_rl.envs import make_fluid_env
 from fluidgym.wrappers import FlattenObservation
 from fluidgym_rl.comparison.evaluate import NominalEnv
 raw=make_fluid_env('CylinderJet2D-easy-v0',flatten=False,mode='test',cuda_device=torch.device('cuda:0'))
 need(not raw._differentiable and not raw._use_marl and raw._enable_actions,'nominal flags mismatch')
 obs=EnvObserver(raw,trace_actuator=trace_actuator)
 cfg=SimpleNamespace(obs_shape=[302],action_dim=1,episode_length=80,device='cuda:0')
 env=NominalEnv(FlattenObservation(raw),cfg)
 return env,raw,obs

def effective(raw):
 import numpy as np
 d={'dt':scalar(raw.dt),'calls_per_control':int(raw._n_sim_steps),'step_length':scalar(raw.step_length),'viscosity':scalar(raw._domain.viscosity),'cfl':scalar(raw._sim.adaptive_CFL),'pressure_tol':scalar(raw._sim.pressure_tol),'shapes':[list(b.velocity.shape) for b in raw._domain.getBlocks()],'reward_reference':float(raw._cd_ref),'lift_penalty':float(raw._lift_penalty),'smoothing':float(raw._action_smoothing_alpha)}
 need(np.isclose(d['dt'],.01) and d['calls_per_control']==25 and np.isclose(d['step_length'],.25) and np.isclose(d['viscosity'],.01) and np.isclose(d['cfl'],.8) and np.isclose(d['pressure_tol'],1e-5),'effective physics mismatch')
 for block in raw._domain.getBlocks():
  if block.hasViscosity():need(np.allclose(npv(block.viscosity),.01,atol=1e-9,rtol=1e-5),'block viscosity differs')
 return d

def native_obs(flat,info):
 import numpy as np
 v,p=npv(info['original_velocity']),npv(info['original_pressure']);f=npv(flat)
 need(v.shape==(151,2) and p.shape==(151,) and f.shape==(302,),'raw/flat observation shape mismatch')
 need(all(x.dtype==np.float32 and np.isfinite(x).all() for x in [v,p,f]),'observation dtype/nonfinite')
 need(np.array_equal(f,v.reshape(-1)),'official flatten differs from velocity sensor-major input')
 return v,p,f

def run_b(repo,out):
 import torch,numpy as np
 trials={};rows=[];attempted=completed=0
 with SolverObserver() as solver:
  for kind,actions in B_SEQUENCE.items():
   for rep in range(2):
    name=f'{kind}_{rep}';env,raw,observer=environment();arrays=[];records=[]
    try:
     reset_calls=solver.calls;flat,info=env.reset(seed=0,randomize=True);eff=effective(raw);identity=observer.identity();v,p,f=native_obs(flat,info)
     need(np.allclose(observer.memory(),0,atol=0),'actuator reset memory nonzero')
     arrays.append((v,p,f));records.append({'step':0,**observer.clock(),'memory_measured':observer.memory().tolist(),'identity':identity,'solver_calls_reset':solver.calls-reset_calls,'effective':eff})
     for step,a in enumerate(actions,1):
      before=observer.memory();clock=observer.clock();observer.applied=[];calls=solver.calls;attempted+=1
      write(out/'cost.json',{'attempted_control_steps':attempted,'completed_control_steps':completed,'solver_calls_including_resets':solver.calls})
      flat,reward,term,trunc,info=env.step(np.array([a],np.float32));completed+=1
      v,p,f=native_obs(flat,info);arrays.append((v,p,f));now=observer.clock()
      need(not term and not trunc and now['control_steps']==step,'unexpected B boundary')
      need(now['solver_time']>clock['solver_time'] and now['solver_steps']-clock['solver_steps']>=25,'invalid solver clock/counters')
      need(solver.calls-calls==25 and len(observer.applied)==25,'outer solver/action count differs')
      # Source-derived recurrence evaluated separately; not reported as runtime observation.
      expected=[];x=before.astype(np.float32).copy();command=np.array([a],np.float32)
      for _ in range(25):x=x+np.float32(.1)*(command-x);expected.append(x.copy())
      measured=np.array(observer.applied,np.float32);need(np.allclose(measured,np.array(expected),atol=1e-7,rtol=1e-6),'measured applied controls differ from source recurrence')
      need(np.allclose(observer.memory(),measured[-1],atol=0,rtol=0),'actuator memory differs from last runtime applied argument')
      record={'step':step,'requested_action':command.tolist(),'expected_applied_source_derived':np.array(expected).tolist(),'runtime_applied_arguments':observer.applied,'runtime_actuator_memory':observer.memory().tolist(),'reward':reward,'drag':scalar(info['drag']),'lift':scalar(info['lift']),**now,'solver_calls':solver.calls-calls,'solver_failures':solver.failed}
      records.append(record);write(out/(name+'_records.json'),records)
     np.savez(out/(name+'_observations.npz'),velocity=np.stack([x[0] for x in arrays]),pressure=np.stack([x[1] for x in arrays]),flat=np.stack([x[2] for x in arrays]))
     trials[name]=(arrays,records);write(out/(name+'_records.json'),records)
    finally:
     if arrays:np.savez(out/(name+'_observations.npz'),velocity=np.stack([x[0] for x in arrays]),pressure=np.stack([x[1] for x in arrays]),flat=np.stack([x[2] for x in arrays]))
     write(out/(name+'_records.json'),records);write(out/'cost.json',{'attempted_control_steps':attempted,'completed_control_steps':completed,'solver_calls_including_resets':solver.calls,'solver_failures':solver.failed});env.close()
  for kind in B_SEQUENCE:
   aa,ar=trials[kind+'_0'];bb,br=trials[kind+'_1'];need(ar[0]['identity']==br[0]['identity'],'repeat IC differs')
   for i in range(5):
    for key,j in [('velocity',0),('pressure',1),('flat',2)]:
     c=compare(aa[i][j],bb[i][j]);rows.append({'sequence':kind,'step':i,'quantity':key,**c})
    for key in ['solver_time','solver_steps','control_steps']+(['reward','drag','lift','runtime_actuator_memory'] if i else ['memory_measured']):
     c=compare(ar[i][key],br[i][key],atol=0 if key.endswith('steps') else ATOL,rtol=0 if key.endswith('steps') else RTOL);rows.append({'sequence':kind,'step':i,'quantity':key,**c})
  dumpcsv(out/'repeat_differences.csv',rows)
  write(out/'cost.json',{'attempted_control_steps':attempted,'completed_control_steps':completed,'solver_calls_including_resets':solver.calls,'solver_failures':solver.failed})
  need(all(x['allclose'] for x in rows),'same-environment repeat exceeded declared numerical tolerance; see repeat_differences.csv')
  need(completed==16,'B budget/count mismatch')
 return {'status':'passed','control_steps':completed,'solver_calls_including_resets':solver.calls,'tolerance':{'atol':ATOL,'rtol':RTOL},'trials':4,'scope':'numerical reproducibility on one seeded randomized TEST IC, not physical accuracy','measured_actuator':'runtime private smoothing memory and actual _apply_action argument; no independent boundary flux measurement'}

@contextlib.contextmanager
def agent_policy(repo,alg):
 from fluidgym_rl.tdmpc.evaluation import Policy,load_checkpoint
 from fluidgym_rl.comparison.evaluate import training_identity
 spec=EXPECTED['policies'][alg];ck=repo/spec['checkpoint'];identity=training_identity(alg,repo/spec['training_run'],ck)
 need(identity['environment_steps']==spec['step'],'wrong checkpoint step')
 selection=read(repo/spec['selection']);need(selection['checkpoint_sha256']==sha(ck),'frozen selection hash mismatch')
 cwd=pathlib.Path.cwd()
 with tempfile.TemporaryDirectory(prefix='fgrl_p1_inference_') as tmp:
  try:
   if alg=='tdmpc':
    from types import SimpleNamespace
    from fluidgym_rl.tdmpc.runtime import make_workcopy,load_algorithm
    work=pathlib.Path(tmp)/'tdmpc';provenance_=make_workcopy(work);Agent,_=load_algorithm(work)
    cfg=SimpleNamespace(**identity['source_config']['parameters']);cfg.device='cuda:0';agent=Agent(cfg);load_checkpoint(agent,ck);policy=Policy(agent,spec['step']);config=vars(cfg)
   else:
    from fluidgym_rl.upstream import make_tdmpc2_workcopy
    from fluidgym_rl.tdmpc2_runtime import load_agent,TDMPC2Policy
    from fluidgym_rl.training_logging import wandb_config
    wc=make_tdmpc2_workcopy(pathlib.Path(tmp)/'tdmpc2');cfg,agent,_=load_agent(ck,ck.parent/'run_config.json',wc['code_dir'],pathlib.Path(tmp)/'run');policy=TDMPC2Policy(agent,'plan');config=wandb_config(cfg);provenance_=wc
    need(cfg.mpc and cfg.horizon==3 and cfg.num_samples==512 and cfg.num_pi_trajs==24,'saved TD-MPC2 planner config mismatch')
   yield agent,policy,config,provenance_
  finally:os.chdir(cwd)

def model_hash(agent):
 h=hashlib.sha256()
 for name,p in sorted(agent.model.state_dict().items()):
  if p.device.type=='meta':continue
  h.update(name.encode());h.update(npv(p).tobytes())
 return h.hexdigest()
def set_memory(agent,state):
 import torch
 if state is None:
  if hasattr(agent,'_prev_mean'):
   if isinstance(agent,torch.nn.Module):agent._prev_mean.zero_()
   else:del agent._prev_mean
 elif hasattr(agent,'_prev_mean'):agent._prev_mean.copy_(state)
 else:agent._prev_mean=state.clone()

def run_a(repo,out,alg,bdir):
 import numpy as np,torch
 need(read(bdir/'status.json')['status']=='passed','B gate not passed')
 saved=np.load(bdir/'bounded_0_observations.npz');observations=[saved['flat'][0].copy(),saved['flat'][2].copy()]
 comparisons=[];calls=[];outputs={};call_count=0
 with agent_policy(repo,alg) as (agent,policy,config,source):
  write(out/'resolved_config.json',config);write(out/'loader_source.json',source);write(out/'runtime_flags_after_load.json',flags());before=model_hash(agent)
  for oi,o in enumerate(observations):
   x=torch.as_tensor(o,device=agent.device).unsqueeze(0)
   with torch.no_grad():
    z=agent.model.h(x) if alg=='tdmpc' else agent.model.encode(x,None)
    zz=agent.model.h(x) if alg=='tdmpc' else agent.model.encode(x,None)
    need(torch.isfinite(z).all().item() and torch.equal(z,zz),'encoder repeat/finite failure')
    if alg=='tdmpc':
     prior=agent.model.pi(z,0);next_z,reward=agent.model.next(z,prior);qs=agent.model.Q(z,prior);params={'prior_mean':npv(prior)}
    else:
     prior,pi=agent.model.pi(z,None);next_z=agent.model.next(z,prior,None);reward=agent.model.reward(z,prior,None);qs=(agent.model.Q(z,prior,None,return_type='all'),);params={k:npv(pi[k]) for k in ['mean','log_std']}
     from common.math import two_hot_inv
     params['reward_decoded']=npv(two_hot_inv(reward,agent.cfg));params['q_decoded']=npv(two_hot_inv(qs[0],agent.cfg))
    values={'encoder':npv(z),'next_z':npv(next_z),'reward':npv(reward),**params,**{f'q{i}':npv(q) for i,q in enumerate(qs)}}
    need(all(np.isfinite(v).all() for v in values.values()),'nonfinite frozen model output');np.savez(out/f'model_outputs_obs{oi}.npz',**values)
   def invoke(name,seed,memory,t0):
    nonlocal call_count
    seed_policy(policy,seed);set_memory(agent,memory);pre=rng();mh=None if not hasattr(agent,'_prev_mean') else npv(agent._prev_mean)
    torch.cuda.synchronize();start=time.monotonic();action=npv(policy.act(o,t0));torch.cuda.synchronize();elapsed=time.monotonic()-start
    post=rng();need(np.isfinite(action).all() and action.shape==(1,) and np.abs(action).max()<=1,'invalid planner action');call_count+=1
    label=f'obs{oi}_{name}';torch.save({'before':pre,'after':post},out/(label+'_rng.pt'))
    calls.append({'label':label,'seed':seed,'t0':t0,'rng_before':rng_hash(pre),'rng_after':rng_hash(post),'warm_before':None if mh is None else mh.tolist(),'warm_after':npv(agent._prev_mean).tolist(),'action':action.tolist(),'wall_seconds':elapsed})
    write(out/'calls.json',calls);outputs[label]=action.copy();return action,agent._prev_mean.detach().clone()
   # Native compiled path is warmed up before controlled comparisons; no config/algorithm override.
   invoke('compile_warmup',7,None,True)
   _,wa=invoke('valid_warm_a',11,None,True)
   invoke('warm_branch_compile',7,wa,False)
   _,wb=invoke('valid_warm_b',12,None,True)
   a,_=invoke('reset_same_rng_1',100,None,True);b,_=invoke('reset_same_rng_2',100,None,True)
   comparisons.append({'observation':oi,'contrast':'same_rng_reset_state','required_to_pass':True,**compare(a,b,1e-6,1e-6)})
   c,_=invoke('changed_rng_reset_state',101,None,True)
   comparisons.append({'observation':oi,'contrast':'changed_rng_reset_state','required_to_pass':False,**compare(a,c,1e-6,1e-6)})
   d,_=invoke('warm_a_same_rng_1',100,wa,False);e,_=invoke('warm_a_same_rng_2',100,wa,False)
   comparisons.append({'observation':oi,'contrast':'same_rng_same_warm','required_to_pass':True,**compare(d,e,1e-6,1e-6)})
   g,_=invoke('warm_a_changed_rng',101,wa,False)
   comparisons.append({'observation':oi,'contrast':'changed_rng_same_warm','required_to_pass':False,**compare(d,g,1e-6,1e-6)})
   f,_=invoke('warm_b_same_rng',100,wb,False)
   comparisons.append({'observation':oi,'contrast':'same_rng_changed_valid_warm','required_to_pass':False,**compare(d,f,1e-6,1e-6)})
   need(model_hash(agent)==before,'frozen model state changed during inference')
  dumpcsv(out/'contrasts.csv',comparisons);np.savez(out/'selected_actions.npz',**outputs)
  need(all(x['allclose'] for x in comparisons if x['required_to_pass']),'controlled fixed-input inference not reproducible within declared tolerance')
 return {'status':'passed','algorithm':alg,'control_steps':0,'planner_calls':call_count,'observations':2,'frozen_model_hash':before,'contrasts':comparisons,'rng_sources':['Python global','NumPy legacy global','Torch CPU','Torch CUDA all devices'],'limitations':['Gaussian candidates/prior/Q subsampling/elite selection share RNG streams; not independently varied','Compiled planner intermediate candidates not exposed; no hooks or graph changes used','Different RNG or memory need not change the selected action','Finite outputs and repeatability do not establish prediction accuracy']}

class ClosedObserver:
 def __init__(self,env,raw,observer,solver,out):
  self.env=env;self.raw=raw;self.observer=observer;self.solver=solver;self.out=out;self.episodes=[];self.attempted=0;self.completed=0;self.partial_rows=[];self.samples=[]
 def __getattr__(self,k):return getattr(self.env,k)
 def reset(self,**kw):
  n=self.solver.calls;self.samples=[];self.episode_start=time.monotonic();flat,info=self.env.reset(**kw);native_obs(flat,info);effective(self.raw)
  self.episodes.append({'episode':len(self.episodes),'reset_kwargs':kw,'initial_domain':self.observer.identity(),'clock_reset':self.observer.clock(),'reset_solver_calls':self.solver.calls-n,'solver_calls_start':self.solver.calls});write(self.out/'initializations.json',self.episodes);return flat,info
 def step(self,action):
  need(self.attempted<800,'per-algorithm C budget exceeded');self.attempted+=1;write(self.out/'cost.json',{'attempted_control_steps':self.attempted,'completed_control_steps':self.completed,'solver_calls_including_resets':self.solver.calls})
  clock=self.observer.clock();calls=self.solver.calls;result=self.env.step(action);self.completed+=1;flat,r,term,trunc,info=result;native_obs(flat,info);now=self.observer.clock()
  need(now['solver_time']>clock['solver_time'] and now['solver_steps']-clock['solver_steps']>=25 and self.solver.calls-calls==25,'C clock/solver-call contract failure')
  self.samples.append((float(r),scalar(info['drag']),scalar(info['lift'])))
  if term or trunc:
   import numpy as np
   a=np.asarray(self.samples);self.partial_rows.append({'episode':len(self.episodes)-1,'length':len(a),'return':float(a[:,0].sum()),'mean_reward':float(a[:,0].mean()),'mean_drag':float(a[:,1].mean()),'mean_lift':float(a[:,2].mean()),'mean_abs_lift':float(np.abs(a[:,2]).mean()),'terminated':bool(term),'truncated':bool(trunc),'wall_seconds':time.monotonic()-self.episode_start});dumpcsv(self.out/'completed_episodes_partial.csv',self.partial_rows)
   self.episodes[-1].update(clock_end=now,solver_calls_episode=self.solver.calls-self.episodes[-1]['solver_calls_start']);write(self.out/'initializations.json',self.episodes)
  # discard per-call actuator observation in C; B owns bounded transition logging.
  self.observer.applied.clear();return result

def run_c(repo,out,alg,root):
 for stage in ['B','A_tdmpc','A_tdmpc2']:need(read(root/stage/'status.json')['status']=='passed','C gate not passed: '+stage)
 import torch
 from fluidgym_rl.evalcontract import EvalContract,run_evaluation,write_run
 from fluidgym_rl.comparison.evaluate import aggregate
 with agent_policy(repo,alg) as (agent,policy,config,source),SolverObserver() as solver:
  before=model_hash(agent);write(out/'resolved_policy_config.json',config);write(out/'loader_source.json',source);write(out/'runtime_flags_after_load.json',flags())
  rng_events=[]
  class PolicyObserver:
   name=policy.name;mode=policy.mode
   def reset(self,episode_seed):
    seed_policy(policy,episode_seed);rng_events.append({'episode':len(rng_events),'seed':episode_seed,'rng_after_reset':rng_hash(rng()),'runtime_flags':flags(),'warm_start_note':'native t0=True on first action; previous mean ignored'});write(out/'planner_rng.json',rng_events)
   def act(self,obs,t0):
    action=policy.act(obs,t0)
    if checked.observer.clock()['control_steps']==79:
     rng_events[-1]['rng_after_final_action']=rng_hash(rng());write(out/'planner_rng.json',rng_events)
    return action
  env,raw,observer=environment(trace_actuator=False);checked=ClosedObserver(env,raw,observer,solver,out)
  try:
   contract=EvalContract(env_id='CylinderJet2D-easy-v0',split='test',n_episodes=10,protocol='seeded',base_seed=0)
   records=run_evaluation(checked,PolicyObserver(),contract,metric_names=['drag','lift','abs_lift'],action_low=env.action_space.low,action_high=env.action_space.high)
   summary=aggregate(records)
   need(checked.completed==800 and len(records)==10 and solver.failed==0,'C completion/budget failure')
   need(model_hash(agent)==before,'model weights/state changed during C')
   manifest={'scope':'current-pinned-environment replication, not exact historical software replay','algorithm':alg,'checkpoint':EXPECTED['policies'][alg],'checkpoint_sha256':sha(repo/EXPECTED['policies'][alg]['checkpoint']),'selection_sha256':sha(repo/EXPECTED['policies'][alg]['selection']),'frozen_model_hash':before,'contract':vars(contract),'solver_failures':solver.failed}
   write_run(out,config,records,summary,manifest)
   historical=list(csv.DictReader((repo/EXPECTED['policies'][alg]['historical']/'episodes.csv').open()));new=list(csv.DictReader((out/'episodes.csv').open()));diff=[]
   for old,now in zip(historical,new):
    need(old['episode']==now['episode'] and json.loads(old['reset_kwargs'])==json.loads(now['reset_kwargs']),'historical protocol pairing mismatch')
    for key in ['return','mean_reward','mean_drag','mean_lift','mean_abs_lift']:
     diff.append({'algorithm':alg,'episode':int(now['episode']),'metric':key,'historical':float(old[key]),'current':float(now[key]),'difference':float(now[key])-float(old[key]),'initial_domain_index':checked.episodes[int(now['episode'])]['initial_domain']['index']})
   dumpcsv(out/'historical_comparison.csv',diff)
   return {'status':'passed','algorithm':alg,'control_steps':checked.completed,'attempted_control_steps':checked.attempted,'solver_calls_including_resets':solver.calls,'solver_failures':solver.failed,'mean_reward_per_step':summary['mean_reward_per_step']['mean'],'interpretation':'contract/restoration/completion passed; reward difference is not a failure or robustness result'}
  finally:
   write(out/'cost.json',{'attempted_control_steps':checked.attempted,'completed_control_steps':checked.completed,'solver_calls_including_resets':solver.calls,'solver_failures':solver.failed});env.close()

REPORTS={'B':'P1_PRESCRIBED_ACTION.md','A':'P1_FROZEN_POLICY_RNG.md','C':'P1_CLOSED_LOOP_REPLICATION.md'}
def reports(root,overall):
 stages={k:read(root/k/'status.json') if (root/k/'status.json').exists() else {'status':'NOT_RUN'} for k in ['B','A_tdmpc','A_tdmpc2','C_tdmpc','C_tdmpc2']}
 for category,filename in REPORTS.items():
  subset={k:v for k,v in stages.items() if k.startswith(category)}
  lines=[f'# {filename[:-3]}','', 'Scope: current-pinned-environment verification. Historical binary equivalence is UNRESOLVED.','']
  for k,s in subset.items():
   label='VERIFIED' if s['status']=='passed' else 'UNRESOLVED';lines += [f'## {k}: {label} — {s["status"]}','','```json',json.dumps(s,indent=2),'```','']
  for k in subset:
   table=root/k/('repeat_differences.csv' if category=='B' else 'contrasts.csv' if category=='A' else 'historical_comparison.csv')
   if table.exists():
    lines += ['',f'Numerical comparison table: [{k}/{table.name}]({k}/{table.name}).']
   if category=='C' and (root/k/'summary.json').exists():
    summary=read(root/k/'summary.json');lines += [f"VERIFIED: reward/step={summary['mean_reward_per_step']['mean']:.9f}; Cd={summary['metrics_per_step_mean']['drag']['mean']:.9f}; abs Cl={summary['metrics_per_step_mean']['abs_lift']['mean']:.9f}.",f'Per-episode results: [{k}/episodes.csv]({k}/episodes.csv). Initial-domain identities: [{k}/initializations.json]({k}/initializations.json).']
  lines+=['INFERRED: differences describe this fixed runtime and the tested initializations only.','UNRESOLVED: physical accuracy, representation robustness and dynamics-shift sensitivity.','Numerical gate thresholds were declared before execution; failed/unrun stages are not passes.']
  (root/filename).write_text('\n'.join(lines)+'\n')
 b=stages['B'].get('control_steps',0);c=sum(stages[k].get('control_steps',0) for k in ['C_tdmpc','C_tdmpc2'])
 lines=['# P0_P1_FINAL_ASSESSMENT','',f'Overall execution: {overall}.','',f'Completed control steps from passed stages: P1-B={b}, P1-C={c}. For failures, cost.json is authoritative for partial/attempted steps. Reset/internal solver calls are separate.','', 'VERIFIED: no checkpoint selection, training, physics/source replacement or upstream algorithm changes requested by this runner. Protected bytes rechecked per worker and after the run.','', 'UNRESOLVED: exact historical software replay. Current pinned source differs from historical wheel; numerical repeatability is not physical accuracy.','', 'Priorities after reviewing the evidence: resolve any failed gate first; inspect per-episode current-minus-historical differences; quantify the stated sampling and IC limits before nominal model/planning interpretation. No broader transition logger, new metric, physical shift or representation analysis was implemented.','', 'Stage statuses:','```json',json.dumps(stages,indent=2),'```']
 (root/'P0_P1_FINAL_ASSESSMENT.md').write_text('\n'.join(lines));write(root/'verification.json',{'scope':'current-pinned-environment replication','status':overall,'stages':stages,'budgets':{'B_control_steps_max':16,'C_control_steps_max':1600},'no_training':True})

def worker(args):
 repo=args.repo.resolve();root=args.output.resolve();out=root/args.worker;out.mkdir(exist_ok=False);start=time.monotonic();status=None
 sys.path.insert(0,str(repo/'src'));os.environ['FGRL_ROOT']=str(repo)
 try:
  provenance(repo,out)
  if args.worker=='B':status=run_b(repo,out)
  elif args.worker.startswith('A_'):status=run_a(repo,out,args.worker[2:],root/'B')
  else:status=run_c(repo,out,args.worker[2:],root)
  protect(repo)
  import fluidgym
  package=pathlib.Path(fluidgym.__file__).resolve().parent
  need({str(p.relative_to(package)):sha(p) for p in package.rglob('*.py')}==EXPECTED['sources'],'FluidGym source changed during stage')
 except BaseException as exc:
  (out/'traceback.txt').write_text(traceback.format_exc());status={'status':'failed','error':repr(exc),'stage':args.worker};traceback.print_exc()
 finally:
  status=status or {'status':'failed','error':'worker ended without result'};status['wall_seconds']=time.monotonic()-start
  try:
   import torch
   status['peak_torch_cuda_allocated_bytes']=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
  except Exception:pass
  write(out/'status.json',status)
 return 0 if status['status']=='passed' else 1

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path);ap.add_argument('--execute',action='store_true');ap.add_argument('--worker',choices=['B','A_tdmpc','A_tdmpc2','C_tdmpc','C_tdmpc2']);args=ap.parse_args()
 if args.worker:
  need(args.execute and args.output is not None,'worker needs explicit execution and output');return worker(args)
 if not args.execute:
  print(json.dumps({'B':{'trials':4,'steps_per_trial':4,'max_control_steps':16,'reset':'test, seed=0, randomize=True, actual domain index observed','sequences':B_SEQUENCE,'gate_atol':ATOL,'gate_rtol':RTOL},'A':{'real_observations':2,'planner_calls_per_algorithm':22,'CFD_steps':0,'same_rng_gate_atol':1e-6,'same_rng_gate_rtol':1e-6},'C':{'gate':'B and both A pass','episodes_each':10,'horizon':80,'algorithms':2,'max_control_steps':1600},'reset_warmup':'additional native solver calls measured separately, not hidden in control budget','execution':'sequential fresh processes; stop on first failure; no retry/resume','estimated_rollout_cost':'C historical ~21–24 min plus startup/compile; not a walltime guarantee'},indent=2));return 0
 repo=args.repo.resolve();root=args.output.resolve() if args.output else repo/'reports'/('p1_run_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);root.mkdir(parents=True,exist_ok=False)
 write(root/'execution_request.json',{'argv':sys.argv,'script_sha256':sha(__file__),'expected_sha256':sha(HERE/'expected.json'),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'B_budget':16,'C_budget':1600})
 overall='failed';start=time.monotonic()
 try:
  for stage in ['B','A_tdmpc','A_tdmpc2','C_tdmpc','C_tdmpc2']:
   print('START',stage,root,flush=True)
   cmd=[sys.executable,'-B',str(pathlib.Path(__file__).resolve()),'--repo',str(repo),'--output',str(root),'--worker',stage,'--execute']
   with (root/(stage+'.log')).open('w') as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    try:
     while True:
      try:rc=child.wait(timeout=30);break
      except subprocess.TimeoutExpired:print('RUNNING',stage,'elapsed_s',round(time.monotonic()-start),flush=True)
    except BaseException:
     child.terminate()
     try:child.wait(timeout=30)
     except subprocess.TimeoutExpired:child.kill();child.wait()
     raise
   print('END',stage,'exit',rc,flush=True)
   if rc!=0 and not (root/stage/'status.json').exists():
    (root/stage).mkdir(exist_ok=True);write(root/stage/'status.json',{'status':'failed','error':'worker exited without status','returncode':rc})
   reports(root,'running')
   need(rc==0,'stage failed: '+stage+'; see '+str(root/(stage+'.log')))
  overall='passed'
 except BaseException as exc:
  write(root/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc()});print(str(exc),flush=True)
 finally:
  try:protect(repo);write(root/'preservation.json',{'status':'passed','protected_files':len(EXPECTED['required_artifacts'])+len(EXPECTED['checkpoints'])})
  except Exception as exc:overall='failed';write(root/'preservation.json',{'status':'failed','error':repr(exc)})
  reports(root,overall);write(root/'execution_cost.json',{'wall_seconds':time.monotonic()-start,'new_training_updates':0})
  archive=root.with_suffix('.tar.gz')
  with tarfile.open(archive,'x:gz') as t:t.add(root,arcname=root.name)
  print('RETURN_ARCHIVE:',archive,flush=True)
 return 0 if overall=='passed' else 1
if __name__=='__main__':sys.exit(main())
