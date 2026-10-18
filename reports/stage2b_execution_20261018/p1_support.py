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

