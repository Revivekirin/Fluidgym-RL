"""CPU fixtures only: control gates/accounting, never CFD evidence."""
import importlib.util,pathlib,json,types
from unittest.mock import patch
import numpy as np
import pytest
P=pathlib.Path(__file__).with_name('run_p1.py')
spec=importlib.util.spec_from_file_location('p1',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Solver:
 active=None
 def __init__(self):self.calls=0;self.failed=0
 def __enter__(self):Solver.active=self;return self
 def __exit__(self,*a):pass
class Observer:
 def __init__(self):self.t=0;self.s=0;self.u=np.zeros(1,np.float32);self.applied=[]
 def memory(self):return self.u.copy()
 def clock(self):return dict(solver_time=self.s*.01,solver_steps=self.s,control_steps=self.t)
 def identity(self):return dict(index=8,split='test',domain_id='fixture')
class Env:
 def __init__(self,o):self.o=o
 def obs(self):
  v=np.full((151,2),self.o.u[0],np.float32);p=np.zeros(151,np.float32)
  return v.reshape(-1).copy(),dict(original_velocity=v,original_pressure=p)
 def reset(self,**kw):return self.obs()
 def close(self):pass
 def step(self,a):
  for _ in range(25):
   self.o.u=self.o.u+np.float32(.1)*(a-self.o.u);self.o.applied.append(self.o.u.tolist());self.o.s+=1;Solver.active.calls+=1
  self.o.t+=1;v,info=self.obs();info.update(drag=3.,lift=.1);return v,.2,False,False,info

def factory(**kw):
 o=Observer();return Env(o),object(),o

def test_b_budget_and_real_arrays(tmp_path):
 with patch.object(m,'environment',factory),patch.object(m,'effective',lambda raw:{}),patch.object(m,'SolverObserver',Solver):r=m.run_b(tmp_path,tmp_path)
 assert r['status']=='passed' and r['control_steps']==16 and r['solver_calls_including_resets']==400
 assert np.load(tmp_path/'bounded_0_observations.npz')['flat'].shape==(5,302)
 rows=m.read(tmp_path/'bounded_0_records.json');assert len(rows[2]['runtime_applied_arguments'])==25
 assert 'expected_applied_source_derived' in rows[2]

def test_b_fails_on_bad_actuator_and_preserves_partial(tmp_path):
 def bad_factory(**kw):
  env,raw,o=factory();orig=env.step
  def bad(a):
   r=orig(a);o.u+=1;return r
  env.step=bad;return env,raw,o
 with patch.object(m,'environment',bad_factory),patch.object(m,'effective',lambda raw:{}),patch.object(m,'SolverObserver',Solver),pytest.raises(RuntimeError,match='actuator memory'):
  m.run_b(tmp_path,tmp_path)
 assert m.read(tmp_path/'cost.json')['attempted_control_steps']==1
 assert (tmp_path/'zero_0_observations.npz').exists()

def test_native_flatten_rejects_reordered_observations():
 v=np.arange(302,dtype=np.float32).reshape(151,2);info=dict(original_velocity=v,original_pressure=np.zeros(151,np.float32))
 m.native_obs(v.ravel(),info)
 with pytest.raises(RuntimeError,match='flatten'):m.native_obs(v.T.ravel(),info)

def test_c_budget_blocks_before_call(tmp_path):
 env,raw,o=factory();s=Solver();c=m.ClosedObserver(env,raw,o,s,tmp_path);c.attempted=800
 with pytest.raises(RuntimeError,match='budget'):c.step(np.zeros(1,np.float32))
 assert o.t==0

def test_c_missing_gate_no_loader(tmp_path):
 for stage in ['B','A_tdmpc','A_tdmpc2']:
  (tmp_path/stage).mkdir();m.write(tmp_path/stage/'status.json',{'status':'failed' if stage=='A_tdmpc' else 'passed'})
 with pytest.raises(RuntimeError,match='gate'):m.run_c(tmp_path,tmp_path,'tdmpc',tmp_path)

def test_failed_stage_report_not_pass(tmp_path):
 (tmp_path/'B').mkdir();m.write(tmp_path/'B/status.json',{'status':'failed','error':'fixture'})
 m.reports(tmp_path,'failed');r=m.read(tmp_path/'verification.json')
 assert r['stages']['C_tdmpc']['status']=='NOT_RUN'
 assert 'UNRESOLVED — failed' in (tmp_path/'P1_PRESCRIBED_ACTION.md').read_text()

def test_identical_compare_and_nan_fail():
 assert m.compare([1],[1])['bitwise_equal']
 assert not m.compare([float('nan')],[float('nan')])['allclose']

def test_checkpoint_protection_detects_mutation(tmp_path):
 (tmp_path/'ck').write_bytes(b'original');e={'required_artifacts':{},'checkpoints':{'ck':m.sha(tmp_path/'ck')}}
 with patch.object(m,'EXPECTED',e):
  m.protect(tmp_path);(tmp_path/'ck').write_bytes(b'changed')
  with pytest.raises(RuntimeError,match='protected'):m.protect(tmp_path)

@pytest.mark.parametrize('hidden_randomness',[False,True])
def test_fixed_policy_rng_gate_and_warm_state(tmp_path,hidden_randomness):
 import contextlib,torch
 bdir=tmp_path/'B';bdir.mkdir();out=tmp_path/'A';out.mkdir();m.write(bdir/'status.json',{'status':'passed'})
 np.savez(bdir/'bounded_0_observations.npz',flat=np.ones((5,302),np.float32))
 class Model(torch.nn.Module):
  def __init__(self):super().__init__();self.encoder=torch.nn.Linear(302,2)
  def h(self,x):return self.encoder(x)
  def pi(self,z,std):return z[:,:1].tanh()
  def next(self,z,a):return z+a,a
  def Q(self,z,a):return a,a
 agent=types.SimpleNamespace(model=Model(),device='cpu')
 class Policy:
  def reset(self,seed):
   torch.manual_seed(seed);np.random.seed(seed)
   if hasattr(agent,'_prev_mean'):del agent._prev_mean
  def act(self,o,t0):
   offset=0. if t0 else agent._prev_mean.mean().item()
   draw=float(np.random.rand())
   if hidden_randomness:draw+=private_rng.random()*.01
   agent._prev_mean=torch.full((3,1),draw*.5+offset*.1)
   return torch.tensor([draw*.5+offset*.1])
 private_rng=np.random.default_rng(999)
 @contextlib.contextmanager
 def loader(*args):yield agent,Policy(),{},{}
 with patch.object(m,'agent_policy',loader),patch.object(torch.cuda,'synchronize',lambda:None):
  if hidden_randomness:
   with pytest.raises(RuntimeError,match='controlled fixed-input'):m.run_a(tmp_path,out,'tdmpc',bdir)
  else:
   r=m.run_a(tmp_path,out,'tdmpc',bdir);assert r['status']=='passed' and r['planner_calls']==22
   contrasts={x['contrast']:x for x in r['contrasts']}
   assert contrasts['same_rng_reset_state']['allclose']
   assert not contrasts['changed_rng_same_warm']['allclose']
   assert not contrasts['same_rng_changed_valid_warm']['allclose']
