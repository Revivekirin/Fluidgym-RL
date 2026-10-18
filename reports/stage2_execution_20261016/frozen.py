"""Frozen TD-MPC 50k only. Existing loader and inference interfaces, no trainer."""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import p1_support as p1
from adapter import require, array

HERE=Path(__file__).resolve().parent
REFERENCE=json.loads((HERE/'frozen_reference.json').read_text())


def weight_hash(agent):
    h=hashlib.sha256()
    for network in ('model','model_target'):
        for name,tensor in sorted(getattr(agent,network).state_dict().items()):
            value=array(tensor)
            require(np.isfinite(value).all(),'nonfinite frozen parameter/buffer')
            h.update(network.encode());h.update(name.encode());h.update(str((value.shape,str(value.dtype))).encode());h.update(value.tobytes())
    return h.hexdigest()


def invariant(agent, baseline):
    require(weight_hash(agent)==baseline,'frozen online/target weights changed')
    require(vars(agent.cfg)==REFERENCE['parameters'],'planner configuration changed')
    require(float(agent.std)==REFERENCE['planner_std'],'saved planner std changed')
    require(not agent.model.training and not agent.model_target.training,'model evaluation mode changed')
    require(not agent.optim.state and not agent.pi_optim.state,'optimizer state created: unexpected update')


@contextmanager
def factory(repo):
    from fluidgym_rl.tdmpc.runtime import make_workcopy,load_algorithm
    from fluidgym_rl.tdmpc.evaluation import Policy,load_checkpoint
    from fluidgym_rl.comparison.evaluate import training_identity
    spec=p1.EXPECTED['policies']['tdmpc'];ck=repo/spec['checkpoint']
    actual=p1.read(ck.with_suffix('.json'))
    require(actual==REFERENCE,'checkpoint sidecar/config drift')
    identity=training_identity('tdmpc',repo/spec['training_run'],ck)
    require(identity['environment_steps']==50000,'not frozen 50k checkpoint')
    require(p1.read(repo/spec['selection'])['checkpoint_sha256']==p1.sha(ck)==REFERENCE['sha256'],'selection/checkpoint mismatch')
    with tempfile.TemporaryDirectory(prefix='fgrl_stage2_tdmpc_') as tmp:
        work=Path(tmp)/'tdmpc';source=make_workcopy(work);Agent,_=load_algorithm(work)
        def create():
            cfg=SimpleNamespace(**copy.deepcopy(REFERENCE['parameters']))
            require(cfg.device=='cuda:0','checkpoint device differs; no configuration override')
            agent=Agent(cfg);load_checkpoint(agent,ck);baseline=weight_hash(agent);invariant(agent,baseline)
            return agent,Policy(agent,50000),baseline
        yield create,{'training_identity':identity,'upstream':source,'parameters':REFERENCE['parameters'],
                      'planner_std':REFERENCE['planner_std'],'checkpoint_sha256':p1.sha(ck),
                      'policy_mode':'mpc_plan, eval_mode=True, step=50000, t0 only at first transition'}


def planner_capture(agent,policy,observation,t,directory):
    import torch
    before=p1.rng();warm_before=array(agent._prev_mean) if hasattr(agent,'_prev_mean') else np.empty(0,np.float32)
    with torch.no_grad(): action=array(policy.act(observation,t==0)).astype(np.float32)
    after=p1.rng();warm_after=array(agent._prev_mean)
    require(action.shape==(1,) and np.isfinite(action).all(),'invalid requested policy action')
    require(np.isfinite(warm_before).all() and np.isfinite(warm_after).all(),'nonfinite planner warm state')
    # No candidate/latent/value diagnostic calls are added.
    torch.save({'before':before,'after':after},directory/f'planner_rng_{t:02d}.pt')
    np.savez(directory/f'planner_{t:02d}.npz',warm_before=warm_before,warm_after=warm_after,
             requested_action=action,clipped_action=np.clip(action,-1,1))
    record={'t':t,'t0':t==0,'rng_before':p1.rng_hash(before),'rng_after':p1.rng_hash(after),
            'warm_before_present':bool(warm_before.size),'requested_action':action.tolist(),
            'clipped_action':np.clip(action,-1,1).tolist()}
    p1.write(directory/f'planner_{t:02d}.json',record)
    return action,record
