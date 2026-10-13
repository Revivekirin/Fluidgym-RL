"""SB3 callbacks and diagnostics. SAC.train/collect_rollouts are not replaced."""
from contextlib import contextmanager
from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from fluidgym_rl.comparison.schema import write_json, select_best, independent_rows
from fluidgym_rl.sac.runtime import save_checkpoint, load_checkpoint


@contextmanager
def preserve_rng():
    py, npstate, cpu=random.getstate(),np.random.get_state(),torch.get_rng_state()
    gpu=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try: yield
    finally:
        random.setstate(py); np.random.set_state(npstate); torch.set_rng_state(cpu)
        if gpu is not None: torch.cuda.set_rng_state_all(gpu)


class GradientChecks:
    def __init__(self, model):
        self.counts={'actor':0,'critic':0,'entropy':0}; self.norms={}; self.handles=[]
        for name,module in [('actor',model.actor),('critic',model.critic)]:
            for p in module.parameters(): self.handles.append(p.register_hook(self.hook(name)))
        if model.log_ent_coef is not None: self.handles.append(model.log_ent_coef.register_hook(self.hook('entropy')))
    def hook(self,name):
        @torch.utils.hooks.unserializable_hook
        def check(grad):
            if not torch.isfinite(grad).all(): raise FloatingPointError(f'nonfinite {name} gradient')
            self.counts[name]+=1; self.norms[name]=float(grad.norm())
            return None  # Observe only; never alter gradients.
        return check
    def close(self):
        for h in self.handles: h.remove()


class Tracking:
    def __init__(self,out,settings,config,manifest):
        self.run=None
        if settings['mode']=='disabled': return
        import os,uuid,wandb
        tags=os.environ.pop('WANDB_TAGS',None)
        try:
            self.run=wandb.init(**settings,id=uuid.uuid4().hex,name=out.name,dir=str(out),
                config={'parameters':config,'provenance':manifest},resume='never' if settings['mode']=='online' else None)
        finally:
            if tags is not None: os.environ['WANDB_TAGS']=tags
        self.run.define_metric('compare/environment_steps')
        for prefix in ('compare/*','diagnostics/sac/*','train/*','resources/*'):
            self.run.define_metric(prefix,step_metric='compare/environment_steps')
        write_json(out/'wandb_run.json',dict(id=self.run.id,project=settings['project'],group=settings['group'],mode=settings['mode']))
    def log(self,steps,values):
        if self.run:
            self.run.log({'compare/environment_steps':steps,**{k:v for k,v in values.items() if v is not None}})
    def finish(self,result,code):
        if self.run:
            try: self.run.summary.update(result)
            finally: self.run.finish(exit_code=code)


def append(path,row):
    exists=path.exists()
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row))
        if not exists: w.writeheader()
        w.writerow(row)


def evaluate_checkpoint(checkpoint,out,config,root,reference=None):
    from fluidgym_rl.comparison.evaluate import evaluate_to_directory
    out.mkdir()
    args=SimpleNamespace(algorithm='sac',training_run=root,checkpoint=checkpoint,
        environment_steps=None,device=config['device'],split='val',role='candidate',
        episodes=config['eval_episodes'],base_seed=config['eval_seed'],reference=reference,selection=None)
    with preserve_rng(): evaluate_to_directory(args,out)
    return json.loads((out/'summary.json').read_text())


class NominalCallback(BaseCallback):
    def __init__(self,out,config,physics,tracker,evaluator=evaluate_checkpoint,reference=None):
        super().__init__(); self.out=out; self.config=config; self.physics=physics; self.tracker=tracker
        self.evaluator=evaluator; self.reference=reference; self.started=time.perf_counter()
        self.vals=[]; self.ret=0.; self.length=0; self.last_eval=-1; self.last_update=-1
    def _on_training_start(self):
        self.gradients=GradientChecks(self.model)
        self.initial={name:[p.detach().cpu().clone() for p in getattr(self.model,name).parameters()] for name in ('actor','critic')}
    def observe_update(self):
        if self.last_update==self.model._n_updates: return
        self.last_update=self.model._n_updates
        values={}
        for key,value in self.model.logger.name_to_value.items():
            if key.startswith('train/') and isinstance(value,(float,int,np.number)):
                if not math.isfinite(float(value)): raise FloatingPointError(f'nonfinite SB3 metric {key}')
                values['diagnostics/sac/'+key.removeprefix('train/')]=float(value)
        values['diagnostics/sac/learner_updates']=self.model._n_updates
        with (self.out/'diagnostics.jsonl').open('a') as f:
            f.write(json.dumps({'environment_steps':self.model.num_timesteps,**values})+'\n')
        self.tracker.log(self.model.num_timesteps,values)
    def _on_rollout_start(self):
        self.observe_update()
        n=self.model.num_timesteps
        if n and n%self.config['eval_freq']==0: self.validate(f'step_{n}')
    def _on_step(self):
        self.ret+=float(self.locals['rewards'][0]); self.length+=1
        if self.locals['dones'][0]:
            row=dict(environment_steps=self.model.num_timesteps,learner_updates=self.model._n_updates,
                episode_return=self.ret,mean_reward_per_step=self.ret/self.length,episode_length=self.length,
                wall_seconds=time.perf_counter()-self.started)
            append(self.out/'train.csv',row)
            # Training rollouts stay train/*, never masquerade as independent compare/train.
            self.tracker.log(self.model.num_timesteps,{'train/'+k:v for k,v in row.items()})
            print(f"sac env_steps={self.model.num_timesteps}/{self.config['steps']} updates={self.model._n_updates} return={self.ret:.4f}",flush=True)
            self.ret=0.; self.length=0
        return True
    def validate(self,name):
        ck=self.out/'checkpoints'/f'{name}.zip'
        save_checkpoint(self.model,ck,self.config,self.out.name,self.physics,time.perf_counter()-self.started)
        dest=self.out/f'validation_{self.model.num_timesteps}'
        summary=self.evaluator(ck,dest,self.config,self.out,self.reference)
        self.vals.append(dest); self.last_eval=self.model.num_timesteps
        append(self.out/'val.csv',dict(environment_steps=self.model.num_timesteps,split='val',
            mean_reward_per_step=summary['mean_reward_per_step']['mean'],episode_return=summary['episode_return']['mean'],
            drag=summary['metrics_per_step_mean']['drag']['mean'],abs_lift=summary['metrics_per_step_mean']['abs_lift']['mean'],
            drag_reduction_percent=summary.get('drag_reduction_percent'),episode_length=summary['episode_length']['mean'],
            evaluation_seed=self.config['eval_seed'],evaluation_wall_seconds=summary['eval_wall_seconds'],
            checkpoint=str(ck),checkpoint_sha256=json.loads(ck.with_suffix('.json').read_text())['sha256']))
        for row in independent_rows(dest):
            if row['status']=='available': self.tracker.log(self.model.num_timesteps,{row['metric']:row['value']})
    def finish(self):
        self.observe_update()
        self.validate('final')
        selection=select_best(self.vals,self.out/'selection.json')
        chosen=Path(selection['checkpoint'])
        shutil.copy2(chosen,self.out/'checkpoints/best.zip')
        shutil.copy2(chosen.with_suffix('.json'),self.out/'checkpoints/best.json')
        with preserve_rng(): load_checkpoint(self.out/'checkpoints/final.zip',self.config['device'])
        if self.model.num_timesteps!=self.config['steps']: raise ValueError('incomplete environment-step budget')
        if self.model._n_updates<=0 or self.model.replay_buffer.size()<=0: raise ValueError('no replay learning')
        if not all(self.gradients.counts.values()): raise ValueError('missing actor/critic/entropy gradients')
        for name,initial in self.initial.items():
            params=list(getattr(self.model,name).parameters())
            if not all(torch.isfinite(p).all() for p in params): raise FloatingPointError(f'nonfinite {name} parameters')
            if not any(not torch.equal(a,p.detach().cpu()) for a,p in zip(initial,params)): raise ValueError(f'{name} parameters unchanged')
        replay=self.model.replay_buffer
        if not replay.handle_timeout_termination: raise ValueError('horizon must bootstrap as a timeout')
        n=replay.size()
        if not np.isfinite(replay.observations[:n]).all() or not np.isfinite(replay.actions[:n]).all(): raise ValueError('invalid replay data')
        if (np.abs(replay.actions[:n])>1+1e-6).any(): raise ValueError('illegal replay actions')
        if not replay.timeouts[:n].any(): raise ValueError('no horizon truncation in replay')
        seconds=time.perf_counter()-self.started
        return dict(status='completed',algorithm='sac',environment_steps=self.model.num_timesteps,
            learner_updates=self.model._n_updates,wall_seconds=seconds,environment_steps_per_wall_second=self.model.num_timesteps/seconds,
            gradient_hook_calls=self.gradients.counts,checks={k:'passed' for k in
            ('replay_learning','finite_gradients','parameters_changed','legal_actions','horizon_timeouts','checkpoint_roundtrip','shared_validation')},
            independent_test='not_run; use compare_nominal.py after validation selection',checkpoint=str(self.out/'checkpoints/final.zip'))
