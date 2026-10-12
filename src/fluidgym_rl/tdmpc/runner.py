"""Bounded experiment orchestration; upstream TOLD losses/planner/PER are unchanged."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace
import numpy as np
import torch
from fluidgym_rl.tdmpc.evaluation import Policy, evaluate, save_checkpoint

TRAIN_FIELDS = ['environment_steps','learner_updates','episode','episode_return','mean_reward_per_step','episode_length','wall_seconds','environment_seconds','learner_seconds','environment_steps_per_second','learner_updates_per_second','grad_norm','consistency_loss','reward_loss','value_loss','pi_loss','total_loss','weighted_loss','action_clipped_fraction','learning_rate','gpu_peak_memory_bytes']
EVAL_FIELDS = ['environment_steps','learner_updates','split','episodes','mean_reward_per_step','episode_return','drag','lift','abs_lift','checkpoint']


def append_csv(path, fields, row):
    exists = Path(path).exists()
    with open(path,'a',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k:row.get(k,'') for k in fields})


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def finite_update(agent, buffer, step):
    metrics = agent.update(buffer, step)
    if not all(math.isfinite(float(v)) for v in metrics.values()):
        raise FloatingPointError(f'non-finite learner metrics: {metrics}')
    for name, param in agent.model.named_parameters():
        if not torch.isfinite(param).all() or (param.grad is not None and not torch.isfinite(param.grad).all()):
            raise FloatingPointError(f'non-finite parameter/gradient: {name}')
    return metrics


def run_training(cfg, agent, helper, env, out, manifest, *, budget, smoke=False):
    if budget % cfg.episode_length:
        raise ValueError('budget must consist of whole episodes')
    # Native ring requires whole-episode capacity. Keep an unused episode for small
    # smoke buffers so upstream sampling-without-replacement is not under-populated.
    capacity = max(budget, (cfg.batch_size//(cfg.episode_length-cfg.horizon)+2)*cfg.episode_length)
    bc = SimpleNamespace(**vars(cfg)); bc.train_steps = capacity; bc.max_buffer_size = capacity
    buffer = helper.ReplayBuffer(bc)
    initial = {n:p.detach().clone() for n,p in agent.model.named_parameters()} if smoke else None
    updates = steps = 0
    best = -float('inf')
    elapsed_env = elapsed_learn = 0.
    start = time.perf_counter()
    next_eval, next_save = cfg.eval_freq, cfg.save_freq
    validations = []
    for ep in range(budget//cfg.episode_length):
        env.train()
        before_env = time.perf_counter()
        obs, _ = env.reset(seed=cfg.seed+ep, randomize=True)
        episode = helper.Episode(cfg, obs)
        clipped = 0
        episode_start = steps
        # Upstream changes horizon at episode boundaries, avoiding warm-start shape changes.
        while not episode.done:
            with torch.no_grad():
                if steps < cfg.seed_steps:
                    raw = torch.empty(cfg.action_dim,device=cfg.device).uniform_(-1,1)
                else:
                    raw = agent.plan(obs, step=max(cfg.seed_steps,episode_start), t0=episode.first)
            if not torch.isfinite(raw).all():
                raise FloatingPointError('non-finite planned action')
            action = raw.clamp(-1,1)
            clipped += int(not torch.equal(raw, action))
            obs, reward, term, trunc, _ = env.step(action)
            episode += (obs, action, reward, term or trunc)
            steps += 1
        if len(episode) != cfg.episode_length:
            raise ValueError('incomplete episode cannot enter original replay')
        buffer.add(episode)
        elapsed_env += time.perf_counter()-before_env
        metrics = {}
        if steps >= cfg.seed_steps:
            t = time.perf_counter()
            # 100 is not divisible by 80: warm-up is completed at the next replay
            # episode boundary. Subsequent episodes retain upstream 1 update/step.
            count = cfg.seed_steps if updates == 0 else cfg.episode_length
            for i in range(count):
                metrics = finite_update(agent, buffer, episode_start+i)
                updates += 1
            elapsed_learn += time.perf_counter()-t
        row = dict(environment_steps=steps,learner_updates=updates,episode=ep+1,
                   episode_return=episode.cumulative_reward, mean_reward_per_step=episode.cumulative_reward/len(episode),
                   episode_length=len(episode),wall_seconds=time.perf_counter()-start,
                   environment_seconds=elapsed_env,learner_seconds=elapsed_learn,
                   environment_steps_per_second=steps/elapsed_env,
                   action_clipped_fraction=clipped/len(episode),learning_rate=cfg.lr,**metrics)
        if updates:
            row['learner_updates_per_second']=updates/elapsed_learn
        if str(cfg.device).startswith('cuda'):
            row['gpu_peak_memory_bytes']=torch.cuda.max_memory_allocated(cfg.device)
        append_csv(out/'train.csv',TRAIN_FIELDS,row)
        print(f'train env_steps={steps}/{budget} updates={updates} return={episode.cumulative_reward:.4f}',flush=True)
        if steps >= next_eval or steps == budget:
            result = evaluate(env,Policy(agent,steps),cfg,out/'validation'/f'step_{steps}',
                              split='val',episodes=1 if smoke else cfg.eval_episodes,seed=cfg.seed+10000,manifest=manifest)
            score = result['mean_reward_per_step']['mean']
            checkpoint = ''
            if score > best:
                best = score
                checkpoint = str(out/'checkpoints/best.pt')
                save_checkpoint(agent,checkpoint,cfg,steps,manifest)
            values = result['metrics_per_step_mean']
            append_csv(out/'eval.csv',EVAL_FIELDS,dict(environment_steps=steps,learner_updates=updates,split='val',episodes=result['n_episodes'],
                       mean_reward_per_step=score,episode_return=result['episode_return']['mean'],checkpoint=checkpoint,
                       **{m:values[m]['mean'] for m in ('drag','lift','abs_lift')}))
            validations.append(dict(step=steps,score=score))
            next_eval = (steps//cfg.eval_freq+1)*cfg.eval_freq
        if steps >= next_save and steps != budget:
            # Single rolling periodic file avoids large duplicate artifact sets.
            save_checkpoint(agent,out/'checkpoints/latest.pt',cfg,steps,manifest)
            next_save = (steps//cfg.save_freq+1)*cfg.save_freq
    save_checkpoint(agent,out/'checkpoints/final.pt',cfg,steps,manifest)
    if updates == 0:
        raise RuntimeError('no learner updates; experiment incomplete')
    if smoke:
        if not any(not torch.equal(initial[n],p) for n,p in agent.model.named_parameters()):
            raise RuntimeError('model parameters did not change')
        batch = buffer.sample()
        if batch[0].shape != (cfg.batch_size,*cfg.obs_shape) or not all(torch.isfinite(t).all() for t in batch):
            raise RuntimeError('invalid sampled replay batch')
        idxs = batch[-2]
        if not torch.all(idxs%cfg.episode_length < cfg.episode_length-cfg.horizon):
            raise RuntimeError('replay sequence crosses episode boundary')
    return dict(status='completed',environment_steps=steps,learner_updates=updates,wall_seconds=time.perf_counter()-start,
                best_validation_reward_per_step=best,validation=validations,replay_capacity=capacity,
                checkpoint=str(out/'checkpoints/final.pt'),checkpoint_semantics='inference only; no exact training resume',
                gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(cfg.device) if str(cfg.device).startswith('cuda') else None)
