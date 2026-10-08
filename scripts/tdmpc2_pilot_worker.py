#!/usr/bin/env python
"""Subprocess entry point: unchanged upstream trainer/agent, observed by the harness."""
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

# Match upstream train.py process setup before importing Torch.
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = '1'

from fluidgym_rl.training_logging import make_logger_class, scalar_metrics, wandb_config
from fluidgym_rl.tdmpc2_runtime import compose_cfg, save_run_config, smoke_replay_config


class Observer:
    def __init__(self, env, agent, cfg, torch):
        self.agent, self.cfg, self.torch = agent, cfg, torch
        self.training_steps = self.eval_steps = self.updates = 0
        self.env_seconds = self.update_seconds = 0.0
        self.started = time.perf_counter()
        self.evaluating = False
        self.eval_values = {}
        self.latest_update = {}
        self.episode_return = 0.0
        self.episode_steps = 0
        self.completed_episode = {}
        original_step, original_update = env.step, agent.update

        def step(action):
            start = time.perf_counter()
            result = original_step(action)
            self.env_seconds += time.perf_counter() - start
            if self.evaluating:
                self.eval_steps += 1
                self.eval_values.setdefault('reward', []).append(float(result[1]))
                for key in ('drag', 'lift'):
                    if key in result[3]:
                        self.eval_values.setdefault(key, []).append(float(result[3][key]))
            else:
                self.training_steps += 1
                self.episode_return += float(result[1])
                self.episode_steps += 1
                if result[2]:
                    self.completed_episode = {'episode_reward': self.episode_return, 'episode_length': self.episode_steps}
                    self.episode_return, self.episode_steps = 0.0, 0
            return result

        def update(buffer):
            from fluidgym_rl.tdmpc2_checks import check_losses
            start = time.perf_counter()
            result = original_update(buffer)
            # update() returns scalar CPU metrics upstream; timing includes that synchronization.
            self.update_seconds += time.perf_counter() - start
            self.latest_update = check_losses(result)
            self.updates += 1
            return result
        env.step, agent.update = step, update

    def metrics(self):
        elapsed = time.perf_counter() - self.started
        data = {'environment_steps': self.training_steps, 'learner_updates': self.updates,
                'evaluation_environment_steps': self.eval_steps,
                'environment_steps_per_wall_second': self.training_steps / elapsed,
                'learner_updates_per_wall_second': self.updates / elapsed,
                'episode_horizon': self.cfg.episode_length,
                'gpu_peak_allocated_bytes': self.torch.cuda.max_memory_allocated()}
        if self.env_seconds:
            data['simulator_steps_per_second'] = (self.training_steps + self.eval_steps) / self.env_seconds
        if self.updates and self.update_seconds:
            data['learner_updates_per_update_second'] = self.updates / self.update_seconds
        # Record every exposed optimizer's actual LR, without inventing missing optimizers.
        for name, value in vars(self.agent).items():
            if isinstance(value, self.torch.optim.Optimizer):
                for index, group in enumerate(value.param_groups):
                    data[f'learning_rate/{name}/{index}'] = group['lr']
        return data

    def wrap_eval(self, trainer):
        original = trainer.eval
        def evaluate():
            self.evaluating = True
            self.eval_values = {}
            try:
                result = original()
            finally:
                self.evaluating = False
            for key, values in self.eval_values.items():
                if values:
                    result['mean_reward_per_step' if key == 'reward' else key] = sum(values) / len(values)
            return result
        trainer.eval = evaluate


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    out, a = Path(spec['out']), spec['args']
    status_path = out / 'status.json'
    logger = None
    try:
        from argparse import Namespace
        from fluidgym_rl.pilot import blockers, hydra_overrides
        errors = blockers(Namespace(**a))
        if errors:
            raise RuntimeError('TRAINING BLOCKED: ' + '; '.join(errors))
        if spec['overrides'] != hydra_overrides(Namespace(**a), spec['run_id']):
            raise RuntimeError('launch overrides do not match the bounded CLI specification')
        cfg = compose_cfg(spec['workcopy']['code_dir'], spec['overrides'], out / 'run')
        # Hydra rejects unsupported overrides before any simulator is constructed.
        (out / 'hydra_config.json').write_text(json.dumps(wandb_config(cfg), indent=2))
        if a['validate_only']:
            status_path.write_text(json.dumps({'status': 'validated', 'simulation_executed': False}))
            return
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no pilot was executed')
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        from common.buffer import Buffer
        from common.logger import Logger
        from envs import make_env
        from tdmpc2 import TDMPC2
        from trainer.online_trainer import OnlineTrainer
        env = make_env(cfg)
        if a['steps'] % cfg.episode_length or a['eval_freq'] % cfg.episode_length or a['save_freq'] % cfg.episode_length:
            raise ValueError('steps, eval-freq and save-freq must be multiples of the environment episode length')
        if a['steps'] <= cfg.seed_steps:
            raise ValueError(f'budget must exceed upstream seed_steps={cfg.seed_steps}; no warmup reduction is implicit')
        cfg.work_dir = out / 'upstream'
        agent = TDMPC2(cfg)
        buffer_cfg = deepcopy(cfg)
        buffer_cfg.steps = a['steps']
        if a['mode'] == 'pilot':
            buffer_cfg = smoke_replay_config(buffer_cfg)
        buffer = Buffer(buffer_cfg)
        logger = make_logger_class(Logger)(cfg, spec)
        observer = Observer(env, agent, cfg, torch)
        logger.observer = observer
        trainer = OnlineTrainer(cfg=cfg, env=env, agent=agent, buffer=buffer, logger=logger)
        observer.wrap_eval(trainer)
        manifest = {'experiment_id': spec['run_id'], 'mode': a['mode'], 'seed': a['seed'],
                    'requested_training_environment_steps': a['steps'], 'upstream_loop_limit': cfg.steps,
                    'episode_horizon': cfg.episode_length, 'replay_capacity_rows': buffer.capacity,
                    'observation_preprocessing': 'fluidgym.wrappers.FlattenObservation -> float32; upstream TensorWrapper',
                    'reward_specification': 'unmodified FluidGym native reward; no harness normalization or rescaling',
                    'reward_formula': ('C_D,ref - mean(C_D) - mean(abs(C_L))'
                                       if a['env_id'] == 'CylinderJet2D-easy-v0' else 'see pinned environment definition'),
                    'training_split': 'train', 'periodic_evaluation_split': 'train (upstream protocol)',
                    'independent_evaluation': 'use evaluate_checkpoint.py for the common test-split contract',
                    'checkpoint_semantics': spec['checkpoint_semantics'], 'metadata': spec['metadata'],
                    'wandb': {k: a[k] for k in a if k.startswith('wandb_')}}
        if logger._wandb:
            manifest['wandb_run_id'] = logger._wandb.run.id
            manifest['wandb_run_directory'] = logger._wandb.run.dir
        (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
        (out / 'resolved_config.json').write_text(json.dumps(wandb_config(cfg), indent=2))
        import yaml
        (out / 'resolved_config.yaml').write_text(yaml.safe_dump(wandb_config(cfg)))
        save_run_config(Path(logger.model_dir) / 'run_config.json', cfg, spec['overrides'], manifest)
        if logger._wandb:
            logger._wandb.config.update({'manifest': manifest}, allow_val_change=True)
        status_path.write_text(json.dumps({'status': 'running'}))
        torch.cuda.reset_peak_memory_stats()
        trainer.train()
        if observer.training_steps != a['steps'] or observer.updates == 0:
            raise RuntimeError(f'incomplete experiment: steps={observer.training_steps}, updates={observer.updates}')
        # Upstream exits before emitting the just-completed last episode.
        if logger.last_train_step < observer.training_steps:
            final_train = {**observer.latest_update, **observer.completed_episode, **trainer.common_metrics()}
            logger.log(final_train, 'train')
        logger.save_agent(agent, identifier='final')
        final_eval = trainer.eval()
        final_eval.update(trainer.common_metrics())
        logger.log(final_eval, 'eval')
        result = {'status': 'completed', 'checkpoint': logger.checkpoints[-1],
                  'evaluation': scalar_metrics(final_eval), **observer.metrics()}
        (out / 'evaluation.json').write_text(json.dumps(result['evaluation'], indent=2))
        if logger._wandb:
            logger._wandb.run.summary.update({'status': 'completed', 'final_evaluation': result['evaluation']})
        logger.close()
        status_path.write_text(json.dumps(result, indent=2))
    except BaseException as exc:
        status_path.write_text(json.dumps({'status': 'failed', 'error': repr(exc)}))
        if logger is not None:
            try:
                logger.close(exit_code=1)
            except Exception:
                import traceback
                traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
