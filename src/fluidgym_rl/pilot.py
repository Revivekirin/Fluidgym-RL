"""Pure configuration helpers for bounded nominal runs (no simulator imports)."""
from __future__ import annotations

import argparse
import json

from fluidgym_rl.safety import full_training_blockers

PILOT_MAX_STEPS = 5000


def parser():
    p = argparse.ArgumentParser(description='Bounded TD-MPC2 pilot or explicitly gated full training.')
    p.add_argument('--mode', choices=['smoke', 'pilot', 'full'], default='smoke', help='smoke is a non-launching safety default; use tdmpc2_smoke_train.py for validation')
    p.add_argument('--allow-full-training', action='store_true')
    p.add_argument('--steps', type=int, required=True, help='training environment steps; pilot maximum 5000')
    p.add_argument('--env-id', default='CylinderJet2D-easy-v0')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--model-size', type=int, default=5)
    p.add_argument('--out', help='new, empty output directory (default: unique directory under results/tdmpc2_<mode>)')
    p.add_argument('--work-dir', help='parent of this run’s isolated patched upstream copy')
    p.add_argument('--wandb-mode', choices=['disabled', 'offline', 'online'], default='disabled')
    p.add_argument('--wandb-project', default='fluidgym-rl')
    p.add_argument('--wandb-entity')
    p.add_argument('--wandb-group', default='nominal')
    p.add_argument('--wandb-tags', nargs='*', default=[], help='space-separated tags')
    p.add_argument('--eval-freq', type=int, default=800, help='periodic upstream evaluation interval in environment steps')
    p.add_argument('--eval-episodes', type=int, default=1)
    p.add_argument('--save-freq', type=int, default=800, help='checkpoint interval in environment steps, at episode log boundaries')
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=None, help='torch.compile: pilot default off, full default on')
    p.add_argument('--validate-only', action='store_true', help='compose/validate upstream Hydra config without simulation or W&B init')
    return p


def blockers(a):
    errors = []
    if a.steps <= 0:
        errors.append('--steps must be positive')
    if a.eval_freq <= 0 or a.eval_episodes <= 0 or a.save_freq <= 0:
        errors.append('evaluation frequency and episode count must be positive')
    if a.wandb_mode != 'disabled' and (not a.wandb_project or a.wandb_project == 'none'):
        errors.append('enabled W&B requires a project other than none')
    if any(',' in tag for tag in a.wandb_tags):
        errors.append('tags must not contain commas; pass separate space-delimited tags')
    if any(not tag.strip() or len(tag) > 64 for tag in a.wandb_tags):
        errors.append('each W&B tag must contain 1–64 characters and not be blank')
    if a.mode == 'pilot':
        if a.steps > PILOT_MAX_STEPS:
            errors.append(f'pilot is limited to {PILOT_MAX_STEPS} environment steps')
    else:
        errors.extend(full_training_blockers(a.mode, a.allow_full_training))
    return errors


def wandb_environment(a, inherited=None):
    """CLI tags replace inherited tags; an omitted list must not become ['']."""
    environment = dict(inherited or {})
    environment.pop('WANDB_TAGS', None)
    environment.update(WANDB_MODE=a.wandb_mode, WANDB_RUN_GROUP=a.wandb_group)
    if a.wandb_tags:
        environment['WANDB_TAGS'] = ','.join(a.wandb_tags)
    return environment


def string_override(key, value):
    # Hydra quoted strings preserve commas, spaces, equals signs, and brackets.
    return key + '=' + json.dumps(value)


def hydra_overrides(a, run_id):
    return [string_override('task', 'fluidgym-' + a.env_id), f'steps={a.steps - 1}',
            f'seed={a.seed}', f'model_size={a.model_size}', f'eval_freq={a.eval_freq}',
            f'eval_episodes={a.eval_episodes}', 'save_video=false', 'save_agent=true', 'save_csv=true',
            f'enable_wandb={str(a.wandb_mode != "disabled").lower()}',
            string_override('wandb_project', a.wandb_project),
            'wandb_entity=null' if a.wandb_entity is None else string_override('wandb_entity', a.wandb_entity),
            string_override('exp_name', run_id), 'data_dir=none', 'checkpoint=none',
            f'compile={str(a.mode == "full" if a.compile is None else a.compile).lower()}']
