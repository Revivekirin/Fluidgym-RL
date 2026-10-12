"""Strict, isolated compatibility layer around the pinned original TD-MPC."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from fluidgym_rl.upstream import lock, repo_root, third_party

DEFAULT_CONFIG = repo_root() / 'configs/tdmpc/cylinderjet2d_easy.yaml'


def read_config(path=DEFAULT_CONFIG, *, seed=None, device=None):
    # JSON is YAML 1.2; accept normal YAML when PyYAML is installed as well.
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import yaml
        data = yaml.safe_load(text)
    p = data['parameters']
    if set(p) != set(data['provenance']):
        raise ValueError('every configuration parameter must have provenance')
    for key, value in [('seed', seed), ('device', device)]:
        if value is not None:
            p[key] = value
            data['provenance'][key] = dict(classification='FluidGym-specific adaptation', source='CLI override')
    if p['task'] != 'CylinderJet2D-easy-v0' or p['modality'] != 'state' or p['action_repeat'] != 1:
        raise ValueError('only nominal CylinderJet2D state observations, action_repeat=1 are supported')
    for key in ('train_steps','episode_length','horizon','batch_size','num_samples','num_elites','iterations','eval_freq','eval_episodes','save_freq','max_buffer_size','update_freq'):
        if type(p[key]) is not int or p[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if p['episode_length'] != 80 or p['obs_shape'] != [302] or p['action_dim'] != 1:
        raise ValueError('nominal contract requires horizon 80, observation 302, action 1')
    if not 0 <= p['seed_steps'] < p['train_steps'] or not 0 < p['horizon'] < p['episode_length']:
        raise ValueError('invalid seed steps or replay horizon')
    if p['train_steps'] % p['episode_length'] or p['max_buffer_size'] < p['train_steps']:
        raise ValueError('budget must contain whole episodes and buffer must hold the complete nominal run')
    if not 0 < p['num_elites'] <= p['num_samples'] or not 0 < p['discount'] <= 1:
        raise ValueError('invalid planning/discount configuration')
    if not (p['device'] == 'cpu' or p['device'] == 'cuda' or (p['device'].startswith('cuda:') and p['device'][5:].isdigit())):
        raise ValueError('device must be cpu, cuda or cuda:N')
    return data


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_workcopy(dest, source=None):
    src = Path(source or os.environ.get('FGRL_TDMPC_SOURCE') or third_party() / 'tdmpc').resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(src), *args], text=True).strip()
    expected = lock()['tdmpc'][1]
    if git('rev-parse','HEAD') != expected or git('status','--porcelain'):
        raise RuntimeError('original TD-MPC checkout must be clean and at ' + expected)
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    # Only import algorithm code; no gym 0.21 / dm-control / old upstream trainer.
    shutil.copytree(src / 'src', dest / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(src / 'LICENSE', dest / 'LICENSE')
    patch = repo_root() / 'patches/tdmpc_device.patch'
    subprocess.run(['git','init','-q',str(dest)], check=True)
    subprocess.run(['git','apply','--check',str(patch)], cwd=dest, check=True)
    subprocess.run(['git','apply',str(patch)], cwd=dest, check=True)
    before = {str(p.relative_to(src)): sha256(p) for p in (src/'src').rglob('*.py')}
    changed = [p for p, h in before.items() if sha256(dest/p) != h]
    if sorted(changed) != ['src/algorithm/helper.py','src/algorithm/tdmpc.py']:
        raise RuntimeError(f'unexpected patched files: {changed}')
    return dict(upstream_url=lock()['tdmpc'][0], upstream_sha=expected, license='MIT',
                patch_sha256=sha256(patch), source_hashes=before,
                patched_hashes={p:sha256(dest/p) for p in before})


def load_algorithm(work):
    # A separate CLI process owns this upstream namespace; refuse collisions.
    directory = str(Path(work).resolve() / 'src')
    if 'algorithm' in sys.modules and not str(sys.modules['algorithm'].__file__).startswith(directory):
        raise RuntimeError('another upstream algorithm is already imported; use a fresh process')
    sys.path.insert(0, directory)
    return importlib.import_module('algorithm.tdmpc').TDMPC, importlib.import_module('algorithm.helper')


def namespace(data):
    return SimpleNamespace(**data['parameters'])
