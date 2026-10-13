#!/usr/bin/env python
"""Opt-in, validation-only frozen TD-MPC2 diagnostics; never trains or selects on test."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-run', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--mode', choices=['plan', 'actor'], default='plan')
    p.add_argument('--horizon', type=int, choices=[1, 3, 5, 10])
    p.add_argument('--episodes', type=int, choices=range(1, 11), default=2)
    p.add_argument('--env-seed', type=int, default=0)
    p.add_argument('--planner-seed', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--execute', action='store_true', help='otherwise print budget only; no imports of CUDA/simulator')
    return p


def diagnostic_overrides(saved, horizon):
    overrides = list(saved)
    if horizon is not None:
        overrides = [x for x in overrides if x.split('=', 1)[0] != 'horizon']
        overrides.append(f'horizon={horizon}')
    return overrides


def main():
    a = parser().parse_args()
    if a.mode == 'actor' and a.horizon is not None:
        raise ValueError('--horizon only applies to plan mode')
    for k in ('training_run', 'checkpoint', 'out'):
        setattr(a, k, getattr(a, k).resolve())
    print(json.dumps(dict(split='val', episodes=a.episodes, simulator_steps=80*a.episodes,
                         mode=a.mode, horizon=a.horizon or 'saved',
                         env_seeds=list(range(a.env_seed,a.env_seed+a.episodes)),
                         planner_seeds=list(range(a.planner_seed,a.planner_seed+a.episodes)),
                         execution_requested=a.execute), indent=2), flush=True)
    if not a.execute:
        return 0
    a.out.mkdir(parents=True, exist_ok=False)
    try:
        import numpy as np
        import torch
        from types import SimpleNamespace
        from fluidgym_rl.comparison.evaluate import training_identity, NominalEnv, physics_manifest, aggregate
        from fluidgym_rl.comparison.schema import file_hash, write_json
        from fluidgym_rl.envs import make_fluid_env
        from fluidgym_rl.evalcontract import EvalContract, run_evaluation, write_run
        from fluidgym_rl.tdmpc2_runtime import load_agent, TDMPC2Policy
        from fluidgym_rl.training_logging import wandb_config
        from fluidgym_rl.upstream import make_tdmpc2_workcopy
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; diagnostic not executed')
        torch.cuda.set_device(0)
        # Match the common evaluator's default numerical policy, not a new TF32 override.
        identity = training_identity('tdmpc2', a.training_run, a.checkpoint)
        before = file_hash(a.checkpoint)
        rc = json.loads((a.checkpoint.parent / 'run_config.json').read_text())
        saved_overrides = list(rc['overrides'])
        rc['overrides'] = diagnostic_overrides(saved_overrides, a.horizon)
        contract = EvalContract(split='val', n_episodes=a.episodes, base_seed=a.env_seed, protocol='seeded')
        latencies = []
        with tempfile.TemporaryDirectory(prefix='fgrl_diagnostic_') as tmp:
            temp = Path(tmp)
            wc = make_tdmpc2_workcopy(temp / 'upstream')
            config_path = temp / 'run_config.json'
            config_path.write_text(json.dumps(rc))
            cfg, agent, _ = load_agent(a.checkpoint, config_path, wc['code_dir'], temp / 'run')
            if list(cfg.obs_shape['state']) != [302] or cfg.action_dim != 1 or cfg.episode_length != 80:
                raise ValueError('checkpoint is not compatible with nominal contract')
            inner = TDMPC2Policy(agent, a.mode)
            class TimedPolicy:
                name, mode = 'tdmpc2', inner.mode
                def reset(self, seed):
                    planner_seed = a.planner_seed + seed - a.env_seed
                    random.seed(planner_seed)
                    np.random.seed(planner_seed)
                    inner.reset(planner_seed)
                def act(self, obs, t0):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    action = inner.act(obs, t0)
                    torch.cuda.synchronize()
                    latencies.append(time.perf_counter() - start)
                    return action
            env = NominalEnv(make_fluid_env(mode='val', cuda_device=torch.device('cuda:0')),
                             SimpleNamespace(obs_shape=[302], action_dim=1, episode_length=80, device='cuda:0'))
            try:
                physics = physics_manifest(env)
                records = run_evaluation(env, TimedPolicy(), contract, metric_names=['drag','lift','abs_lift'],
                                         action_low=env.action_space.low, action_high=env.action_space.high)
            finally:
                env.close()
            assert file_hash(a.checkpoint) == before, 'checkpoint changed'
            summary = aggregate(records)
            summary['action_latency_seconds'] = dict(first=latencies[0], mean=float(np.mean(latencies)),
                median=float(np.median(latencies)), p95=float(np.quantile(latencies,.95)),
                mean_excluding_first=float(np.mean(latencies[1:])),
                note='synchronized wall latency includes compilation/copy; excluding first is not guaranteed steady state')
            manifest = dict(status='completed', kind='validation_only_diagnostic', identity=identity,
                checkpoint_sha256=before, physics=physics, contract=asdict(contract),
                policy_mode=inner.mode, planner_base_seed=a.planner_seed, resolved_config=wandb_config(cfg),
                saved_overrides=saved_overrides, effective_overrides=rc['overrides'],
                upstream=wc['upstream'], patch_sha256=file_hash(wc['patch']),
                torch_version=torch.__version__, device=torch.cuda.get_device_name(0),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                script_sha256=file_hash(Path(__file__)),
                interpretation='inference-only ablation; changed horizons are outside the saved training configuration; no causal world-model claim')
            write_run(a.out, wandb_config(cfg), records, summary, manifest)
            write_json(a.out / 'action_latencies.json', latencies)
            manifest['result_hashes'] = {n:file_hash(a.out/n) for n in
                ('episodes.csv','summary.json','resolved_config.yaml','run_manifest.json','action_latencies.json')}
            write_json(a.out / 'diagnostic_manifest.json', manifest)
        return 0
    except BaseException as exc:
        (a.out / 'traceback.txt').write_text(traceback.format_exc())
        (a.out / 'failure.json').write_text(json.dumps(dict(status='failed', error=repr(exc))))
        raise


if __name__ == '__main__':
    sys.exit(main())
