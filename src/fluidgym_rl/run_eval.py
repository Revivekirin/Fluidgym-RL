"""Shared driver: build env under the evaluation contract, run a Policy, write the standard result directory."""
from __future__ import annotations

import json
import logging
import platform
import sys
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

from fluidgym_rl.evalcontract import (EvalContract, attach_run_log, new_run_dir, run_evaluation, summarize, write_run)


def _ver(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def environment_versions() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(), **{p: _ver(p) for p in ("fluidgym", "torch", "stable-baselines3", "gymnasium", "numpy", "huggingface_hub")}}


def evaluate_and_write(*, policy, contract: EvalContract, out_root: str | Path, tag: str, algorithm: str, checkpoint: dict, checkpoint_sha256: str | None,
                       extra_config: dict, extra_manifest: dict, references: list[dict] | None = None, device: str = "cuda", env=None) -> Path:
    """Runs the contract on the real FluidGym env (unless `env` is injected for tests) and writes episodes.csv/summary.json/..."""
    from fluidgym_rl.report import collect_metadata
    run_dir = new_run_dir(out_root, f"{tag}_{contract.digest()[:6]}")
    log = attach_run_log(run_dir)
    log.info("run dir %s | contract %s | policy %s/%s", run_dir, contract.digest(), policy.name, policy.mode)
    if env is None:
        from fluidgym_rl.envs import make_fluid_env
        env = make_fluid_env(contract.env_id, flatten=contract.flatten_observation)
    metrics = []
    for obj in (getattr(env, "unwrapped", None), env):
        if obj is not None and getattr(obj, "metrics", None):
            metrics = list(obj.metrics)
            break
    if not metrics:
        log.warning("env exposes no `metrics`; physical metrics (drag/lift) will NOT be recorded")
    sp = env.action_space
    recs = run_evaluation(env, policy, contract, metric_names=metrics, action_low=sp.low, action_high=sp.high, log=log)
    summary = summarize(recs)
    summary["policy"] = {"algorithm": algorithm, "name": policy.name, "inference_mode": policy.mode}
    summary["physical_metrics_note"] = f"only metrics exposed by the env in info: {metrics}; drag reduction is NOT inferred from reward"
    manifest = {"algorithm": algorithm, "contract": asdict(contract), "contract_digest": contract.digest(), "checkpoint": checkpoint, "checkpoint_sha256": checkpoint_sha256,
                "train_seed": checkpoint.get("train_seed"), "simulator_version": _ver("fluidgym"), "versions": environment_versions(),
                "inference_mode": policy.mode, "metric_names": metrics, "run_metadata": collect_metadata(), **extra_manifest}
    if references:
        from fluidgym_rl.evalcontract import compare_to_reference
        summary["comparison_to_published"] = [compare_to_reference(summary, r) for r in references]
    write_run(run_dir, {"contract": asdict(contract), "algorithm": algorithm, "policy_mode": policy.mode, "device": device, **extra_config}, recs, summary, manifest)
    log.info("summary: %s", json.dumps({k: summary[k] for k in ("n_episodes", "mean_reward_per_step")}, default=str))
    logging.shutdown()
    return run_dir
