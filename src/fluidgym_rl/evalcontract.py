"""Common evaluation contract for FluidGym checkpoints (SAC today; TD-MPC2, PPO, D-MPC later).

Everything algorithm-independent lives here: which env/split/reset protocol/seeds are used, which per-episode quantities are
recorded, how results are written, and how two results may (or may not) be compared. Algorithm-specific inference stays in a
`Policy` object. Torch is only touched lazily so the logic is unit-testable on CPU.

Reset protocol 'official_callback' mirrors fluidgym.integration.sb3.EvalCallback/evaluate_model as read at the pinned SHA:
env.test() split, episode 0 reset(randomize=False), later episodes reset(randomize=True), env.seed(base_seed) once, metrics
taken from info[m] for m in env.unwrapped.metrics (cylinders: 'drag', 'lift'), mean_reward = mean over steps of reward.
Whether the *paper's* test numbers used exactly this protocol is NOT verified (see docs/REPRODUCTION_AUDIT.md)."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

CONTRACT_VERSION = 1
PROTOCOLS = ("official_callback", "seeded")


class EvaluationError(RuntimeError):
    """Raised for invalid evaluation data (non-finite values, contract violations). Never swallowed."""


def to_np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def scalar(x: Any) -> float:
    a = to_np(x).astype(np.float64)
    if a.size == 0:
        raise EvaluationError("empty value where a scalar was expected")
    return float(a.mean())


@dataclass(frozen=True)
class EvalContract:
    env_id: str = "CylinderJet2D-easy-v0"
    split: str = "test"
    n_episodes: int = 10
    protocol: str = "official_callback"
    base_seed: int = 0
    flatten_observation: bool = True
    version: int = CONTRACT_VERSION

    def __post_init__(self):
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}")
        if self.split not in ("train", "val", "test"):
            raise ValueError("split must be train|val|test")
        if self.n_episodes < 1:
            raise ValueError("n_episodes must be >= 1")

    def reset_kwargs(self, i: int) -> dict:
        if self.protocol == "official_callback":
            return {"randomize": i > 0}
        return {"seed": self.base_seed + i, "randomize": True}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


class Policy(Protocol):
    name: str
    mode: str                       # e.g. 'deterministic', 'stochastic', 'mpc_plan', 'actor_mean'

    def reset(self, episode_seed: int) -> None: ...
    def act(self, obs: Any, t0: bool) -> Any: ...


@dataclass
class EpisodeRecord:
    episode: int
    reset_kwargs: dict
    ret: float
    mean_reward: float
    length: int
    terminated: bool
    truncated: bool
    action_mean: float
    action_std: float
    action_abs_max: float
    action_clipped_frac: float
    metrics: dict = field(default_factory=dict)       # per-step mean of info[m] (physical metrics reported by the env only)
    wall_seconds: float = 0.0


class _EnvIO:
    """Tensor/array conversion for real FluidGym envs (torch, CUDA) and CPU stand-ins."""

    def __init__(self, env):
        self.env, self.torch_backend = env, hasattr(env, "cuda_device")

    def action_in(self, a: np.ndarray):
        if not self.torch_backend:
            return a
        import torch
        return torch.as_tensor(a, dtype=torch.float32, device=self.env.cuda_device)


def run_evaluation(env, policy: Policy, contract: EvalContract, *, metric_names: list[str], action_low, action_high,
                   log: logging.Logger | None = None) -> list[EpisodeRecord]:
    log = log or logging.getLogger("fluidgym_rl.eval")
    io = _EnvIO(env)
    getattr(env, contract.split)()                      # env.train()/val()/test()
    if contract.protocol == "official_callback" and hasattr(env, "seed"):
        env.seed(contract.base_seed)
    low, high = np.asarray(action_low, np.float32), np.asarray(action_high, np.float32)
    max_steps = 10 * int(getattr(env, "episode_length", 10_000))
    recs: list[EpisodeRecord] = []
    for i in range(contract.n_episodes):
        t0 = time.perf_counter()
        kw = contract.reset_kwargs(i)
        out = env.reset(**kw)
        obs = out[0] if isinstance(out, tuple) else out
        if not np.isfinite(to_np(obs)).all():
            raise EvaluationError(f"episode {i}: non-finite observation after reset")
        policy.reset(contract.base_seed + i)
        rewards, acts, clipped, mvals = [], [], 0, {m: [] for m in metric_names}
        term = trunc = False
        step = 0
        while not (term or trunc):
            raw = np.asarray(to_np(policy.act(obs, t0=(step == 0))), np.float32).reshape(low.shape)
            if not np.isfinite(raw).all():
                raise EvaluationError(f"episode {i} step {step}: policy produced non-finite action {raw}")
            act = np.clip(raw, low, high)
            clipped += int((act != raw).any())
            obs, r, term, trunc, info = env.step(io.action_in(act))
            term, trunc = bool(term), bool(trunc)
            r = scalar(r)
            if not (math.isfinite(r) and np.isfinite(to_np(obs)).all()):
                raise EvaluationError(f"episode {i} step {step}: non-finite reward/observation")
            rewards.append(r); acts.append(act.copy())
            for m in metric_names:
                if m not in info:
                    raise EvaluationError(f"metric '{m}' missing from info (keys: {sorted(info)})")
                mvals[m].append(scalar(info[m]))
            step += 1
            if step > max_steps:
                raise EvaluationError(f"episode {i} exceeded {max_steps} steps without terminating/truncating")
        a = np.stack(acts)
        rec = EpisodeRecord(i, kw, float(np.sum(rewards)), float(np.mean(rewards)), step, term, trunc, float(a.mean()), float(a.std()),
                            float(np.abs(a).max()), clipped / step, {m: float(np.mean(v)) for m, v in mvals.items()}, time.perf_counter() - t0)
        recs.append(rec)
        log.info("episode %d: return=%.4f mean_reward=%.4f len=%d term=%s trunc=%s %s", i, rec.ret, rec.mean_reward, step, term, trunc, rec.metrics)
    return recs


def iqm(x) -> float:
    x = np.sort(np.asarray(x, float)); n = len(x); lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def summarize(recs: list[EpisodeRecord]) -> dict:
    def stats(v):
        v = np.asarray(v, float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "median": float(np.median(v)), "iqm": iqm(v),
                "min": float(v.min()), "max": float(v.max())}
    names = sorted({m for r in recs for m in r.metrics})
    return {"n_episodes": len(recs), "mean_reward_per_step": stats([r.mean_reward for r in recs]), "episode_return": stats([r.ret for r in recs]),
            "episode_length": stats([r.length for r in recs]), "metrics_per_step_mean": {m: stats([r.metrics[m] for r in recs]) for m in names},
            "action": {"mean": float(np.mean([r.action_mean for r in recs])), "std_within_episode": float(np.mean([r.action_std for r in recs])),
                       "abs_max": float(max(r.action_abs_max for r in recs)), "clipped_step_fraction": float(np.mean([r.action_clipped_frac for r in recs]))},
            "terminated_episodes": int(sum(r.terminated for r in recs)), "truncated_episodes": int(sum(r.truncated for r in recs)),
            "eval_wall_seconds": float(sum(r.wall_seconds for r in recs))}


def new_run_dir(root: str | Path, tag: str) -> Path:
    """Create results/<root>/<UTC stamp>_<tag>[_n]; never reuses an existing directory."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    base = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{tag}"
    for n in range(100):
        d = root / (base if n == 0 else f"{base}_{n}")
        try:
            d.mkdir()
            return d
        except FileExistsError:
            continue
    raise RuntimeError("could not create a unique run directory")


def attach_run_log(run_dir: Path) -> logging.Logger:
    lg = logging.getLogger("fluidgym_rl.eval"); lg.setLevel(logging.INFO)
    for h in list(lg.handlers):
        lg.removeHandler(h)
    for h in (logging.FileHandler(run_dir / "evaluation.log"), logging.StreamHandler()):
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s")); lg.addHandler(h)
    return lg


def write_run(run_dir: Path, resolved_config: dict, recs: list[EpisodeRecord], summary: dict, manifest: dict) -> None:
    """resolved_config.yaml is JSON (valid YAML 1.2) so no PyYAML dependency is required."""
    (run_dir / "resolved_config.yaml").write_text(json.dumps(resolved_config, indent=2, default=str))
    metric_names = sorted({m for r in recs for m in r.metrics})
    with open(run_dir / "episodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reset_kwargs", "return", "mean_reward", "length", "terminated", "truncated", "action_mean", "action_std",
                    "action_abs_max", "action_clipped_frac", *[f"mean_{m}" for m in metric_names], "wall_seconds"])
        for r in recs:
            w.writerow([r.episode, json.dumps(r.reset_kwargs), r.ret, r.mean_reward, r.length, r.terminated, r.truncated, r.action_mean, r.action_std,
                        r.action_abs_max, r.action_clipped_frac, *[r.metrics.get(m, "") for m in metric_names], r.wall_seconds])
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


# ----------------------------------------------------------------------------- comparison

def classify_comparison(a: dict, b: dict) -> dict:
    """Decide what kind of comparison two run manifests permit. Keys read from each manifest:
    contract (dict), algorithm, checkpoint_sha256, train_seed, simulator_version (installed fluidgym)."""
    reasons: list[str] = []
    ca, cb = a["contract"], b["contract"]
    for k in ("env_id", "split", "protocol", "flatten_observation", "n_episodes", "base_seed"):
        if ca.get(k) != cb.get(k):
            reasons.append(f"contract.{k} differs: {ca.get(k)!r} vs {cb.get(k)!r}")
    if reasons:
        return {"class": "non_comparable", "reasons": reasons}
    if a["algorithm"] != b["algorithm"]:
        return {"class": "cross_algorithm_same_contract", "reasons": ["same env/split/protocol/seeds, different algorithm; algorithm-specific inference modes differ"]}
    same_ckpt = a.get("checkpoint_sha256") == b.get("checkpoint_sha256") and a.get("checkpoint_sha256") is not None
    same_sim = a.get("simulator_version") == b.get("simulator_version")
    if same_ckpt and same_sim:
        return {"class": "same_checkpoint_same_contract", "reasons": []}
    if same_ckpt:
        return {"class": "same_environment_different_implementation_version", "reasons": [f"simulator {a.get('simulator_version')} vs {b.get('simulator_version')}"]}
    return {"class": "same_algorithm_different_training_seed" if a.get("train_seed") != b.get("train_seed") else "same_algorithm_different_checkpoint",
            "reasons": ["different checkpoint artifacts"]}


def compare_to_reference(summary: dict, reference: dict, *, indicative_tol: float = 0.01) -> dict:
    """Report differences to published numbers. Deliberately makes NO pass/fail 'reproduced' claim."""
    got = summary["mean_reward_per_step"]["mean"]
    out = {"reference": reference.get("source"), "reference_conditions": reference.get("conditions"), "claims_reproduction": False, "rows": []}
    for key, label in (("mean_reward", "mean_reward_per_step"),):
        if key in reference:
            ref = reference[key]
            out["rows"].append({"metric": label, "measured": got, "published": ref, "abs_diff": got - ref, "within_indicative_tol": abs(got - ref) <= indicative_tol,
                                "indicative_tol": indicative_tol})
    for m, key in (("drag", "drag"), ("lift", "lift")):
        if key in reference and m in summary["metrics_per_step_mean"]:
            got_m = summary["metrics_per_step_mean"][m]["mean"]
            out["rows"].append({"metric": f"mean_{m}", "measured": got_m, "published": reference[key], "abs_diff": got_m - reference[key]})
    return out
