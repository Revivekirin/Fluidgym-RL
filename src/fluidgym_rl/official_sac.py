"""Resolution and contract validation of the OFFICIAL FluidGym SB3 checkpoints.

Layout verified from the Hugging Face model card (2026-10-08): repo `safe-autonomous-systems/{algo}-{EnvId}` contains one
sub-directory per training seed (`0/ … 4/`), each with `ckpt_latest.zip`. Models were trained with fluidgym==0.0.2 and need
`fluidgym.wrappers.FlattenObservation` on newer versions. File names/revisions are resolved at run time from the Hub (the
resolved commit SHA is recorded) — nothing is hard-coded beyond the documented convention, and a missing file is a hard error."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np

HF_ORG = "safe-autonomous-systems"
CKPT_NAME = "ckpt_latest.zip"
TRAINED_WITH_FLUIDGYM = "0.0.2"          # per model card; not verifiable from the zip itself
TRAIN_SEEDS = (0, 1, 2, 3, 4)

# PUBLISHED numbers (transcribed; sources noted). Used only for side-by-side reporting.
PUBLISHED = {
    "model_card_per_seed": {   # HF model card, 'Per-Seed Statistics' (mean reward per step, rounded to 2 decimals; 'Std Dev' as printed)
        "source": "HF model card safe-autonomous-systems/sac-CylinderJet2D-easy-v0 (2026-10-08)",
        0: {"mean_reward": 0.05, "std": 0.39}, 1: {"mean_reward": 0.05, "std": 0.40}, 2: {"mean_reward": 0.05, "std": 0.37},
        3: {"mean_reward": 0.04, "std": 0.38}, 4: {"mean_reward": 0.05, "std": 0.39},
        "aggregate": {"mean_reward": 0.05},
        "conditions": "aggregated across 5 seeds; episode count/protocol/determinism not stated on the card (unverified)"},
    "paper_table8_iqm": {      # arXiv:2601.15015 Table 8 (IQM over seeds x test episodes)
        "source": "arXiv:2601.15015v1 Table 8, CylinderJet2D-easy-v0, SAC",
        "mean_reward": 0.051, "drag": 3.105, "lift": 0.032, "baseflow_drag": 3.328,
        "conditions": "5 training seeds x 10 test episodes, IQM; determinism of inference not stated in text read (unverified)"},
}


class CheckpointContractError(RuntimeError):
    """The checkpoint cannot be shown to match the evaluation contract. Message lists every mismatch found."""


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def official_repo_id(env_id: str, algo: str = "sac") -> str:
    return f"{HF_ORG}/{algo}-{env_id}"


def resolve_official_checkpoint(env_id: str, train_seed: int, algo: str = "sac", revision: str | None = None, cache_dir: str | None = None) -> dict:
    """Download `<seed>/ckpt_latest.zip` from the official repo; returns identity + local path. Raises with a clear message on failure."""
    if train_seed not in TRAIN_SEEDS:
        raise ValueError(f"train_seed must be one of {TRAIN_SEEDS} (model card lists seeds 0-4)")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:
        raise RuntimeError("huggingface_hub missing: pip install 'huggingface_hub>=1.0.0' (or pass --checkpoint-path)") from e
    repo, fname = official_repo_id(env_id, algo), f"{train_seed}/{CKPT_NAME}"
    api = HfApi()
    info = api.model_info(repo, revision=revision)
    files = [s.rfilename for s in info.siblings]
    if fname not in files:
        raise CheckpointContractError(f"{repo}@{info.sha} has no '{fname}'. Files in that seed dir: {sorted(f for f in files if f.startswith(f'{train_seed}/'))}; "
                                      f"top-level layout: {sorted(files)[:20]}")
    siblings = sorted(f for f in files if f.startswith(f"{train_seed}/"))
    path = hf_hub_download(repo, fname, revision=info.sha, cache_dir=cache_dir)
    return {"source": "huggingface", "repo_id": repo, "filename": fname, "revision_sha": info.sha, "path": path, "seed_dir_files": siblings,
            "train_seed": train_seed}


def local_checkpoint(path: str | Path, train_seed: int | None = None) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    return {"source": "local", "path": str(p), "filename": p.name, "revision_sha": None, "train_seed": train_seed, "seed_dir_files": sorted(x.name for x in p.parent.iterdir())}


def inspect_zip(path: str | Path) -> dict:
    """Read SB3 save metadata WITHOUT unpickling anything."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        read = lambda n: z.read(n).decode(errors="replace").strip() if n in names else None  # noqa: E731
        return {"members": names, "sb3_version_saved": read("_stable_baselines3_version"), "system_info": read("system_info.txt")}


def validate_contract(model, env, *, ckpt_files: list[str] | None = None, installed_sb3: str | None = None, saved_sb3: str | None = None) -> dict:
    """Check observation/action space compatibility of a loaded SB3 model against the (already FlattenObservation-wrapped) env.
    Collects ALL mismatches and raises CheckpointContractError. Never reshapes, pads, or reorders observations."""
    problems, notes = [], {}
    ms, es = model.observation_space, env.observation_space
    notes["model_obs_space"], notes["env_obs_space"] = str(ms), str(es)
    if type(ms).__name__ != "Box":
        problems.append(f"checkpoint observation space is {type(ms).__name__}, but the contract is a flat Box via FlattenObservation; "
                        "structured (v0.1 dict) observations would need an explicit, verified field-order mapping — refusing to guess")
    elif tuple(ms.shape) != tuple(es.shape):
        problems.append(f"observation dimension mismatch: checkpoint {tuple(ms.shape)} vs env {tuple(es.shape)}")
    ma, ea = model.action_space, env.action_space
    notes["model_action_space"], notes["env_action_space"] = str(ma), str(ea)
    if type(ma).__name__ != "Box" or tuple(ma.shape) != tuple(ea.shape):
        problems.append(f"action space mismatch: checkpoint {ma} vs env {ea}")
    elif not (np.allclose(ma.low, ea.low) and np.allclose(ma.high, ea.high)):
        problems.append(f"action bounds mismatch: checkpoint [{ma.low},{ma.high}] vs env [{ea.low},{ea.high}]")
    for f in ckpt_files or []:
        if any(t in f.lower() for t in ("vecnormalize", "vec_normalize", "normalize", "running_stats")):
            problems.append(f"found normalization artifact '{f}' next to the checkpoint; observation/reward normalization is NOT applied by this evaluator")
    if saved_sb3 and installed_sb3 and saved_sb3.split(".")[0] != installed_sb3.split(".")[0]:
        problems.append(f"SB3 major version differs: saved {saved_sb3} vs installed {installed_sb3}")
    notes["sb3_saved"], notes["sb3_installed"] = saved_sb3, installed_sb3
    notes["observation_field_order"] = "UNVERIFIED beyond dimension: relies on the official FlattenObservation wrapper at the installed FluidGym version"
    if problems:
        raise CheckpointContractError("checkpoint/environment contract violated:\n  - " + "\n  - ".join(problems))
    return notes


class SB3Policy:
    """Deterministic (default) or stochastic SB3 inference. Observation conversion GPU->CPU numpy happens per step (SB3 requirement)."""

    def __init__(self, model, deterministic: bool = True):
        self.model, self.deterministic = model, deterministic
        self.name, self.mode = "sac", "deterministic" if deterministic else "stochastic"

    def reset(self, episode_seed: int) -> None:
        if not self.deterministic:
            import torch
            torch.manual_seed(episode_seed)

    def act(self, obs, t0: bool):
        o = obs.detach().cpu().numpy() if hasattr(obs, "detach") else np.asarray(obs)
        a, _ = self.model.predict(o.astype(np.float32), deterministic=self.deterministic)
        return a
