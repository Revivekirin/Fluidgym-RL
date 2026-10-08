"""Torch-free validation helpers for the TD-MPC2 smoke test (operate on arrays/tensors via .detach().cpu().numpy())."""
from __future__ import annotations

import hashlib
import math

import numpy as np

from fluidgym_rl.evalcontract import to_np

REQUIRED_LOSS_KEYS = ("consistency_loss", "reward_loss", "value_loss", "pi_loss", "total_loss")


class ReplayValidationError(AssertionError):
    pass


class NonFiniteLossError(AssertionError):
    pass


def check_losses(info: dict, required=REQUIRED_LOSS_KEYS) -> dict:
    """All required keys present and every entry finite. Returns plain floats."""
    missing = [k for k in required if k not in info]
    if missing:
        raise NonFiniteLossError(f"loss keys missing from update(): {missing}; got {sorted(info)}")
    vals = {k: float(to_np(v).mean()) for k, v in info.items()}
    bad = {k: v for k, v in vals.items() if not math.isfinite(v)}
    if bad:
        raise NonFiniteLossError(f"non-finite training statistics: {bad}")
    return vals


def validate_stored_episodes(td_obs, td_action, td_reward, td_episode, *, episode_length: int, obs_dim: int, act_dim: int, n_episodes: int) -> dict:
    """Stored layout (upstream convention): per episode episode_length+1 rows; row 0 = reset obs with NaN action/reward."""
    obs, act, rew, ep = (to_np(x) for x in (td_obs, td_action, td_reward, td_episode))
    L = episode_length + 1
    if obs.shape != (n_episodes * L, obs_dim) or act.shape != (n_episodes * L, act_dim):
        raise ReplayValidationError(f"stored shapes obs{obs.shape} act{act.shape} != expected ({n_episodes * L},{obs_dim}) / ({n_episodes * L},{act_dim})")
    ep = ep.reshape(n_episodes, L)
    if not (ep == ep[:, :1]).all() or sorted(set(ep[:, 0].tolist())) != list(range(n_episodes)):
        raise ReplayValidationError("episode ids are not constant within an episode / not 0..n-1")
    act, rew = act.reshape(n_episodes, L, act_dim), rew.reshape(n_episodes, L)
    if not (np.isnan(act[:, 0]).all() and np.isnan(rew[:, 0]).all()):
        raise ReplayValidationError("row 0 of each episode must carry NaN action/reward (upstream convention)")
    if not (np.isfinite(obs).all() and np.isfinite(act[:, 1:]).all() and np.isfinite(rew[:, 1:]).all()):
        raise ReplayValidationError("non-finite values in stored transitions")
    return {"episodes": n_episodes, "rows": n_episodes * L, "rows_per_episode": L}


def validate_sampled_batch(obs, action, reward, terminated, episode_ids, *, horizon: int, batch: int, obs_dim: int, act_dim: int) -> dict:
    """obs [H+1,B,D], action [H,B,A], reward [H,B,1], terminated [H,B,1]; episode_ids [H+1,B] must be constant along time (no sequence
    crosses an episode boundary)."""
    o, a, r, t, e = (to_np(x) for x in (obs, action, reward, terminated, episode_ids))
    exp = {"obs": ((horizon + 1, batch, obs_dim), o), "action": ((horizon, batch, act_dim), a), "reward": ((horizon, batch, 1), r), "terminated": ((horizon, batch, 1), t)}
    for k, (shape, arr) in exp.items():
        if arr.shape != shape:
            raise ReplayValidationError(f"sampled {k} shape {arr.shape} != {shape}")
    if e.shape != (horizon + 1, batch) or not (e == e[:1]).all():
        raise ReplayValidationError("a sampled sequence spans more than one episode")
    for k, (_, arr) in exp.items():
        if not np.isfinite(arr).all():
            raise ReplayValidationError(f"non-finite values in sampled {k} (NaN placeholders leaked into the training batch?)")
    return {"horizon": horizon, "batch": batch, "distinct_episodes_in_batch": int(len(np.unique(e[0])))}


def validate_actions(actions, act_dim: int, low=-1.0, high=1.0) -> dict:
    a = np.stack([to_np(x).reshape(-1) for x in actions])
    if a.shape[1] != act_dim or not np.isfinite(a).all() or a.min() < low - 1e-6 or a.max() > high + 1e-6:
        raise AssertionError(f"illegal planned actions: shape {a.shape}, range [{a.min() if a.size else None}, {a.max() if a.size else None}]")
    return {"n": int(a.shape[0]), "min": float(a.min()), "max": float(a.max())}


def tensor_digest(x) -> str:
    a = np.ascontiguousarray(to_np(x))
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def state_digests(state_dict: dict) -> dict:
    return {k: tensor_digest(v) for k, v in state_dict.items()}


def diff_digests(saved: dict, loaded: dict) -> dict:
    return {"missing_in_loaded": sorted(set(saved) - set(loaded)), "unexpected_in_loaded": sorted(set(loaded) - set(saved)),
            "mismatched": sorted(k for k in set(saved) & set(loaded) if saved[k] != loaded[k]), "n_compared": len(set(saved) & set(loaded))}
