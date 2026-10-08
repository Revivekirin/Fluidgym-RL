#!/usr/bin/env python
"""Test the TD-MPC2 adapter against the REAL FluidGym env, through the patched TD-MPC2 work copy
(upstream envs.make_env + TensorWrapper). Pinned upstream clone is never modified."""
import argparse, os, sys, tempfile, types
from pathlib import Path

import numpy as np

from fluidgym_rl.report import CheckFailure, FatalStop, Report, collect_metadata
from fluidgym_rl.upstream import make_tdmpc2_workcopy

ap = argparse.ArgumentParser()
ap.add_argument("--env-id", default="CylinderJet2D-easy-v0")
ap.add_argument("--out", default=os.environ.get("FGRL_OUT_DIR", "results") + "/integration")
ap.add_argument("--work-dir", default=os.environ.get("FGRL_WORK_DIR") or str(Path(tempfile.gettempdir()) / "fgrl_work"))
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
PLANNED = ["cuda_available", "patched_workcopy", "upstream_make_env", "obs_action_conversion", "action_bounds_clipping",
           "tensorwrapper_roundtrip", "full_episode_termination"]
rep = Report(out / "report.json", f"integration:{a.env_id}", planned=PLANNED)
rep.set_meta(args=vars(a), **collect_metadata())
S = {}


class Cfg(types.SimpleNamespace):  # minimal stand-in for the hydra/dataclass cfg make_env needs
    def get(self, k, d=None): return getattr(self, k, d)


def cuda():
    import torch
    if not torch.cuda.is_available(): raise CheckFailure("CUDA unavailable")


def workcopy():
    info = make_tdmpc2_workcopy(Path(a.work_dir) / "tdmpc2_patched"); S["wc"] = info
    sys.path.insert(0, info["code_dir"]); return info


def upstream_make_env():
    import envs  # TD-MPC2's envs package from the patched work copy
    assert str(S["wc"]["code_dir"]) in os.path.abspath(envs.__file__), envs.__file__
    cfg = Cfg(task=f"fluidgym-{a.env_id}", multitask=False, obs="state", seed=0); S["cfg"] = cfg
    S["env"] = envs.make_env(cfg)
    return {"cfg_obs_shape": {k: list(v) for k, v in cfg.obs_shape.items()}, "action_dim": cfg.action_dim, "episode_length": cfg.episode_length,
            "seed_steps": cfg.seed_steps, "env_class": type(S["env"]).__name__}


def conversion():
    import torch
    inner = S["env"].env; S["inner"] = inner
    o = inner.reset()
    d = {"obs_type": type(o).__name__, "obs_dtype": str(o.dtype), "obs_shape": list(o.shape), "action_dtype": str(inner.action_space.dtype)}
    if not isinstance(o, np.ndarray) or o.dtype != np.float32 or o.ndim != 1: raise CheckFailure("adapter obs must be 1-D float32 numpy", d)
    if not np.isfinite(o).all(): raise CheckFailure("non-finite obs", d)
    if o.shape != inner.observation_space.shape: raise CheckFailure("obs shape != observation_space", d)
    o2, r, done, info = inner.step(inner.rand_act())
    d.update(reward=r, done=done, info_keys=sorted(info), info={k: v for k, v in info.items()})
    if not (np.isfinite(o2).all() and np.isfinite(r)): raise CheckFailure("non-finite step output", d)
    if not {"success", "terminated"} <= set(info): raise CheckFailure("info missing success/terminated required by TD-MPC2", d)
    d["internal_env_device"] = str(inner._env.cuda_device)
    return d


def clipping():
    inner = S["inner"]; inner.reset(); a1 = inner.step(np.ones(inner.action_space.shape, np.float32))[0]
    inner.reset(); a9 = inner.step(9 * np.ones(inner.action_space.shape, np.float32))[0]
    d = {"max_abs_diff_hi_vs_9x": float(np.abs(a1 - a9).max()), "low": inner.action_space.low.tolist(), "high": inner.action_space.high.tolist()}
    if d["max_abs_diff_hi_vs_9x"] > 1e-6: raise CheckFailure("adapter did not clip out-of-range actions", d)
    return d


def tw():
    import torch
    env = S["env"]; obs = env.reset()
    d = {"obs_type": type(obs).__name__, "obs_dtype": str(getattr(obs, "dtype", None))}
    if not isinstance(obs, torch.Tensor) or obs.dtype != torch.float32: raise CheckFailure("TensorWrapper.reset must give float32 tensor", d)
    act = env.rand_act()
    if not isinstance(act, torch.Tensor) or act.dtype != torch.float32 or act.device.type != "cpu": raise CheckFailure("rand_act must be CPU float32 tensor", d)
    obs, r, done, info = env.step(act)
    d.update(reward_type=type(r).__name__, terminated=str(info["terminated"]))
    if not isinstance(r, torch.Tensor) or not isinstance(info["terminated"], torch.Tensor): raise CheckFailure("TensorWrapper outputs wrong types", d)
    return d


def episode():
    import torch
    env = S["env"]; env.reset(); n = 0; rs = []
    while True:
        o, r, done, info = env.step(env.rand_act()); n += 1; rs.append(float(r))
        if done: break
        if n > 10 * S["cfg"].episode_length: raise CheckFailure("episode never ended")
    d = {"steps": n, "expected": S["cfg"].episode_length, "terminated": float(info["terminated"]), "reward_sum": sum(rs), "all_finite": bool(np.isfinite(rs).all())}
    if n != S["cfg"].episode_length or float(info["terminated"]) != 0.0 or not d["all_finite"]: raise CheckFailure("episode contract violated", d)
    return d


try:
    rep.check("cuda_available", cuda, fatal=True); rep.check("patched_workcopy", workcopy, fatal=True)
    rep.check("upstream_make_env", upstream_make_env, fatal=True); rep.check("obs_action_conversion", conversion, fatal=True)
    rep.check("action_bounds_clipping", clipping); rep.check("tensorwrapper_roundtrip", tw); rep.check("full_episode_termination", episode)
except FatalStop:
    pass
status = rep.finalize(); print(f"integration status: {status}"); sys.exit(0 if status == "passed" else 1)
