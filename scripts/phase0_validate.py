#!/usr/bin/env python
"""Phase 0 CUDA validation of a real FluidGym env. Every check is recorded in <out>/<env>/report.json.
Exit: 0 all planned checks passed, 1 failures/errors, 2 blocked (no CUDA). Nothing is assumed passed."""
import argparse, os, sys, time
from pathlib import Path

import numpy as np

from fluidgym_rl.artifacts import verify_gif, verify_png
from fluidgym_rl.report import CheckFailure, FatalStop, Report, collect_metadata
from fluidgym_rl.safety import write_gate

ap = argparse.ArgumentParser()
ap.add_argument("--env-id", default="CylinderJet2D-easy-v0")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--bench-steps", type=int, default=int(os.environ.get("FGRL_BENCH_STEPS", 20)))
ap.add_argument("--repro-tol", type=float, default=1e-5)
ap.add_argument("--out", default=os.environ.get("FGRL_OUT_DIR", "results") + "/phase0")
a = ap.parse_args()
out = Path(a.out) / a.env_id
out.mkdir(parents=True, exist_ok=True)

PLANNED = ["cuda_available", "gpu_tensor_op", "env_create", "reset_obs_contract", "action_contract", "step_contract",
           "action_bounds_behavior", "action_affects_state", "episode_termination", "reproducibility", "seed_sensitivity",
           "throughput_and_memory", "render_png", "flow_field_png", "episode_gif"]
rep = Report(out / "report.json", f"phase0:{a.env_id}", planned=PLANNED)
rep.set_meta(args=vars(a), **collect_metadata())

try:
    import torch
except ImportError:  # recorded as a failed cuda_available check below, never hidden
    torch = None

S = {}  # shared state between checks


def cuda_available():
    if torch is None:
        raise CheckFailure("torch is not importable in this environment")
    if not torch.cuda.is_available():
        raise CheckFailure("torch.cuda.is_available() is False")
    return {"device": torch.cuda.get_device_name(0)}


def gpu_tensor_op():
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(512, 512, device="cuda", dtype=torch.float64, generator=g)
    y = (x @ x).sum(); torch.cuda.synchronize()
    ref = (x.cpu() @ x.cpu()).sum()
    err = abs(float(y.cpu()) - float(ref)) / max(1.0, abs(float(ref)))
    if not torch.isfinite(y) or err > 1e-8:
        raise CheckFailure(f"GPU matmul mismatch vs CPU, rel err {err}")
    return {"rel_err_vs_cpu": err}


def env_create():
    import fluidgym
    from fluidgym_rl.envs import make_fluid_env
    S["env"] = make_fluid_env(a.env_id)
    e = S["env"]
    return {"episode_length": e.episode_length, "cuda_device": str(e.cuda_device), "use_marl": e.use_marl,
            "obs_space": str(e.observation_space), "action_space": str(e.action_space), "metrics": list(getattr(e, "metrics", []))}


def reset_obs_contract():
    e = S["env"]; obs, info = e.reset(seed=a.seed)
    S["obs0"] = obs
    d = {"type": type(obs).__name__, "shape": list(obs.shape), "dtype": str(obs.dtype), "device": str(obs.device), "info_keys": list(info)}
    if not isinstance(obs, torch.Tensor): raise CheckFailure("obs is not a torch.Tensor", d)
    if obs.ndim != 1: raise CheckFailure("expected flat 1-D obs (FlattenObservation)", d)
    if tuple(obs.shape) != tuple(e.observation_space.shape): raise CheckFailure("obs shape != observation_space.shape", d)
    if obs.device.type != "cuda": raise CheckFailure("obs not on CUDA", d)
    if obs.device != e.cuda_device: raise CheckFailure("obs device != env.cuda_device", d)
    if not torch.isfinite(obs).all(): raise CheckFailure("non-finite obs after reset", d)
    return d


def action_contract():
    e = S["env"]; act = e.sample_action(); sp = e.action_space
    d = {"shape": list(act.shape), "dtype": str(act.dtype), "device": str(act.device), "low": sp.low.tolist(), "high": sp.high.tolist()}
    if tuple(act.shape) != tuple(sp.shape): raise CheckFailure("sample_action shape != action_space.shape", d)
    if act.device != e.cuda_device: raise CheckFailure("sample_action not on env.cuda_device", d)
    lo, hi = torch.as_tensor(sp.low, device=act.device), torch.as_tensor(sp.high, device=act.device)
    if not ((act >= lo) & (act <= hi)).all(): raise CheckFailure("sampled action outside action_space bounds", d)
    S["zero"] = torch.zeros_like(act)
    return d


def step_contract():
    e = S["env"]; e.reset(seed=a.seed)
    obs, r, te, tr, info = e.step(torch.full_like(S["zero"], 0.3))
    d = {"obs_shape": list(obs.shape), "obs_dtype": str(obs.dtype), "obs_device": str(obs.device), "reward": float(r),
         "reward_type": type(r).__name__, "reward_shape": list(getattr(r, "shape", [])), "terminated": te, "truncated": tr,
         "info": {k: {"shape": list(getattr(v, "shape", [])), "device": str(getattr(v, "device", ""))} for k, v in info.items()}}
    if tuple(obs.shape) != tuple(S["obs0"].shape): raise CheckFailure("obs shape changed between reset and step", d)
    if not (torch.isfinite(obs).all() and torch.isfinite(torch.as_tensor(r)).all()): raise CheckFailure("non-finite obs/reward", d)
    if not isinstance(te, bool) or not isinstance(tr, bool): raise CheckFailure("terminated/truncated are not Python bools", d)
    if torch.as_tensor(r).numel() != 1: raise CheckFailure("reward is not scalar", d)
    return d


def action_bounds_behavior():
    """Out-of-range actions: record (not judge) whether the env clips, errors, or passes them through."""
    e = S["env"]; res = {}
    for name, val in (("hi", 1.0), ("beyond_hi", 5.0)):
        e.reset(seed=a.seed)
        try:
            res[name] = e.step(torch.full_like(S["zero"], val))[0].clone()
        except Exception as ex:  # noqa: BLE001
            res[name] = repr(ex)
    if isinstance(res["hi"], torch.Tensor) and isinstance(res["beyond_hi"], torch.Tensor):
        return {"behavior": "clipped" if torch.allclose(res["hi"], res["beyond_hi"], atol=1e-6) else "passed_through",
                "max_abs_diff": float((res["hi"] - res["beyond_hi"]).abs().max())}
    return {"behavior": "raises", "detail": {k: v if isinstance(v, str) else "ok" for k, v in res.items()}}


def action_affects_state():
    e = S["env"]; outs = []
    for v in (0.0, 1.0):
        e.reset(seed=a.seed)
        for _ in range(3): o = e.step(torch.full_like(S["zero"], v))[0]
        outs.append(o.clone())
    d = float((outs[0] - outs[1]).abs().max())
    if d == 0.0: raise CheckFailure("zero and max actuation produce identical observations", {"max_abs_diff": d})
    return {"max_abs_diff": d}


def episode_termination():
    e = S["env"]; e.reset(seed=a.seed); flags, finite = [], True
    for _ in range(e.episode_length):
        o, r, te, tr, _ = e.step(S["zero"]); finite &= bool(torch.isfinite(o).all() and torch.isfinite(r).all()); flags.append((te, tr))
    d = {"steps": len(flags), "all_finite": finite, "terminated_any": any(t for t, _ in flags), "truncated_at_last": flags[-1][1],
         "truncated_before_last": any(t for _, t in flags[:-1])}
    try:
        e.step(S["zero"]); d["step_after_done_raises"] = False
    except RuntimeError:
        d["step_after_done_raises"] = True
    if not finite or not d["truncated_at_last"] or d["truncated_before_last"] or not d["step_after_done_raises"]:
        raise CheckFailure("termination/truncation contract violated", d)
    return d


def _roll(seed, n=5):
    e = S["env"]; e.reset(seed=seed)
    return torch.stack([e.step(torch.full_like(S["zero"], 0.3))[0].clone() for _ in range(n)])


def reproducibility():
    d = float((_roll(123) - _roll(123)).abs().max())
    if d > a.repro_tol: raise CheckFailure(f"same seed+actions differ by {d} > tol {a.repro_tol}", {"max_abs_diff": d})
    return {"max_abs_diff": d, "tol": a.repro_tol}


def seed_sensitivity():
    d = float((_roll(123) - _roll(124)).abs().max())
    if d == 0.0: raise CheckFailure("different seeds gave identical rollouts")
    return {"max_abs_diff": d}


def throughput_and_memory():
    e = S["env"]; n = min(a.bench_steps, e.episode_length - 2)
    e.reset(seed=a.seed)
    for _ in range(2): e.step(e.sample_action())  # warmup
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); free0, total = torch.cuda.mem_get_info(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); e.step(e.sample_action()); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    free1, _ = torch.cuda.mem_get_info(); ts = np.array(ts)
    return {"steps": n, "env_steps_per_s": float(1 / ts.mean()), "s_per_step_mean": float(ts.mean()), "s_per_step_median": float(np.median(ts)),
            "s_per_step_p95": float(np.percentile(ts, 95)), "torch_peak_alloc_MiB": torch.cuda.max_memory_allocated() / 2**20,
            "device_used_MiB_delta_during_bench": (free0 - free1) / 2**20, "device_total_MiB": total / 2**20,
            "note": "device-level delta is authoritative if the solver allocates outside torch's allocator; paper reports ~2 s/step on A100 (not a pass criterion)"}


def render_png():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.image as mpimg
    e = S["env"]; e.reset(seed=a.seed)
    for _ in range(20): e.step(e.sample_action())
    frame = np.asarray(e.render())
    S["frame_shape"] = list(frame.shape)
    if frame.ndim not in (2, 3): raise CheckFailure(f"unexpected render() shape {frame.shape}")
    f = frame if frame.dtype == np.uint8 else (np.clip(frame, 0, 1) if frame.max() <= 1.0 else frame.astype(np.float32) / max(float(frame.max()), 1.0))
    mpimg.imsave(out / "render.png", f)
    return {"render_shape": list(frame.shape), **verify_png(out / "render.png")}


def flow_field_png():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    e = S["env"]  # state left by render_png (20 steps in)
    w = e.get_vorticity().detach().cpu().numpy().squeeze(); u = e.get_velocity().detach().cpu().numpy().squeeze()
    d = {"vorticity_shape": list(w.shape), "velocity_shape": list(u.shape), "vorticity_finite": bool(np.isfinite(w).all()),
         "vorticity_std": float(w.std()), "velocity_finite": bool(np.isfinite(u).all())}
    if w.ndim != 2: raise CheckFailure("expected 2-D vorticity for a 2-D env", d)
    ch = [i for i, s in enumerate(u.shape) if s == 2]
    if u.ndim != 3 or not ch: raise CheckFailure("cannot identify velocity component axis", d)
    speed = np.sqrt((u.astype(np.float64) ** 2).sum(ch[0])); d["speed_max"] = float(speed.max())
    if not d["vorticity_finite"] or not d["velocity_finite"] or d["vorticity_std"] == 0.0: raise CheckFailure("degenerate flow fields", d)
    def orient(x): return x.T if x.shape[0] > x.shape[1] else x  # heuristic: wide domain; array axis order is an unverified assumption
    fig, ax = plt.subplots(2, 1, figsize=(10, 5)); v = np.abs(w).max()
    ax[0].imshow(orient(w), origin="lower", cmap="RdBu_r", vmin=-v, vmax=v); ax[0].set_title("vorticity (axis order assumed)")
    ax[1].imshow(orient(speed), origin="lower"); ax[1].set_title("|u|"); fig.tight_layout(); fig.savefig(out / "flow_fields.png", dpi=150); plt.close(fig)
    return {**d, **verify_png(out / "flow_fields.png")}


def episode_gif():
    e = S["env"]; e.reset(seed=a.seed)
    for _ in range(e.episode_length):
        e.step(e.sample_action()); e.render()
    e.save_gif("episode.gif", output_path=out)
    return verify_gif(out / "episode.gif")


try:
    for name, fn, fatal in [("cuda_available", cuda_available, True), ("gpu_tensor_op", gpu_tensor_op, True), ("env_create", env_create, True),
                            ("reset_obs_contract", reset_obs_contract, True), ("action_contract", action_contract, True),
                            ("step_contract", step_contract, True), ("action_bounds_behavior", action_bounds_behavior, False),
                            ("action_affects_state", action_affects_state, False), ("episode_termination", episode_termination, False),
                            ("reproducibility", reproducibility, False), ("seed_sensitivity", seed_sensitivity, False),
                            ("throughput_and_memory", throughput_and_memory, False), ("render_png", render_png, False),
                            ("flow_field_png", flow_field_png, False), ("episode_gif", episode_gif, False)]:
        rep.check(name, fn, fatal=fatal)
except FatalStop:
    pass
status = rep.finalize()
if status == "passed":
    write_gate("phase0", out / "report.json")
print(f"phase0 status: {status}  (report: {out / 'report.json'})")
sys.exit(0 if status == "passed" else (2 if rep.data["checks"].get("cuda_available", {}).get("status") != "passed" else 1))
