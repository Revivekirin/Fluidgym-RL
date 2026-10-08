#!/usr/bin/env python
"""SHORT TD-MPC2 training smoke test on the real FluidGym env, using UPSTREAM TDMPC2/Buffer/OnlineTrainer/Logger classes from the
patched work copy (no algorithm code is changed). Only harness-level settings are reduced: seed_steps, total steps, model_size,
eval_episodes, compile off. This is NOT a training run and its returns mean nothing. ~ (seed+train episodes + 2 eval episodes) x
episode_length env steps. Not executed in the authoring sandbox."""
import argparse, dataclasses, json, os, sys, tempfile, time
from pathlib import Path

from fluidgym_rl.report import CheckFailure, FatalStop, Report, collect_metadata
from fluidgym_rl.upstream import make_tdmpc2_workcopy

ap = argparse.ArgumentParser()
ap.add_argument("--env-id", default="CylinderJet2D-easy-v0")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--seed-episodes", type=int, default=int(os.environ.get("FGRL_SMOKE_SEED_EPISODES", 2)), help="random-action episodes before updates (>=2 so the buffer is non-empty)")
ap.add_argument("--train-episodes", type=int, default=int(os.environ.get("FGRL_SMOKE_TRAIN_EPISODES", 1)))
ap.add_argument("--eval-episodes", type=int, default=1)
ap.add_argument("--model-size", type=int, default=1)
ap.add_argument("--compile", action="store_true", help="keep upstream default torch.compile=true (slower to start)")
ap.add_argument("--out", default=os.environ.get("FGRL_OUT_DIR", "results") + "/tdmpc2_smoke")
ap.add_argument("--work-dir", default=os.environ.get("FGRL_WORK_DIR") or str(Path(tempfile.gettempdir()) / "fgrl_work"))
a = ap.parse_args()
out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
PLANNED = ["cuda_available", "patched_workcopy", "build_trainer", "train_loop", "checkpoint_save", "checkpoint_load_roundtrip", "post_train_eval", "metrics_written"]
rep = Report(out / "report.json", f"tdmpc2_smoke:{a.env_id}", planned=PLANNED)
rep.set_meta(args=vars(a), mode="smoke", full_training=False, **collect_metadata())
S = {"env_steps": 0}


def cuda():
    import torch
    if not torch.cuda.is_available(): raise CheckFailure("CUDA unavailable")


def workcopy():
    S["wc"] = make_tdmpc2_workcopy(Path(a.work_dir) / "tdmpc2_patched"); sys.path.insert(0, S["wc"]["code_dir"]); return S["wc"]


def build():
    import hydra, torch
    from hydra import compose, initialize_config_dir
    run = out / "run"; run.mkdir(exist_ok=True); os.chdir(run)
    hydra.utils.get_original_cwd = os.getcwd  # harness: we compose the config ourselves instead of using @hydra.main
    with initialize_config_dir(version_base=None, config_dir=S["wc"]["code_dir"]):
        cfg = compose(config_name="config", overrides=[f"task=fluidgym-{a.env_id}", f"model_size={a.model_size}", f"seed={a.seed}", "enable_wandb=false",
                     "save_video=false", f"compile={str(a.compile).lower()}", f"eval_episodes={a.eval_episodes}", "exp_name=smoke",
                     "wandb_project=none", "wandb_entity=none", "data_dir=none", "checkpoint=none"])
    from common.parser import parse_cfg
    from common.seed import set_seed
    from common.buffer import Buffer
    from common.logger import Logger
    from envs import make_env
    from tdmpc2 import TDMPC2
    from trainer.online_trainer import OnlineTrainer
    cfg = parse_cfg(cfg); set_seed(cfg.seed)
    env = make_env(cfg)
    S["upstream_seed_steps"] = cfg.seed_steps
    cfg.seed_steps = a.seed_episodes * cfg.episode_length                       # harness-level reduction (documented)
    cfg.steps = cfg.seed_steps + a.train_episodes * cfg.episode_length
    orig = env.step
    def counted(x):
        S["env_steps"] += 1; return orig(x)
    env.step = counted
    S.update(cfg=cfg, TDMPC2=TDMPC2, torch=torch)
    S["agent"] = TDMPC2(cfg)
    S["trainer"] = OnlineTrainer(cfg=cfg, env=env, agent=S["agent"], buffer=Buffer(cfg), logger=Logger(cfg))
    return {"episode_length": cfg.episode_length, "seed_steps_used": cfg.seed_steps, "upstream_default_seed_steps": S["upstream_seed_steps"], "steps": cfg.steps,
            "horizon": cfg.horizon, "lr": cfg.lr, "batch_size": cfg.batch_size, "model_size": a.model_size, "compile": a.compile,
            "work_dir": str(cfg.work_dir), "note": "optimization hyperparameters are upstream defaults; only seed_steps/steps/eval_episodes/model_size/compile reduced"}


def train():
    t0 = time.time(); S["trainer"].train(); dt = time.time() - t0
    S["train_seconds"] = dt
    if S["trainer"]._step <= S["cfg"].seed_steps: raise CheckFailure("trainer ended before any post-seed step")
    return {"trainer_steps": S["trainer"]._step, "env_steps_taken_incl_eval": S["env_steps"], "seconds": dt, "env_steps_per_s_incl_updates_and_eval": S["env_steps"] / dt}


def ckpt_save():
    d = out / "checkpoints"; d.mkdir(exist_ok=True); p = d / "tdmpc2_smoke.pt"
    S["agent"].save(str(p))
    (d / "cfg.json").write_text(json.dumps(dataclasses.asdict(S["cfg"]), indent=2, default=str))
    if p.stat().st_size == 0: raise CheckFailure("empty checkpoint")
    S["ckpt"] = p; return {"path": str(p), "bytes": p.stat().st_size}


def ckpt_roundtrip():
    torch = S["torch"]; new = S["TDMPC2"](S["cfg"]); new.load(str(S["ckpt"]))
    a_sd, b_sd = S["agent"].model.state_dict(), new.model.state_dict()
    if a_sd.keys() != b_sd.keys(): raise CheckFailure("state_dict keys differ after load")
    bad = [k for k in a_sd if not torch.equal(a_sd[k].cpu(), b_sd[k].cpu())]
    if bad: raise CheckFailure("tensors differ after load", {"n_bad": len(bad), "first": bad[:5]})
    return {"tensors_compared": len(a_sd)}


def post_eval():
    m = S["trainer"].eval(); m = {k: float(v) for k, v in m.items()}; S["eval"] = m
    import math
    if not all(math.isfinite(v) for v in m.values()): raise CheckFailure("non-finite eval metrics", m)
    return {**m, "note": "smoke-level agent; value is meaningless as a performance number"}


def metrics():
    f = out / "metrics.json"; f.write_text(json.dumps({"eval": S.get("eval"), "train_seconds": S.get("train_seconds"), "env_steps": S["env_steps"],
                                                       "csv_logs_dir": str(S["cfg"].work_dir)}, indent=2)); return {"path": str(f)}


try:
    rep.check("cuda_available", cuda, fatal=True); rep.check("patched_workcopy", workcopy, fatal=True); rep.check("build_trainer", build, fatal=True)
    rep.check("train_loop", train, fatal=True); rep.check("checkpoint_save", ckpt_save); rep.check("checkpoint_load_roundtrip", ckpt_roundtrip)
    rep.check("post_train_eval", post_eval); rep.check("metrics_written", metrics)
except FatalStop:
    pass
status = rep.finalize(); print(f"tdmpc2 smoke status: {status}"); sys.exit(0 if status == "passed" else 1)
