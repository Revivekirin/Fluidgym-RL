#!/usr/bin/env python
"""BOUNDED TD-MPC2 smoke on the REAL FluidGym CUDA env using UPSTREAM TDMPC2/Buffer/OnlineTrainer/Logger (patched work copy).
Only harness-level knobs are reduced (seed_steps, steps, eval_episodes, model_size, compile); no algorithm code is touched. We *observe*
upstream via thin wrappers (agent.update / agent.act record outputs, then delegate unchanged).
Success requires that every check below genuinely executed. Not executed in the authoring sandbox (needs CUDA).
Checkpoint note: upstream TDMPC2.save() stores model weights ONLY (no optimizer/step/RNG) => inference restoration, NOT exact training resume."""
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path

from fluidgym_rl.report import CheckFailure, FatalStop, Report, collect_metadata
from fluidgym_rl.upstream import make_tdmpc2_workcopy

ap = argparse.ArgumentParser()
ap.add_argument("--env-id", default="CylinderJet2D-easy-v0")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--seed-episodes", type=int, default=int(os.environ.get("FGRL_SMOKE_SEED_EPISODES", 2)), help="random-action episodes before updates (>=2)")
ap.add_argument("--train-episodes", type=int, default=int(os.environ.get("FGRL_SMOKE_TRAIN_EPISODES", 1)))
ap.add_argument("--eval-episodes", type=int, default=1)
ap.add_argument("--model-size", type=int, default=1)
ap.add_argument("--compile", action="store_true", help="keep upstream default torch.compile=true")
ap.add_argument("--out-root", default=os.environ.get("FGRL_RESULTS_DIR", "results") + "/tdmpc2_smoke")
ap.add_argument("--out", help="exact output dir (default: <out-root>/<UTC stamp>)")
ap.add_argument("--work-dir", default=os.environ.get("FGRL_WORK_DIR") or str(Path(tempfile.gettempdir()) / "fgrl_work"))
a = ap.parse_args()
if a.seed_episodes < 2:
    sys.exit("--seed-episodes must be >= 2 so the replay buffer has complete episodes before the first update")
from fluidgym_rl.evalcontract import new_run_dir
out = (Path(a.out) if a.out else new_run_dir(a.out_root, "smoke")).resolve(); out.mkdir(parents=True, exist_ok=True)
PLANNED = ["cuda_available", "patched_workcopy", "build_trainer", "agent_env_device_match", "train_loop", "real_transitions", "replay_episodes_valid", "sampled_batch_valid",
           "optimizer_updates_finite", "parameters_changed", "planned_actions_legal", "post_train_eval_real_env", "checkpoint_save", "checkpoint_reload_same_process",
           "checkpoint_fresh_process", "metrics_written"]
rep = Report(out / "report.json", f"tdmpc2_smoke:{a.env_id}", planned=PLANNED)
rep.set_meta(args=vars(a), mode="smoke", full_training=False, checkpoint_semantics="inference-only restore (upstream save() = model weights only)", **collect_metadata())
S = {"env_steps": 0, "losses": [], "actions": [], "updates": 0}


def cuda():
    import torch
    if not torch.cuda.is_available(): raise CheckFailure("CUDA unavailable")
    t = torch.randn(64, 64, device="cuda"); assert bool(torch.isfinite((t @ t).sum()))


def workcopy():
    S["wc"] = make_tdmpc2_workcopy(Path(a.work_dir) / "tdmpc2_patched"); return S["wc"]


def build():
    import torch
    from fluidgym_rl import tdmpc2_runtime as rt
    from fluidgym_rl.tdmpc2_checks import check_losses, state_digests
    S["overrides"] = rt.default_overrides(a.env_id, model_size=a.model_size, seed=a.seed, eval_episodes=a.eval_episodes, compile=a.compile)
    cfg = rt.compose_cfg(S["wc"]["code_dir"], S["overrides"], out / "run")
    from common.buffer import Buffer
    from common.logger import Logger
    from envs import make_env
    from tdmpc2 import TDMPC2
    from trainer.online_trainer import OnlineTrainer
    env = make_env(cfg)
    S["upstream_seed_steps"] = cfg.seed_steps
    cfg.seed_steps = a.seed_episodes * cfg.episode_length                    # harness reduction (documented)
    cfg.steps = cfg.seed_steps + a.train_episodes * cfg.episode_length
    orig_step = env.step
    def counted(x):
        S["env_steps"] += 1; return orig_step(x)
    env.step = counted
    agent = TDMPC2(cfg)
    orig_update, orig_act = agent.update, agent.act
    def update(buffer):                                                        # observe, then delegate unchanged
        info = orig_update(buffer); S["updates"] += 1
        S["losses"].append(check_losses(info))                                 # raises on missing/non-finite statistics
        return info
    def act(obs, t0=False, eval_mode=False, task=None):
        a_ = orig_act(obs, t0=t0, eval_mode=eval_mode, task=task)
        if len(S["actions"]) < 2000: S["actions"].append(a_.detach().clone().cpu())
        return a_
    agent.update, agent.act = update, act
    S.update(cfg=cfg, rt=rt, env=env, agent=agent, TDMPC2=TDMPC2, torch=torch, digests_before=state_digests(agent.model.state_dict()))
    S["trainer"] = OnlineTrainer(cfg=cfg, env=env, agent=agent, buffer=Buffer(cfg), logger=Logger(cfg))
    rt.save_run_config(out / "run_config.json", cfg, S["overrides"], {"smoke": True})
    return {"episode_length": cfg.episode_length, "obs_shape": {k: list(v) for k, v in cfg.obs_shape.items()}, "action_dim": cfg.action_dim, "seed_steps_used": cfg.seed_steps,
            "upstream_default_seed_steps": S["upstream_seed_steps"], "steps": cfg.steps, "horizon": cfg.horizon, "batch_size": cfg.batch_size, "lr": cfg.lr,
            "model_size": a.model_size, "compile": a.compile, "episodic": cfg.episodic, "work_dir": str(cfg.work_dir),
            "note": "optimization hyperparameters = upstream defaults; reduced: seed_steps, steps, eval_episodes, model_size, compile"}


def dev_match():
    from fluidgym_rl.device import resolve_device, same_device
    inner = S["env"].env._env   # FluidGym env behind adapter
    d = {"agent_device": str(S["agent"].device), "env_cuda_device": str(inner.cuda_device), "resolved_agent": str(resolve_device(S["agent"].device)), "resolved_env": str(resolve_device(inner.cuda_device))}
    if not same_device(S["agent"].device, inner.cuda_device): raise CheckFailure("agent and simulator are on different CUDA devices", d)
    return d


def train():
    import torch
    torch.cuda.reset_peak_memory_stats(); t0 = time.time(); S["trainer"].train(); S["train_seconds"] = time.time() - t0
    S["peak_alloc_MiB"] = torch.cuda.max_memory_allocated() / 2**20
    if S["trainer"]._step <= S["cfg"].seed_steps: raise CheckFailure("trainer ended before any post-seed step")
    return {"trainer_steps": S["trainer"]._step, "seconds": S["train_seconds"], "peak_torch_alloc_MiB": S["peak_alloc_MiB"], "env_steps_incl_eval": S["env_steps"]}


def real_transitions():
    cfg, b = S["cfg"], S["trainer"].buffer
    exp_eps = a.seed_episodes + a.train_episodes
    d = {"env_steps_counted": S["env_steps"], "replay_episodes": b.num_eps, "expected_replay_episodes": exp_eps, "learner_updates": S["updates"],
         "train_steps_per_s_incl_updates_and_eval": S["env_steps"] / S["train_seconds"]}
    if b.num_eps != exp_eps or S["env_steps"] < exp_eps * cfg.episode_length or S["updates"] < 1: raise CheckFailure("transition/update counts inconsistent", d)
    return d


def replay_valid():
    from fluidgym_rl.tdmpc2_checks import validate_stored_episodes
    cfg, buf = S["cfg"], S["trainer"].buffer._buffer
    td = buf[: len(buf)]
    return validate_stored_episodes(td["obs"], td["action"], td["reward"], td["episode"], episode_length=cfg.episode_length, obs_dim=cfg.obs_shape["state"][0],
                                    act_dim=cfg.action_dim, n_episodes=S["trainer"].buffer.num_eps)


def batch_valid():
    from fluidgym_rl.tdmpc2_checks import validate_sampled_batch
    cfg, B = S["cfg"], S["trainer"].buffer
    td = B._buffer.sample().view(-1, cfg.horizon + 1).permute(1, 0)          # exactly upstream Buffer.sample()
    obs, action, reward, terminated, _ = B._prepare_batch(td)
    return validate_sampled_batch(obs, action, reward, terminated, td["episode"], horizon=cfg.horizon, batch=cfg.batch_size, obs_dim=cfg.obs_shape["state"][0], act_dim=cfg.action_dim)


def updates_finite():
    ls = S["losses"]; keys = ["consistency_loss", "reward_loss", "value_loss", "pi_loss", "total_loss"]
    return {"n_updates": len(ls), "first": {k: ls[0][k] for k in keys}, "last": {k: ls[-1][k] for k in keys}, "history_file": "loss_history.json"} if ls else (_ for _ in ()).throw(CheckFailure("no updates ran"))


def params_changed():
    from fluidgym_rl.tdmpc2_checks import diff_digests, state_digests
    d = diff_digests(S["digests_before"], state_digests(S["agent"].model.state_dict()))
    if not d["mismatched"]: raise CheckFailure("no model tensor changed during training", d)
    return {"tensors_changed": len(d["mismatched"]), "tensors_compared": d["n_compared"]}


def actions_legal():
    from fluidgym_rl.tdmpc2_checks import validate_actions
    return validate_actions(S["actions"], S["cfg"].action_dim)


def post_eval():
    import math
    m = {k: float(v) for k, v in S["trainer"].eval().items()}
    if not all(math.isfinite(v) for v in m.values()): raise CheckFailure("non-finite eval metrics", m)
    S["eval"] = m
    return {**m, "note": "smoke-level agent; meaningless as performance; upstream eval = MPPI planning (stochastic) on FluidGym TRAIN split"}


def ckpt_save():
    import torch
    from fluidgym_rl.official_sac import sha256_file
    from fluidgym_rl.tdmpc2_checks import state_digests
    d = out / "checkpoints"; d.mkdir(exist_ok=True); p = d / "tdmpc2_smoke.pt"
    S["agent"].save(str(p)); S["ckpt"] = p
    (out / "run_config.json").replace(d / "run_config.json")
    S["digests_saved"] = state_digests(S["agent"].model.state_dict())
    obs = S["env"].reset(); S["ref_obs"] = obs.cpu()
    torch.manual_seed(1234); plan = S["agent"].act(S["ref_obs"], t0=True, eval_mode=True)
    from fluidgym_rl.tdmpc2_runtime import TDMPC2Policy
    actor = TDMPC2Policy(S["agent"], "actor").act(S["ref_obs"], True)
    (d / "reference.json").write_text(json.dumps({"obs": S["ref_obs"].tolist(), "plan_action_seed1234": plan.tolist(), "actor_action": actor.tolist(), "digests": S["digests_saved"]}))
    S["sha"] = sha256_file(p)
    if p.stat().st_size == 0: raise CheckFailure("empty checkpoint")
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": S["sha"], "contains": "model weights only (no optimizer / step / RNG)"}


def ckpt_same_proc():
    from fluidgym_rl.tdmpc2_checks import diff_digests, state_digests
    new = S["TDMPC2"](S["cfg"]); new.load(str(S["ckpt"]))
    d = diff_digests(S["digests_saved"], state_digests(new.model.state_dict()))
    if d["mismatched"] or d["missing_in_loaded"] or d["unexpected_in_loaded"]: raise CheckFailure("loaded parameters differ from saved", d)
    return d


def ckpt_fresh():
    r = subprocess.run([sys.executable, str(Path(__file__).with_name("tdmpc2_ckpt_verify.py")), "--ckpt-dir", str(out / "checkpoints"), "--work-dir", a.work_dir],
                       capture_output=True, text=True)
    (out / "fresh_process_verify.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    try:
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        raise CheckFailure(f"fresh-process verifier produced no JSON (rc={r.returncode}); see fresh_process_verify.log")
    if r.returncode != 0: raise CheckFailure("fresh-process checkpoint verification failed", res)
    return res


def metrics():
    (out / "loss_history.json").write_text(json.dumps(S["losses"]))
    f = out / "metrics.json"
    f.write_text(json.dumps({"eval": S.get("eval"), "train_seconds": S.get("train_seconds"), "env_steps": S["env_steps"], "learner_updates": S["updates"], "replay_episodes": S["trainer"].buffer.num_eps,
                             "peak_torch_alloc_MiB": S.get("peak_alloc_MiB"), "checkpoint": str(S.get("ckpt")), "checkpoint_sha256": S.get("sha"), "upstream_csv_logs": str(S["cfg"].work_dir)}, indent=2))
    return {"path": str(f)}


try:
    rep.check("cuda_available", cuda, fatal=True); rep.check("patched_workcopy", workcopy, fatal=True); rep.check("build_trainer", build, fatal=True)
    rep.check("agent_env_device_match", dev_match, fatal=True); rep.check("train_loop", train, fatal=True)
    for n, f in [("real_transitions", real_transitions), ("replay_episodes_valid", replay_valid), ("sampled_batch_valid", batch_valid), ("optimizer_updates_finite", updates_finite),
                 ("parameters_changed", params_changed), ("planned_actions_legal", actions_legal), ("post_train_eval_real_env", post_eval), ("checkpoint_save", ckpt_save)]:
        rep.check(n, f)
    rep.check("checkpoint_reload_same_process", ckpt_same_proc); rep.check("checkpoint_fresh_process", ckpt_fresh); rep.check("metrics_written", metrics)
except FatalStop:
    pass
status = rep.finalize(); print(f"tdmpc2 smoke status: {status}  ({out})"); sys.exit(0 if status == "passed" else 1)
