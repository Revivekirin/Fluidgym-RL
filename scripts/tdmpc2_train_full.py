#!/usr/bin/env python
"""FULL TD-MPC2 training via upstream train.py (patched work copy). BLOCKED unless every safety condition in fluidgym_rl/safety.py holds.
Default invocation does nothing but print the blockers and the estimated cost."""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

from fluidgym_rl.report import collect_metadata
from fluidgym_rl.safety import full_training_blockers
from fluidgym_rl.upstream import make_tdmpc2_workcopy

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
ap.add_argument("--allow-full-training", action="store_true")
ap.add_argument("--env-id", default="CylinderJet2D-easy-v0")
ap.add_argument("--steps", type=int, required=True, help="explicit env-step budget (no default on purpose)")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--model-size", type=int, default=5)
ap.add_argument("--out", default=os.environ.get("FGRL_OUT_DIR", "results") + "/tdmpc2_full")
ap.add_argument("--work-dir", default=os.environ.get("FGRL_WORK_DIR") or str(Path(tempfile.gettempdir()) / "fgrl_work"))
a = ap.parse_args()
out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)

s_per_step = None
try:
    p0 = Path(os.environ.get("FGRL_OUT_DIR", "results")) / "phase0" / a.env_id / "report.json"
    s_per_step = json.loads(p0.read_text())["checks"]["throughput_and_memory"]["details"]["s_per_step_mean"]
except Exception:  # noqa: BLE001
    pass
print(f"steps={a.steps}  est. env time >= {a.steps * s_per_step / 3600:.1f} GPU-h (sim only, from phase0 bench)" if s_per_step else f"steps={a.steps}  cost unknown (no phase0 bench)")
blockers = full_training_blockers(a.mode, a.allow_full_training)
if blockers:
    print("FULL TRAINING BLOCKED:\n  - " + "\n  - ".join(blockers)); sys.exit(3)

wc = make_tdmpc2_workcopy(Path(a.work_dir) / "tdmpc2_patched")
cmd = [sys.executable, str(Path(wc["code_dir"]) / "train.py"), f"task=fluidgym-{a.env_id}", f"steps={a.steps}", f"seed={a.seed}", f"model_size={a.model_size}",
       "enable_wandb=false", "save_video=false", "save_agent=true", "save_csv=true", "wandb_project=none", "wandb_entity=none", "data_dir=none", "exp_name=full"]
(out / "launch.json").write_text(json.dumps({"cmd": cmd, "workcopy": wc, "meta": collect_metadata()}, indent=2, default=str))
env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [wc["code_dir"], os.environ.get("PYTHONPATH")]))}
with open(out / "stdout.log", "w") as so, open(out / "stderr.log", "w") as se:
    rc = subprocess.run(cmd, cwd=out, env=env, stdout=so, stderr=se).returncode
(out / "exit_code.txt").write_text(str(rc)); print("exit code", rc); sys.exit(rc)
