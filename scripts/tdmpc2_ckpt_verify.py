#!/usr/bin/env python
"""Fresh-process verification of a TD-MPC2 checkpoint written by tdmpc2_smoke_train.py: load into a NEW model, compare every parameter digest with
the saved ones, and check that deterministic actor inference matches the reference action recorded before saving (and, informationally, that
seeded MPPI planning reproduces). Prints one JSON line; exit 0 only if parameters match exactly and the actor action matches."""
import argparse, json, sys, tempfile
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt-dir", required=True); ap.add_argument("--work-dir", default=str(Path(tempfile.gettempdir()) / "fgrl_work"))
ap.add_argument("--actor-tol", type=float, default=1e-6); ap.add_argument("--plan-tol", type=float, default=1e-4)
a = ap.parse_args()
d = Path(a.ckpt_dir); res = {"ok": False}
try:
    import torch
    from fluidgym_rl.tdmpc2_checks import diff_digests, state_digests
    from fluidgym_rl.tdmpc2_runtime import TDMPC2Policy, load_agent
    from fluidgym_rl.upstream import make_tdmpc2_workcopy
    wc = make_tdmpc2_workcopy(Path(a.work_dir) / "tdmpc2_patched_verify")
    cfg, agent, rc = load_agent(d / "tdmpc2_smoke.pt", d / "run_config.json", wc["code_dir"], Path(a.work_dir) / "verify_run")
    ref = json.loads((d / "reference.json").read_text())
    dd = diff_digests(ref["digests"], state_digests(agent.model.state_dict()))
    obs = torch.tensor(ref["obs"], dtype=torch.float32)
    actor = TDMPC2Policy(agent, "actor").act(obs, True).numpy(); actor_diff = float(np.abs(actor - np.array(ref["actor_action"])).max())
    torch.manual_seed(1234); plan = agent.act(obs, t0=True, eval_mode=True).numpy(); plan_diff = float(np.abs(plan - np.array(ref["plan_action_seed1234"])).max())
    res = {"digest_diff": dd, "actor_action_max_abs_diff": actor_diff, "plan_action_max_abs_diff_same_seed": plan_diff, "actor_tol": a.actor_tol, "plan_tol": a.plan_tol,
           "plan_reproducible": plan_diff <= a.plan_tol, "fresh_process_pid_note": "separate interpreter; model rebuilt from run_config.json + checkpoint only"}
    res["ok"] = (not dd["mismatched"] and not dd["missing_in_loaded"] and not dd["unexpected_in_loaded"] and actor_diff <= a.actor_tol)
except Exception as e:  # noqa: BLE001
    import traceback; res = {"ok": False, "error": repr(e), "traceback": traceback.format_exc()}
print(json.dumps(res)); sys.exit(0 if res["ok"] else 1)
