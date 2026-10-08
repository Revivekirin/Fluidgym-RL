#!/usr/bin/env python
"""Unified evaluation entry point (same env/split/protocol/seeds/metrics for every algorithm; inference mode stays algorithm-specific).
  sac    : any local SB3 SAC .zip (for official checkpoints prefer scripts/evaluate_official_sac.py)
  tdmpc2 : checkpoint written by scripts/tdmpc2_smoke_train.py / upstream train.py + its run_config.json
  compare: classify whether two result directories are comparable
Not executed in the authoring sandbox (needs CUDA)."""
import argparse, json, os, sys, tempfile
from pathlib import Path

from fluidgym_rl.evalcontract import EvalContract, classify_comparison
from fluidgym_rl.run_eval import evaluate_and_write

ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest="cmd", required=True)
for name in ("sac", "tdmpc2"):
    p = sub.add_parser(name)
    p.add_argument("--checkpoint", required=True); p.add_argument("--env-id", default="CylinderJet2D-easy-v0")
    p.add_argument("--episodes", type=int, default=10); p.add_argument("--protocol", choices=["official_callback", "seeded"], default="official_callback")
    p.add_argument("--base-seed", type=int, default=0); p.add_argument("--out-root", default=os.environ.get("FGRL_RESULTS_DIR", "results") + "/eval")
    p.add_argument("--train-seed", type=int)
    if name == "sac":
        p.add_argument("--stochastic", action="store_true")
    else:
        p.add_argument("--run-config", help="run_config.json next to the checkpoint (default: <ckpt dir>/run_config.json)")
        p.add_argument("--mode", choices=["plan", "actor"], default="plan", help="plan = upstream MPPI (stochastic, seeded per episode); actor = deterministic policy mean")
        p.add_argument("--work-dir", default=os.environ.get("FGRL_WORK_DIR") or str(Path(tempfile.gettempdir()) / "fgrl_work"))
c = sub.add_parser("compare"); c.add_argument("run_a"); c.add_argument("run_b")
a = ap.parse_args()

if a.cmd == "compare":
    ma, mb = (json.loads((Path(r) / "run_manifest.json").read_text()) for r in (a.run_a, a.run_b))
    sa, sb = (json.loads((Path(r) / "summary.json").read_text()) for r in (a.run_a, a.run_b))
    print(json.dumps({"comparability": classify_comparison(ma, mb), "mean_reward_per_step": [sa["mean_reward_per_step"]["mean"], sb["mean_reward_per_step"]["mean"]],
                      "inference_modes": [ma["inference_mode"], mb["inference_mode"]]}, indent=2)); sys.exit(0)

contract = EvalContract(env_id=a.env_id, split="test", n_episodes=a.episodes, protocol=a.protocol, base_seed=a.base_seed)
from fluidgym_rl.official_sac import local_checkpoint, sha256_file
ck = local_checkpoint(a.checkpoint, a.train_seed)
if a.cmd == "sac":
    from stable_baselines3 import SAC
    import stable_baselines3
    from fluidgym_rl.envs import make_fluid_env
    from fluidgym_rl.official_sac import SB3Policy, inspect_zip, validate_contract
    env = make_fluid_env(a.env_id); model = SAC.load(a.checkpoint, device="cuda")
    notes = validate_contract(model, env, ckpt_files=ck["seed_dir_files"], installed_sb3=stable_baselines3.__version__, saved_sb3=inspect_zip(a.checkpoint)["sb3_version_saved"])
    run = evaluate_and_write(policy=SB3Policy(model, not a.stochastic), contract=contract, out_root=a.out_root, tag="sac", algorithm="sac", checkpoint=ck,
                             checkpoint_sha256=sha256_file(a.checkpoint), extra_config={"checkpoint": ck}, extra_manifest={"contract_notes": notes}, env=env)
else:
    from fluidgym_rl.tdmpc2_runtime import TDMPC2Policy, load_agent
    from fluidgym_rl.upstream import make_tdmpc2_workcopy
    from fluidgym_rl.device import same_device
    from fluidgym_rl.envs import make_fluid_env
    wc = make_tdmpc2_workcopy(Path(a.work_dir) / "tdmpc2_patched")
    rc_path = a.run_config or str(Path(a.checkpoint).parent / "run_config.json")
    cfg, agent, rc = load_agent(a.checkpoint, rc_path, wc["code_dir"], Path(a.work_dir) / "eval_run")
    env = make_fluid_env(a.env_id)
    if not same_device(agent.device, env.cuda_device):
        raise SystemExit(f"agent device {agent.device} is not the env device {env.cuda_device}")
    if tuple(env.observation_space.shape) != tuple(rc["obs_shape"]["state"]) or env.action_space.shape[0] != rc["action_dim"]:
        raise SystemExit(f"checkpoint was trained for obs{rc['obs_shape']} act{rc['action_dim']}, env has obs{env.observation_space.shape} act{env.action_space.shape}")
    run = evaluate_and_write(policy=TDMPC2Policy(agent, a.mode), contract=contract, out_root=a.out_root, tag=f"tdmpc2_{a.mode}", algorithm="tdmpc2", checkpoint=ck,
                             checkpoint_sha256=sha256_file(a.checkpoint), extra_config={"checkpoint": ck, "run_config": rc, "workcopy": wc},
                             extra_manifest={"tdmpc2_upstream": wc["upstream"], "note": "upstream save() stores model weights only; evaluation = inference restoration"}, env=env)
print("wrote", run)
