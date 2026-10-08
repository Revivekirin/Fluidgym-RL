#!/usr/bin/env python
"""Evaluate the OFFICIAL FluidGym SAC checkpoint(s) for CylinderJet2D-easy-v0 under the common evaluation contract.
Needs CUDA + fluidgym + stable-baselines3 (+ huggingface_hub unless --checkpoint-path). Not executed in the authoring sandbox.
Exit: 0 ok | 2 setup/download problem | 3 checkpoint/contract incompatibility (diagnostic printed) | 5 phase0 gate missing."""
import argparse, os, sys, traceback
from pathlib import Path

import numpy as np

from fluidgym_rl.evalcontract import EvalContract, classify_comparison, iqm, new_run_dir
from fluidgym_rl.official_sac import (CheckpointContractError, PUBLISHED, SB3Policy, TRAIN_SEEDS, inspect_zip, local_checkpoint, resolve_official_checkpoint,
                                      sha256_file, validate_contract, TRAINED_WITH_FLUIDGYM)
from fluidgym_rl.run_eval import _ver, evaluate_and_write

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--env-id", default="CylinderJet2D-easy-v0")
ap.add_argument("--train-seed", default="0", help="0-4 or 'all' (official checkpoints are per training seed)")
ap.add_argument("--checkpoint-path", help="local SB3 .zip instead of downloading (then --train-seed is only a label)")
ap.add_argument("--revision", help="HF revision/commit to pin (default: current main; the resolved SHA is recorded)")
ap.add_argument("--cache-dir")
ap.add_argument("--episodes", type=int, default=10, help="paper: 10 test episodes per run")
ap.add_argument("--protocol", choices=["official_callback", "seeded"], default="official_callback")
ap.add_argument("--base-seed", type=int, default=0)
ap.add_argument("--stochastic", action="store_true", help="sample actions instead of deterministic inference (official default: deterministic)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--out-root", default=os.environ.get("FGRL_RESULTS_DIR", "results") + "/official_sac_eval")
ap.add_argument("--dry-run", action="store_true", help="resolve+load+validate the contract on the real env but run no episodes")
ap.add_argument("--skip-phase0-check", action="store_true")
a = ap.parse_args()

if not a.skip_phase0_check:
    from fluidgym_rl.status import phase0_status
    st = phase0_status([os.environ.get("FGRL_OUT_DIR", "results")], a.env_id)
    if st["status"] != "passed":
        print(f"Phase 0 evidence is '{st['status']}' ({st.get('reason', st.get('non_passed_checks'))}). Confirm it first (scripts/check_phase0_status.py) or pass --skip-phase0-check.", file=sys.stderr)
        sys.exit(5)

seeds = list(TRAIN_SEEDS) if a.train_seed == "all" else [int(a.train_seed)]
if a.checkpoint_path and len(seeds) != 1:
    print("--checkpoint-path needs a single --train-seed", file=sys.stderr); sys.exit(2)
contract = EvalContract(env_id=a.env_id, split="test", n_episodes=a.episodes, protocol=a.protocol, base_seed=a.base_seed)
runs = []
try:
    import stable_baselines3
    from stable_baselines3 import SAC
    from fluidgym_rl.envs import make_fluid_env
    for s in seeds:
        ck = local_checkpoint(a.checkpoint_path, s) if a.checkpoint_path else resolve_official_checkpoint(a.env_id, s, "sac", a.revision, a.cache_dir)
        sha = sha256_file(ck["path"]); zi = inspect_zip(ck["path"])
        env = make_fluid_env(a.env_id, flatten=True)
        model = SAC.load(ck["path"], device=a.device)
        notes = validate_contract(model, env, ckpt_files=ck["seed_dir_files"], installed_sb3=stable_baselines3.__version__, saved_sb3=zi["sb3_version_saved"])
        print(f"[seed {s}] contract OK: {notes}")
        if a.dry_run:
            d = new_run_dir(a.out_root, f"dryrun_seed{s}"); import json
            (d / "dry_run.json").write_text(json.dumps({"checkpoint": ck, "sha256": sha, "zip": {k: v for k, v in zi.items() if k != "members"}, "contract_notes": notes,
                                                      "trained_with_fluidgym": TRAINED_WITH_FLUIDGYM, "installed_fluidgym": _ver("fluidgym")}, indent=2, default=str))
            runs.append(d); continue
        refs = [dict(PUBLISHED["model_card_per_seed"][s], source=PUBLISHED["model_card_per_seed"]["source"], conditions=PUBLISHED["model_card_per_seed"]["conditions"]) if s in PUBLISHED["model_card_per_seed"] else None,
                PUBLISHED["paper_table8_iqm"]]
        refs = [r for r in refs if r]
        run = evaluate_and_write(policy=SB3Policy(model, deterministic=not a.stochastic), contract=contract, out_root=a.out_root, tag=f"official_sac_seed{s}", algorithm="sac",
                                 checkpoint=ck, checkpoint_sha256=sha, extra_config={"checkpoint": ck, "sb3_zip": {k: v for k, v in zi.items() if k != "members"}},
                                 extra_manifest={"checkpoint_trained_with_fluidgym": TRAINED_WITH_FLUIDGYM, "wrapper": "fluidgym.wrappers.FlattenObservation", "contract_notes": notes},
                                 references=refs, device=a.device, env=env)
        runs.append(run); print("wrote", run)
except CheckpointContractError as e:
    print("CHECKPOINT CONTRACT ERROR\n" + str(e), file=sys.stderr); sys.exit(3)
except Exception as e:  # noqa: BLE001
    traceback.print_exc(); print(f"setup/evaluation failed: {e!r}", file=sys.stderr); sys.exit(2)

if len(runs) > 1 and not a.dry_run:
    import json
    sums = [json.loads((r / "summary.json").read_text()) for r in runs]; mans = [json.loads((r / "run_manifest.json").read_text()) for r in runs]
    per_seed = [x["mean_reward_per_step"]["mean"] for x in sums]
    d = new_run_dir(a.out_root, "official_sac_aggregate")
    (d / "aggregate.json").write_text(json.dumps({"runs": [str(r) for r in runs], "per_seed_mean_reward": per_seed, "mean_over_seeds": float(np.mean(per_seed)), "iqm_over_seeds": iqm(per_seed),
        "seed_pair_comparability": classify_comparison(mans[0], mans[1]), "published": PUBLISHED["paper_table8_iqm"],
        "note": "IQM over 5 seed means is NOT the paper's IQM over seeds x episodes; compare episode-level via episodes.csv if exact aggregation is needed"}, indent=2, default=str))
    print("aggregate:", d)
