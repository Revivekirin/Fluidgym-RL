# Nominal comparison audit — 2026-10-13

Verdict: **NOT READY for dynamics-shift experiments.** Both training runs reached 50,000 real environment steps, but no shared `evaluation_manifest.json` exists in the supplied results. No independent validation/test scores or TD-MPC2 validation-selected winner can be reported yet. No training, simulator evaluation, remote connection, or W&B write was launched in this audit. Algorithms and training code were not changed.

## Executed evidence

- `tests/test_comparison.py`: **10 passed**, see tests.log. These include synthetic evaluation/provenance/selection contracts; they are not real CUDA tests.
- TD-MPC best and final: **PASS**, independent CPU processes using actual saved weights, loader and encoder-probe checks; see restore_best.log, restore_final.log and corresponding output directories. This verifies inference restoration, not optimizer/replay/RNG resume.
- Actual shared TD-MPC loader: **FAIL in local environment**, `ModuleNotFoundError: No module named 'omegaconf'` during config serialization. Full traceback: shared_loader.log. The proposed repeated seeded-action check was NOT reached.
- TD-MPC2: 13/13 files deserialize on CPU, 72 tensors each, all finite. **SKIP model restoration**: local CUDA unavailable and pinned TD-MPC2 checkout absent. Tensor checks are not a substitute for restoration or planning.
- Actual historical-artifact report: **PASS**, CSV, Parquet, PNG/SVG figures, mapping and HTML generated in dashboard/. Independent performance panels correctly say unavailable. No historical score is presented as a shared evaluation point.
- Original artifacts: all files inventoried before/after and unchanged; preservation.json and original_hashes.json record evidence.

## Requirement status

| Requirement | Result | Evidence / limitation |
|---|---|---|
| 1 Locate checkpoints | PASS | checkpoint_inventory.json, exact paths/hashes below |
| 2 Fresh-process restoration, both algorithms | SKIP | TD-MPC CPU PASS; TD-MPC2 CUDA not executed; shared loader dependency failure recorded |
| 3 Action selection / reset / determinism | SKIP | Adapter/static audit below; live shared planner repeatability unverified |
| 4 Identical physics, initial conditions, reward, horizon, seeds | SKIP | Common code contract checked; installed remote physics/dataset hashes and real rollouts not verified |
| 5 Never merge historical evaluation splits | PASS | Historical metrics exported incompatible; independent-only figure filters; tests passed |
| 6 Shared validation reevaluation | SKIP | Requires explicit remote CUDA evaluation budget |
| 7 TD-MPC2 validation-only selection | SKIP | Selector tests pass, no real common validation candidates yet |
| 8 Best/final independent test | SKIP | No test episodes executed |
| 9 Raw results, provenance, figures | SKIP | Artifact-only report PASS; real per-episode results unavailable |
| 10 Missing historical checkpoints | PASS | Recorded below; no synthetic/interpolated curve points |
| 11 Preserve originals | PASS | Before/after SHA-256 inventory |
| 12 PASS/FAIL/SKIP report | PASS | This table |
| 13 Exact remote commands | PASS | remote_evaluation.sh uses implemented CLIs; syntax checked, not executed |
| 14 No expensive automatic experiments | PASS | No simulation or training launched |

## Checkpoint inventory

Paths below are relative to `/Users/jihyekim/Desktop/Fluidgym-RL/`; the same suffixes are under `/nfs4/jhkim/repos/Fluidgym-RL/` on the server. checkpoint_inventory.json contains full absolute local paths and provenance/step evidence hashes.

| Algorithm | File | Real steps | SHA-256 |
|---|---|---:|---|
| tdmpc | `results/tdmpc_full/20261012T060719Z_full/checkpoints/best.pt` | 50000 | `1405b10e3601f987500d0c25575d64c7db85f4a7de6f9e6e5f1a724327239a0b` |
| tdmpc | `results/tdmpc_full/20261012T060719Z_full/checkpoints/final.pt` | 50000 | `48ec36ffd30b760c6f13e0c039d5ef6d57be0c348896d9cd2be18aa9961e00e4` |
| tdmpc | `results/tdmpc_full/20261012T060719Z_full/checkpoints/latest.pt` | 48000 | `81585e852000cacb4ae02bff8627ee0e9b1a6df4da979f672f9a952eaae9ab21` |
| tdmpc2 | `results/tdmpc2_full/20261008T124249Z_4be46ae7/upstream/models/final.pt` | 50000 | `865b6fffb7c9bf80e0a5ae1aa861711cb38fd10f5a4e79ebbdce14253f6f90ff` |
| tdmpc2 | `results/tdmpc2_full/20261008T124249Z_4be46ae7/upstream/models/step_12000.pt` | 12000 | `25ff524150f0df80b6152301dd8844ded6944dded3652cbfb47b4010b5e3fa3f` |
| tdmpc2 | `results/tdmpc2_full/20261008T124249Z_4be46ae7/upstream/models/step_16000.pt` | 16000 | `6fe0e9cf6efd39d87d8199e06e5fb2116c739f96949cd6dc9f85e3db850ca762` |
| tdmpc2 | `results/tdmpc2_full/20261008T124249Z_4be46ae7/upstream/models/step_20000.pt` | 20000 | `6af12c5d9e7c3915e9adf32d266a7ae11ee371909cbf80f0bcfa70acc2dcb0ba` |
| tdmpc2 | `results/tdmpc2_full/20261008T124249Z_4be46ae7/upstream/models/step_24000.pt` | 24000 | `18b4e35bd5cbaca8c693bea998265b6dae45cbf8f739d227389fd43b5de2d5f0` |
| tdmpc2 | `results/tdmpc2_full/20261008T124249Z_4be46ae7/upstream/models/step_28000.pt` | 28000 | `c6cf2a286c24607e22345da4c618795b70cab24478234b420822435c9b89ce84` |
| tdmpc2 | `results/tdmpc2_full/20261008T124249Z_4be46ae7/upstream/models/step_32000.pt` | 32000 | `75a8b35d0280150ed4047587ecd0bd467e9dce973241e43e95cda39dc116fb19` |
| tdmpc2 | `results/tdmpc2_full/20261008T124249Z_4be46ae7/upstream/models/step_36000.pt` | 36000 | `b17fc4c26508ac12272ea7f9f3b5431c9f3b23e997735ce549eab9b2a48487e6` |
| tdmpc2 | `results/tdmpc2_full/20261008T124249Z_4be46ae7/upstream/models/step_4000.pt` | 4000 | `7c745870b20861974de007282a0b831539c16be8c4fc173144fec1f97951e1fc` |
| tdmpc2 | `results/tdmpc2_full/20261008T124249Z_4be46ae7/upstream/models/step_40000.pt` | 40000 | `492184f09fa244b963607f77b97af0b7324a1272a92c30013c28eba5233983f4` |
| tdmpc2 | `results/tdmpc2_full/20261008T124249Z_4be46ae7/upstream/models/step_44000.pt` | 44000 | `825ae5e1fed69dad532357a3326156acc65733cf926bb3ae5ab851f20f4cf3c3` |
| tdmpc2 | `results/tdmpc2_full/20261008T124249Z_4be46ae7/upstream/models/step_48000.pt` | 48000 | `6d136fe26b516dc2958136b7ed5ed84c97d67a5316402e7bb470b4a6f44019ec` |
| tdmpc2 | `results/tdmpc2_full/20261008T124249Z_4be46ae7/upstream/models/step_8000.pt` | 8000 | `d6393b65673809cb190e21c8b67f94f89e75d9148d3b7e6921a21c05a64439af` |

TD-MPC best is the **historical** validation-selected 50k model (reported reward/step 0.03206033026101067). Final is also 50k, but file hashes differ; neither is deduplicated without evidence. Latest is 48k. Checkpoints at 4k–44k are absent, although historical validation logs exist. The shared selector can only select among surviving candidates, not recover a historical optimum under the new contract. TD-MPC2 has every 4k checkpoint through 48k plus 50k final; no step-zero checkpoint. No TD-MPC2 shared-validation winner exists yet.

Source pins: TD-MPC `f4d85eca7419039b71bab2234ffc8aca378dd313`; TD-MPC2 `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`. Training repository revisions: TD-MPC `be4478f494a5bc4d4f6de3550c28fa90d81b588f`; TD-MPC2 `d0f0686114e60954b41bf3c935132fbc0901404f`; manifests report dirty working trees, so SHA alone does not fully identify training source. Retain recorded source hashes/patch metadata.

## Common evaluation protocol and adapter audit

Use `CylinderJet2D-easy-v0`, Re=100, 302 float32 velocity observations, sensor-major [vx,vy], no normalization/frame stacking, one normalized action in [-1,1], native smoothing alpha=0.1, control interval 0.25, 80 steps per episode. No planner hyperparameters are changed. Shared evaluation uses the saved model configuration and each checkpoint's real environment-step counter (important for TD-MPC planning horizon schedules).

Use 10 seeded episodes per split: reset(seed=i, randomize=True) and reset policy RNG with i, i=0..9. Explicit train/val/test modes select separate precomputed initial-condition fields. Matching numeric seeds do not make test the validation split. The native reset samples initial-domain indices with replacement: ten episodes do not guarantee ten distinct fields. Remote installed environment and all initial-domain files must match; manifests hash these. Current historical artifacts do not prove identical historical datasets.

TD-MPC adapter calls `plan(eval_mode=True, step=checkpoint_steps, t0=...)`, resets NumPy/Torch and deletes `_prev_mean`. Pinned planner ignores warm-start when t0 is true. TD-MPC2 adapter seeds Torch/CUDA and calls `act(eval_mode=True, t0=...)`; it relies on upstream t0 handling rather than explicitly clearing planner state. Pinned TD-MPC2 planner source and runtime reset behavior remain a local verification gap. The shared wrapper also seeds Python/NumPy. Both are **seeded sampling planners**, not noise-free deterministic actor policies; eval_mode does not remove candidate-trajectory sampling. Bitwise reproducibility across hardware/library versions is not asserted.

The pinned native reward is `Cd_ref - Cd - lift_penalty*abs(Cl)`, using control-interval mean Cd/Cl, lift_penalty=1. Shared NominalEnv checks this against every actual reward. Mean absolute lift is mean over control steps of abs(interval-mean Cl), not abs(episode-mean Cl). An old TD-MPC2 metadata description is insufficient to infer a reward change; live verification is still required.

Comparison drag reduction is **100*(1 - measured mean Cd / measured uncontrolled-training mean Cd)**, with matched physics/seeds/budget. This is an explicit cross-split training-reference definition, not a same-initial-state paired test reduction and not derived from reward. The native Cd_ref used for reward is separately recorded and must not be silently substituted for the measured reference.

## Comparable and non-comparable quantities

After remote evaluation, compare split-specific reward/step, episode return, Cd, mean absolute Cl, measured-reference drag reduction, episode length, action mean/std/max/clipped fraction and evaluation wall cost. Primary axis is training real environment steps; evaluation interactions and learner updates are separate counters. Episodes/raw summaries and manifest hashes identify every checkpoint result.

Historical TD-MPC validation uses the legacy official_callback reset protocol. TD-MPC2 `eval/*` runs on train split. Neither is a shared-contract validation curve. Signed mean lift cannot be converted to mean absolute lift. TD-MPC scalar value/model/planning diagnostics and TD-MPC2 distributional/latent losses stay in separate panels; their magnitudes are not comparable. Throughput timing scopes also differ. See dashboard/metric_mapping.csv for units, reductions, axes and split mapping.

Only one training seed exists per algorithm; no across-training-seed uncertainty interval is supported. Evaluation episodes must not be counted as training seeds. Independent train-vs-validation plots require extra frozen-policy train evaluations (`--include-train`); legacy training rollouts cannot fill them.

## Remote execution and remaining blockers

Run remote_evaluation.sh manually from the existing configured CUDA environment after copying this audit directory to the repository. It never trains. It checks dependencies, restores TD-MPC, evaluates an uncontrolled training reference, runs each candidate in a fresh subprocess, freezes validation-only selection, evaluates best/final on test, and creates a new dashboard. Each evaluation failure saves a full traceback and stops the suite.

Default cost: 16 candidates × 10 × 80 validation steps, four test evaluations × 10 × 80, plus 800 reference steps = **16,800 simulator steps**. Optional independent train evaluations add 12,800. No cost estimate is a promise of runtime. Use a new output directory for every attempt; existing results are never overwritten.

Additional guardrail limitation: standalone best-validation test invocation verifies selection identities and candidate evidence, but does not enforce that the new test episode budget/base seed matches selected validation. Use the supplied batch commands with identical options and inspect manifests/cohorts; do not mix manually changed contracts. Physics cohorts separate incompatible results rather than proving all runs matched automatically.

Readiness requires successful real common validation/test results, TD-MPC2 fresh-process restoration/planner reset checks, matching remote physics/data provenance, and review of resulting nominal performance. Current status is **NOT READY**, not a finding that either trained algorithm failed.
