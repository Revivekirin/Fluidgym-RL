# IMPLEMENTATION_STATUS (updated 2026-10-08, milestone: official SAC evaluation + TD-MPC2 CUDA smoke)

**Evidence rule:** a stage counts only if it executed and passed. `not_run` / `skipped` / `unverified` are never passes. This authoring environment has no GPU, no torch and no Hugging Face file access, so **nothing below marked "remote" has been executed by the code author**; the commands to produce that evidence are listed in `docs/REMOTE_EXECUTION.md` §Milestone.

## Phase 0 (CUDA validation of `CylinderJet2D-easy-v0`)
- Status in this snapshot: **`not_run` (no report.json present)** — *not* a failure. The repo owner reports Phase 0 was completed on the remote server after fixing the `cuda` vs `cuda:0` comparison (`src/fluidgym_rl/device.py`) and the slice-prefixed GIF check; that report was not part of the uploaded snapshot, so it is unconfirmed here.
- Confirm without re-running the simulator: `python scripts/check_phase0_status.py --roots $FGRL_OUT_DIR` (exit 0 passed / 4 not_run / 1 failed or invalid; it validates that all 15 required checks are present and passed).
- **Snapshot defect found and fixed:** the uploaded `scripts/phase0_validate.py` ended after `episode_gif()`: the runner block (execute checks, finalize report, write gate) was missing, so that file ran *no checks and exited 0 without a report*. The runner was restored verbatim from the earlier version; also its `fluidgym_rl.device` import is now guarded so a torch-less machine produces a visible `failed` report instead of a crash. If your remote copy already has the runner, keep yours and compare.
- CUDA-alias normalization: reused the existing `fluidgym_rl/device.py` (resolves implicit `cuda` to `torch.cuda.current_device()`); regression tests added with a stand-in torch (alias equality, cpu-vs-cuda, distinct indices, current-device sensitivity, no-CUDA error) plus real-torch tests that skip without CUDA/multi-GPU.

## P1 official SAC evaluation — implemented, NOT executed
- Official artifact located (verified from the HF model card, not guessed): repo `safe-autonomous-systems/sac-CylinderJet2D-easy-v0`, per training seed `0..4/ckpt_latest.zip`, trained with `fluidgym==0.0.2`, needs `FlattenObservation` on newer versions. The Hub commit SHA is resolved and recorded at run time.
- `scripts/evaluate_official_sac.py` (+ `src/fluidgym_rl/{official_sac,evalcontract,run_eval}.py`): downloads/validates/evaluates; `--checkpoint-path` for offline use; `--dry-run` validates the observation/action contract on the real env without episodes; fails with exit 3 and a full diagnostic list on any mismatch (obs dim, structured Dict obs — refused, never flattened by guesswork —, action bounds, normalization artifacts such as vecnormalize, SB3 major-version drift). Writes `results/official_sac_eval/<run_id>/{resolved_config.yaml,episodes.csv,summary.json,run_manifest.json,evaluation.log}`; run dirs are never overwritten.
- Evaluation protocol = what FluidGym's own `evaluate_model`/`EvalCallback` do at the pinned SHA (test split, episode 0 `randomize=False`, then `True`, deterministic inference, per-step `info['drag']`/`info['lift']`). **Whether the paper's published numbers used exactly this protocol is unverified** (docs/REPRODUCTION_AUDIT.md).
- Physical metrics: only `drag` and `lift` as exposed by the env. Drag reduction is *not* inferred.
- Published numbers are stored side by side (model-card per-seed values; paper Table 8) and the report states `claims_reproduction: false`; differences are reported, never forced. Comparability is classified (same checkpoint+contract / different simulator version / different training seed / cross-algorithm / non-comparable).
- Not verified: field ordering of the flattened observation beyond dimension equality (recorded as UNVERIFIED in every manifest); `fluidgym-experiments` dataset layout (7.9 GB; its viewer is broken, file list not inspected) so no learning-curve reproduction yet.

## P2 TD-MPC2 integration — extended, NOT executed on GPU
- `scripts/tdmpc2_smoke_train.py` now (on the real CUDA env, upstream trainer/agent/buffer unchanged, observed through delegating wrappers) checks: agent/simulator device match; real transitions & replay-episode counts; stored-episode layout (NaN first row convention, finite rest); sampled batches (shapes, finite, **no sequence spans two episodes**, using the episode ids of the very samples upstream draws); ≥1 real optimizer update with finite consistency/reward/value/pi/total losses; parameters actually changed; planned actions finite and within [-1,1]; post-train evaluation on the real simulator; checkpoint save; same-process reload with per-tensor SHA-256 equality; **fresh-process** reload (`scripts/tdmpc2_ckpt_verify.py`) with digest equality, deterministic actor-action equality and seeded-planning reproducibility.
- Checkpoint semantics: upstream `TDMPC2.save()` stores **model weights only** → this is *inference restoration*. Optimizer state, step counters and RNG are not saved, so exact training resume is **not supported** and not claimed.
- Planning (`mpc=True`) is stochastic; evaluation records the mode (`mpc_plan_seeded` = MPPI with torch RNG seeded per episode, or `actor_mean` = deterministic policy mean without planning).
- Not verified: algorithm-semantics preservation is argued by construction (no upstream code modified except the 1-file env-registration patch; wrappers only record and delegate), plus the smoke checks above; it is not proven by diffing training curves against a reference implementation. `terminated` masking: FluidGym only truncates, so upstream's `cfg.episodic=False` path is exercised (termination masking is not).
- Upstream trains/evals on the FluidGym **train** split; the new evaluator uses the **test** split with the common contract, so training-time eval numbers are not comparable to it.

## P3 unified evaluation — implemented, NOT executed
- `scripts/evaluate_checkpoint.py {sac|tdmpc2|compare}` shares `EvalContract` with the official evaluator (same env, split, reset protocol, seeds, metrics, writer). Algorithm-specific inference stays explicit and is recorded in manifests (`deterministic`/`stochastic` for SAC; `mpc_plan_seeded`/`actor_mean` for TD-MPC2). PPO and D-MPC can plug in via the `Policy` protocol (not implemented).

## Tests (local, CPU) — 46 passed, 3 skipped
Covered: contract/protocol semantics, determinism, clipping, scalar reward conversion, truncation vs termination, non-finite handling, missing-metric error, no-overwrite run dirs, comparability classes, checkpoint metadata validation (missing file, dimension/bounds/Dict/normalization/SB3 drift), replay-sequence validity incl. cross-episode and NaN-leak detection, non-finite loss detection, checkpoint digest round-trip/tamper detection, report-status correctness (`not_run` vs `failed`), CUDA alias normalization, adapter contract. **Skipped (not passes):** real-torch/CUDA tests (`tests/test_fluidgym_smoke.py`, device tests needing a GPU, patch-application test when `third_party/tdmpc2` is absent).

## Still missing
- Any remote results for P1/P2/P3; Phase 1 gate (`scripts/mark_gate.py phase1`) is deliberately manual after you review the SAC evaluation evidence.
- Learning-curve reproduction from `fluidgym-experiments`; D-MPC; matched SAC-vs-TD-MPC2 study at real training budgets; full TD-MPC2 training (still blocked by `safety.py` gates).
- Upgrading `torch.compile=false` smoke to the upstream-default compiled path.

## 2026-10-08 remote failure follow-up
The supplied server logs show SAC loading blocked by missing `omegaconf` and TD-MPC2 trainer construction blocked by missing `hydra`. SAC's extra now declares OmegaConf; TD-MPC2 additionally requires `requirements/tdmpc2_extra.txt` (see the recovery commands in `docs/REMOTE_EXECUTION.md`). No training updates or checkpoint restoration were reached in that run. CUDA and patched-workcopy checks passed in the supplied smoke report; full smoke and independent evaluation remain blocked pending a successful server rerun. No upstream algorithm or patch was changed by this fix.

Follow-up validation on this Mac: Python compilation, `bash -n scripts/remote_milestone.sh`, and `git diff --check` passed. A standard-library-only regression run passed five Phase 0 recovery cases and one failure diagnostic/report check. The full pytest suite was **not_run**: pytest is absent and permission to download test dependencies was declined. These checks do not validate SAC loading or CUDA training.

## Nominal pilot and configurable W&B

The user reports the CUDA smoke passed every validation check, including real
updates and fresh-process checkpoint restoration. A bounded pilot launcher now
supports disabled/offline/online W&B, project/entity/group/tags, upstream periodic
evaluation and harness-scheduled saves via the upstream serializer. Full-training
gates are unchanged. Environment transitions and learner updates have independent
counters; manifests distinguish train-split in-run evaluation from independent
test-split evaluation. No learning-algorithm code is changed.

New pilot runtime and W&B CUDA validation remain **not_run locally** (Torch/CUDA
unavailable). Hydra composition also requires the server's dependencies; use
`--validate-only` before the offline pilot. No full training has been launched.

Pilot change validation: `PYTHONPATH=src python tests/test_pilot.py -v` ->
8 passed, 1 skipped (Hydra absent). Existing stdlib-compatible regressions:
smoke budget 3 passed; supplied-source factory 1 passed / pinned-source factory
1 skipped; real meta/gradient test 1 skipped (Torch absent). Total: 12 passed,
3 skipped. Python syntax checks and CLI help passed; UPSTREAM.lock is unchanged.
The full pytest suite was not run (local NumPy/pytest unavailable).
