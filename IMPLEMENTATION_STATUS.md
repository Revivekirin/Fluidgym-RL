# IMPLEMENTATION_STATUS (updated 2026-10-08, after remote-execution refactor)

**Scope rule:** local runs are CPU *interface* tests only. FluidGym simulation, evaluation, training, rendering and benchmarking run on a remote CUDA server. **No CUDA/FluidGym check has been executed yet**, so every GPU-dependent item below is "written, not run".
Decisions in force: TD-MPC2 only (no TD-MPC v1); reproduce paper D-MPC (App. D.3), no learned DPC; no full training before Phase 0 CUDA validation + Phase 1 checkpoint evaluation.

## Verified (executed locally, CPU, interface level only)
- Upstream SHAs/licenses/deps recorded (`UPSTREAM.lock`, `docs/UPSTREAM_AUDIT.md`).
- `patches/tdmpc2_fluidgym_task.patch` applies to the pinned TD-MPC2 SHA via `make_tdmpc2_workcopy`; exactly `tdmpc2/envs/__init__.py` changes; the pinned clone stays clean (`tests/test_report_safety_upstream.py`).
- Adapter logic on a stand-in env: shapes/dtypes, `done = terminated or truncated`, true-terminal flag preserved, action clipping to bounds, metric pass-through, seeded reset.
- Report machinery: failures/errors/skips are recorded and never counted as pass; overall `passed` requires every planned check to pass.
- Safety gates: full training refused unless all conditions hold (`tdmpc2_train_full.py` exits 3 by default).
- Without CUDA/torch, `phase0_validate.py` produces a `failed` report (exit 2) and writes no gate; a full local `remote_smoke.sh` run ends `NOT PASSED` (cuda_pytests counted as failed because 0 tests passed).
- PNG/GIF verifier detects blank PNG, missing file, single-frame/static GIF.

## Written but UNVERIFIED on GPU (will only be evidence after a remote run)
- `scripts/remote_setup.sh`, `scripts/remote_smoke.sh` (bash syntax-checked only; setup never run).
- `scripts/phase0_validate.py`: CUDA availability + GPU tensor op, reset/step/action/reward/termination/device contracts, out-of-range-action behavior, reproducibility, throughput + peak memory, render PNG, flow-field PNG, episode GIF, metadata (simulator version, git SHAs, GPU, driver, torch).
- `scripts/integration_cuda.py`: adapter against the real env through the patched upstream `make_env` + `TensorWrapper`; device/dtype conversion; action clipping; full-episode termination.
- `scripts/tdmpc2_smoke_train.py`: upstream `TDMPC2`/`Buffer`/`OnlineTrainer`/`Logger` for ~3 training + 2 eval episodes (harness-reduced `seed_steps/steps/eval_episodes/model_size/compile`), checkpoint save + load round-trip, post-train eval, metrics.
- `tests/test_fluidgym_smoke.py` (`pytest -m cuda`).
- Whether TD-MPC2 imports and runs under torch 2.9 with the unpinned tensordict/torchrl.

## Missing / not started
- Any actual remote results (Phase 0 numbers, figures, GIF, integration verdict, smoke-train verdict).
- Phase 1: published-checkpoint download and SAC evaluation, learning-curve reproduction from HF data, GPU/wall-clock tracking, published/reproduced/new tables. (HF repo layout still unknown.) The `phase1` gate can therefore not be set.
- Phase 2: SAC-vs-TD-MPC2 matched evaluation (needs a test-split evaluator; upstream trains/evals on the train split), TD-MPC2 flow visualization during evaluation.
- Phase 3: D-MPC implementation (`src/fluidgym_rl/dmpc/` is an empty, documented placeholder, deliberately independent of TD-MPC2); measurement of multi-step differentiable-rollout memory.

## Open items for the repo owner
1. Commit/push the repo (sandbox copy has no git history) so the server can clone it.
2. Run `scripts/remote_setup.sh` then `scripts/remote_smoke.sh` and send back `SUMMARY.md` + `report.json` files; see `docs/REMOTE_EXECUTION.md` §5 for risks that may need manual action (wheel-vs-SHA, HF access from compute nodes, tensordict/torchrl pins).
3. Phase 1 evaluation design needs the HF model/dataset layout, which could only be checked from a machine with HF access.
