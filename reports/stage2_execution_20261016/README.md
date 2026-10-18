# Stage 2 — frozen TD-MPC bounded pilot

**Prepared locally; real Stage 2 CFD and policy inference NOT RUN.** Remote execution remains under the user's control. This package implements only the approved 12 runs × 80 controls = **960 controls**, external watchdog **2,700 s / 45 minutes**, zero training updates, no retry. Stage 1 sources/results, Physics, checkpoints and original evaluators are not overwritten.

## Fixed experiment

VAL `CylinderJet2D-easy-v0`, original horizon 80, reset `randomize=True`, frozen validation-selected TD-MPC 50k. Environment seeds {0,1} × planner seeds {0,1}, conditions S0, S1, S2 in that order for each pair. S0 gain1 throughout; S1 gain.5 starting t0 after nominal reset; S2 gain1 at t0–39 and gain.5 at t40–79. First changed transition is o40→o41. The policy receives only the official 302-value observation. There is no planner/environment/memory reset at t40.

Checkpoint parameters are read unchanged from `best.json`: horizon20, 512 samples, 64 elites, 6 iterations, discount.99, and saved std .05000900000000001. The existing `load_checkpoint`, `Policy.reset`, `Policy.act(eval_mode=True, step=50000)` and pinned patched-upstream loader are reused. The loader's existing zero-input probe is an integrity check, not evidence of physical model accuracy. No new latent/value/candidate diagnostics are added. Online and target weights, eval mode, optimizer state, exact config and std are checked before each physical action. A fresh agent/environment is used per run.

## Stage 1 reuse and gates

`PostSmoothingAdapter`, its reset/apply/step implementation and all helpers are identical to the verified Stage 1 version. Only the isolated `gain_at` table/bounds changes from 12 controls/t6 to the approved 80 controls/t40. Before CFD, AST equality outside `gain_at` and the original Stage 1 adapter/source hashes are checked. `stage1_support.py` is the exact verified Stage 1 pilot source, reused only for numerical instrumentation and provenance helpers; its old standalone pilot is never invoked. No upstream source is edited.

The actual returned Stage 1 directory is required. Its **784 file hashes** must match the locally reviewed fixed manifest. Status/cost must show Stage 1 passed. Current pinned FluidGym, compiled PISOtorch, dependency versions, dataset, checkpoint/config/selection/result and inference-module hashes are checked before native environment reset. Stage 1's verified gain1/bare equivalence is the pre-execution evidence; Stage 2 does not spend extra CFD controls repeating that experiment.

Every reset uses gain1 and verifies zero memory/boundary, fresh live boundary references and immutable profiles. Every native control call validates smoother recurrence, independent physical boundary velocity readback, flux consistency, persistence through the solver and finite fields/success. Expected profiles are separate from measured boundary values. Direct C++ setter arguments are not intercepted. Constructor divergence correction runs before the single_step observer; exceptions propagate, but its internal solve is not independently certified.

For each pair, S0 is the reference. S1/S2 reset observations, exposed native state and RNG are checked against it. S2 planner action, RNG, warm state, observations, force/reward, applied boundaries and clocks are compared **after each nominal transition**; planner output comparison occurs before that transition's CFD. A discrepancy stops immediately. The o40 state gate must pass before the first reduced-gain transition. Identical seeds are not accepted as a substitute for these checks.

Prefix numerical tolerances remain atol=rtol=1e-5; clock atol=1e-6/rtol=0; counters and RNG identity exact. Actuator readback atol=1e-7/rtol=1e-6. Maximum absolute and relative errors plus bitwise identity are reported. The relative denominator floor 1e-12 is descriptive only, not a relaxed acceptance threshold. No post-hoc tuning.

Environment resets are seeded first; the existing policy reset seeds NumPy/Torch and removes warm memory, with Python random seeded by the existing P1 helper **after** environment reset. Private environment RNGs and global planner RNG hashes/states are recorded. Sampling components share the original RNG streams and are not independently reseeded or modified.

## Exact server instructions

Transfer `stage2_execution_bundle.tar.gz` to `/tmp/stage2_execution_bundle.tar.gz`. Keep the Stage 1 result directory intact on the server. Its original printed location was under `reports/`; if you moved it, supply its actual path through `--stage1` (no search/guessing is performed by the worker).

```bash
STAGE2_PACKAGE=$(mktemp -d /tmp/fgrl-stage2.XXXXXX)
tar -xzf /tmp/stage2_execution_bundle.tar.gz \
  -C "$STAGE2_PACKAGE" --strip-components=1
STAGE1_RESULTS=/nfs4/jhkim/repos/Fluidgym-RL/reports/stage1_pilot_20261016T124327Z_f587f185

# Plan only: no FluidGym import, checkpoint loading or CFD.
/nfs4/jhkim/conda_envs/fluidgym-test/bin/python -B \
  "$STAGE2_PACKAGE/launch.py" --repo /nfs4/jhkim/repos/Fluidgym-RL \
  --stage1 "$STAGE1_RESULTS"

# Execute exactly once. Do not rerun automatically after a failure.
/nfs4/jhkim/conda_envs/fluidgym-test/bin/python -B \
  "$STAGE2_PACKAGE/launch.py" --repo /nfs4/jhkim/repos/Fluidgym-RL \
  --stage1 "$STAGE1_RESULTS" --execute
```

The launcher creates `reports/stage2_frozen_<UTC>_<UUID>/`, prints OUTPUT_DIRECTORY and RETURN_DIRECTORY, and emits a heartbeat every 30 seconds. One child process group has a 45-minute external watchdog covering preflight, all resets, planning, logging and comparisons. At expiry it is killed without retry. Report/hash finalization afterward performs no CFD. Runtime is not guaranteed; the earlier nominal rollout rate suggests about 13 minutes for CFD/planning, plus substantial state logging and repeated provenance/weight checks, within the hard 45-minute cap.

Return the entire printed directory. A failed run is evidence to inspect, not authorization to restart the 960-control budget. No reward/performance threshold terminates a numerically valid run, and legitimate saturation is logged rather than rejected.

## Evidence and accounting

- `steps.json`, `episodes.json`, `paired_outcomes.json`: original reward, Cd, signed Cl and absolute control-interval Cl, requested/clipped actions, gain, native memory, clipping/saturation frequencies, whole/pre40/post40 metrics, paired S2–S0 post40 differences. Clipping means actual requested≠clipped; saturation means |clipped|≥1, no invented near-boundary threshold.
- `obs_*.npz`, `state_*.npz/.json`, `profiles.npz`, `actuator_*.json`, `solver.jsonl`: raw/flat observations, exposed state, distinct expected/measured actuator values and native status. These are scoped feasibility records, not an independent broad training dataset.
- `planner_*.npz/.json`, `planner_rng_*.pt`: inference warm state and global RNG before/after, with fixed seed identities. No candidate/latent diagnostics.
- `provenance.json`, `frozen_provenance.json`, `weights_final.json`, `post_preservation.json`, package/output hashes: source/backend/data/config/checkpoint and preservation evidence.
- `cost.json`: attempted/completed controls, successful reset/control outer calls, actual per-call accepted counter increments and initial offsets separately. Offsets are **not added** to executed increments. Solver journal entries include attempts; interrupted counters are lower bounds and partial steps are not completed episodes. Initialization divergence correction is not included in observed single_step increments.
- Three required Stage 2 reports are generated even on orderly failure/watchdog termination. Power loss or loss of the parent process may prevent finalization; retain the partial unique directory.

Physical accuracy, complete hidden C++ state identity and statistical/population robustness remain outside scope. A gain reduction can improve reward. Paired closed-loop effects cannot uniquely distinguish authority loss, planner response and learned-model mismatch. Two planner seeds for one reset are not independent physical cases.

The post-run report recommends the next smallest **unexecuted** step: if gates pass, a separately approved zero-CFD offline one-step reward-prediction check on these at most 960 real transitions; if a gate fails, resolve that exact blocker offline first. No such diagnostic is implemented or run by this package. No TD-MPC2, history fitting, new schedule, training or additional CFD is launched.
