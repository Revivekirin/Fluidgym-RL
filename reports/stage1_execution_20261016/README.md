# Stage 1 bounded paired transition package

**Local status: implemented and tested without CFD. Actual native CUDA/boundary/solver acceptance remains UNRESOLVED.** This package is opt-in and is not imported by training or official evaluation. Only the user runs the server command. No model is loaded, trained or optimized.

## Fixed protocol

CylinderJet2D-easy-v0, VAL, seed=0, randomize=True, original horizon 80. Fresh native environment for each run. Actual selected domain index and dataset hashes are recorded; selection is not overridden. All conditions use `[0,.2,.2,-.2,-.2,.2,.2,-.2,-.2,.2,.2,0]`.

Order: N-bare ×2, N-hook ×2, fixed gain .5 ×2, jump gain 1→.5 at transition t=6 ×2. Each has 12 controls, total **96 maximum**, with reset solver calls separately counted. The jump affects the seventh transition, o6→o7; a5=a6=.2. Nominal gain=1 reset initialization is identical for all arms. No reset or planner/controller change at the gain jump. No automatic retry, amplitude/schedule/tolerance change, or extension.

`adapter.py` patches only a selected instance's `_apply_action`, `reset` and `step`. Original `step(action)` receives its unchanged command; original smoothing computes its own memory. Only the argument to original `_apply_action` is multiplied by gain. Profile tensors are not modified. Detach restores the prior instance namespace. Reset disarms the schedule, uses gain=1, reads zero memory/boundary, then arms t=0. Boundary handles are read from the current domain, never retained across resets.

`NumericalObserver` wraps each newly constructed simulator instance's `single_step` before randomized-reset warmups. It logs and rejects native False/exception, nonfinite fields or native flux-balance violations. It does not alter the solver. Constructor `make_divergence_free()` precedes this wrapper; its exceptions propagate, but its internal solve is not counted as `single_step` or independently certified. No deliberate native solver failure is induced; False/exception handling is fault-injection tested with CPU doubles.

## Gates and evidence

- Package hashes; 72 FluidGym Python source hashes; pinned Git commit; reviewed compiled backend; versions; data; checkpoint/config/result hashes; environment import modules and selected C++ source hashes must match. The known historical and Stage-0 environment-contract refactor hashes are accepted explicitly, recorded at runtime, and never silently updated. Missing `environment_contract.py` is accepted only with the exact historical environment module that does not import it.
- Every control's 25 outer calls record requested command, separately reconstructed smoother recurrence, measured native memory, gain, and measured effective argument supplied to native `_apply_action` for hooked arms. **Actual C++ setter argument interception is unavailable** and explicitly false. Native `.velocity` and `getVelocity(False)` are independently read; source-derived profile products are never substituted for measured readback. Effective scalar is also projected from the measured boundary tensor onto its fixed profile.
- `GetFluxes()` is checked against measured computational velocity times the native transform determinant, selecting the y component. Outward sign is negative at top block `-y`, positive at bottom block `+y`. Domain net flux is checked against the unchanged native `flux_balance_tol`; exact jet cancellation is not assumed. Before/after solver boundary checks detect overwrites.
- Official raw velocity/pressure and flattened 302-vector, original reward/Cd/Cl, termination flags, solver clocks/counts and finite u/p are saved. Profiles and exposed state are saved at all 13 control boundaries. Unknown C++ caches/packed storage are explicitly excluded; matched-reset/prefix evidence is not full-state restoration.
- All resets match N-bare_0; same-condition repeats match numerically; both hooked nominal runs match bare before F/J. J checks the full o0..o6 available prefix **before** taking its first reduced-gain transition. A failed gate stops subsequent runs.
- Repeat atol=rtol=1e-5; clock atol=1e-6, rtol=0; integer counters exact. Actuator/readback atol=1e-7, rtol=1e-6. Downstream criterion: flat-velocity RMS, absolute Cd or Cl difference exceeds 5×max(nominal repeat noise, numerical floor) for two consecutive post-treatment transitions. Both repeats are reported and must satisfy this operational flag. No new inferential/statistical claim follows from it.

## Exact server launch

Copy `stage1_execution_bundle.tar.gz` to `/tmp/stage1_execution_bundle.tar.gz` on the existing server, without replacing the server repo or Physics checkout. Then run:

```bash
STAGE1_PACKAGE=$(mktemp -d /tmp/fgrl-stage1.XXXXXX)
tar -xzf /tmp/stage1_execution_bundle.tar.gz -C "$STAGE1_PACKAGE" --strip-components=1

# Plan only: imports no FluidGym and starts no CFD.
/nfs4/jhkim/conda_envs/fluidgym-test/bin/python -B \
  "$STAGE1_PACKAGE/launch.py" --repo /nfs4/jhkim/repos/Fluidgym-RL

# The one authorized bounded pilot. Do not rerun after failure.
/nfs4/jhkim/conda_envs/fluidgym-test/bin/python -B \
  "$STAGE1_PACKAGE/launch.py" --repo /nfs4/jhkim/repos/Fluidgym-RL --execute
```

The external parent starts one child process group and enforces **600 seconds maximum** for the whole worker, including preflight, resets, controls, logging and comparisons. It kills the process group at the deadline, without retry or extra CFD. Parent-side report/hash finalization may finish after 600 seconds and performs no CFD. Output is a new `reports/stage1_pilot_<UTC>_<UUID>/`, printed as `RETURN_DIRECTORY`. Existing official outputs are never overwritten. Direct worker execution is rejected unless launched under the watchdog environment.

Return the printed directory, including all files. It contains the three requested reports, provenance, per-run observations/state NPZ, actuator and solver JSON/JSONL, cost, gate/comparison tables, downstream criteria, statuses, post-preservation checks and `output_hashes.json`. JSONL is flushed each outer call; a hard kill can leave the last transition incomplete. Attempted controls are reserved before each call, completed counters can lag on abrupt termination, and partial records must not be reported as a completed 96-control pilot. Disk/power/process-parent loss can prevent finalization; retain the unique directory and never auto-retry.

`STAGE1_GO_NO_GO.md` never authorizes Stage 2. No TD-MPC2 comparison, frozen-policy run, history fit, or new schedule is in this package.

## Local verification

Run only CPU doubles (requires NumPy, standard-library unittest):

```bash
python -B test_stage1.py
```

These fixtures exercise software contracts and orchestration, not physical accuracy, real nominal observations, or CUDA solver reliability. See `local_validation.json` and the local implementation report for measured test results and preserved-file hashes.
