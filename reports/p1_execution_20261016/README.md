# Bounded P1 execution package

Status: prepared and CPU-fixture tested; real CFD and CUDA inference NOT RUN locally. User executes the server command. No remote connection is made by the authoring agent.

- B: 4 trials × 4 controls = **16 controls**. Same TEST seed 0/randomize=True; zero sequence and [0,.2,-.2,0], each repeated. Actual initial-domain load is observed. Solver reset/internal calls are reported separately.
- A: two actual B observations, 22 planner calls per algorithm, **0 CFD controls**. Native planner/config/compile retained; compile warmup plus valid warm-state generation are counted. Mean/std and finite model outputs are saved. Identical RNG/state contrasts must meet atol=rtol=1e-6. Changed RNG/warm-state contrasts are observations, not mandatory differences. Intermediate compiled candidates are unavailable without intrusive hooks and are not logged.
- C: only after B and both A gates pass; one shared TEST replication per algorithm, 10 seeds (0–9) × 80 = **1,600 controls maximum**. Saved validation winner/configs fixed. Historical reward differences are never pass/fail gates.
- Maximum whole task: **1,616 control steps**, plus separately measured reset/warmup native solver calls. No retraining or retries. Fresh worker processes isolate algorithm import namespaces. Failure stops the sequence and packages logs/status/partial outputs. Re-running the launcher starts a NEW budget, so do not rerun blindly after an error.

B observation/reward repeats use atol=rtol=1e-5; integer counters must match exactly. These are predeclared float32 repeatability thresholds, not physical accuracy/convergence tolerances. Actuator recurrence uses atol=1e-7, rtol=1e-6. Runtime private actuator memory and observed `_apply_action` arguments are directly read; the source-derived expected recurrence is separately labeled. Boundary flux is not independently measured.

Instrumentation is in-memory observation around official methods: initial-domain load, applied-control call arguments (B only), and solver-return guard. Original methods are called unchanged; source files and algorithms are never edited. This does add timing overhead. All policies use existing inference/reset wrappers. Python/NumPy/Torch CPU/CUDA RNG streams are recorded, but individual planner components share streams and are not independently perturbed. Valid warm states come from native planning calls; no arbitrary all-ones warm state is injected.

Current preflight package/source/backend/data/dependency hashes and core evaluator hashes are fail-closed inputs in expected.json. Other harness files are fingerprinted for provenance. Checkpoints, their configs, selection evidence and original test/validation artifacts are rehashed before and after stages. No installed package rollback or update occurs. Models use scratch patched copies of their pinned upstream sources just like the existing loaders. The Physics checkout is read-only.

The server needs both `run_p1.py` and `expected.json` in the same directory. Do not replace its repo with this local working tree.

```bash
# On the server after copying this package to /tmp/fgrl-p1-20261016:
/nfs4/jhkim/conda_envs/fluidgym-test/bin/python -B \
  /tmp/fgrl-p1-20261016/run_p1.py \
  --repo /nfs4/jhkim/repos/Fluidgym-RL

# Execute the printed bounded sequence:
/nfs4/jhkim/conda_envs/fluidgym-test/bin/python -B \
  /tmp/fgrl-p1-20261016/run_p1.py \
  --repo /nfs4/jhkim/repos/Fluidgym-RL --execute
```

A new UUID-suffixed output directory is created under the repository's reports/. The launcher emits START/END and a 30-second heartbeat. Child logs are stage-specific. The final RETURN_ARCHIVE path contains four requested Markdown reports, machine-readable verification/provenance, per-episode comparisons and scoped B arrays. If interrupted, allow the process to finalize before collecting the archive. Machine/power loss cannot guarantee final packaging.

Historical C rollout time suggests ~21–24 minutes plus loading/compile and observer overhead; this is an estimate, not a measured P1 runtime or timeout. Actual stage/wall time, attempted/completed controls, reset/solver calls and peak PyTorch CUDA allocation are recorded. PyTorch allocation is not total device memory.

Local validation: 10 CPU fixture tests passed, covering B budget/actuator mismatch/partial artifact retention, flatten-order rejection, C budget and upstream-stage gates, report failure classification, artifact mutation detection, RNG/warm separation, and uncontrolled hidden-randomness rejection. Fixtures are NOT nominal fluid observations and are not included as physical evidence.
