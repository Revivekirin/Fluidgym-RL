# Stage 2B launch package

Prepared locally; **no real CFD or checkpoint inference has been executed for Stage 2B**. The user runs the remote experiment. This is a new outcome-focused pilot, not the original failed Stage 2. Read STAGE2B_EXPERIMENT_CONTRACT.md before execution. Frozen checkpoint/config and unchanged adapter are inherited from the verified package. Only cross-run comparisons become diagnostics; per-run solver, finite-state, actuator, configuration and resource gates remain enforced.

Transfer `stage2b_execution_bundle.tar.gz` to `/tmp/stage2b_execution_bundle.tar.gz` on the server. Use the existing Stage 1 returned directory; if it moved, set STAGE1_RESULTS to its actual location.

```bash
STAGE2B_PACKAGE=$(mktemp -d /tmp/fgrl-stage2b.XXXXXX)
tar -xzf /tmp/stage2b_execution_bundle.tar.gz -C "$STAGE2B_PACKAGE" --strip-components=1
STAGE1_RESULTS=/nfs4/jhkim/repos/Fluidgym-RL/reports/stage1_pilot_20261016T124327Z_f587f185

# Plan only: no simulator import or inference.
/nfs4/jhkim/conda_envs/fluidgym-test/bin/python -B "$STAGE2B_PACKAGE/launch.py" \
  --repo /nfs4/jhkim/repos/Fluidgym-RL --stage1 "$STAGE1_RESULTS"

# Execute ONCE. No automatic retry after failure.
/nfs4/jhkim/conda_envs/fluidgym-test/bin/python -B "$STAGE2B_PACKAGE/launch.py" \
  --repo /nfs4/jhkim/repos/Fluidgym-RL --stage1 "$STAGE1_RESULTS" --execute
```

Output: `/nfs4/jhkim/repos/Fluidgym-RL/reports/stage2b_frozen_<UTC>_<UUID>/`; use the exact printed OUTPUT_DIRECTORY / RETURN_DIRECTORY. Return the entire directory, including failure or partial logs. Expected execution approximately 20 minutes based on the prior 163-control/198-second run (~19.5 minutes for 960 controls); logging and numerical-diagnostic overhead can increase this. Hard external cap remains 45 minutes, no retry. It is not an execution-time guarantee.

The launcher saves the contract/package manifest; the worker saves verified source/backend/checkpoint/config/data provenance and preregistration_ready.json before environment construction. Missing or mismatched prerequisite stops before CFD. Expected hash manifests are part of the archive. Existing original Stage 2 results and analyses are never written by these scripts.

Generated deliverables: STAGE2B_EXPERIMENT_CONTRACT.md, STAGE2B_FROZEN_CONTROL_RESULTS.md, STAGE2B_RESEARCH_DECISION.md, outcome_analysis.json (all complete paired comparisons, descriptive seed variation and missing episodes), raw per-run records, native diagnostics, provenance and output hashes. Failed episodes remain partial and are not aggregated as complete. Original native tolerance remains a diagnostic; its exceedance cannot reclassify original Stage 2 as passed.

Local tests use only CPU software fixtures, no physical or model-accuracy evidence. Full-native prefix differences deliberately continue to the approved horizon in these fixtures; frozen-weight violations and nonfinite data still stop. No Stage 3 or other follow-up is launched.
