# Bounded diagnostic plan (NOT executed on CUDA)

**Verified:** optional CLI `scripts/audit_tdmpc2_diagnostics.py` exists and CPU argument/override tests passed. By default it prints a budget only. `--execute` is required for actual CUDA evaluation. It hardcodes validation, limits each invocation to1–10 episodes, creates a new output directory, preserves checkpoint hashes, records raw episodes, actual config/source/physics hashes and synchronized action latency. It imports no trainer and never calls update. Full GPU behavior is **unverified locally**. Complete exceptions are saved in traceback.txt/failure.json. This is a diagnostic format, not a new test-set benchmark result.

## Priority A: MPC vs policy-prior mean

**Hypothesis:** use of the learned model/value for planning changes nominal performance materially. Actor-vs-MPC measures planner use, not the causal usefulness of the world model: the policy itself was learned using latent/Q objectives. Start with44k fixed checkpoint, two identical validation initial-condition seeds, mode plan with saved horizon3 and mode actor (tanh policy mean). Same weights, no optimizer/RNG restoration claim. Repeat at48k only if informative. Each pair:2 modes×2 episodes×80=**320 simulator steps**, 320 action calls. Historically800-step validations cost~666–714 s, so~4.4–4.8 minutes of comparable rollout cost per pair, PLUS fresh loading/compilation and any runtime variation. Not a guaranteed runtime bound.

## Priority B: frozen horizon sensitivity

H∈{1,3,5,10}, same44k weights, mode plan, environment seeds0–1, planner seeds0–1. **640 simulator steps**, four fresh processes; reusing A's matching H3 leaves480 new steps. Keep num_samples512 total, num_pi_trajs24, iterations6, elites64, temperature.5, discount.95 and compile setting unchanged. Candidate budget is unchanged but candidate rollout COMPUTE increases with horizon (roughly512×H×6 latent reward/dynamics evaluations per action, plus prior/terminal-Q overhead). New horizons are inference-only, outside the trained horizon3 setting; no claims of retrained-algorithm fairness. Latency must accompany reward. Do not choose a test configuration from these results and then call the old test set untouched.

## Priority C: planner variability vs checkpoint drift

44k/48k/50k × planner base seeds{0,1000,2000} × fixed env seeds{0,1} ×80 = **1,440 simulator steps**,9 processes. Keep environment seeds FIXED; change only planner seed offsets. Configurations already generated in A can be reused exactly. This estimates conditional planner sampling variability with only two initial conditions; it is not training-seed variability or a robust uncertainty bound. Run same config/seed twice in fresh processes to test residual GPU/CFD nondeterminism if needed: another160 steps per repeat. Record model/config hashes and precision flags before interpreting differences.

## Priority D: predicted vs realized returns

**Unavailable / deferred.** Native FluidGym has state access methods, but exact state cloning, actuator memory, all RNG state, GPU solver replay equivalence and open-loop action replay have not been validated by this audit. Existing artifacts contain per-episode aggregates, not aligned candidate trajectories, predicted rewards/Qs or simulator snapshots. Do not implement a guessed snapshot contract or equate observed policy reward with candidate predicted value. Minimum prerequisite: same-state/same-actions deterministic replay and complete state/RNG checksum test, then bounded paired open-loop rollouts on validation.

## Exact optional remote commands

Use the existing allocated CUDA shell; no remote connection or diagnostic is started automatically. Copy the new script to the repository first. Without --execute every command below is a budget-only preview. Run only the chosen phase, never all by default.

```bash
conda activate /nfs4/jhkim/conda_envs/fluidgym-test
cd /nfs4/jhkim/repos/Fluidgym-RL
T2="$PWD/results/tdmpc2_full/20261008T124249Z_4be46ae7"
D="$PWD/results/nominal_diagnostics/$(date -u +%Y%m%dT%H%M%SZ)"
CK="$T2/upstream/models/step_44000.pt"
# Preview (no CUDA work, no output directory created)
python scripts/audit_tdmpc2_diagnostics.py --training-run "$T2" --checkpoint "$CK" --mode plan --episodes 2 --out "$D/plan44"
# A: 320 simulator steps total, opt-in execution
python scripts/audit_tdmpc2_diagnostics.py --training-run "$T2" --checkpoint "$CK" --mode plan --episodes 2 --out "$D/plan44" --execute
python scripts/audit_tdmpc2_diagnostics.py --training-run "$T2" --checkpoint "$CK" --mode actor --episodes 2 --out "$D/actor44" --execute
```

Optional B (480 extra steps after A; H3=A's plan44):

```bash
for H in 1 5 10; do
  python scripts/audit_tdmpc2_diagnostics.py --training-run "$T2" --checkpoint "$CK" --mode plan --horizon "$H" --episodes 2 --out "$D/horizon_$H" --execute || break
done
```

Optional C (1,440 steps if all new; output directories distinct from A):

```bash
for NAME in step_44000 step_48000 final; do
  for SEED in 0 1000 2000; do
    python scripts/audit_tdmpc2_diagnostics.py --training-run "$T2" --checkpoint "$T2/upstream/models/$NAME.pt" --mode plan --episodes 2 --env-seed 0 --planner-seed "$SEED" --out "$D/repeat_${NAME}_${SEED}" --execute || exit 1
  done
done
```

Every result is validation exploratory evidence. Training weights, reward, candidate count and original logs remain unchanged. Compile/cold-start latency is retained (first, all-call mean, median,p95, mean excluding first); excluding first does not guarantee compilation is fully excluded. No historical latency was fabricated.
