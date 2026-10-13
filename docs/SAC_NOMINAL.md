# Reproducible SB3 SAC nominal baseline

## Audit and readiness

**Implementation ready for a remote CUDA smoke; not yet certified for full
FluidGym training.** Local CPU tests exercise real SB3 replay/optimization on a
synthetic 302-dimensional environment, not CFD. No remote login, CUDA smoke, or
full training was launched during implementation. Full mode retains the existing
Phase 0/Phase 1 gates, explicit flag/environment opt-in, and additionally requires
a completed **SAC CUDA smoke** from the same source, SB3 version, physics and SAC
hyperparameters. No gate is written automatically by this new trainer.

Inspected `envs.py`, `evalcontract.py`, `run_eval.py`, `official_sac.py`,
`evaluate_official_sac.py`, original TD-MPC and TD-MPC2 entry points, tracking code,
and the shared comparison pipeline. Reused the official flattening wrapper,
`NominalEnv` physical/reward checks, shared seeded evaluation, `SB3Policy`,
checkpoint-selection manifests and W&B conventions. No TD model, objective,
planner or environment physics code is changed.

### Public checkpoint incompatibility: what is and is not established

User clarification (2026-10-13): no public SAC checkpoint was downloaded.
Consequently there is **no demonstrated checkpoint incompatibility** to diagnose.
No public SAC zip or current loader failure is present locally. Earlier pasted
evaluation logs do not establish the status of a newly downloaded checkpoint or
its compatibility with today's independent contract. The following are distinct
possible checks, not claims that an unseen checkpoint failed.

- The earlier `ModuleNotFoundError: omegaconf` occurs during SB3 deserialization.
  That is a loader dependency problem, not proof of a physical or space mismatch.
  Version drift/cloudpickle dependencies can also fail before any environment runs.
- A loaded model may genuinely have a different Box shape, action bounds, or
  normalization requirement. Existing `validate_contract` rejects those differences.
  Dimension equality alone cannot certify sensor order, reward or physics identity.
- Official metadata says FluidGym 0.0.2; the current harness uses 0.1.2. This version
  difference needs verification; it is not by itself proof of a physics change.
- Historical official evaluation used `official_callback`; shared comparison uses
  explicit `seeded` resets, split and source/dataset fingerprints. The old evaluation
  is not interchangeable with new common evaluation evidence.

Do not reshape, pad, reorder features, use `custom_objects` to replace spaces, or
rewrite the official zip. For a trusted checkpoint, obtain an exact read-only CPU
loader/space diagnostic (no downloads, no FluidGym simulation):

```bash
python scripts/audit_sac_checkpoint.py /path/to/ckpt_latest.zip --out public_sac_audit.json
```

The fresh nominal baseline starts from random weights and never loads public SAC
weights. Thus it does not depend on solving their incompatibility.

### Official repository training code audit

Inspected the clean official checkout at pinned commit
`50071140b73a2a917fb96d33522a769f903f8c27` (pin unchanged), including:

- [SB3 integration](https://github.com/safe-autonomous-systems/fluidgym/tree/50071140b73a2a917fb96d33522a769f903f8c27/src/fluidgym/integration/sb3):
  `VecFluidEnv`, evaluation callback and evaluation utilities.
- [SB3 example](https://github.com/safe-autonomous-systems/fluidgym/blob/50071140b73a2a917fb96d33522a769f903f8c27/examples/interfaces/sb3_vec_env.py):
  a MARL Airfoil random-action loop, **not SAC learning**.
- `integration/gymnasium.py`: `GymFluidEnv` handles NumPy/tensor conversion for
  single-agent SB3. The new adapter uses equivalent conversion through this repo's
  existing checked nominal wrapper, adding explicit seeds/shape/reward/boundary
  checks and real-step counting. It does not change physical actuation.

No standalone SAC training entry point/config or `SAC(...)` constructor was found
in that pinned source tree. The README advertises reference PPO/SAC baselines,
but integration helpers and released benchmark results are not a complete runnable
SAC experiment driver at this revision. This is a claim about the inspected pin,
not every branch or external experiment repository.

The official EvalCallback starts evaluation with `randomize=False`, then randomized
resets. Reusing it directly would change the already shared explicit-seed contract.
We therefore use SB3's original SAC learner with the repo's shared evaluator; no
custom SAC algorithm is implemented. The source revision remains pinned and untouched.

## Configuration and reference differences

`configs/sac_nominal.json` sets one seed (0), 50,000 real training environment steps,
validation every 4,000 steps plus final, and ten seeded validation episodes.

[FluidGym v1 Table 6](https://arxiv.org/html/2601.15015v1#A4.T6) specifies MlpPolicy,
lr 3e-4, gamma .99, tau .005, replay 1e6, batch 256, learning starts 100,
train frequency 1, gradient steps -1, auto entropy/target entropy, target update
interval 1 and CUDA. These are explicit constructor options. The one-seed run is
not a reproduction of the paper's five-seed aggregate.

Installed local SB3 **2.9.0** signatures/source were inspected. `sb3_audit()` repeats
that inspection on the server and records constructor defaults, policy signature,
version and source hashes; drift in relevant defaults fails. SB3's default gradient
steps is 1, overridden to the paper's -1 (equivalent with one environment and
train_freq=1). Non-paper details follow SB3: two 256-unit hidden layers, ReLU, Adam,
two critics, FlattenExtractor, no SDE/action noise/VecNormalize, one-step replay,
non-shared feature extractors. Architecture is explicit; policy defaults are checked.
Float32 flat state observations are not image-normalized. SB3 >=2.7,<3 is accepted
by the API checks; only 2.9.0 was exercised locally. Exact runtime versions are saved.

Other declared differences: current FluidGym/Python/CUDA versions and server
hardware, deterministic evaluation actions, explicit per-episode seeds, validation
selection, and a one-seed budget. These are nominal comparisons, not claims of exact
reference reproduction. Smoke uses the same SAC hyperparameters; only its step and
evaluation episode budgets are smaller. CPU fixtures alone use a smaller replay.

## Environment, learning and evaluation

`SACEnv` is a Gymnasium adapter over `NominalEnv(make_fluid_env(...))`. It preserves
302 raw flattened velocity values, [-1,1] normalized action scaling, native reward,
80-step horizon and upstream smoothing. SB3 gets float32 NumPy observations/actions;
the existing adapter transfers actions to the simulator device. Training resets use
`seed=train_seed+episode_index`, randomized **train** initial domains.

SB3's own SAC `learn`, replay buffer, entropy tuning and optimizer updates are used.
No learning function is replaced. Finite-gradient hooks observe and return no
modified gradient. Truncations stay `terminated=False, truncated=True`; SB3 stores
terminal observations and timeout flags and bootstraps at the horizon. Real steps
are counted by the adapter and checked against `num_timesteps`; evaluation steps do
not enter that counter or replay. The default replay's two observation arrays can
reserve roughly 2.4 GB of host RAM.

Validation loads immutable saved checkpoint candidates through the shared evaluator
in a separate environment; training RNG state is restored afterward. It may temporarily
allocate a second simulator/model on the same GPU. Test is **never** invoked during
training. `selection.json` freezes maximum validation mean per-step reward, with the
existing deterministic tie rule. `best.zip` copies that candidate; `final.zip` is saved
after all requested updates. A failed evaluation or failed checkpoint save fails the
run; the full traceback is captured in `stderr.log` and `summary.json`.

An SB3 zip includes model and optimizer tensors, but this workflow does not save
replay, simulator or RNG state and offers no exact-resume claim or resume CLI.
Checkpoint sidecars contain SHA-256, real steps, training seed, physical fingerprint,
resolved config and deterministic-action probe. Smoke also reloads in a fresh CPU
process; real simulator validation still runs on CUDA in the parent.

All algorithms use the same independent `seeded` contract (`eval_seed=0`, ten
80-step episodes by default), explicit train/val/test split, source/dataset hashes
and physical checks. Metrics and drag-reduction formula are in
[NOMINAL_COMPARISON.md](NOMINAL_COMPARISON.md). Drag reduction is unavailable without
a matched measured zero-action **train** reference; it is never inferred from reward.

## W&B and artifact schema

Default project remains `fluidgym-rl`, group `sac-nominal`, with a new explicit UUID
run ID. Existing TD-MPC/TD-MPC2 runs cannot be resumed or overwritten. CLI supports
disabled/offline/online, project/entity/group/tags. Empty inherited WANDB_TAGS is
handled consistently with the existing integrations.

| Keys | Axis / meaning |
|---|---|
| `compare/val/{mean_reward_per_step,episode_return,drag,abs_lift,...}` | `compare/environment_steps`; independent shared validation |
| `compare/test/*` | Separate post-training report, same axis and frozen selection |
| `train/*` | Exploratory training rollout, **not** interchangeable with independent `compare/train/*` |
| `diagnostics/sac/{actor_loss,critic_loss,ent_coef_loss,...}` | Latest exposed SB3 update; separate SAC diagnostic panel |
| `diagnostics/sac/learner_updates` | Optimizer-update counter, not environment steps |
| `resources/*` | Wall time, training steps / elapsed wall time, peak CUDA allocation |

CSV/Parquet metric mapping extends the existing comparison pipeline with SAC.
No missing metric is recorded as zero. Wall time includes validation; peak allocation
may include the evaluation model. Neither is isolated learner compute performance.

Each new run saves resolved_config.yaml (JSON-compatible YAML), run_manifest.json,
train.csv, val.csv, diagnostics.jsonl, stdout/stderr, exit code, summary, checkpoint
zips/sidecars, complete shared validation directories, selection.json and optional
wandb_run.json. Post-training evaluation saves independent test results separately.
Existing directories are rejected rather than reused.

## Exact remote commands

Run in the existing CUDA environment from the repository root. Do not run these on
the local authoring machine. Keep the installed CUDA torch build when adding extras:

```bash
python -m pip freeze | sed -n '/^torch==/p' > /tmp/fgrl-sac-torch-constraint.txt
python -m pip install -c /tmp/fgrl-sac-torch-constraint.txt -e '.[sac,dev,tracking,comparison]'
PYTHONPATH=src python -m pytest tests/test_sac_nominal.py tests/test_comparison.py -q

# 1. Configuration validation only: imports SB3, does not create FluidGym or W&B.
python scripts/sac_train.py --mode full --print-config-only

ROOT="${FGRL_RESULTS_DIR:-results}/sac_nominal/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$ROOT"
# Optional real adapter check: one complete episode plus SB3 check_env probes.
python -u scripts/sac_train.py --check-env-only --out "$ROOT/check_env"

# 2. Bounded CUDA smoke: 240 training steps + one 80-step validation episode.
python -u scripts/sac_train.py --mode smoke --steps 240 --eval-episodes 1 \
  --wandb-mode offline --out "$ROOT/smoke"
cat "$ROOT/smoke/summary.json"

# Shared drag reference, matching the ten-episode nominal evaluation contract.
python -u scripts/compare_nominal.py evaluate --algorithm uncontrolled \
  --role reference --split train --episodes 10 --base-seed 0 \
  --device cuda:0 --out "$ROOT/reference"

# 3. Pilot (allowed range 2000..5000); default physical/SAC settings unchanged.
python -u scripts/sac_train.py --mode pilot --steps 2400 --eval-freq 800 \
  --eval-episodes 10 --reference "$ROOT/reference" \
  --wandb-mode offline --out "$ROOT/pilot"

# 4. Full nominal run: only after reviewing successful CUDA smoke and pilot.
# Existing FGRL_OUT_DIR/FGRL_GATE_DIR must still point to the reviewed phase gates.
FGRL_ALLOW_FULL_TRAINING=1 python -u scripts/sac_train.py \
  --mode full --allow-full-training --steps 50000 --seed 0 \
  --smoke-evidence "$ROOT/smoke/summary.json" --reference "$ROOT/reference" \
  --wandb-mode online --wandb-project fluidgym-rl --wandb-group sac-nominal \
  --out "$ROOT/full"

# 5. Independent best-validation and final TEST; no model selection using test.
RUN="$ROOT/full"
python -u scripts/compare_nominal.py evaluate --algorithm sac --training-run "$RUN" \
  --checkpoint "$RUN/checkpoints/best.zip" --split test --role best_validation \
  --selection "$RUN/selection.json" --episodes 10 --base-seed 0 \
  --reference "$ROOT/reference" --out "$ROOT/test_best"
python -u scripts/compare_nominal.py evaluate --algorithm sac --training-run "$RUN" \
  --checkpoint "$RUN/checkpoints/final.zip" --split test --role final \
  --episodes 10 --base-seed 0 --reference "$ROOT/reference" --out "$ROOT/test_final"

# Dashboard; Bash/GNU find on the remote Linux server.
mapfile -d '' EVALS < <(find "$RUN" "$ROOT/test_best" "$ROOT/test_final" \
  -name evaluation_manifest.json -printf '%h\0')
python scripts/compare_nominal.py report --evaluations "${EVALS[@]}" \
  --training-runs "$RUN" --out "$ROOT/dashboard"
# Add --wandb-mode online --wandb-project fluidgym-rl for NEW derived comparison runs.
# Include existing TD-MPC/TD-MPC2 independent evaluation directories in --evaluations
# to compare all three; incompatible contracts are separated, never silently merged.

# 6. Archive completed training plus independent best/final tests, with file hashes.
python scripts/package_sac_results.py --run "$RUN" \
  --evaluations "$ROOT/test_best" "$ROOT/test_final" --out "$ROOT/sac_results.tar.gz"
```

Stop at any nonzero command (use `set -e` in a saved script). Do not create a fake
Phase 1 gate if a public checkpoint cannot be evaluated; the legacy gate requires
review and is intentionally unchanged. A missing gate blocks full training.
The smoke's hyperparameters, SB3 version, source hashes and physical fingerprint
must match full mode; rerun smoke after relevant code changes. The pilot is bounded
but not an automatic substitute for the required smoke.

For a common checkpoint sweep later, `compare_nominal_run.py --algorithm sac`
also accepts completed SAC runs. Avoid mixing duplicate re-evaluations of the same
training seed/step/role in one report. Independent test results are evidence only
once the separate commands have finished; a completed training summary explicitly
records test as not_run.

## Executed local validation

2026-10-13: full CPU suite **101 passed, 9 skipped, 20 subtests passed**; configuration-only CLI completed with SB3 2.9.0. SAC tests used synthetic observations/physics, collected 240 real fixture transitions, ran 140 SB3 updates, checked gradients/parameters, legal actions, replay timeouts, deterministic round-trip and fresh-process restore, shared validation aggregation, RNG preservation and mocked W&B identity/axis behavior. No real CUDA FluidGym smoke, pilot, training or test evaluation was executed. A real W&B SDK/network run was not executed for SAC (sink behavior was mocked).
