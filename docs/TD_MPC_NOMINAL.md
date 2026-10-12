# Original TD-MPC nominal baseline

This integration runs the original Hansen et al. TD-MPC algorithm. It does not rename,
import, or modify TD-MPC2. Local validation uses synthetic CPU data, not CFD training.
Remote environment validation and CUDA smoke remain required before a 50k experiment.

## Repository audit

Reusable components are `envs.make_fluid_env` (official flattening and split selection),
`evalcontract` (reset protocol, episode records, reward/action statistics and result
files), `device` (logical CUDA aliases), `UPSTREAM.lock`, and the full-training safety
gates. The existing SAC implementation evaluates published checkpoints; it is not a
SAC trainer. No PPO trainer exists. `dmpc` is a placeholder, not a DPC baseline.
TD-MPC2 already has its own patches, replay checks, logging and checkpoint evaluator.
Those components and existing results remain unchanged.

The new `fluidgym_rl.tdmpc` package supplies a separate environment contract, strict
original-upstream loader, experiment loop and shared-contract policy. The original
upstream trainer imports obsolete Gym/dm-control and assumes fixed DMC episodes;
it is not imported. No CI configuration was found. Tests follow the existing pytest /
unittest conventions. The default configuration is JSON-formatted YAML 1.2, matching
the existing result serialization convention. Normal YAML can also be read with PyYAML.

## Sources, license and compatibility

- [Official TD-MPC](https://github.com/nicklashansen/tdmpc/tree/f4d85eca7419039b71bab2234ffc8aca378dd313),
  commit `f4d85eca7419039b71bab2234ffc8aca378dd313`, MIT, copyright Nicklas Hansen.
  The checkout is verified clean and copied into a temporary directory. The upstream
  license is copied with it. `UPSTREAM.lock` is unchanged.
- `patches/tdmpc_device.patch` changes only device transfers and checkpoint loading
  (`map_location`, `weights_only=True`). No architecture, objective, optimizer,
  target update, priority formula or planning equation is patched.
- Original environment specification: Python 3.8, PyTorch 1.9, CUDA 11.1, Gym 0.21.
  Do **not** install its `environment.yaml` into the FluidGym environment. Only its
  Torch/NumPy algorithm modules are imported. The remote target is the existing
  Python 3.12 / Torch 2.9.1 CUDA environment; actual compatibility requires smoke.
- [FluidGym pinned source](https://github.com/safe-autonomous-systems/fluidgym/tree/50071140b73a2a917fb96d33522a769f903f8c27)
  was inspected without modifying it. Installed wheel version and hashes of the
  installed cylinder/flattening code are recorded separately; a wheel is not asserted
  identical to that Git SHA.

## Paper configuration and provenance

The available [FluidGym v2 HTML](https://arxiv.org/html/2601.15015v2#A4) places the
TD-MPC hyperparameters in **D.2 Table 7**, not D.4. v1 has no TD-MPC baseline.
The nominal configuration uses Table 7 and the 50,000-step budget in Table 8.
Every configuration key has an explicit provenance classification in
`configs/tdmpc/cylinderjet2d_easy.yaml`; each run saves this mapping again.

Table 7 specifies seed steps 100, batch 512, discount .99, samples 512, elites 64,
iterations 6, encoder 256, latent 50, PER alpha .6 / beta .4. Both schedules last
50,000 steps: standard deviation .5→.05 and planning horizon 1→20. Remaining
model/loss/planning constants are also transcribed there. `kappa=.1` is retained
for provenance but is unused by this pinned upstream implementation.

One unresolved detail is the learner unroll horizon: the paper specifies the
planning schedule, while upstream uses one `horizon` for both its planning cap and
fixed learner unroll. This integration sets it to 20 for both. The independent
FluidGym TD-MPC training implementation was absent from the pinned FluidGym tree.
No claim of exact reproduction is made for that unreported detail.

## Environment contract

- `CylinderJet2D-easy-v0`, nominal Re=100, one action in [-1,1], float32.
- Official `FlattenObservation` includes **velocity only**, excludes pressure,
  and flattens sensor-major `[vx, vy]` at 151 sensors to `(302,)`. No normalization,
  frame stack or actuator-state augmentation is added.
- The upstream smoothing coefficient is .1 per solver step. The hidden smoothed
  control remains simulator state, so the interface is partially observed.
- Control interval .25; nominal solver dt .01 with adaptive CFL .8. No solver,
  actuation, physical parameters or reward are changed.
- Pinned source reward is `Cd_ref - mean_substep(Cd) - abs(mean_substep(Cl))`.
  Reward scaling is one. It is **not** the solver-substep mean absolute lift.
- 80 control steps per episode. Fixed-horizon truncation keeps original bootstrap
  behavior. An actual termination or early truncation fails loudly: original replay
  and TD target have no true-terminal mask, so silently accepting it would be wrong.
- Training uses `train()` and seed `train_seed + episode_index`, randomized resets.
  Validation uses `val()` and the existing official-callback reset protocol with
  seed `train_seed+10000`. Independent evaluation defaults to `test()` with the
  saved train seed. Both evaluation seeds/reset arguments are saved in result files.
  No test data participates in checkpoint selection.

`abs_lift` means the mean over control steps of `abs(info['lift'])`, where the 2D
info value is already averaged over solver substeps. Signed lift is recorded too.
This avoids confusing the requested absolute metric with the paper's signed column.

## Algorithm and experimental comparison

| Item | Original TD-MPC pin | FluidGym paper setup | Existing TD-MPC2 nominal run | New integration |
|---|---|---|---|---|
| Model | ELU TOLD, latent 50; twin scalar Q; policy prior | Table 7 dimensions 256/512/50 | SimNorm, latent 512, 5 distributional Q; model size 5 | Original TOLD, paper dimensions |
| Objectives | Latent MSE, scalar reward MSE, twin TD losses, Q-maximizing policy, EMA target | Table 7 coefficients | Distributional reward/value and TD-MPC2 objectives | Original losses unchanged |
| Replay | Prioritized trajectory subsequences, alpha .6 / beta .4 | Same reported PER | TorchRL SliceSampler | Original PER, fixed episode boundaries |
| Planning | Policy-guided weighted elite sampling, 512/64/6 | Same, horizon 1→20 | TD-MPC2 planner, horizon 3 | Original planner, episode-fixed scheduled horizon |
| Budget | Default 500k/action repeat | 50k control steps | Completed 50k in supplied local status | Exactly 50k; no extra final episode |
| Validation/test | Upstream DMC evaluation | Held-out splits and aggregate test protocol | Existing training logger evaluates its training env; separate test CLI exists | Explicit val selection; independent shared-contract test |
| Cost | GPU model/planner + original environment | Hardware-dependent reported experiment | Supplied run: 49,368 s wall, ~13.71 h | Measured per-run environment/learner/wall time and memory; not yet measured on CFD |

The existing local `results/tdmpc2_full/20261008T124249Z_4be46ae7/status.json`
reports completion: 50,000 training environment steps, 49,999 learner updates and
11,200 evaluation environment steps. Its final logged evaluation is return
-7.14338, per-step reward -0.0892923, drag 3.228326 and signed lift 0.006749.
`resolved_config.json` confirms model size 5, batch 256, discount .95, horizon 3,
learning rate .0003 and seed 1. This is the existing training-split evaluation,
not an independently verified held-out test result. Reported peak allocated memory
225,788,928 bytes is PyTorch allocation accounting, not total process/CFD VRAM.
These files were read only and were not altered.

The paper's [Table 9](https://arxiv.org/html/2601.15015v2#A5.S3) provides reference
TD-MPC test targets for CylinderJet2D-easy: per-step reward **.024**, drag **3.116**,
signed lift **.036**, drag reduction **6.360%**. These are IQM aggregates across
seeds/episodes, not one seed's arithmetic mean or episodic return. They are
**reference targets, not results of this implementation**. The baseline drag is
computed from uncontrolled training episodes; rounded published baseflow values
should not be substituted for matched measured evidence.

## Explicit harness adaptations and unresolved details

1. Upstream's inclusive training loop can exceed the requested budget; this harness
   stops after exactly the configured number of control transitions (whole episodes).
2. The 100 random actions are counted exactly. Replay only accepts complete episodes,
   so the initial 100 learner updates occur after step 160, then 80 updates per completed
   episode. This resolves upstream's `step == seed_steps` branch being unreachable when
   episode starts are multiples of 80. This cadence is an explicit adaptation, not a
   verified reproduction of an unavailable FluidGym training loop. A 50k run records
   49,940 learner updates separately from its 50,000 environment steps.
3. Planning schedules stay fixed within each episode (as in upstream) so CEM warm-start
   tensors never change length mid-episode. Evaluation uses the checkpoint's recorded
   environment-step count. `eval_mode` still samples candidate trajectories/elites;
   seeded evaluation is not a claim of deterministic action selection.
4. Upstream may add exploration noise outside [-1,1]. The harness clips before both
   simulator execution and replay insertion and records the clipped fraction. Thus
   replay contains the action actually sent to the simulator.
5. Small smoke buffers reserve enough whole episodes to avoid upstream's full-buffer
   without-replacement sampling requesting more nonzero-priority starts than exist.
   Full-run capacity stays 50,000. `horizon+1` sample allocation and exclusion of the last
   `horizon` starts of each episode are preserved, including the upstream convention
   that learner consumes only the first `horizon` samples.
6. Validation cadence (4,000), seed (default 1), best selection and a rolling periodic
   checkpoint are harness choices. Use five separately named seed runs for an aggregate
   study; one run cannot establish the paper's aggregate reproduction.
7. Original checkpoint contents are online and target model weights only. Adam state,
   replay, random generator state and planner history are not restored. There is no
   exact training-resume feature. Sidecars bind config, step count, provenance, hash
   the scheduled planner standard deviation, and a numerical encoder probe to weights.
   The standard deviation is restored from the sidecar so checkpoint evaluation does
   not accidentally reset exploration/planning spread to its initial .5 value. Bare foreign checkpoints require an
   explicitly verified matching sidecar/config; the CLI will not guess.
8. Each checkpoint retains the paper's nominal schedules even during a short smoke.
   Smoke is compatibility evidence, not a learning-performance measurement.

## Outputs and failure semantics

Each command creates a new timestamped directory under the output root and never
reuses an existing run. A training run contains `resolved_config.yaml`,
`run_manifest.json`, `train.csv`, `eval.csv`, `summary.json`, `stdout.log`,
`stderr.log`, `exit_code.txt`, `validation/step_*/`, and `checkpoints/{best,final}.pt`
with JSON sidecars. `latest.pt` is a rolling periodic save to limit storage.
Periodic save/evaluation thresholds are serviced at complete episode boundaries.
Evaluation has its own per-episode CSV, config, contract and summary.

The manifest records repository SHA/dirty status, imported upstream SHA/source hashes,
patch hash, local integration hashes, dependency versions, GPU identity, preprocessing,
reward, seed, reset protocol and run identity. A long step/update has no console progress
until the next completed episode; the process and log timestamps can be monitored.
Exceptions retain complete tracebacks and nonzero exit codes. Interrupted/failed runs
cannot produce a completed summary. An interrupted run may leave a valid earlier
checkpoint but must not be counted as a completed experiment.

## Remote commands

Run in the existing CUDA environment. These commands do not stash, reset or overwrite
local edits. The pull block stops when tracked local changes exist; reconcile them first.

```bash
cd /nfs4/jhkim/repos/Fluidgym-RL
if ! git diff --quiet || ! git diff --cached --quiet; then
  git status --short
  echo 'Local tracked changes exist: reconcile them before pulling.'
else
  git pull --ff-only
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /nfs4/jhkim/conda_envs/fluidgym-test
python -m pip install -e '.[dev]'
FGRL_UPSTREAMS=tdmpc bash scripts/fetch_upstream.sh
git -C third_party/tdmpc rev-parse HEAD
# Must be f4d85eca7419039b71bab2234ffc8aca378dd313

export CUDA_VISIBLE_DEVICES=0
python scripts/tdmpc_train.py --config configs/tdmpc/cylinderjet2d_easy.yaml --seed 1 --device cuda:0 --print-config
PYTHONPATH=src python -m pytest tests/test_tdmpc.py -q
```

Environment-only validation runs one 80-step zero-action episode, with no model updates:

```bash
python -u scripts/tdmpc_train.py --mode validate --device cuda:0 --output-root results/tdmpc_validate
```

Bounded real CUDA smoke runs 240 training transitions, 180 learner updates, one validation
episode and same/fresh-process checkpoint checks. It retains the nominal model and loss
configuration. This is deliberately more substantial than a fake tensor-only test.

```bash
python -u scripts/tdmpc_train.py --mode smoke --smoke-steps 240 --device cuda:0 --output-root results/tdmpc_smoke
SMOKE=$(ls -dt results/tdmpc_smoke/*_smoke | head -n 1)
cat "$SMOKE/summary.json"
cat "$SMOKE/exit_code.txt"
tail -n 80 "$SMOKE/stderr.log"
# Require status=completed, environment_steps=240, all checks passed, exit_code=0.
```

Only after smoke succeeds, run full training. Existing phase0/phase1 gates must already
have valid evidence; do not fabricate or bypass them. The environment variable and CLI
flag are both required. W&B is optional and disabled by default; see the online command below.

```bash
FGRL_ALLOW_FULL_TRAINING=1 python -u scripts/tdmpc_train.py \
  --config configs/tdmpc/cylinderjet2d_easy.yaml \
  --mode full --allow-full-training --seed 1 --device cuda:0 \
  --output-root results/tdmpc_full
```

In a second terminal:

```bash
cd /nfs4/jhkim/repos/Fluidgym-RL
RUN=$(ls -dt results/tdmpc_full/*_full | head -n 1)
echo "$RUN"
tail -n 30 "$RUN/train.csv"
tail -n 30 "$RUN/eval.csv"
tail -f "$RUN/stdout.log" "$RUN/stderr.log"
# In a separate terminal: nvidia-smi
# On completion:
cat "$RUN/summary.json"
cat "$RUN/exit_code.txt"
```

Measure an optional matched uncontrolled reference (10 training episodes), then evaluate
best and final independently on test. `--checkpoint` never starts training. Test results
must not be used to replace the validation-selected best checkpoint.

```bash
python -u scripts/tdmpc_train.py --uncontrolled --seed 1 --device cuda:0 --output-root results/tdmpc_baseflow
BASE=$(ls -dt results/tdmpc_baseflow/*_eval | head -n 1)/evaluation
RUN=$(ls -dt results/tdmpc_full/*_full | head -n 1)
python -u scripts/tdmpc_train.py --checkpoint "$RUN/checkpoints/best.pt" \
  --split test --episodes 10 --device cuda:0 --reference "$BASE" --output-root results/tdmpc_test_best
python -u scripts/tdmpc_train.py --checkpoint "$RUN/checkpoints/final.pt" \
  --split test --episodes 10 --device cuda:0 --reference "$BASE" --output-root results/tdmpc_test_final
BEST=$(ls -dt results/tdmpc_test_best/*_eval | head -n 1)/evaluation
FINAL=$(ls -dt results/tdmpc_test_final/*_eval | head -n 1)/evaluation
cat "$BEST/summary.json"
cat "$FINAL/summary.json"
# Existing comparison CLI can compare these evaluation directories to SAC/TD-MPC2 directories:
# python scripts/evaluate_checkpoint.py compare "$FINAL" <existing_evaluation_directory>
```

Package small results; model binaries, videos, replay and temporary upstream copies are
excluded. Include explicit directories so unrelated existing experiments are not bundled:

```bash
python scripts/package_tdmpc_results.py "$RUN" "$SMOKE" "$BASE" "$BEST" "$FINAL" \
  --output "results/tdmpc_analysis_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
```

## Local verification record

See `TD_MPC_VALIDATION.md` for executed commands, dependencies, pass/skip results and
remote-only checks. Nothing in this change asserts a completed nominal CFD training run,
a measured speedup, or reproduced test performance.


## Optional W&B tracking for original TD-MPC

The CLI accepts `--wandb-mode disabled|offline|online`, `--wandb-project`
(default `fluidgym-rl`), optional `--wandb-entity`, `--wandb-group` (default
`tdmpc-nominal`), and comma-separated `--wandb-tags`. These settings are saved in
resolved config and the run manifest. Empty inherited `WANDB_TAGS` is excluded
while starting the SDK, avoiding the earlier empty-tag Settings error.

W&B uses the same records as the existing CSV writer. It adds no optimizer or model
hooks. `environment_steps` is the explicit chart axis; `learner_updates` is a separate
counter. Training records appear after each 80-step episode, including returns,
reward/step, episode length, last-update losses and gradient norm, learning rate,
environment/learner throughput and GPU peak allocated memory. Losses are the latest
update in that episode, not an average over updates. Before updates begin they are
absent, not zero. Validation records include reward/step, episodic return, drag,
signed/absolute lift, split, episode count and best-checkpoint path. Validation runs
at the configured evaluation cadence, not every training episode.

The original pinned upstream logger is not used: it belongs to the unused DMC training
entry point. This sink attaches to this integration's existing CSV records so there is
one metrics collection path. Training/evaluation records at the same environment step
use separate SDK history indices; no records are dropped by a decreasing SDK step.

Configuration includes seed, provenance, source/patch hashes, versions and environment
contract. `wandb_run.json` stores the SDK run ID/URL. Final config, manifest, CSV and
summary files are saved to W&B; model binaries remain local. Checkpoint paths are
logged, but W&B artifact upload of weights is not enabled. Checkpoint semantics remain
inference-only. A failure, interruption, or failed fresh-process smoke verification
finishes tracking with a nonzero exit code. Logging errors are propagated rather than
silently claiming a fully recorded run. Disabled mode does not import the SDK.

W&B is enabled only for smoke/full training. Independent evaluation and environment
validation retain their local evidence output. Online mode requires working server
credentials/connectivity; offline mode writes an unsynced local W&B run.

After pulling the changes and activating the existing server environment:

```bash
cd /nfs4/jhkim/repos/Fluidgym-RL
python -m pip install -e '.[dev,tracking]'
python scripts/tdmpc_train.py --mode full --wandb-mode online --print-config

# Optional bounded check of the new logging integration:
python -u scripts/tdmpc_train.py --mode smoke --smoke-steps 240 --device cuda:0 \
  --wandb-mode offline --wandb-project fluidgym-rl --wandb-group tdmpc-logging-smoke \
  --output-root results/tdmpc_smoke

# 50k training with online logging (full-training gates remain required):
FGRL_ALLOW_FULL_TRAINING=1 python -u scripts/tdmpc_train.py \
  --config configs/tdmpc/cylinderjet2d_easy.yaml \
  --mode full --allow-full-training --seed 1 --device cuda:0 \
  --wandb-mode online --wandb-project fluidgym-rl \
  --wandb-group tdmpc-nominal-full --wandb-tags tdmpc,nominal,seed1 \
  --output-root results/tdmpc_full
```

The SDK prints the run URL after initialization. Expect the first training point only
after the first 80-step episode, and the first losses after the update block at step 160.
No full training or online run was started during implementation.
