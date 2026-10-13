# Shared nominal evaluation and comparison

This pipeline reads existing checkpoints. It never trains, rewrites original logs,
changes model objectives/planning parameters, or resumes an existing W&B run.
Outputs require new paths. Run each evaluator in a separate process so the two
upstream Python namespaces cannot collide.

## Repository audit (2026-10-12)

The implementation was inspected against `tdmpc/runner.py`, `tdmpc/tracking.py`,
`training_logging.py`, `tdmpc2_pilot_worker.py`, both checkpoint loaders, and the
shared `evalcontract.py`. This local checkout had **no results directory and no
third_party checkout** at audit time. Earlier pasted server logs are not proof
of current completed checkpoint contents. The `audit` command below inventories
actual server artifacts, completion status, sizes and SHA-256 hashes without
contacting or modifying W&B.

| Existing evidence | Actual interpretation | Common comparison treatment |
|---|---|---|
| TD-MPC `train/*` | Exploratory training rollout; real `environment_steps` | Historical/incompatible with frozen-policy evaluation |
| TD-MPC `val/*` | Validation, legacy `official_callback` reset protocol | Preserve val; independently re-evaluate for common curves |
| TD-MPC2 `train/*` | Training rollout; `train/environment_steps` is the real counter | Historical/incompatible |
| TD-MPC2 `eval/*` | **Train split**, because upstream eval does not switch the environment | Never relabel as val/test |
| TD-MPC2 `eval/lift` | Signed mean Cl | Cannot reconstruct mean absolute Cl |
| Both learners' losses | Different objectives and reductions | Separate algorithm diagnostic panels |
| TD-MPC `latest.pt` | Rolling periodic checkpoint | Missing historical checkpoints cannot be reconstructed |
| TD-MPC2 `cfg.steps` | Upstream loop bound, potentially budget minus one | Never substitute for real environment steps |

The full machine-readable mapping, including units, reductions, axes, resource
metrics and incompatibility reasons, is exported as `metric_mapping.csv` by
`report`. Its source is `comparison/audit.py`. Original metrics are not renamed
in their existing files or W&B runs. New performance keys are `compare/train/*`,
`compare/val/*`, and `compare/test/*`; new W&B axes use
`compare/environment_steps`. Learner updates are not environment steps.

## Independent evaluation contract

- `CylinderJet2D-easy-v0`, Re=100, 80 control steps, control interval 0.25,
  upstream solver and action smoothing unchanged.
- 302 raw flattened velocity components (sensor-major vx/vy), no observation
  normalization/stacking or actuator-state augmentation; normalized action [-1,1].
- Explicit `train`, `val`, or `test` initial-condition split. Default ten episodes,
  reset `seed=0..9, randomize=True`, identical across algorithms. The same seed
  labels in different splits do **not** imply the same underlying initial state.
- Original TD-MPC `plan(eval_mode=True)` and TD-MPC2 `act(eval_mode=True)` with
  saved planning/model parameters. Planning sampling remains stochastic, seeded
  per episode. TD-MPC restores the saved planner standard deviation too.
- Installed FluidGym Python sources, initial-domain dataset files (all splits),
  numerical package versions, physical settings and reward reference enter the
  compatibility fingerprint. File hashing adds setup I/O but does not alter data.
- Contracts differing in physics, dataset, seeds or episode count form separate
  cohorts. Splits remain separate even within a cohort. Different model/planner
  configurations have separate experiment signatures for seed aggregation.
- Validation selection maximizes mean reward per control step; ties choose the
  earlier real step then checkpoint hash. Selection is frozen in a hashed manifest
  before test. Test accepts only `final` or a verified `best_validation` selection.
  Selection is over **available candidates**, not over lost historical checkpoints.

### Physical metrics and formulas

Verified against FluidGym source pin
`50071140b73a2a917fb96d33522a769f903f8c27`,
`src/fluidgym/envs/cylinder/cylinder_env_base.py` (`_cd_ref`, `_step_impl`), and
`fluid_env.py` (split-specific initial domains, seeded selection, statistics).
For this single-cylinder 2D environment:

```
r_t = Cd_ref - mean_solver_substeps(Cd) - abs(mean_solver_substeps(Cl))
mean_reward_per_step = mean_t(r_t)
episode_return = sum_t(r_t)
mean_drag = mean_t(info.drag)
mean_abs_lift = mean_t(abs(info.lift))
drag_reduction_percent = 100 * (1 - mean_controlled_drag / mean_uncontrolled_train_drag)
```

`info.drag` and `info.lift` already contain control-interval means. Therefore
`abs(mean(Cl))`, `mean(abs(Cl))` across control intervals, and mean absolute
solver-substep lift are different quantities. The last is unavailable.
The evaluator checks the native reward equality at every step. The native
`Cd_ref` comes from the environment's stored statistics; it is **not** used as
an invented measured drag-reduction denominator. Drag reduction needs a separate
zero-action **training-split** reference with identical source/dataset, seeds and
episode budget. Different reference hashes are kept separate for ratio metrics.
No reference means null/unavailable, never zero. Some historical TD-MPC2 metadata
used a `mean(abs(Cl))` reward description; the original artifact is preserved and
this derived manifest records the source-verified formula instead.

Each episode reports return, per-step reward, mean drag, mean absolute lift,
length, evaluation wall seconds, executed-action mean/std/max and clipped-step
fraction. Aggregation uses arithmetic means of episode statistics (all horizons
are 80). Action std means within-episode population std; max is across all episodes.
Evaluation wall cost includes reset, planning and simulation, excludes model/source
setup. Training wall hours are taken only from an exact matching real-step record;
missing timing is unavailable. TD-MPC's timing includes action selection and
simulation; TD-MPC2's simulator timing wraps env.step only, so those throughput
metrics are incompatible. Neither wall timing is an algorithm-only compute measure.

## Exact server workflow

Run from the repository in the existing CUDA environment. These commands perform
**evaluation only**. A suite evaluates each unique saved checkpoint for ten
validation episodes, optionally ten train episodes, then two ten-episode test
reports; account for simulator time before starting it. Do not run concurrently
with an expensive training process on the same GPU.

```bash
python -m pip install -e '.[comparison]'
PYTHONPATH=src python -m pytest tests/test_comparison.py -q

# Select latest COMPLETED full runs, not failed/in-progress directories.
TD1=$(python - <<'PY'
import json, os
from pathlib import Path
root=Path(os.environ.get('FGRL_RESULTS_DIR','results'))/'tdmpc_full'
paths=[p for p in root.glob('*/summary.json') if json.loads(p.read_text()).get('status')=='completed' and (p.parent/'checkpoints/final.pt').is_file()]
if not paths: raise SystemExit('No completed TD-MPC full run; set TD1 to a verified run directory.')
print(sorted(paths)[-1].parent.resolve())
PY
)
TD2=$(python - <<'PY'
import json, os
from pathlib import Path
root=Path(os.environ.get('FGRL_RESULTS_DIR','results'))/'tdmpc2_full'
paths=[p for p in root.glob('*/status.json') if json.loads(p.read_text()).get('status')=='completed' and (p.parent/'upstream/models/final.pt').is_file()]
if not paths: raise SystemExit('No completed TD-MPC2 full run; set TD2 to a verified run directory.')
print(sorted(paths)[-1].parent.resolve())
PY
)
: "${TD1:?completed TD-MPC run required}" "${TD2:?completed TD-MPC2 run required}"
CMP="${FGRL_RESULTS_DIR:-results}/nominal_compare/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$CMP"
python scripts/compare_nominal.py audit --training-runs "$TD1" "$TD2" --out "$CMP/source_audit.json"

# A single reference shared by BOTH algorithms.
python -u scripts/compare_nominal.py evaluate \
  --algorithm uncontrolled --role reference --split train \
  --episodes 10 --base-seed 0 --device cuda:0 --out "$CMP/reference"

# Keep --include-train to produce independent train-vs-validation panels.
# Removing it saves evaluation time; missing train data is explicitly marked.
python -u scripts/compare_nominal_run.py --algorithm tdmpc \
  --training-run "$TD1" --reference "$CMP/reference" \
  --episodes 10 --base-seed 0 --include-train --out "$CMP/tdmpc"
python -u scripts/compare_nominal_run.py --algorithm tdmpc2 \
  --training-run "$TD2" --reference "$CMP/reference" \
  --episodes 10 --base-seed 0 --include-train --out "$CMP/tdmpc2"

# Bash on the Linux server; null-delimited paths support spaces.
mapfile -d '' EVALS < <(find "$CMP/tdmpc" "$CMP/tdmpc2" -name evaluation_manifest.json -printf '%h\0')
python scripts/compare_nominal.py report \
  --evaluations "${EVALS[@]}" --training-runs "$TD1" "$TD2" --out "$CMP/dashboard"
```

Use `&&` between stages, or `set -e` in a saved shell script, to stop on failures.
Each child evaluator returns nonzero and preserves a full `traceback.txt` on failure.
An interrupted/incomplete suite has no completed `suite_manifest.json`.
Original checkpoint loaders require the existing clean pinned upstream checkouts
and existing CUDA dependencies; this command does not fetch or repin them.

### Evaluate just one saved checkpoint

```bash
python -u scripts/compare_nominal.py evaluate \
  --algorithm tdmpc2 --training-run "$TD2" \
  --checkpoint "$TD2/upstream/models/final.pt" \
  --split val --role candidate --episodes 10 --base-seed 0 \
  --reference "$CMP/reference" --out "$CMP/single_final_val"
```

For TD-MPC substitute `--algorithm tdmpc`, `--training-run "$TD1"` and
`--checkpoint "$TD1/checkpoints/final.pt"`. Optional `--environment-steps`
asserts agreement with the recorded real counter; it cannot replace missing
evidence. Checkpoints without a verifiable environment-step identity are rejected.
Do not use learner updates or the Hydra `steps` field for this value.

### Dashboard, local figures, and optional derived W&B runs

Copy completed evaluation directories (all files, including manifests) to a local
machine, then run the same `report` command with their local paths. Reporting needs
no CUDA, upstream models, or W&B account. It verifies result hashes first and writes:

- `comparison.csv` / `.parquet`: long-form `compare/{split}/{metric}` values, unit,
  reduction, availability/reason, real steps, seed, run/checkpoint and source hashes.
- `seed_summary.csv` / `.parquet`: seed-level means; 95% percentile bootstrap
  intervals only with at least **three distinct training seeds** at the same real
  step/contract/model configuration/role. No interpolation or episode-as-seed error.
- `dashboard.html`, six PNG/SVG performance panels per compatible cohort, and
  separate TD-MPC / TD-MPC2 diagnostic panels when actual losses exist.
- `metric_mapping.csv`, `report_manifest.json`, input and output hashes.

Repeated seed/run evidence cannot inflate sample size. With one seed no uncertainty
band is claimed. Test results at different checkpoint steps are separate bars;
missing best/final or train/val evidence is labeled. Old training logs remain in the
export as historical/incompatible and never replace independent evaluations.
Sparse TD-MPC saved checkpoints necessarily give sparse independent curves.

Distinct saved candidates (for example best and final) can have the same training
seed and real step. In that case the summary preserves checkpoint-specific rows
with `series=checkpoint:<sha256>`, `n_seeds=1`, and no confidence interval. They
appear as separate points, not an average or a performance-selected point. Other
unambiguous steps still support normal seed aggregation. Duplicate evaluations of
the same checkpoint/seed remain rejected rather than counted as independent seeds.
If reporting fails after both evaluation suites completed, reuse their
`suite_manifest.json` evaluation lists and run only `compare_nominal.py report`
with a new output directory; do not rerun simulation or overwrite the failed report.

To publish **new derived runs** (original runs remain untouched), use a new output
path and append `--wandb-mode online --wandb-project fluidgym-compare` to `report`.
For an offline SDK check use `--wandb-mode offline`. Unique explicit run IDs prevent
inherited `WANDB_RUN_ID` from resuming an original run. Missing/incompatible metrics
are excluded from W&B numeric plots and remain visible in the local tables.

Manifest fields include pipeline commit and file hashes, upstream source SHA and
compatibility patch hash, checkpoint SHA/config/step evidence, inference semantics,
FluidGym source/dataset versions, explicit seeds/split/budget, physical settings,
reference/selection identity and result hashes. Checkpoints restore inference only;
optimizer, replay and RNG resume are not claimed. No original files are modified.

## Fresh SAC baseline

The same evaluation/select/report CLI accepts `--algorithm sac` for the new nominal SB3 `.zip` checkpoints and hashed JSON sidecars. Training, reference differences and exact remote commands are in [SAC_NOMINAL.md](SAC_NOMINAL.md). Public checkpoints without this provenance are not silently converted. Original TD-MPC and TD-MPC2 loading behavior remains unchanged.
