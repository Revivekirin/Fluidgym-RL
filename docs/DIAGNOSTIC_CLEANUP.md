# Nominal diagnostic cleanup — 2026-10-13

This maintenance change removes the standalone frozen-checkpoint diagnostic CLI.
It does not change algorithms, training, physics, checkpoint restoration, shared
evaluation, comparison outputs or W&B behavior. No training or GPU experiment ran.

## Removed

| Path / component | Original purpose | Reason / dependency verification |
|---|---|---|
| `scripts/audit_tdmpc2_diagnostics.py` (133 lines) | One-off validation MPC/prior contrast, horizon sweep and planner-seed repeats, including `parser`, `diagnostic_overrides`, `main` and nested `TimedPolicy` latency wrapper | Experimental entry point added for the nominal investigation; no retained production module or script imports/invokes it. Repository-wide search found only its dedicated test and historical audit documents/manifests as references. No package entry point or configuration depends on it. |
| `tests/test_nominal_audit.py::AuditTests.test_diagnostic_override_preserves_candidate_budget` and its two imports (14 lines) | Import the removed script and test its override helper/default CLI | Exclusively tests the deleted CLI. The other two adapter regression tests and their fake environment remain. |

No new experimental method or replacement diagnostic was introduced. No untracked
file was deleted. The deleted script remains retrievable from Git history; history
was not rewritten and no archive hashes were changed.

## Preserved (KEEP)

- `scripts/tdmpc_train.py`, `scripts/tdmpc2_train_full.py`,
  `scripts/tdmpc2_pilot_worker.py`, nominal configs, both runners and replay paths.
- `src/fluidgym_rl/tdmpc/*`, `tdmpc2_runtime.py`, `tdmpc2_adapter.py`, `envs.py`,
  upstream lock and compatibility patches. Device/meta/Gymnasium fixes are core
  compatibility requirements, not temporary diagnostic instrumentation.
- `scripts/tdmpc2_smoke_train.py`, `tdmpc2_ckpt_verify.py` and existing smoke tests.
- `evalcontract.py`, `run_eval.py`, `comparison/*`, `compare_nominal*.py`,
  `evaluate_checkpoint.py` and official SAC evaluation/integration. None imports
  the removed CLI. SAC functionality is unrelated to this cleanup and remains.
- Core logger observers, counters, losses and resource metrics. Comparison's
  `diagnostics/<algorithm>/*` panels consume actual training logs; the namespace
  does not make them obsolete diagnostic code. CSV/JSON/Parquet and W&B schemas
  are unchanged.
- Two tests in `test_nominal_audit.py` covering adapter transition/boundary I/O and
  reset/seeding behavior. They exercise retained production adapters.
- All `results/`, `outputs/`, `checkpoints/`, `wandb/` and existing `reports/`
  artifacts, including the completed MPC/prior experiment and upstream snapshots.

## Review required (preserved, not automatically removed)

| Component | Why retained / callers |
|---|---|
| `TDMPC2Policy(mode='actor')` | General-purpose deterministic inference is used in `tdmpc2_smoke_train.py:170` and `tdmpc2_ckpt_verify.py:25` for checkpoint probes. Removing it would break validated restoration checks. The shared evaluator uses the same class in plan mode. |
| `scripts/evaluate_checkpoint.py --mode actor` | Established public interface used by `scripts/remote_milestone.sh:44` and documented remote workflows. It predates the removed standalone investigation tool. Whether to retire its milestone actor evaluation requires an explicit workflow decision; untouched here. |
| `scripts/evaluate_checkpoint.py` / `run_eval.py` versus newer comparison evaluator | Overlapping evaluation responsibilities, but the milestone and official SAC path actively use them. Not dead duplicate helpers. |
| `reports/nominal_implementation_audit_20261013/build_evidence.py`, source snapshots, diagnostic plan and audit manifests | Archived analysis provenance, with recorded file hashes and historical reproduction commands. Removing or editing them would damage the existing audit archive. They are not production entry points. |
| `reports/nominal_audit_20261013/remote_evaluation.sh` | Archived common-evaluation workflow; not a horizon/actor experiment. |
| `.gitignore`, `.DS_Store`, `results/.DS_Store` modifications present at start | Unrelated user changes; preserved exactly as found. |

Historical audit reports may mention the now-deleted CLI. Those are archival
records, **not current execution instructions**. Active training/evaluation docs
contain no references to that CLI. The retained general actor inference interface
is distinct from the removed checkpoint-investigation command.

## Verification

Interpreter: `/Users/jihyekim/miniconda3/envs/basilisk-rl/bin/python`.
Environment: `PYTHONPATH=src:/tmp/fgrl-compare-deps`,
`MPLCONFIGDIR=/tmp/fgrl-compare-mpl`, `PYTHONDONTWRITEBYTECODE=1`,
`FGRL_TDMPC_SOURCE=/tmp/fgrl-original-tdmpc` for tests.

| Check | Actual result |
|---|---|
| Full existing suite: `python -m pytest -q -rs` | **PASS: 104 passed, 9 skipped, 20 subtests passed** |
| Python syntax/compile (in memory; no bytecode writes) | **PASS: 68 source/script/test files** |
| Core imports (environment, evaluator, checkpoint runtime, runners, tracking and comparison) | **PASS: 20 modules** |
| TD-MPC nominal config `read_config()` | **PASS** |
| TD-MPC2 pure parser, blockers and Hydra override generation | **PASS**; does not imply real Hydra composition |
| Shared evaluator/comparison, checkpoint-loader unit/round-trip tests, logging schema tests | **PASS** in full suite; no new remote restoration claimed |
| Non-CUDA adapter, safety, replay-check and smoke-budget regressions | **PASS** in full suite |
| Real FluidGym integration smoke | **SKIPPED:** FluidGym package unavailable |
| Real CUDA/device/simulator checks | **SKIPPED:** local CUDA unavailable (2 tests) |
| Actual TD-MPC2 Hydra composition | **SKIPPED:** Hydra unavailable |
| TD-MPC2 pinned-location factory/workcopy/logging checks | **SKIPPED:** checkout absent at configured `third_party` location (3 tests); no clone/fetch performed |
| TensorDict meta-template regression | **SKIPPED:** TensorDict unavailable |
| Real W&B offline SDK test | **SKIPPED:** opt-in test not enabled; mock/schema checks passed |
| Protected files before/after SHA-256 check | **PASS: 547 files unchanged**, including existing reports and diagnostic results |
| Active dependency/reference search | **PASS:** no retained executable/test references to deleted entry point |
| `git diff --check` | **PASS** |

No failed test was suppressed. No full training, CUDA evaluation, package install,
W&B upload, commit, push or destructive Git cleanup was performed.

## Git scope

- Deleted: `scripts/audit_tdmpc2_diagnostics.py`.
- Modified: `tests/test_nominal_audit.py` (diagnostic-only test and imports removed).
- Added documentation: this cleanup report.
- Code removed: **147 lines**. No production module changed.
- Remaining diagnostic-capable public entry point: the retained general
  `evaluate_checkpoint.py --mode actor`; dependencies/review rationale above.
- Existing analysis utilities inside protected report archives remain archival.
- Pre-existing unrelated `.gitignore` and `.DS_Store` diffs are excluded from the
  cleanup scope.
