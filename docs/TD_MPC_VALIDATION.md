# Original TD-MPC local verification — 2026-10-12

Executed without CFD training or CUDA initialization. Python 3.11.16,
Torch 2.14.0 (macOS CPU), NumPy 2.4.6, Gymnasium 1.3.0, pytest 9.1.1.
This differs from the intended remote Python 3.12 / Torch 2.9.1+cu126 environment;
CPU success does not certify that environment.

Pinned source was fetched into `/tmp/fgrl-original-tdmpc`, checked out at
`f4d85eca7419039b71bab2234ffc8aca378dd313`, and verified clean after tests.
The workcopy uses the checked-in unified patch, then verifies the exact changed-file
allowlist and records pre/post hashes. `UPSTREAM.lock` was not changed.

Executed command:

```bash
PYTHONDONTWRITEBYTECODE=1 FGRL_TDMPC_SOURCE=/tmp/fgrl-original-tdmpc PYTHONPATH=src \
  /Users/jihyekim/miniconda3/envs/basilisk-rl/bin/python -m pytest -q -rs \
  -m 'not cuda' --ignore=tests/test_fluidgym_smoke.py
```

Result: **79 passed, 7 skipped, 18 subtests passed**, 4.88 seconds.
The new original TD-MPC module contributes **11 passed, 1 skipped**.

Executed original TD-MPC checks:

- Config/provenance loading, schedule endpoints and config-only CLI.
- Full-training gate and invalid non-episode-aligned smoke budget rejection.
- Synthetic `(302,)` float32 / `[-1,1]` environment interface, exact horizon truncation,
  true termination rejection and absolute-lift statistic.
- CPU and mocked CUDA device alias handling (no real CUDA call).
- Applying the compatibility patch to the real pinned upstream checkout.
- Original episode storage, terminal observation, PER priorities / normalized importance
  weights, sequence boundaries, full-buffer sampling without replacement.
- Original model forward pass, all returned losses finite, gradients finite and nonzero,
  actual optimizer parameter changes, legal evaluation planned actions.
- Model/target checkpoint restoration, planner standard deviation and seeded planned-action equivalence, numerical probe, and separate Python-process load.
- Shared independent evaluator reward/step versus return, test split and episode count.
- Bounded synthetic experiment orchestration, separate environment/update counters,
  validation best/final checkpoint creation and CSV output. Synthetic tests reduce
  architecture and batch sizes; they are not CFD training or nominal-performance evidence.

Skipped / not executed:

| Check | Reason |
|---|---|
| Original real FluidGym environment / CUDA smoke | `remote_required`; explicitly skipped |
| Existing pilot Hydra override test | Hydra unavailable in chosen CPU environment |
| Three existing pinned TD-MPC2 workcopy tests | TD-MPC2 checkout absent locally |
| Existing real CUDA alias test | CUDA unavailable |
| Existing TD-MPC2 meta/ensemble test | TensorDict unavailable |
| Entire `test_fluidgym_smoke.py` module | Explicitly excluded to avoid simulator/GPU initialization |
| Full 50k TD-MPC training and independent CFD test evaluation | Intentionally not launched |

No new runtime failure occurred in executed CPU tests. The full traceback of any future
remote failure is retained in that command's `stderr.log` and `summary.json`.
See `TD_MPC_NOMINAL.md` for the exact remote validation/smoke/full/evaluation workflow.
