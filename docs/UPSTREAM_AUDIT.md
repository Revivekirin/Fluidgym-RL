# Upstream audit (Phase 0.1 / 2.1 / 3.1) — audited 2026-10-08

| Repo | SHA (pinned in UPSTREAM.lock) | License | Python / torch | Notes |
|---|---|---|---|---|
| fluidgym | 50071140… (2026-09-29, v0.1.2) | MIT; files under `src/fluidgym/simulation/pict/` carry Apache-2.0 headers (PICT) | >=3.10,<3.14 / torch 2.9.* (cu128) | Custom CUDA kernels (PICT/PISOtorch). **CPU-only execution unsupported** (paper §5). |
| tdmpc (v1) | f4d85eca… (2023-11-25) | MIT | py3.8, torch 1.9, cudatoolkit 11.1, gym==0.21, dm-control | dm_control only (`src/env.py`); **dependency stack incompatible with FluidGym** (py>=3.10, torch 2.9, gymnasium). |
| tdmpc2 | e9f59321… (2026-07-13) | MIT | py3.11, torch 2.7.1 (docker env), gymnasium 0.29.1, hydra, tensordict 0.8.3, torchrl 0.8.1, numpy 1.24.4 | Env factory `tdmpc2/envs/__init__.py::make_env`; `torch.compile` on by default. |

## FluidGym supported algorithms (as shipped)
Integrations: Gymnasium, PettingZoo, SB3 (vec env), TorchRL. Published baselines: **PPO and SAC (SB3 defaults)**, MA-PPO/MA-SAC, and a **D-MPC** controller (gradient-only, paper App. D.3). No experiment/training config files (no yaml besides CI/pre-commit) live in the GitHub repo at the pinned SHA; `pyproject` has an `experiments` extra (hydra, wandb, rliable) but no `experiments/` dir. SB3 helpers exist in `src/fluidgym/integration/sb3/` (`eval_callback.py`, `vec_env.py`). Training configs are therefore the SB3 defaults in paper Tables 5/6 + the HF dataset (unverified, see below).

## Environment facts relevant to us (from source + paper)
- `CylinderJet2D-easy-v0`: Re=100, episode_length=80 control steps, 302 sensor values, 1 action in [-1,1], reward `C_D,ref − <C_D> − 1.0·<|C_L|>`.
- Observation is a `Dict` in v0.1; `FlattenObservation` is required to match HF models trained with v0.0.2.
- `step()` returns `(obs, reward, terminated, truncated, info)`; `truncated` fires at `episode_length`; stepping afterwards raises `RuntimeError`.
- Splits: train/val/test, 10 initial domains each, downloaded from Hugging Face on first use.
- `differentiable=True` enables backprop through `step()` (examples/interfaces/gradient_based_methods.py).

## TD-MPC vs TD-MPC2 for this task
| | TD-MPC (v1; dropped by decision, audit-only) | TD-MPC2 |
|---|---|---|
| Env contract | dm_control `TimeStep` + gym 0.21 | gymnasium-style inner env wrapped by `TensorWrapper`/`Timeout`; `reset()->obs`, `step()->(obs,r,done,info)` with `info['success']`, `info['terminated']`, `max_episode_steps` |
| Defaults to preserve | horizon 5, 512 samples/64 elites, 6 iters, lr 1e-3, batch 512, seed_steps 5000, update_freq 2 | horizon 3, 512 samples/64 elites, 6 iters, lr 3e-4, batch 256, `model_size` required, `compile: true` |
| Integration cost | High (python/torch/gym generation gap; needs a compat port or separate venv + process bridge) | Low (one factory + 4-line patch) |
| Recommendation | Defer; needs a decision (see IMPLEMENTATION_STATUS "Open decisions") | Integrate first |

## DPC (Phase 3)
See `REPRODUCTION_AUDIT.md §DPC`. Short version: the paper contains **D-MPC (Alg. 1, App. D.3)**, not a trained DPC policy; no DPC checkpoints exist to download.
