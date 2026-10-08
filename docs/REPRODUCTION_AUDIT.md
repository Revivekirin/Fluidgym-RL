# Reproduction audit: paper vs. what exists (2026-10-08)

Source: arXiv:2601.15015v1 (Becktepe, Franz, Thuerey, Peitz; ICML 2026). **Everything below labelled PUBLISHED is transcribed from the paper; nothing here has been reproduced yet.**

## Published SAC/PPO setup (to be matched)
- SB3 defaults. SAC (Table 6): lr 3e-4, γ 0.99, τ 0.005, buffer 1e6, batch 256, learning_starts 100, train_freq 1, gradient_steps −1, auto entropy, device CUDA. PPO (Table 5): n_steps 2048, batch 64, 10 epochs, device CPU.
- 50 000 env steps, 5 seeds for CylinderJet2D; 10 test-set episodes per run; reported as mean reward/step, IQM across seeds×episodes.
- Hardware: A100, py3.10, torch 2.9.1, CUDA 12.8.

## PUBLISHED test metrics — CylinderJet2D-easy-v0 (Table 8, IQM)
| Algo | reward/step | C_D | C_L | drag reduction |
|---|---|---|---|---|
| Baseflow | – | 3.328 | −0.042 | – |
| PPO | −0.052 | 3.141 | 0.065 | 5.638 % |
| SAC | 0.051 | 3.105 | 0.032 | 6.697 % |
Fig. 5 text quotes ≈8 % for one SAC test episode (single episode, different statistic than the IQM table — not a discrepancy, but do not compare the two).

## Cost reality check
Paper Table 2: ~2.01 s/step for CylinderJet2D on an A100 ⇒ **50 000 steps ≈ 28 GPU-hours per run**; 5 seeds ≈ 140 GPU-hours per algorithm. This is why Phase 1 evaluates published checkpoints first and nothing is retrained until tests pass.

## Published-data availability
| Item | Claimed location | Verified here? |
|---|---|---|
| Trained models | HF collection `safe-autonomous-systems/fluidgym-benchmark-models` | **Layout verified from the web pages (2026-10-08):** repo `safe-autonomous-systems/sac-CylinderJet2D-easy-v0`, per-seed `0..4/ckpt_latest.zip`, trained with fluidgym 0.0.2, `FlattenObservation` required. Files themselves not downloaded/hashed by the author. Model card lists per-seed mean reward 0.05/0.05/0.05/0.04/0.05 (std 0.39/0.40/0.37/0.38/0.39). |
| Experiment data (curves, test results) | HF dataset `safe-autonomous-systems/fluidgym-experiments` | Exists (MIT, 7.91 GB, train/test splits); dataset viewer errors on CSV parsing; **file layout not inspected** -> curve reproduction not started. |
| Initial domains | HF dataset `safe-autonomous-systems/fluidgym-data` | **No** (same). |
| Training configs | paper Tables 5/6 + App. D | Yes (text only); no config files found in repo at pinned SHA. |
Caveat from FluidGym README: published models were trained with v0.0.2 ⇒ use `FlattenObservation` (we do) or install v0.0.2. Whether v0.1.2 reproduces v0.0.2 numbers is itself something the evaluation must test.

## DPC
- **What the paper actually has:** "D-MPC", *inspired by* DPC (Drgoňa et al. 2022): per control step, optimize an H-step action sequence by gradient ascent through the differentiable simulator; execute the first action; shift horizon. **No policy network, value function, or learning across steps.** Hyperparameters: H=20, N=10 iterations, lr α=0.1, γ=0.999 (optimizer: "gradient descent/Adam" — unspecified which); evaluated on 10 seeds × 1 test episode for all three 2D cylinder envs. Algorithm 1 in App. D.3.
- **Not in the paper:** a learned DPC policy (the paper lists DPC as *future work*), training hyperparameters for one, or any checkpoint. ⇒ **No published DPC checkpoints exist.** D-MPC needs none (it is test-time optimization).
- **Repo support:** `differentiable=True`, `env.detach()`, `get_state()/set_state()` exist (needed by Alg. 1's "set env to state s0"). Differentiable multi-step rollout: documented by example for one step + `detach()`; **multi-step backprop through H=20 steps has not been verified** (needs GPU memory/time measurement).
- Ambiguity to resolve before implementing: "DPC" in the task could mean (a) reproduce D-MPC, or (b) build a new learned DPC policy. (a) is specified by the paper; (b) is not.

## Paper-internal inconsistencies noticed
- Airfoil hard Reynolds number: Table 4 says 5×10³, App. C.3 text says 3×10⁵. (Not relevant to cylinders.)
- Table 7 "#Algorithms" column holds non-integer values (runtime ratios), so "GPU hours" there is not simply steps×seeds×algos.
