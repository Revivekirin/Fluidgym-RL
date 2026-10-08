# Remote execution guide

FluidGym needs a CUDA GPU; the local development machine has none. **All simulation, evaluation, training, rendering and benchmarking happens on the remote CUDA server.** Locally you only edit code and run CPU *interface* tests (`pytest -m "not cuda"`), which prove nothing about FluidGym compatibility. This repo is independent of the existing DRL project: it shares no code, venv, or paths with it, and nothing here writes outside `FGRL_*` locations.

Nothing below has been executed on a GPU yet; the first remote run is what produces the evidence.

## 0. Get the code onto the server
Locally (the sandbox copy is not a git repo yet):
```bash
cd fluidgym-rl && git init && git add -A && git commit -m "fluidgym-rl harness" && git remote add origin <YOUR_REMOTE_URL> && git push -u origin HEAD
```
On the server (no sudo needed; needs git + Python 3.10–3.13 + outbound access to PyPI, download.pytorch.org, github.com, huggingface.co):
```bash
git clone <YOUR_REMOTE_URL> fluidgym-rl && cd fluidgym-rl
```
(Alternative without a git remote: `rsync -a --exclude .venv --exclude third_party --exclude results ./ server:~/fluidgym-rl/`.)

## 1. Create the environment (once)
```bash
# optional overrides, all have defaults — see header of scripts/remote_setup.sh
export FGRL_VENV=$HOME/envs/fgrl           # default <repo>/.venv
export FGRL_OUT_DIR=/scratch/$USER/fgrl    # default <repo>/results   (results, gates, logs)
export FGRL_PYTHON=python3.11              # default python3
# export FGRL_TORCH_INDEX=https://download.pytorch.org/whl/cu121   # if the driver is older than CUDA 12.8 wheels need
scripts/remote_setup.sh
```
What it does: venv → `torch==2.9.*` (cu128 index) → records `torch==<installed>` as a pip constraint → `fluidgym==0.1.2` (prebuilt PyPI wheel; no nvcc/CUDA toolkit needed) → TD-MPC2 runtime deps from `requirements/tdmpc2_extra.txt` → this package (editable) → clones **only** `fluidgym` and `tdmpc2` at the SHAs in `UPSTREAM.lock` into `FGRL_THIRD_PARTY` → writes `$FGRL_OUT_DIR/setup/pip_freeze.txt` → prints a CUDA matmul sanity check. It can run on a login node, but `torch.cuda` must be visible for the check to say so.

Pinned versions: torch 2.9.* (FluidGym requirement; TD-MPC2's own docker env uses 2.7.1 — a deliberate deviation), fluidgym 0.1.2, hydra-core 1.3.2, omegaconf 2.3.0, submitit 1.5.1, hydra-submitit-launcher 1.2.0, termcolor 2.4.0. `tensordict`/`torchrl` are unpinned (see Known risks).

## 2. Run the validation (on a GPU node)
```bash
. $FGRL_VENV/bin/activate
scripts/remote_smoke.sh                       # ~ tens of minutes; see cost note
# useful switches:
FGRL_SKIP_TRAIN_SMOKE=1 scripts/remote_smoke.sh      # Phase 0 + integration only (fastest useful run)
FGRL_ENV_ID=CylinderJet2D-easy-v0 FGRL_BENCH_STEPS=20 scripts/remote_smoke.sh
```
Under a scheduler (no scheduler is assumed; adapt to yours), e.g. `srun --gres=gpu:1 -t 02:00:00 --pty bash` then the commands above, or wrap `scripts/remote_smoke.sh` in your batch script.

Individual stages can be run on their own:
```bash
python scripts/phase0_validate.py     --out $FGRL_OUT_DIR/phase0
python scripts/integration_cuda.py    --out $FGRL_OUT_DIR/integration
python scripts/tdmpc2_smoke_train.py  --out $FGRL_OUT_DIR/tdmpc2_smoke
```
**Cost note.** The FluidGym paper reports ≈2 s per CylinderJet2D step on an A100 (Table 2; your GPU will differ). Episode length is 80 steps. Rough env-step counts: phase0 ≈ 300, integration ≈ 100, TD-MPC2 smoke ≈ 400 (3 training episodes + 2 eval episodes). Treat as estimates, not measurements; `phase0` records the real s/step.

## 3. Outputs (`$FGRL_OUT_DIR/remote_smoke/<UTC-stamp>/`)
| file | content |
|---|---|
| `SUMMARY.md`, `summary.json` | per-stage verdict; **overall is `passed` only if every stage ran and passed** (skipped/not-run ≠ pass) |
| `preflight.json` | versions, GPU, driver, torch/CUDA, FluidGym version + install source, repo SHA, upstream SHAs/dirty flags |
| `phase0/<env>/report.json`, `render.png`, `flow_fields.png`, `episode.gif` | CUDA validation; every check is `passed/failed/error/skipped` with traceback on error |
| `integration/report.json` | adapter vs real env through patched upstream `make_env` + `TensorWrapper` |
| `tdmpc2_smoke/report.json`, `metrics.json`, `checkpoints/` | smoke training, checkpoint save + load round-trip |
| `*.log`, `stage_rc.txt`, `cuda_pytests.xml` | raw logs / exit codes / junit |

`report.json` check list — phase0: `cuda_available, gpu_tensor_op, env_create, reset_obs_contract, action_contract, step_contract, action_bounds_behavior, action_affects_state, episode_termination, reproducibility, seed_sensitivity, throughput_and_memory, render_png, flow_field_png, episode_gif`. PNG/GIF are decoded and checked (non-blank; GIF has ≥2 distinct frames). Pull results back: `rsync -a server:$FGRL_OUT_DIR/remote_smoke/ ./results/remote_smoke/`.

## 4. Safety defaults
- Every script defaults to smoke scale. `scripts/tdmpc2_smoke_train.py` never runs more than a few episodes.
- Full training is `scripts/tdmpc2_train_full.py --mode full --allow-full-training --steps N` **and** `FGRL_ALLOW_FULL_TRAINING=1` **and** gate files `phase0` + `phase1` under `$FGRL_OUT_DIR/gates/`. `phase0` is written automatically by a passing `phase0_validate.py`; `phase1` only by `python scripts/mark_gate.py phase1 --evidence <file>` after you have reviewed published-checkpoint evaluation — which is **not implemented yet**, so full training is currently impossible by design. `--steps` has no default.
- The pinned upstream clones are never edited: the TD-MPC2 patch is applied to a throw-away copy under `FGRL_WORK_DIR` (default `$TMPDIR/fgrl_work`), and the run verifies that exactly `tdmpc2/envs/__init__.py` differs.

## 5. Known risks / manual intervention
1. **Wheel vs pinned SHA.** The simulator under test is the PyPI wheel `fluidgym==0.1.2`; `UPSTREAM.lock` pins git `50071140` (also version 0.1.2). That they are the same code is *unverified*; `preflight.json` records both. For an exact-source build see the FluidGym README ("Build from Source": conda CUDA toolkit 12.8 + gcc, `make install`) — needs a toolchain you may not have without sudo.
2. **Hugging Face access.** FluidGym downloads initial domains on first `reset()`. If the compute node has no internet, pre-populate FluidGym's cache from a node that has (cache location not verified here) or set `HF_HOME` to shared storage.
3. **Dependency resolution.** TD-MPC2 pins `tensordict==0.8.3`/`torchrl==0.8.1` for torch 2.7.1. We keep torch 2.9 and leave those unpinned under a torch constraint, so setup fails loudly if no compatible release exists. If it does, pin versions in a copy of `requirements/tdmpc2_extra.txt` and pass `FGRL_TDMPC2_REQS`.
4. **TD-MPC2 harness assumptions (unverified).** The smoke test composes the upstream hydra config itself (needs `hydra-submitit-launcher`), sets `hydra.utils.get_original_cwd = os.getcwd`, and overrides `seed_steps`/`steps` after `make_env`. Upstream training evaluates once at step 0, and trains+evaluates on FluidGym's *train* split; SAC-vs-TD-MPC2 comparison on the *test* split needs a separate evaluator (not written).
5. **GPU memory numbers.** FluidGym's CUDA kernels may allocate outside PyTorch's allocator; compare `torch_peak_alloc_MiB` with `device_used_MiB_delta_during_bench`.
6. **Flow-field PNG axis order** is a heuristic (`orient()` in phase0). `render.png` (FluidGym's own renderer) is the authoritative picture; check both by eye — automated checks only prove non-blank.
7. **Determinism.** `reproducibility` fails if same seed + same actions differ by more than 1e-5; GPU non-determinism would show up there rather than be hidden. Adjust `--repro-tol` deliberately, not silently.


## Milestone: official SAC evaluation + TD-MPC2 CUDA smoke (run on the GPU server)

All commands assume the venv from `scripts/remote_setup.sh` is active, `pip install 'stable-baselines3[extra]>=2.7.0' 'huggingface_hub>=1.0.0'` has been done (the `sac` extra), and `FGRL_OUT_DIR` points where your Phase 0 report lives.

```bash
git pull && pip install -e ".[dev,sac]"
# 0) what do we already know about Phase 0? (no simulator run; exit 0 passed, 4 not_run, 1 failed/invalid)
python scripts/check_phase0_status.py --roots $FGRL_OUT_DIR
# if it is not 'passed', confirm on the GPU:  python scripts/phase0_validate.py --out $FGRL_OUT_DIR/phase0

# 1) one-shot, in the required order (static -> CPU tests -> CUDA tests -> Phase 0 evidence -> official SAC -> TD-MPC2 smoke -> reload + eval -> report)
scripts/remote_milestone.sh                       # report: results/milestone/<stamp>/MILESTONE_REPORT.md
#   knobs: FGRL_SAC_SEEDS=all  FGRL_SAC_EPISODES=10  FGRL_TDMPC2_EVAL_EPISODES=10  FGRL_REVISION=<hf commit>  FGRL_SAC_CHECKPOINT=/path/ckpt_latest.zip  FGRL_SKIP_TRAIN_SMOKE=1

# 2) or stage by stage
python scripts/evaluate_official_sac.py --train-seed 0 --dry-run          # download + contract check only (needs env creation, no episodes)
python scripts/evaluate_official_sac.py --train-seed 0 --episodes 10      # official checkpoint, deterministic, test split
python scripts/evaluate_official_sac.py --train-seed all --episodes 10    # all 5 official seeds + aggregate.json
python scripts/tdmpc2_smoke_train.py                                      # bounded CUDA smoke -> results/tdmpc2_smoke/<stamp>/
CK=$(ls -dt results/tdmpc2_smoke/*/checkpoints | head -1)
python scripts/tdmpc2_ckpt_verify.py --ckpt-dir $CK                       # fresh-process reload check (also run inside the smoke)
python scripts/evaluate_checkpoint.py tdmpc2 --checkpoint $CK/tdmpc2_smoke.pt --episodes 10 --mode plan
python scripts/evaluate_checkpoint.py tdmpc2 --checkpoint $CK/tdmpc2_smoke.pt --episodes 10 --mode actor
python scripts/evaluate_checkpoint.py compare results/official_sac_eval/<run> results/eval/<run>   # comparability class
```
Expected cost (estimates from the paper's ~2 s/step on an A100; measure with your Phase 0 numbers): SAC eval ≈ 10 episodes × 80 steps; TD-MPC2 smoke ≈ 400 steps plus ≈ 240 tiny updates; each TD-MPC2 evaluation additionally runs MPPI planning every step.
Exit codes of `evaluate_official_sac.py`: 0 ok, 2 setup/download error, 3 checkpoint/contract incompatibility (diagnostics printed), 5 Phase 0 evidence missing/not passed. **Do not bypass exit 3** by reshaping observations — send the diagnostic instead.
Same-contract comparison needs equal `--episodes`/`--base-seed`/`--protocol` (otherwise the comparison class is `non_comparable`).
Setting the Phase 1 gate (required for any full training) stays manual: `python scripts/mark_gate.py phase1 --evidence <path to the reviewed official_sac_eval summary.json>`.
