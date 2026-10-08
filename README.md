# fluidgym-rl
Reproduction + TD-MPC2 integration harness for [FluidGym](https://github.com/safe-autonomous-systems/fluidgym). Independent of any other DRL project. Upstream repos are pinned (`UPSTREAM.lock`), cloned into `third_party/`, and never edited; compatibility patches in `patches/` are applied only to throw-away copies at run time; the optional training logging patch changes no learning code.

**FluidGym only runs on a remote CUDA server.** Local machine = editing + CPU interface tests. Read `docs/REMOTE_EXECUTION.md` and `IMPLEMENTATION_STATUS.md` first.

```bash
# local (no GPU): interface tests only -- NOT evidence of FluidGym compatibility
pip install -e ".[dev]" pillow && pytest -m "not cuda"

# remote (GPU server)
scripts/remote_setup.sh          # once
scripts/remote_smoke.sh          # Phase 0 CUDA validation + adapter integration + TD-MPC2 smoke training
```
Scope decisions: TD-MPC2 only; D-MPC (paper App. D.3) is the DPC target and lives separately in `src/fluidgym_rl/dmpc/` (not implemented); full training is gated (`src/fluidgym_rl/safety.py`).
Milestone (official SAC + TD-MPC2 smoke): `scripts/remote_milestone.sh`; see docs/REMOTE_EXECUTION.md Milestone section.
Docs: `docs/UPSTREAM_AUDIT.md`, `docs/REPRODUCTION_AUDIT.md`, `docs/REMOTE_EXECUTION.md`.

## 2026-10-08 remote failure follow-up
The supplied server logs show SAC loading blocked by missing `omegaconf` and TD-MPC2 trainer construction blocked by missing `hydra`. SAC's extra now declares OmegaConf; TD-MPC2 additionally requires `requirements/tdmpc2_extra.txt` (see the recovery commands in `docs/REMOTE_EXECUTION.md`). No training updates or checkpoint restoration were reached in that run. CUDA and patched-workcopy checks passed in the supplied smoke report; full smoke and independent evaluation remain blocked pending a successful server rerun. No upstream algorithm or patch was changed by this fix.

Bounded nominal training and optional W&B logging are available through
`scripts/tdmpc2_train_full.py --mode pilot --steps 2400 --wandb-mode offline`.
Pilot training is capped at 5,000 environment transitions; full training retains
all existing gates. See `docs/REMOTE_EXECUTION.md` under "Configurable W&B and
bounded nominal pilot" for installation, validation, periodic saves and evaluation.
