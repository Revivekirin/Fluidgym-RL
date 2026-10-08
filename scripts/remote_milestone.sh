#!/usr/bin/env bash
# Remote milestone validation in the required order. Runs ON THE GPU SERVER inside the activated venv.
#   A static/imports -> B+C CPU & interface tests -> D Phase 0 (only if no passing evidence exists) -> E official SAC eval
#   -> F bounded TD-MPC2 CUDA smoke -> G checkpoint restoration + independent evaluation -> H consolidated report.
# Never uses FGRL_FORCE; a stage whose prerequisite did not pass is recorded as SKIPPED (not passed).
# Env (optional): FGRL_OUT_DIR (phase0 evidence/gates), FGRL_RESULTS_DIR (default results), FGRL_ENV_ID, FGRL_SAC_SEEDS (default 0; "all" = 0-4),
#   FGRL_SAC_EPISODES (10), FGRL_TDMPC2_EVAL_EPISODES (2; set equal to FGRL_SAC_EPISODES for a same-contract comparison),
#   FGRL_SKIP_TRAIN_SMOKE=1, FGRL_REVISION (pin HF commit), FGRL_SAC_CHECKPOINT (local zip instead of download)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
: "${FGRL_OUT_DIR:=$ROOT/results}" "${FGRL_RESULTS_DIR:=$ROOT/results}" "${FGRL_ENV_ID:=CylinderJet2D-easy-v0}" "${FGRL_SAC_SEEDS:=0}" "${FGRL_SAC_EPISODES:=10}" "${FGRL_TDMPC2_EVAL_EPISODES:=2}"
export FGRL_OUT_DIR FGRL_RESULTS_DIR
[ -f "${FGRL_VENV:-$ROOT/.venv}/bin/activate" ] && . "${FGRL_VENV:-$ROOT/.venv}/bin/activate"
PY="${FGRL_PYTHON:-python}"
RUN="$FGRL_RESULTS_DIR/milestone/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$RUN"; touch "$RUN/.start"; : > "$RUN/stage_rc.txt"; : > "$RUN/stage_skipped.txt"
echo "[milestone] run dir $RUN"
stage() { local n="$1"; shift; echo "[milestone] >>> $n"; "$@" >"$RUN/$n.log" 2>&1; local rc=$?; echo "$n $rc" >>"$RUN/stage_rc.txt"; echo "[milestone] <<< $n rc=$rc"; return $rc; }
skip()  { echo "$1 $2" >>"$RUN/stage_skipped.txt"; echo "[milestone] --- $1 SKIPPED: $2"; }
rc_of() { awk -v n="$1" '$1==n {print $2}' "$RUN/stage_rc.txt"; }

stage A_static        bash -c "$PY -m compileall -q src scripts && $PY -c 'import fluidgym_rl.evalcontract, fluidgym_rl.official_sac, fluidgym_rl.tdmpc2_checks, fluidgym_rl.status, fluidgym_rl.run_eval, fluidgym_rl.device'"
stage BC_cpu_tests    "$PY" -m pytest -q -m "not cuda" -p no:cacheprovider
stage CUDA_tests      "$PY" -m pytest -q -m cuda -p no:cacheprovider --junitxml "$RUN/cuda_tests.xml"

# D: reuse existing Phase 0 evidence; only (re)run the simulator validation if it is absent or not passing
"$PY" scripts/check_phase0_status.py --roots "$FGRL_OUT_DIR" --env-id "$FGRL_ENV_ID" >"$RUN/D0_phase0_status.json" 2>"$RUN/D0_phase0_status.err"; P0=$?; echo "D0_phase0_status $P0" >>"$RUN/stage_rc.txt"
if [ $P0 -ne 0 ]; then stage D_phase0_validate "$PY" scripts/phase0_validate.py --env-id "$FGRL_ENV_ID" --out "$FGRL_OUT_DIR/phase0"
  "$PY" scripts/check_phase0_status.py --roots "$FGRL_OUT_DIR" --env-id "$FGRL_ENV_ID" >"$RUN/D1_phase0_status.json" 2>"$RUN/D1_phase0_status.err"; P0=$?; echo "D1_phase0_status $P0" >>"$RUN/stage_rc.txt"; fi
if [ $P0 -ne 0 ]; then
  for s in E_official_sac_dry E_official_sac_eval F_tdmpc2_smoke G_tdmpc2_eval; do skip $s "Phase 0 not passed (status rc=$P0)"; done
else
  EXTRA=(); [ -n "${FGRL_REVISION:-}" ] && EXTRA+=(--revision "$FGRL_REVISION"); [ -n "${FGRL_SAC_CHECKPOINT:-}" ] && EXTRA+=(--checkpoint-path "$FGRL_SAC_CHECKPOINT")
  stage E_official_sac_dry  "$PY" scripts/evaluate_official_sac.py --env-id "$FGRL_ENV_ID" --train-seed "${FGRL_SAC_SEEDS%%,*}" --dry-run "${EXTRA[@]}"
  if [ "$(rc_of E_official_sac_dry)" = "0" ]; then
    stage E_official_sac_eval "$PY" scripts/evaluate_official_sac.py --env-id "$FGRL_ENV_ID" --train-seed "$FGRL_SAC_SEEDS" --episodes "$FGRL_SAC_EPISODES" "${EXTRA[@]}"
  else skip E_official_sac_eval "contract/dry-run failed (see E_official_sac_dry.log; rc=3 = checkpoint incompatibility)"; fi
  if [ "${FGRL_SKIP_TRAIN_SMOKE:-0}" = "1" ]; then skip F_tdmpc2_smoke "FGRL_SKIP_TRAIN_SMOKE=1"; skip G_tdmpc2_eval "no smoke checkpoint"
  else
    SMOKE="$FGRL_RESULTS_DIR/tdmpc2_smoke/$(basename "$RUN")_milestone"
    stage F_tdmpc2_smoke "$PY" scripts/tdmpc2_smoke_train.py --env-id "$FGRL_ENV_ID" --out "$SMOKE"
    CK="$SMOKE/checkpoints"
    if [ "$(rc_of F_tdmpc2_smoke)" = "0" ] && [ -f "$CK/tdmpc2_smoke.pt" ]; then
      stage G_tdmpc2_eval_plan  "$PY" scripts/evaluate_checkpoint.py tdmpc2 --checkpoint "$CK/tdmpc2_smoke.pt" --episodes "$FGRL_TDMPC2_EVAL_EPISODES" --mode plan  --env-id "$FGRL_ENV_ID"
      stage G_tdmpc2_eval_actor "$PY" scripts/evaluate_checkpoint.py tdmpc2 --checkpoint "$CK/tdmpc2_smoke.pt" --episodes "$FGRL_TDMPC2_EVAL_EPISODES" --mode actor --env-id "$FGRL_ENV_ID"
    else skip G_tdmpc2_eval "smoke did not pass"; fi
  fi
fi
"$PY" scripts/consolidate_milestone.py "$RUN"; exit $?
