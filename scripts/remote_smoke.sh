#!/usr/bin/env bash
# Remote CUDA validation. Runs every stage even if an earlier one fails (except where noted) and exits non-zero unless ALL stages
# actually executed and passed. Results: $FGRL_OUT_DIR/remote_smoke/<UTC stamp>/ (SUMMARY.md, summary.json, per-stage logs + reports).
# Configuration (optional):
#   FGRL_VENV, FGRL_OUT_DIR, FGRL_THIRD_PARTY, FGRL_PYTHON   as in remote_setup.sh (venv is activated if it exists)
#   FGRL_ENV_ID=CylinderJet2D-easy-v0     environment under test
#   FGRL_BENCH_STEPS=20                   env steps timed for throughput
#   FGRL_SKIP_TRAIN_SMOKE=1               skip the TD-MPC2 training smoke (slowest stage)
#   FGRL_SMOKE_SEED_EPISODES=2 FGRL_SMOKE_TRAIN_EPISODES=1   smoke-training size (episodes of env.episode_length steps)
#   FGRL_FORCE=1                          run later stages even if phase0/integration failed
#   FGRL_WORK_DIR                         scratch dir for the patched TD-MPC2 copy (default: $TMPDIR/fgrl_work)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${FGRL_VENV:=$ROOT/.venv}" "${FGRL_OUT_DIR:=$ROOT/results}" "${FGRL_THIRD_PARTY:=$ROOT/third_party}" "${FGRL_ENV_ID:=CylinderJet2D-easy-v0}"
export FGRL_OUT_DIR FGRL_THIRD_PARTY FGRL_BENCH_STEPS="${FGRL_BENCH_STEPS:-20}"
[ -f "$FGRL_VENV/bin/activate" ] && . "$FGRL_VENV/bin/activate"
PY="${FGRL_PYTHON:-python}"
RUN="$FGRL_OUT_DIR/remote_smoke/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$RUN"; : > "$RUN/stage_rc.txt"; : > "$RUN/stage_skipped.txt"
cd "$ROOT"
echo "[remote_smoke] run dir: $RUN"
"$PY" - > "$RUN/preflight.json" 2>"$RUN/preflight.err" <<'PYEOF'
import json
from fluidgym_rl.report import collect_metadata
print(json.dumps(collect_metadata(), indent=2, default=str))
PYEOF

stage() { local name="$1"; shift; echo "[remote_smoke] >>> $name"; "$@" > "$RUN/$name.log" 2>&1; local rc=$?; echo "$name $rc" >> "$RUN/stage_rc.txt"; echo "[remote_smoke] <<< $name rc=$rc"; return $rc; }
skip()  { echo "$1 $2" >> "$RUN/stage_skipped.txt"; echo "[remote_smoke] --- $1 SKIPPED: $2"; }
rc_of() { awk -v n="$1" '$1==n {print $2}' "$RUN/stage_rc.txt"; }

stage interface_tests "$PY" -m pytest -q -m "not cuda" -p no:cacheprovider                      # interface tests only
stage cuda_pytests    "$PY" -m pytest -q -m cuda -p no:cacheprovider --junitxml "$RUN/cuda_pytests.xml"
stage phase0          "$PY" scripts/phase0_validate.py --env-id "$FGRL_ENV_ID" --out "$RUN/phase0"
if [ "$(rc_of phase0)" = "0" ] || [ "${FGRL_FORCE:-0}" = "1" ]; then
  stage integration   "$PY" scripts/integration_cuda.py --env-id "$FGRL_ENV_ID" --out "$RUN/integration"
else skip integration "phase0 did not pass (set FGRL_FORCE=1 to override)"; fi
if [ "${FGRL_SKIP_TRAIN_SMOKE:-0}" = "1" ]; then skip tdmpc2_smoke "FGRL_SKIP_TRAIN_SMOKE=1"
elif { [ "$(rc_of phase0)" = "0" ] && [ "$(rc_of integration)" = "0" ]; } || [ "${FGRL_FORCE:-0}" = "1" ]; then
  stage tdmpc2_smoke  "$PY" scripts/tdmpc2_smoke_train.py --env-id "$FGRL_ENV_ID" --out "$RUN/tdmpc2_smoke"
else skip tdmpc2_smoke "phase0/integration did not pass (set FGRL_FORCE=1 to override)"; fi

"$PY" scripts/summarize_remote.py "$RUN"; exit $?
