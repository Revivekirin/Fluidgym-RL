#!/usr/bin/env bash
# One-time environment setup on the REMOTE CUDA server. No sudo; everything lives in user-writable paths.
# Configuration (all optional):
#   FGRL_VENV          venv location                      (default: <repo>/.venv)
#   FGRL_PYTHON        python used to create the venv      (default: python3; must be 3.10-3.13)
#   FGRL_OUT_DIR       results/gates/logs root             (default: <repo>/results)
#   FGRL_THIRD_PARTY   where upstream clones live          (default: <repo>/third_party)
#   FGRL_TORCH_INDEX   pip index for torch                 (default: https://download.pytorch.org/whl/cu128; "" = default PyPI)
#   FGRL_TORCH_SPEC    torch requirement                   (default: torch==2.9.*)
#   FGRL_FLUIDGYM_SPEC fluidgym requirement                (default: fluidgym==0.1.2, prebuilt wheel)
#   FGRL_TDMPC2_REQS   extra TD-MPC2 requirements file     (default: requirements/tdmpc2_extra.txt)
#   FGRL_PIP_ARGS      extra args for every pip install    (e.g. "--proxy http://...", "--no-cache-dir")
#   FGRL_SKIP_UPSTREAM=1  do not clone upstream repos
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${FGRL_VENV:=$ROOT/.venv}" "${FGRL_PYTHON:=python3}" "${FGRL_OUT_DIR:=$ROOT/results}" "${FGRL_THIRD_PARTY:=$ROOT/third_party}"
: "${FGRL_TORCH_INDEX=https://download.pytorch.org/whl/cu128}" "${FGRL_TORCH_SPEC:=torch==2.9.*}" "${FGRL_FLUIDGYM_SPEC:=fluidgym==0.1.2}"
: "${FGRL_TDMPC2_REQS:=$ROOT/requirements/tdmpc2_extra.txt}" "${FGRL_PIP_ARGS:=}"
export FGRL_OUT_DIR FGRL_THIRD_PARTY
SETUP_DIR="$FGRL_OUT_DIR/setup"; mkdir -p "$SETUP_DIR"
log() { printf '[remote_setup] %s\n' "$*"; }
log "root=$ROOT venv=$FGRL_VENV out=$FGRL_OUT_DIR third_party=$FGRL_THIRD_PARTY"

command -v "$FGRL_PYTHON" >/dev/null || { log "ERROR: $FGRL_PYTHON not found (set FGRL_PYTHON)"; exit 1; }
"$FGRL_PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,14), sys.version' \
  || { log "ERROR: need Python 3.10-3.13 (FluidGym requirement). Try a conda/pyenv python via FGRL_PYTHON."; exit 1; }
command -v git >/dev/null || { log "ERROR: git not found"; exit 1; }
if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee "$SETUP_DIR/nvidia_smi.txt"
else log "WARNING: nvidia-smi not on PATH (login node without GPU?). CUDA is verified later by torch."; fi

[ -d "$FGRL_VENV" ] || "$FGRL_PYTHON" -m venv "$FGRL_VENV"
# shellcheck disable=SC1091
. "$FGRL_VENV/bin/activate"
python -m pip install --quiet --upgrade pip $FGRL_PIP_ARGS

log "installing $FGRL_TORCH_SPEC"
if [ -n "$FGRL_TORCH_INDEX" ]; then python -m pip install $FGRL_PIP_ARGS "$FGRL_TORCH_SPEC" --index-url "$FGRL_TORCH_INDEX"
else python -m pip install $FGRL_PIP_ARGS "$FGRL_TORCH_SPEC"; fi
TORCH_VER="$(python -c 'import torch; print(torch.__version__)')"
echo "torch==$TORCH_VER" > "$SETUP_DIR/constraints.txt"   # keep torch fixed for every later install
log "torch $TORCH_VER pinned via constraints"

log "installing $FGRL_FLUIDGYM_SPEC"
python -m pip install $FGRL_PIP_ARGS -c "$SETUP_DIR/constraints.txt" "$FGRL_FLUIDGYM_SPEC"
log "installing TD-MPC2 runtime deps from $FGRL_TDMPC2_REQS"
python -m pip install $FGRL_PIP_ARGS -c "$SETUP_DIR/constraints.txt" -r "$FGRL_TDMPC2_REQS"
log "installing fluidgym-rl (editable) + dev deps"
python -m pip install $FGRL_PIP_ARGS -c "$SETUP_DIR/constraints.txt" -e "$ROOT[dev]" pillow

if [ "${FGRL_SKIP_UPSTREAM:-0}" != "1" ]; then log "fetching pinned upstream repos"; "$ROOT/scripts/fetch_upstream.sh"; fi

python -m pip freeze > "$SETUP_DIR/pip_freeze.txt"
python - <<'PY'
import sys, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| torch CUDA runtime", torch.version.cuda)
ok = torch.cuda.is_available()
print("cuda available:", ok)
if not ok:
    print("WARNING: CUDA not available in this shell. Run on a GPU node (scheduler/allocation) before remote_smoke.sh.")
    sys.exit(0)
x = torch.randn(256, 256, device="cuda"); print("gpu matmul ok:", bool(torch.isfinite((x @ x).sum())), "| device:", torch.cuda.get_device_name(0))
PY
log "done. Activate with: . $FGRL_VENV/bin/activate ; then run scripts/remote_smoke.sh on a GPU node. pip freeze: $SETUP_DIR/pip_freeze.txt"
