#!/usr/bin/env bash
# Manual remote invocation only. No training. About 16,800 simulator steps,
# before optional independent train evaluations; budget hours, not seconds.
set -euo pipefail
cd /nfs4/jhkim/repos/Fluidgym-RL
T1="$PWD/results/tdmpc_full/20261012T060719Z_full"
T2="$PWD/results/tdmpc2_full/20261008T124249Z_4be46ae7"
OUT="$PWD/results/nominal_comparison/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
python -c 'import torch, omegaconf, hydra, pyarrow; assert torch.cuda.is_available()'
PYTHONPATH=src python -m pytest tests/test_comparison.py -q
python scripts/tdmpc_train.py --checkpoint "$T1/checkpoints/best.pt" --verify-checkpoint --device cuda:0 --output-root "$OUT/restore_best"
python scripts/tdmpc_train.py --checkpoint "$T1/checkpoints/final.pt" --verify-checkpoint --device cuda:0 --output-root "$OUT/restore_final"
python scripts/compare_nominal.py audit --training-runs "$T1" "$T2" --out "$OUT/original_audit.json"
python scripts/compare_nominal.py evaluate --algorithm uncontrolled --split train --role reference --episodes 10 --base-seed 0 --device cuda:0 --out "$OUT/reference"
# Each candidate gets its own fresh process, then validation selection is frozen.
# Both suites evaluate selected and final models on test, with no training.
python scripts/compare_nominal_run.py --algorithm tdmpc --training-run "$T1" --reference "$OUT/reference" --episodes 10 --base-seed 0 --device cuda:0 --out "$OUT/tdmpc"
python scripts/compare_nominal_run.py --algorithm tdmpc2 --training-run "$T2" --reference "$OUT/reference" --episodes 10 --base-seed 0 --device cuda:0 --out "$OUT/tdmpc2"
# To obtain independent train-vs-val panels, add --include-train to BOTH suite
# commands above before running: another 12,800 simulator steps with these files.
python - "$OUT" "$T1" "$T2" <<'PY'
import json,subprocess,sys
from pathlib import Path
out=Path(sys.argv[1]); evaluations=[]
for alg in ('tdmpc','tdmpc2'):
    manifest=json.loads((out/alg/'suite_manifest.json').read_text())
    assert manifest['status']=='completed'
    evaluations.extend(manifest['evaluations'])
subprocess.run([sys.executable,'scripts/compare_nominal.py','report','--evaluations',*evaluations,
 '--training-runs',*sys.argv[2:],'--out',str(out/'dashboard')],check=True)
PY
printf 'Comparison output: %s\n' "$OUT"
