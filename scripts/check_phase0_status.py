#!/usr/bin/env python
"""Inspect existing Phase 0 evidence without running anything. Exit 0 = passed, 4 = not_run (no evidence: UNKNOWN, not a failure), 1 = failed/invalid."""
import argparse, json, os, sys
from fluidgym_rl.status import phase0_status

ap = argparse.ArgumentParser()
ap.add_argument("--roots", nargs="*", default=[os.environ.get("FGRL_OUT_DIR", "results")])
ap.add_argument("--env-id", default="CylinderJet2D-easy-v0")
a = ap.parse_args()
st = phase0_status(a.roots, a.env_id)
print(json.dumps(st, indent=2, default=str))
if st["status"] != "passed":
    print("\nTo (re)confirm on the GPU server:\n  python scripts/phase0_validate.py --env-id %s --out $FGRL_OUT_DIR/phase0" % a.env_id, file=sys.stderr)
sys.exit({"passed": 0, "not_run": 4}.get(st["status"], 1))
