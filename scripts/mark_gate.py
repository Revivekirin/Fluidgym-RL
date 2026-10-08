#!/usr/bin/env python
"""Human-in-the-loop gate. Phase 0 is written only by a passed phase0_validate run; Phase 1 only here, after you have inspected the
published-checkpoint evaluation evidence (Phase 1 evaluation is NOT implemented yet)."""
import argparse
from fluidgym_rl.safety import write_gate
ap = argparse.ArgumentParser(); ap.add_argument("gate", choices=["phase1"]); ap.add_argument("--evidence", required=True)
a = ap.parse_args(); print("wrote", write_gate(a.gate, a.evidence))
