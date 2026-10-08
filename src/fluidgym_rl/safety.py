"""Experiment safety gates. Full training is blocked unless ALL of the following hold:
  1. --mode full AND --allow-full-training on the command line,
  2. FGRL_ALLOW_FULL_TRAINING=1 in the environment,
  3. gate files for Phase 0 (written automatically by a *passed* phase0_validate run on CUDA)
     and Phase 1 (written only by `scripts/mark_gate.py phase1 --evidence <file>`, i.e. a human decision
     after published-checkpoint evaluation) exist."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from fluidgym_rl.upstream import repo_root

REQUIRED = ("phase0", "phase1")


def gate_dir() -> Path:
    return Path(os.environ.get("FGRL_GATE_DIR") or Path(os.environ.get("FGRL_OUT_DIR") or repo_root() / "results") / "gates")


def write_gate(name: str, evidence: str | Path) -> Path:
    ev = Path(evidence)
    if not ev.is_file():
        raise FileNotFoundError(ev)
    d = gate_dir(); d.mkdir(parents=True, exist_ok=True)
    out = d / f"{name}.passed.json"
    out.write_text(json.dumps({"gate": name, "evidence": str(ev.resolve()), "sha256": hashlib.sha256(ev.read_bytes()).hexdigest()}, indent=2))
    return out


def missing_gates() -> list[str]:
    return [g for g in REQUIRED if not (gate_dir() / f"{g}.passed.json").is_file()]


def full_training_blockers(mode: str, allow_flag: bool) -> list[str]:
    b = []
    if mode != "full":
        b.append("mode is not 'full'")
    if not allow_flag:
        b.append("missing --allow-full-training")
    if os.environ.get("FGRL_ALLOW_FULL_TRAINING") != "1":
        b.append("FGRL_ALLOW_FULL_TRAINING=1 not set")
    b += [f"gate '{g}' not passed ({gate_dir() / (g + '.passed.json')})" for g in missing_gates()]
    return b
