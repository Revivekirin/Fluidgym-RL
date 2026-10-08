"""Regression coverage for Phase 0 recovery and missing evidence."""
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("initial,final,expected", [(1, 0, 0), (4, 0, 0), (1, 1, 1), (4, 4, 1), (4, None, 1)])
def test_final_phase0_evidence_controls_recovery(tmp_path, initial, final, expected):
    run = tmp_path / "milestone" / "run"
    run.mkdir(parents=True)
    (run / ".start").touch()
    stages = f"D0_phase0_status {initial}\n"
    if final is not None:
        stages += f"D_phase0_validate 0\nD1_phase0_status {final}\n"
    (run / "stage_rc.txt").write_text(stages)
    (run / "stage_skipped.txt").write_text("")
    script = Path(__file__).resolve().parents[1] / "scripts/consolidate_milestone.py"
    result = subprocess.run([sys.executable, str(script), str(run)], capture_output=True, text=True)
    assert result.returncode == expected, result.stdout + result.stderr
    if final is not None:
        assert "superseded" in result.stdout and "D1_phase0_status" in result.stdout
