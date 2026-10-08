"""CPU INTERFACE tests for the harness (reporting, gates, patch application, artifact verification).
They do not touch FluidGym and are NOT evidence of FluidGym/CUDA compatibility."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fluidgym_rl.report import CheckFailure, FatalStop, Report
from fluidgym_rl import safety
from fluidgym_rl.upstream import lock, make_tdmpc2_workcopy, repo_root, third_party, upstream_status


def test_report_statuses_and_visible_failures(tmp_path, capsys):
    r = Report(tmp_path / "r.json", "t", planned=["ok", "bad", "boom", "never"])
    r.check("ok", lambda: {"x": 1})
    r.check("bad", lambda: (_ for _ in ()).throw(CheckFailure("nope", {"d": 2})))
    r.check("boom", lambda: 1 / 0)
    assert r.finalize() == "failed"
    d = json.loads((tmp_path / "r.json").read_text())
    assert d["checks"]["ok"]["status"] == "passed" and d["checks"]["bad"]["details"] == {"d": 2}
    assert d["checks"]["boom"]["status"] == "error" and "ZeroDivisionError" in d["checks"]["boom"]["traceback"]
    assert d["checks"]["never"]["status"] == "skipped"
    stderr = capsys.readouterr().err
    assert "boom" in stderr and "ZeroDivisionError" in stderr and "nope" in stderr


def test_all_pass_needed_for_passed_and_skips_are_incomplete(tmp_path):
    r = Report(tmp_path / "a.json", "t", planned=["a", "b"]); r.check("a", lambda: True)
    assert r.finalize() == "incomplete"
    r = Report(tmp_path / "b.json", "t", planned=["a"]); r.check("a", lambda: True)
    assert r.finalize() == "passed"


def test_fatal_stops_and_marks_rest_skipped(tmp_path):
    r = Report(tmp_path / "c.json", "t", planned=["a", "b"])
    with pytest.raises(FatalStop):
        r.check("a", lambda: False, fatal=True)
    assert r.finalize() == "failed" and r.data["checks"]["b"]["status"] == "skipped"


def test_full_training_blocked_by_default_and_needs_every_condition(tmp_path, monkeypatch):
    monkeypatch.setenv("FGRL_GATE_DIR", str(tmp_path / "gates")); monkeypatch.delenv("FGRL_ALLOW_FULL_TRAINING", raising=False)
    assert len(safety.full_training_blockers("smoke", False)) >= 4
    ev = tmp_path / "ev.json"; ev.write_text("{}")
    safety.write_gate("phase0", ev)
    assert any("phase1" in b for b in safety.full_training_blockers("full", True))
    safety.write_gate("phase1", ev); monkeypatch.setenv("FGRL_ALLOW_FULL_TRAINING", "1")
    assert safety.full_training_blockers("full", True) == []
    assert safety.full_training_blockers("full", False) != []


def test_train_full_script_refuses(tmp_path):
    env = {**os.environ, "FGRL_GATE_DIR": str(tmp_path / "g"), "FGRL_OUT_DIR": str(tmp_path)}; env.pop("FGRL_ALLOW_FULL_TRAINING", None)
    p = subprocess.run([sys.executable, str(repo_root() / "scripts/tdmpc2_train_full.py"), "--steps", "50000", "--mode", "full", "--allow-full-training",
                        "--out", str(tmp_path / "o")], capture_output=True, text=True, env=env)
    assert p.returncode == 3 and "BLOCKED" in p.stdout


def test_phase0_without_cuda_is_visibly_not_passed(tmp_path):
    try:
        import torch
        if torch.cuda.is_available(): pytest.skip("CUDA present")
    except ImportError:
        pass
    env = {**os.environ, "FGRL_OUT_DIR": str(tmp_path), "FGRL_GATE_DIR": str(tmp_path / "gates")}
    p = subprocess.run([sys.executable, str(repo_root() / "scripts/phase0_validate.py"), "--out", str(tmp_path / "p0")], capture_output=True, text=True, env=env)
    assert p.returncode == 2, p.stderr
    d = json.loads(next((tmp_path / "p0").rglob("report.json")).read_text())
    assert d["status"] == "failed" and d["checks"]["cuda_available"]["status"] == "failed"
    assert all(c["status"] in ("failed", "skipped") for c in d["checks"].values())
    assert not (tmp_path / "gates" / "phase0.passed.json").exists()


@pytest.mark.skipif(not (third_party() / "tdmpc2" / ".git").exists(), reason="third_party/tdmpc2 not fetched")
def test_workcopy_applies_patch_and_leaves_upstream_untouched(tmp_path):
    before = upstream_status("tdmpc2")
    info = make_tdmpc2_workcopy(tmp_path / "wc")
    assert info["changed_files"] == ["tdmpc2/envs/__init__.py"]
    assert "make_fluidgym_env" in (Path(info["code_dir"]) / "envs" / "__init__.py").read_text()
    after = upstream_status("tdmpc2")
    assert after["head"] == before["head"] == lock()["tdmpc2"][1] and after["dirty"] is False


def test_artifact_verification(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    from fluidgym_rl.artifacts import verify_gif, verify_png
    rng = np.random.default_rng(0)
    PIL.fromarray((rng.random((20, 30, 3)) * 255).astype("uint8")).save(tmp_path / "a.png")
    assert verify_png(tmp_path / "a.png")["shape"] == [20, 30, 3]
    PIL.fromarray(np.zeros((20, 30, 3), "uint8")).save(tmp_path / "blank.png")
    with pytest.raises(CheckFailure): verify_png(tmp_path / "blank.png")
    with pytest.raises(CheckFailure): verify_png(tmp_path / "missing.png")
    fr = [PIL.fromarray(np.full((10, 10, 3), 40 * i, "uint8")) for i in range(4)]
    fr[0].save(tmp_path / "g.gif", save_all=True, append_images=fr[1:], duration=50)
    assert verify_gif(tmp_path / "g.gif")["frames"] == 4
    fr[0].save(tmp_path / "one.gif")
    with pytest.raises(CheckFailure): verify_gif(tmp_path / "one.gif")
