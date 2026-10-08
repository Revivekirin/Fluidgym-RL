import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from fluidgym_rl.status import PHASE0_REQUIRED, phase0_status


def _report(root: Path, env="CylinderJet2D-easy-v0", status="passed", bad=None, drop=None, name="phase0"):
    d = root / "phase0" / env; d.mkdir(parents=True)
    checks = {c: {"status": "passed"} for c in PHASE0_REQUIRED}
    for c in (drop or []): checks.pop(c)
    for c in (bad or []): checks[c] = {"status": "failed"}
    (d / "report.json").write_text(json.dumps({"name": f"{name}:{env}", "status": status, "finished_utc": "t", "checks": checks, "meta": {"gpu": {"name": "X"}, "torch": {"version": "2.9"}}}))
    return d / "report.json"


def test_missing_report_is_not_run_not_failed(tmp_path):
    st = phase0_status([tmp_path]); assert st["status"] == "not_run"
    assert phase0_status([tmp_path / "does-not-exist"])["status"] == "not_run"


def test_passed_failed_invalid_reports(tmp_path):
    assert phase0_status([_report(tmp_path / "a")])["status"] == "passed"
    st = phase0_status([_report(tmp_path / "b", status="failed", bad=["episode_gif"])]); assert st["status"] == "failed" and "episode_gif" in st["non_passed_checks"]
    assert phase0_status([_report(tmp_path / "c", drop=["render_png"])])["status"] == "invalid_report"
    assert phase0_status([_report(tmp_path / "d", status="running")])["status"] == "invalid_report"
    p = _report(tmp_path / "e"); p.write_text("{not json"); assert phase0_status([p])["status"] == "invalid_report"


def test_all_checks_passing_but_report_status_failed_is_not_passed(tmp_path):
    assert phase0_status([_report(tmp_path, status="failed")])["status"] == "failed"


# ---- CUDA alias normalization on the project's own fluidgym_rl/device.py, with a stand-in torch so it runs without torch/CUDA
class _Dev:
    def __init__(self, t, i=None):
        if isinstance(t, _Dev): t, i = t.type, t.index
        elif ":" in t: t, i = t.split(":")[0], int(t.split(":")[1])
        self.type, self.index = t, i
    def __eq__(self, o): return (self.type, self.index) == (o.type, o.index)
    def __hash__(self): return hash((self.type, self.index))


def _load_device_module(current=0, available=True):
    fake = types.ModuleType("torch")
    fake.device = _Dev
    fake.cuda = types.SimpleNamespace(is_available=lambda: available, current_device=lambda: current)
    old = sys.modules.get("torch"); sys.modules["torch"] = fake
    try:
        spec = importlib.util.spec_from_file_location("device_under_test", Path(__file__).parents[1] / "src/fluidgym_rl/device.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    finally:
        sys.modules.pop("torch") if old is None else sys.modules.__setitem__("torch", old)
    return m


def test_cuda_alias_equals_explicit_index_on_single_gpu():
    m = _load_device_module(current=0)
    assert m.same_device("cuda", "cuda:0") and m.same_device("cuda:0", "cuda") and m.same_device("cuda", "cuda")


def test_cpu_vs_cuda_and_distinct_indices_are_mismatches():
    m = _load_device_module(current=0)
    assert not m.same_device("cpu", "cuda:0") and not m.same_device("cuda:0", "cuda:1")


def test_alias_follows_current_device_so_real_mismatch_survives():
    m = _load_device_module(current=1)
    assert m.same_device("cuda", "cuda:1") and not m.same_device("cuda", "cuda:0")


def test_alias_without_cuda_raises_instead_of_guessing():
    m = _load_device_module(available=False)
    with pytest.raises(RuntimeError): m.same_device("cuda", "cuda:0")


def test_real_torch_cuda_alias_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available(): pytest.skip("CUDA required (skipped != passed)")
    from fluidgym_rl.device import same_device
    assert same_device("cuda", f"cuda:{torch.cuda.current_device()}") and not same_device("cpu", "cuda")
    if torch.cuda.device_count() > 1:
        assert not same_device("cuda:0", "cuda:1")
