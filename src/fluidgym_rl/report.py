"""JSON reporting for remote validation. Design rule: a check is 'passed' ONLY if it executed and
its assertions held. Exceptions are recorded as 'error' (with traceback), never swallowed, and checks that
were planned but never reached are 'skipped'. Overall status is 'passed' only if every planned check passed."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


class CheckFailure(Exception):
    def __init__(self, msg: str, details: dict | None = None):
        super().__init__(msg)
        self.details = details


class FatalStop(Exception):
    """Raised by Report.check(fatal=True) when a prerequisite check did not pass."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Report:
    def __init__(self, path, name: str, planned: list[str] | None = None, scope: str = "remote-cuda"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.planned = list(planned or [])
        self.data = {"name": name, "scope": scope, "started_utc": _now(), "meta": {}, "checks": {}, "status": "running"}
        self._flush()

    def _flush(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str))
        os.replace(tmp, self.path)

    def set_meta(self, **kw) -> None:
        self.data["meta"].update(kw)
        self._flush()

    def check(self, name: str, fn, *, fatal: bool = False) -> dict:
        t0 = time.time()
        try:
            res = fn()
            if res is False:
                rec = {"status": "failed", "error": "check returned False"}
            else:
                rec = {"status": "passed"}
                if res is not None and res is not True:
                    rec["details"] = res
        except CheckFailure as e:
            rec = {"status": "failed", "error": str(e)}
            if e.details is not None:
                rec["details"] = e.details
        except Exception as e:  # noqa: BLE001 - recorded, not hidden
            rec = {"status": "error", "error": repr(e), "traceback": traceback.format_exc()}
        rec["seconds"] = round(time.time() - t0, 3)
        self.data["checks"][name] = rec
        self._flush()
        if rec["status"] in ("failed", "error"):
            print(f"[{self.data['name']}] {name}: {rec['error']}", file=sys.stderr)
            if rec.get("traceback"):
                print(rec["traceback"], file=sys.stderr, end="")
        if fatal and rec["status"] != "passed":
            self.data["fatal_stop"] = name
            self._flush()
            raise FatalStop(name)
        return rec

    def skip(self, name: str, reason: str) -> None:
        self.data["checks"][name] = {"status": "skipped", "reason": reason}
        self._flush()

    def finalize(self) -> str:
        for n in self.planned:
            if n not in self.data["checks"]:
                self.data["checks"][n] = {"status": "skipped", "reason": "not reached (earlier fatal failure)"}
        sts = [c["status"] for c in self.data["checks"].values()]
        if any(s in ("failed", "error") for s in sts):
            status = "failed"
        elif any(s == "skipped" for s in sts) or not sts:
            status = "incomplete"
        else:
            status = "passed"
        self.data.update(status=status, finished_utc=_now(), counts={s: sts.count(s) for s in set(sts)})
        self._flush()
        return status


def _run(cmd: list[str], cwd=None) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=cwd).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def git_info(path) -> dict:
    p = str(path)
    if not Path(p).exists():
        return {"path": p, "present": False}
    head = _run(["git", "rev-parse", "HEAD"], cwd=p)
    dirty = _run(["git", "status", "--porcelain"], cwd=p)
    return {"path": p, "present": True, "head": head, "dirty": bool(dirty) if head else None}


def collect_metadata() -> dict:
    from fluidgym_rl.upstream import repo_root, third_party, lock

    meta = {"timestamp_utc": _now(), "host": platform.node(), "platform": platform.platform(),
            "python": sys.version.split()[0], "repo": git_info(repo_root()),
            "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "FGRL_OUT_DIR", "FGRL_THIRD_PARTY")}}
    meta["upstream_pins"] = {n: {"expected_sha": sha, "local_clone": git_info(third_party() / n)} for n, (_, sha) in lock().items()}
    meta["dependency_versions"] = {}
    for package in ("tensordict", "torchrl", "gymnasium", "hydra-core", "omegaconf", "wandb", "numpy", "pandas"):
        try:
            meta["dependency_versions"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            meta["dependency_versions"][package] = None
    try:
        import torch
        meta["torch"] = {"version": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            meta["gpu"] = {"name": props.name, "capability": f"{props.major}.{props.minor}", "total_MiB": props.total_memory / 2**20,
                           "device_count": torch.cuda.device_count()}
    except Exception as e:  # noqa: BLE001
        meta["torch"] = {"error": repr(e)}
    meta["nvidia_smi"] = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    try:
        dist = metadata.distribution("fluidgym")
        import fluidgym
        meta["fluidgym"] = {"version": dist.version, "module_path": getattr(fluidgym, "__file__", None),
                            "direct_url": dist.read_text("direct_url.json"),
                            "note": "installed artifact (e.g. PyPI wheel) is NOT guaranteed identical to the pinned git SHA in upstream_pins"}
    except Exception as e:  # noqa: BLE001
        meta["fluidgym"] = {"error": repr(e)}
    return meta
