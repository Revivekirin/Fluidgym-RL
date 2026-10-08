"""Locating pinned upstream clones and building a patched, throw-away TD-MPC2 work copy.
The pinned clone in third_party/ is never modified."""
from __future__ import annotations

import filecmp
import os
import shutil
import subprocess
from pathlib import Path


def repo_root() -> Path:
    return Path(os.environ.get("FGRL_ROOT") or Path(__file__).resolve().parents[2])


def third_party() -> Path:
    return Path(os.environ.get("FGRL_THIRD_PARTY") or repo_root() / "third_party")


def lock() -> dict[str, tuple[str, str]]:
    out = {}
    f = repo_root() / "UPSTREAM.lock"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                name, url, sha = line.split()[:3]
                out[name] = (url, sha)
    return out


def upstream_status(name: str) -> dict:
    path, expected = third_party() / name, lock()[name][1]
    def g(*a):
        return subprocess.run(["git", *a], cwd=path, capture_output=True, text=True).stdout.strip()
    head = g("rev-parse", "HEAD") if path.exists() else None
    return {"path": str(path), "expected": expected, "head": head, "matches": head == expected,
            "dirty": bool(g("status", "--porcelain")) if head else None}


def _changed_files(a: Path, b: Path) -> list[str]:
    out = []
    def walk(d: filecmp.dircmp, rel=""):
        out.extend(f"{rel}{f}" for f in d.diff_files + d.left_only + d.right_only)
        for n, sub in d.subdirs.items():
            walk(sub, f"{rel}{n}/")
    walk(filecmp.dircmp(a, b, ignore=[".git", "__pycache__"]))
    return sorted(out)


def make_tdmpc2_workcopy(dest: str | Path) -> dict:
    """Copy pinned tdmpc2 (without .git) to dest, apply patches/tdmpc2_fluidgym_task.patch, and verify that
    exactly the expected file changed. Raises if the clone is not at the pinned SHA or is dirty."""
    st = upstream_status("tdmpc2")
    if not st["matches"] or st["dirty"]:
        raise RuntimeError(f"third_party/tdmpc2 must be clean and at the pinned SHA: {st}")
    src, dest = Path(st["path"]), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    patch = repo_root() / "patches" / "tdmpc2_fluidgym_task.patch"
    subprocess.run(["git", "init", "-q", str(dest)], check=True)  # own throw-away repo => unambiguous `git apply` root
    subprocess.run(["git", "apply", str(patch)], cwd=dest, check=True)
    shutil.rmtree(dest / ".git")
    changed = _changed_files(src, dest)
    if changed != ["tdmpc2/envs/__init__.py"]:
        raise RuntimeError(f"unexpected files changed by patch: {changed}")
    return {"workcopy": str(dest), "code_dir": str(dest / "tdmpc2"), "patch": str(patch), "changed_files": changed, "upstream": st}
