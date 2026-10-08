"""Phase 0 evidence inspection. A missing report is `not_run` (unknown), NEVER `failed`."""
from __future__ import annotations

import json
from pathlib import Path

PHASE0_REQUIRED = ["cuda_available", "gpu_tensor_op", "env_create", "reset_obs_contract", "action_contract", "step_contract", "action_bounds_behavior",
                   "action_affects_state", "episode_termination", "reproducibility", "seed_sensitivity", "throughput_and_memory", "render_png",
                   "flow_field_png", "episode_gif"]


def find_phase0_reports(roots, env_id: str) -> list[Path]:
    out = []
    for r in roots:
        r = Path(r)
        if r.is_file():
            out.append(r)
        elif r.is_dir():
            out += [p for p in r.rglob("report.json") if p.parent.name == env_id and p.parent.parent.name == "phase0" or p.parent.name == env_id]
    return sorted(set(out), key=lambda p: p.stat().st_mtime, reverse=True)


def phase0_status(roots, env_id: str = "CylinderJet2D-easy-v0", required=PHASE0_REQUIRED) -> dict:
    reports = find_phase0_reports(roots, env_id)
    if not reports:
        return {"status": "not_run", "reason": f"no phase0 report.json for {env_id} under {[str(r) for r in roots]}", "required_checks": required}
    for p in reports:                                   # newest first
        try:
            d = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            return {"status": "invalid_report", "report": str(p), "reason": repr(e)}
        if not str(d.get("name", "")).startswith("phase0"):
            continue
        checks = d.get("checks", {})
        missing = [c for c in required if c not in checks]
        bad = {c: v.get("status") for c, v in checks.items() if v.get("status") != "passed"}
        meta = d.get("meta", {})
        base = {"report": str(p), "report_status": d.get("status"), "finished_utc": d.get("finished_utc"), "torch": meta.get("torch"), "gpu": meta.get("gpu"),
                "fluidgym": (meta.get("fluidgym") or {}).get("version"), "repo_sha": (meta.get("repo") or {}).get("head")}
        if d.get("status") == "running" or "status" not in d:
            return {**base, "status": "invalid_report", "reason": "report never finalized (run crashed or still running)"}
        if missing:
            return {**base, "status": "invalid_report", "reason": f"required checks absent from report: {missing}"}
        if d.get("status") == "passed" and not bad:
            return {**base, "status": "passed"}
        return {**base, "status": "failed", "non_passed_checks": bad}
    return {"status": "not_run", "reason": "reports found but none is a phase0 report", "reports": [str(p) for p in reports]}
