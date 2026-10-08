#!/usr/bin/env python
"""Aggregate stage results of one remote_smoke run into SUMMARY.md / summary.json. Only reports what the stage files contain."""
import json, sys, xml.etree.ElementTree as ET
from pathlib import Path

run = Path(sys.argv[1])
rc = {k: int(v) for k, v in (l.split() for l in (run / "stage_rc.txt").read_text().splitlines() if l.strip())} if (run / "stage_rc.txt").exists() else {}
skipped = {l.split(maxsplit=1)[0]: l.split(maxsplit=1)[1] for l in (run / "stage_skipped.txt").read_text().splitlines() if l.strip()} if (run / "stage_skipped.txt").exists() else {}
stages = {}
for name in ("interface_tests", "cuda_pytests", "phase0", "integration", "tdmpc2_smoke"):
    if name in skipped:
        stages[name] = {"status": "skipped", "reason": skipped[name]}; continue
    if name not in rc:
        stages[name] = {"status": "not_run"}; continue
    s = {"exit_code": rc[name], "status": "passed" if rc[name] == 0 else "failed"}
    rj = {"phase0": "phase0/*/report.json", "integration": "integration/report.json", "tdmpc2_smoke": "tdmpc2_smoke/report.json"}.get(name)
    if rj:
        fs = list(run.glob(rj)); s["report"] = str(fs[0].relative_to(run)) if fs else None
        if fs:
            d = json.loads(fs[0].read_text()); s["report_status"] = d["status"]; s["counts"] = d.get("counts")
            if d["status"] != "passed": s["status"] = "failed" if d["status"] == "failed" else d["status"]
        else:
            s["status"] = "failed"; s["note"] = "no report.json produced"
    if name == "cuda_pytests":
        x = run / "cuda_pytests.xml"
        if x.exists():
            r = ET.parse(x).getroot(); ts = r if r.tag == "testsuite" else r.find("testsuite")
            n, f, e, k = (int(ts.get(t, 0)) for t in ("tests", "failures", "errors", "skipped")); s.update(tests=n, failures=f, errors=e, skipped=k, passed=n - f - e - k)
            if s["passed"] == 0: s["status"] = "failed"; s["note"] = "zero CUDA tests actually passed (skipped tests are not evidence)"
        else:
            s["status"] = "failed"; s["note"] = "no junit xml"
    stages[name] = s
overall = "passed" if all(v["status"] == "passed" for k, v in stages.items()) else "NOT PASSED"
(run / "summary.json").write_text(json.dumps({"overall": overall, "stages": stages}, indent=2))
lines = [f"# remote_smoke summary: {overall}", "", "| stage | status | detail |", "|---|---|---|"]
for k, v in stages.items():
    lines.append(f"| {k} | {v['status']} | {v.get('reason') or v.get('note') or v.get('counts') or ''} |")
lines += ["", "`interface_tests` are CPU-style interface tests and are NOT evidence of FluidGym compatibility.", "Only the CUDA stages (cuda_pytests, phase0, integration, tdmpc2_smoke) count as evidence; 'skipped'/'not_run' is not a pass."]
(run / "SUMMARY.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines)); sys.exit(0 if overall == "passed" else 1)
