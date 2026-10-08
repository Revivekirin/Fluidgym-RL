#!/usr/bin/env python
"""H: consolidated report for one remote_milestone.sh run. Only states what stage files/results contain; skipped/not_run are never passes."""
import json, sys, xml.etree.ElementTree as ET
from pathlib import Path

run = Path(sys.argv[1]); res_root = run.parent.parent
start = (run / ".start").stat().st_mtime
rc = dict(l.split() for l in (run / "stage_rc.txt").read_text().splitlines() if l.strip())
skipped = dict(l.split(maxsplit=1) for l in (run / "stage_skipped.txt").read_text().splitlines() if l.strip())
rows, lines = [], []
def newer(sub):   # result dirs created during this run
    d = res_root / sub
    return sorted([p for p in d.iterdir() if p.is_dir() and p.stat().st_mtime >= start], key=lambda p: p.stat().st_mtime) if d.exists() else []
def jl(p): return json.loads(Path(p).read_text())
def add(name, status, detail=""): rows.append((name, status, detail))

for n in sorted(set(rc) | set(skipped)):
    if n in skipped: add(n, "skipped", skipped[n]); continue
    if n == "D0_phase0_status":
        if "D1_phase0_status" in rc:
            add(n, "superseded", f"initial rc={rc[n]}; final evidence: D1_phase0_status"); continue
        add(n, {"0": "passed", "4": "not_run (no evidence) -> validation was run"}.get(rc[n], "failed/invalid")); continue
    add(n, "passed" if rc[n] == "0" else f"failed (rc={rc[n]})")
x = run / "cuda_tests.xml"
if x.exists():
    r = ET.parse(x).getroot(); ts = r if r.tag == "testsuite" else r.find("testsuite"); n_, f_, e_, k_ = (int(ts.get(t, 0)) for t in ("tests", "failures", "errors", "skipped"))
    rows = [(n, ("failed (0 CUDA tests passed; skipped != passed)" if n == "CUDA_tests" and n_ - f_ - e_ - k_ == 0 else s), d) if n == "CUDA_tests" else (n, s, d) for n, s, d in rows]

lines += ["# Milestone report", f"run: `{run}`", "", "| stage | status | detail |", "|---|---|---|", *[f"| {a} | {b} | {c} |" for a, b, c in rows], ""]
for d in newer("official_sac_eval"):
    if (d / "summary.json").exists():
        s, m = jl(d / "summary.json"), jl(d / "run_manifest.json")
        lines += [f"## Official SAC: `{d.name}`", f"- checkpoint: {m['checkpoint'].get('repo_id')}/{m['checkpoint'].get('filename')} @ {m['checkpoint'].get('revision_sha')} (sha256 {m['checkpoint_sha256'][:12]}…)",
                  f"- trained with fluidgym {m.get('checkpoint_trained_with_fluidgym')}, evaluated with {m['simulator_version']}; inference: {m['inference_mode']}; contract `{m['contract_digest']}` ({m['contract']['n_episodes']} episodes, {m['contract']['protocol']})",
                  f"- mean reward/step: {s['mean_reward_per_step']['mean']:.4f} (episode std {s['mean_reward_per_step']['std']:.4f}); metrics: " + ", ".join(f"{k}={v['mean']:.4f}" for k, v in s['metrics_per_step_mean'].items())]
        for cmp in s.get("comparison_to_published", []):
            for r in cmp["rows"]: lines.append(f"- vs {cmp['reference']}: {r['metric']} measured {r['measured']:.4f} / published {r['published']} (diff {r['abs_diff']:+.4f}); no reproduction claim made")
        lines.append("")
for d in newer("tdmpc2_smoke"):
    if (d / "report.json").exists():
        r = jl(d / "report.json"); lines += [f"## TD-MPC2 smoke: `{d.name}` -> {r['status']}"] + [f"- {k}: {v['status']}" for k, v in r["checks"].items()] + [""]
        for k, v in r["checks"].items():
            if v.get("error"):
                lines += [f"### {k} diagnostic", "", "```text", v.get("traceback") or v["error"], "```", ""]
evs = newer("eval")
for d in evs:
    if (d / "summary.json").exists():
        s, m = jl(d / "summary.json"), jl(d / "run_manifest.json")
        lines.append(f"## Eval `{d.name}`: {m['algorithm']}/{m['inference_mode']} mean reward/step {s['mean_reward_per_step']['mean']:.4f} over {s['n_episodes']} episodes (contract `{m['contract_digest']}`)")
if evs:
    lines.append("")
sac = [d for d in newer("official_sac_eval") if (d / "run_manifest.json").exists()]
if sac and evs:
    from fluidgym_rl.evalcontract import classify_comparison
    for d in evs:
        if (d / "run_manifest.json").exists():
            lines.append(f"- comparability SAC `{sac[0].name}` vs `{d.name}`: **{classify_comparison(jl(sac[0] / 'run_manifest.json'), jl(d / 'run_manifest.json'))['class']}**")
overall = "PASSED" if rows and all(s in ("passed", "superseded") for _, s, _ in rows) else "NOT PASSED"
lines = [f"**Overall: {overall}** (only executed-and-passed stages count)", ""] + lines
(run / "MILESTONE_REPORT.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines)); sys.exit(0 if overall == "PASSED" else 1)
