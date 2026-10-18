"""Saved-evidence reporting only; no policy or simulator imports."""
import json
from pathlib import Path
from statistics import mean,pstdev

def read(p,default):return json.loads(p.read_text()) if p.exists() else default

def report(out,status):
    episodes=read(out/'episodes.json',[]);cost=read(out/'cost.json',{}); comparisons=[]
    lookup={e['run']:e for e in episodes}
    for es in (0,1):
        for ps in (0,1):
            base=lookup.get(f'e{es}_p{ps}_S0')
            if base is None:continue
            for condition in ('S1','S2'):
                other=lookup.get(f'e{es}_p{ps}_{condition}')
                if other is None:continue
                comparisons.append({'environment_seed':es,'planner_seed':ps,'comparison':condition+'-S0',
                    'differences':{segment:{k:v-base['metrics'][segment][k] for k,v in other['metrics'][segment].items() if k!='controls'} for segment in ('whole','pre40','post40')}})
    variation={}
    for condition in ('S0','S1','S2'):
        selected=[e for e in episodes if e['condition']==condition]
        variation[condition]={'completed_episodes':len(selected)}
        for segment in ('whole','pre40','post40'):
            vals=[e['metrics'][segment]['reward_sum'] for e in selected]
            variation[condition][segment]={'n':len(vals),'mean':mean(vals) if vals else None,'population_sd_descriptive':pstdev(vals) if vals else None,'min':min(vals) if vals else None,'max':max(vals) if vals else None}
    diagnostics=[]
    for p in sorted(out.glob('*/prefix_differences.json')):
        rows=read(p,[])
        diagnostics.append({'run':p.parent.name,'comparisons':len(rows),'numeric_disagreements':sum(r.get('allclose') is False for r in rows),'identity_disagreements':sum(r.get('exact_identity') is False for r in rows),'first_disagreement':next((r for r in rows if r.get('allclose') is False or r.get('exact_identity') is False),None)})
    attempts=[]
    for d in sorted(out.glob('e*_p*_S*')):
        rows=read(d/'steps.json',[])
        attempts.append({'run':d.name,'recorded_controls':len(rows),'completed':d.name in lookup,'last_t':rows[-1]['t'] if rows else None})
    result={'classification':'VERIFIED recorded results; exploratory only','launcher_status':status,'comparisons':comparisons,'variation':variation,'attempts':attempts,'prefix_diagnostics':diagnostics,'missing_episodes':[f'e{e}_p{p}_{c}' for e in (0,1) for p in (0,1) for c in ('S0','S1','S2') if f'e{e}_p{p}_{c}' not in lookup]}
    (out/'outcome_analysis.json').write_text(json.dumps(result,indent=2))
    text='# Stage 2B frozen-control results\n\n'
    text+='Execution status: **'+status['status']+'**. Original Stage 2 remains failed; this report cannot reclassify it.\n\nVERIFIED: recorded completed episodes '+str(len(episodes))+'/12. Missing or failed trajectories are not replaced. Exact launcher/worker status:\n```json\n'+json.dumps(status,indent=2)+'\n```\n\n'
    text+='| Run | Physical domain | Full return | Pre40 | Post40 | Mean drag | Mean absolute lift | Saturation |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for e in episodes:
        w=e['metrics']['whole'];text+=f"| {e['run']} | {e['identity']['index']} | {w['reward_sum']:.9g} | {e['metrics']['pre40']['reward_sum']:.9g} | {e['metrics']['post40']['reward_sum']:.9g} | {w['mean_drag']:.9g} | {w['mean_abs_control_interval_lift']:.9g} | {w['saturation_frequency']:.6g} |\n"
    text+='\nVERIFIED paired comparisons (only complete pairs):\n\n| Pair | Comparison | Full return difference | Pre40 difference | Post40 difference |\n|---|---|---:|---:|---:|\n'
    for c in comparisons:
        d=c['differences'];text+=f"| e{c['environment_seed']}_p{c['planner_seed']} | {c['comparison']} | {d['whole']['reward_sum']:.9g} | {d['pre40']['reward_sum']:.9g} | {d['post40']['reward_sum']:.9g} |\n"
    text+='\nVariation across the fixed seed matrix (descriptive SD, no independence/significance claim):\n```json\n'+json.dumps(variation,indent=2)+'\n```\n\nNumerical prefix diagnostics (non-stopping):\n```json\n'+json.dumps(diagnostics,indent=2)+'\n```\n\nActual cost, including separate resets and accepted solver increments:\n```json\n'+json.dumps(cost,indent=2)+'\n```\n'
    text+='\nRequested/clipped action magnitudes, clipping/saturation, drag and lift changes by pre/post segment are in episodes.json and outcome_analysis.json. Measured smoothed/effective boundary actuation is in each actuator file, distinct from expected profile products. Native arrays, solver status, physical clocks, actual raw/flat observations, RNG and warm states remain in per-run logs. Prefix differences retain the original 1e-5 envelope as diagnostics.\n\nINFERRED: paired return differences characterize this fixed controller under the assigned gain schedules plus numerical variability. UNRESOLVED: isolation of learned-model error, control-authority limits, causal need for history, population robustness and physical accuracy. S2/S0 pre40 differences quantify observed baseline divergence but are not subtracted post hoc from the primary post40 estimand. S1/S0 comparisons are descriptive. Two planner seeds per environment seed are not four independent physical initializations.\n'
    (out/'STAGE2B_FROZEN_CONTROL_RESULTS.md').write_text(text)
    passed=status['status']=='passed' and len(episodes)==12
    decision='# Stage 2B research decision\n\n'
    decision+=('VERIFIED: all 12 episodes completed within the execution contract. **CONDITIONAL GO** for separately preregistered Stage 3 history-identifiability data collection.\n' if passed else '**HOLD / UNRESOLVED:** full Stage 2B execution is incomplete or a hard gate failed. Review the recorded blocker and partial evidence before authorizing further collection. No automatic replacement or retry.\n')
    decision+='\nOriginal Stage 2 strict-native failure remains FAILED. Stage 2B completion is execution feasibility, not robustness or identifiability certification.\n\nINFERRED recommendation: Stage 3 may use prescribed-action trajectories and does not require TD-MPC failure or strict native-state trajectory identity. Preregister causal history/target timing, excitation, matched or randomized physical initialization, actuator labels obtained from runtime readback, independent trajectory-level train/validation/test splits, and controls against schedule/time/initialization leakage. Compare history information against current-observation/action baselines and nominal repeat variability. Use the observed pre40 divergence and seed variation above to plan a small collection; do not infer identifiability from return degradation alone, and do not require degradation to study it.\n\nUNRESOLVED: whether history predicts hidden actuator effectiveness beyond these baselines. No estimator, dataset collection, TD-MPC2 comparison, new schedule or training is launched. Any further CFD requires separate authorization.\n'
    (out/'STAGE2B_RESEARCH_DECISION.md').write_text(decision)
