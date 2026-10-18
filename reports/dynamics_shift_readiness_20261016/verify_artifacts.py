from pathlib import Path
import json,csv,hashlib
R=Path('/Users/jihyekim/Desktop/Fluidgym-RL'); O=R/'reports/dynamics_shift_readiness_20261016'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def local(s): return R/s.split('/Fluidgym-RL/',1)[-1]
checks=[]
for row in csv.DictReader((R/'reports/nominal_implementation_audit_20261013/checkpoint_audit.csv').open()):
 p=local(row['path']); checks.append(dict(path=str(p),sha256=sha(p),matches=sha(p)==row['sha256'],step=row['step']))
evals=[]
for p in sorted((R/'results/nominal_comparison').rglob('evaluation_manifest.json')):
 m=read(p); s=read(p.parent/'summary.json'); rows=list(csv.DictReader((p.parent/'episodes.csv').open()))
 evals.append(dict(path=str(p),hashes_match=all(sha(p.parent/k)==v for k,v in m['result_hashes'].items()),checkpoint_match=not m.get('checkpoint') or sha(local(m['checkpoint']))==m['checkpoint_sha256'],reward_error=abs(sum(float(x['mean_reward']) for x in rows)/len(rows)-s['mean_reward_per_step']['mean']),cohort=m['cohort']))
selection=[]
for p in (R/'results/nominal_comparison').rglob('selection.json'):
 m=read(p); cs=[]
 for c in m['candidates']:
  q=local(c['path']); a=read(q/'evaluation_manifest.json'); b=read(q/'summary.json'); assert a['contract']['split']=='val'; assert sha(q/'evaluation_manifest.json')==c['manifest_sha256']; cs.append((a,b))
 w=min(cs,key=lambda x:(-x[1]['mean_reward_per_step']['mean'],x[0]['environment_steps'],x[0]['checkpoint_sha256']))[0]
 selection.append(dict(path=str(p),candidates=len(cs),winner_matches=w['checkpoint_sha256']==m['checkpoint_sha256']))
diags=[]
for p in (R/'results/nominal_diagnostics').rglob('diagnostic_manifest.json'):
 m=read(p); s=read(p.parent/'summary.json'); diags.append(dict(path=str(p),hashes_match=all(sha(p.parent/k)==v for k,v in m['result_hashes'].items()),checkpoint_match=sha(local(m['identity']['checkpoint']))==m['checkpoint_sha256'],mode=m['policy_mode'],reward=s['mean_reward_per_step']['mean']))
m=read(next((R/'results/nominal_comparison').glob('*/tdmpc/test_best/evaluation_manifest.json')))
P=Path('/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym/src/fluidgym')
drift=[dict(path=k,rl_installed_hash=v,physics_checkout_hash=sha(P/k) if (P/k).exists() else None) for k,v in m['physics']['installed_source_hashes'].items() if not (P/k).exists() or sha(P/k)!=v]
result=dict(checkpoints=checks,evaluations=evals,selection=selection,diagnostics=diags,rl_installed_vs_physics_pinned_source_drift=drift)
(O/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(checkpoints=len(checks),checkpoint_hashes_match=all(x['matches'] for x in checks),evaluations=len(evals),evaluation_hashes_match=all(x['hashes_match'] and x['checkpoint_match'] for x in evals),max_reward_error=max(x['reward_error'] for x in evals),cohorts=len(set(x['cohort'] for x in evals)),selection=selection,diagnostics=diags,source_differences=len(drift)),indent=2))
