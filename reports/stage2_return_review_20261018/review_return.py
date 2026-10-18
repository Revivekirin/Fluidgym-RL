import pathlib,json,hashlib,numpy as np
R=pathlib.Path('/Users/jihyekim/Desktop/Fluidgym-RL');P=R/'reports/stage2_frozen_20261016T131654Z_c1efaf57';O=R/'reports/stage2_return_review_20261018';O.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
manifest=read(P/'output_hashes.json');bad=[k for k,h in manifest.items() if sha(P/k)!=h];assert not bad,bad
assert read(P/'package_hashes.json')==read(R/'reports/stage2_execution_20261016/package_hashes.json')
preserve=read(P/'post_preservation.json');assert all(x['actual']==x['expected'] for x in preserve['checks'].values())
A=P/'e0_p0_S0';B=P/'e0_p0_S2';trace=[]
for t in range(3):
 row={'t':t};aa,bb=read(A/f'planner_{t:02d}.json'),read(B/f'planner_{t:02d}.json');row['rng_before_equal']=aa['rng_before']==bb['rng_before'];row['rng_after_equal']=aa['rng_after']==bb['rng_after']
 with np.load(A/f'planner_{t:02d}.npz') as x,np.load(B/f'planner_{t:02d}.npz') as y:
  row['planner_max_abs']={k:float(np.max(np.abs(x[k]-y[k]))) if x[k].size else 0. for k in x.files}
  row['requested_actions']=[x['requested_action'].tolist(),y['requested_action'].tolist()]
 for i in [t,t+1]:
  with np.load(A/f'obs_{i:02d}.npz') as x,np.load(B/f'obs_{i:02d}.npz') as y:row[f'obs{i}_max_abs']={k:float(np.max(np.abs(x[k]-y[k]))) for k in x.files}
 trace.append(row)
with np.load(A/'state_03.npz') as x,np.load(B/'state_03.npz') as y:
 a,b=x['b4_pressure'],y['b4_pressure'];d=(b-a).astype(np.float64);allowed=1e-5+1e-5*np.abs(b);fail=np.abs(d)>allowed;idx=np.unravel_index(np.argmax(np.abs(d)),d.shape)
 pressure={'shape':list(a.shape),'elements':a.size,'exceeding_elements':int(fail.sum()),'max_abs':float(np.abs(d).max()),'rms':float(np.sqrt(np.mean(d*d))),'mean_signed':float(d.mean()),'std_difference':float(d.std()),'max_abs_index':list(map(int,idx)),'at_max_S0':float(a[idx]),'at_max_S2':float(b[idx]),'allowed_at_max':float(allowed[idx]),'max_error_to_tolerance':float(np.max(np.abs(d)/allowed)),'mean_centered_difference_max':float(np.max(np.abs(d-d.mean())))}
counts=0;solver=0;reset=0;accepted=0;all_weights=True;metrics={}
for run in sorted(P.glob('e*')):
 if not run.is_dir():continue
 records=read(run/'steps.json');counts+=len(records);w=read(run/'weights_final.json');assert w['expected']==w['actual']
 for r in records:
  assert r['native_solver_success'] and r['outer_calls']==25
  assert r['gain']==(.5 if run.name.endswith('S1') else 1.)
  meta=read(run/'initialization.json');expected=meta['effective']['reward_reference']-r['drag']-abs(r['lift']);assert np.isclose(r['reward'],expected,atol=1e-5,rtol=1e-5)
  events=read(run/f"actuator_{r['t']:02d}.json");assert len(events)==25
  with np.load(run/'profiles.npz') as profiles:
   for e in events:
    assert e['native_success'] and e['native_fields_finite'];u=np.asarray(e['smoothed_memory_measured'],np.float32)
    for side in ('top','bottom'):assert np.allclose(e['boundary'][side]['velocity_measured'],profiles[side]*(u*r['gain']),atol=1e-7,rtol=1e-6)
 events=[json.loads(l) for l in (run/'solver.jsonl').read_text().splitlines()];success=[e for e in events if e['status']=='success'];assert not any(e['status']=='failed' for e in events);solver+=len(success);reset+=sum(e['phase']=='reset' for e in success);accepted+=sum(e['clock_after']['solver_steps']-e['clock_before']['solver_steps'] for e in success)
 metrics[run.name]={'controls':len(records),'return_or_partial_sum':sum(r['reward'] for r in records),'domain':meta['identity'],'weights_unchanged':True}
cost=read(P/'cost.json');assert counts==cost['completed_controls']==cost['attempted_controls']==163;assert solver==4138 and reset==63 and accepted==4138
v={'classification':'VERIFIED contract-driven early stop; Stage 2 incomplete','files_hash_verified':len(manifest),'package_identity_match':True,'protected_checks_passed':len(preserve['checks']),'cost':cost,'launcher_seconds':read(P/'launcher_status.json')['wall_seconds'],'failed_field':pressure,'early_prefix':trace,'runs':metrics,'paired_Delta_post_available':False,'S2_reduced_gain_controls':0,'additional_CFD_controls':0,'UNRESOLVED':['cause of initial numerical differences','whether acceptable closed-loop prefix reproducibility can be achieved under unchanged implementation','S2 post-shift effect','model mismatch/history need']}
(O/'verification.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
