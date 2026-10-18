from pathlib import Path
import json,csv,hashlib,numpy as np
R=Path.cwd();O=R/'reports/p1_return_review_20261016';D=O/'p1_run_20261016T072205Z_3a71b74d';E=json.loads((R/'reports/p1_execution_20261016/expected.json').read_text())
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(p.open()))
checks=[]
def check(name,ok):
 checks.append({'check':name,'passed':bool(ok)})
 if not ok:raise AssertionError(name)
check('execution script hash',read(D/'execution_request.json')['script_sha256']==sha(R/'reports/p1_execution_20261016/run_p1.py'))
check('expectation hash',read(D/'execution_request.json')['expected_sha256']==sha(R/'reports/p1_execution_20261016/expected.json'))
for stage in ['B','A_tdmpc','A_tdmpc2','C_tdmpc','C_tdmpc2']:
 p=read(D/stage/'provenance.json');check(stage+' sources',p['source_hashes']==E['sources']);check(stage+' backend',p['backend']['sha256']==E['backend_hash']);check(stage+' checkpoint hashes',p['checkpoint_hashes']==E['checkpoints']);check(stage+' versions',p['versions']==E['versions'])
B={};all_b=[]
for kind in ['zero','bounded']:
 aa=np.load(D/'B'/f'{kind}_0_observations.npz');bb=np.load(D/'B'/f'{kind}_1_observations.npz');ar=read(D/'B'/f'{kind}_0_records.json');br=read(D/'B'/f'{kind}_1_records.json')
 d={k:float(np.max(np.abs(aa[k]-bb[k]))) for k in aa.files}
 for k in aa.files:check(kind+' '+k+' repeat',np.allclose(aa[k],bb[k],atol=1e-5,rtol=1e-5))
 for a in [aa,bb]:check(kind+' native flatten',np.array_equal(a['flat'],a['velocity'].reshape(5,302)))
 check(kind+' identity',ar[0]['identity']==br[0]['identity'])
 scalars={k:max(abs(x[k]-y[k]) for x,y in zip(ar[1:],br[1:])) for k in ['reward','drag','lift','solver_time','solver_steps','control_steps']}
 for rec in [ar,br]:
  for x in rec[1:]:
   check('reward formula',np.isclose(x['reward'],rec[0]['effective']['reward_reference']-x['drag']-abs(x['lift']),atol=1e-5,rtol=1e-5))
   check('actuator',np.allclose(x['runtime_applied_arguments'],x['expected_applied_source_derived'],atol=1e-7,rtol=1e-6))
   check('memory',np.array_equal(x['runtime_actuator_memory'],x['runtime_applied_arguments'][-1]))
 B[kind]={'array_max_abs':d,'scalar_max_abs':scalars,'identity':ar[0]['identity'],'clock_reset':ar[0]['solver_time'],'clock_end':ar[-1]['solver_time'],'applied_end':[x['runtime_actuator_memory'] for x in ar[1:]]}
A={}
for alg in ['tdmpc','tdmpc2']:
 root=D/('A_'+alg);calls=read(root/'calls.json');contrasts=[]
 for i in range(2):
  z=np.load(root/f'model_outputs_obs{i}.npz');check(alg+' model finite',all(np.isfinite(z[k]).all() for k in z.files))
  by={x['label'].split('_',1)[1]:x for x in calls if x['label'].startswith(f'obs{i}_')}
  for a,b,same_rng,same_mem in [('reset_same_rng_1','reset_same_rng_2',True,True),('reset_same_rng_1','changed_rng_reset_state',False,True),('warm_a_same_rng_1','warm_a_same_rng_2',True,True),('warm_a_same_rng_1','warm_a_changed_rng',False,True),('warm_a_same_rng_1','warm_b_same_rng',True,False)]:
   x,y=by[a],by[b];eqrng=x['rng_before']==y['rng_before'];eqmem=x['warm_before']==y['warm_before'];check(alg+' rng control '+a+'/'+b,eqrng==same_rng);check(alg+' memory control '+a+'/'+b,eqmem==same_mem)
   delta=float(np.max(np.abs(np.array(x['action'])-np.array(y['action']))));contrasts.append({'obs':i,'pair':a+'/'+b,'rng_equal':eqrng,'memory_equal':eqmem,'action_max_abs':delta})
   if same_rng and same_mem:check(alg+' controlled action exact',delta==0)
 A[alg]=contrasts
C={};table=[]
for alg in ['tdmpc','tdmpc2']:
 root=D/('C_'+alg);rr=rows(root/'episodes.csv');hist=rows(R/E['policies'][alg]['historical']/'episodes.csv');init=read(root/'initializations.json');summary=read(root/'summary.json');check(alg+' episode count',len(rr)==len(hist)==len(init)==10)
 check(alg+' model hash A C',read(D/('A_'+alg)/'status.json')['frozen_model_hash']==read(root/'run_manifest.json')['frozen_model_hash'])
 vals={}
 for metric in ['mean_reward','return','mean_drag','mean_lift','mean_abs_lift']:
  v=np.array([float(x[metric]) for x in rr]);h=np.array([float(x[metric]) for x in hist]);diff=v-h
  vals[metric]={'historical_mean':float(h.mean()),'current_mean':float(v.mean()),'mean_delta':float(diff.mean()),'max_abs_episode_delta':float(np.abs(diff).max()),'min_delta':float(diff.min()),'max_delta':float(diff.max())}
  for i in range(10):table.append({'algorithm':alg,'episode':i,'domain_index':init[i]['initial_domain']['index'],'metric':metric,'historical':h[i],'current':v[i],'difference':diff[i]})
 for i,x in enumerate(rr):
  check(alg+' boundary',x['length']=='80' and x['terminated']=='False' and x['truncated']=='True');check(alg+' seed',json.loads(x['reset_kwargs'])=={'seed':i,'randomize':True});check(alg+' return consistency',np.isclose(float(x['return']),80*float(x['mean_reward'])));check(alg+' matching init',init[i]['reset_kwargs']==json.loads(x['reset_kwargs']))
 check(alg+' summary recompute',np.isclose(vals['mean_reward']['current_mean'],summary['mean_reward_per_step']['mean'],rtol=0,atol=1e-14))
 C[alg]={'metrics':vals,'domain_indices':[x['initial_domain']['index'] for x in init],'wall_seconds':read(root/'status.json')['wall_seconds']}
check('C paired actual ICs',C['tdmpc']['domain_indices']==C['tdmpc2']['domain_indices'])
check('preservation',read(D/'preservation.json')['status']=='passed')
result={'archive_sha256':sha(R/'reports/p1_run_20261016T072205Z_3a71b74d.tar.gz'),'checks':checks,'B':B,'A':A,'C':C,'cost':read(D/'execution_cost.json'),'scope':'offline recheck of returned current-pinned-environment P1; no new CFD','all_passed':True}
(O/'verified_results.json').write_text(json.dumps(result,indent=2))
with (O/'per_episode_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
print(json.dumps({'checks':len(checks),'B':B,'C':C,'wall':result['cost']},indent=2))
