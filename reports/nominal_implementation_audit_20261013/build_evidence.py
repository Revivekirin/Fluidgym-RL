"""Read-only evidence extraction; run from repository root. Never runs simulation."""
import csv,hashlib,json,os,shutil,subprocess,tempfile
from pathlib import Path
import numpy as np
import torch
from fluidgym_rl.comparison.schema import verify_result,file_hash
from fluidgym_rl.comparison.evaluate import training_identity
ROOT=Path.cwd(); OUT=Path(__file__).resolve().parent
T1=ROOT/'results/tdmpc_full/20261012T060719Z_full'; T2=ROOT/'results/tdmpc2_full/20261008T124249Z_4be46ae7'; EV=ROOT/'results/nominal_comparison/20261013T061228Z'
def dump(name,x): (OUT/name).write_text(json.dumps(x,indent=2,default=str))
def table(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read(p): return json.loads(p.read_text())
sources={'tdmpc':Path('/tmp/fgrl-original-tdmpc'),'tdmpc2':Path('/tmp/fgrl-audit-tdmpc2'),'fluidgym':Path('/tmp/fgrl-original-fluidgym')}
source_info={}
for name,p in sources.items():
 sha=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
 dirty=subprocess.check_output(['git','-C',str(p),'status','--porcelain'],text=True).strip()
 source_info[name]={'path':str(p),'sha':sha,'dirty':dirty}
 for f in p.rglob('*'):
  if f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts and f.suffix in ('.py','.yaml'):
   target=OUT/'source_snapshots'/name/f.relative_to(p);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
dump('upstream_sources.json',source_info)
hashes={str(p):file_hash(p) for root in (T1,T2,EV) for p in root.rglob('*') if p.is_file()}
dump('input_hashes.json',hashes)
inventory=[]
for p,h in hashes.items():
 path=Path(p); alg='tdmpc2' if path.is_relative_to(T2) else 'tdmpc' if path.is_relative_to(T1) else 'shared evaluation'
 role='W&B offline history/artifact' if 'wandb' in path.parts else 'checkpoint' if path.suffix=='.pt' else 'run/evaluation artifact'
 inventory.append(dict(path=p,role=role,source_revision=source_info.get(alg,{}).get('sha','see evaluation manifest'),checkpoint_step='',format=path.suffix,verification='SHA256 recorded; content checks separately',sha256=h))
for p in [*ROOT.glob('scripts/*tdmpc*.py'),*ROOT.glob('src/fluidgym_rl/tdmpc/*.py'),*ROOT.glob('src/fluidgym_rl/*tdmpc*.py'),*ROOT.glob('patches/*'),*ROOT.glob('tests/test*tdmpc*.py'),ROOT/'src/fluidgym_rl/envs.py',ROOT/'src/fluidgym_rl/evalcontract.py']:
 if p.is_file(): inventory.append(dict(path=str(p),role='current source/patch/test',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),checkpoint_step='',format=p.suffix,verification='current source; training hash comparison separately',sha256=file_hash(p)))
cks=[]; states={}
for alg,root in [('tdmpc',T1),('tdmpc2',T2)]:
 for ck in sorted(root.rglob('*.pt')):
  identity=training_identity(alg,root,ck); state=torch.load(ck,map_location='cpu',weights_only=True)
  tensors={f'{g}.{k}':v for g,d in state.items() for k,v in d.items() if isinstance(v,torch.Tensor)}
  states[(alg,ck.name)]=tensors
  cks.append(dict(algorithm=alg,path=str(ck),step=identity['environment_steps'],sha256=identity['checkpoint_sha256'],format='torch.save model+model_target' if alg=='tdmpc' else 'torch.save model',tensor_count=len(tensors),finite=all(bool(torch.isfinite(v).all()) for v in tensors.values()),restore_evidence='remote fresh-process common evaluation completed; local CPU tensor inspection',normalization='none obs; TDMP2 RunningScale training-only not saved',source_revision=source_info[alg]['sha']))
  for row in inventory:
   if row['path']==str(ck): row.update(checkpoint_step=identity['environment_steps'],verification='hash + CPU weights_only deserialize + tensor finite + remote evaluation manifest')
table('artifact_inventory.csv',inventory);table('checkpoint_audit.csv',cks)
a=states['tdmpc','best.pt'];b=states['tdmpc','final.pt'];dump('tdmpc_best_final_weights.json',dict(all_tensors_bitwise_equal=all(torch.equal(v,b[k]) for k,v in a.items()),tensor_count=len(a),file_hashes_different=True,note='Different archive bytes do not imply different model parameters; planner_std sidecars also checked below',planner_stds=[read(T1/'checkpoints'/f'{n}.json')['planner_std'] for n in ('best','final')]))
verifications=[]; evaluation_rows=[]
ref=read(EV/'reference/summary.json')['metrics_per_step_mean']['drag']['mean']
for p in sorted(EV.rglob('evaluation_manifest.json')):
 m,s=verify_result(p.parent)
 records=list(csv.DictReader((p.parent/'episodes.csv').open()))
 def mean(key): return float(np.mean([float(r[key]) for r in records]))
 errors={}
 for field,col in [('mean_reward_per_step','mean_reward'),('episode_return','return'),('episode_length','length')]: errors[field]=abs(mean(col)-s[field]['mean'])
 for field in ('drag','lift','abs_lift'): errors[field]=abs(mean('mean_'+field)-s['metrics_per_step_mean'][field]['mean'])
 errors['wall']=abs(sum(float(r['wall_seconds']) for r in records)-s['eval_wall_seconds'])
 for r in records: assert abs(float(r['return'])/int(r['length'])-float(r['mean_reward']))<1e-12
 if m['algorithm']!='uncontrolled': errors['drag_reduction']=abs(100*(1-mean('mean_drag')/ref)-s['drag_reduction_percent'])
 assert max(errors.values())<1e-10,(p,errors)
 if m['checkpoint']:
  local=ROOT/Path(m['checkpoint']).relative_to('/nfs4/jhkim/repos/Fluidgym-RL')
  assert file_hash(local)==m['checkpoint_sha256']
 verifications.append(dict(path=str(p),cohort=m['cohort'],contract=m['contract'],errors=errors,hash_status='passed'))
 evaluation_rows.append(dict(algorithm=m['algorithm'],split=m['contract']['split'],role=m['checkpoint_role'],step=m['environment_steps'],checkpoint=m.get('checkpoint'),reward=mean('mean_reward'),reward_std=float(np.std([float(r['mean_reward']) for r in records],ddof=1)),drag=mean('mean_drag'),abs_lift=mean('mean_abs_lift'),action_mean=mean('action_mean'),action_std=mean('action_std'),wall_seconds=s['eval_wall_seconds'],path=str(p.parent)))
table('evaluation_recomputed.csv',evaluation_rows);dump('evaluation_verification.json',verifications)
selection=[]
for alg in ('tdmpc','tdmpc2'):
 s=read(EV/alg/'selection.json');candidates=[]
 for c in s['candidates']:
  p=EV/alg/Path(c['path']).name;assert file_hash(p/'evaluation_manifest.json')==c['manifest_sha256'];m,v=verify_result(p); assert m['contract']['split']=='val'; candidates.append((v['mean_reward_per_step']['mean'],m))
 winner=min(candidates,key=lambda x:(-x[0],x[1]['environment_steps'],x[1]['checkpoint_sha256']))
 assert winner[1]['checkpoint_sha256']==s['checkpoint_sha256'] and winner[0]==s['score']
 for label in ('test_best','test_final'):
  m=read(EV/alg/label/'evaluation_manifest.json')
  if label=='test_best': assert m['checkpoint_sha256']==s['checkpoint_sha256'] and m['selection_sha256']==file_hash(EV/alg/'selection.json')
  assert m['contract']['split']=='test'
 selection.append(dict(algorithm=alg,status='passed',selected_checkpoint=s['checkpoint'],score=s['score']))
dump('selection_verification.json',selection)
# Historical metrics: never mix the eval split with common validation.
tr1=list(csv.DictReader((T1/'train.csv').open())); tr2=[json.loads(x) for x in (T2/'metrics.jsonl').read_text().splitlines()]
windows=[]
for alg,rows in [('tdmpc',tr1),('tdmpc2',[x for x in tr2 if x['category']=='train'])]:
 keys=['episode_return','mean_reward_per_step','episode_reward','consistency_loss','reward_loss','value_loss','pi_loss','total_loss','weighted_loss','grad_norm','pi_grad_norm','pi_entropy','pi_scaled_entropy','pi_scale']
 for lo,hi in [(0,4000),(40000,44000),(44000,48000),(48000,50000)]:
  subset=[x for x in rows if lo<int(x['environment_steps'])<=hi]
  for k in keys:
   vals=[float(x[k]) for x in subset if x.get(k) not in (None,'')]
   if vals: windows.append(dict(algorithm=alg,interval=f'({lo},{hi}]',metric=k,count=len(vals),mean=np.mean(vals),median=np.median(vals),min=min(vals),max=max(vals),reduction='over logged latest-update scalars, not all optimizer updates' if 'loss' in k or 'grad' in k or 'pi_' in k else 'over training episode records'))
table('training_windows.csv',windows)
table('tdmpc2_training_eval.csv',[{k:r.get(k) for k in ('environment_steps','mean_reward_per_step','drag','lift','episode_length','elapsed_time')} for r in tr2 if r['category']=='eval'])
drift=[]
for label,m in [('tdmpc_training',read(T1/'run_manifest.json')),('shared_eval',read(EV/'tdmpc2/test_best/evaluation_manifest.json'))]:
 for path,h in m.get('local_source_hashes',m.get('pipeline_source_hashes',{})).items():
  current=ROOT/path; drift.append(dict(evidence=label,path=path,recorded_sha256=h,current_sha256=file_hash(current) if current.is_file() else None,matches=current.is_file() and file_hash(current)==h))
table('source_drift.csv',drift)
# Work copies only; original upstream trees remain clean.
from fluidgym_rl.upstream import make_tdmpc2_workcopy
from fluidgym_rl.tdmpc.runtime import make_workcopy
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp);(p/'sources').mkdir();(p/'sources/tdmpc2').symlink_to(sources['tdmpc2'],target_is_directory=True)
 os.environ['FGRL_THIRD_PARTY']=str(p/'sources')
 a=make_workcopy(p/'tdmpc',source=sources['tdmpc']);b=make_tdmpc2_workcopy(p/'tdmpc2')
 subprocess.run(['git','init','-q',str(p/'tdmpc2')],check=True)
 subprocess.run(['git','apply','--check',str(ROOT/'patches/tdmpc2_logging.patch')],cwd=p/'tdmpc2',check=True)
 dump('patch_verification.json',dict(tdmpc=a,tdmpc2=b,logging_patch_check='passed',note='only disposable copies modified'))
dump('preservation.json',dict(files=len(hashes),changed=[p for p,h in hashes.items() if file_hash(p)!=h]))
print('evidence complete',len(hashes),'input files;',len(verifications),'evaluations;',len(cks),'checkpoints')
