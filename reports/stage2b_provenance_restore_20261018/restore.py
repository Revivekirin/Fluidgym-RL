"""Restore verified missing metadata only. Never overwrite or execute CFD."""
import argparse,hashlib,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
here=pathlib.Path(__file__).resolve().parent
m=json.loads((here/'manifest.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for k,v in m['required_artifacts'].items():
 assert sha(here/'payload'/k)==v, 'bundle mismatch: '+k
rows=[]
for group in ('required_artifacts','checkpoints'):
 for k,v in m[group].items():
  f=a.repo/k
  state='MATCH' if f.is_file() and sha(f)==v else 'CONFLICT' if f.exists() or f.is_symlink() else 'MISSING'
  rows.append({'group':group,'path':k,'status':state})
conflicts=[r for r in rows if r['status']=='CONFLICT']
if a.apply and conflicts:
 print(json.dumps({'status':'STOP: existing conflicting files; nothing written','files':conflicts},indent=2));raise SystemExit(1)
for r in rows:
 if a.apply and r['group']=='required_artifacts' and r['status']=='MISSING':
  f=a.repo/r['path'];f.parent.mkdir(parents=True,exist_ok=True)
  with f.open('xb') as out:out.write((here/'payload'/r['path']).read_bytes())
  assert sha(f)==m['required_artifacts'][r['path']]
  r['status']='RESTORED'
print(json.dumps({'mode':'restore_missing_metadata' if a.apply else 'read_only_audit','counts':{s:sum(r['status']==s for r in rows) for s in ('MATCH','MISSING','CONFLICT','RESTORED')},'nonmatching_files':[r for r in rows if r['status']!='MATCH'],'CFD':0,'checkpoint_inference':0,'experiment_retry':False},indent=2))
