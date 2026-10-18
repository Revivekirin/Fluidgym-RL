"""Read-only runtime provenance collection. No environment construction or CFD."""
import argparse, datetime, hashlib, importlib.metadata as md, importlib.util
import json, os, pathlib, subprocess, sys, time, traceback, uuid
sys.dont_write_bytecode = True
p=argparse.ArgumentParser();p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--data-root',type=pathlib.Path,required=True);a=p.parse_args()
start=time.monotonic();root=a.repo.resolve();out=root/'reports'/('p0_p1_preflight_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def command(args):
 try:
  x=subprocess.run(args,capture_output=True,text=True,timeout=30);return dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr)
 except Exception as e:return dict(error=repr(e))
r=dict(status='collecting',python=sys.version,executable=sys.executable,cwd=str(pathlib.Path.cwd()),repo=str(root),cfd_steps=0,training_updates=0,git_head=command(['git','-C',str(root),'rev-parse','HEAD']),git_status=command(['git','-C',str(root),'status','--porcelain']),versions={},errors=[])
for name in ['fluidgym','torch','numpy','gymnasium','hoptorch','hydra-core','omegaconf','tensordict','torchrl']:
 try:r['versions'][name]=md.version(name)
 except md.PackageNotFoundError:r['versions'][name]=None
try:
 dist=md.distribution('fluidgym');r['fluidgym_distribution']={'root':str(dist.locate_file('')),'direct_url':dist.read_text('direct_url.json'),'installer':dist.read_text('INSTALLER')};(out/'fluidgym_RECORD.txt').write_text(dist.read_text('RECORD') or '')
 spec=importlib.util.find_spec('fluidgym');r['fluidgym_spec_origin']=spec.origin if spec else None
 if spec and spec.origin:
  package=pathlib.Path(spec.origin).resolve().parent;r['source_hashes']={}
  for f in sorted(package.rglob('*.py')):
   rel=f.relative_to(package);r['source_hashes'][str(rel)]=digest(f);q=out/'installed_sources'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(f.read_bytes())
  r['package_binary_hashes']={str(f):digest(f) for f in package.rglob('*.so')}
except Exception:r['errors'].append(traceback.format_exc())
try:
 import torch
 r['torch']={'version':torch.__version__,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'matmul_precision':torch.get_float32_matmul_precision(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32}
except Exception:r['errors'].append(traceback.format_exc())
r['import_probe']=command([sys.executable,'-B','-c',"import json,fluidgym; from fluidgym.config import config; print(json.dumps({'fluidgym_file':fluidgym.__file__,'initial_domains_path':str(config.initial_domains_path)}))"])
r['gpu']=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,memory.used','--format=csv,noheader'])
r['checkpoints']={str(f.relative_to(root)):digest(f) for f in sorted((root/'results').rglob('*.pt'))}
r['existing_evaluation_files']={str(f.relative_to(root)):digest(f) for dirname in ['nominal_comparison','nominal_diagnostics'] for f in (root/'results'/dirname).rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.yaml')}
domain=a.data_root.resolve()/'cylinder_2D_Re100_Res24';r['requested_domain_root']=str(domain);r['domain_files']={str(f.relative_to(domain)):digest(f) for f in sorted(domain.rglob('*')) if f.is_file()} if domain.is_dir() else {};r['data_available']=bool(r['domain_files'])
r['elapsed_seconds']=time.monotonic()-start;r['status']='collected_with_errors' if r['errors'] or r['import_probe'].get('returncode')!=0 or not r['data_available'] else 'collected';(out/'verification.json').write_text(json.dumps(r,indent=2));print(out);print('CFD steps: 0; status:',r['status'])
import tarfile
archive=out.with_suffix('.tar.gz')
with tarfile.open(archive,'x:gz') as tar:
 tar.add(out,arcname=out.name)
print('RETURN_ARCHIVE:',archive)
