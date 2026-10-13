#!/usr/bin/env python
"""Create a new archive with source-file hashes; never modify run evidence."""
import argparse,json,tarfile,io,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fluidgym_rl.comparison.schema import file_hash,verify_result
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--evaluations',nargs='*',type=Path,default=[])
p.add_argument('--out',type=Path,required=True)
a=p.parse_args(); root=a.run.resolve(); dest=a.out.resolve()
if json.loads((root/'summary.json').read_text()).get('status')!='completed': raise ValueError('training run incomplete')
if dest.is_relative_to(root): raise ValueError('archive must be outside the source run')
for ev in a.evaluations: verify_result(ev)
files={}
for label,path in [('training',root),*[(f'evaluation_{i}',v.resolve()) for i,v in enumerate(a.evaluations)]]:
    for f in sorted(path.rglob('*')):
        if f.is_file() and not f.is_symlink(): files[f'{label}/{f.relative_to(path)}']=f
hashes={k:file_hash(v) for k,v in files.items()}
with dest.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:gz') as archive:
    for name,f in files.items(): archive.add(f,arcname=name,recursive=False)
    payload=json.dumps({'algorithm':'sac','file_sha256':hashes,'independent_test_included':any(verify_result(v)[0]['contract']['split']=='test' for v in a.evaluations)},indent=2).encode()
    info=tarfile.TarInfo('PACKAGE_MANIFEST.json'); info.size=len(payload); archive.addfile(info,io.BytesIO(payload))
print(dest, file_hash(dest))
