#!/usr/bin/env python3
"""Package explicitly selected small experiment evidence, never model/replay binaries."""
import argparse
from pathlib import Path
import tarfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('runs',nargs='+',type=Path)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args()
roots=[r.resolve(strict=True) for r in a.runs]
if not all(r.is_dir() for r in roots):
    p.error('all inputs must be result directories')
if a.output.exists():
    p.error('output exists; choose a new archive name')
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('xb') as raw:
    with tarfile.open(fileobj=raw,mode='w:gz') as tar:
        for i,root in enumerate(roots):
            for f in sorted(root.rglob('*')):
                if f.is_file() and not f.is_symlink() and f.suffix in {'.csv','.json','.yaml','.log','.txt'}:
                    if f.stat().st_size > 10*1024*1024:
                        print('Skipped >10MiB:',f)
                        continue
                    tar.add(f,arcname=str(Path(f'{i}_{root.name}')/f.relative_to(root)),recursive=False)
print(a.output)
