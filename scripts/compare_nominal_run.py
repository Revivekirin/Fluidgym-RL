#!/usr/bin/env python
"""Evaluate saved candidates, freeze validation selection, then report test.

Each checkpoint is loaded in a fresh child process. No training is launched.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fluidgym_rl.comparison.schema import file_hash, write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--algorithm',choices=['tdmpc','tdmpc2'],required=True)
    ap.add_argument('--training-run',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--episodes',type=int,default=10)
    ap.add_argument('--base-seed',type=int,default=0)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--include-train',action='store_true',help='also evaluate frozen candidates on train split')
    a=ap.parse_args(); root=a.training_run.resolve(); out=a.out.resolve()
    status=root/('summary.json' if a.algorithm=='tdmpc' else 'status.json')
    if json.loads(status.read_text()).get('status')!='completed':
        raise ValueError('batch evaluation requires a completed training run')
    directory=root/('checkpoints' if a.algorithm=='tdmpc' else 'upstream/models')
    final=directory/'final.pt'
    if not final.is_file(): raise FileNotFoundError(final)
    candidates=sorted(directory.glob('*.pt'))
    from fluidgym_rl.comparison.evaluate import training_identity
    seen=set(); checkpoints=[]
    for ck in candidates:
        identity=training_identity(a.algorithm,root,ck)
        key=(identity['checkpoint_sha256'],identity['environment_steps'])
        if key not in seen:
            checkpoints.append(ck); seen.add(key)
    out.mkdir(parents=True,exist_ok=False)
    evaluations=[]
    cli=Path(__file__).with_name('compare_nominal.py')
    common=['--algorithm',a.algorithm,'--training-run',str(root),'--episodes',str(a.episodes),
            '--base-seed',str(a.base_seed),'--device',a.device,'--reference',str(a.reference.resolve())]
    def run(*args):
        subprocess.run([sys.executable,'-u',str(cli),*map(str,args)],check=True)
    def evaluate(ck,split,role,name,selection=None):
        dest=out/name
        command=['evaluate',*common,'--checkpoint',str(ck),'--split',split,'--role',role,'--out',str(dest)]
        if selection: command+=['--selection',str(selection)]
        run(*command); evaluations.append(str(dest)); return dest
    print(f'{len(checkpoints)} unique saved candidates; {a.episodes} episodes per evaluation; train included: {a.include_train}',flush=True)
    vals=[]
    for i,ck in enumerate(checkpoints):
        vals.append(evaluate(ck,'val','candidate',f'val_{i:04d}'))
        if a.include_train: evaluate(ck,'train','candidate',f'train_{i:04d}')
    selection=out/'selection.json'
    run('select','--evaluations',*vals,'--out',selection)
    chosen=Path(json.loads(selection.read_text())['checkpoint'])
    evaluate(chosen,'test','best_validation','test_best',selection)
    evaluate(final,'test','final','test_final')
    write_json(out/'suite_manifest.json',dict(status='completed',algorithm=a.algorithm,
        training_run=str(root),selection_sha256=file_hash(selection),evaluations=evaluations))
    print(f'Completed evaluation suite: {out}')


if __name__=='__main__': main()
