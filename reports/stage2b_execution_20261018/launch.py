#!/usr/bin/env python3
"""External 2700-second watchdog. Defaults to plan-only; no remote execution."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def write(path, data):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def post_preservation(repo, out):
    expected=json.loads((HERE/'expected.json').read_text())
    checks={}
    for relative,h in {**expected['required_artifacts'],**expected['checkpoints']}.items():
        path=repo/relative
        checks[relative]={'expected':h,'actual':sha(path) if path.is_file() else None}
    provenance=out/'provenance.json'
    if provenance.exists():
        p=json.loads(provenance.read_text());package=Path(p['fluidgym_file']).parent
        for relative,h in p['source_hashes'].items():
            path=package/relative
            checks[str(path)]={'expected':h,'actual':sha(path) if path.is_file() else None}
        backend=Path(p['backend']['path'])
        checks[str(backend)]={'expected':p['backend']['sha256'],'actual':sha(backend) if backend.is_file() else None}
        for relative,h in p['pipeline_hashes'].items():
            path=repo/relative;checks[str(path)]={'expected':h,'actual':sha(path) if path.is_file() else None}
        for relative,h in p['stage1_source_gate']['cpp_sources'].items():
            path=package/relative;checks[str(path)]={'expected':h,'actual':sha(path) if path.is_file() else None}
    result={'classification':'VERIFIED' if all(c['expected']==c['actual'] for c in checks.values()) else 'FAILED',
            'checks':checks,'source_baseline_available':provenance.exists()}
    write(out/'post_preservation.json',result)
    return result['classification']=='VERIFIED'


from reporting import report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--stage1',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    print('Frozen TD-MPC 50k; VAL env seeds 0,1 x planner seeds 0,1 x S0/S1/S2 x80; maximum 960 controls; watchdog 2700 s; no retry.',flush=True)
    if not args.execute:
        print('PLAN ONLY: add --execute to run this fixed protocol on the authorized server.');return
    manifest=json.loads((HERE/'package_hashes.json').read_text())
    for relative,expected in manifest.items():
        if sha(HERE/relative)!=expected:raise RuntimeError('package hash differs: '+relative)
    repo=args.repo.resolve()
    out=repo/'reports'/('stage2b_frozen_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    write(out/'package_hashes.json',manifest)
    (out/'STAGE2B_EXPERIMENT_CONTRACT.md').write_bytes((HERE/'STAGE2B_EXPERIMENT_CONTRACT.md').read_bytes())
    write(out/'launcher_status.json',{'status':'running','watchdog_seconds':2700,'output':str(out)})
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',FGRL_STAGE2B_WATCHDOG='2700')
    env['PYTHONPATH']=str(repo/'src')+os.pathsep+env.get('PYTHONPATH','')
    start=time.monotonic();status={'status':'failed'};child=None
    print('OUTPUT_DIRECTORY='+str(out),flush=True)
    try:
        with (out/'worker.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B',str(HERE/'run_stage2b.py'),str(repo),str(args.stage1.resolve()),str(out)],
                                   stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            while True:
                remaining=2700-(time.monotonic()-start)
                if remaining<=0:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                    status={'status':'watchdog_timeout','error':'External 2700-second limit reached; killed process group. No retry.'};break
                try:
                    rc=child.wait(timeout=min(30,remaining))
                    worker=json.loads((out/'worker_status.json').read_text()) if (out/'worker_status.json').exists() else {}
                    status={'status':'passed' if rc==0 and worker.get('status')=='passed' else 'failed',
                            'returncode':rc,'worker':worker};break
                except subprocess.TimeoutExpired:
                    print(f'Running bounded Stage 2B: {time.monotonic()-start:.0f}s; '+str(out),flush=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL);child.wait()
        status={'status':'interrupted','error':repr(exc)}
    finally:
        status['wall_seconds']=time.monotonic()-start
        status['automatic_retries']=0
        write(out/'launcher_status.json',status)
        try:
            preserved=post_preservation(repo,out)
            if not preserved and status['status']=='passed':
                status['status']='failed';status['error']='post-run artifact preservation check failed'
        except BaseException as exc:
            write(out/'post_preservation.json',{'classification':'UNRESOLVED','error':repr(exc)})
            if status['status']=='passed':status['status']='failed';status['error']='post-run preservation unavailable'
        write(out/'launcher_status.json',status)
        report(out,status)
        weight_checks={str(p.parent.name):json.loads(p.read_text()) for p in out.glob('*/weights_final.json')}
        verification={'execution_status':status['status'],'classification':'VERIFIED completed bounded pilot' if status['status']=='passed' else 'UNRESOLVED complete acceptance; see exact failure status',
                      'maximum_controls':960,'watchdog_seconds':2700,'automatic_retries':0,
                      'weight_checks':weight_checks,'diagnostic_prefix_pairs':4,'native_prefix_gate_required':False,
                      'stage1_gate':json.loads((out/'stage1_gate.json').read_text()) if (out/'stage1_gate.json').exists() else None,
                      'cost':json.loads((out/'cost.json').read_text()) if (out/'cost.json').exists() else None,
                      'model_error_history_claims':'UNRESOLVED; not measured by this pilot'}
        write(out/'verification.json',verification)
        hashes={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='output_hashes.json'}
        write(out/'output_hashes.json',hashes)
        print('RESULT='+status['status']+'\nRETURN_DIRECTORY='+str(out),flush=True)
    if status['status']!='passed':sys.exit(1)


if __name__=='__main__':main()
