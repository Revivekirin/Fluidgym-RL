#!/usr/bin/env python3
"""External 600-second watchdog. Defaults to plan-only; no remote execution."""
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
        for relative,h in p['stage1_source_gate']['cpp_sources'].items():
            path=package/relative;checks[str(path)]={'expected':h,'actual':sha(path) if path.is_file() else None}
    result={'classification':'VERIFIED' if all(c['expected']==c['actual'] for c in checks.values()) else 'FAILED',
            'checks':checks,'source_baseline_available':provenance.exists()}
    write(out/'post_preservation.json',result)
    return result['classification']=='VERIFIED'


def report(out, status):
    passed=status['status']=='passed'
    cost=json.loads((out/'cost.json').read_text()) if (out/'cost.json').exists() else {}
    implementation='VERIFIED runtime gates passed' if passed else 'UNRESOLVED: inspect worker_status.json and solver.jsonl for the last completed gate'
    (out/'STAGE1_IMPLEMENTATION_VALIDATION.md').write_text(
        '# Stage 1 implementation validation\n\n'+implementation+'\n\n'
        'Instance-only post-smoothing gain; original smoother, profiles, solver, reward and observations retained. '
        'Source/backend/dependency/data/config hashes: provenance.json. Native boundary velocity and flux readback: per-run solver.jsonl and actuator_*.json. '
        'Setter tensor arguments are not directly intercepted; expected profiles are separately labeled. '
        'UNRESOLVED: physical accuracy and hidden solver cache/full-state replay identity.\n')
    (out/'STAGE1_PAIRED_TRANSITION_RESULTS.md').write_text(
        '# Stage 1 paired transition results\n\nStatus: '+status['status']+'\n\n'
        'Cost (attempted controls include an interrupted transition):\n```json\n'+json.dumps(cost,indent=2)+'\n```\n\n'
        + ('VERIFIED: all nominal/repeat/prefix, measured actuator and numerical gates passed. '
           'INFERRED: downstream_effect.json reports the predeclared operational sensor-effect criterion; this is not physical accuracy or history identifiability.\n' if passed else
           'UNRESOLVED: complete pilot acceptance. Partial data are preserved; no retry or budget extension occurred. See worker_status.json for the exact blocker, or launcher_status.json for watchdog termination.\n'))
    results=out/'STAGE1_PAIRED_TRANSITION_RESULTS.md'
    with results.open('a') as f:
        f.write('\n| Run | Domain split/index | Completed controls | Sum reward | Mean drag | Mean lift |\n|---|---|---:|---:|---:|---:|\n')
        for path in sorted(out.glob('*/records.json')):
            records=json.loads(path.read_text())
            if not records:continue
            initial=records[0]['identity'];steps=records[1:];n=len(steps)
            f.write(f"| {path.parent.name} | {initial['split']}/{initial['index']} | {n} | {sum(x['reward'] for x in steps):.9g} | {sum(x['drag'] for x in steps)/n if n else 0:.9g} | {sum(x['lift'] for x in steps)/n if n else 0:.9g} |\n")
        effects=out/'downstream_effect.json'
        if effects.exists():
            for arm,value in json.loads(effects.read_text()).items():
                f.write(f"\n{arm}: operational downstream signal = {value['operational_downstream_signal']}. Full per-transition thresholds and differences: downstream_effect.json.\n")
    (out/'STAGE1_GO_NO_GO.md').write_text(
        '# Stage 1 GO / NO-GO\n\n'+('VERIFIED: bounded implementation/transition verification passed. ' if passed else 'NO-GO: failed or incomplete contract verification. ')
        +'UNRESOLVED: history necessity, hidden-gain identifiability and robust-control performance. '
        'Stage 2 is not launched or authorized by this result. Any further experiment requires a separate decision.\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    print('VAL seed=0 randomize=True; N-bare/N-hook/F/J, 2 x 12 each; 96 controls maximum; watchdog 600 s; no retry.',flush=True)
    if not args.execute:
        print('PLAN ONLY: add --execute to run this fixed protocol on the authorized server.');return
    manifest=json.loads((HERE/'package_hashes.json').read_text())
    for relative,expected in manifest.items():
        if sha(HERE/relative)!=expected:raise RuntimeError('package hash differs: '+relative)
    repo=args.repo.resolve()
    out=repo/'reports'/('stage1_pilot_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    write(out/'package_hashes.json',manifest)
    write(out/'launcher_status.json',{'status':'running','watchdog_seconds':600,'output':str(out)})
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',FGRL_STAGE1_WATCHDOG='600')
    env['PYTHONPATH']=str(repo/'src')+os.pathsep+env.get('PYTHONPATH','')
    start=time.monotonic();status={'status':'failed'};child=None
    print('OUTPUT_DIRECTORY='+str(out),flush=True)
    try:
        with (out/'worker.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B',str(HERE/'pilot.py'),str(repo),str(out)],
                                   stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            while True:
                remaining=600-(time.monotonic()-start)
                if remaining<=0:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                    status={'status':'watchdog_timeout','error':'External 600-second limit reached; killed process group. No retry.'};break
                try:
                    rc=child.wait(timeout=min(30,remaining))
                    worker=json.loads((out/'worker_status.json').read_text()) if (out/'worker_status.json').exists() else {}
                    status={'status':'passed' if rc==0 and worker.get('status')=='passed' else 'failed',
                            'returncode':rc,'worker':worker};break
                except subprocess.TimeoutExpired:
                    print(f'Running bounded Stage 1: {time.monotonic()-start:.0f}s; '+str(out),flush=True)
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
        hashes={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='output_hashes.json'}
        write(out/'output_hashes.json',hashes)
        print('RESULT='+status['status']+'\nRETURN_DIRECTORY='+str(out),flush=True)
    if status['status']!='passed':sys.exit(1)


if __name__=='__main__':main()
