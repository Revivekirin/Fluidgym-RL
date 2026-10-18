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


def report(out,status):
    passed=status['status']=='passed'
    cost=json.loads((out/'cost.json').read_text()) if (out/'cost.json').exists() else {}
    episodes=json.loads((out/'episodes.json').read_text()) if (out/'episodes.json').exists() else []
    paired=json.loads((out/'paired_outcomes.json').read_text()) if (out/'paired_outcomes.json').exists() else []
    text='# Stage 2 frozen TD-MPC control results\n\nStatus: '+status['status']+'\n\n'
    text+='VERIFIED: recorded execution cost (partial on failure):\n```json\n'+json.dumps(cost,indent=2)+'\n```\n\n'
    text+='| Run | VAL domain | Episode return | Pre40 return | Post40 return | Mean Cd | Mean signed Cl | Mean absolute interval Cl | Clipping | Saturation |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for e in episodes:
        w=e['metrics']['whole']
        text+=f"| {e['run']} | {e['identity']['index']} | {w['reward_sum']:.9g} | {e['metrics']['pre40']['reward_sum']:.9g} | {e['metrics']['post40']['reward_sum']:.9g} | {w['mean_drag']:.9g} | {w['mean_signed_lift']:.9g} | {w['mean_abs_control_interval_lift']:.9g} | {w['clipping_frequency']:.6g} | {w['saturation_frequency']:.6g} |\n"
    text+='\nPaired post-shift outcomes (not isolated learned-model errors):\n```json\n'+json.dumps(paired,indent=2)+'\n```\n'
    text+='\nFull whole/pre40/post40 action and force metrics: episodes.json. Per-step actions, memory, reward and time: each run steps.json; independent measured boundary tensors: actuator_*.json and solver.jsonl. Successful outer calls and actual accepted counter increments are separate; constructor offsets are excluded from increments. Partial/failing runs may only have steps.json.\n'
    text+='\nINFERRED: gain delivery concerns physical actuator amplitude; changed actions and trajectories describe this frozen controller/planner response. UNRESOLVED: learned-model mismatch, irreducible control-authority loss, optimal achievable reward, history necessity and physical accuracy. Reward improvement or decrease is not itself a gate. Two environment seeds with two planner seeds form a small exploratory cohort; planner repeats are not independent physical initial conditions. No statistical significance or generalization claim.\n'
    (out/'STAGE2_FROZEN_CONTROL_RESULTS.md').write_text(text)
    validation='# Stage 2 prefix and actuator validation\n\n'+('VERIFIED: all four matched S0/S2 nominal-prefix gates and all native actuator/numerical gates passed. ' if passed else 'UNRESOLVED: complete acceptance; inspect worker_status.json for the exact blocker and saved partial prefix_differences.json. ')
    validation+='Per-run prefix_differences.json includes maximum absolute/relative errors, bitwise numeric equality and exact RNG identity; the relative denominator floor is 1e-12 for reporting only. Acceptance stays atol=rtol=1e-5, clocks atol=1e-6/rtol=0, counters exact; actuator readback atol=1e-7/rtol=1e-6.\n\n'
    validation+='Requested actions, clipped commands, native smoother memory, expected profile products and measured boundary readback are separate. C++ setter arguments are not directly intercepted. Planner RNG and warm state before/after inference are recorded; no candidate, latent or value diagnostics added. Frozen online/target weight hashes and config/std invariants are checked before each physical control. Native source/solver are unchanged. Full hidden-cache replay and constructor divergence-correction internals remain UNRESOLVED.\n'
    validation+='\nPrefix difference summaries (full field detail remains in JSON):\n'
    for path in sorted(out.glob('*_S2/prefix_differences.json')):
        rows=json.loads(path.read_text());groups={}
        exact=[r['exact_identity'] for r in rows if 'exact_identity' in r]
        validation+=f"\n{path.parent.name}: exact identity checks {sum(exact)}/{len(exact)}.\n\n"
        validation+='| Pair | Field category | Max absolute difference | Max relative difference | Numeric checks pass |\n|---|---|---:|---:|---|\n'
        for row in rows:
            if 'max_abs' not in row:continue
            field=row['field']
            category=('native state' if field.startswith('native_state') else 'planner action/warm' if field.startswith('planner') else 'observations/clocks' if field.startswith('o') else 'transition/actuator')
            groups.setdefault(category,[]).append(row)
        for category,items in groups.items():
            absolute=max((r['max_abs'] for r in items if r['max_abs'] is not None),default=0.)
            relative=max((r['max_relative'] for r in items if r['max_relative'] is not None),default=0.)
            validation+=f"| {path.parent.name} | {category} | {absolute:.9g} | {relative:.9g} | {all(r['allclose'] for r in items)} |\n"
    (out/'STAGE2_PREFIX_AND_ACTUATOR_VALIDATION.md').write_text(validation)
    recommendation=('Next smallest bounded experiment recommendation (NOT RUN): a zero-CFD, read-only offline comparison of one-step reward predictions against the recorded real transitions, with fixed frozen checkpoint, original 302-vector observations and actual clipped commands; compare predeclared S0 versus S2 pre/post40 errors within each seed pair. Limit to these 960 transitions, no fitting, no candidate replanning or new rollouts. A separate approval and target/alignment review are required; reward residuals alone do not uniquely identify dynamics mismatch. This recommendation adds no diagnostic implementation to this pilot.' if passed else 'Next smallest action: resolve the exact failing provenance/prefix/numerical gate using preserved evidence and zero new CFD. Do not restart the matrix or weaken tolerance without a separate decision.')
    (out/'STAGE2_GO_NO_GO.md').write_text('# Stage 2 GO / NO-GO\n\n'+('VERIFIED: bounded Stage 2 execution passed. ' if passed else 'NO-GO: execution failed or incomplete. ')+ 'Scientific model-mismatch/history/adaptation claims remain UNRESOLVED.\n\n'+recommendation+'\n\nNo TD-MPC2, training, additional CFD, or new schedules were executed or authorized by this report.\n')


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
    out=repo/'reports'/('stage2_frozen_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    write(out/'package_hashes.json',manifest)
    write(out/'launcher_status.json',{'status':'running','watchdog_seconds':2700,'output':str(out)})
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',FGRL_STAGE2_WATCHDOG='2700')
    env['PYTHONPATH']=str(repo/'src')+os.pathsep+env.get('PYTHONPATH','')
    start=time.monotonic();status={'status':'failed'};child=None
    print('OUTPUT_DIRECTORY='+str(out),flush=True)
    try:
        with (out/'worker.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B',str(HERE/'run_stage2.py'),str(repo),str(args.stage1.resolve()),str(out)],
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
                    print(f'Running bounded Stage 2: {time.monotonic()-start:.0f}s; '+str(out),flush=True)
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
                      'weight_checks':weight_checks,'prefix_gate_required_pairs':4,
                      'stage1_gate':json.loads((out/'stage1_gate.json').read_text()) if (out/'stage1_gate.json').exists() else None,
                      'cost':json.loads((out/'cost.json').read_text()) if (out/'cost.json').exists() else None,
                      'model_error_history_claims':'UNRESOLVED; not measured by this pilot'}
        write(out/'verification.json',verification)
        hashes={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='output_hashes.json'}
        write(out/'output_hashes.json',hashes)
        print('RESULT='+status['status']+'\nRETURN_DIRECTORY='+str(out),flush=True)
    if status['status']!='passed':sys.exit(1)


if __name__=='__main__':main()
