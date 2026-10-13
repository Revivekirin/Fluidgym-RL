#!/usr/bin/env python
"""Remote-only SB3 SAC nominal trainer; local configuration and unit checks are safe."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fluidgym_rl.sac.runtime import DEFAULT_CONFIG,read_config,sb3_audit
from fluidgym_rl.comparison.schema import file_hash,write_json


class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,text):
        for stream in self.streams: stream.write(text); stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams: stream.flush()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--seed',type=int); p.add_argument('--steps',type=int)
    p.add_argument('--eval-freq',type=int); p.add_argument('--eval-episodes',type=int)
    p.add_argument('--device',default=None)
    p.add_argument('--out',type=Path)
    p.add_argument('--output-root',type=Path,default=Path(os.environ.get('FGRL_RESULTS_DIR','results'))/'sac_nominal')
    p.add_argument('--mode',choices=['smoke','pilot','full'],default='smoke')
    p.add_argument('--print-config-only','--print-config',action='store_true')
    p.add_argument('--check-env-only',action='store_true')
    p.add_argument('--verify-checkpoint',type=Path)
    p.add_argument('--reference',type=Path,help='shared uncontrolled training reference (optional; required for drag reduction)')
    p.add_argument('--allow-full-training',action='store_true')
    p.add_argument('--smoke-evidence',type=Path)
    p.add_argument('--wandb-mode',choices=['disabled','offline','online'],default='disabled')
    p.add_argument('--wandb-project',default='fluidgym-rl'); p.add_argument('--wandb-entity')
    p.add_argument('--wandb-group',default='sac-nominal'); p.add_argument('--wandb-tags',default='sac,nominal')
    return p


def run(args,config,audit,out):
    import torch
    from stable_baselines3 import SAC
    from fluidgym_rl.envs import make_fluid_env
    from fluidgym_rl.sac.environment import SACEnv
    from fluidgym_rl.sac.training import NominalCallback,Tracking
    from fluidgym_rl.comparison.evaluate import physics_manifest
    from fluidgym_rl.run_eval import environment_versions
    from fluidgym_rl.upstream import repo_root
    if not torch.cuda.is_available() or not config['device'].startswith('cuda'):
        raise RuntimeError('remote_required: FluidGym CUDA only; no local simulator/training run performed')
    torch.cuda.set_device(torch.device(config['device'])); torch.cuda.reset_peak_memory_stats()
    env=SACEnv(make_fluid_env(config['env_id'],mode='train',cuda_device=torch.device(config['device'])),config['seed'],config['device'])
    tracker=None; callback=None; result={'status':'failed'}; code=1
    try:
        physics=physics_manifest(env.inner)
        manifest=dict(algorithm='sac',run_id=out.name,mode=args.mode,versions=environment_versions(),sb3=audit,
            git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=repo_root()).strip(),
            git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True,cwd=repo_root()).strip()),
            physics=physics,seed=config['seed'],cuda=True,gpu_name=torch.cuda.get_device_name(),cuda_runtime=torch.version.cuda,
            source_hashes={str(p.relative_to(repo_root())):file_hash(p) for p in [Path(__file__).resolve(),*(repo_root()/'src/fluidgym_rl/sac').glob('*.py')]},
            training_reset='train split; seed=train_seed+episode index, randomize=True',
            checkpoint_semantics='SB3 zip; no replay/environment/RNG restoration and no resume CLI',
            smoke_evidence_sha256=file_hash(args.smoke_evidence) if args.smoke_evidence else None)
        write_json(out/'run_manifest.json',manifest)
        if args.smoke_evidence:
            prev=json.loads((args.smoke_evidence.parent/'run_manifest.json').read_text())
            if prev.get('mode')!='smoke' or not prev.get('cuda'): raise ValueError('evidence is not a real CUDA SAC smoke')
            if prev.get('source_hashes')!=manifest['source_hashes']: raise ValueError('SAC implementation changed since smoke; rerun smoke')
            if prev['physics']!=physics or prev['sb3']['version']!=audit['version']: raise ValueError('smoke environment/SB3 version differs')
        if args.check_env_only:
            from stable_baselines3.common.env_checker import check_env
            from fluidgym_rl.evalcontract import EvalContract,run_evaluation
            from fluidgym_rl.tdmpc.evaluation import ZeroPolicy
            check_env(env,warn=True,skip_render_check=True)
            rec=run_evaluation(env.inner,ZeroPolicy(),EvalContract(split='train',protocol='seeded',n_episodes=1),metric_names=['drag','lift','abs_lift'],action_low=env.action_space.low,action_high=env.action_space.high)
            result=dict(status='validated',real_episode_length=rec[0].length); code=0
            return result
        tracker=Tracking(out,args.tracking,config,manifest)
        model=SAC('MlpPolicy',env,seed=config['seed'],device=config['device'],verbose=0,**config['sac'])
        callback=NominalCallback(out,config,physics,tracker,reference=args.reference)
        model.learn(total_timesteps=config['steps'],callback=callback,log_interval=1)
        result=callback.finish()
        if env.real_steps!=config['steps']: raise ValueError('SB3 counter != actual simulator training calls')
        if args.mode=='smoke':
            cmd=[sys.executable,str(Path(__file__).resolve()),'--verify-checkpoint',str(out/'checkpoints/final.zip')]
            subprocess.run(cmd,check=True)
            result['checks']['checkpoint_fresh_process']='passed'
        result['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        tracker.log(config['steps'],{'resources/'+k:result[k] for k in ('wall_seconds','environment_steps_per_wall_second','gpu_peak_allocated_bytes')})
        code=0; return result
    finally:
        if callback and hasattr(callback,'gradients'): callback.gradients.close()
        env.close()
        if code: result={**result,'status':'failed'}
        if tracker: tracker.finish(result,code)


def main(argv=None):
    args=parser().parse_args(argv)
    if args.verify_checkpoint:
        from fluidgym_rl.sac.runtime import load_checkpoint
        load_checkpoint(args.verify_checkpoint.resolve(),'cpu')
        print('Fresh-process SAC inference probe passed (CPU; no simulator)'); return 0
    config=read_config(args.config,seed=args.seed,steps=(240 if args.mode=='smoke' and args.steps is None else args.steps),eval_freq=args.eval_freq,
                       eval_episodes=args.eval_episodes,device=args.device)
    audit=sb3_audit()
    from fluidgym_rl.tdmpc.tracking import options
    args.tracking=options(args)
    if args.print_config_only:
        print(json.dumps({'parameters':config,'sb3':audit,'tracking':args.tracking},indent=2)); return 0
    if not args.check_env_only:
        if args.mode=='smoke' and not 160<=config['steps']<=800: raise ValueError('smoke requires 160..800 steps')
        if args.mode=='pilot' and not 2000<=config['steps']<=5000: raise ValueError('pilot requires 2000..5000 steps')
        if args.mode=='full':
            from fluidgym_rl.safety import full_training_blockers
            blockers=full_training_blockers('full',args.allow_full_training)
            if blockers: raise RuntimeError('TRAINING BLOCKED: '+'; '.join(blockers))
            if not args.smoke_evidence: raise ValueError('--smoke-evidence <smoke/summary.json> required')
            evidence=json.loads(args.smoke_evidence.read_text())
            previous=json.loads((args.smoke_evidence.parent/'resolved_config.yaml').read_text())['parameters']
            if evidence.get('status')!='completed' or any(evidence.get('checks',{}).get(k)!='passed' for k in ('replay_learning','finite_gradients','parameters_changed','legal_actions','horizon_timeouts','checkpoint_roundtrip','shared_validation','checkpoint_fresh_process')):
                raise ValueError('SAC smoke validation incomplete')
            if previous['sac']!=config['sac']: raise ValueError('SAC smoke hyperparameters differ')
    from fluidgym_rl.evalcontract import new_run_dir
    if args.reference: args.reference=args.reference.resolve()
    if args.smoke_evidence: args.smoke_evidence=args.smoke_evidence.resolve()
    if args.out:
        out=args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    else: out=new_run_dir(args.output_root.resolve(),args.mode)
    write_json(out/'resolved_config.yaml',{'parameters':config,'sb3':audit,'tracking':args.tracking})
    result={'status':'failed'}; code=1
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        with redirect_stdout(Tee(sys.stdout,stdout)),redirect_stderr(Tee(sys.stderr,stderr)):
            try:
                result=run(args,config,audit,out); code=0
            except BaseException as exc:
                traceback.print_exc(); result={'status':'failed','error':repr(exc),'traceback':traceback.format_exc()}
                code=130 if isinstance(exc,KeyboardInterrupt) else 1
    write_json(out/'summary.json',result)
    (out/'exit_code.txt').write_text(str(code)+'\n')
    print(f'Exit code: {code} | evidence: {out}')
    return code


if __name__=='__main__': sys.exit(main())
