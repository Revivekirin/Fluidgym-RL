#!/usr/bin/env python3
"""Original TD-MPC nominal baseline; importing/printing config never initializes CUDA."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout, redirect_stderr
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fluidgym_rl.tdmpc.runtime import DEFAULT_CONFIG, read_config, namespace, make_workcopy, load_algorithm, repo_root


class Tee:
    def __init__(self, stream, file):
        self.stream,self.file = stream,file
    def write(self, text):
        self.stream.write(text); self.file.write(text); self.file.flush()
        return len(text)
    def flush(self):
        self.stream.flush(); self.file.flush()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--seed',type=int)
    p.add_argument('--output-root',type=Path,default=repo_root()/'results/tdmpc')
    p.add_argument('--device',default=None)
    p.add_argument('--mode',choices=['validate','smoke','full'],default='validate')
    p.add_argument('--print-config',action='store_true')
    p.add_argument('--smoke-steps',type=int,default=240)
    p.add_argument('--allow-full-training',action='store_true')
    p.add_argument('--checkpoint',type=Path,help='independent checkpoint evaluation; no training')
    p.add_argument('--verify-checkpoint',action='store_true',help='fresh-process weight/probe check only; no simulator')
    p.add_argument('--uncontrolled',action='store_true',help='10 zero-action train episodes to establish drag reference')
    p.add_argument('--reference',type=Path,help='uncontrolled evaluation directory for drag reduction')
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--wandb-mode',choices=['disabled','offline','online'],default='disabled')
    p.add_argument('--wandb-project',default='fluidgym-rl')
    p.add_argument('--wandb-entity')
    p.add_argument('--wandb-group',default='tdmpc-nominal')
    p.add_argument('--wandb-tags',default='',help='comma-separated tags')
    return p


def run(args, data, out):
    import torch
    from fluidgym_rl.tdmpc.environment import make_env, CONTRACT
    from fluidgym_rl.tdmpc.evaluation import Policy, ZeroPolicy, evaluate, load_checkpoint
    from fluidgym_rl.tdmpc.runner import run_training, seed_all
    cfg=namespace(data)
    if args.checkpoint:
        sidecar=json.loads(args.checkpoint.with_suffix('.json').read_text())
        cfg=type(cfg)(**sidecar['parameters'])
        data['provenance']=sidecar.get('provenance',{}).get('parameter_provenance',data['provenance'])
        if args.device:
            cfg.device=args.device
        if args.seed is not None:
            raise ValueError('--seed cannot override checkpoint train seed; evaluation seed follows saved train seed')
    if cfg.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('remote_required: CUDA unavailable; no simulator validation performed')
        torch.cuda.set_device(torch.device(cfg.device)) if ':' in cfg.device else None
        cfg.device=str(torch.device('cuda',torch.cuda.current_device()))
        torch.cuda.reset_peak_memory_stats(cfg.device)
    elif not args.verify_checkpoint:
        raise RuntimeError('remote_required: FluidGym requires CUDA; CPU algorithm tests are in tests/test_tdmpc.py')
    seed_all(cfg.seed)
    versions={'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda}
    for name in ('fluidgym','gymnasium','numpy','wandb'):
        try: versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError: versions[name]=None
    git=lambda *a:subprocess.check_output(['git',*a],cwd=repo_root(),text=True).strip()
    manifest=dict(algorithm='tdmpc',repository_sha=git('rev-parse','HEAD'),repository_dirty=bool(git('status','--porcelain')),
                  versions=versions,environment_contract=CONTRACT,seed=cfg.seed,parameter_provenance=data['provenance'],
                  evaluation_selection='validation only; test evaluations require separate CLI invocation',
                  run_id=out.name,mode=args.mode,tracking=data['tracking'],
                  requested_environment_steps=args.smoke_steps if args.mode=='smoke' else cfg.train_steps,
                  training_reset_protocol='seed=train_seed+episode_index, randomize=True; train split',
                  checkpoint_semantics='inference only: online+target weights')
    manifest['local_source_hashes'] = {}
    if cfg.device.startswith('cuda'):
        manifest['gpu'] = dict(name=torch.cuda.get_device_name(cfg.device),device=cfg.device,capability=torch.cuda.get_device_capability(cfg.device))
    from fluidgym_rl.tdmpc.runtime import sha256
    for p in [Path(__file__),*(repo_root()/'src/fluidgym_rl/tdmpc').glob('*.py')]:
        manifest['local_source_hashes'][str(p.relative_to(repo_root()))]=sha256(p)
    with tempfile.TemporaryDirectory(prefix='fgrl_tdmpc_') as tmp:
        work=Path(tmp)/'upstream'
        manifest['upstream']=make_workcopy(work)
        Agent,helper=load_algorithm(work)
        (out/'resolved_config.yaml').write_text(json.dumps(dict(parameters=vars(cfg),provenance=data['provenance'],tracking=data['tracking']),indent=2))
        (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
        if args.verify_checkpoint:
            if not args.checkpoint:
                raise ValueError('--verify-checkpoint requires --checkpoint')
            agent=Agent(cfg); load_checkpoint(agent,args.checkpoint)
            return dict(status='verified',checkpoint=str(args.checkpoint),check='fresh-process inference probe')
        env=make_env(cfg)
        import inspect
        import fluidgym.wrappers.flatten_obs as flatten_module
        import fluidgym.envs.cylinder.cylinder_env_base as cylinder_module
        manifest['installed_environment_source_hashes'] = {m.__name__:sha256(inspect.getfile(m)) for m in (flatten_module,cylinder_module)}
        (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
        try:
            if args.mode == 'validate' and not args.checkpoint and not args.uncontrolled:
                # One full real episode verifies boundaries and physical info; no learner.
                result=evaluate(env,ZeroPolicy(),cfg,out/'environment_validation',split='train',episodes=1,seed=cfg.seed,manifest=manifest)
                return dict(status='validated',checks=dict(environment_contract='passed',real_episode='passed',horizon_truncation='passed'),evaluation=result)
            if args.uncontrolled:
                return dict(status='evaluated',evaluation=evaluate(env,ZeroPolicy(),cfg,out/'evaluation',split='train',episodes=10,seed=cfg.seed,manifest=manifest))
            agent=Agent(cfg)
            if args.checkpoint:
                sc=load_checkpoint(agent,args.checkpoint)
                return dict(status='evaluated',evaluation=evaluate(env,Policy(agent,sc['environment_steps']),cfg,out/'evaluation',split=args.split,
                    episodes=args.episodes,seed=cfg.seed,checkpoint=args.checkpoint,manifest=manifest,reference=args.reference))
            from fluidgym_rl.tdmpc.tracking import Tracker
            args._tracker = Tracker(out)
            args._tracker.start(data['tracking'],vars(cfg),manifest)
            budget=args.smoke_steps if args.mode=='smoke' else cfg.train_steps
            result=run_training(cfg,agent,helper,env,out,manifest,budget=budget,smoke=args.mode=='smoke',tracker=args._tracker)
            if args.mode=='smoke':
                load_checkpoint(Agent(cfg),out/'checkpoints/final.pt')
                cmd=[sys.executable,str(Path(__file__).resolve()),'--checkpoint',str(out/'checkpoints/final.pt'),
                     '--verify-checkpoint','--device',cfg.device,'--output-root',str(out/'fresh_process')]
                # Parent model/environment remain allocated; child validates small model only.
                subprocess.run(cmd,check=True)
                result['checks']={k:'passed' for k in ('real_environment','replay_sequences','finite_losses_gradients','optimizer_updates','parameters_changed',
                     'bounded_actions','validation_evaluation','checkpoint_same_process','checkpoint_fresh_process')}
            return result
        finally:
            env.close()


def main(argv=None):
    args=parser().parse_args(argv)
    data=read_config(args.config,seed=args.seed,device=args.device)
    from fluidgym_rl.tdmpc.tracking import options
    data['tracking']=options(args)
    if args.print_config:
        print(json.dumps(data,indent=2)); return 0
    if args.wandb_mode != 'disabled' and (args.mode == 'validate' or args.checkpoint or args.uncontrolled or args.verify_checkpoint):
        raise ValueError('W&B tracking is supported for --mode smoke/full training only')
    if args.checkpoint:
        args.checkpoint=args.checkpoint.resolve()
    if args.verify_checkpoint and not args.checkpoint:
        raise ValueError('--verify-checkpoint requires --checkpoint')
    if args.smoke_steps % data['parameters']['episode_length']:
        raise ValueError('smoke budget must contain whole episodes')
    if args.uncontrolled and args.checkpoint:
        raise ValueError('--uncontrolled and --checkpoint are mutually exclusive')
    if args.mode=='full' and not (args.checkpoint or args.uncontrolled):
        from fluidgym_rl.safety import full_training_blockers
        blockers=full_training_blockers('full',args.allow_full_training)
        if blockers:
            raise RuntimeError('TRAINING BLOCKED: '+ '; '.join(blockers))
    if args.mode=='smoke' and not 160 <= args.smoke_steps <= 800:
        raise ValueError('smoke budget must be 160..800 steps')
    from fluidgym_rl.evalcontract import new_run_dir
    out=new_run_dir(args.output_root.resolve(), 'verify' if args.verify_checkpoint else 'eval' if args.checkpoint or args.uncontrolled else args.mode)
    print('Run directory:',out,flush=True)
    result={'status':'running'}; code=1
    (out/'summary.json').write_text(json.dumps(result))
    with open(out/'stdout.log','w') as stdout,open(out/'stderr.log','w') as stderr:
        with redirect_stdout(Tee(sys.stdout,stdout)),redirect_stderr(Tee(sys.stderr,stderr)):
            try:
                result=run(args,data,out); code=0
            except BaseException as e:
                traceback.print_exc()
                result=dict(status='failed',error=repr(e),traceback=traceback.format_exc())
                code=130 if isinstance(e,KeyboardInterrupt) else 1
            finally:
                (out/'summary.json').write_text(json.dumps(result,indent=2))
                tracker=getattr(args,'_tracker',None)
                if tracker is not None:
                    try:
                        tracker.finish(result,code)
                    except BaseException as e:
                        traceback.print_exc()
                        result.update(status='failed',tracking_error=repr(e),tracking_traceback=traceback.format_exc())
                        code=code or 1
                        (out/'summary.json').write_text(json.dumps(result,indent=2))
                (out/'exit_code.txt').write_text(str(code)+'\n')
    print(f'Exit code: {code} | evidence: {out}',flush=True)
    return code


if __name__=='__main__':
    sys.exit(main())
