"""Independent evaluations only. Never imports a trainer or mutates a checkpoint."""
from contextlib import contextmanager
from dataclasses import asdict
from importlib import metadata
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

from fluidgym_rl.evalcontract import EvalContract, run_evaluation, summarize, write_run, scalar
from fluidgym_rl.tdmpc.environment import CheckedEnv, CONTRACT
from fluidgym_rl.comparison.schema import VERSION, digest, file_hash, read_json, verify_result, write_json


class NominalEnv(CheckedEnv):
    def step(self, action):
        obs,reward,term,trunc,info=super().step(action)
        # The 2D environment exposes control-interval averages, not substep samples.
        expected=float(self.env._cd_ref)-scalar(info['drag'])-float(self.env._lift_penalty)*abs(scalar(info['lift']))
        if not math.isclose(reward,expected,rel_tol=1e-5,abs_tol=1e-5):
            raise ValueError(f'live native reward formula mismatch: {reward} vs {expected}')
        return obs,reward,term,trunc,info


def training_identity(algorithm, root, checkpoint, declared_steps=None):
    root,checkpoint=Path(root).resolve(),Path(checkpoint).resolve()
    checkpoint.relative_to(root)  # Prevent accidentally associating a different run.
    if algorithm=='tdmpc':
        sidecar=read_json(checkpoint.with_suffix('.json'))
        steps=sidecar['environment_steps']; seed=sidecar['parameters']['seed']
        if sidecar['sha256']!=file_hash(checkpoint): raise ValueError('checkpoint sidecar hash mismatch')
        config=sidecar
        run_id=sidecar.get('provenance',{}).get('run_id',root.name)
        step_source=str(checkpoint.with_suffix('.json'))
    else:
        rc=checkpoint.parent/'run_config.json'
        config=read_json(rc)
        seed=config.get('seed')
        if seed is None:
            for override in config.get('overrides',[]):
                if override.startswith('seed='): seed=int(override.split('=',1)[1])
        if seed is None: raise ValueError('training seed unavailable in run_config')
        run_id=config.get('experiment_id',root.name)
        if checkpoint.stem=='final' and (root/'status.json').exists():
            status=read_json(root/'status.json')
            if status.get('status')!='completed': raise ValueError('final checkpoint run is not completed')
            steps=status['environment_steps']; step_source=str(root/'status.json')
        elif checkpoint.stem.startswith('step_') and (root/'metrics.jsonl').exists():
            steps=int(checkpoint.stem[5:]); step_source=str(root/'metrics.jsonl')
            matches=[read_json_line for line in (root/'metrics.jsonl').read_text().splitlines()
                     if (read_json_line:=json.loads(line)).get('category')=='train' and read_json_line.get('environment_steps')==steps]
            if not matches: raise ValueError('checkpoint step not supported by recorded real environment counter')
        else:
            raise ValueError('real environment steps unavailable in original artifacts; cannot certify this checkpoint for comparison')
    if type(steps) is not int or steps<=0: raise ValueError('invalid checkpoint environment steps')
    if declared_steps is not None and steps!=declared_steps: raise ValueError('declared steps contradict checkpoint evidence')
    wall=None; wall_source=None
    path=root/('train.csv' if algorithm=='tdmpc' else 'metrics.jsonl')
    if path.exists():
        records=list(csv.DictReader(path.open())) if path.suffix=='.csv' else [json.loads(x) for x in path.read_text().splitlines()]
        for row in records:
            if row.get('category','train')=='train' and int(float(row.get('environment_steps',-1)))==steps:
                value=row.get('wall_seconds' if algorithm=='tdmpc' else 'elapsed_time')
                if value not in (None,''):
                    wall=float(value)/3600; wall_source={'path':str(path),'sha256':file_hash(path)}
    return dict(algorithm=algorithm,training_run_id=run_id,train_seed=int(seed),environment_steps=steps,step_source=step_source,
                step_source_sha256=file_hash(step_source) if Path(step_source).is_file() else None,
                training_wall_hours=wall,training_wall_source=wall_source,checkpoint=str(checkpoint),
                checkpoint_sha256=file_hash(checkpoint),training_run=str(root),source_config=config)


@contextmanager
def load_policy(algorithm, checkpoint, identity, device):
    import torch
    from fluidgym_rl.upstream import lock
    from fluidgym_rl.tdmpc.evaluation import Policy, ZeroPolicy, load_checkpoint
    if algorithm=='uncontrolled':
        yield ZeroPolicy(),{},None
        return
    cwd=Path.cwd()
    with tempfile.TemporaryDirectory(prefix='fgrl_compare_') as tmp:
        try:
            if algorithm=='tdmpc':
                from fluidgym_rl.tdmpc.runtime import make_workcopy,load_algorithm
                work=Path(tmp)/'tdmpc'; provenance=make_workcopy(work)
                Agent,_=load_algorithm(work)
                cfg=SimpleNamespace(**identity['source_config']['parameters']); cfg.device=device
                agent=Agent(cfg); load_checkpoint(agent,checkpoint)
                policy=Policy(agent,identity['environment_steps'])
            else:
                from fluidgym_rl.upstream import make_tdmpc2_workcopy
                from fluidgym_rl.tdmpc2_runtime import load_agent,TDMPC2Policy
                wc=make_tdmpc2_workcopy(Path(tmp)/'tdmpc2')
                cfg,agent,_=load_agent(str(checkpoint),str(checkpoint.parent/'run_config.json'),wc['code_dir'],Path(tmp)/'run')
                if not cfg.mpc: raise ValueError('shared nominal comparison requires saved MPC planning configuration')
                provenance={'upstream_sha':lock()['tdmpc2'][1],'patch_sha256':file_hash(wc['patch'])}
                policy=TDMPC2Policy(agent,'plan')
            from fluidgym_rl.device import same_device
            if not same_device(agent.device,device): raise ValueError('learner device does not match requested simulator device')
            shape=cfg.obs_shape if algorithm=='tdmpc' else cfg.obs_shape['state']
            if list(shape)!=[302] or cfg.action_dim!=1 or cfg.episode_length!=80:
                raise ValueError('checkpoint incompatible with nominal observation/action/horizon')
            class SeededPolicy:
                name, mode = policy.name, policy.mode
                def reset(self, episode_seed):
                    import random
                    import numpy as np
                    random.seed(episode_seed)
                    np.random.seed(episode_seed)
                    policy.reset(episode_seed)
                def act(self, obs, t0):
                    return policy.act(obs, t0)
            from fluidgym_rl.training_logging import wandb_config
            yield SeededPolicy(),provenance,wandb_config(cfg)
        finally:
            os.chdir(cwd)


def physics_manifest(env):
    import fluidgym
    from fluidgym.config import config as fluidgym_config
    package=Path(fluidgym.__file__).parent
    sources={str(p.relative_to(package)):file_hash(p) for p in sorted(package.rglob('*.py'))}
    domains=Path(fluidgym_config.initial_domains_path)/env.initial_domain_id
    dataset={str(p.relative_to(domains)):file_hash(p) for p in sorted(domains.rglob('*')) if p.is_file()}
    if not dataset: raise ValueError('initial-condition dataset provenance unavailable')
    numerical_versions={}
    for name in ('torch','numpy','hoptorch'):
        try: numerical_versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError: numerical_versions[name]=None
    return dict(contract=CONTRACT,fluidgym_version=metadata.version('fluidgym'),numerical_versions=numerical_versions,
                initial_domain_id=env.initial_domain_id,initial_domain_file_hashes=dataset,
                native_reward_drag_reference=float(env._cd_ref),
                installed_python_source_sha256=digest(sources),installed_source_hashes=sources,
                step_length=float(env.step_length),episode_length=env.episode_length,
                reynolds_number=float(env._reynolds_number),lift_penalty=float(env._lift_penalty),
                smoothing_alpha=float(env._action_smoothing_alpha),action_bounds=[-1,1],observation_shape=[302])


def aggregate(records, reference=None):
    if not records: raise ValueError('no completed evaluation episodes')
    for r in records:
        values=[r.ret,r.mean_reward,r.length,r.wall_seconds,r.action_mean,r.action_std,r.action_abs_max,r.action_clipped_frac,*r.metrics.values()]
        if not all(math.isfinite(float(v)) for v in values): raise ValueError('non-finite evaluation data')
        if r.length!=80 or r.terminated or not r.truncated: raise ValueError('nominal episode boundary violation')
    result=summarize(records)
    result['drag_reduction_percent']=None
    result['drag_reduction_status']='unavailable: no matched uncontrolled training reference'
    if reference is not None:
        drag=float(reference)
        if not math.isfinite(drag) or drag<=0: raise ValueError('invalid uncontrolled mean drag')
        result['drag_reduction_percent']=100*(1-result['metrics_per_step_mean']['drag']['mean']/drag)
        result['drag_reduction_status']='available: ratio of arithmetic means to measured uncontrolled training reference'
    result['claims_reproduction']=False
    return result


def check_reference(root, contract, physics):
    m,s=verify_result(root)
    expected={**asdict(contract),'split':'train'}
    if m['algorithm']!='uncontrolled' or m['contract']!=expected or m['physics']!=physics:
        raise ValueError('reference must use uncontrolled train split with identical physics, seeds, protocol and episode budget')
    return s['metrics_per_step_mean']['drag']['mean']


def validate_selection(path, identity):
    selected=read_json(path)
    candidates=[]
    for candidate in selected['candidates']:
        root=Path(candidate['path'])
        if file_hash(root/'evaluation_manifest.json')!=candidate['manifest_sha256']:
            raise ValueError('validation selection evidence changed')
        m,s=verify_result(root)
        if m['contract']['split']!='val': raise ValueError('test data cannot select a checkpoint')
        candidates.append((m,s))
    if not candidates: raise ValueError('empty selection')
    winner=min(candidates,key=lambda x:(-x[1]['mean_reward_per_step']['mean'],x[0]['environment_steps'],x[0]['checkpoint_sha256']))[0]
    for k in ('algorithm','training_run_id','train_seed','checkpoint_sha256','environment_steps'):
        if winner[k]!=identity[k] or selected[k]!=identity[k]: raise ValueError(f'best-validation checkpoint mismatch: {k}')
    if len({(m['algorithm'],m['training_run_id'],m['train_seed'],m['cohort']) for m,_ in candidates})!=1:
        raise ValueError('mixed validation candidates')


def evaluate_to_directory(args, out):
    import torch
    from fluidgym_rl.tdmpc.runtime import repo_root
    from fluidgym_rl.envs import make_fluid_env
    if not torch.cuda.is_available(): raise RuntimeError('remote_required: real FluidGym evaluation needs CUDA')
    torch.cuda.set_device(torch.device(args.device))
    contract=EvalContract(split=args.split,n_episodes=args.episodes,base_seed=args.base_seed,protocol='seeded')
    if args.algorithm=='uncontrolled':
        if args.split!='train' or args.role!='reference': raise ValueError('uncontrolled reference must use training split and reference role')
        identity=dict(training_run_id='uncontrolled',train_seed=None,environment_steps=0,checkpoint=None,checkpoint_sha256=None,training_wall_hours=None)
        checkpoint=None
    else:
        if args.role=='reference': raise ValueError('reference role requires uncontrolled policy')
        if not args.training_run or not args.checkpoint: raise ValueError('--training-run and --checkpoint required')
        checkpoint=args.checkpoint.resolve()
        identity=training_identity(args.algorithm,args.training_run,checkpoint,args.environment_steps)
    if args.role=='best_validation':
        if not args.selection: raise ValueError('--selection required for best_validation reporting')
        validate_selection(args.selection,identity)
    if args.role=='final' and (checkpoint is None or checkpoint.stem!='final'):
        raise ValueError('final role requires the saved final.pt; do not relabel a selected checkpoint')
    if args.split=='test' and args.role not in ('best_validation','final'):
        raise ValueError('test is reserved for final or frozen best-validation checkpoints')
    cfg=SimpleNamespace(task=contract.env_id,obs_shape=[302],action_dim=1,episode_length=80,device=args.device)
    with load_policy(args.algorithm,checkpoint,identity,args.device) as (policy,source,config):
        env=NominalEnv(make_fluid_env(contract.env_id,mode=contract.split,cuda_device=torch.device(args.device)),cfg)
        try:
            physics=physics_manifest(env)
            reference=check_reference(args.reference,contract,physics) if args.reference else None
            records=run_evaluation(env,policy,contract,metric_names=['drag','lift','abs_lift'],action_low=env.action_space.low,action_high=env.action_space.high)
            summary=aggregate(records,reference)
        finally:
            env.close()
    versions={}
    for name in ('torch','numpy','gymnasium','fluidgym','hydra-core','tensordict'):
        try: versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError: versions[name]=None
    manifest={**identity,'algorithm':args.algorithm,'schema_version':VERSION,'status':'completed',
              'source':source,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo_root(),text=True).strip(),
              'contract':asdict(contract),'physics':physics,'versions':versions,'inference_mode':policy.mode,
              'cohort':digest({'contract':{k:v for k,v in asdict(contract).items() if k!='split'},'physics':physics}),
              'checkpoint_role':args.role,'seeds':[args.base_seed+i for i in range(args.episodes)],
              'selection_sha256':file_hash(args.selection) if args.selection else None,
              'reference_manifest_sha256':file_hash(args.reference/'evaluation_manifest.json') if args.reference else None,
              'reward_formula_verified_live':True,'checkpoint_semantics':'inference only; no training resume',
              'metric_definitions':{'abs_lift':'abs(control interval mean Cl), not mean abs solver-substep Cl',
                                   'drag_reduction':'100*(1-mean_controlled_drag/mean_uncontrolled_train_drag)'}}
    manifest['pipeline_source_hashes']={str(p.relative_to(repo_root())):file_hash(p)
        for p in sorted((repo_root()/'src/fluidgym_rl').rglob('*.py'))}
    manifest['pipeline_source_hashes']['scripts/compare_nominal.py']=file_hash(repo_root()/'scripts/compare_nominal.py')
    # Do not aggregate seeds with different model/planner/learning settings.
    excluded={'seed','device','exp_name','work_dir','task_title','steps','checkpoint','data_dir',
              'eval_episodes','eval_freq','save_freq','enable_wandb','save_csv','save_video','save_agent'}
    signature={k:v for k,v in (config or {}).items() if k not in excluded and not k.startswith('wandb')}
    manifest['experiment_signature']=digest(signature)
    if checkpoint and file_hash(checkpoint)!=identity['checkpoint_sha256']:
        raise ValueError('checkpoint changed during evaluation')
    write_run(out,{'contract':asdict(contract),'inference_config':config},records,summary,manifest)
    manifest['result_hashes']={name:file_hash(out/name) for name in ('episodes.csv','summary.json','resolved_config.yaml')}
    write_json(out/'evaluation_manifest.json',manifest)
    return manifest
