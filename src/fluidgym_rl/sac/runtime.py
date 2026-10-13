"""Configuration and inference identity for new nominal SB3 SAC runs."""
import inspect
from importlib import metadata
import json
from pathlib import Path
import numpy as np
from fluidgym_rl.comparison.schema import file_hash, write_json

DEFAULT_CONFIG=Path(__file__).resolve().parents[3]/'configs/sac_nominal.json'


def read_config(path=DEFAULT_CONFIG, **overrides):
    data=json.loads(Path(path).read_text())
    data.update({k:v for k,v in overrides.items() if v is not None})
    if data['env_id']!='CylinderJet2D-easy-v0': raise ValueError('nominal CylinderJet2D only')
    for k in ('steps','eval_freq','eval_episodes'):
        if type(data[k]) is not int or data[k]<=0: raise ValueError(f'{k} must be a positive integer')
    for k in ('seed','eval_seed'):
        if type(data[k]) is not int or not 0<=data[k]<2**32: raise ValueError(f'invalid {k}')
    if data['sac']['train_freq']!=1 or data['sac']['gradient_steps']!=-1:
        raise ValueError('nominal real-step accounting requires train_freq=1, gradient_steps=-1')
    for key in ('buffer_size','batch_size','target_update_interval'):
        if type(data['sac'][key]) is not int or data['sac'][key]<=0: raise ValueError(f'invalid SAC {key}')
    if type(data['sac']['learning_starts']) is not int or data['sac']['learning_starts']<0: raise ValueError('invalid learning_starts')
    if not 0<data['sac']['gamma']<=1 or not 0<data['sac']['tau']<=1 or not 0<data['sac']['learning_rate']<1: raise ValueError('invalid SAC learning rate/discount/tau')
    if data['sac']['ent_coef']!='auto': raise ValueError('nominal SAC requires automatic entropy tuning')
    if data['steps']<=data['sac']['learning_starts']: raise ValueError('budget must reach replay learning')
    return data


def sb3_audit():
    import stable_baselines3 as sb3
    from stable_baselines3.sac.policies import SACPolicy
    version=tuple(int(x) for x in sb3.__version__.split('.')[:2])
    if not (2,7)<=version<(3,0): raise RuntimeError('tested API requires SB3 >=2.7,<3')
    defaults={k:repr(v.default) for k,v in inspect.signature(sb3.SAC.__init__).parameters.items() if v.default is not inspect.Parameter.empty}
    expected={'action_noise':None,'use_sde':False,'sde_sample_freq':-1,'use_sde_at_warmup':False,
              'optimize_memory_usage':False,'target_update_interval':1,'target_entropy':'auto'}
    signature=inspect.signature(sb3.SAC.__init__).parameters
    for key,value in expected.items():
        if signature[key].default!=value: raise RuntimeError(f'SB3 default drift: {key}')
    if 'n_steps' in signature and signature['n_steps'].default!=1: raise RuntimeError('SB3 n-step return default drift')
    import torch
    from stable_baselines3.common.torch_layers import FlattenExtractor
    policy_defaults=inspect.signature(SACPolicy.__init__).parameters
    for key,value in {'activation_fn':torch.nn.ReLU,'optimizer_class':torch.optim.Adam,'n_critics':2,
                      'share_features_extractor':False,'features_extractor_class':FlattenExtractor,
                      'normalize_images':True,'log_std_init':-3,'optimizer_kwargs':None}.items():
        if policy_defaults[key].default!=value: raise RuntimeError(f'SB3 policy default drift: {key}')
    return {'version':sb3.__version__,'constructor_defaults':defaults,
            'policy_signature':str(inspect.signature(SACPolicy.__init__)),
            'sac_source_sha256':file_hash(inspect.getfile(sb3.SAC)),
            'policy_source_sha256':file_hash(inspect.getfile(SACPolicy)),
            'paper':'https://arxiv.org/html/2601.15015v1#A4.T6',
            'explicit_paper_difference_from_sb3_default':'gradient_steps=-1 (SB3 default 1); equivalent with one env/train_freq=1'}


def save_checkpoint(model, path, config, run_id, physics, wall_seconds):
    path=Path(path)
    if path.exists() or path.with_suffix('.json').exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    model.save(path)
    probe=np.zeros((1,302),np.float32)
    action=model.predict(probe,deterministic=True)[0].tolist()
    sidecar=dict(algorithm='sac',sb3_version=metadata.version('stable-baselines3'),parameters={'seed':config['seed']},config=config,
        provenance={'run_id':run_id},environment_steps=int(model.num_timesteps),
        learner_updates=int(model._n_updates),sha256=file_hash(path),probe_action=action,
        physics=physics,training_wall_hours=wall_seconds/3600,
        semantics='SB3 zip includes optimizer state; replay/environment/RNG not saved, inference restoration only')
    write_json(path.with_suffix('.json'),sidecar)
    return sidecar


def load_checkpoint(path, device='cpu'):
    from stable_baselines3 import SAC
    path=Path(path); data=json.loads(path.with_suffix('.json').read_text())
    if data['algorithm']!='sac' or data['sha256']!=file_hash(path): raise ValueError('SAC checkpoint identity mismatch')
    if data['sb3_version']!=metadata.version('stable-baselines3'): raise ValueError('SAC saved/runtime SB3 versions differ; use recorded training environment')
    model=SAC.load(path,device=device)
    if model.observation_space.shape!=(302,) or model.action_space.shape!=(1,): raise ValueError('SAC observation/action mismatch')
    if not (np.all(model.action_space.low==-1) and np.all(model.action_space.high==1)): raise ValueError('SAC action bounds mismatch')
    action=model.predict(np.zeros((1,302),np.float32),deterministic=True)[0]
    if not np.isfinite(action).all() or not np.allclose(action,data['probe_action'],rtol=1e-5,atol=1e-6):
        raise ValueError('SAC deterministic checkpoint probe mismatch')
    return model,data
