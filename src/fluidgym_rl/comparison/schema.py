"""Versioned comparison schema. Missing and incompatible values stay explicit."""
import hashlib
import json
from pathlib import Path

VERSION = 1
METRICS = {
    'mean_reward_per_step': ('reward/control step','arithmetic mean of episode per-step means'),
    'episode_return': ('reward/episode','arithmetic mean of summed episode rewards'),
    'drag': ('dimensionless Cd','mean control-interval Cd, then mean episodes'),
    'abs_lift': ('dimensionless |Cl|','abs(control-interval mean Cl), then mean steps/episodes'),
    'lift': ('dimensionless Cl','signed control-interval mean, then mean steps/episodes'),
    'drag_reduction_percent': ('percent','100 * (1 - mean Cd / uncontrolled mean Cd)'),
    'episode_length': ('control steps/episode','mean episode length'),
    'evaluation_wall_seconds': ('seconds','sum episode wall times including reset and planning'),
    'action_mean': ('normalized action','mean across steps/actions, then episodes'),
    'action_std': ('normalized action','mean within-episode population standard deviation'),
    'action_abs_max': ('normalized action','maximum absolute executed action'),
    'action_clipped_fraction': ('fraction','mean fraction of clipped steps per episode'),
}
LOSS_NAMES = ('consistency_loss','reward_loss','value_loss','pi_loss','policy_loss','total_loss','weighted_loss','grad_norm')


def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path,data):
    with Path(path).open('x') as stream: json.dump(data,stream,indent=2,allow_nan=False)


def verify_result(root):
    root=Path(root)
    m=read_json(root/'evaluation_manifest.json')
    if m.get('status')!='completed' or m.get('schema_version')!=VERSION:
        raise ValueError(f'not a completed comparison evaluation: {root}')
    for name in ('episodes.csv','summary.json','resolved_config.yaml'):
        if m['result_hashes'].get(name)!=file_hash(root/name):
            raise ValueError(f'evaluation evidence hash mismatch: {root/name}')
    expected=digest({'contract':{k:v for k,v in m['contract'].items() if k!='split'},'physics':m['physics']})
    if expected!=m['cohort']:
        raise ValueError('comparison cohort does not match evaluation contract')
    return m,read_json(root/'summary.json')


def metric_rows(meta, values, *, unavailable_reason='not recorded'):
    for metric,(unit,reduction) in METRICS.items():
        value=values.get(metric)
        status='available' if value is not None else 'unavailable'
        if value is not None:
            import math
            if not math.isfinite(float(value)):
                raise ValueError(f'non-finite comparison metric: {metric}')
        yield {**meta,'metric':f"compare/{meta['split']}/{metric}",'unit':unit,'reduction':reduction,
               'value':value,'status':status,'reason':'' if status=='available' else unavailable_reason}


def independent_rows(root):
    root=Path(root).resolve(); m,s=verify_result(root)
    meta=dict(algorithm=m['algorithm'],run_id=m['training_run_id'],train_seed=m['train_seed'],
              split=m['contract']['split'],kind='independent_evaluation',role=m['checkpoint_role'],
              environment_steps=m['environment_steps'],training_wall_hours=m.get('training_wall_hours'),
              cohort=m['cohort'],experiment=m.get('experiment_signature','unverified:'+m['training_run_id']),
              checkpoint_sha256=m['checkpoint_sha256'],source=str(root),
              source_sha256=file_hash(root/'evaluation_manifest.json'))
    values={k:s[k]['mean'] for k in ('mean_reward_per_step','episode_return','episode_length')}
    values.update({k:s['metrics_per_step_mean'][k]['mean'] for k in ('drag','lift','abs_lift')})
    values.update(drag_reduction_percent=s.get('drag_reduction_percent'),evaluation_wall_seconds=s['eval_wall_seconds'],
                  action_mean=s['action']['mean'],action_std=s['action']['std_within_episode'],
                  action_abs_max=s['action']['abs_max'],action_clipped_fraction=s['action']['clipped_step_fraction'])
    rows=list(metric_rows(meta,values,unavailable_reason='matched uncontrolled training reference unavailable'))
    # Ratios with different measured denominators are distinct comparison cohorts.
    for row in rows:
        if row['metric'].endswith('/drag_reduction_percent'):
            row['cohort']=digest({'cohort':m['cohort'],'reference':m.get('reference_manifest_sha256')})
    return rows


def select_best(roots, output):
    candidates=[]
    for root in roots:
        m,s=verify_result(root)
        if m['contract']['split']!='val' or m['algorithm']=='uncontrolled':
            raise ValueError('checkpoint selection accepts validation evaluations only')
        candidates.append((m,s,Path(root).resolve()))
    if not candidates: raise ValueError('no validation candidates')
    groups={(m['algorithm'],m['training_run_id'],m['train_seed'],m['cohort']) for m,_,_ in candidates}
    if len(groups)!=1: raise ValueError('selection must use one training run and one validation contract')
    m,s,root=min(candidates,key=lambda x:(-x[1]['mean_reward_per_step']['mean'],x[0]['environment_steps'],x[0]['checkpoint_sha256']))
    result=dict(schema_version=VERSION,rule='maximum validation mean_reward_per_step; ties earliest environment step then hash',
                algorithm=m['algorithm'],training_run_id=m['training_run_id'],train_seed=m['train_seed'],
                checkpoint=m['checkpoint'],checkpoint_sha256=m['checkpoint_sha256'],environment_steps=m['environment_steps'],
                candidates=[{'path':str(r),'manifest_sha256':file_hash(r/'evaluation_manifest.json')} for _,_,r in candidates],
                selected_evaluation=str(root),score=s['mean_reward_per_step']['mean'])
    write_json(output,result)
    return result
