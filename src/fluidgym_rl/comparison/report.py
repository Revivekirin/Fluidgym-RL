"""Comparison tables and static scientific figures. No access to original W&B runs."""
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fluidgym_rl.comparison.schema import independent_rows, file_hash, write_json, digest
from fluidgym_rl.comparison.audit import legacy_rows, audit_run, write_mapping

COLUMNS=['algorithm','run_id','train_seed','split','kind','role','environment_steps','training_wall_hours','cohort','experiment','checkpoint_sha256','source','source_sha256','metric','unit','reduction','value','status','reason']


def aggregate_seeds(frame):
    """Bootstrap independent training-seed means, never evaluation episodes."""
    valid=frame[(frame.status=='available') & (frame.kind=='independent_evaluation') & (frame.algorithm!='uncontrolled')]
    keys=['algorithm','cohort','experiment','split','role','metric','environment_steps']
    rows=[]
    for group,data in valid.groupby(keys,dropna=False):
        if data.train_seed.isna().any():
            raise ValueError('repeated or unknown training seed at the same metric/step/role; do not count repeated runs as independent seeds')
        if data.train_seed.duplicated().any():
            # Saved best/final candidates can occupy the same training step.
            # Preserve each checkpoint, without averaging them or choosing by
            # performance. This ambiguous group has no across-seed estimate.
            if (data.checkpoint_sha256.isna().any() or
                    data.duplicated(['train_seed','checkpoint_sha256']).any()):
                raise ValueError('repeated or unknown training seed at the same metric/step/role; do not count repeated runs as independent seeds')
            for _,record in data.iterrows():
                rows.append(dict(zip(keys,group),mean=float(record.value),n_seeds=1,
                                 ci_low=None,ci_high=None,
                                 series=f'checkpoint:{record.checkpoint_sha256}',
                                 uncertainty='unavailable: multiple checkpoints at one seed/step; displayed individually',
                                 training_wall_hours=record.training_wall_hours))
            continue
        values=data.value.to_numpy(float); n=len(values)
        lo=hi=None
        if n>=3:
            draws=np.random.default_rng(2026).choice(values,(4000,n),replace=True).mean(axis=1)
            lo,hi=[float(x) for x in np.quantile(draws,[.025,.975])]
        rows.append(dict(zip(keys,group),mean=float(values.mean()),n_seeds=n,ci_low=lo,ci_high=hi,
                         series='seed_mean',
                         uncertainty='95% percentile bootstrap over training seeds' if n>=3 else 'unavailable: fewer than 3 independent training seeds',
                         training_wall_hours=float(data.training_wall_hours.mean()) if data.training_wall_hours.notna().all() else None))
    return pd.DataFrame(rows,columns=keys+['mean','n_seeds','ci_low','ci_high','series','uncertainty','training_wall_hours'])


def render_figures(frame,stats,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels=[]
    performance=stats[~stats.metric.str.endswith('/drag_reduction_percent')]
    cohorts=sorted(performance.cohort.unique()) if not performance.empty else ['unavailable']
    for cohort in cohorts:
        data=stats[stats.cohort==cohort]
        suffix=digest(cohort)[:10]
        specs=[('validation_reward','val','mean_reward_per_step','environment_steps','Validation reward vs real environment steps'),
               ('validation_drag','val','drag','environment_steps','Validation drag vs real environment steps'),
               ('reward_wall_hours','val','mean_reward_per_step','training_wall_hours','Validation reward vs training wall-clock hours'),
               ('train_vs_validation',None,'mean_reward_per_step','environment_steps','Independent train vs validation reward'),
               ('test_best_vs_final','test','mean_reward_per_step',None,'Test: validation-selected vs final checkpoint'),
               ('multi_seed','val','mean_reward_per_step','environment_steps','Multi-seed validation mean and 95% bootstrap interval')]
        for name,split,metric,x,title in specs:
            subset=data[data.metric.str.endswith('/'+metric)]
            subset=subset[subset.split==split] if split else subset[subset.split.isin(['train','val'])]
            if name=='multi_seed': subset=subset[subset.n_seeds>=3]
            if name=='test_best_vs_final': subset=subset[subset.role.isin(['best_validation','final'])]
            if x: subset=subset[subset[x].notna()]
            fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
            reason=''
            if subset.empty:
                reason='Unavailable: no matching independent evaluation evidence'
                if name=='multi_seed': reason='Unavailable: fewer than 3 independent training seeds at shared steps'
                ax.text(.5,.5,reason,ha='center',va='center',wrap=True,transform=ax.transAxes)
            elif x is None:
                labels=[]
                for i,(_,row) in enumerate(subset.iterrows()):
                    ax.bar(i,row['mean']); labels.append(f"{row.algorithm}\n{row.role}\nn={row.n_seeds}")
                    if pd.notna(row.ci_low): ax.errorbar(i,row['mean'],yerr=[[row['mean']-row.ci_low],[row.ci_high-row['mean']]],color='black',capsize=4)
                ax.set_xticks(range(len(labels)),labels)
                if set(subset.role)!= {'best_validation','final'}:
                    reason='Partial evidence: best-validation or final test result missing'
            else:
                for (algorithm,experiment,split_name,role,series),group in subset.groupby(['algorithm','experiment','split','role','series']):
                    group=group.sort_values(x)
                    label=f'{algorithm} / {str(experiment)[:8]} / {split_name} / {role}'
                    if series!='seed_mean': label+=f' / {series[:23]}'
                    ax.plot(group[x],group['mean'],marker='o',linestyle='-' if series=='seed_mean' else 'None',label=label)
                    ci=group[group.ci_low.notna()]
                    if not ci.empty: ax.fill_between(ci[x].to_numpy(float),ci.ci_low.to_numpy(float),ci.ci_high.to_numpy(float),alpha=.18)
                ax.set_xlabel('Real training environment steps' if x=='environment_steps' else 'Recorded training wall-clock hours')
                ax.legend(fontsize=8)
                if name=='train_vs_validation' and set(subset.split)!= {'train','val'}:
                    reason='Partial evidence: independent train or validation evaluation missing; no legacy substitution'
            ax.set_title(title); ax.set_ylabel('Mean Cd' if metric=='drag' else 'Mean reward / control step')
            ax.grid(alpha=.2)
            stem=f'{name}_{suffix}'
            for ext in ('png','svg'): fig.savefig(out/f'{stem}.{ext}',dpi=150)
            plt.close(fig)
            panels.append(dict(name=title,cohort=cohort,image=f'{stem}.png',status=reason or 'available'))
    # Each objective has its own axis and each algorithm its own figure.
    diag=frame[(frame.status=='diagnostic') & frame.environment_steps.notna()]
    for algorithm,data in diag.groupby('algorithm'):
        metrics=sorted(data.metric.unique())
        fig,axes=plt.subplots(len(metrics),1,figsize=(8,3*len(metrics)),squeeze=False,layout='constrained')
        for ax,metric in zip(axes[:,0],metrics):
            for run_id,group in data[data.metric==metric].groupby('run_id'):
                group=group.sort_values('environment_steps')
                ax.plot(group.environment_steps,group.value,label=run_id)
            ax.set_title(f'{algorithm}: {metric.split("/")[-1]} (algorithm-specific diagnostic)')
            ax.set_xlabel('Real training environment steps'); ax.legend(fontsize=7)
        stem=f'diagnostics_{algorithm}'; fig.savefig(out/f'{stem}.png',dpi=150); fig.savefig(out/f'{stem}.svg'); plt.close(fig)
        panels.append(dict(name=f'{algorithm} loss diagnostics — not cross-algorithm comparable',cohort='diagnostics',image=f'{stem}.png',status='diagnostic'))
    return panels


def publish(frame,out,mode,project,entity):
    if mode=='disabled': return []
    import os
    import uuid
    import wandb
    identities=[]
    records=frame[frame.status.isin(['available','diagnostic']) & frame.environment_steps.notna()]
    for (algorithm,run_id,cohort,role),group in records.groupby(['algorithm','run_id','cohort','role']):
        # Explicit fresh ID: even inherited WANDB_RUN_ID cannot resume an original run.
        tags=os.environ.pop('WANDB_TAGS',None)
        try:
            run=wandb.init(id=uuid.uuid4().hex,mode=mode,project=project,entity=entity,dir=str(out),
                           name=f'compare-{algorithm}-{run_id}-{role}',group='nominal-independent-comparison',tags=['derived','comparison'],
                           resume='never' if mode=='online' else None,
                           config=dict(algorithm=algorithm,source_run=run_id,cohort=cohort,checkpoint_role=role,source_sha256s=sorted(group.source_sha256.unique())))
        finally:
            if tags is not None: os.environ['WANDB_TAGS']=tags
        code=1
        try:
            run.define_metric('compare/environment_steps')
            run.define_metric('compare/*',step_metric='compare/environment_steps')
            run.define_metric(f'diagnostics/{algorithm}/*',step_metric='compare/environment_steps')
            for _,row in group.sort_values('environment_steps').iterrows():
                run.log({'compare/environment_steps':int(row.environment_steps),row.metric:float(row.value)})
            identities.append(dict(id=run.id,source_run=run_id,algorithm=algorithm,role=role))
            code=0
        finally: run.finish(exit_code=code)
    return identities


def build_report(evaluations,training_runs,out,*,wandb_mode='disabled',wandb_project='fluidgym-compare',wandb_entity=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    rows=[]; snapshots=[]
    for root in evaluations:
        rows.extend(independent_rows(root))
        p=Path(root)/'evaluation_manifest.json'; snapshots.append({'path':str(p.resolve()),'sha256':file_hash(p)})
    audits=[audit_run(root) for root in training_runs]
    for root in training_runs: rows.extend(legacy_rows(root))
    frame=pd.DataFrame(rows,columns=COLUMNS)
    # Repeated copies of the same evidence must not inflate seed counts.
    frame=frame.drop_duplicates(subset=['algorithm','run_id','train_seed','split','kind','role','environment_steps','metric','source_sha256'])
    stats=aggregate_seeds(frame)
    frame.to_csv(out/'comparison.csv',index=False); stats.to_csv(out/'seed_summary.csv',index=False)
    # Fail explicitly if the optional comparison dependency was not installed.
    frame.to_parquet(out/'comparison.parquet',index=False); stats.to_parquet(out/'seed_summary.parquet',index=False)
    write_mapping(out/'metric_mapping.csv')
    panels=render_figures(frame,stats,out)
    identities=publish(frame,out,wandb_mode,wandb_project,wandb_entity)
    manifest=dict(schema_version=1,status='completed',source_evaluations=snapshots,training_audits=audits,
                  wandb_derived_runs=identities,plots=panels,uncertainty='4000 bootstrap draws, seed=2026, minimum 3 independent training seeds; no step interpolation',
                  result_hashes={p.name:file_hash(p) for p in out.iterdir() if p.suffix in ('.csv','.parquet','.png','.svg')})
    esc=html.escape
    body=''.join(f'<section><h2>{esc(p["name"])}</h2><p>Cohort: {esc(p["cohort"][:24])} · {esc(p["status"])}</p><img src="{esc(p["image"])}" alt="{esc(p["name"])}"></section>' for p in panels)
    table=frame[['algorithm','run_id','split','kind','metric','value','status','reason']].head(200).to_html(index=False,escape=True)
    (out/'dashboard.html').write_text('<!doctype html><html><meta charset="utf-8"><title>FluidGym nominal comparison</title>'
        '<style>body{font:16px system-ui;margin:2em;max-width:1200px}img{max-width:100%}td,th{padding:.4em;text-align:left}section{border-top:1px solid #ddd;margin-top:2em}table{font-size:12px}</style>'
        '<h1>FluidGym nominal comparison</h1><p>Independent evaluations only in performance figures. Splits and incompatible cohorts stay separate. Missing evidence is not zero.</p>'
        '<p><a href="comparison.csv">CSV</a> · <a href="comparison.parquet">Parquet</a> · <a href="metric_mapping.csv">Metric mapping</a> · <a href="report_manifest.json">Provenance</a></p>'
        +body+'<h2>Records (first 200)</h2>'+table+'</html>')
    manifest['result_hashes']['dashboard.html']=file_hash(out/'dashboard.html')
    write_json(out/'report_manifest.json',manifest)
    return manifest
