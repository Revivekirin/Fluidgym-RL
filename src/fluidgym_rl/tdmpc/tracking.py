"""Optional W&B sink for existing TD-MPC metrics; no learning hooks."""
import json
import os
from pathlib import Path


def options(args):
    tags = [] if not args.wandb_tags else args.wandb_tags.split(',')
    if any(not t.strip() or len(t.strip()) > 64 for t in tags):
        raise ValueError('--wandb-tags must contain nonempty comma-separated tags of 1..64 characters')
    if args.wandb_mode != 'disabled' and not args.wandb_project.strip():
        raise ValueError('--wandb-project must be nonempty')
    return dict(mode=args.wandb_mode, project=args.wandb_project, entity=args.wandb_entity,
                group=args.wandb_group, tags=[t.strip() for t in tags])


class Tracker:
    def __init__(self, out):
        self.run = None
        self.out = Path(out)

    def start(self, settings, config, manifest):
        if settings['mode'] == 'disabled':
            return
        # An inherited empty WANDB_TAGS caused a previous SDK Settings failure.
        # CLI tags own this run; restore the caller's environment after initialization.
        inherited = os.environ.pop('WANDB_TAGS', None)
        try:
            import wandb
            self.run = wandb.init(**settings, name=self.out.name, dir=str(self.out),
                                  config=dict(parameters=config, provenance=manifest),
                                  **({'resume':'never'} if settings['mode']=='online' else {}))
        finally:
            if inherited is not None:
                os.environ['WANDB_TAGS'] = inherited
        self.run.define_metric('environment_steps')
        self.run.define_metric('learner_updates', step_metric='environment_steps')
        for namespace in ('train', 'val', 'checkpoint'):
            self.run.define_metric(namespace+'/*', step_metric='environment_steps')
        identity = dict(id=self.run.id, url=self.run.url if settings['mode']=='online' else None, mode=settings['mode'])
        (self.out/'wandb_run.json').write_text(json.dumps(identity, indent=2))
        self.run.summary['status'] = 'running'
        self.run.log(dict(environment_steps=0, learner_updates=0))

    def log(self, category, row):
        if self.run is None:
            return
        data = {k:row[k] for k in ('environment_steps','learner_updates') if k in row}
        data.update({f'{category}/{k}':v for k,v in row.items()
                     if k not in data and v is not None and v != ''})
        # SDK history index stays monotonic even when train/eval share an env step.
        self.run.log(data)

    def finish(self, result, exit_code):
        if self.run is None:
            return
        try:
            self.run.summary.update(result)
            for name in ('resolved_config.yaml','run_manifest.json','train.csv','eval.csv','summary.json'):
                path = self.out/name
                if path.exists():
                    self.run.save(str(path), base_path=str(self.out), policy='now')
        except BaseException:
            self.run.summary['status'] = 'failed'
            self.run.finish(exit_code=exit_code or 1)
            raise
        else:
            self.run.finish(exit_code=exit_code)
