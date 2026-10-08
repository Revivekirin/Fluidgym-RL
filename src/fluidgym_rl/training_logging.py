"""Small additions to upstream Logger; no optimizer or policy changes."""
import dataclasses
import json
import math
from pathlib import Path


def wandb_config(cfg):
    if dataclasses.is_dataclass(cfg):
        data = dataclasses.asdict(cfg)
    else:
        from omegaconf import OmegaConf
        data = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else vars(cfg)
    return json.loads(json.dumps(data, default=str))


def scalar_metrics(data):
    result = {}
    for name, value in data.items():
        if isinstance(value, (str, bool, int)):
            result[name] = value
        elif value is not None:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f'non-finite metric {name}: {value}')
            result[name] = value
    return result


def make_logger_class(base):
    class ExperimentLogger(base):
        def __init__(self, cfg, spec):
            super().__init__(cfg)
            self.spec = spec
            self.observer = None
            self.next_save = spec['args']['save_freq']
            self.records = Path(spec['out']) / 'metrics.jsonl'
            self.checkpoints = []
            self.last_train_step = -1
            # Upstream mutates cfg.save_agent when W&B is disabled even though
            # _save_agent was captured as True. Keep the resolved config accurate.
            cfg.save_agent = True
            if self._wandb:
                self._wandb.config.update({'experiment': spec}, allow_val_change=True)
                for category in ('train', 'eval'):
                    self._wandb.define_metric(category + '/environment_steps')
                    self._wandb.define_metric(category + '/*', step_metric=category + '/environment_steps')

        def log(self, data, category='train'):
            values = dict(data)
            if self.observer is not None:
                values.update(self.observer.metrics())
                if category == 'train':
                    values.update(self.observer.completed_episode)
            values = scalar_metrics(values)
            with self.records.open('a') as stream:
                stream.write(json.dumps({'category': category, **values}) + '\n')
            super().log(values, category)  # The only W&B scalar log call.
            step = int(values.get('environment_steps', values.get('step', 0)))
            if category == 'train':
                self.last_train_step = step
            if category == 'train' and step >= self.next_save:
                self.save_agent(self.observer.agent, identifier=f'step_{step}')
                self.next_save = (step // self.spec['args']['save_freq'] + 1) * self.spec['args']['save_freq']

        def save_agent(self, agent=None, identifier='final'):
            super().save_agent(agent, identifier)
            path = Path(self.model_dir) / f'{identifier}.pt'
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError(f'checkpoint was not saved: {path}')
            self.checkpoints.append(str(path))
            Path(self.spec['out'], 'checkpoints.json').write_text(json.dumps(self.checkpoints, indent=2))
            if self._wandb:
                self._wandb.run.summary['checkpoint_path'] = str(path)

        def finish(self, agent=None):
            # Upstream finish swallows save failures; propagate instead and keep
            # the same W&B run open for final evaluation and completion evidence.
            self.pending_final_agent = agent

        def close(self, exit_code=0):
            if self._wandb:
                self._wandb.finish(exit_code=exit_code)
    return ExperimentLogger
