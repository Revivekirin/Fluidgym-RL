import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fluidgym_rl.pilot import parser, blockers, hydra_overrides, wandb_environment
from fluidgym_rl.training_logging import make_logger_class, scalar_metrics
from fluidgym_rl.upstream import repo_root, third_party, upstream_status, make_tdmpc2_workcopy


class PilotTests(unittest.TestCase):
    def args(self, *extra):
        return parser().parse_args(['--mode', 'pilot', '--steps', '2400', *extra])

    def test_bound_and_full_gates(self):
        self.assertEqual(blockers(self.args()), [])
        self.assertTrue(blockers(self.args('--steps', '5001')))
        self.assertTrue(blockers(self.args('--steps', '0')))
        with patch.dict(os.environ, {'FGRL_ALLOW_FULL_TRAINING': '0'}):
            self.assertTrue(blockers(self.args('--mode', 'full', '--allow-full-training')))

    def test_modes_and_quoted_strings(self):
        for mode in ('disabled', 'offline', 'online'):
            a = self.args('--wandb-mode', mode, '--wandb-project', 'name, with=punctuation', '--wandb-tags', 'pilot', 'nominal')
            overrides = hydra_overrides(a, 'identity')
            self.assertIn('steps=2399', overrides)
            self.assertIn('wandb_entity=null', overrides)
            self.assertIn('enable_wandb=' + str(mode != 'disabled').lower(), overrides)
            self.assertEqual(wandb_environment(a)['WANDB_MODE'], mode)
            self.assertEqual(wandb_environment(a)['WANDB_TAGS'], 'pilot,nominal')
            self.assertEqual(json.loads(next(x.split('=', 1)[1] for x in overrides if x.startswith('wandb_project='))), a.wandb_project)

    def test_missing_metrics_are_not_zero_and_nonfinite_fails(self):
        self.assertEqual(scalar_metrics({'loss': 2.0, 'drag': None}), {'loss': 2.0})
        with self.assertRaises(ValueError): scalar_metrics({'loss': float('nan')})

    def test_omitted_tags_remove_inherited_value_without_mutating_parent(self):
        for tags in ('', 'stale-tag'):
            inherited = {'WANDB_TAGS': tags, 'PATH': '/bin'}
            result = wandb_environment(self.args('--wandb-mode', 'offline'), inherited)
            self.assertNotIn('WANDB_TAGS', result)
            self.assertEqual(result['PATH'], '/bin')
            self.assertEqual(inherited['WANDB_TAGS'], tags)
        self.assertNotIn('WANDB_TAGS', wandb_environment(self.args('--wandb-tags')))

    def test_explicit_tags_replace_inherited_tags(self):
        result = wandb_environment(self.args('--wandb-tags', 'nominal', 'pilot'), {'WANDB_TAGS': ''})
        self.assertEqual(result['WANDB_TAGS'], 'nominal,pilot')

    def test_invalid_tags_rejected_before_launch(self):
        for tag in ('', '   ', 'x' * 65, 'one,two'):
            with self.subTest(tag=tag):
                self.assertTrue(blockers(self.args('--wandb-tags', tag)))
        self.assertEqual(blockers(self.args('--wandb-tags', 'x' * 64)), [])

    def test_logging_patch_applies(self, header_variant=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'tdmpc2/common/logger.py'
            source.parent.mkdir(parents=True)
            content = (repo_root() / 'tests/fixtures/tdmpc2_logging_source.py.txt').read_bytes()
            self.assertEqual(content.count(b'\r\n'), content.count(b'\n'))
            if header_variant:
                content = b'# Different file preamble/import order must not matter.\r\n' + content.replace(b'import dataclasses\r\nimport os', b'import os\r\nimport dataclasses')
            source.write_bytes(content)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            subprocess.run(['git', 'apply', '--check', str(repo_root() / 'patches/tdmpc2_logging.patch')], cwd=root, check=True)
            subprocess.run(['git', 'apply', str(repo_root() / 'patches/tdmpc2_logging.patch')], cwd=root, check=True)
            ast.parse(source.read_text())
            self.assertIn('config=wandb_config(cfg)', source.read_text())
            result = source.read_bytes()
            self.assertEqual(result.count(b'\r\n'), result.count(b'\n'))

    def test_lf_patch_reproduces_crlf_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'tdmpc2/common/logger.py'
            source.parent.mkdir(parents=True)
            source.write_bytes((repo_root() / 'tests/fixtures/tdmpc2_logging_source.py.txt').read_bytes())
            broken = root / 'normalized.patch'
            broken.write_bytes((repo_root() / 'patches/tdmpc2_logging.patch').read_bytes().replace(b'\r\n', b'\n'))
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            result = subprocess.run(['git', 'apply', '--check', str(broken)], cwd=root, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('patch does not apply', result.stderr)

    def test_logging_patch_does_not_depend_on_file_header(self):
        self.test_logging_patch_applies(header_variant=True)

    @unittest.skipUnless((third_party() / 'tdmpc2' / '.git').exists(), 'pinned upstream checkout unavailable')
    def test_logging_patch_on_pinned_workcopy(self):
        before = upstream_status('tdmpc2')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'work'
            make_tdmpc2_workcopy(root)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            logging_patch = str(repo_root() / 'patches/tdmpc2_logging.patch')
            subprocess.run(['git', 'apply', '--check', logging_patch], cwd=root, check=True)
            subprocess.run(['git', 'apply', logging_patch], cwd=root, check=True)
            source = (root / 'tdmpc2/common/logger.py').read_text()
            ast.parse(source)
            self.assertIn('config=wandb_config(cfg)', source)
        self.assertEqual(upstream_status('tdmpc2'), before)
        self.assertTrue(before['matches'])
        self.assertFalse(before['dirty'])

    def test_single_log_path_and_save_failure_propagation(self):
        with tempfile.TemporaryDirectory() as tmp:
            class Base:
                def __init__(self, cfg):
                    self._wandb = None
                    self.model_dir = Path(tmp)
                    self.calls = []
                def log(self, data, category): self.calls.append((data, category))
                def save_agent(self, agent, identifier):
                    (self.model_dir / (identifier + '.pt')).write_bytes(b'model')
            logger = make_logger_class(Base)(SimpleNamespace(), {'out': tmp, 'args': {'save_freq': 80}})
            logger.observer = SimpleNamespace(agent=object(), completed_episode={'episode_length': 80}, metrics=lambda: {'environment_steps': 80, 'learner_updates': 12})
            logger.log({'step': 80, 'episode_reward': 4, 'value_loss': 1})
            self.assertEqual(len(logger.calls), 1)
            self.assertIsInstance(logger.calls[0][0]['step'], int)
            self.assertEqual(logger.calls[0][0]['environment_steps'], 80)
            self.assertEqual(logger.calls[0][0]['learner_updates'], 12)
            self.assertEqual(len(logger.checkpoints), 1)
            with patch.object(Base, 'save_agent', side_effect=OSError('disk full')):
                with self.assertRaises(OSError): logger.save_agent(object())

    @unittest.skipUnless(importlib.util.find_spec('hydra'), 'Hydra not installed')
    def test_hydra_composes_overrides(self):
        from hydra import compose, initialize_config_dir
        for mode in ('disabled', 'offline', 'online'):
            with initialize_config_dir(version_base=None, config_dir=str(repo_root() / 'tests/fixtures')):
                cfg = compose(config_name='tdmpc2_pilot_config', overrides=hydra_overrides(self.args('--wandb-mode', mode), 'test'))
            self.assertEqual(cfg.steps, 2399)
            self.assertEqual(cfg.enable_wandb, mode != 'disabled')




class WorkerObservationTests(unittest.TestCase):
    def test_environment_and_learner_counters_are_separate(self):
        import types
        spec = importlib.util.spec_from_file_location('pilot_worker_test', repo_root() / 'scripts/tdmpc2_pilot_worker.py')
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        class Optimizer: pass
        fake_torch = SimpleNamespace(optim=SimpleNamespace(Optimizer=Optimizer),
                                     cuda=SimpleNamespace(max_memory_allocated=lambda: 123))
        env = SimpleNamespace(step=lambda action: (None, 2., True, {'drag': 3., 'lift': -.1}))
        agent = SimpleNamespace(update=lambda buffer: {'value_loss': 1.})
        checks = types.ModuleType('fluidgym_rl.tdmpc2_checks')
        checks.check_losses = lambda metrics: dict(metrics)
        observer = worker.Observer(env, agent, SimpleNamespace(episode_length=80), fake_torch)
        with patch.dict(sys.modules, {'fluidgym_rl.tdmpc2_checks': checks}):
            env.step(None)
            agent.update(None)
            agent.update(None)
        observer.evaluating = True
        env.step(None)
        result = observer.metrics()
        self.assertEqual(result['environment_steps'], 1)
        self.assertEqual(result['learner_updates'], 2)
        self.assertEqual(result['evaluation_environment_steps'], 1)
        self.assertEqual(observer.eval_values['drag'], [3.])
        self.assertEqual(observer.completed_episode['episode_reward'], 2.)

    def test_subprocess_failure_is_propagated(self, returncode=7):
        spec = importlib.util.spec_from_file_location('pilot_launcher_test', repo_root() / 'scripts/tdmpc2_train_full.py')
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            wc = {'code_dir': tmp, 'changed_files': ['tdmpc2/common/layers.py', 'tdmpc2/envs/__init__.py'], 'upstream': {'path': tmp}}
            expected = sorted(wc['changed_files'] + ['tdmpc2/common/logger.py'])
            def run(command, **kwargs):
                if command[0] == sys.executable:
                    kwargs['stderr'].write('complete child traceback\n')
                    return SimpleNamespace(returncode=returncode)
                return SimpleNamespace(returncode=0)
            with patch.object(sys, 'argv', ['launcher', '--mode', 'pilot', '--steps', '2400', '--out', str(out)]), \
                 patch.object(launcher, 'make_tdmpc2_workcopy', return_value=wc), \
                 patch.object(launcher, '_changed_files', return_value=expected), \
                 patch.object(launcher, 'collect_metadata', return_value={}), \
                 patch.object(launcher.subprocess, 'run', side_effect=run):
                if returncode:
                    self.assertEqual(launcher.main(), returncode)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'without completion evidence'):
                        launcher.main()
            self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'failed')
            self.assertIn('complete child traceback', (out / 'stderr.log').read_text())

    def test_zero_exit_without_completion_evidence_is_failure(self):
        self.test_subprocess_failure_is_propagated(returncode=0)


if __name__ == '__main__': unittest.main()
