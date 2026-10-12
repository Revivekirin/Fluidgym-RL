"""Tracking API and CLI failure semantics, without CUDA or online communication."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fluidgym_rl.tdmpc.runtime import repo_root
from fluidgym_rl.tdmpc.tracking import Tracker, options


def args(**overrides):
    return SimpleNamespace(**{**dict(wandb_mode='offline',wandb_project='fluidgym-rl',wandb_entity=None,
                                    wandb_group='nominal',wandb_tags=''),**overrides})


class TrackingTests(unittest.TestCase):
    def test_options_and_empty_tags(self):
        self.assertEqual(options(args())['tags'],[])
        self.assertEqual(options(args(wandb_tags='tdmpc, seed1'))['tags'],['tdmpc','seed1'])
        for tags in ('a,,b',' ','x'*65):
            with self.assertRaises(ValueError): options(args(wandb_tags=tags))

    def test_disabled_does_not_import_wandb(self):
        tracker=Tracker('/unused')
        with patch.dict('sys.modules',{'wandb':None}):
            tracker.start(options(args(wandb_mode='disabled')),{}, {})
            tracker.log('train',dict(environment_steps=80))
            tracker.finish({'status':'completed'},0)

    def test_metrics_axis_identity_missing_values_and_finish(self):
        run=Mock(id='abc',url='https://example.invalid/run'); run.summary={}
        sdk=Mock()
        def init(**kwargs):
            self.assertNotIn('WANDB_TAGS',os.environ)
            self.assertEqual(kwargs['tags'],[])
            self.assertIsNone(kwargs['entity'])
            return run
        sdk.init.side_effect=init
        with tempfile.TemporaryDirectory() as tmp,patch.dict('sys.modules',{'wandb':sdk}),patch.dict(os.environ,{'WANDB_TAGS':''}):
            tracker=Tracker(tmp); tracker.start(options(args()),{'seed':1},{'upstream_sha':'pin'})
            self.assertEqual(os.environ['WANDB_TAGS'],'')
            tracker.log('train',dict(environment_steps=80,learner_updates=0,episode_return=-2))
            tracker.log('val',dict(environment_steps=80,mean_reward_per_step=.1,checkpoint='',missing=None))
            calls=run.log.call_args_list
            self.assertNotIn('train/value_loss',calls[1].args[0])
            self.assertEqual(calls[1].args[0]['learner_updates'],0)
            self.assertNotIn('val/missing',calls[2].args[0])
            self.assertNotIn('val/checkpoint',calls[2].args[0])
            self.assertTrue(all(not c.kwargs for c in calls))
            run.define_metric.assert_any_call('val/*',step_metric='environment_steps')
            self.assertEqual(json.loads((Path(tmp)/'wandb_run.json').read_text())['id'],'abc')
            (Path(tmp)/'summary.json').write_text('{}')
            tracker.finish({'status':'failed'},1)
            run.finish.assert_called_once_with(exit_code=1)
            self.assertEqual(run.summary['status'],'failed')

    def test_upload_failure_marks_wandb_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker=Tracker(tmp); tracker.run=Mock(); tracker.run.summary={}
            (Path(tmp)/'summary.json').write_text('{}')
            tracker.run.save.side_effect=RuntimeError('upload failed')
            with self.assertRaisesRegex(RuntimeError,'upload failed'):
                tracker.finish({'status':'completed'},0)
            tracker.run.finish.assert_called_once_with(exit_code=1)
            self.assertEqual(tracker.run.summary['status'],'failed')

    def test_cli_failure_and_finalize_failure_propagate(self):
        spec=importlib.util.spec_from_file_location('tdmpc_cli',repo_root()/'scripts/tdmpc_train.py')
        cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        for failure in ('training','finalization'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                tracker=Mock()
                def fake_run(a,data,out):
                    a._tracker=tracker
                    if failure=='training': raise RuntimeError('training failure')
                    return {'status':'completed'}
                if failure=='finalization': tracker.finish.side_effect=RuntimeError('finalization failure')
                with patch.object(cli,'run',side_effect=fake_run):
                    rc=cli.main(['--mode','smoke','--wandb-mode','offline','--output-root',tmp])
                self.assertEqual(rc,1)
                summary=json.loads(next(Path(tmp).glob('*/summary.json')).read_text())
                self.assertEqual(summary['status'],'failed')
                self.assertIn(failure+' failure',str(summary))
                self.assertEqual(tracker.finish.call_args.args[1],1 if failure=='training' else 0)

    @unittest.skipUnless(os.environ.get('FGRL_TEST_WANDB_OFFLINE')=='1','opt-in real W&B offline SDK test')
    def test_real_offline_sdk(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'WANDB_TAGS':''}):
            tracker=Tracker(tmp)
            tracker.start(options(args()),{'seed':1},{'algorithm':'tdmpc','test':'CPU logging only'})
            tracker.log('train',dict(environment_steps=80,learner_updates=0,episode_return=-1))
            tracker.log('train',dict(environment_steps=160,learner_updates=100,value_loss=.2))
            tracker.log('val',dict(environment_steps=160,mean_reward_per_step=.1,drag=3.1))
            result={'status':'completed','environment_steps':160,'learner_updates':100}
            (Path(tmp)/'summary.json').write_text(json.dumps(result))
            captured={}
            finish=tracker.run.finish
            def capture_finish(**kwargs):
                captured.update(dict(tracker.run.summary))
                return finish(**kwargs)
            with patch.object(tracker.run,'finish',side_effect=capture_finish):
                tracker.finish(result,0)
            self.assertTrue(list(Path(tmp).glob('wandb/offline-run-*/*.wandb')))
            self.assertEqual(captured['status'],'completed')
            self.assertEqual(captured['environment_steps'],160)
            self.assertEqual(captured['val/drag'],3.1)


if __name__=='__main__': unittest.main()
