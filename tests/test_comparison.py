"""CPU-only contract tests; fixture numbers are synthetic, never experiment evidence."""
from dataclasses import asdict
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

from fluidgym_rl.evalcontract import EpisodeRecord, EvalContract, write_run
from fluidgym_rl.comparison.schema import digest, file_hash, write_json, verify_result, select_best, independent_rows
from fluidgym_rl.comparison.evaluate import aggregate, check_reference, training_identity, validate_selection
from fluidgym_rl.comparison.audit import legacy_rows


def fixture(root, *, split='val', seed=1, step=80, reward=1., algorithm='tdmpc', physics=None):
    root.mkdir()
    contract=asdict(EvalContract(split=split, n_episodes=1, protocol='seeded'))
    records=[EpisodeRecord(0, {'seed':0,'randomize':True}, reward*80,reward,80,False,True,
                           0.,.1,.2,0.,{'drag':3.,'lift':-.1,'abs_lift':.2},2.)]
    summary=aggregate(records)
    physics=physics or {'version':'test-fixture'}
    m=dict(schema_version=1,status='completed',algorithm=algorithm,training_run_id='run',train_seed=seed,
           environment_steps=step,checkpoint=str(root/'fixture.pt'),checkpoint_sha256=digest(str(root)),
           checkpoint_role='candidate',contract=contract,physics=physics,training_wall_hours=step/3600,
           cohort=digest({'contract':{k:v for k,v in contract.items() if k!='split'},'physics':physics}))
    write_run(root,{},records,summary,m)
    m['result_hashes']={n:file_hash(root/n) for n in ('episodes.csv','summary.json','resolved_config.yaml')}
    write_json(root/'evaluation_manifest.json',m)
    return m,records


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def test_aggregation_abs_lift_and_reference(self):
        _,records=fixture(self.root/'eval')
        s=aggregate(records,4.)
        self.assertEqual(s['drag_reduction_percent'],25.)
        self.assertEqual(s['metrics_per_step_mean']['abs_lift']['mean'],.2)
        self.assertNotEqual(abs(s['metrics_per_step_mean']['lift']['mean']),.2)
        self.assertIsNone(aggregate(records)['drag_reduction_percent'])
        with self.assertRaises(ValueError): aggregate(records,0)
        records[0].length=79
        with self.assertRaises(ValueError): aggregate(records)

    def test_provenance_tampering(self):
        fixture(self.root/'eval'); verify_result(self.root/'eval')
        (self.root/'eval'/'summary.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'hash mismatch'): verify_result(self.root/'eval')

    def test_native_reward_formula_and_signed_lift(self):
        import numpy as np
        from gymnasium.spaces import Box
        from types import SimpleNamespace
        from fluidgym_rl.comparison.evaluate import NominalEnv
        class Env:
            episode_length=80
            observation_space=Box(-np.inf,np.inf,(302,),dtype=np.float32)
            action_space=Box(-1,1,(1,),dtype=np.float32)
            _cd_ref=4.
            _lift_penalty=1.
            reward=.5
            def step(self, action):
                return np.zeros(302,np.float32),self.reward,False,False,{'drag':3.,'lift':-.5}
        source=Env()
        env=NominalEnv(source,SimpleNamespace(obs_shape=[302],action_dim=1,episode_length=80))
        self.assertEqual(env.step(np.zeros(1))[4]['abs_lift'],.5)
        source.reward=1.5  # Incorrectly subtracting signed lift must fail.
        with self.assertRaisesRegex(ValueError,'reward formula mismatch'): env.step(np.zeros(1))

    @unittest.skipUnless(importlib.util.find_spec('pandas'), 'requires comparison extra')
    def test_distinct_model_configurations_are_not_seed_replicates(self):
        import pandas as pd
        from fluidgym_rl.comparison.report import aggregate_seeds
        rows=[]
        for seed in (1,2,3):
            p=self.root/str(seed); fixture(p,seed=seed)
            batch=independent_rows(p)
            for row in batch: row['experiment']=f'model-{seed}'
            rows+=batch
        stats=aggregate_seeds(pd.DataFrame(rows))
        self.assertTrue((stats.n_seeds==1).all())
        self.assertTrue(stats.ci_low.isna().all())

    def test_selection_validation_only_and_consistent_contract(self):
        a,_=fixture(self.root/'a',reward=1)
        b,_=fixture(self.root/'b',reward=2,step=160)
        result=select_best([self.root/'a',self.root/'b'],self.root/'selection.json')
        self.assertEqual(result['checkpoint_sha256'],b['checkpoint_sha256'])
        validate_selection(self.root/'selection.json',b)
        with self.assertRaises(ValueError): validate_selection(self.root/'selection.json',a)
        fixture(self.root/'test',split='test',reward=100)
        with self.assertRaises(ValueError): select_best([self.root/'test'],self.root/'bad.json')
        fixture(self.root/'other',physics={'version':'different'})
        with self.assertRaises(ValueError): select_best([self.root/'a',self.root/'other'],self.root/'bad.json')

    def test_reference_must_match_physics_seeds_and_split(self):
        m,_=fixture(self.root/'ref',split='train',algorithm='uncontrolled')
        contract=EvalContract(split='val',n_episodes=1,protocol='seeded')
        self.assertEqual(check_reference(self.root/'ref',contract,m['physics']),3)
        with self.assertRaises(ValueError): check_reference(self.root/'ref',contract,{'version':'other'})
        with self.assertRaises(ValueError): check_reference(self.root/'ref',EvalContract(split='val',n_episodes=2,protocol='seeded'),m['physics'])

    def test_real_steps_not_upstream_loop_limit(self):
        root=self.root/'run'; models=root/'models'; models.mkdir(parents=True)
        ck=models/'final.pt'; ck.write_bytes(b'fixture')
        write_json(models/'run_config.json',{'seed':1,'overrides':['steps=49999']})
        write_json(root/'status.json',{'status':'completed','environment_steps':50000})
        identity=training_identity('tdmpc2',root,ck)
        self.assertEqual(identity['environment_steps'],50000)
        self.assertIsNone(identity['training_wall_hours'])
        with self.assertRaises(ValueError): training_identity('tdmpc2',root,ck,49999)

    def test_legacy_eval_is_train_and_missing_not_zero(self):
        root=self.root/'run'; root.mkdir()
        write_json(root/'status.json',{'status':'completed'})
        (root/'metrics.jsonl').write_text(json.dumps({'category':'eval','environment_steps':800,'episode_reward':80,'lift':-.1})+'\n')
        rows=legacy_rows(root)
        self.assertEqual({r['split'] for r in rows},{'train'})
        lift=next(r for r in rows if r['metric'].endswith('/abs_lift'))
        self.assertEqual(lift['status'],'unavailable'); self.assertIsNone(lift['value'])
        reward=next(r for r in rows if r['metric'].endswith('/episode_return'))
        self.assertEqual(reward['status'],'incompatible')

    @unittest.skipUnless(importlib.util.find_spec('pandas'), 'requires comparison extra')
    def test_seed_intervals_do_not_count_episodes_or_mix_splits(self):
        import pandas as pd
        from fluidgym_rl.comparison.report import aggregate_seeds
        rows=[]
        for seed in (1,2,3):
            path=self.root/str(seed); fixture(path,seed=seed,reward=seed); rows+=independent_rows(path)
        stats=aggregate_seeds(pd.DataFrame(rows))
        reward=stats[stats.metric.str.endswith('/mean_reward_per_step')].iloc[0]
        self.assertEqual(reward.n_seeds,3); self.assertEqual(reward['mean'],2)
        self.assertLessEqual(reward.ci_low,2); self.assertGreaterEqual(reward.ci_high,2)
        one=aggregate_seeds(pd.DataFrame(independent_rows(self.root/'1')))
        self.assertTrue(one.ci_low.isna().all())
        with self.assertRaises(ValueError): aggregate_seeds(pd.DataFrame(rows+independent_rows(self.root/'1')))
        path=self.root/'train'; fixture(path,split='train',reward=100)
        split_stats=aggregate_seeds(pd.DataFrame(rows+independent_rows(path)))
        self.assertEqual(set(split_stats.split),{'train','val'})

    @unittest.skipUnless(all(importlib.util.find_spec(p) for p in ('pandas','matplotlib','pyarrow')), 'requires comparison extra')
    def test_report_distinct_checkpoints_at_same_seed_and_step(self):
        from fluidgym_rl.comparison.report import build_report
        import pandas as pd
        first=self.root/'best'; second=self.root/'final'
        fixture(first,step=50000,reward=1.)
        fixture(second,step=50000,reward=3.)
        before={str(p):file_hash(p) for root in (first,second) for p in root.iterdir()}
        build_report([first,second],[],self.root/'report')
        stats=pd.read_csv(self.root/'report/seed_summary.csv')
        rewards=stats[stats.metric.str.endswith('/mean_reward_per_step')]
        self.assertEqual(set(rewards['mean']),{1.,3.})
        self.assertTrue((rewards.n_seeds==1).all())
        self.assertTrue(rewards.ci_low.isna().all())
        self.assertEqual(rewards.series.nunique(),2)
        self.assertEqual(len(pd.read_parquet(self.root/'report/comparison.parquet')),24)
        self.assertEqual(before,{str(p):file_hash(p) for root in (first,second) for p in root.iterdir()})

    @unittest.skipUnless(all(importlib.util.find_spec(p) for p in ('pandas','matplotlib','pyarrow')), 'requires comparison extra')
    def test_report_exports_preserve_sources(self):
        from fluidgym_rl.comparison.report import build_report
        import pandas as pd
        path=self.root/'eval'; fixture(path)
        before={p.name:file_hash(p) for p in path.iterdir()}
        result=build_report([path],[],self.root/'report')
        self.assertEqual(before,{p.name:file_hash(p) for p in path.iterdir()})
        self.assertEqual(len(result['plots']),6)
        self.assertEqual(len(pd.read_parquet(self.root/'report/comparison.parquet')),12)
        self.assertTrue((self.root/'report/dashboard.html').is_file())
        with self.assertRaises(FileExistsError): build_report([path],[],self.root/'report')


if __name__=='__main__': unittest.main()
