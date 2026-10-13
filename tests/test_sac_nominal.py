"""CPU stand-in tests only; no FluidGym/CUDA success is implied."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import gymnasium as gym

HAVE_SB3=importlib.util.find_spec('stable_baselines3') is not None

class NativeFixture:
    episode_length=80
    observation_space=gym.spaces.Box(-np.inf,np.inf,(302,),dtype=np.float32)
    action_space=gym.spaces.Box(-1,1,(1,),dtype=np.float32)
    _cd_ref=4.; _lift_penalty=1.
    def reset(self,seed=None,randomize=True):
        self.t=0; self.rng=np.random.default_rng(seed)
        return self.rng.normal(0,.1,302).astype(np.float32),{}
    def step(self,action):
        self.t+=1; drag=3.+float(action[0])**2; lift=float(action[0])*.1
        return self.rng.normal(0,.1,302).astype(np.float32),4.-drag-abs(lift),False,self.t==80,dict(drag=drag,lift=lift)
    def train(self): pass
    def val(self): pass
    def test(self): pass


@unittest.skipUnless(HAVE_SB3,'SB3 extra unavailable')
class SACTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        import torch
        self.threads=torch.get_num_threads(); torch.set_num_threads(1)
    def tearDown(self):
        import torch
        torch.set_num_threads(self.threads); self.tmp.cleanup()

    def test_adapter_timeout_and_actions(self):
        from stable_baselines3.common.env_checker import check_env
        from fluidgym_rl.sac.environment import SACEnv
        env=SACEnv(NativeFixture(),device='cpu')
        check_env(env)
        a,_=env.reset(seed=7); b,_=env.reset(seed=7)
        np.testing.assert_array_equal(a,b)
        for i in range(80): _,_,terminated,truncated,_=env.step(np.zeros(1,np.float32))
        self.assertFalse(terminated); self.assertTrue(truncated)
        with self.assertRaises(ValueError): env.step(np.array([2.]))

    def test_actual_sb3_learning_checkpoint_and_shared_validation(self):
        import torch
        from stable_baselines3 import SAC
        from fluidgym_rl.sac.runtime import read_config,load_checkpoint,sb3_audit
        from fluidgym_rl.sac.environment import SACEnv
        from fluidgym_rl.sac.training import NominalCallback
        from fluidgym_rl.comparison.evaluate import aggregate,training_identity
        from fluidgym_rl.comparison.schema import digest,file_hash,write_json,verify_result
        from fluidgym_rl.evalcontract import EvalContract,run_evaluation,write_run
        from fluidgym_rl.official_sac import SB3Policy
        config=read_config(steps=240,device='cpu',eval_freq=160,eval_episodes=1)
        config['sac']['buffer_size']=512  # CPU fixture only; nominal config remains 1e6.
        env=SACEnv(NativeFixture(),device='cpu')
        model=SAC('MlpPolicy',env,device='cpu',seed=0,**config['sac'])
        self.assertEqual(sb3_audit()['version'],__import__('stable_baselines3').__version__)
        self.assertEqual([layer.out_features for layer in model.actor.latent_pi if isinstance(layer,torch.nn.Linear)],[256,256])
        physics={'fixture':True}
        def evaluate(ck,out,cfg,root,reference):
            out.mkdir(); restored,_=load_checkpoint(ck)
            c=EvalContract(split='val',protocol='seeded',n_episodes=1)
            val=SACEnv(NativeFixture(),device='cpu').inner
            records=run_evaluation(val,SB3Policy(restored),c,metric_names=['drag','lift','abs_lift'],action_low=[-1],action_high=[1])
            summary=aggregate(records)
            m={**training_identity('sac',root,ck),'schema_version':1,'status':'completed','algorithm':'sac',
                'contract':asdict(c),'physics':physics,'checkpoint_role':'candidate',
                'cohort':digest({'contract':{k:v for k,v in asdict(c).items() if k!='split'},'physics':physics})}
            write_run(out,{},records,summary,m)
            m['result_hashes']={n:file_hash(out/n) for n in ('episodes.csv','summary.json','resolved_config.yaml')}
            write_json(out/'evaluation_manifest.json',m)
            return summary
        tracker=SimpleNamespace(log=lambda *a:None)
        cb=NominalCallback(self.root,config,physics,tracker,evaluator=evaluate)
        try:
            model.learn(240,callback=cb)
            result=cb.finish()
        finally: cb.gradients.close()
        self.assertEqual(env.real_steps,240)
        self.assertEqual(result['learner_updates'],140)
        self.assertEqual(result['checks']['finite_gradients'],'passed')
        self.assertEqual(result['checks']['checkpoint_roundtrip'],'passed')
        self.assertEqual(len(cb.vals),2)
        import subprocess,sys
        subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/sac_train.py'),
                        '--verify-checkpoint',str(self.root/'checkpoints/final.zip')],check=True,capture_output=True)
        for p in cb.vals: self.assertEqual(verify_result(p)[0]['contract']['split'],'val')
        restored,sidecar=load_checkpoint(self.root/'checkpoints/best.zip')
        self.assertEqual(sidecar['sha256'],file_hash(self.root/'checkpoints/best.zip'))
        with self.assertRaises(ValueError): training_identity('sac',self.root,self.root/'checkpoints/best.zip',123)
        # Read-only public-loader audit succeeds on this new fixture archive; no physics claim.
        subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/audit_sac_checkpoint.py'),
                        str(self.root/'checkpoints/final.zip'),'--out',str(self.root/'loader_audit.json')],check=True,capture_output=True)
        self.assertIn('physics/preprocessing/protocol remain unverified',json.loads((self.root/'loader_audit.json').read_text())['status'])
        write_json(self.root/'summary.json',result)
        import tarfile
        with tempfile.TemporaryDirectory() as package_dir:
            archive=Path(package_dir)/'results.tar.gz'
            subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/package_sac_results.py'),
                            '--run',str(self.root),'--out',str(archive)],check=True,capture_output=True)
            with tarfile.open(archive) as tar:
                manifest=json.load(tar.extractfile('PACKAGE_MANIFEST.json'))
                self.assertEqual(manifest['file_sha256']['training/checkpoints/final.zip'],file_hash(self.root/'checkpoints/final.zip'))
                self.assertFalse(manifest['independent_test_included'])

    def test_tracking_uses_new_run_and_explicit_axis(self):
        from unittest.mock import Mock,patch
        from fluidgym_rl.sac.training import Tracking
        import os
        run=Mock(); run.id='new-run'; sdk=Mock(); sdk.init.return_value=run
        settings=dict(mode='offline',project='fluidgym-rl',entity=None,group='sac-nominal',tags=['sac'])
        with patch.dict('sys.modules',{'wandb':sdk}),patch.dict(os.environ,{'WANDB_RUN_ID':'original','WANDB_TAGS':''}):
            tracker=Tracking(self.root,settings,{},{}); tracker.log(80,{'compare/val/drag':3.,'missing':None})
            self.assertNotEqual(sdk.init.call_args.kwargs['id'],'original')
            self.assertEqual(sdk.init.call_args.kwargs['project'],'fluidgym-rl')
            self.assertEqual(run.log.call_args.args[0],{'compare/environment_steps':80,'compare/val/drag':3.})
            self.assertEqual(os.environ['WANDB_TAGS'],'')
            tracker.finish({'status':'completed'},0)
            run.finish.assert_called_once_with(exit_code=0)

    def test_rng_preservation(self):
        import torch,random
        from fluidgym_rl.sac.training import preserve_rng
        random.seed(5); np.random.seed(5); torch.manual_seed(5)
        with preserve_rng(): random.random(); np.random.rand(); torch.rand(1)
        observed=(random.random(),np.random.rand(),torch.rand(1))
        random.seed(5); np.random.seed(5); torch.manual_seed(5)
        expected=(random.random(),np.random.rand(),torch.rand(1))
        self.assertEqual(observed,expected)

    def test_config_and_gradient_failure(self):
        from fluidgym_rl.sac.runtime import read_config
        from fluidgym_rl.sac.training import GradientChecks
        import torch
        with self.assertRaises(ValueError): read_config(steps=50)
        checks=object.__new__(GradientChecks); checks.counts={'actor':0}; checks.norms={}
        with self.assertRaises(FloatingPointError): checks.hook('actor')(torch.tensor(float('nan')))
        self.assertIsNone(checks.hook('actor')(torch.ones(2)))


if __name__=='__main__': unittest.main()
