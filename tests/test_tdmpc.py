"""CPU/synthetic verification only; real FluidGym checks are remote_required."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
import numpy as np
import torch
from gymnasium.spaces import Box

from fluidgym_rl.tdmpc.runtime import read_config, namespace, make_workcopy, load_algorithm, repo_root
from fluidgym_rl.tdmpc.environment import CheckedEnv
from fluidgym_rl.tdmpc.evaluation import Policy, evaluate, save_checkpoint, load_checkpoint
from fluidgym_rl.tdmpc.runner import finite_update, run_training


class FakeEnv:
    observation_space=Box(-np.inf,np.inf,(302,),np.float32)
    action_space=Box(-1,1,(1,),np.float32)
    episode_length=80
    def train(self): self.split='train'
    def val(self): self.split='val'
    def test(self): self.split='test'
    def seed(self,seed): pass
    def reset(self,**kwargs):
        self.t=0
        return np.zeros(302,np.float32),{}
    def step(self,a):
        self.t+=1
        return np.full(302,self.t/80,np.float32),0.1,False,self.t==80,dict(drag=3.1,lift=(-1)**self.t*.2)


class ContractTests(unittest.TestCase):
    def test_close_is_optional_and_preserves_original_failure(self):
        env=CheckedEnv(FakeEnv(),namespace(read_config(device='cpu')))
        env.close()
        with self.assertRaisesRegex(RuntimeError,'original failure'):
            try:
                raise RuntimeError('original failure')
            finally:
                env.close()

    def test_close_forwards_and_does_not_swallow_cleanup_failure(self):
        raw=FakeEnv()
        env=CheckedEnv(raw,namespace(read_config(device='cpu')))
        with patch.object(raw,'close',create=True) as close:
            env.close()
            close.assert_called_once_with()
        with patch.object(raw,'close',create=True,side_effect=RuntimeError('cleanup failure')):
            with self.assertRaisesRegex(RuntimeError,'cleanup failure'):
                env.close()

    def test_config_and_no_cuda_print(self):
        data=read_config(device='cpu',seed=9)
        self.assertEqual(data['parameters']['horizon_schedule'],'linear(1,20,50000)')
        self.assertEqual(data['parameters']['seed'],9)
        with patch('torch.cuda.is_available',side_effect=AssertionError('CUDA touched')):
            read_config(device='cuda')
        p=subprocess.run([sys.executable,str(repo_root()/'scripts/tdmpc_train.py'),'--print-config'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout)['parameters']['batch_size'],512)

    def test_full_gate_and_non_multiple_smoke_fail_before_creation(self):
        env=dict(os.environ); env.pop('FGRL_ALLOW_FULL_TRAINING',None)
        for args, expected in [(['--mode','full'],'TRAINING BLOCKED'),(['--mode','smoke','--smoke-steps','161'],'whole episodes')]:
            p=subprocess.run([sys.executable,str(repo_root()/'scripts/tdmpc_train.py'),*args],env=env,capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn(expected,p.stderr)
            self.assertNotIn('Run directory:',p.stdout)

    def test_env_boundary_dtype_and_abs_lift(self):
        cfg=namespace(read_config(device='cpu')); raw=FakeEnv(); env=CheckedEnv(raw,cfg)
        self.assertEqual(env.reset()[0].dtype,np.float32)
        for t in range(80):
            _,_,term,trunc,info=env.step(np.array([2],np.float32))
            self.assertFalse(term); self.assertEqual(trunc,t==79)
            self.assertAlmostEqual(info['abs_lift'],.2)
        with self.assertRaises(RuntimeError): env.step(np.zeros(1))
        env.reset()
        with patch.object(raw,'step',return_value=(np.zeros(302,np.float32),0,True,False,{})):
            with self.assertRaisesRegex(RuntimeError,'termination'): env.step(np.zeros(1))
        with patch.object(raw,'step',return_value=(np.zeros(302,np.float64),0,False,False,{})):
            with self.assertRaises(ValueError): env.step(np.zeros(1))

    def test_device_alias_without_cuda_initialization(self):
        from fluidgym_rl.device import same_device
        self.assertTrue(same_device('cpu',torch.device('cpu')))
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.current_device',return_value=1):
            self.assertTrue(same_device('cuda','cuda:1'))
            self.assertFalse(same_device('cuda','cuda:0'))

    @unittest.skip('remote_required: real FluidGym CUDA creation, CFD episode and CUDA aliases; run --mode validate then --mode smoke')
    def test_real_cuda_fluidgym(self): pass


class AlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=Path(os.environ.get('FGRL_TDMPC_SOURCE',repo_root()/'third_party/tdmpc'))
        if not source.exists():
            raise unittest.SkipTest('pinned original upstream unavailable; fetch with FGRL_UPSTREAMS=tdmpc scripts/fetch_upstream.sh')
        torch.set_num_threads(1)
        cls.tmp=tempfile.TemporaryDirectory()
        cls.work=Path(cls.tmp.name)/'work'
        cls.provenance=make_workcopy(cls.work,source)
        cls.Agent,cls.h=load_algorithm(cls.work)

    @classmethod
    def tearDownClass(cls):
        for k in list(sys.modules):
            if k=='algorithm' or k.startswith('algorithm.'):
                del sys.modules[k]
        cls.tmp.cleanup()

    def setUp(self):
        torch.manual_seed(2); np.random.seed(2)
        self.cfg=namespace(read_config(device='cpu'))
        # Explicit synthetic-test dimensions, never nominal config changes.
        self.cfg.enc_dim=16; self.cfg.mlp_dim=24; self.cfg.latent_dim=8
        self.cfg.horizon=3; self.cfg.horizon_schedule='3'; self.cfg.batch_size=8
        self.cfg.num_samples=16; self.cfg.num_elites=4; self.cfg.iterations=2
        self.cfg.train_steps=320; self.cfg.max_buffer_size=320

    def buffer(self):
        b=self.h.ReplayBuffer(self.cfg)
        for ep in range(2):
            episode=self.h.Episode(self.cfg,np.full(302,ep*100,np.float32))
            for t in range(80):
                episode.add(np.full(302,ep*100+t+1,np.float32),torch.zeros(1),.1,t==79)
            b.add(episode)
        return b

    def test_patch_provenance_and_algorithm_scope(self):
        self.assertEqual(self.provenance['upstream_sha'],'f4d85eca7419039b71bab2234ffc8aca378dd313')
        text=(repo_root()/'patches/tdmpc_device.patch').read_text()
        self.assertNotIn('tdmpc2',text)
        self.assertNotIn('weighted_loss',text)

    def test_episode_replay_per_sequences_and_weights(self):
        b=self.buffer()
        b.update_priorities(torch.tensor([0]),torch.tensor([[12.]]))
        self.assertGreater(b._priorities[0],b._priorities[1])
        for _ in range(10):
            obs,nxt,actions,rewards,idx,weights=b.sample()
            self.assertEqual(nxt.shape,(4,8,302))
            self.assertTrue(torch.all(idx%80<77))
            self.assertTrue(torch.all(weights>0)); self.assertLessEqual(weights.max(),1)
            self.assertTrue(torch.equal(nxt[0,:,0],obs[:,0]+1))
            self.assertTrue(torch.equal(nxt[2,:,0],obs[:,0]+3))
        probs=b._priorities[:b.idx]**self.cfg.per_alpha; probs/=probs.sum()
        expected=(b.idx*probs[idx])**(-self.cfg.per_beta); expected/=expected.max()
        torch.testing.assert_close(weights,expected)

    def test_terminal_observation_and_full_buffer_sampling(self):
        b=self.buffer()
        b._priorities.zero_(); b._priorities[76]=1
        obs,nxt,_,_,idx,_=b.sample()
        self.assertTrue(torch.all(idx==76))
        self.assertTrue(torch.all(nxt[-1,:,0]==80))
        # Four complete episodes fill the native ring; valid starts exceed batch size.
        b=self.buffer()
        for ep in range(2):
            episode=self.h.Episode(self.cfg,np.zeros(302,np.float32))
            for t in range(80):
                episode.add(np.zeros(302,np.float32),torch.zeros(1),0,t==79)
            b.add(episode)
        self.assertTrue(b._full)
        batch=b.sample()
        self.assertEqual(len(torch.unique(batch[-2])),self.cfg.batch_size)

    def test_schedule_endpoints(self):
        self.assertEqual(self.h.linear_schedule('linear(1,20,50000)',0),1)
        self.assertEqual(self.h.linear_schedule('linear(1,20,50000)',50000),20)

    def test_forward_loss_gradients_optimizer_and_planner(self):
        agent=self.Agent(self.cfg); b=self.buffer()
        before=copy.deepcopy(agent.model.state_dict())
        z=agent.model.h(torch.zeros(2,302)); nz,r=agent.model.next(z,torch.zeros(2,1))
        self.assertEqual(nz.shape,(2,8)); self.assertEqual(r.shape,(2,1))
        metrics=finite_update(agent,b,160)
        self.assertEqual(set(metrics),{'consistency_loss','reward_loss','value_loss','pi_loss','total_loss','weighted_loss','grad_norm'})
        self.assertTrue(any(not torch.equal(v,agent.model.state_dict()[k]) for k,v in before.items()))
        for t in range(2):
            a=agent.plan(np.zeros(302,np.float32),eval_mode=True,step=160,t0=t==0)
            self.assertTrue(torch.isfinite(a).all()); self.assertLessEqual(a.abs().max(),1)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in agent.model.parameters()))

    def test_checkpoint_and_shared_evaluator(self):
        agent=self.Agent(self.cfg); agent.std=.123
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'final.pt'; save_checkpoint(agent,path,self.cfg,160,{})
            other=self.Agent(self.cfg); load_checkpoint(other,path)
            self.assertEqual(other.std,agent.std)
            first,second=Policy(agent,160),Policy(other,160)
            first.reset(7); a=first.act(np.zeros(302,np.float32),True)
            second.reset(7); b=second.act(np.zeros(302,np.float32),True)
            torch.testing.assert_close(a,b)
            for k,v in agent.model_target.state_dict().items(): torch.testing.assert_close(v,other.model_target.state_dict()[k])
            for k,v in agent.model.state_dict().items(): torch.testing.assert_close(v,other.model.state_dict()[k])
            env=CheckedEnv(FakeEnv(),self.cfg)
            result=evaluate(env,Policy(other,160),self.cfg,Path(tmp)/'evaluation',split='test',episodes=2,seed=5)
            self.assertAlmostEqual(result['mean_reward_per_step']['mean'],.1)
            self.assertAlmostEqual(result['episode_return']['mean'],8)
            self.assertEqual(result['truncated_episodes'],2)
            self.assertAlmostEqual(result['metrics_per_step_mean']['abs_lift']['mean'],.2)
            self.assertIsNone(result['drag_reduction_percent'])
            manifest=json.loads((Path(tmp)/'evaluation/run_manifest.json').read_text())
            self.assertEqual(manifest['contract']['split'],'test')
            child=subprocess.run([sys.executable,str(repo_root()/'scripts/tdmpc_train.py'),'--checkpoint',str(path),'--verify-checkpoint','--device','cpu','--output-root',str(Path(tmp)/'fresh')],capture_output=True,text=True)
            self.assertEqual(child.returncode,0,child.stdout+child.stderr)

    def test_bounded_synthetic_harness(self):
        cfg=self.cfg; cfg.seed_steps=100
        agent=self.Agent(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            tracker=Mock()
            result=run_training(cfg,agent,self.h,CheckedEnv(FakeEnv(),cfg),out,{},budget=160,smoke=True,tracker=tracker)
            calls=tracker.log.call_args_list
            self.assertEqual([c.args[0] for c in calls],['train','train','val','checkpoint'])
            self.assertEqual(calls[0].args[1]['learner_updates'],0)
            self.assertNotIn('value_loss',calls[0].args[1])
            self.assertEqual(calls[1].args[1]['learner_updates'],100)
            self.assertEqual(calls[2].args[1]['environment_steps'],160)
            self.assertEqual(result['environment_steps'],160)
            self.assertEqual(result['learner_updates'],100)
            self.assertTrue((out/'checkpoints/best.pt').exists())
            self.assertTrue((out/'checkpoints/final.pt').exists())
            self.assertTrue((out/'eval.csv').exists())
            self.assertEqual(result['validation'][0]['step'],160)


if __name__=='__main__': unittest.main()
