"""CPU audit probes. Fake transitions do not establish CFD equivalence."""
from types import SimpleNamespace
import unittest

import numpy as np
from gymnasium.spaces import Box
from fluidgym_rl.tdmpc.environment import CheckedEnv
from fluidgym_rl.tdmpc2_adapter import FluidGymTDMPC2Env


class FakeEnv:
    episode_length = 80
    observation_space = Box(-np.inf, np.inf, (302,), dtype=np.float32)
    action_space = Box(-1, 1, (1,), dtype=np.float32)
    def seed(self, seed): self.seed_value = seed
    def reset(self, seed=None, randomize=None):
        self.reset_args = (seed, randomize)
        self.t = 0
        self.x = np.zeros(302, np.float32)
        return self.x.copy(), {}
    def step(self, action):
        self.t += 1
        self.x += float(action[0])
        return self.x.copy(), float(action[0]), False, self.t == 80, {'drag':3.,'lift':-.2}


class AuditTests(unittest.TestCase):
    def test_same_command_sequence_wrapper_io(self):
        left = CheckedEnv(FakeEnv(), SimpleNamespace(obs_shape=[302],action_dim=1,episode_length=80))
        right = FluidGymTDMPC2Env(FakeEnv(), seed=1)
        np.testing.assert_array_equal(left.reset(seed=1)[0], right.reset(seed=1))
        for i in range(80):
            action=np.array([(-1)**i * 1.1],np.float32)
            a,r,term,trunc,info=left.step(action)
            b,s,done,other=right.step(action)
            np.testing.assert_array_equal(a,b)
            self.assertEqual(r,s)
            self.assertEqual(done,term or trunc)
            self.assertEqual(other['terminated'],float(term))
            self.assertEqual(info['drag'],other['drag'])
            self.assertEqual(info['lift'],other['lift'])

    def test_tdmpc2_reset_drops_randomize_and_box_seed_is_unset(self):
        source=FakeEnv(); wrapped=FluidGymTDMPC2Env(source,seed=1)
        wrapped.reset(seed=2,randomize=False)
        self.assertEqual(source.reset_args,(2,None))
        self.assertIsNone(wrapped.action_space._np_random)


if __name__=='__main__': unittest.main()
