"""Smoke replay sizing, runnable without Torch, NumPy, or pytest."""
import unittest
from types import SimpleNamespace

from fluidgym_rl.tdmpc2_runtime import smoke_replay_config


class SmokeReplayBudgetTests(unittest.TestCase):
    def test_reset_rows_fit_without_extending_training(self):
        cfg = SimpleNamespace(steps=240, episode_length=80, buffer_size=1000000,
                              seed_steps=160, horizon=3, batch_size=256)
        replay = smoke_replay_config(cfg)
        self.assertEqual(min(replay.buffer_size, replay.steps), 243)
        self.assertEqual(cfg.steps, 240)
        self.assertEqual(cfg.seed_steps, 160)
        self.assertEqual((replay.horizon, replay.batch_size), (3, 256))
        self.assertIsNot(replay, cfg)

    def test_small_buffer_and_other_episode_counts(self):
        for length, episodes in ((1, 3), (5, 4), (80, 5)):
            with self.subTest(length=length, episodes=episodes):
                cfg = SimpleNamespace(steps=length * episodes, episode_length=length, buffer_size=1)
                replay = smoke_replay_config(cfg)
                self.assertEqual(min(replay.buffer_size, replay.steps), (length + 1) * episodes)
                self.assertEqual(cfg.buffer_size, 1)
                self.assertEqual(cfg.steps, length * episodes)

    def test_incomplete_or_invalid_budget_rejected(self):
        for length, steps in ((80, 241), (0, 240), (80, 0), (80, -80), (80, 240.0)):
            with self.subTest(length=length, steps=steps), self.assertRaises(ValueError):
                smoke_replay_config(SimpleNamespace(steps=steps, episode_length=length, buffer_size=1000))


if __name__ == '__main__':
    unittest.main()
