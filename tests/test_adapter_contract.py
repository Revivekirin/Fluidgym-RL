"""CPU-only contract tests for the TD-MPC2 adapter, using a stand-in env that mimics the
FluidGym API (reset(seed)->(obs,info); step->(obs,r,terminated,truncated,info); episode_length).
These verify OUR adapter logic only; they say nothing about FluidGym physics."""
import gymnasium as gym
import numpy as np
import pytest

from fluidgym_rl.tdmpc2_adapter import FluidGymTDMPC2Env


class FakeFluidEnv:
    episode_length = 5

    def __init__(self, terminate_at=None):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (6,), np.float64)
        self.action_space = gym.spaces.Box(-1, 1, (1,), np.float32)
        self.terminate_at, self._rng, self._n = terminate_at, np.random.default_rng(0), 0

    def seed(self, s):
        self._rng = np.random.default_rng(s)

    def reset(self, seed=None, randomize=None):
        if seed is not None:
            self.seed(seed)
        self._n = 0
        return self._rng.normal(size=6), {}

    def step(self, a):
        if self._n >= self.episode_length:
            raise RuntimeError("Episode has already terminated.")
        self._n += 1
        term = self.terminate_at == self._n
        return self._rng.normal(size=6), float(-abs(a[0])), term, self._n >= self.episode_length, {"cd": np.array(3.2), "field": np.zeros(4)}


def test_shapes_dtypes_and_spaces():
    env = FluidGymTDMPC2Env(FakeFluidEnv())
    obs = env.reset()
    assert obs.shape == (6,) and obs.dtype == np.float32
    assert env.observation_space.shape == (6,) and env.action_space.shape == (1,)
    assert env.max_episode_steps == 5
    o, r, done, info = env.step(env.rand_act())
    assert o.dtype == np.float32 and isinstance(r, float) and isinstance(done, bool)
    assert {"success", "terminated"} <= info.keys()


def test_truncation_is_done_but_not_terminated():
    env = FluidGymTDMPC2Env(FakeFluidEnv())
    env.reset()
    for t in range(5):
        _, _, done, info = env.step(np.zeros(1, np.float32))
    assert done and info["terminated"] == 0.0 and info["truncated"] == 1.0
    with pytest.raises(RuntimeError):
        env.step(np.zeros(1, np.float32))


def test_true_termination_flag_preserved():
    env = FluidGymTDMPC2Env(FakeFluidEnv(terminate_at=3))
    env.reset()
    for _ in range(3):
        _, _, done, info = env.step(np.zeros(1, np.float32))
    assert done and info["terminated"] == 1.0


def test_scalar_metrics_pass_through_non_scalars_dropped():
    env = FluidGymTDMPC2Env(FakeFluidEnv())
    env.reset()
    _, _, _, info = env.step(np.zeros(1, np.float32))
    assert info["cd"] == pytest.approx(3.2) and "field" not in info


def test_seeded_reset_is_reproducible():
    a, b = FluidGymTDMPC2Env(FakeFluidEnv(), seed=7), FluidGymTDMPC2Env(FakeFluidEnv(), seed=7)
    np.testing.assert_array_equal(a.reset(), b.reset())
    np.testing.assert_array_equal(a.reset(seed=3), b.reset(seed=3))


def test_dict_obs_space_rejected():
    e = FakeFluidEnv()
    e.observation_space = gym.spaces.Dict({"velocity": gym.spaces.Box(-1, 1, (2,))})
    with pytest.raises(ValueError):
        FluidGymTDMPC2Env(e)


def test_out_of_range_actions_are_clipped_to_bounds():
    class Rec(FakeFluidEnv):
        def step(self, a):
            self.last = np.array(a); return super().step(a)
    e = Rec(); env = FluidGymTDMPC2Env(e); env.reset()
    env.step(np.array([9.0], np.float32)); assert e.last[0] == 1.0
    env.step(np.array([-9.0], np.float32)); assert e.last[0] == -1.0
