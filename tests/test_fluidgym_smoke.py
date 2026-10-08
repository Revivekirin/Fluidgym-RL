"""Smoke tests against the real FluidGym env. Skipped unless CUDA + fluidgym are present.
(NOT executed in the authoring sandbox: it has no GPU.)"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fluidgym")
pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

ENV_ID = "CylinderJet2D-easy-v0"


@pytest.fixture(scope="module")
def env():
    from fluidgym_rl.envs import make_fluid_env
    return make_fluid_env(ENV_ID)


def test_reset_step_shapes_finite(env):
    obs, _ = env.reset(seed=0)
    assert torch.isfinite(obs).all() and obs.ndim == 1
    obs2, r, term, trunc, _ = env.step(env.sample_action())
    assert obs2.shape == obs.shape and torch.isfinite(obs2).all() and torch.isfinite(r).all()


def test_truncates_at_episode_length(env):
    env.reset(seed=0)
    for i in range(env.episode_length):
        _, _, term, trunc, _ = env.step(torch.zeros_like(env.sample_action()))
    assert trunc and not term
    with pytest.raises(RuntimeError):
        env.step(torch.zeros_like(env.sample_action()))


def test_same_seed_same_trajectory(env):
    runs = []
    for _ in range(2):
        env.reset(seed=123)
        runs.append(torch.stack([env.step(torch.full_like(env.sample_action(), 0.3))[0] for _ in range(5)]))
    assert torch.allclose(runs[0], runs[1], atol=1e-5)


def test_tdmpc2_adapter_on_real_env():
    from fluidgym_rl.envs import make_fluid_env
    from fluidgym_rl.tdmpc2_adapter import FluidGymTDMPC2Env
    e = FluidGymTDMPC2Env(make_fluid_env(ENV_ID), seed=0)
    o = e.reset()
    assert o.dtype == np.float32 and np.isfinite(o).all()
    o, r, done, info = e.step(e.rand_act())
    assert np.isfinite(o).all() and np.isfinite(r)
