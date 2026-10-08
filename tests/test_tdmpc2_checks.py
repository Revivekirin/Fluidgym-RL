import numpy as np
import pytest

from fluidgym_rl.tdmpc2_checks import (NonFiniteLossError, ReplayValidationError, check_losses, diff_digests, state_digests, tensor_digest, validate_actions,
                                       validate_sampled_batch, validate_stored_episodes)

LOSSES = {k: np.float32(0.5) for k in ("consistency_loss", "reward_loss", "value_loss", "pi_loss", "total_loss", "grad_norm")}


def test_finite_losses_pass_and_missing_or_nan_detected():
    assert check_losses(LOSSES)["total_loss"] == 0.5
    with pytest.raises(NonFiniteLossError, match="missing"): check_losses({k: v for k, v in LOSSES.items() if k != "pi_loss"})
    with pytest.raises(NonFiniteLossError, match="non-finite"): check_losses({**LOSSES, "value_loss": np.float32("nan")})
    with pytest.raises(NonFiniteLossError): check_losses({**LOSSES, "grad_norm": np.float32("inf")})


def _stored(n_eps=3, L=5, D=4, A=1):
    obs = np.random.default_rng(0).normal(size=(n_eps * L, D)); act = np.random.default_rng(1).uniform(-1, 1, (n_eps * L, A)); rew = np.random.default_rng(2).normal(size=n_eps * L)
    act[::L], rew[::L] = np.nan, np.nan
    return obs, act, rew, np.repeat(np.arange(n_eps), L)


def test_stored_episodes_valid_and_invalid_variants():
    o, a, r, e = _stored()
    assert validate_stored_episodes(o, a, r, e, episode_length=4, obs_dim=4, act_dim=1, n_episodes=3)["rows"] == 15
    a2 = a.copy(); a2[0] = 0.0
    with pytest.raises(ReplayValidationError, match="NaN"): validate_stored_episodes(o, a2, r, e, episode_length=4, obs_dim=4, act_dim=1, n_episodes=3)
    e2 = e.copy(); e2[7] = 0
    with pytest.raises(ReplayValidationError, match="episode ids"): validate_stored_episodes(o, a, r, e2, episode_length=4, obs_dim=4, act_dim=1, n_episodes=3)
    with pytest.raises(ReplayValidationError, match="shapes"): validate_stored_episodes(o[:-1], a[:-1], r[:-1], e[:-1], episode_length=4, obs_dim=4, act_dim=1, n_episodes=3)
    o3 = o.copy(); o3[3, 0] = np.inf
    with pytest.raises(ReplayValidationError, match="non-finite"): validate_stored_episodes(o3, a, r, e, episode_length=4, obs_dim=4, act_dim=1, n_episodes=3)


def _batch(H=3, B=6, D=4, A=1):
    return (np.zeros((H + 1, B, D)), np.zeros((H, B, A)), np.zeros((H, B, 1)), np.zeros((H, B, 1)), np.tile(np.arange(B), (H + 1, 1)))


def test_sampled_batch_detects_cross_episode_sequences_and_nan_leaks():
    o, a, r, t, e = _batch()
    assert validate_sampled_batch(o, a, r, t, e, horizon=3, batch=6, obs_dim=4, act_dim=1)["distinct_episodes_in_batch"] == 6
    e2 = e.copy(); e2[2, 1] = 99
    with pytest.raises(ReplayValidationError, match="more than one episode"): validate_sampled_batch(o, a, r, t, e2, horizon=3, batch=6, obs_dim=4, act_dim=1)
    a2 = a.copy(); a2[0, 0, 0] = np.nan
    with pytest.raises(ReplayValidationError, match="non-finite"): validate_sampled_batch(o, a2, r, t, e, horizon=3, batch=6, obs_dim=4, act_dim=1)
    with pytest.raises(ReplayValidationError, match="shape"): validate_sampled_batch(o[:-1], a, r, t, e, horizon=3, batch=6, obs_dim=4, act_dim=1)


def test_planned_actions_must_be_legal():
    assert validate_actions([np.array([0.2]), np.array([-1.0])], 1)["n"] == 2
    for bad in ([np.array([1.5])], [np.array([np.nan])], [np.array([0.1, 0.2])]):
        with pytest.raises(AssertionError): validate_actions(bad, 1)


def test_checkpoint_digests_roundtrip_and_detect_tampering(tmp_path):
    sd = {"w": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.ones(3, np.float32)}
    np.savez(tmp_path / "c.npz", **sd); loaded = dict(np.load(tmp_path / "c.npz"))
    assert diff_digests(state_digests(sd), state_digests(loaded)) == {"missing_in_loaded": [], "unexpected_in_loaded": [], "mismatched": [], "n_compared": 2}
    loaded["w"] = loaded["w"] + 1e-7
    assert diff_digests(state_digests(sd), state_digests(loaded))["mismatched"] == ["w"]
    assert tensor_digest(np.zeros(3, np.float32)) != tensor_digest(np.zeros(3, np.float64))
