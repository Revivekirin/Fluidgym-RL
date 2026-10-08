"""CPU tests of the common evaluation contract using a stand-in FluidGym-like env (interface tests, not simulator evidence)."""
import json
import numpy as np
import gymnasium as gym
import pytest

from fluidgym_rl.evalcontract import (EvalContract, EvaluationError, classify_comparison, compare_to_reference, new_run_dir, run_evaluation, summarize, write_run)


class FakeEnv:
    episode_length = 4
    metrics = ["drag", "lift"]

    def __init__(self, nan_obs_at=None):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (3,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (1,), np.float32)
        self.mode, self._n, self.resets, self.nan_obs_at, self._rng = "train", 0, [], nan_obs_at, np.random.default_rng(0)

    def test(self): self.mode = "test"
    def seed(self, s): self._rng = np.random.default_rng(s)

    def reset(self, seed=None, randomize=None):
        self.resets.append({"seed": seed, "randomize": randomize}); self._n = 0
        if seed is not None: self.seed(seed)
        return self._rng.normal(size=3).astype(np.float32), {}

    def step(self, a):
        self._n += 1
        obs = self._rng.normal(size=3).astype(np.float32)
        if self.nan_obs_at == self._n: obs[0] = np.nan
        return obs, float(-abs(a[0])), False, self._n >= self.episode_length, {"drag": np.array(3.1), "lift": np.array(0.01 * self._n)}


class ConstPolicy:
    name, mode = "const", "deterministic"
    def __init__(self, v=0.5): self.v = v
    def reset(self, episode_seed): pass
    def act(self, obs, t0): return np.array([self.v], np.float32)


def _run(env=None, policy=None, **kw):
    c = EvalContract(n_episodes=3, **kw); env = env or FakeEnv()
    return env, run_evaluation(env, policy or ConstPolicy(), c, metric_names=["drag", "lift"], action_low=[-1], action_high=[1])


def test_official_protocol_first_episode_not_randomized_and_uses_test_split():
    env, recs = _run()
    assert env.mode == "test" and [r["randomize"] for r in env.resets] == [False, True, True] and all(r["seed"] is None for r in env.resets)
    assert [r.length for r in recs] == [4, 4, 4] and all(r.truncated and not r.terminated for r in recs)


def test_seeded_protocol_uses_explicit_seeds():
    env, _ = _run(protocol="seeded", base_seed=10)
    assert [r["seed"] for r in env.resets] == [10, 11, 12]


def test_evaluation_is_deterministic_for_same_contract():
    _, a = _run(); _, b = _run()
    assert [r.ret for r in a] == [r.ret for r in b] and [r.metrics for r in a] == [r.metrics for r in b]


def test_actions_clipped_and_recorded_scalar_rewards_and_metrics():
    _, recs = _run(policy=ConstPolicy(5.0))
    r = recs[0]
    assert r.action_abs_max == 1.0 and r.action_clipped_frac == 1.0 and r.ret == pytest.approx(-4.0) and r.metrics["drag"] == pytest.approx(3.1)
    s = summarize(recs)
    assert s["mean_reward_per_step"]["mean"] == pytest.approx(-1.0) and "drag" in s["metrics_per_step_mean"] and "drag_reduction" not in json.dumps(s)


def test_nonfinite_observation_and_action_fail_loudly():
    with pytest.raises(EvaluationError): _run(env=FakeEnv(nan_obs_at=2))
    with pytest.raises(EvaluationError): _run(policy=ConstPolicy(float("nan")))


def test_missing_metric_is_an_error_not_a_silent_skip():
    env = FakeEnv(); env.metrics = ["drag"]
    class E2(FakeEnv):
        def step(self, a):
            o, r, t, tr, info = super().step(a); info.pop("lift"); return o, r, t, tr, info
    with pytest.raises(EvaluationError, match="lift"):
        run_evaluation(E2(), ConstPolicy(), EvalContract(n_episodes=1), metric_names=["drag", "lift"], action_low=[-1], action_high=[1])


def test_invalid_contract_rejected():
    for kw in ({"protocol": "x"}, {"split": "dev"}, {"n_episodes": 0}):
        with pytest.raises(ValueError): EvalContract(**kw)


def test_run_dirs_never_overwritten(tmp_path):
    a, b = new_run_dir(tmp_path, "t"), new_run_dir(tmp_path, "t")
    assert a != b and a.exists() and b.exists()


def test_write_run_produces_required_files(tmp_path):
    _, recs = _run(); d = new_run_dir(tmp_path, "x")
    write_run(d, {"c": 1}, recs, summarize(recs), {"m": 1})
    assert {p.name for p in d.iterdir()} >= {"resolved_config.yaml", "episodes.csv", "summary.json", "run_manifest.json"}
    assert len((d / "episodes.csv").read_text().strip().splitlines()) == 4


def _man(**kw):
    base = {"contract": {"env_id": "E", "split": "test", "protocol": "official_callback", "flatten_observation": True, "n_episodes": 10, "base_seed": 0},
            "algorithm": "sac", "checkpoint_sha256": "aa", "train_seed": 0, "simulator_version": "0.1.2"}
    base.update(kw); return base


def test_comparison_classes():
    assert classify_comparison(_man(), _man())["class"] == "same_checkpoint_same_contract"
    assert classify_comparison(_man(), _man(simulator_version="0.0.2"))["class"] == "same_environment_different_implementation_version"
    assert classify_comparison(_man(), _man(checkpoint_sha256="bb", train_seed=1))["class"] == "same_algorithm_different_training_seed"
    assert classify_comparison(_man(), _man(algorithm="tdmpc2"))["class"] == "cross_algorithm_same_contract"
    bad = _man(); bad["contract"] = {**bad["contract"], "split": "val"}
    assert classify_comparison(_man(), bad)["class"] == "non_comparable"
    bad2 = _man(); bad2["contract"] = {**bad2["contract"], "env_id": "Other"}
    assert classify_comparison(_man(), bad2)["class"] == "non_comparable"


def test_reference_comparison_never_claims_reproduction():
    _, recs = _run(); out = compare_to_reference(summarize(recs), {"source": "paper", "mean_reward": 0.05})
    assert out["claims_reproduction"] is False and out["rows"][0]["abs_diff"] == pytest.approx(-0.5 - 0.05)
