"""Adapter exposing a FluidGym env through the *inner* env contract TD-MPC2 expects.

TD-MPC2 (envs/__init__.py::make_env) wraps task envs in TensorWrapper, which calls
``env.reset()`` -> obs and ``env.step(np_action)`` -> (obs, reward, done, info) with
info['success'] and info['terminated'], and needs ``max_episode_steps``.

Termination semantics: FluidGym returns (terminated, truncated). TD-MPC2 only has `done`,
so done = terminated or truncated, and info['terminated'] carries the *true* terminal flag
so time-limit truncation is not treated as a terminal state.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

TASK_PREFIX = "fluidgym-"


class FluidGymTDMPC2Env(gym.Env):
    def __init__(self, env, seed: int | None = None):
        super().__init__()
        self._env = env
        self._torch_backend = hasattr(env, "cuda_device")  # real FluidGym env (torch tensors on CUDA)
        obs_space, act_space = env.observation_space, env.action_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise ValueError("Wrap the env with FlattenObservation first (Box observation space required).")
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, obs_space.shape, np.float32)
        self.action_space = gym.spaces.Box(act_space.low.astype(np.float32), act_space.high.astype(np.float32), dtype=np.float32)
        self._low, self._high = self.action_space.low, self.action_space.high
        self.max_episode_steps = int(env.episode_length)
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)

    def _to_np(self, x) -> np.ndarray:
        return (x.detach().cpu().numpy() if self._torch_backend else np.asarray(x)).astype(np.float32)

    def _to_env(self, a: np.ndarray):
        a = np.clip(np.asarray(a, dtype=np.float32).reshape(self.action_space.shape), self._low, self._high)
        if not self._torch_backend:
            return a
        import torch
        return torch.as_tensor(a, device=self._env.cuda_device)

    def reset(self, seed: int | None = None, **_):
        obs, _info = self._env.reset(seed=seed)
        return self._to_np(obs)

    def step(self, action):
        obs, reward, terminated, truncated, raw = self._env.step(self._to_env(action))
        info = {"success": 0.0, "terminated": float(bool(terminated)), "truncated": float(bool(truncated))}
        for k, v in raw.items():  # pass physical metrics (e.g. drag/lift) through when scalar
            arr = self._to_np(v) if hasattr(v, "shape") else np.asarray(v)
            if arr.size == 1:
                info[k] = float(arr.reshape(-1)[0])
        return self._to_np(obs), float(reward), bool(terminated or truncated), info

    def rand_act(self):
        return self.action_space.sample()

    def render(self, *a, **k):
        return self._env.render(*a, **k)


def make_env(cfg):
    """Factory with TD-MPC2's make_env signature. Task name: 'fluidgym-<EnvId>' e.g. fluidgym-CylinderJet2D-easy-v0."""
    task = str(cfg.task)
    if not task.startswith(TASK_PREFIX):
        raise ValueError(f"Unknown task {task}")
    from fluidgym_rl.envs import make_fluid_env

    return FluidGymTDMPC2Env(make_fluid_env(task[len(TASK_PREFIX):]), seed=cfg.get("seed", None))
