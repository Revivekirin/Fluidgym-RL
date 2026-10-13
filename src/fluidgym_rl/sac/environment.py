"""Gymnasium interface only; FluidGym physics/reward/boundaries stay unchanged."""
import gymnasium as gym
import numpy as np
from types import SimpleNamespace
from fluidgym_rl.comparison.evaluate import NominalEnv


class SACEnv(gym.Env):
    metadata={}
    def __init__(self, native, seed=0, device='cuda:0'):
        self.inner=NominalEnv(native,SimpleNamespace(obs_shape=[302],action_dim=1,episode_length=80,device=device))
        self.observation_space=self.inner.observation_space
        self.action_space=self.inner.action_space
        self.train_seed=seed; self.episodes=0; self.real_steps=0
    def reset(self, *, seed=None, options=None):
        seed=self.train_seed+self.episodes if seed is None else seed
        super().reset(seed=seed)
        self.episodes+=1
        return self.inner.reset(seed=seed,randomize=True)
    def step(self, action):
        a=np.asarray(action)
        if a.shape!=(1,) or not np.isfinite(a).all() or (np.abs(a)>1+1e-6).any():
            raise ValueError('illegal SAC action')
        result=self.inner.step(a); self.real_steps+=1
        return result
    def close(self): self.inner.close()
