"""Unmodified FluidGym physics behind a checked, five-value Gymnasium interface."""
from __future__ import annotations
import numpy as np
from fluidgym_rl.evalcontract import to_np, scalar

CONTRACT = dict(observation='FlattenObservation: velocity only, sensor-major [vx,vy], 151 sensors',
                normalization='none', frame_stack=1, actuator_state_augmented=False,
                partial_observation='smoothed actuator memory is not observed',
                reward='Cd_ref - mean_substep(Cd) - abs(mean_substep(Cl)); scale=1',
                control_timestep=0.25, nominal_solver_dt=0.01, adaptive_cfl=0.8,
                smoothing='upstream alpha=0.1 each solver step; no adapter changes',
                upstream_fluidgym_sha='50071140b73a2a917fb96d33522a769f903f8c27',
                abs_lift='mean over control steps of abs(control-interval mean Cl), not solver-substep mean abs Cl',
                termination='unexpected physical termination rejected; horizon truncation bootstraps as upstream')


class CheckedEnv:
    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.episode_length = int(env.episode_length)
        if self.observation_space.shape != tuple(cfg.obs_shape) or self.observation_space.dtype != np.float32:
            raise ValueError('observation space does not match float32 (302,) contract')
        if self.action_space.shape != (cfg.action_dim,) or not np.all(self.action_space.low == -1) or not np.all(self.action_space.high == 1):
            raise ValueError('action bounds/shape mismatch')
        if self.episode_length != cfg.episode_length:
            raise ValueError('episode horizon mismatch')
        if hasattr(env, 'cuda_device'):
            from fluidgym_rl.device import same_device
            for key, expected in {'step_length':0.25, '_reynolds_number':100., '_lift_penalty':1., '_action_smoothing_alpha':0.1}.items():
                if not np.isclose(float(getattr(env,key)), expected):
                    raise ValueError(f'nominal physical contract mismatch: {key}')
            if not same_device(env.cuda_device, cfg.device):
                raise ValueError('simulator and learner device mismatch')
        self.t = 0

    def __getattr__(self, key):
        return getattr(self.env, key)

    def close(self):
        """FluidGym does not require close(); forward it when provided."""
        close = getattr(self.env, 'close', None)
        if callable(close):
            close()

    def _obs(self, obs):
        obs = to_np(obs)
        if obs.shape != tuple(self.cfg.obs_shape) or obs.dtype != np.float32 or not np.isfinite(obs).all():
            raise ValueError('invalid observation shape/dtype/finiteness')
        return obs.copy()

    def reset(self, **kwargs):
        self.t = 0
        obs, info = self.env.reset(**kwargs)
        return self._obs(obs), info

    def step(self, action):
        if self.t >= self.episode_length:
            raise RuntimeError('reset required after episode boundary')
        a = to_np(action).astype(np.float32)
        if a.shape != (self.cfg.action_dim,) or not np.isfinite(a).all():
            raise ValueError('invalid action')
        a = np.clip(a, self.action_space.low, self.action_space.high)
        if hasattr(self.env, 'cuda_device'):
            import torch
            a = torch.as_tensor(a, device=self.env.cuda_device)
        obs, reward, term, trunc, info = self.env.step(a)
        self.t += 1
        if bool(term):
            raise RuntimeError('physical termination unsupported by original fixed-episode TD-MPC; refusing invalid bootstrapping')
        if bool(trunc) != (self.t == self.episode_length):
            raise RuntimeError('unexpected truncation / episode length')
        reward = scalar(reward)
        if not np.isfinite(reward):
            raise ValueError('non-finite reward')
        info = dict(info)
        if 'lift' in info:
            info['abs_lift'] = float(np.abs(to_np(info['lift'])).mean())
        for k in ('drag','lift','abs_lift'):
            if k in info and not np.isfinite(to_np(info[k])).all():
                raise ValueError(f'non-finite {k}')
        return self._obs(obs), reward, bool(term), bool(trunc), info


def make_env(cfg, split='train', **kwargs):
    import torch
    from fluidgym_rl.envs import make_fluid_env
    return CheckedEnv(make_fluid_env(cfg.task, mode=split, cuda_device=torch.device(cfg.device), **kwargs), cfg)
