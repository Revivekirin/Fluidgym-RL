"""Explicit nominal FluidGym contract, independent of any learner.

Physics values are expectations, not constructor overrides. Only the historical
nominal specification is executable until a separate shift protocol is validated.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

CONTRACT = dict(observation='FlattenObservation: velocity only, sensor-major [vx,vy], 151 sensors',
                normalization='none', frame_stack=1, actuator_state_augmented=False,
                partial_observation='smoothed actuator memory is not observed',
                reward='Cd_ref - mean_substep(Cd) - abs(mean_substep(Cl)); scale=1',
                control_timestep=0.25, nominal_solver_dt=0.01, adaptive_cfl=0.8,
                smoothing='upstream alpha=0.1 each solver step; no adapter changes',
                upstream_fluidgym_sha='50071140b73a2a917fb96d33522a769f903f8c27',
                abs_lift='mean over control steps of abs(control-interval mean Cl), not solver-substep mean abs Cl',
                termination='unexpected physical termination rejected; horizon truncation bootstraps as upstream')

NOMINAL_SPEC = {
    "schema_version": 1,
    "environment_id": "CylinderJet2D-easy-v0",
    "physics": {"reynolds_number": 100.0, "control_timestep": 0.25,
                "lift_penalty": 1.0, "action_smoothing_alpha": 0.1},
    "interface": {"observation_shape": [302], "action_dim": 1, "episode_length": 80,
                  "flatten_observation": True},
}


def resolve_environment(spec=None):
    """Resolve an independent JSON specification; fail closed before simulation.

    No arbitrary kwargs, private-attribute assignments or shift execution are
    supported. Return a detached mapping so callers cannot alter the defaults.
    """
    if spec is None:
        return deepcopy(NOMINAL_SPEC)
    if isinstance(spec, (str, Path)):
        spec = json.loads(Path(spec).read_text())

    def check(actual, expected, path):
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(actual) != set(expected):
                raise ValueError(f'unsupported environment keys at {path}')
            for key in expected:
                check(actual[key], expected[key], f'{path}.{key}')
        elif type(actual) is not type(expected) or actual != expected:
            raise ValueError(f'unsupported environment setting {path}: {actual!r}; only nominal is executable')

    check(spec, NOMINAL_SPEC, 'environment')
    return deepcopy(spec)


def validate_tdmpc_parameters(parameters, spec=None):
    """Bridge legacy checkpoint/config fields to environment-owned expectations."""
    spec = resolve_environment(spec)
    interface = spec['interface']
    if parameters['task'] != spec['environment_id'] or parameters['modality'] != 'state' or parameters['action_repeat'] != 1:
        raise ValueError('only nominal CylinderJet2D state observations, action_repeat=1 are supported')
    if (parameters['episode_length'] != interface['episode_length'] or
            parameters['obs_shape'] != interface['observation_shape'] or
            parameters['action_dim'] != interface['action_dim']):
        raise ValueError('nominal contract requires horizon 80, observation 302, action 1')


def validate_physics(env, spec=None):
    """Validate observed physics using the pre-refactor numerical tolerances.

    Synthetic test doubles without cuda_device are handled by the adapter's
    interface checks; this function itself never skips missing physical fields.
    This is not a solver/dataset certification (see preparation documentation).
    """
    physics = resolve_environment(spec)['physics']
    fields = {'step_length': 'control_timestep', '_reynolds_number': 'reynolds_number',
              '_lift_penalty': 'lift_penalty', '_action_smoothing_alpha': 'action_smoothing_alpha'}
    for attribute, key in fields.items():
        if not hasattr(env, attribute) or not np.isclose(float(getattr(env, attribute)), physics[key]):
            raise ValueError(f'nominal physical contract mismatch: {attribute}')


def validate_creation_kwargs(kwargs):
    """The original TD-MPC factory has no supported physical overrides yet."""
    if kwargs:
        raise ValueError(f'unsupported TD-MPC environment constructor overrides: {sorted(kwargs)}')
