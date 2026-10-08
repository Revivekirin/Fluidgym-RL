"""Construction of FluidGym environments. FluidGym itself is used unmodified."""
from __future__ import annotations


def make_fluid_env(env_id: str = "CylinderJet2D-easy-v0", *, flatten: bool = True, mode: str = "train", **kwargs):
    """Create a (single-agent) FluidGym env.

    flatten=True applies fluidgym.wrappers.FlattenObservation, which is required for
    compatibility with the published HF models (trained with FluidGym v0.0.2) and gives
    the flat Box observation TD-MPC/TD-MPC2 expect.
    """
    import fluidgym  # lazy: needs CUDA build
    from fluidgym.wrappers import FlattenObservation

    env = fluidgym.make(env_id, **kwargs)
    env = FlattenObservation(env) if flatten else env
    getattr(env, mode)()  # env.train() / env.val() / env.test() select the initial-domain split
    return env
