"""D-MPC reproduction (FluidGym paper, App. D.3, Algorithm 1) — NOT IMPLEMENTED YET.

Deliberately a separate subpackage: it must not import or depend on TD-MPC2 code. Spec to implement (from the paper):
receding-horizon gradient ascent through the differentiable simulator; H=20, N=10 iterations, step size 0.1, gamma=0.999,
no policy/value network; 10 seeds x 1 test episode on the three 2D cylinder envs. Open spec details: optimizer (plain GD vs Adam)
and how 'set env to state s0' maps onto env.get_state()/set_state()/detach(). Needs Phase 0 CUDA validation first.
"""
