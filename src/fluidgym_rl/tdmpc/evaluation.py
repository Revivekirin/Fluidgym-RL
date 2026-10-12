"""Original TD-MPC policy under the existing shared evaluation contract."""
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from fluidgym_rl.evalcontract import EvalContract, run_evaluation, summarize, write_run, to_np
from fluidgym_rl.tdmpc.runtime import sha256


class Policy:
    name, mode = 'tdmpc', 'mpc_plan'
    def __init__(self, agent, step):
        self.agent, self.step = agent, step
    def reset(self, episode_seed):
        import torch
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if hasattr(self.agent, '_prev_mean'):
            del self.agent._prev_mean
    def act(self, obs, t0):
        # Eval still samples elites/policy trajectories in original upstream.
        return self.agent.plan(to_np(obs), eval_mode=True, step=self.step, t0=t0)


class ZeroPolicy:
    name, mode = 'uncontrolled', 'zero_action'
    def reset(self, episode_seed):
        pass
    def act(self, obs, t0):
        return np.zeros(1, np.float32)


def evaluate(env, policy, cfg, out, *, split, episodes, seed, checkpoint=None, manifest=None, reference=None):
    import torch
    # Validation must not perturb training RNG or planner warm-start state.
    np_state, cpu_state = np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if str(cfg.device).startswith('cuda') else None
    previous = getattr(getattr(policy, 'agent', None), '_prev_mean', None)
    contract = EvalContract(env_id=cfg.task, split=split, n_episodes=episodes, base_seed=seed)
    try:
        records = run_evaluation(env, policy, contract, metric_names=['drag','lift','abs_lift'],
                                 action_low=env.action_space.low, action_high=env.action_space.high)
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if hasattr(policy, 'agent'):
            if previous is not None:
                policy.agent._prev_mean = previous
            elif hasattr(policy.agent, '_prev_mean'):
                del policy.agent._prev_mean
    summary = summarize(records)
    summary['claims_reproduction'] = False
    summary['policy'] = dict(algorithm=policy.name, inference_mode=policy.mode)
    summary['abs_lift_definition'] = 'mean(abs(control-interval mean lift)); solver-substep abs lift unavailable'
    summary['drag_reduction_percent'] = None
    summary['drag_reduction_note'] = 'requires matched uncontrolled training-split reference; not inferred from reward'
    if reference:
        ref = Path(reference)
        rs = json.loads((ref/'summary.json').read_text())
        rm = json.loads((ref/'run_manifest.json').read_text())
        rc = rm['contract']
        if rm['algorithm'] != 'uncontrolled' or rc['split'] != 'train' or rc['env_id'] != cfg.task or rc['n_episodes'] != 10 or rc['protocol'] != 'official_callback':
            raise ValueError('reference must be 10 uncontrolled training episodes of the same environment')
        if rm.get('environment_contract') != (manifest or {}).get('environment_contract') or rm.get('versions',{}).get('fluidgym') != (manifest or {}).get('versions',{}).get('fluidgym'):
            raise ValueError('reference environment contract/version differs')
        if rm.get('installed_environment_source_hashes') != (manifest or {}).get('installed_environment_source_hashes'):
            raise ValueError('reference installed simulator sources differ')
        drag_ref = rs['metrics_per_step_mean']['drag']['mean']
        if not np.isfinite(drag_ref) or drag_ref <= 0:
            raise ValueError('invalid reference drag')
        summary['drag_reduction_percent'] = 100*(1-summary['metrics_per_step_mean']['drag']['mean']/drag_ref)
        summary['drag_reduction_note'] = dict(reference=str(ref.resolve()), sha256=sha256(ref/'summary.json'), drag_ref=drag_ref, aggregation='arithmetic means; paper reports IQM across seeds')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    m = {**(manifest or {}), 'algorithm':policy.name, 'contract':asdict(contract), 'contract_digest':contract.digest(),
         'checkpoint':str(checkpoint) if checkpoint else None, 'checkpoint_sha256':sha256(checkpoint) if checkpoint else None,
         'train_seed':cfg.seed, 'simulator_version':(manifest or {}).get('versions',{}).get('fluidgym'), 'inference_mode':policy.mode}
    write_run(out, dict(parameters=vars(cfg),contract=asdict(contract)), records, summary, m)
    return summary


def save_checkpoint(agent, path, cfg, step, manifest):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(path)
    with torch.no_grad():
        probe = agent.model.h(torch.zeros(1, *cfg.obs_shape, device=cfg.device)).cpu().tolist()
    sidecar = dict(parameters=vars(cfg), environment_steps=step, planner_std=float(agent.std), sha256=sha256(path), probe=probe,
                   semantics='inference only: online+target model weights; no optimizer/replay/RNG resume', provenance=manifest)
    path.with_suffix('.json').write_text(json.dumps(sidecar, indent=2))
    return sidecar


def load_checkpoint(agent, path):
    import torch
    path = Path(path)
    sidecar = json.loads(path.with_suffix('.json').read_text())
    if sidecar['sha256'] != sha256(path):
        raise ValueError('checkpoint hash mismatch')
    agent.load(path)
    planner_std=float(sidecar['planner_std'])
    if not np.isfinite(planner_std) or not 0 < planner_std <= 2:
        raise ValueError('invalid checkpoint planner standard deviation')
    agent.std=planner_std
    with torch.no_grad():
        probe = agent.model.h(torch.zeros(1,*agent.cfg.obs_shape,device=agent.device)).cpu().numpy()
    if not np.allclose(probe, sidecar['probe'], rtol=1e-5, atol=1e-6):
        raise ValueError('checkpoint inference probe mismatch')
    return sidecar
