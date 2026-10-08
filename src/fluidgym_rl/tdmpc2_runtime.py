"""Harness-level helpers around UPSTREAM TD-MPC2 (patched work copy): compose the upstream hydra config, build the agent, load a
checkpoint, and expose TD-MPC2 inference as an evaluation `Policy`. No upstream algorithm code is modified or re-implemented."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUN_CONFIG = "run_config.json"


def default_overrides(env_id: str, *, model_size: int, seed: int, eval_episodes: int, compile: bool, exp_name: str = "smoke") -> list[str]:
    return [f"task=fluidgym-{env_id}", f"model_size={model_size}", f"seed={seed}", "enable_wandb=false", "save_video=false", f"compile={str(compile).lower()}",
            f"eval_episodes={eval_episodes}", f"exp_name={exp_name}", "wandb_project=none", "wandb_entity=none", "data_dir=none", "checkpoint=none"]


def compose_cfg(code_dir: str, overrides: list[str], run_dir: str | Path):
    """Compose + parse the upstream config. `hydra.utils.get_original_cwd` is pointed at os.getcwd because we do not run under @hydra.main."""
    import hydra
    from hydra import compose, initialize_config_dir
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    hydra.utils.get_original_cwd = os.getcwd
    with initialize_config_dir(version_base=None, config_dir=code_dir):
        cfg = compose(config_name="config", overrides=overrides)
    from common.parser import parse_cfg
    from common.seed import set_seed
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    return cfg


def save_run_config(path: str | Path, cfg, overrides: list[str], extra: dict | None = None) -> None:
    Path(path).write_text(json.dumps({"overrides": overrides, "obs_shape": {k: list(v) for k, v in cfg.obs_shape.items()}, "action_dim": int(cfg.action_dim),
                                      "episode_length": int(cfg.episode_length), "model_size": cfg.get("model_size"), **(extra or {})}, indent=2))


def apply_run_config(cfg, run_config: dict) -> None:
    cfg.obs_shape = {k: tuple(v) for k, v in run_config["obs_shape"].items()}
    cfg.action_dim, cfg.episode_length = run_config["action_dim"], run_config["episode_length"]


def load_agent(ckpt: str | Path, run_config_path: str | Path, code_dir: str, run_dir: str | Path):
    """Fresh-model load (inference restoration). Upstream `save()` stores only model weights, so this is NOT an exact training resume."""
    rc = json.loads(Path(run_config_path).read_text())
    cfg = compose_cfg(code_dir, rc["overrides"], run_dir)
    apply_run_config(cfg, rc)
    from tdmpc2 import TDMPC2
    agent = TDMPC2(cfg)
    agent.load(str(ckpt))
    return cfg, agent, rc


class TDMPC2Policy:
    """mode='plan': upstream MPPI planning with eval_mode=True (stochastic: sampling noise; seeded per episode via torch RNG).
    mode='actor': deterministic policy-prior mean, no planning (a diagnostic; NOT what upstream evaluates with when cfg.mpc=True)."""

    def __init__(self, agent, mode: str = "plan"):
        assert mode in ("plan", "actor")
        self.agent, self.name = agent, "tdmpc2"
        self.mode = "mpc_plan_seeded" if mode == "plan" else "actor_mean"
        self._kind = mode

    def reset(self, episode_seed: int) -> None:
        import torch
        torch.manual_seed(episode_seed)
        torch.cuda.manual_seed_all(episode_seed)

    def act(self, obs, t0: bool):
        import torch
        o = obs.detach().float().cpu() if hasattr(obs, "detach") else torch.as_tensor(obs, dtype=torch.float32)
        if self._kind == "plan":
            return self.agent.act(o, t0=t0, eval_mode=True)
        with torch.no_grad():
            z = self.agent.model.encode(o.to(self.agent.device).unsqueeze(0), None)
            return self.agent.model.pi(z, None)[1]["mean"][0].cpu()
