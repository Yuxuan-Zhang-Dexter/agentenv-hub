from __future__ import annotations

from typing import Any

from .registry import load_env as _load_env, load_agent_unit as _load_agent


def make_env(target: str, **params: Any):
    if target.startswith(("http://", "https://")):
        from .transports.http_env import HttpEnv

        return HttpEnv(base_url=target, **params)
    return _load_env(target, **params)


def make_agent_unit(name: str, **params: Any):
    return _load_agent(name, **params)


def bind(agent, env):
    agent.bind(env)
    return agent


def from_config(run_cfg):
    # Lazy import to avoid pydantic dependency during simple imports
    from .config.schema import RunConfig

    if isinstance(run_cfg, dict):
        run_cfg = RunConfig(**run_cfg)

    agent = make_agent_unit(run_cfg.agent.name, **run_cfg.agent.params)
    env = make_env(run_cfg.env.target, **run_cfg.env.params)
    return bind(agent, env)


