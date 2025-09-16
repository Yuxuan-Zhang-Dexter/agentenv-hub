from __future__ import annotations

from typing import Any, List, Optional, Tuple

from agentenv_hub.base import BaseAgentUnit, AgentSpec, AgentIO, BaseEnv


class SokobanAgentUnit(BaseAgentUnit):
    def __init__(self, system_prompt: str, action_regex: str, **hp):
        self._spec = AgentSpec(
            agent_id="sokoban-agent",
            format="react",
            parse_action_regex=action_regex,
            track_conversation=True,
            mask_invalid_actions=hp.get("mask_invalid_actions", True),
        )
        self._env: Optional[BaseEnv] = None
        self._history: List[Tuple[str, str]] = [("system", system_prompt)]
        self._last_obs: Any = None

    def spec(self) -> AgentSpec:
        return self._spec

    def bind(self, env: BaseEnv) -> None:
        self._env = env

    def reset(self, **kw) -> AgentIO:
        assert self._env is not None, "Agent must be bound to an environment before reset()"
        obs, info = self._env.reset(**kw)
        self._last_obs = obs
        return AgentIO(observation=obs, info=info)

    def observe(self) -> AgentIO:
        return AgentIO(observation=self._last_obs)

    def act(self, io: AgentIO) -> str:
        # Placeholder policy: always move left. Replace with LLM/tooling later.
        return "move_left"

    def step(self, action: str) -> AgentIO:
        assert self._env is not None, "Agent must be bound to an environment before step()"
        obs, rew, term, trunc, info = self._env.step(action)
        done = bool(term or trunc)
        self._last_obs = obs
        return AgentIO(observation=obs, action=action, reward=rew, done=done, info=info)

    def close(self) -> None:
        # No-op; override if needed
        pass


