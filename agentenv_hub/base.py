from __future__ import annotations

from typing import Any, Dict, Tuple, Optional, Protocol, List
from pydantic import BaseModel


# --- Env (keep simple; Gymnasium contract)
class EnvSpec(BaseModel):
    id: str
    multiagent: bool = False
    horizon: Optional[int] = None
    observation_space: Any | None = None
    action_space: Any | None = None


class BaseEnv(Protocol):
    def spec(self) -> EnvSpec: ...

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Any, Dict]: ...

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]: ...

    def close(self) -> None: ...


# --- AgentUnit (owns interaction with Env + LLM formatting)
class AgentSpec(BaseModel):
    agent_id: str
    format: str = "react"  # 'react' | 'raw' | 'tool_calls'
    multiagent: bool = False
    parse_action_regex: str | None = r"Action:\s*(.*?)(?=\n|$)"
    track_conversation: bool = True
    mask_invalid_actions: bool = True


class AgentIO(BaseModel):
    observation: Any
    available_actions: Optional[List[str]] = None
    thought: Optional[str] = None
    action: Optional[str] = None
    reward: Optional[float] = None
    done: Optional[bool] = None
    info: Optional[Dict] = None


class BaseAgentUnit(Protocol):
    def spec(self) -> AgentSpec: ...

    def bind(self, env: BaseEnv) -> None: ...

    def reset(self, **kwargs) -> AgentIO: ...

    def observe(self) -> AgentIO: ...

    def act(self, io: AgentIO) -> str: ...  # returns raw action string

    def step(self, action: str) -> AgentIO: ...  # env.step + update convo/masks

    def close(self) -> None: ...


