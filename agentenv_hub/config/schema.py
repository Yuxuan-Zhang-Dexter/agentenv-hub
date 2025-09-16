from __future__ import annotations

from typing import Dict, Any, Optional
from pydantic import BaseModel


class EnvConfig(BaseModel):
    target: str
    params: Dict[str, Any] = {}


class AgentConfig(BaseModel):
    name: str
    format: Optional[str] = None
    multiagent: Optional[bool] = None
    params: Dict[str, Any] = {}


class RunConfig(BaseModel):
    agent: AgentConfig
    env: EnvConfig


