## agentenv-hub
agentenv_hub is a training-free Python library that standardizes agent↔environment interaction and packaging. Each standalone unit (e.g., `units/SokobanAgent/`) contains both the agent and the environment wrapper, plus Hydra-style defaults, so users can import the unit and run an agent↔env loop immediately.

### Principles
- **Env stays dumb & swappable**: Gymnasium step/reset API `(obs, reward, terminated, truncated, info)`.
- **Multi-agent via PettingZoo**: Optional adapter to expose dict-style observations/actions.
- **Plugin discovery with entry points**: Auto-register units on install.
- **Optional service**: FastAPI AgentServer with Pydantic types.
- **Hydra-style configs**: `name + params` with Pydantic validation.

### Repository layout
```text
agentenv_hub/
  agentenv_hub/
    __init__.py
    base.py                 # BaseEnv, BaseAgentUnit, EnvSpec, AgentSpec, AgentIO
    registry.py             # Entry points + manual registry helpers
    loaders.py              # make_env(), make_agent_unit(), bind(), from_config()
    config/
      schema.py             # Pydantic models mirroring agents.yaml style
    adapters/
      gymnasium_env.py      # Gymnasium -> BaseEnv (single-agent)
      pettingzoo_env.py     # PettingZoo (AEC/Parallel) -> BaseEnv
    transports/
      http_env.py           # optional remote env client
      agent_server.py       # optional FastAPI AgentServer
      agent_client.py       # optional HTTP client for AgentServer
    vector/
      pool.py               # async multi-session runner (no training logic)
  units/
    SokobanAgent/
      __init__.py
      agent.py              # SokobanAgentUnit (interaction, parsing, convo)
      env.py                # SokobanEnv (thin Gym wrapper)
      defaults_agent.yaml   # agent defaults (Hydra-friendly)
      defaults_env.yaml     # env defaults
  examples/
    configs/
      defaults.yaml
      agent/sokoban_agent.yaml
      env/sokoban_env.yaml
    quickstart_local.py
    quickstart_service.py
  pyproject.toml            # entry points for agents & envs (units/* register here)
  README.md
```

### Minimal, stable interfaces
```python
from typing import Any, Dict, Tuple, Optional, Protocol, List
from pydantic import BaseModel

class EnvSpec(BaseModel):
    id: str
    multiagent: bool = False
    horizon: Optional[int] = None
    observation_space: Any | None = None
    action_space: Any | None = None

class BaseEnv(Protocol):
    def spec(self) -> EnvSpec: ...
    def reset(self, seed: Optional[int]=None, options: Optional[Dict]=None) -> Tuple[Any, Dict]: ...
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]: ...
    def close(self) -> None: ...

class AgentSpec(BaseModel):
    agent_id: str
    format: str = "react"
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
    def act(self, io: AgentIO) -> str: ...
    def step(self, action: str) -> AgentIO: ...
    def close(self) -> None: ...
```

### Registry and loaders
```python
from importlib.metadata import entry_points

# groups:
#  - agentenv_hub.envs    name -> callable(**kwargs) -> BaseEnv
#  - agentenv_hub.agents  name -> callable(**kwargs) -> BaseAgentUnit
def load_env(name: str, **kw):
    return entry_points(group="agentenv_hub.envs")[name].load()(**kw)

def load_agent_unit(name: str, **kw):
    return entry_points(group="agentenv_hub.agents")[name].load()(**kw)

def make_env(target: str, **params):
    if target.startswith(("http://","https://")):
        from agentenv_hub.transports.http_env import HttpEnv
        return HttpEnv(base_url=target, **params)
    from agentenv_hub.registry import load_env
    return load_env(target, **params)

def make_agent_unit(name: str, **params):
    from agentenv_hub.registry import load_agent_unit
    return load_agent_unit(name, **params)

def bind(agent, env):
    agent.bind(env); return agent
```

### Standalone unit pattern (example)
```python
# units/SokobanAgent/env.py
from agentenv_hub.base import BaseEnv, EnvSpec
import gymnasium as gym

class SokobanEnv(BaseEnv):
    def __init__(self, env_id: str="Sokoban-v0", **kwargs):
        self._env = gym.make(env_id, **kwargs)
        gid = getattr(getattr(self._env, "spec", None), "id", env_id)
        self._spec = EnvSpec(id=gid, multiagent=False)
    def spec(self): return self._spec
    def reset(self, seed=None, options=None): return self._env.reset(seed=seed, options=options)
    def step(self, a):
        obs, r, term, trunc, info = self._env.step(a); return obs, r, term, trunc, info
    def close(self): self._env.close()
```

```python
# units/SokobanAgent/agent.py
from agentenv_hub.base import BaseAgentUnit, AgentSpec, AgentIO

class SokobanAgentUnit(BaseAgentUnit):
    def __init__(self, system_prompt: str, action_regex: str, **hp):
        self._spec = AgentSpec(agent_id="sokoban-agent", format="react",
                               parse_action_regex=action_regex, track_conversation=True,
                               mask_invalid_actions=hp.get("mask_invalid_actions", True))
        self._env = None
        self._history = [("system", system_prompt)]
        self._last_obs = None
    def spec(self): return self._spec
    def bind(self, env): self._env = env
    def reset(self, **kw):
        obs, info = self._env.reset(**kw); self._last_obs = obs
        return AgentIO(observation=obs, info=info)
    def observe(self): return AgentIO(observation=self._last_obs)
    def act(self, io: AgentIO) -> str: return "move_left"
    def step(self, action: str) -> AgentIO:
        obs, rew, term, trunc, info = self._env.step(action)
        done = bool(term or trunc)
        self._last_obs = obs
        return AgentIO(observation=obs, action=action, reward=rew, done=done, info=info)
    def close(self): pass
```

### Entry points (plugins)
```toml
[project.entry-points."agentenv_hub.envs"]
sokoban_env = "units.SokobanAgent:make_env"

[project.entry-points."agentenv_hub.agents"]
sokoban_agent = "units.SokobanAgent:make_agent_unit"
```

### Config styles
Agent-owns-Env: keep both `defaults_*.yaml` inside the unit. External config: select agent and env separately.

Example external configs are in `examples/configs/`. A minimal runner:
```python
from agentenv_hub.loaders import make_env, make_agent_unit, bind
agent = make_agent_unit(cfg.agent.name, **cfg.agent.params)
env   = make_env(cfg.env.target, **cfg.env.params)
bind(agent, env)
io = agent.reset()
while True:
    a  = agent.act(io)
    io = agent.step(a)
    if io.done: break
```

### Quickstarts
- **Local**: `python examples/quickstart_local.py`
- **Service**: `python examples/quickstart_service.py` then drive it with an HTTP client.

### Why this layout is future-proof
- **Clear ownership**: Env = state machine (Gym/PZ). AgentUnit = interaction brain.
- **Standalone units**: Obvious code, defaults, and registry hooks.
- **Hydra-style configs**: Mirrors `agents.yaml` shape; Pydantic validation.
- **Entry points**: Anyone can publish units as pip packages; no hub changes needed.
