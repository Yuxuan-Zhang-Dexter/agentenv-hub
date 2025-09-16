from __future__ import annotations

from typing import Callable, Dict, Any, Iterable

try:
    from importlib.metadata import entry_points, EntryPoint
except Exception:  # pragma: no cover
    from importlib_metadata import entry_points, EntryPoint  # type: ignore


ENV_GROUP = "agentenv_hub.envs"
AGENT_GROUP = "agentenv_hub.agents"


def _get_entry_points(group: str) -> Iterable["EntryPoint"]:
    eps = entry_points()
    # Python 3.10+ supports select(); older returns a dict-like
    try:
        return eps.select(group=group)  # type: ignore[attr-defined]
    except Exception:
        return eps.get(group, [])  # type: ignore[call-arg]


def _load_from_entry_points(group: str, name: str):
    for ep in _get_entry_points(group):
        if ep.name == name:
            return ep.load()
    raise KeyError(f"No entry point named '{name}' in group '{group}'")


class _Registry:
    def __init__(self) -> None:
        self._factories: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        self._factories[name] = factory

    def names(self) -> Dict[str, Callable[..., Any]]:
        return dict(self._factories)

    def create(self, name: str, **kwargs):
        if name in self._factories:
            return self._factories[name](**kwargs)
        # fallback to entry points
        return _load_from_entry_points(
            ENV_GROUP if self is env_registry else AGENT_GROUP, name
        )(**kwargs)


env_registry = _Registry()
agent_registry = _Registry()


def load_env(name: str, **kw):
    return env_registry.create(name, **kw)


def load_agent_unit(name: str, **kw):
    return agent_registry.create(name, **kw)


