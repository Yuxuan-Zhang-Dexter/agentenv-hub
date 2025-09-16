from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import httpx

from ..base import BaseEnv, EnvSpec


class HttpEnv(BaseEnv):
    """
    Minimal HTTP-based env client.

    Expects a remote service exposing:
      - POST {base_url}/reset -> {observation, info}
      - POST {base_url}/step  with {action} -> {observation, reward, terminated, truncated, info}
      - POST {base_url}/close -> {}
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self._spec = EnvSpec(id=f"http:{self._base_url}", multiagent=False)

    def spec(self) -> EnvSpec:
        return self._spec

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Any, Dict]:
        payload: Dict[str, Any] = {}
        if seed is not None:
            payload["seed"] = seed
        if options is not None:
            payload["options"] = options
        r = self._client.post(f"{self._base_url}/reset", json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("observation"), data.get("info", {})

    def step(self, action: Any):
        r = self._client.post(f"{self._base_url}/step", json={"action": action})
        r.raise_for_status()
        d = r.json()
        return (
            d.get("observation"),
            float(d.get("reward", 0.0)),
            bool(d.get("terminated", False)),
            bool(d.get("truncated", False)),
            d.get("info", {}),
        )

    def close(self) -> None:
        try:
            self._client.post(f"{self._base_url}/close")
        finally:
            self._client.close()


