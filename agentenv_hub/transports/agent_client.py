from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class AgentServerClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def create_session(self, run_config: Dict[str, Any]) -> str:
        r = self._client.post(f"{self._base_url}/session", json=run_config)
        r.raise_for_status()
        return r.json()["session_id"]

    def reset(self, session_id: str):
        r = self._client.post(f"{self._base_url}/session/{session_id}/reset")
        r.raise_for_status()
        return r.json()

    def act_step(self, session_id: str, action: Optional[str] = None, llm_reply: Optional[str] = None):
        payload: Dict[str, Any] = {}
        if action is not None:
            payload["action"] = action
        if llm_reply is not None:
            payload["llm_reply"] = llm_reply
        r = self._client.post(f"{self._base_url}/session/{session_id}/act_step", json=payload)
        r.raise_for_status()
        return r.json()

    def close(self, session_id: str):
        r = self._client.post(f"{self._base_url}/session/{session_id}/close")
        r.raise_for_status()
        return r.json()


