from __future__ import annotations

from typing import Dict, Any

from fastapi import FastAPI, HTTPException

from ..loaders import from_config


def create_app() -> FastAPI:
    app = FastAPI(title="AgentEnv Hub - Agent Server")
    sessions: Dict[str, Any] = {}
    counter = 0

    @app.post("/session")
    def create_session(run_cfg: Dict[str, Any]):
        nonlocal counter
        counter += 1
        session_id = f"s{counter}"
        agent = from_config(run_cfg)
        sessions[session_id] = agent
        return {"session_id": session_id}

    @app.post("/session/{session_id}/reset")
    def reset(session_id: str):
        agent = sessions.get(session_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="session not found")
        io = agent.reset()
        return io.model_dump() if hasattr(io, "model_dump") else io

    @app.post("/session/{session_id}/act_step")
    def act_step(session_id: str, payload: Dict[str, Any]):
        agent = sessions.get(session_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="session not found")
        action = payload.get("action")
        if action is None:
            # If using LLM reply flow, call observe->act instead (not implemented here)
            io = agent.observe()
            action = agent.act(io)
        io = agent.step(action)
        return io.model_dump() if hasattr(io, "model_dump") else io

    @app.post("/session/{session_id}/close")
    def close(session_id: str):
        agent = sessions.pop(session_id, None)
        if agent is None:
            raise HTTPException(status_code=404, detail="session not found")
        agent.close()
        return {"ok": True}

    return app


