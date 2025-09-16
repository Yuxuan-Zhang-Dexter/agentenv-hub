from __future__ import annotations

import asyncio
from typing import Callable, Dict, Any, List


async def run_sessions_concurrently(
    num_sessions: int,
    make_agent: Callable[[], Any],
    make_env: Callable[[], Any],
    max_steps: int = 100,
) -> List[Dict[str, Any]]:
    async def _run_one(idx: int) -> Dict[str, Any]:
        agent = make_agent()
        env = make_env()
        agent.bind(env)
        io = agent.reset()
        steps = 0
        total_reward = 0.0
        while steps < max_steps:
            action = agent.act(io)
            io = agent.step(action)
            total_reward += float(io.reward or 0.0)
            steps += 1
            if io.done:
                break
        env.close()
        agent.close()
        return {"session": idx, "steps": steps, "reward": total_reward}

    return await asyncio.gather(*[_run_one(i) for i in range(num_sessions)])


