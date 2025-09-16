from __future__ import annotations

from agentenv_hub.base import BaseEnv, EnvSpec
import gymnasium as gym


class SokobanEnv(BaseEnv):
    def __init__(self, env_id: str = "Sokoban-v0", **kwargs):
        self._env = gym.make(env_id, **kwargs)
        gid = getattr(getattr(self._env, "spec", None), "id", env_id)
        self._spec = EnvSpec(id=gid, multiagent=False)

    def spec(self):
        return self._spec

    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, a):
        obs, r, term, trunc, info = self._env.step(a)
        return obs, r, term, trunc, info

    def close(self):
        self._env.close()


