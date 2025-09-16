from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    from pettingzoo.utils import parallel_to_aec
    from pettingzoo.utils.conversions import aec_to_parallel
except Exception:  # pragma: no cover
    parallel_to_aec = None  # type: ignore
    aec_to_parallel = None  # type: ignore

from ..base import BaseEnv, EnvSpec


class PettingZooParallelEnv(BaseEnv):
    """
    Thin adapter around a PettingZoo ParallelEnv to expose the BaseEnv interface.

    Notes:
      - This presents a dict observation/action API if the underlying env is multi-agent.
      - For single-agent use-cases, callers can select a specific agent key from the dicts.
    """

    def __init__(self, parallel_env) -> None:
        if parallel_env is None:
            raise ValueError("parallel_env must not be None")
        self._env = parallel_env
        self._spec = EnvSpec(id=getattr(parallel_env, "metadata", {}).get("name", "pettingzoo"), multiagent=True)

    def spec(self) -> EnvSpec:
        return self._spec

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Any, Dict]:
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action: Any):
        # Expect dict[str, Any] for multi-agent parallel envs
        obs, rewards, terms, truncs, infos = self._env.step(action)
        done = all(bool(x) for x in terms.values()) if isinstance(terms, dict) else bool(terms)
        trunc = all(bool(x) for x in truncs.values()) if isinstance(truncs, dict) else bool(truncs)
        # For BaseEnv we must return single reward and done flags; provide sum reward and aggregate flags
        reward_sum = sum(float(x) for x in rewards.values()) if isinstance(rewards, dict) else float(rewards)
        return obs, reward_sum, done, trunc, infos

    def close(self) -> None:
        self._env.close()


