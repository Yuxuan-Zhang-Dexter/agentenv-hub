from .env import SokobanEnv
from .agent import SokobanAgentUnit


def make_env(**params):
    return SokobanEnv(**params)


def make_agent_unit(**params):
    return SokobanAgentUnit(**params)


