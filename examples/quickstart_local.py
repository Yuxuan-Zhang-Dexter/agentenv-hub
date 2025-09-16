from __future__ import annotations

from agentenv_hub.loaders import make_env, make_agent_unit, bind
from agentenv_hub.config.schema import RunConfig, AgentConfig, EnvConfig


def main():
    cfg = RunConfig(
        agent=AgentConfig(
            name="sokoban_agent",
            params={
                "system_prompt": "You are a Sokoban solver...",
                "action_regex": r"Action:\s*(.*?)(?=\n|$)",
            },
        ),
        env=EnvConfig(
            target="sokoban_env",
            params={"env_id": "Sokoban-v0"},
        ),
    )

    agent = make_agent_unit(cfg.agent.name, **cfg.agent.params)
    env = make_env(cfg.env.target, **cfg.env.params)
    bind(agent, env)

    io = agent.reset()
    steps = 0
    total_reward = 0.0
    while True:
        action = agent.act(io)
        io = agent.step(action)
        total_reward += float(io.reward or 0.0)
        steps += 1
        if io.done or steps >= 50:
            break
    print({"steps": steps, "reward": total_reward})


if __name__ == "__main__":
    main()


