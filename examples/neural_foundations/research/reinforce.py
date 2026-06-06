from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__:
    from .point_robot import PointRobotEnv
else:
    from point_robot import PointRobotEnv


class PolicyNetwork(nn.Module):
    def __init__(self, observation_size: int = 4, hidden_size: int = 16, action_size: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    returns: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def train_episode(
    env: PointRobotEnv,
    policy: PolicyNetwork,
    optimizer,
    gamma: float,
) -> float:
    seed = int(torch.randint(0, 1_000_000, ()).item())
    observation = env.reset(seed=seed)
    log_probs = []
    rewards = []
    done = False

    while not done:
        observation_tensor = torch.tensor(observation, dtype=torch.float32)
        logits = policy(observation_tensor)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        observation, reward, done, _info = env.step(int(action.item()))
        log_probs.append(distribution.log_prob(action))
        rewards.append(reward)

    returns = discounted_returns(rewards, gamma)
    if returns.numel() > 1 and float(returns.std()) > 0.0:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = -torch.stack(log_probs).mul(returns).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(sum(rewards))


def evaluate(
    env_factory,
    policy: PolicyNetwork,
    seeds: list[int],
) -> dict[str, float]:
    returns = []
    lengths = []
    successes = 0

    policy.eval()
    with torch.no_grad():
        for seed in seeds:
            env = env_factory()
            observation = env.reset(seed=seed)
            total_reward = 0.0
            done = False
            info = {"success": False, "steps": 0}

            while not done:
                observation_tensor = torch.tensor(observation, dtype=torch.float32)
                action = int(torch.argmax(policy(observation_tensor)).item())
                observation, reward, done, info = env.step(action)
                total_reward += reward

            returns.append(total_reward)
            lengths.append(float(info["steps"]))
            successes += int(bool(info["success"]))
    policy.train()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(successes / len(seeds)),
        "mean_episode_length": float(np.mean(lengths)),
    }


def train_demo(
    episodes: int = 180,
    gamma: float = 0.95,
    learning_rate: float = 0.01,
    seed: int = 0,
    evaluation_seeds: list[int] | None = None,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = PointRobotEnv()
    policy = PolicyNetwork()
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    returns = [train_episode(env, policy, optimizer, gamma) for _ in range(episodes)]
    seeds = evaluation_seeds or [0, 1, 2, 3, 4]
    metrics = evaluate(PointRobotEnv, policy, seeds)
    return {
        "episodes": episodes,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "training_seed": seed,
        "evaluation_seeds": seeds,
        "initial_training_return": float(returns[0]),
        "final_training_return": float(returns[-1]),
        "evaluation": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train REINFORCE on the point robot.")
    parser.add_argument("--episodes", type=int, default=180)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("examples/neural_foundations/research/results/reinforce_five_seed.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_demo(
        episodes=args.episodes,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        seed=args.seed,
        evaluation_seeds=args.evaluation_seeds,
    )
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["evaluation"], indent=2))


if __name__ == "__main__":
    main()
