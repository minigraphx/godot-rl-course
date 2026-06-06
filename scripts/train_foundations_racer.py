#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def convert_space(space):
    import gym
    import gymnasium as gymnasium

    if isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(
            low=space.low,
            high=space.high,
            shape=space.shape,
            dtype=space.dtype,
        )
    if isinstance(space, gym.spaces.Discrete):
        return gymnasium.spaces.Discrete(space.n)
    if isinstance(space, gym.spaces.Dict):
        return gymnasium.spaces.Dict({key: convert_space(value) for key, value in space.spaces.items()})
    raise TypeError(f"Unsupported space type: {type(space)!r}")


class GymnasiumSpaceAdapter(VecEnv):
    def __init__(self, env):
        self.env = env
        observation_space = convert_space(env.observation_space)
        action_space = convert_space(env.action_space)
        super().__init__(env.num_envs, observation_space, action_space)

    def reset(self):
        return self.env.reset()

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        return self.env.step(self._actions)

    def step(self, actions):
        return self.env.step(actions)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        return [seed]

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self.env, method_name)(*method_args, **method_kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Foundations racer with SB3 PPO.")
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--speedup", type=int, default=4)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("examples/neural_foundations/game/unit_03_racer/models/foundations_racer.zip"),
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("examples/neural_foundations/game/unit_03_racer/models/foundations_racer.onnx"),
    )
    return parser.parse_args()


def export_policy_onnx(model, onnx_path: Path) -> None:
    import torch

    class PolicyActor(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            distribution = self.policy.get_distribution({"obs": obs})
            return distribution.distribution.mean

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    actor = PolicyActor(model.policy).eval()
    obs_dim = model.observation_space["obs"].shape[0]
    dummy_obs = torch.zeros((1, obs_dim), dtype=torch.float32)
    torch.onnx.export(
        actor,
        dummy_obs,
        onnx_path,
        input_names=["obs"],
        output_names=["out0"],
        dynamic_axes={"obs": {0: "batch"}, "out0": {0: "batch"}},
        opset_version=17,
    )


def main() -> None:
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
    from stable_baselines3 import PPO

    args = parse_args()
    env = GymnasiumSpaceAdapter(StableBaselinesGodotEnv(
        env_path=None,
        show_window=False,
        seed=args.seed,
        speedup=args.speedup,
        action_repeat=args.action_repeat,
    ))
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=64,
        batch_size=32,
        tensorboard_log="logs/foundations_racer",
    )
    model.learn(total_timesteps=args.timesteps)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    export_policy_onnx(model, args.onnx_path)
    env.close()
    print(f"saved model: {args.model_path}")
    print(f"saved onnx: {args.onnx_path}")


if __name__ == "__main__":
    main()
