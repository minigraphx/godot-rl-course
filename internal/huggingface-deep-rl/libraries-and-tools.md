# Libraries and tools — HF Deep RL course

| Library | Role in HF course | Godot course equivalent |
|---------|-------------------|-------------------------|
| [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) | Unit 1 Lunar Lander; Unit 6 A2C | Primary trainer via `gdrl` / SB3 |
| [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) | Unit 3 Atari DQN training, eval, hyperparams | Less central; hub examples use custom scripts |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | Unit 8 PPO from scratch | Extension 11 mention |
| [Sample Factory](https://samplefactory.dev/) | Unit 8b async PPO, VizDoom | Extension 11; same author ecosystem as godot-rl-agents |
| [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) | Units 5, 7 | **godot-rl-agents** + Godot 4 |
| [Gymnasium](https://gymnasium.farama.org/) | Standard env API in Colab | Godot env via socket / exported binary |
| [PyTorch](https://pytorch.org/) | Units 4, 8 from-scratch agents | SB3 backend; ONNX export Unit 10 |
| [Hugging Face Hub](https://huggingface.co/) | Share trained agents, model cards | Optional; not required in Godot 0–10 |

## Typical HF hands-on workflow

1. Open Colab notebook for the unit.
2. Install pinned dependencies (`stable-baselines3`, `gymnasium`, etc.).
3. Train agent; log metrics.
4. Record video replay.
5. Push to Hub with `huggingface_hub` + metadata (mean reward, env id).

## Godot course workflow (contrast for integration copy)

1. Run or export Godot env (editor, `--viz`, or headless binary).
2. `gdrl` launches SB3 (or CleanRL) against socket.
3. TensorBoard on `logs/`.
4. Export ONNX → load in Godot `Sync` (Unit 10).

Authors should explain *why* Godot skips Colab/Hub without dismissing HF workflow as “wrong.”

## Environments reference (HF)

| Environment | HF unit | Properties |
|-------------|---------|------------|
| LunarLander-v2 | 1, 8a | Continuous obs, discrete actions; classic pedagogy |
| FrozenLake-v1 | 2 | Small discrete; sparse reward |
| Taxi-v3 | 2 | Larger discrete; navigation |
| Atari (ALE) | 3 | High-dimensional pixels |
| CartPole-v1 | 4 | Simple policy gradient testbed |
| PixelCopter | 4 | Visual policy gradient |
| ML-Agents snowball / pyramid | 5 | 3D, curiosity on pyramid |
| Fetch (robotics) | 6 | Continuous control |
| Soccer 2v2 | 7 | Multi-agent |
| VizDoom | 8b | FPS, partial observability |
