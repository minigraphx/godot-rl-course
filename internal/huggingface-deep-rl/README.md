# Hugging Face Deep RL Course — internal learnings

**Source:** [🤗 Deep Reinforcement Learning Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) (Thomas Simonini et al., Hugging Face)

**Course state (2026):** Low maintenance — theory and Colab hands-on remain useful; Unit 7 AI-vs-AI leaderboard and global leaderboard are non-functional.

## Contents

| File | Description |
|------|-------------|
| [syllabus.md](syllabus.md) | Full unit list, tools, certification paths |
| [integration-map.md](integration-map.md) | HF unit → Godot RL course unit mapping |
| [concepts-glossary.md](concepts-glossary.md) | Cross-unit terms students should know |
| [libraries-and-tools.md](libraries-and-tools.md) | SB3, RL-Zoo, CleanRL, Sample Factory, Hub |
| [units/](units/) | Per-unit learnings (theory + hands-on + integration notes) |

## What students learn (high level)

1. **RL foundations** — MDP loop, reward, policy, value vs policy methods, exploration.
2. **Tabular RL** — Q-Learning, MC vs TD, small discrete envs.
3. **Deep value-based** — DQN, replay, target nets, Atari-scale state spaces.
4. **Policy gradients** — REINFORCE, variance issues.
5. **Game-engine envs** — Unity ML-Agents (conceptual parallel to Godot + godot-rl-agents).
6. **Actor–critic** — A2C, advantage, variance reduction.
7. **Multi-agent** — MARL basics, cooperative/competitive soccer.
8. **PPO** — clipped objective, CleanRL from scratch, Sample Factory + VizDoom.

## Relevance to this repo

- Same **training stack family**: Stable-Baselines3, optional CleanRL / Sample Factory (Extension 11).
- **Lunar Lander** appears in HF Unit 1 and Godot Unit 2 — strong alignment for MDP vocabulary and reward design.
- **Edward Beeching** (godot-rl-agents) authored HF Unit 8 Part 2 (Sample Factory) — cite when teaching PPO at scale.
- **Do not** copy HF certification, Discord challenges, or Unity-only labs into core Units 0–10 without adaptation.
