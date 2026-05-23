# Godot RL Course

Learn deep reinforcement learning by building and training agents inside real Godot game environments.

## What you'll build

| Phase | Content | What you learn |
|-------|---------|----------------|
| **Foundations** (Units 0–2) | Setup, RL loop, first custom env | How RL works, first agent in Godot |
| **Value-Based** (Q-Learning → DQN) | Bellman, Q-tables, DQN, experience replay | How value methods work from scratch |
| **Policy-Based** (REINFORCE → Actor-Critic → PPO) | Policy gradients, A2C, PPO internals | Why PPO is the standard algorithm |
| **Scale & Complexity** (Units 5–8) | Parallel training, continuous actions, MARL, memory | Real-world training at scale |
| **Beyond Reward** (Units 9–10) | Imitation learning, ONNX/WASM | Alternative learning signals, shipping |

## Three ways to see your AI (every unit)

| Channel | What it shows |
|---------|---------------|
| **Godot** | Agent behavior in the world |
| **TensorBoard** | Learning curves (`tensorboard --logdir=logs`) |
| **AIController source** | Observations, actions, and rewards you changed |

## Units

**Phase 1 — Foundations**

- [Unit 0 — Setup & First Run](unit-00.md)
- [Unit 1 — RL Foundations](unit-01.md)
- [Unit 2 — Build Your First Env](unit-02.md)

**Phase 2 — Value-Based Methods**

- [Q-Learning](unit-q-learning.md)
- [Deep Q-Learning (DQN)](unit-03.md)

**Phase 3 — Policy-Based Methods**

- [Policy Gradients & REINFORCE](unit-policy-gradients.md)
- [Actor-Critic](unit-actor-critic.md)
- [PPO Deep Dive](unit-ppo-deep.md)
- [PPO in Practice (JumperHard)](unit-04.md)

**Phase 4 — Scale & Complexity**

- [Unit 5 — Parallel Training](unit-05.md)
- [Unit 6 — Continuous 3D](unit-06.md)
- [Unit 7 — Multi-Agent](unit-07.md)
- [Unit 8 — Memory & POMDPs](unit-08.md)

**Phase 5 — Beyond Reward**

- [Unit 9 — Imitation Learning](unit-09.md)
- [Unit 10 — Ship Your Brain](unit-10.md)

## After this course

!!! info "Follow-on: model alignment (separate course)"
    This course ends at ONNX vector policies. A planned sequel covers language-model alignment (SFT, preferences, RLHF/DPO). Unit 9 imitation learning is the on-ramp. Optional — Units 0–10 are complete on their own.

    Alignment course repository: *coming soon*.
