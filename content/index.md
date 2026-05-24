# Godot RL Course

Learn deep reinforcement learning by building and training agents inside real Godot game environments.

## What you'll build

| Phase | Content | What you learn |
|-------|---------|----------------|
| **Phase 1 — Foundations** | Setup · RL loop · first custom env · reward design | How RL works, how to design rewards |
| **Phase 2 — Value-Based** | Q-Learning · DQN · curiosity | Bellman equation, Q-tables, DQN, sparse reward exploration |
| **Phase 3 — Policy-Based** | REINFORCE · Actor-Critic · PPO · SAC | Policy gradient theorem, PPO internals, continuous control |
| **Phase 4 — Scale & Complexity** | Parallel · 3D · Multi-agent · Memory | Real-world training at scale |
| **Phase 5 — Beyond Reward** | Imitation learning · ONNX/WASM | Alternative learning signals, shipping |
| **Phase 6 — Robotics** | Robot sensors · HER · sim-to-real | Robot observation/action design, goal-conditioned RL, deploying policies to hardware |
| **Guides** | Debugging | Systematic diagnosis of broken training |

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
- [Reward Engineering](unit-reward-engineering.md)

**Phase 2 — Value-Based Methods**

- [Q-Learning](unit-q-learning.md)
- [Deep Q-Learning (DQN)](unit-03.md)
- [Intrinsic Motivation & Curiosity](unit-curiosity.md)

**Phase 3 — Policy-Based Methods**

- [Policy Gradients & REINFORCE](unit-policy-gradients.md)
- [Actor-Critic](unit-actor-critic.md)
- [PPO Deep Dive](unit-ppo-deep.md)
- [PPO in Practice (JumperHard)](unit-04.md)
- [SAC — Soft Actor-Critic](unit-sac.md)

**Phase 4 — Scale & Complexity**

- [Unit 5 — Parallel Training](unit-05.md)
- [Unit 6 — Continuous 3D](unit-06.md)
- [Unit 7 — Multi-Agent](unit-07.md)
- [Unit 8 — Memory & POMDPs](unit-08.md)

**Phase 5 — Beyond Reward**

- [Unit 9 — Imitation Learning](unit-09.md)
- [Unit 10 — Ship Your Brain](unit-10.md)

**Phase 6 — Robotics**

- [Robot Observations & Sensors](unit-robotics.md)
- [Goal-Conditioned RL & HER](unit-her.md)
- [Sim-to-Real Transfer](unit-sim-to-real.md)

**Guides**

- [Debugging RL Training](unit-debugging.md)

## After this course

!!! info "Follow-on: model alignment (separate course)"
    This course ends at ONNX vector policies. A planned sequel covers language-model alignment (SFT, preferences, RLHF/DPO). Unit 9 imitation learning is the on-ramp. Optional — Units 0–10 are complete on their own.

    Alignment course repository: *coming soon*.
