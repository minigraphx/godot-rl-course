# Godot RL Course

Learn deep reinforcement learning by building and training agents inside real Godot game environments.

## What you'll build

| Phase | Content | What you learn |
|-------|---------|----------------|
| **Phase 1 — Foundations** | Setup · neural networks · RL loop · reward learning · deep dive · first custom env · reward design | Neurons to policies, how RL works, how to design rewards |
| **Phase 2 — Value-Based** | Q-Learning · DQN · curiosity | Bellman equation, Q-tables, DQN, sparse reward exploration |
| **Phase 3 — Policy-Based** | REINFORCE · Actor-Critic · PPO · SAC · Apply It · PPO from scratch (CleanRL) | Policy gradient theorem, PPO internals, continuous control |
| **Phase 4 — Scale & Complexity** | Parallel · 3D · Multi-agent · Memory | Real-world training at scale |
| **Phase 5 — Beyond Reward** | Multi-task RL · Imitation learning · ONNX/WASM · Capstone project | Alternative learning signals, generalist policies, shipping |
| **Phase 6 — Robotics** | Robot sensors · HER · sim-to-real · Safe RL | Robot observation/action design, goal-conditioned RL, deploying policies to hardware safely |
| **Guides** | Debugging · Advanced Evaluation · PBT · World Models | Systematic diagnosis, evaluation, hyperparameter AutoML, model-based RL |

## Three ways to see your AI (every unit)

| Channel | What it shows |
|---------|---------------|
| **Godot** | Agent behavior in the world |
| **TensorBoard** | Learning curves (`tensorboard --logdir=logs`) |
| **AIController source** | Observations, actions, and rewards you changed |

## Units

**Phase 1 — Foundations**

- [Unit 0 — Setup & First Run](unit-00.md)
- [Neural Foundations 1 — One Neuron](unit-neural-01.md)
- [Neural Foundations 2 — Tiny Networks](unit-neural-02.md)
- [RL Essentials](unit-01.md)
- [Neural Foundations 3 — Learn from Reward](unit-neural-03.md)
- [RL Foundations Deep Dive](unit-rl-foundations-deep.md)
- [Reward Engineering](unit-reward-engineering.md)
- [Unit 2 — Build Your First Env](unit-02.md)

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
- [Apply It — SAC vs PPO on JumperHard](unit-sac-applied.md)
- [PPO From Scratch (CleanRL)](unit-cleanrl.md)

**Phase 4 — Scale & Complexity**

- [Unit 5 — Parallel Training](unit-05.md)
- [Unit 6 — Continuous 3D](unit-06.md)
- [Visual Observations](unit-visual-observations.md)
- [Unit 7 — Multi-Agent](unit-07.md)
- [Unit 8 — Memory & POMDPs](unit-08.md)
- [Self-Play](unit-self-play.md)
- [Hierarchical RL](unit-hierarchical.md)

**Phase 5 — Beyond Reward**

- [Multi-Task RL](unit-multitask.md)
- [Unit 9 — Imitation Learning](unit-09.md)
- [Offline RL](unit-offline-rl.md)
- [Unit 10 — Ship Your Brain](unit-10.md)
- [Capstone Project](unit-capstone.md)

**Phase 6 — Robotics**

- [Robot Observations & Sensors](unit-robotics.md)
- [Goal-Conditioned RL & HER](unit-her.md)
- [Sim-to-Real Transfer](unit-sim-to-real.md)
- [Safe RL / Constrained MDPs](unit-safe-rl.md)

**Guides**

- [Debugging RL Training](unit-debugging.md)
- [Advanced Evaluation](unit-evaluation.md)
- [Population-Based Training](unit-pbt.md)
- [World Models / DreamerV3](unit-world-models.md)
- [Foundation Models for Control (VLA)](unit-foundation-models.md)

## After this course

!!! info "Follow-on: model alignment (separate course)"
    This course ends at ONNX vector policies. A planned sequel covers language-model alignment (SFT, preferences, RLHF/DPO). Unit 9 imitation learning is the on-ramp. Optional — Units 0–10 are complete on their own.

    Alignment course repository: *coming soon*.
