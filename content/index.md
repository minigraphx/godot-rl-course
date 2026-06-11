# Godot RL Course

Learn deep reinforcement learning by building and training agents inside real Godot game environments.

## Before you start

**Who this course is for.** Developers who want to train game AI with reinforcement learning. You need basic Python (functions, loops, running scripts) and comfort with a terminal. No prior Godot, machine-learning, or RL experience is required — neurons, networks, and the RL loop are built from scratch in Phase 1. The math stays at high-school algebra, and every formula is walked through with concrete numbers first.

**What you need.** A desktop or laptop running macOS, Windows, or Linux. A GPU helps but is not required — early units train in minutes on CPU, and each unit's time box lists CPU and GPU estimates. See the [Hardware Setup Guide](hardware-setup.md) before buying anything.

**Time commitment.** Every unit opens with a time estimate. Unit 0 plus your first neuron fits in [one evening (~2½–3 h)](unit-00.md#first-evening-script); most units take 1–3 hours of attention, with longer training runs happening in the background.

**Where to start.** Complete [Setup](setup.md) once — clone the course repo, install Godot and the Python environment. Then begin with [Unit 0](unit-00.md), your first training run.

## What you'll build

| Phase | Content | What you learn |
|-------|---------|----------------|
| **Phase 1 — Foundations** | Setup · neural networks · RL loop · reward learning · deep dive · first custom env · reward design | Neurons to policies, how RL works, how to design rewards |
| **Phase 2 — Value-Based** | Q-Learning · DQN · curiosity | Bellman equation, Q-tables, DQN, sparse reward exploration |
| **Phase 3 — Policy-Based** | REINFORCE · Actor-Critic · PPO · SAC · Apply It · PPO from scratch (CleanRL) | Policy gradient theorem, PPO internals, continuous control |
| **Phase 4 — Scale & Complexity** | Parallel · 3D · Multi-agent · Memory | Real-world training at scale |
| **Phase 5 — Beyond Reward** | Multi-task RL · Imitation learning · RLHF · Offline RL · Decision Transformer · ONNX/WASM · Capstone project | Alternative learning signals, generalist policies, shipping |
| **Phase 6 — Robotics** | Robot sensors · Locomotion · Diffusion Policy · HER · sim-to-real · Safe RL | Robot observation/action design, goal-conditioned RL, deploying policies to hardware safely |
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
- [RLHF & Preference Learning](unit-rlhf.md)
- [Offline RL](unit-offline-rl.md)
- [Decision Transformer](unit-decision-transformer.md)
- [Unit 10 — Ship Your Brain](unit-10.md)
- [Capstone Project](unit-capstone.md)

**Phase 6 — Robotics**

- [Robot Observations & Sensors](unit-robotics.md)
- [Locomotion Agents (Walker / Crawler / Worm)](unit-locomotion.md)
- [Diffusion Policy](unit-diffusion-policy.md)
- [Goal-Conditioned RL & HER](unit-her.md)
- [Sim-to-Real Transfer](unit-sim-to-real.md)
- [Safe RL / Constrained MDPs](unit-safe-rl.md)

**Guides**

- [Debugging RL Training](unit-debugging.md)
- [Advanced Evaluation](unit-evaluation.md)
- [Experiment Tracking (W&B / MLflow)](unit-experiment-tracking.md)
- [GPU-Accelerated Environments](unit-gpu-envs.md)
- [Population-Based Training](unit-pbt.md)
- [World Models / DreamerV3](unit-world-models.md)
- [Foundation Models for Control (VLA)](unit-foundation-models.md)

## After this course

!!! info "Follow-on: model alignment (separate course)"
    This course ends at ONNX vector policies. A planned sequel covers language-model alignment (SFT, preferences, RLHF/DPO). Unit 9 imitation learning is the on-ramp. Optional — Units 0–10 are complete on their own.

    Alignment course repository: *coming soon*.
