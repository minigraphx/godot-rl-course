# Curriculum

Godot-only deep RL course using **godot-rl-agents** examples in increasing complexity.

## First success (Unit 0, one session)

Students should leave Unit 0 with:

1. **Godot** — agent moving (hub binary or `--viz`)
2. **Python** — training log shows steps / rising `ep_rew_mean`
3. **TensorBoard** (optional same day) — a learning curve

Unit 0 remains the self-contained first-session success. The first reward tweak
and visible behavior change move to **RL Essentials** after Neural Foundations
2, where learners can connect the reward-driven update to the network they now
understand.

## Curriculum decision: neural-network foundations

**Status:** Published in Phase 1 navigation (see `content/unit-neural-*.md`).
Main course units still use `godot-rl-agents`; native migration tracked in
[#71](https://github.com/minigraphx/godot-rl-course/issues/71).

The course includes an integrated three-unit **Neural-Network Foundations
Track** immediately after Unit 0, split around a compact RL Essentials bridge.
Learners choose one practical path and view short comparison demonstrations
from the other:

1. one neuron — Python decision boundary or Godot enemy decision;
2. a tiny network — Python nonlinear classifier or Godot arena collector;
3. reward learning — Python REINFORCE point robot or a PPO-trained Godot racer.

Intended sequence:

1. Unit 0 — first successful training;
2. Neural Foundations 1–2;
3. RL Essentials — agent, environment, observations, actions, reward, policy,
   episodes, return, and intuitive exploration;
4. Neural Foundations 3;
5. RL Foundations Deep Dive — MC versus TD, bootstrapping, exploration
   mechanisms, and algorithm taxonomy;
6. Reward Engineering and current Unit 2 onward.

Current Unit 1 will be split across steps 3 and 5. This prevents Foundations 3
from introducing RL and neural networks simultaneously while avoiding a long
theory block before learners have a concrete policy to analyze.

The track is required for beginners and skippable through a diagnostic for
learners who already understand forward passes and backpropagation. It is not a
separate course because these concepts are prerequisites for DQN, policy
gradients, PPO, SAC, and ONNX inference.

The Game path uses `godot-native-rl` for C#-free training integration and native
ncnn deployment. ONNX is a mandatory inspection and parity stage between the
Python policy and ncnn. Existing units retain their current stack until a
separate migration project is approved.

The planned language-model sequel will reuse this track as a prerequisite and
map the concepts to embeddings, token logits, and transformer training rather
than duplicating the material.

See [neural-foundations-plan.md](neural-foundations-plan.md) for the overview
and the linked approved teaching design.

## Three ways to see your AI

Use in every unit so training never feels like a black box:

| Channel | What it shows | Typical mode |
|---------|---------------|--------------|
| **Godot** | Agent in the world | Editor or `--viz` (Units 0–2); occasional `--viz` after headless units |
| **TensorBoard** | Learning curves | `tensorboard --logdir=logs` from Unit 1 on |
| **AIController** | Obs, actions, rewards you changed | Open source each unit |

Optional stretch: inspect exported **ONNX** in [Netron](https://netron.app) before Unit 10.

## Units (aligned with HTML files)

| Unit | File | Title | Example | Algorithm |
|------|------|-------|---------|-----------|
| 0 | `Unit-00.html` | Setup & first run | BallChase | PPO (smoke test) |
| 1 | `Unit-01.html` | RL foundations + first tweak | BallChase | PPO |
| 2 | `Unit-02.html` | SimpleReachGoal warm-up → build Lunar Lander | SimpleReachGoal, Lander | PPO |
| 3 | `Unit-03.html` | CrossTheRoad & DQN | CrossTheRoad | DQN |
| 4 | `Unit-04.html` | JumperHard & PPO | JumperHard | PPO |
| 5 | `Unit-05.html` | Parallel training | BallChase (source) | PPO |
| 6 | `Unit-06.html` | Continuous 3D | FlyBy, HovercraftRacing | PPO |
| 7 | `Unit-07.html` | Multi-agent | Racer, MultiAgentSimple | PPO |
| 8 | `Unit-08.html` | Memory & POMDPs | FPS / RobotFPS | RecurrentPPO |
| 9 | `Unit-09.html` | Imitation learning | MultiLevelRobot | BC / GAIL |
| 10 | `Unit-10.html` | Ship your brain into the game | Any trained model | ONNX (+ optional HTML5/WASM) |

**Pacing:** ~10–12 weeks.

## Unit pacing notes

- **RL Essentials (from current Unit 1):** operational loop + one reward tweak
  before Foundations 3; target 45–60 min.
- **RL Foundations Deep Dive (from current Unit 1):** MC/TD, exploration, and
  taxonomy after Foundations 3; target 40–60 min.
- **Unit 2:** Phase A — run/tweak **SimpleReachGoal** hub or source; Phase B — build Lunar Lander from scratch.
- **Unit 5:** Same BallChase env, new skill — parallel rollouts (`n_parallel`), not a repeat lesson.
- **After headless units (3+):** Schedule a short **viz checkpoint** — re-run with `--viz` or editor to screenshot before/after behavior.

## Extensions

| Unit | Title |
|------|-------|
| 11 | CleanRL & Sample Factory |
| 12 | Capstone (stretch pool) |
| 13 | Self-play (AirHockey, RobotVolleyball) |

## After this course

This course teaches **Godot + godot-rl-agents + SB3/CleanRL → ONNX**. It does **not** include LLM APIs, Hugging Face training, or RLHF labs.

A separate follow-on course will cover **language-model alignment** (SFT, preference modeling, RLHF/DPO). **Unit 9** (imitation / BC) is the conceptual bridge. Finishing Units 0–10 here is a complete path; the sequel is optional.

| Godot RL | Alignment sequel |
|----------|------------------|
| Reward \(r_t\) | Preference / learned reward model |
| Policy \(\pi(a \mid s)\) | LM policy \(\pi(y \mid x)\) |
| Expert demos (Unit 9) | Supervised fine-tuning |
| PPO improvement | RLHF / DPO |

Alignment course repository: *coming soon* (separate repo when published).

## RL concepts checklist (authoring Units 4–10)

When writing each unit HTML, include:

| Unit | Required content |
|------|------------------|
| 4 | Hyperparameters (learning rate, `n_steps`, clip range); link to [training-stalled](Unit-03.html#training-stalled) callout |
| 5 | **Eval protocol**: fixed seed, deterministic policy, report mean episodic return over N episodes |
| 6 | **Normalization**: clip/scale observations and actions for continuous 3D |
| 9 | One sentence: BC demos ↔ SFT in alignment sequel (optional enrollment) |
| 10 | Load checkpoint and resume training; ONNX export |

Do not add LLM, Hugging Face, or RLHF content to Units 0–10.

## Training workflow

| Phase | Units | Godot | Python |
|-------|-------|-------|--------|
| Explore | 0–2 | Editor / `--viz` | gdrl, SB3 |
| Scale | 3–8 | Headless binary | gdrl, `n_parallel` |
| Ship | 9–10 | ONNX in Sync | No Python at runtime |

## Docs

- [example-progression.md](example-progression.md)
- [architecture.md](architecture.md)
- [html-units.md](html-units.md)
- [neural-foundations-plan.md](neural-foundations-plan.md)
