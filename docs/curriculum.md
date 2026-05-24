# Curriculum

Godot-only deep RL course using **godot-rl-agents** examples in increasing complexity.

## First success (Unit 0, one session)

Students should leave Unit 0 with:

1. **Godot** — agent moving (hub binary or `--viz`)
2. **Python** — training log shows steps / rising `ep_rew_mean`
3. **TensorBoard** (optional same day) — a learning curve
4. **Ownership** (Unit 1) — one reward tweak with visible behavior change

**Timed path:** [Unit-00.html#first-evening](../Unit-00.html#first-evening) — ~2½–3 h covering Unit 0 + Unit 1 fast path in one sitting.

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

- **Unit 1:** Skim MDP loop (~15 min) → tweak one BallChase reward → read deeper theory while training runs.
- **Unit 2:** Phase A — run/tweak **SimpleReachGoal** hub or source; Phase B — build Lunar Lander from scratch.
- **After Unit 3 (DQN):** If CrossTheRoad reward curve is flat → assign the **Curiosity unit** (Intrinsic Motivation & RND) before Unit 4. Sparse-reward diagnosis is the direct motivation; students should read it while their headless training runs.
- **Unit 5:** Same BallChase env, new skill — parallel rollouts (`n_parallel`), not a repeat lesson.
- **After headless units (3+):** Schedule a short **viz checkpoint** — re-run with `--viz` or editor to screenshot before/after behavior.

## Extensions

| Unit | Title |
|------|-------|
| 11 | CleanRL & Sample Factory |
| 12 | Capstone (stretch pool) |
| 13 | Self-play (AirHockey, RobotVolleyball) |
| 14 | Locomotion Agents — Walker, Crawler, Worm (AI Warehouse parity) |

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
