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
- **Unit 5:** Same BallChase env, new skill — parallel rollouts (`n_parallel`), not a repeat lesson.
- **After headless units (3+):** Schedule a short **viz checkpoint** — re-run with `--viz` or editor to screenshot before/after behavior.

## Extensions

| Unit | Title |
|------|-------|
| 11 | CleanRL & Sample Factory |
| 12 | Capstone (stretch pool) |
| 13 | Self-play (AirHockey, RobotVolleyball) |

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
