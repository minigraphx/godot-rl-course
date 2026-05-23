# Curriculum

Godot-only deep RL course using **godot-rl-agents** examples in increasing complexity.

## Units (aligned with HTML files)

| Unit | File | Title | Example | Algorithm |
|------|------|-------|---------|-----------|
| 0 | `Unit-00.html` | Setup & first run | BallChase | PPO (smoke test) |
| 1 | `Unit-01.html` | RL foundations | BallChase recap | — |
| 2 | `Unit-02.html` | Build Lunar Lander | SimpleReachGoal patterns | PPO |
| 3 | `Unit-03.html` | CrossTheRoad & DQN | CrossTheRoad | DQN |
| 4 | `Unit-04.html` | JumperHard & PPO | JumperHard | PPO |
| 5 | `Unit-05.html` | Parallel training | BallChase (source) | PPO |
| 6 | `Unit-06.html` | Continuous 3D | FlyBy, HovercraftRacing | PPO |
| 7 | `Unit-07.html` | Multi-agent | Racer, MultiAgentSimple | PPO |
| 8 | `Unit-08.html` | Memory & POMDPs | FPS / RobotFPS | RecurrentPPO |
| 9 | `Unit-09.html` | Imitation learning | MultiLevelRobot | BC / GAIL |
| 10 | `Unit-10.html` | Deploy to web | Any trained model | ONNX |

**Pacing:** ~10–12 weeks.

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
