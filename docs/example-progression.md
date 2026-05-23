# Example Progression Ladder

The course teaches **godot-rl-agents** by walking through official examples in order of increasing complexity.

Source: [godot_rl_agents_examples](https://github.com/edbeeching/godot_rl_agents_examples). Hub binaries: `gdrl.env_from_hub`.

## Step → unit mapping

| Step | Example | HTML unit | New skill |
|------|---------|-----------|-----------|
| 0 | — | Unit 0 | Conda, plugin, BallChase run (first success checklist) |
| 1 | BallChase | Unit 1 | MDP vocabulary + **one reward tweak** |
| 2 | SimpleReachGoal | Unit 2 (Phase A) | Hub/source run, tweak sensor or reward |
| 3 | Lunar Lander | Unit 2 (Phase B) | Full scene from scratch |
| 4 | CrossTheRoad | Unit 3 | DQN, sparse 2D rewards |
| 5 | JumperHard | Unit 4 | PPO benchmark, headless export |
| 6 | BallChase (source) | Unit 5 | Parallel instances |
| 7 | FlyBy / HovercraftRacing | Unit 6 | Continuous 3D |
| 8 | Racer | Unit 7 | Mixed actions |
| 9 | MultiAgentSimple | Unit 7 | Multi-agent |
| 10 | FPS / RobotFPS | Unit 8 | RecurrentPPO |
| 11 | MultiLevelRobot | Unit 9 | Imitation learning |
| 12 | Any model | Unit 10 | ONNX + WASM |

## Per-unit rhythm

1. Run hub binary or exported build (headless from Unit 3+)
2. Open source in Godot .NET
3. Read `AIController` → Sync → training scene
4. Tweak one reward or sensor
5. Retrain and compare in **three views**: Godot behavior · TensorBoard · what changed in code

## Viz checkpoints (Units 3+)

Default training is headless for speed. After each major unit, spend ~5 minutes with `--viz` or the editor to confirm behavior matches the curve — avoids “TensorBoard up, game invisible” fatigue.

## Training mode

| Units | Default |
|-------|---------|
| 0–2 | Editor or `--viz` |
| 3–8 | Headless exported binary |
| 9–10 | ONNX inference in Godot |

## Stretch capstone pool

Ships · ZombieGame · ItemSortingCart · AirHockey · VirtualCamera

## File status

| File | Status |
|------|--------|
| `Unit-00.html` | Done |
| `Unit-01.html` | Done |
| `Unit-02.html` | Done |
| `Unit-03.html` | Done |
| `Unit-04.html`–`Unit-10.html` | Planned |
