# Example Progression Ladder

The course teaches **godot-rl-agents** by walking through official examples in order of increasing complexity.

Source: [godot_rl_agents_examples](https://github.com/edbeeching/godot_rl_agents_examples). Hub binaries: `gdrl.env_from_hub`.

## Step → unit mapping

| Step | Example | HTML unit | New skill |
|------|---------|-----------|-----------|
| 0 | — | Unit 0 | Conda, plugin, BallChase run (first success checklist) |
| 1 | — | Neural Foundations 1–2 | One neuron, tiny MLP, visible forward pass and gradients |
| 2 | BallChase | RL Essentials | MDP vocabulary + **one reward tweak** |
| 3 | — | Neural Foundations 3 | REINFORCE point robot or PPO racer + ONNX/ncnn parity |
| 4 | — | RL Foundations Deep Dive | MC/TD, exploration, algorithm taxonomy |
| 5 | SimpleReachGoal | Unit 2 (Phase A) | Hub/source run, tweak sensor or reward |
| 6 | Lunar Lander | Unit 2 (Phase B) | Full scene from scratch |
| 7 | CrossTheRoad | Unit 3 | DQN, sparse 2D rewards |
| 8 | JumperHard | Unit 4 | PPO benchmark, headless export |
| 9 | BallChase (source) | Unit 5 | Parallel instances |
| 10 | FlyBy / HovercraftRacing | Unit 6 | Continuous 3D |
| 11 | Racer | Unit 7 | Mixed actions |
| 12 | MultiAgentSimple | Unit 7 | Multi-agent |
| 13 | FPS / RobotFPS | Unit 8 | RecurrentPPO |
| 14 | MultiLevelRobot | Unit 9 | Imitation learning |
| 15 | Any model | Unit 10 | ONNX + WASM |

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
| `content/unit-00.md` | Done |
| `content/unit-neural-01.md` | Done |
| `content/unit-neural-02.md` | Done |
| `content/unit-01.md` | Done (RL Essentials) |
| `content/unit-neural-03.md` | Done |
| `content/unit-rl-foundations-deep.md` | Done |
| `content/unit-02.md` | Done |
| `content/unit-03.md` | Done |
| `content/unit-04.md` | Done |
| `content/unit-05.md` | Done |
| `content/unit-06.md` | Done |
| `content/unit-07.md` | Done |
| `content/unit-08.md` | Done |
| `content/unit-09.md` | Done |
| `content/unit-10.md` | Done |
