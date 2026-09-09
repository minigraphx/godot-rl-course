# Architecture

## Single-stack design: godot-native-rl

The course runs on **godot-native-rl**, a Godot addon whose training bridge is pure GDScript. It is *not* committed: `scripts/fetch-native-runner.sh` installs the pinned release into `examples/neural_foundations/game/addons/godot_native_rl/`, which is gitignored. It replaces the earlier C# `godot_rl_agents_plugin`, which required the Godot .NET edition and the .NET SDK. With the native stack, students use the **Standard Godot build (4.5+)** — no C#, no MSBuild, no NuGet.

**Training phase (local):**

- Godot runs the game environment and sends observations/rewards to Python over a local socket
- GDScript side: `NcnnAIController2D/3D` nodes + the `NcnnSync` node (training mode) — speaks godot-rl's socket wire protocol
- Python side: `godot-rl`'s `StableBaselinesGodotEnv` wraps the socket as a Gymnasium-compatible env (the Python bridge package is unchanged)

> **Known protocol-version skew.** `sync.gd` declares wire protocol **0.7** (tracking
> godot-rl 0.8.x); the course pins godot-rl **0.5.0**, which declares **0.3**. The handshake
> logs `NcnnSync: minor version mismatch (got 3, expected 7)` and then proceeds — the
> exchanged messages are compatible. Verified end-to-end: a 2048-step foundations-racer run
> completed and exported both the `.zip` checkpoint and the ONNX file. Cosmetic, but it is
> the first thing a student sees, so Unit 0 and Troubleshooting both name it.
- Training uses `stable-baselines3` or `cleanrl` (PPO / DQN) with PyTorch

**Inference phase (native):**

- Trained model exported to ONNX (mandatory inspection/parity artifact), converted to **ncnn**
- `NcnnSync` in inference mode runs the ncnn `.param`/`.bin` files through the addon's GDExtension — zero Python at runtime
- Platform scope: prebuilt runner libraries ship for **macOS arm64, Windows x86_64 and Linux x86_64** (plus iOS, Android, Web), which clears the multi-platform prerequisite #81 named for retiring the legacy path
- The libraries are not committed — they outweigh the rest of the repository. `scripts/fetch-native-runner.sh` installs the release pinned in `examples/neural_foundations/game/GODOT_NATIVE_RL_VERSION` and verifies its checksum

Key technologies: Godot 4.5+ (GDScript), godot-native-rl addon (pinned release — see `examples/neural_foundations/game/GODOT_NATIVE_RL_VERSION`), Python, `godot-rl` (socket bridge), `stable-baselines3`, PyTorch, ONNX, ncnn, TensorBoard.

**Migration status:** Setup and Unit 0 run fully on the native stack. Units from RL Essentials onward still use the legacy `godot_rl_agents_examples` (C# plugin) until migrated unit-by-unit — tracked in issue #71.

## Example-driven learning

The course does not treat examples as a side catalog. Each unit centers on one official example (or a student-built env that reuses the same patterns). Complexity increases step by step — see [example-progression.md](example-progression.md).

## Training workflow by phase

| Phase | Units | Godot mode | Python mode |
|-------|-------|------------|-------------|
| Explore | 0–2 | Editor open, Play Scene | SB3 trainer script or `gdrl` with visualization |
| Scale | 3–9 | Exported binary, `--headless` | `gdrl` without `--viz`, `n_parallel` |
| Ship | 10 | Native ncnn inference via NcnnSync | No Python at runtime |

## godot-native-rl stack (conceptual)

```
┌─────────────────────────────────────────┐
│  Godot scene (Standard build, GDScript) │
│  ├─ Environment (physics, visuals)      │
│  ├─ NcnnAIController (obs/action/rew)   │
│  └─ NcnnSync node (TCP ↔ Python)        │
└──────────────┬──────────────────────────┘
               │ observations, rewards
               ▼
┌─────────────────────────────────────────┐
│  Python (Conda env)                     │
│  ├─ godot-rl StableBaselinesGodotEnv    │
│  └─ PPO / DQN / RecurrentPPO            │
└──────────────┬──────────────────────────┘
               │ ONNX export → ncnn conversion
               ▼
┌─────────────────────────────────────────┐
│  Godot native inference                 │
│  └─ NcnnSync (inference mode) + ncnn    │
│     GDExtension — no Python at runtime  │
└─────────────────────────────────────────┘
```
