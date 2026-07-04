# Architecture

## Single-stack design: godot-native-rl

The course runs on **godot-native-rl** — a pure-GDScript Godot addon bundled with the course repo (`examples/neural_foundations/game/addons/godot_native_rl/`). It replaces the earlier C# `godot_rl_agents_plugin`, which required the Godot .NET edition and the .NET SDK. With the native stack, students use the **Standard Godot build (4.5+)** — no C#, no MSBuild, no NuGet.

**Training phase (local):**

- Godot runs the game environment and sends observations/rewards to Python over a local socket
- GDScript side: `NcnnAIController2D/3D` nodes + the `NcnnSync` node (training mode) — speaks the same wire protocol (0.3) as godot-rl
- Python side: `godot-rl`'s `StableBaselinesGodotEnv` wraps the socket as a Gymnasium-compatible env (the Python bridge package is unchanged)
- Training uses `stable-baselines3` or `cleanrl` (PPO / DQN) with PyTorch

**Inference phase (native):**

- Trained model exported to ONNX (mandatory inspection/parity artifact), converted to **ncnn**
- `NcnnSync` in inference mode runs the ncnn `.param`/`.bin` files through a bundled GDExtension — zero Python at runtime
- Platform scope: ncnn runner binaries currently ship for **macOS arm64 only**; Windows/Linux runners are a prerequisite for retiring the legacy path entirely (tracked in #71/#81)

Key technologies: Godot 4.5+ (GDScript), godot-native-rl addon (pinned commit — see `examples/neural_foundations/game/GODOT_NATIVE_RL_VERSION`), Python, `godot-rl` (socket bridge), `stable-baselines3`, PyTorch, ONNX, ncnn, TensorBoard.

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
