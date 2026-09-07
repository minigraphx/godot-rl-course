# Architecture

The course is mid-migration from a two-runtime, .NET-dependent stack
(**godot-rl-agents**) to a single GDScript+native stack (**godot-native-rl**,
ncnn inference). Both are documented below: the **current stack** is what the
units run today; the **target stack** is the design the migration is moving to
(tracked in issues #81 / #71).

## Example-driven learning

The course does not treat examples as a side catalog. Each unit centers on one
official example (or a student-built env that reuses the same patterns).
Complexity increases step by step — see [example-progression.md](example-progression.md).

---

## Target stack — single-stack native (godot-native-rl)

The goal of the migration is to remove the Godot **.NET edition** and the C# /
MSBuild bridge entirely. The native runner is statically-linked C++ (ncnn) with
no C#/.NET and no external runtime, so the **standard Godot build** suffices.

**Toolchain**

- **Godot 4.5+, standard build** — no Mono/.NET edition, no .NET SDK.
- **Godot Native RL** addon — installed from the Godot **AssetLib** (or the
  release `.zip`), enabled under Project → Project Settings → Plugins. The course
  does not vendor the addon or its binaries; students install the pinned release.
- Prebuilt ncnn runner binaries ship for **Linux x86_64, macOS arm64,
  Windows x86_64, and Web/WASM** (all CI-verified as of plugin `v0.3.1`);
  Android/iOS are in progress.

**Training phase (local)**

- Godot runs the environment with an `NcnnSync` node (training mode) that speaks
  the `godot_rl` socket wire protocol — the same observation/action/reward
  exchange as before, minus the C# Sync node.
- Python side: Stable-Baselines3 (PPO / SAC / …) over the socket bridge. The
  exact course Python pin for the native bridge is settled per-unit during the
  migration (#71).

**Inference phase (native + web)**

- The trained policy is exported to **ncnn** (PyTorch → ncnn; the Neural
  Foundations path goes PyTorch → ONNX → ncnn so the graph stays inspectable).
- Godot loads the `.ncnn` model via the addon's `NcnnRunner` and runs inference
  natively — including HTML5/WASM export — with **no Python and no .NET at
  runtime**.

```
┌─────────────────────────────────────────────┐
│  Godot (standard build, 4.5+)               │
│  ├─ Environment (physics, visuals)          │
│  ├─ Sensors / reward (declarative, GDScript)│
│  └─ NcnnSync  (training: TCP ↔ Python)      │
│              (inference: native ncnn runner)│
└──────────────┬──────────────────────────────┘
               │ observations, rewards (training only)
               ▼
┌─────────────────────────────────────────────┐
│  Python (Conda env)                         │
│  └─ Stable-Baselines3  (PPO / SAC / …)      │
└──────────────┬──────────────────────────────┘
               │ export_to_ncnn.py  (TorchScript → ncnn)
               ▼
┌─────────────────────────────────────────────┐
│  Godot native inference (NcnnRunner)        │
│  └─ desktop + HTML5 / WASM, no runtime deps │
└─────────────────────────────────────────────┘
```

The Neural Foundations 3 **Game path** already uses this stack (PPO racer,
PyTorch → ONNX → ncnn parity). The migration extends it to Units 0–10, reusing
the native example environments the plugin now ships (`ball_chase`,
`chase_the_target`, `fly_by`, `3dball`, `quadruped_walk`, …).

---

## Current stack — hybrid godot-rl-agents (.NET)

This is what Units 0–10 run today, until each is migrated.

**Training phase (local):**
- Godot runs the game environment and sends observations/rewards to Python over a local socket
- GDScript side: `AIController` node + `godot_rl_agents_plugin` Sync node (C#)
- Python side: `gdrl.env_from_hub(...)` or an exported binary path wraps the socket as a Gymnasium-compatible env
- Training uses `stable-baselines3` or `cleanrl` (PPO / DQN) with PyTorch

**Inference phase (web):**
- Trained model exported to ONNX, loaded back into Godot
- Project exported to HTML5/WASM for browser-based play with zero Python dependency

Key technologies: Godot 4 (.NET edition, GDScript / C#), `godot-rl-agents` plugin, Python, `stable-baselines3`, PyTorch, ONNX, TensorBoard.

```
┌─────────────────────────────────────┐
│  Godot scene (.NET edition)         │
│  ├─ Environment (physics, visuals)  │
│  ├─ AIController (obs/action/rew)   │
│  └─ Sync node (C#, TCP ↔ Python)    │
└──────────────┬──────────────────────┘
               │ observations, rewards
               ▼
┌─────────────────────────────────────┐
│  Python (Conda env)                 │
│  ├─ StableBaselinesGodotEnv         │
│  └─ PPO / DQN / RecurrentPPO        │
└──────────────┬──────────────────────┘
               │ ONNX export (after training)
               ▼
┌─────────────────────────────────────┐
│  Godot inference (Sync → ONNX path) │
│  └─ HTML5 / WASM export             │
└─────────────────────────────────────┘
```

---

## Training workflow by phase

| Phase | Units | Godot mode | Python mode |
|-------|-------|------------|-------------|
| Explore | 0–2 | Editor open, optional `--viz` | `gdrl` or SB3 script with visualization |
| Scale | 3–9 | Exported binary, `--headless` | `gdrl` without `--viz`, `n_parallel` |
| Ship | 10 | ncnn inference in the native runner | No Python at runtime |
