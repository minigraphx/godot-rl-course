# Architecture

## Hybrid Training / Inference Model

**Training phase (local):**
- Godot runs the game environment and sends observations/rewards to Python over a local socket
- GDScript side: `AIController` node + `godot_rl_agents_plugin` Sync node
- Python side: `gdrl.env_from_hub(...)` or an exported binary path wraps the socket as a Gymnasium-compatible env
- Training uses `stable-baselines3` or `cleanrl` (PPO / DQN) with PyTorch

**Inference phase (web):**
- Trained model exported to ONNX, loaded back into Godot
- Project exported to HTML5/WASM for browser-based play with zero Python dependency

Key technologies: Godot 4 (GDScript / C#), `godot-rl-agents` plugin, Python, `stable-baselines3`, PyTorch, ONNX, TensorBoard.

## Example-driven learning

The course does not treat examples as a side catalog. Each unit centers on one official example (or a student-built env that reuses the same patterns). Complexity increases step by step — see [example-progression.md](example-progression.md).

## Training workflow by phase

| Phase | Units | Godot mode | Python mode |
|-------|-------|------------|-------------|
| Explore | 0–2 | Editor open, optional `--viz` | `gdrl` or SB3 script with visualization |
| Scale | 3–9 | Exported binary, `--headless` | `gdrl` without `--viz`, `n_parallel` |
| Ship | 10 | ONNX inference in Sync node | No Python at runtime |

## godot-rl-agents stack (conceptual)

```
┌─────────────────────────────────────┐
│  Godot scene                        │
│  ├─ Environment (physics, visuals)  │
│  ├─ AIController (obs/action/rew)   │
│  └─ Sync node (TCP ↔ Python)       │
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
