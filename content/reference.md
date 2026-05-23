# Reference

Quick lookup for `gdrl` CLI flags, `AIController` API, and ONNX export.

!!! note "Full reference"
    See `godot_rl_course_reference.html` for the complete plugin API reference.

## `gdrl` common flags

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment_name` | — | Name for logs and saved model |
| `--viz` | off | Show Godot window during training |
| `--timesteps` | 1 000 000 | Total environment steps |
| `--speedup` | 1 | Time scale multiplier (headless only) |
| `--n_parallel` | 1 | Number of parallel Godot instances |
| `--save_model_path` | — | Path to save trained `.zip` model |
| `--onnx_export_path` | — | Path to export `.onnx` for Godot inference |

## AIController lifecycle

```
_ready()          → register with Sync node
get_obs() → Array → return observation vector
set_action()      → receive and apply action
get_reward() → float → return scalar reward
get_done() → bool → return episode-end flag
```

## ONNX inference in Godot

1. Train and export: `--onnx_export_path=brain.onnx`
2. Copy `brain.onnx` into the Godot project folder
3. Sync node → **Control Mode** → `Onnx Inference`
4. Sync node → **Onnx Model Path** → path to `.onnx`
5. Play scene — no Python required
