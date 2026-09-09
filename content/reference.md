# Reference

Quick lookup for the SB3 training script's flags, the `AIController` API, and ONNX export.

!!! note "Full reference"
    See `godot_rl_course_reference.html` for the complete plugin API reference.

## The training script — not `gdrl`

The flags below belong to **`stable_baselines3_example.py`**, which lives in the godot-rl
*repository* and is **not** part of the `pip install`. The installed `gdrl` command accepts
none of them: it trains for a fixed 200 000 steps and saves nothing. Upstream deprecated it —
running it prints *"This use of gdrl is deprecated and will be removed in version 1.0, please
refer to the examples in the github repo."*

Fetch the script once, into the folder you train from:

```bash
curl -O https://raw.githubusercontent.com/edbeeching/godot_rl_agents/main/examples/stable_baselines3_example.py
```

Then every training command in this course reads:

```bash
python stable_baselines3_example.py --experiment_name=my-run --viz
```

## Common training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--env_path` | — | Exported Godot binary. Omit it to train against the editor (press F6) |
| `--experiment_name` | `experiment` | Name shown in TensorBoard |
| `--experiment_dir` | `logs/sb3` | Where TensorBoard logs are written |
| `--viz` | off | Show a Godot window during training |
| `--timesteps` | 1 000 000 | Total environment steps |
| `--speedup` | 1 | Physics time-scale multiplier |
| `--n_parallel` | 1 | Number of parallel Godot instances (needs `--env_path`) |
| `--save_model_path` | — | Path to save the trained `.zip` model |
| `--onnx_export_path` | — | Path to export `.onnx` for Godot inference |
| `--resume_model_path` | — | Continue from a previously saved model |
| `--inference` | off | Run a loaded model instead of training |
| `--n_steps` | 64 | Steps per environment per update |
| `--batch_size` | 64 | Minibatch size |
| `--learning_rate` | 0.0003 | Learning rate |
| `--ent_coef` | 0.0001 | Entropy coefficient |
| `--clip_range` | 0.2 | PPO clipping range |

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
