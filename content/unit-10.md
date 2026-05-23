# Unit 10 — Ship Your Brain

Your policy is trained. Now get it **out of Python and into Godot** — running at full speed without a Python process, as a self-contained game. Learn the ONNX export pipeline, load and resume checkpoints, and optionally publish a playable HTML5 demo.

[← Unit 9: Imitation Learning](unit-09.md) · [Course home](index.md)

---

!!! info "Three ways to see your AI"
    Godot (inference mode — no Python running, agent plays live) · ONNX inspector (Netron — visualise the exported graph) · HTML5 export (share a URL, let anyone play against your agent)

---

## 1 · The shipping pipeline

```
Python training  →  .zip checkpoint  →  ONNX export  →  Godot inference
                        ↕
                   resume training
```

**ONNX (Open Neural Network Exchange)** is a standard model format. Godot RL Agents ships a GDScript ONNX runtime — no Python required at inference time.

---

## 2 · Save and resume checkpoints

Before exporting, make sure you have a final saved model. You can also resume training from any checkpoint if you interrupted a run.

```bash
# Save with checkpoints every 100k steps and export ONNX at the end
gdrl --env_path=./BallChase.x86_64 \
  --experiment_name=ballchase_final \
  --timesteps=1_000_000 \
  --save_model_path=ballchase_final \
  --save_checkpoint_frequency=100000 \
  --onnx_export_path=ballchase_final.onnx \
  --n_parallel=8 \
  --speedup=20
```

```bash
# Resume from a checkpoint if training was interrupted
gdrl --env_path=./BallChase.x86_64 \
  --resume_model_path=ballchase_final.zip \
  --experiment_name=ballchase_resumed \
  --timesteps=500_000 \
  --onnx_export_path=ballchase_final.onnx
```

The `--resume_model_path` flag loads weights, optimizer state, and step count — training continues exactly where it left off.

---

## 3 · Export ONNX from a saved model

If you already have a `.zip` model and want to export ONNX separately:

```python
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="./BallChase.x86_64", n_parallel=1, speedup=1)
model = PPO.load("logs/sb3/ballchase_final/best_model", env=env)

# Export to ONNX
model.policy.to("cpu")
import torch

obs = torch.zeros(1, *env.observation_space.shape)
torch.onnx.export(
    model.policy,
    obs,
    "ballchase_final.onnx",
    input_names=["obs"],
    output_names=["action"],
    opset_version=15,
)
print("Exported: ballchase_final.onnx")
env.close()
```

Alternatively, use the built-in gdrl export:

```bash
gdrl --env_path=./BallChase.x86_64 \
  --resume_model_path=ballchase_final.zip \
  --onnx_export_path=ballchase_final.onnx \
  --timesteps=0
```

Setting `--timesteps=0` skips training and just exports.

---

## 4 · Inspect the ONNX graph (optional)

Install [Netron](https://netron.app/) — a browser-based ONNX viewer — and drag your `.onnx` file into it. You'll see:

- Input node: observation shape
- MLP layers: weight matrices and activation functions
- Output node: action distribution parameters

This is useful for debugging shape mismatches when switching between SB3 versions.

---

## 5 · Load ONNX in Godot

1. Copy `ballchase_final.onnx` into your Godot project folder (e.g., `res://models/`)
2. Select the `Sync` node in your scene
3. Set `Control Mode` to `ONNX_INFERENCE`
4. Set `Onnx Model Path` to `res://models/ballchase_final.onnx`
5. Run the scene — the agent plays without any Python process

```
Sync node properties:
  Control Mode:      ONNX_INFERENCE
  Onnx Model Path:   res://models/ballchase_final.onnx
  Speed Up:          1          ← real-time, not accelerated
```

The agent runs at game speed. You can add human-controlled characters, obstacles, or UI around the AI agent — it's now just another Godot node.

---

## 6 · Export for desktop

Export a standard desktop binary with the AI baked in:

1. Project → Export → Add preset (Linux / Windows / macOS)
2. Resources tab: make sure `*.onnx` files are included in the export filter
3. Export → Export Project (not PCK)

The result is a standalone executable — no Python, no conda environment needed.

---

## 7 · HTML5 / WASM export (optional)

Godot can export to WebAssembly, making the game playable in a browser. The ONNX runtime works in WASM.

**Requirements:**

- Godot 4.x with the HTML5 export template installed
- A web server (GitHub Pages, itch.io, Netlify — all work)

**Steps:**

1. Project → Export → Add preset → Web
2. Enable `Export Type: Release`
3. Resources: include `*.onnx`
4. Export → Export Project → choose `index.html` output path
5. Upload the output folder to your web server

!!! warning "SharedArrayBuffer"
    HTML5 Godot exports require `SharedArrayBuffer`, which needs specific HTTP headers (`Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Embedder-Policy: require-corp`). GitHub Pages supports this; basic file hosting does not. Check your host's documentation.

**Publish on itch.io:**

1. Create a project on [itch.io](https://itch.io)
2. Kind: HTML
3. Upload the exported folder as a zip
4. Check "This file will be played in the browser"
5. Share the URL

Anyone with a browser can now watch — and interact with — your trained agent.

---

## 8 · Swap agents at runtime (advanced)

You can load different ONNX models at runtime using GDScript:

```gdscript
# In a scene script — swap agent brain on button press
@onready var sync_node = $Sync

func _on_swap_pressed():
    sync_node.onnx_model_path = "res://models/alternative_brain.onnx"
    sync_node.reload_model()
```

This lets you build "replay demos" that switch between a random policy, a BC clone, and a fine-tuned PPO agent — all in one scene, no Python restarts.

---

## 9 · Viz checkpoint

Run your exported binary or HTML5 build for 5 minutes:

- Is the agent's behavior identical to what you saw during training? (It should be — ONNX is deterministic)
- Frame rate stable? ONNX inference adds < 1ms per step for MLP policies
- For HTML5: test on mobile and a low-end laptop — the WASM runtime is leaner than desktop Godot

---

## 10 · Stretch goals

- **A/B test in-browser** — export two models (PPO vs BC fine-tune from Unit 9); build a Godot UI that lets users switch between them and vote on which looks more natural
- **Quantize the ONNX model** — use `onnxruntime` tools to reduce the model to INT8; measure size and speed difference
- **Continuous deployment** — set up a GitHub Actions workflow: push new training results → auto-export ONNX → deploy HTML5 to GitHub Pages

---

## What's next

You've completed the course. Here's what you've built:

| Unit | Skill |
|------|-------|
| 0 | Run a Godot RL environment |
| 1 | Understand the agent–environment loop |
| 2 | Build a custom Godot RL env from scratch |
| 3 | DQN for sparse discrete tasks |
| 4 | PPO hyperparameter tuning |
| 5 | Parallel training at scale |
| 6 | Continuous action spaces + normalization |
| 7 | Multi-agent: cooperative and competitive |
| 8 | Memory and POMDPs with RecurrentPPO |
| 9 | Imitation learning: BC and GAIL |
| 10 | ONNX export + Godot inference + HTML5 |

**What comes next:** The alignment sequel builds directly on Unit 9's imitation learning foundation — moving from BC on game demonstrations to SFT on language, then RLHF, reward modelling, and Constitutional AI.

[← Back to course home](index.md)
