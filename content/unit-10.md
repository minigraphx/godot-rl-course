# Unit 10 — Ship Your Brain

Your policy is trained. Now get it **out of Python and into Godot** — running at full speed without a Python process, as a self-contained game. Learn the ONNX export pipeline, load and resume checkpoints, and optionally publish a playable HTML5 demo.

[← Offline RL](unit-offline-rl.md) · [Course home](index.md)

!!! note "Prerequisites"
    - **Any trained agent from Units 2–9** — you need a `.zip` SB3 checkpoint to export
    - **[Unit 2](unit-02.md)** §11 already showed the first ONNX export — this unit makes it production-shaped
    - Comfort with Godot export presets (desktop and/or HTML5)
    - No PyTorch internals needed; no ONNX background required

!!! info "Time"
    Reading: ~20 min · Training: ~10 min GPU / ~30 min CPU

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

!!! check "Done when"
    The scene runs in `ONNX_INFERENCE` mode and the agent plays competently with **no Python process alive** — no `gdrl` in your process list, no conda environment activated, nothing to Ctrl-C. At the Section 9 viz checkpoint the behavior matches what you saw at the end of training (ONNX is deterministic, so it should look identical). An agent that jitters randomly or freezes points to a wrong model path or an observation-shape mismatch (Section 4), not a broken runtime.

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

**What comes next:** Put everything together in the capstone — pick your own environment, design your own reward, train and ship it.

!!! info "Self-check before you move on"
    Can you answer these in your own words?

    1. Why does the shipping path go through ONNX instead of loading the SB3 `.zip` directly in Godot?
    2. What three things must match between training and inference so the policy doesn't silently produce garbage actions?
    3. What's the difference between a *resume-training* checkpoint and an *export-for-inference* model — what state does each contain?
    4. When would you choose HTML5/WASM export over a desktop binary, and what does that cost the agent?
    5. After export, how would you sanity-check that the Godot-side inference matches the Python-side training rollouts?

    If you can answer all five — you're ready for the capstone.

??? success "Self-check answers"
    1. The SB3 `.zip` is a PyTorch artifact — loading it requires a live Python process. **ONNX** is a standard interchange format, and Godot RL Agents ships a GDScript ONNX runtime, so the shipped game runs the policy with no Python at all (Section 1).
    2. The **observation shape**, the action space, and the meaning/order of the values your Godot env feeds the model. If any of these drift between training and the shipped scene, the graph still runs — it just maps garbage in to garbage out. Netron (Section 4) catches the shape part.
    3. A *resume-training* checkpoint (`--resume_model_path`) contains **weights, optimizer state, and step count**, so training continues exactly where it stopped (Section 2). The exported ONNX contains only the policy's forward pass — enough to act, not enough to keep learning.
    4. Choose **HTML5/WASM** when you want anyone to play via a URL (itch.io, GitHub Pages) without installing anything. It costs hosting constraints (`SharedArrayBuffer` headers) and performance headroom — which is why the viz checkpoint tells you to test on mobile and a low-end laptop.
    5. Run the Section 9 **viz checkpoint**: play the exported build for several minutes and compare against your training rollouts. ONNX inference is deterministic, so the behavior should be identical — visible drift signals an export problem, not randomness.

[→ Capstone Project](unit-capstone.md) · [← Back to course home](index.md)
