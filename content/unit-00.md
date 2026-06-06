# Unit 0 — Setup & First Run

!!! info "Time"
    Reading: ~20 min · Training: ~15 min GPU / ~1 h CPU

Install the Godot .NET editor, Python toolchain, and godot-rl-agents plugin. Run your first training session with the **BallChase** example and confirm the Godot ↔ Python socket works.

---

!!! success "First success (one session)"
    1. **Godot** — agent moving (hub binary or `--viz`)
    2. **Python** — terminal shows steps; `ep_rew_mean` trends up
    3. **TensorBoard** (optional) — `tensorboard --logdir=logs` shows a curve
    4. **Neural Foundations** — you will build visible neurons before the RL loop

!!! info "Three ways to see your AI (every unit)"
    Godot behavior · TensorBoard curves · what you change in `AIController`

!!! tip "Short on time?"
    Follow the [first evening script](#first-evening-script) (~2½–3 h) to finish Unit 0 and start Neural Foundations 1 in one sitting.

---

## 1 · Split architecture

Two runtimes talk over a local socket — Godot sends observations and receives actions; Python runs the training loop.

| Component | Role | Runtime |
|-----------|------|---------|
| **Godot** | Physics, observations, rewards | Godot 4 .NET + GDScript |
| **Plugin (C#)** | Sync node, ONNX bridge | .NET / MSBuild |
| **Python** | PPO / DQN training (SB3) | Conda, Python 3.10 |

!!! warning "Godot .NET edition required"
    The standard Godot build cannot load the plugin's C# / NuGet dependencies. Download the **.NET** build from [godotengine.org](https://godotengine.org) and install the [.NET SDK](https://dotnet.microsoft.com/download).

---

## 2 · Conda environment

!!! info "First time here?"
    Full installation instructions — Miniconda, `godot_env`, and the Godot plugin — are in [Setup](setup.md). Complete that page first, then return here.

Quick reminder for every new terminal:

```bash
conda activate godot_env
```

---

## 3 · Godot project & plugin

**Option A — Hub binary (fastest smoke test)**

```bash
python -c "from godot_rl.env_from_hub import env_from_hub; env_from_hub('edbeeching/godot_rl_BallChase')"
chmod +x examples/godot_rl_BallChase/bin/BallChase.x86_64
```

On macOS, use the downloaded binary path in training commands below.

**Option B — Open example source (recommended for learning)**

1. Clone [godot_rl_agents_examples](https://github.com/edbeeching/godot_rl_agents_examples)
2. Godot → Import → `examples/BallChase` → open `project.godot`
3. Project → Project Settings → Plugins → enable **Godot RL Agents**
4. Wait for MSBuild to finish

---

## 4 · First training run

**A — Headless hub binary (console-first)**

```bash
python examples/stable_baselines3_example.py \
  --env_path=examples/godot_rl_BallChase/bin/BallChase.x86_64 \
  --experiment_name=BallChase_smoke \
  --timesteps=50000 \
  --speedup=8
```

Omit `--viz` for headless training. Watch the terminal for rising episode reward.

**B — In-editor (macOS / debugging)**

Terminal 1 — start Python listener:

```bash
gdrl --experiment_name=BallChase_Mac --viz \
  --save_model_path=ballchase_brain \
  --onnx_export_path=ballchase_brain.onnx
```

Wait for *"Waiting for connection from Godot…"*

Terminal 2 optional: `tensorboard --logdir=logs`

Godot — open the training scene, press **F6** (Play Scene). The agent should connect and learn.

!!! success "Success criteria"
    Agent visible in Godot (or hub window); episode reward trends upward; no socket errors; TensorBoard curve optional but recommended.

---

## 5 · Training modes for this course

| Phase | Units | Default |
|-------|-------|---------|
| Explore | 0–2 | Editor or `--viz` — see physics and reward bugs |
| Scale | 3–8 | Exported binary + `--headless` — faster rollouts |
| Ship | 9–10 | ONNX in Sync node — no Python at runtime |

**ONNX preview (optional)**

After training, copy `ballchase_brain.onnx` into the Godot project. On the Sync node: Control Mode → **ONNX Inference**, set ONNX Model Path. Play scene — agent runs without Python.

---

## First evening script (~2½–3 hours) { #first-evening-script }

One sitting: tooling works, agent learns, you change a reward. Times are guides — install steps vary by machine. Use **Option A (hub binary)** in Section 3 unless you already have Godot .NET open.

| Block | Time | Do this | Done when |
|-------|------|---------|-----------|
| **1 · Install** | 45–75 min | [Section 2](#2-conda-environment) — install Miniconda → create `godot_env` → `pip install`<br>[Section 3](#3-godot-project-plugin) — hub download BallChase | `import godot_rl` prints ok; binary path exists |
| **2 · First train** | 30–45 min | [Section 4B](#4-first-training-run) — `gdrl --viz` + Godot F6<br>Second terminal: `tensorboard --logdir=logs` | Agent moves; `ep_rew_mean` rises; no socket errors |
| **3 · Start Foundations 1** | 45–60 min | Open [Neural Foundations 1](unit-neural-01.md)<br>Predict the hand calculation (~15 min) → run the research plot or Godot enemy scene<br>While exploring: read Sections 2–3 | You can name each contribution, weighted sum, and activation output |

**Minimal command cheat sheet (Block 2)**

```bash
conda activate godot_env
tensorboard --logdir=logs &

gdrl --experiment_name=evening_ballchase --viz \
  --save_model_path=ballchase_brain \
  --onnx_export_path=ballchase_brain.onnx

# Godot: BallChase training scene → F6 (Play Scene)
```

Headless alternative (no Godot window): Section 4A with `--timesteps=50000 --speedup=8`.

**End-of-evening checklist**

- [ ] Godot — saw the agent act
- [ ] Python — training ran without connection errors
- [ ] TensorBoard — opened at least once (localhost:6006)
- [ ] Code — ran one neuron forward-pass test or visual example

Tomorrow: finish Foundations 1–2, then [RL Essentials](unit-01.md) (BallChase reward tweak) and Unit 2 Phase A (SimpleReachGoal).

!!! warning "Stuck?"
    Most first-evening blockers: wrong Godot build (need **.NET**), plugin not enabled, Python started before Godot F6, or firewall blocking localhost socket. Re-read Section 1 if the split architecture is unclear.

---

## Stretch Goals

**Run the other training mode.** If you used Option A (headless hub binary), try Option B (in-editor with `--viz`) — or vice versa. Same agent, same algorithm; the point is to feel the difference between *console-first* speed and *editor* visibility. Note in the terminal which mode reaches a given `ep_rew_mean` first.

**Inspect the exported ONNX.** Open `ballchase_brain.onnx` in [Netron](https://netron.app/) (drag the file into the browser tab — no install). Identify the input tensor shape, the output tensor shape, and the activation between layers. This is the file Godot loads at inference time in Phase 3 — knowing what's inside it now pays off in Unit 9.

**Try a second example from the hub.** Pull a different binary and run a short 50k-step smoke test:

```python
from godot_rl.env_from_hub import env_from_hub
env_from_hub("edbeeching/godot_rl_JumperHard")
```

Then point `--env_path` at the new binary. The goal isn't to train it well — it's to confirm your install handles more than one environment and to compare reward curves on TensorBoard.

## What's next

Tooling works. In **Neural Foundations 1** you'll build one visible neuron, then connect networks to the RL loop in later units.

[→ Neural Foundations 1](unit-neural-01.md)
