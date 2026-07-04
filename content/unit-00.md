# Unit 0 — Setup & First Run

[Course home](index.md)

!!! info "Time"
    Reading: ~20 min · Training: ~15 min GPU / ~1 h CPU

Install the Godot editor (Standard build) and Python toolchain. Run your first training session with the bundled **foundations racer** and confirm the Godot ↔ Python socket works.

---

!!! success "First success (one session)"
    1. **Godot** — the racer moving on its track (Play Scene while Python trains)
    2. **Python** — terminal shows rollout tables; `ep_rew_mean` appears
    3. **TensorBoard** (optional) — `tensorboard --logdir=logs/foundations_racer` shows a curve
    4. **Neural Foundations** — you will build visible neurons before the RL loop

!!! info "Three ways to see your AI (every unit)"
    Godot behavior · TensorBoard curves · what you change in `AIController`

!!! tip "Short on time?"
    Follow the [first evening script](#first-evening-script) (~2½–3 h) to finish Unit 0 and start Neural Foundations 1 in one sitting.

---

## 1 · Split architecture

Two runtimes talk over a local socket — Godot sends observations and receives actions; Python runs the training loop.

<div class="diagram-scroll">

<svg class="course-diagram" viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="Split architecture: the Godot game process (scene plus NcnnSync node) exchanges observations, rewards, and actions with the Python training process over a local socket; after training the NcnnSync node runs the converted brain natively instead">
  <defs>
    <marker id="arS" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 1 L10 5 L0 9 z" fill="#8892b0"/>
    </marker>
  </defs>
  <rect x="20" y="30" width="250" height="200" rx="14" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="145" y="58" text-anchor="middle" fill="#e2e8f0" font-size="16" font-weight="700">Godot</text>
  <text x="145" y="78" text-anchor="middle" fill="#8892b0" font-size="13">game process</text>
  <rect x="40" y="95" width="210" height="50" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="145" y="125" text-anchor="middle" fill="#e2e8f0" font-size="14">scene · physics · rewards</text>
  <rect x="40" y="160" width="210" height="50" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="145" y="190" text-anchor="middle" fill="#e2e8f0" font-size="14">NcnnSync node (GDScript addon)</text>
  <rect x="450" y="30" width="250" height="200" rx="14" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="575" y="58" text-anchor="middle" fill="#e2e8f0" font-size="16" font-weight="700">Python</text>
  <text x="575" y="78" text-anchor="middle" fill="#8892b0" font-size="13">training process</text>
  <rect x="470" y="95" width="210" height="50" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="575" y="125" text-anchor="middle" fill="#e2e8f0" font-size="14">godot-rl wrapper</text>
  <rect x="470" y="160" width="210" height="50" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="575" y="190" text-anchor="middle" fill="#e2e8f0" font-size="14">SB3 — PPO / DQN training</text>
  <path d="M270 120 L450 120" fill="none" stroke="#8892b0" stroke-width="1.8" marker-end="url(#arS)"/>
  <text x="360" y="106" text-anchor="middle" fill="#4ecca3" font-size="13" font-weight="700">observations + reward</text>
  <path d="M450 190 L270 190" fill="none" stroke="#8892b0" stroke-width="1.8" marker-end="url(#arS)"/>
  <text x="360" y="208" text-anchor="middle" fill="#6c8ef7" font-size="13" font-weight="700">actions</text>
  <text x="360" y="252" text-anchor="middle" fill="#8892b0" font-size="13">local socket · port 11008</text>
  <text x="360" y="284" text-anchor="middle" fill="#8892b0" font-size="13" font-style="italic">after training: NcnnSync runs the converted brain natively — no Python at runtime (see §5)</text>
</svg>

</div>

| Component | Role | Runtime |
|-----------|------|---------|
| **Godot** | Physics, observations, rewards | Godot 4 Standard, GDScript |
| **Addon (GDScript)** | NcnnSync node, native ncnn bridge | godot-native-rl (bundled with course repo) |
| **Python** | PPO training (SB3) | Conda, Python 3.10 |

!!! tip "The Standard Godot build is enough"
    Everything on the Godot side is GDScript — no C#, no .NET SDK, no build step. Download the **Standard** build (4.5+) from [godotengine.org](https://godotengine.org); [Setup](setup.md) has the details.

---

## 2 · Conda environment

!!! info "First time here?"
    Full installation instructions — Miniconda, `godot_env`, and the Godot addon — are in [Setup](setup.md). Complete that page first, then return here.

Quick reminder for every new terminal:

```bash
conda activate godot_env
```

---

## 3 · Godot project & addon

Unit 0 and the Neural Foundations units train inside the course repo's own Godot project — the **godot-native-rl** addon is already bundled in it, so there is nothing to install or enable:

1. Godot → Import → browse to `godot-rl-course/examples/neural_foundations/game/project.godot`
2. Let the first import finish (no build step — the addon is pure GDScript)
3. Verify: Add Node → search `NcnnSync`. If it appears, the addon is loaded.

!!! info "Examples repo for later units"
    Units from [RL Essentials](unit-01.md) onward currently use environments from the separate **godot_rl_agents_examples** repo (the legacy stack — migration is tracked in the course repo's issues). Clone it as a sibling of the course repo when you reach those units:

    ```bash
    cd ..   # step out of godot-rl-course
    git clone https://github.com/edbeeching/godot_rl_agents_examples.git
    ```

---

## 4 · First training run

**In-editor**

Terminal 1 — start the Python trainer (from the course repo root):

```bash
conda activate godot_env
python scripts/train_foundations_racer.py --timesteps 2048
```

Wait for *"waiting for remote GODOT connection"* in the log output.

Terminal 2 optional: `tensorboard --logdir=logs/foundations_racer`

Godot — open `unit_03_racer/racer_train.tscn`, press **F6** (Play Scene). The racer should connect and start stepping; SB3 prints rollout tables in Terminal 1. The default 2048 timesteps is a **smoke test** (a few minutes) — it proves the socket, the trainer, and the scene work together. For visible learning, re-run with `--timesteps 50000`.

When the run finishes, the trainer saves a checkpoint and an ONNX export under `examples/neural_foundations/game/unit_03_racer/models/`.

!!! success "Success criteria"
    Racer visible in Godot; rollout tables print with `ep_rew_mean`; no socket errors; TensorBoard curve optional but recommended.

---

## 5 · Training modes for this course

| Phase | Units | Default |
|-------|-------|---------|
| Explore | 0–2 | Editor Play Scene — see physics and reward bugs |
| Scale | 3–8 | Exported binary + `--headless` — faster rollouts |
| Ship | 9–10 | Native ncnn inference in NcnnSync — no Python at runtime |

**Native inference preview (optional, macOS Apple Silicon only for now)**

After training, the exported ONNX can be converted to ncnn and run natively: the evaluation scene `unit_03_racer/racer_eval.tscn` uses `NcnnSync` in inference mode — the agent runs without Python. [Neural Foundations 3](unit-neural-03.md) walks through the full export → verify → deploy pipeline. On Windows/Linux, skip this preview — the native inference binaries ship for macOS Apple Silicon only right now.

---

## First evening script (~2½–3 hours) { #first-evening-script }

One sitting: tooling works, agent learns, you change a reward. Times are guides — install steps vary by machine.

| Block | Time | Do this | Done when |
|-------|------|---------|-----------|
| **1 · Install** | 45–75 min | [Section 2](#2-conda-environment) — install Miniconda → create `godot_env` → `pip install`<br>[Section 3](#3-godot-project-addon) — open the course game project in Godot | `import godot_rl` prints ok; game project opens and `NcnnSync` appears in Add Node |
| **2 · First train** | 30–45 min | [Section 4](#4-first-training-run) — Python trainer + Godot F6<br>Second terminal: `tensorboard --logdir=logs/foundations_racer` | Racer moves; rollout tables print; no socket errors |
| **3 · Start Foundations 1** | 45–60 min | Open [Neural Foundations 1](unit-neural-01.md)<br>Predict the hand calculation (~15 min) → run the research plot or Godot enemy scene<br>While exploring: read Sections 2–3 | You can name each contribution, weighted sum, and activation output |

**Minimal command cheat sheet (Block 2)**

```bash
conda activate godot_env
tensorboard --logdir=logs/foundations_racer &

python scripts/train_foundations_racer.py --timesteps 2048

# Godot: unit_03_racer/racer_train.tscn → F6 (Play Scene)
```

**End-of-evening checklist**

- [ ] Godot — saw the agent act
- [ ] Python — training ran without connection errors
- [ ] TensorBoard — opened at least once (localhost:6006)
- [ ] Code — ran one neuron forward-pass test or visual example

Tomorrow: finish Foundations 1–2, then [RL Essentials](unit-01.md) (BallChase reward tweak) and Unit 2 Phase A (SimpleReachGoal).

!!! warning "Stuck?"
    Most first-evening blockers: Godot older than **4.5**, Godot F6 pressed before the Python trainer was waiting, or firewall blocking the localhost socket. Re-read Section 1 if the split architecture is unclear.

---

## Stretch Goals

**Inspect the exported ONNX.** Open `examples/neural_foundations/game/unit_03_racer/models/foundations_racer.onnx` in [Netron](https://netron.app/) (drag the file into the browser tab — no install). Identify the input tensor shape (`obs`), the output tensor shape (`out0`), and the activation between layers. This graph is the portable form of the brain — the export → verify → deploy pipeline in [Neural Foundations 3](unit-neural-03.md) builds on it.

**Run the deterministic math tests.** The game project ships headless tests for the racer's observation and reward math:

```bash
godot --headless --path examples/neural_foundations/game --script res://test/test_racer_math.gd
```

The goal isn't the tests themselves — it's to confirm your `godot` command-line setup from [Setup](setup.md#godot-cli) works, which every later unit relies on.

## What's next

Tooling works. In **Neural Foundations 1** you'll build one visible neuron, then connect networks to the RL loop in later units.

[→ Neural Foundations 1](unit-neural-01.md)
