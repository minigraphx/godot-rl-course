# Godot Reinforcement Learning Course 

To help visualize how the local code, the engine, and the training loop communicate with each other, see the design diagram below.

### Visualizing the Hybrid Architecture
This architectural diagram outlines how the **Godot Game Client** runs independently of the **Python PyTorch Training Loop**, exchanging observations and actions over a fast local network socket.

```html
<svg width="100%" height="400" viewBox="0 0 680 400" xmlns="http://www.w3.org/2000/svg">
  <!-- Definitions for Markers -->
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 2 L 10 5 L 0 8 z" fill="var(--color-text-primary)" />
    </marker>
  </defs>

  <!-- Title -->
  <text class="th" x="340" y="30" text-anchor="middle" dominant-baseline="central" style="font-size: 16px; fill: var(--color-text-primary);">Godot-RL Hybrid Training Architecture</text>
  <text class="ts" x="340" y="50" text-anchor="middle" dominant-baseline="central" style="fill: var(--color-text-secondary);">Real-Time Local Loop (Training) vs. ONNX Web Export (Inference)</text>

  <!-- CONTAINER 1: GODOT RIG (C#/.NET) -->
  <g class="c-teal">
    <rect x="40" y="80" width="240" height="240" rx="16" fill="var(--color-background-secondary)" stroke="var(--color-border-secondary)" stroke-width="1.5" />
    <text class="th" x="160" y="105" text-anchor="middle" dominant-baseline="central" style="fill: var(--color-text-primary); font-size: 14px;">Godot Engine (Game/Env)</text>
    
    <!-- Sub-node: Godot Scene Nodes -->
    <rect x="60" y="130" width="200" height="50" rx="8" fill="var(--color-background-tertiary)" stroke="var(--color-border-tertiary)" stroke-width="1" />
    <text class="t" x="160" y="145" text-anchor="middle" dominant-baseline="central" style="fill: var(--color-text-primary);">Player AI Controller</text>
    <text class="ts" x="160" y="162" text-anchor="middle" dominant-baseline="central" style="fill: var(--color-text-secondary);">RayCast Sensors &amp; Node3D</text>

    <!-- Sub-node: Sync Node / Socket Client -->
    <rect x="60" y="240" width="200" height="60" rx="8" fill="var(--color-background-tertiary)" stroke="var(--color-border-tertiary)" stroke-width="1" />
    <text class="t" x="160" y="258" text-anchor="middle" dominant-baseline="central" style="fill: var(--color-text-primary);">GDRL Sync Node</text>
    <text class="ts" x="160" y="275" text-anchor="middle" dominant-baseline="central" style="fill: var(--color-text-secondary); font-family: monospace;">godot_rl_agents_plugin</text>
  </g>

  <!-- COMMUNICATIONS / SOCKET BRIDGE -->
  <!-- Forward Arrow (Observations / Rewards) -->
  <path d="M 280 150 L 390 150" fill="none" stroke="var(--color-text-info)" stroke-width="2" marker-end="url(#arrow)" />
  <rect x="290" y="125" width="90" height="20" rx="4" fill="var(--color-background-primary)" />
  <text class="ts" x="335" y="135" text-anchor="middle" dominant-baseline="central" style="fill: var(--color-text-info); font-size: 10px;">Observations</text>

  <!-- Backward Arrow (Actions) -->
  <path d="M 400 270 L 290 270" fill="none" stroke="var(--color-text-success)" stroke-width="2" marker-end="url(#arrow)" />
  <rect x="295" y="278" width="80" height="20" rx="4" fill="var(--color-background-primary)" />
  <text class="ts" x="335" y="288" text-anchor="middle" dominant-baseline="central" style="fill: var(--color-text-success); font-size: 10px;">RL Actions</text>

  <!-- CONTAINER 2: PYTHON TRAINING RUNTIME -->
  <g class="c-blue">
    <rect x="400" y="80" width="240" height="240" rx="16" fill="var(--color-background-secondary)" stroke="var(--color-border-secondary)" stroke-width="1.5" />
    <text class="th" x="520" y="105" text-anchor="middle" dominant-baseline="central" style="fill: var(--color-text-primary); font-size: 14px;">Python Runtime (Model)</text>
    
    <!-- Sub-node: Wrapper -->
    <rect x="420" y="130" width="200" height="50" rx="8" fill="var(--color-background-tertiary)" stroke="var(--color-border-tertiary)" stroke-width="1" />
    <text class="t" x="520" y="155" text-anchor="middle" dominant-baseline="central" style="fill: var(--color-text-primary);">GDRL Env Entrypoint</text>
    <text class="ts" x="520" y="168" text-anchor="middle" dominant-baseline="central" style="fill: var(--color-text-secondary); font-family: monospace;">gdrl.env_from_hub</text>

    <!-- Sub-node: Core RL Library -->
    <rect x="420" y="240" width="200" height="60" rx="8" fill="var(--color-background-tertiary)" stroke="var(--color-border-tertiary)" stroke-width="1" />
    <text class="t" x="520" y="258" text-anchor="middle" dominant-baseline="central" style="fill: var(--color-text-primary);">StableBaselines3 / PyTorch</text>
    <text class="ts" x="520" y="275" text-anchor="middle" dominant-baseline="central" style="fill: var(--color-text-secondary);">PPO / DQN / ONNX Export</text>
  </g>

  <!-- ONNX Export Link at the bottom -->
  <path d="M 520 320 C 520 370, 160 370, 160 320" fill="none" stroke="var(--color-text-warning)" stroke-width="2" stroke-dasharray="4,4" marker-end="url(#arrow)" />
  <rect x="230" y="345" width="220" height="24" rx="4" fill="var(--color-background-secondary)" stroke="var(--color-border-tertiary)" stroke-width="0.5" />
  <text class="ts" x="340" y="357" text-anchor="middle" dominant-baseline="central" style="fill: var(--color-text-warning);">ONNX Model Export (Deployed for Web/WASM)</text>
</svg>
```

---

### Part 1: Hybrid training strategy

Training in the browser is impractical. Use a **hybrid model**:

* **Training (local):** Godot env + Python over a socket; headless binaries from Unit 3 onward.
* **Inference (web):** ONNX → Godot Sync node → HTML5/WASM without Python.

---

### Course approach

Example-driven progression through official **godot-rl-agents** environments. Start in [index.html](index.html). Full detail: [docs/curriculum.md](docs/curriculum.md).

---

### Part 2: Curriculum (10 core units)

| Unit | Topic | Example | Training mode |
|------|-------|---------|---------------|
| 0 | Setup & first run | BallChase | Hub binary / in-editor — first success checklist |
| 1 | RL foundations + tweak | BallChase | Fast path: skim MDP → reward tweak → theory |
| 2 | SimpleReachGoal → Lunar Lander | SimpleReachGoal, Lander | Phase A warm-up, then build |
| 3 | DQN & discrete spaces | CrossTheRoad | Headless binary |
| 4 | PPO & platforms | JumperHard | Headless |
| 5 | Parallel training | BallChase (source) | `n_parallel` |
| 6 | Continuous 3D | FlyBy, HovercraftRacing | Headless |
| 7 | Multi-agent | Racer, MultiAgentSimple | Headless |
| 8 | Memory / POMDPs | FPS, RobotFPS | RecurrentPPO |
| 9 | Imitation learning | MultiLevelRobot | Expert demos |
| 10 | Ship your brain into the game | Any model | ONNX in Godot; optional WASM |

**Extensions:** CleanRL / Sample Factory · capstone from stretch pool · self-play (AirHockey).

**Three views (every unit):** Godot behavior · TensorBoard · `AIController` changes.

**Rhythm (every unit):** run → open source → read `AIController` → tweak → retrain → viz checkpoint (Units 3+).