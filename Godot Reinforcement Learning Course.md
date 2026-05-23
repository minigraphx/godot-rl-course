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

### Course Transition Blueprint: Classic Gym to Godot RL

Converting legacy Python-only libraries into interactive, engine-supported modules represents a leap forward in teaching quality. This document maps out your migration path, practical limitations, and a 6-week curriculum.

---

### Part 1: Conversion Feasibility & Strategy

Running deep learning algorithms synchronously alongside full visual rendering is computationally expensive. Attempting to train RL models directly inside a standard web browser (such as compiling Python via Pyodide or JupyterLite inside static pages) is highly impractical due to extreme slow-downs, sandboxing CPU thresholds, and lacking proper PyTorch GPU acceleration in browser engines.

Instead, a perfect **"Hybrid Engine Model"** should be utilized:
* **The Training Phase (Local-first):** Students clone environments containing Godot source templates locally. They use Python scripts (utilizing `stable-baselines3` or `cleanrl`) on their own PCs to execute rapid vectorized training loops connected to local Godot executable outputs.
* **The Showcase / Interactive Phase (Web-first):** Once trained, the optimized parameters are exported to an **ONNX model** file. The student loads this model directly into the Godot project and exports their clean, responsive game with WASM compatibility to run real-time inference seamlessly inside any web browser.

---

### Part 2: Curriculum Outline (6-Week Core)

This syllabus takes students from basic Reinforcement Learning principles up to compiling high-quality autonomous game actors.

#### Week 1: Introduction to MDPs & The Godot-Python Link
* **Concepts:** Markov Decision Processes, actions, states, environmental bounds, and basic scalar rewards.
* **Local Setup:** Installing Anaconda/Miniconda, creating virtual environments, installing `godot-rl`, and initiating standard environments from the hub.
* **Hands-on Lab:** Run training using pre-existing assets (`JumperHard`) using CMD tools while monitoring local runtime rendering flags on screen.

#### Week 2: Deep Q-Networks (DQN) & Discretized Spaces
* **Concepts:** Q-learning, exploration vs. exploitation ($\epsilon$-greedy), neural network approximation, and experience replay buffers.
* **Godot Integration:** Constructing discrete game boundaries inside 2D coordinates (e.g., Grid navigation or "Cross the Road" variants).
* **Hands-on Lab:** Implement custom reward wrappers to train a 2D vehicle navigating through traffic obstacles.

#### Week 3: Policy Gradient Methods & Continuous Actions
* **Concepts:** Policy Gradients, Actor-Critic structures, continuous spaces, and standard deviations of predictive actions.
* **Godot Integration:** Using the AI dynamic sensors (RayCast2D / RayCast3D) to sample continuous ranges of distances from nearby geometric collision panels.
* **Hands-on Lab:** Standardizing reward systems to teach a 3D Hovercraft how to guide itself through a windy, high-speed speedway.

#### Week 4: PPO Explained & Godot Multi-Agent Sync
* **Concepts:** Clip ratios, trust regions, advantages, and training multiple agents inside a single vectorized thread simultaneously to save system resources.
* **Godot Integration:** Instantiating parallel arena nodes inside Godot to accelerate step generation and gather training samples quickly.
* **Hands-on Lab:** Parallelize 16 spatial instances of a 3D Ball-chasing agent using Stable-Baselines3, cutting full validation down to minutes.

#### Week 5: Memory Models (LSTMs) & Navigation Tasks
* **Concepts:** Partially Observable Markov Decision Processes (POMDPs), embedding sequence memory into networks, and recurrent layers.
* **Godot Integration:** Designing mazes where targets are temporarily obstructed or location-sensitive flags rely on timing sequences.
* **Hands-on Lab:** Training an agent using a recurrent visual policy to navigate dynamic obstacle paths requiring temporal memory.

#### Week 6: Web Assembly (WASM) Deployment
* **Concepts:** ONNX model serialization, compiling assets for client-side environments, and handling light inference runtimes in WebGL frameworks.
* **Godot Integration:** Loading the trained network weights directly within Godot's Mono version, binding the file runtime, and setting up HTML5 exports.
* **Hands-on Lab:** Building a beautiful, browser-ready portfolio showing the virtual agent running on self-guided pathfinding loops.