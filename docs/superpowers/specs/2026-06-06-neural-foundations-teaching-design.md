# Design: Visual Neural Foundations for Research and Game Development

**Date:** 2026-06-06
**Status:** Approved design; implementation plan not yet written
**Decision:** Add a three-unit neural-foundations track with one shared conceptual
spine and two audience-specific practical paths.

---

## Summary

The neural-foundations track teaches the same concepts through two distinct
professional contexts:

- **Research path:** Python, NumPy, PyTorch, Matplotlib, controlled experiments,
  and a hand-written REINFORCE agent.
- **Game-development path:** Godot, GDScript, visible game behavior, small
  hand-written networks, and a final PPO-trained arcade racer.

Learners complete one primary path and view short comparison demonstrations
from the other. Every unit contains a running visual example. The recurring
learning loop is:

> **Predict → Play → Build → Break → Explain**

The Game path uses `godot-native-rl` as its standard RL integration. Training
still runs in Python through the compatible `godot-rl` protocol. Deployment is:

> **PyTorch/SB3 → ONNX inspection and parity check → ncnn conversion → native
> Godot inference**

This adoption is limited to the new foundations track. Existing course units
retain their current `godot-rl-agents` stack until a separate migration design
approves replacement or an alternative native route.

---

## Problem with the previous plan

The original plan correctly identified the missing conceptual bridge between
the RL loop and deep-RL algorithms, but it had four weaknesses:

1. It used essentially the same scientific-style visualizations for both
   researchers and game developers.
2. Its Godot sequence resembled a neural-network laboratory rather than actual
   game-AI work.
3. It let the final racer grow mechanically from earlier racing exercises
   instead of testing whether learners could transfer concepts between
   different problems.
4. It assumed the existing .NET-based inference route even though a native,
   C#-free course stack is available.

The revised design keeps the shared theory while changing the examples,
evidence, and tooling to match each audience.

---

## Goals

1. Teach neurons, layers, forward passes, loss, gradient descent, and
   backpropagation without assuming prior ML knowledge.
2. Make every mathematical operation produce an immediate visual consequence.
3. Require only basic programming knowledge in the learner's chosen language.
4. Keep every project small enough to build from an empty scene or script in
   one session.
5. Give researchers practice with hypotheses, seeds, variance, generalization,
   and ablations.
6. Give game developers practice with visible behavior, instrumentation,
   failure diagnosis, and robustness.
7. Use supervised mini-problems to isolate network training, then use
   reward-only RL in the final projects.
8. Make the final Godot racer a genuine synthesis task.
9. Introduce a C#-free Godot training and deployment path through
   `godot-native-rl`.
10. Preserve ONNX as a visible, inspectable, framework-neutral intermediate.

---

## Non-goals

- Teaching convolutional networks, attention, transformers, tokenization, or
  language-model training.
- Implementing PPO from scratch.
- Training neural networks inside the ncnn runtime.
- Building production game architecture, polished art, menus, save systems, or
  realistic vehicle physics.
- Migrating all existing course units to `godot-native-rl` in this project.
- Teaching both practical paths in full to every learner.
- Optimizing for large models, GPU training, distributed training, or
  production-scale experiment infrastructure.

---

## Audience and prerequisites

### Shared prerequisites

- Basic programming knowledge: variables, functions, conditionals, loops, and
  arrays.
- Ability to run a script or Godot scene.
- No prior machine-learning, neural-network, calculus, or linear-algebra course
  required.

### Research path

- Basic Python syntax.
- No prior NumPy, PyTorch, or Matplotlib experience required; only the small
  subset used in the unit is introduced.

### Game-development path

- Basic GDScript or transferable programming knowledge.
- Familiarity with Godot's scene tree is helpful but not required.
- Standard Godot 4.5+; no Godot .NET editor or C# SDK required.

---

## Placement and course boundary

The track remains conceptually placed after **Unit 2 — Build Your First Env**
and before **Q-Learning / DQN**.

This ordering preserves the existing fast-start principle:

1. Learners first observe a working RL agent.
2. They understand observations, actions, rewards, and episodes.
3. They open the policy black box and build its neural components.
4. Q-Learning supplies the tabular baseline.
5. DQN becomes a comprehensible replacement of the table with a network.

During the transition period:

- Units 0–2 retain their existing stack.
- The foundations track has a self-contained Standard-Godot setup using
  `godot-native-rl`.
- A short bridge explains that both plugins communicate with the same
  `godot-rl` Python ecosystem but use different Godot-side inference runtimes.
- A separate future design decides whether existing units gain a Native
  alternative or migrate completely.

Experienced learners may take a diagnostic and skip the track if they can
already explain and calculate a forward pass, loss gradient, and basic
backpropagation update.

---

## Track structure

The track contains three units of **90–120 minutes each**.

Learners:

1. complete either the Research or Game path;
2. watch a comparison demonstration of at most ten minutes from the other path
   in each unit;
3. join a shared final comparison of approximately twenty minutes.

The comparison demonstrations show conceptual equivalence. They do not require
setting up the other toolchain.

---

## Teaching method

### Recurring loop

Every major concept follows the same five steps:

1. **Predict** — learners state what they expect before running the example.
2. **Play** — they run or manipulate the visual system.
3. **Build** — they implement one central computation from scratch.
4. **Break** — they deliberately change one variable to produce a failure.
5. **Explain** — they use visible evidence and a metric to explain the result.

### Mathematical sequence

The common sequence is **intuition first, complete equation second**:

1. Show the behavior.
2. Name the inputs and output.
3. Display the current numerical calculation.
4. Introduce the equation.
5. Implement the equation.
6. Generalize it to vector or matrix notation.

No equation is introduced without a synchronized visual that displays:

- its current inputs;
- the numerical intermediate result;
- the final output;
- the resulting behavioral or graphical effect.

### Controlled complexity

Hand-written networks must not exceed:

- `2 → 4 → 1` for scalar output examples;
- `3 → 4 → 2` for two-action examples.

The small sizes allow learners to inspect every weight and calculate selected
updates manually.

---

## Shared opening

The track begins with a tool-neutral visual demonstration of one neuron:

$$
z = w_1x_1 + w_2x_2 + b
$$

The demonstration shows:

- two normalized inputs;
- positive and negative weighted contributions;
- bias;
- weighted sum;
- activation;
- one visible decision.

Learners first predict the output from the displayed values. The opening then
maps the same computation to:

- a research classification decision;
- a game-agent behavior decision.

The core message is:

> A neuron is a small differentiable decision function, not a miniature brain.

---

## Unit 1 — One Neuron Makes One Decision

### Shared concepts

- input features;
- normalization;
- weight;
- bias;
- weighted sum;
- step, sigmoid, and tanh activations;
- decision threshold;
- linear decision boundary;
- inference as a forward calculation.

### Research path: visual linear decision

**Question:** Can one neuron separate safe and unsafe experimental conditions?

Learners generate a small two-feature dataset representing abstract
measurements such as temperature and pressure. The domain is deliberately
generic so the lesson remains applicable across research disciplines.

They build:

- a NumPy weighted-sum function;
- a selectable activation;
- a Matplotlib scatter plot;
- a live decision boundary;
- simple sliders or repeated parameter runs for weights and bias.

**Visible result**

- points are colored by predicted class;
- the boundary rotates when weights change;
- the boundary shifts when bias changes;
- misclassified points are highlighted.

**Controlled experiments**

1. Reverse one weight and predict which region changes class.
2. Multiply one input feature by 100 and observe why normalization matters.
3. Compare step, sigmoid, and tanh outputs near the boundary.

**Research evidence**

- classification accuracy;
- explicit parameter table;
- short hypothesis before each run.

### Game path: enemy decision

**Question:** Can one neuron choose whether an enemy should chase or retreat?

Learners build a minimal top-down room with:

- one player represented by a primitive shape;
- one enemy represented by a different primitive shape;
- normalized player distance;
- normalized enemy health;
- one neuron implemented directly in GDScript;
- two behaviors: chase and retreat.

**Visible result**

- a line displays player distance;
- a health bar displays the second input;
- labels display both weights, bias, weighted sum, and activation;
- the enemy's color and motion show the selected behavior;
- changing exported weights updates behavior immediately.

**Controlled experiments**

1. Reverse the health weight and explain the resulting bad behavior.
2. Add positive bias and identify when the enemy becomes over-aggressive.
3. Move the player across the decision threshold and observe behavior
   oscillation.

**Game-development evidence**

- expected versus actual behavior in three test positions;
- one diagnosed failure using the overlay;
- explanation of how the same decision could be authored with rules and why a
  differentiable neuron becomes useful later.

### Unit 1 completion check

Learners must:

- calculate one output by hand;
- predict the effect of changing one weight or the bias;
- implement the forward pass without copying a finished project;
- identify why unnormalized inputs can dominate;
- explain the equivalent roles of the research boundary and enemy behavior.

---

## Unit 2 — A Tiny Network Learns from Examples

### Shared concepts

- multiple neurons;
- hidden layers;
- vectors and tensor shapes;
- nonlinear decision regions;
- target values;
- loss;
- learning rate;
- gradient;
- chain rule;
- backpropagation;
- training versus inference;
- generalization and overfitting.

This unit uses fixed examples so backpropagation can be understood without
simultaneously introducing reward, exploration, and policy gradients.

### Research path: nonlinear classification laboratory

**Question:** Why can a hidden layer solve a pattern that one neuron cannot?

Learners build a `2 → 4 → 1` network on a small nonlinear dataset such as XOR,
two moons, or concentric regions. The canonical reference exercise uses a fixed
two-moons dataset because the decision boundary is visually intuitive.

They:

1. calculate one forward pass by hand;
2. calculate one selected output-layer gradient by hand;
3. implement the small network and training loop;
4. reproduce the model with PyTorch autograd;
5. compare manual and autograd results.

**Visible result**

- class points and decision regions;
- current loss;
- selected gradient arrows or signs;
- boundary snapshots before, during, and after training;
- training and validation curves.

**Controlled experiments**

1. Compare hidden sizes `1`, `2`, and `4`.
2. Compare a learning rate that is too small, useful, and too large.
3. Add measurement noise and compare train versus validation error.
4. Repeat with fixed seeds and report variation.

**Research evidence**

- loss and validation accuracy;
- seed table;
- one-variable-at-a-time experiment;
- explanation of whether poor performance is underfitting, optimization
  failure, or overfitting.

### Game path: arena collector

**Question:** Can a tiny network learn to collect gems while avoiding hazards?

Learners build a single-screen top-down arena containing:

- one moving agent;
- one gem;
- one hazard;
- normalized relative direction to the gem;
- normalized relative direction or proximity to the hazard;
- two continuous movement outputs;
- a small `3 → 4 → 2` GDScript MLP;
- a minimal GDScript backpropagation implementation.

To isolate supervised learning, a simple scripted teacher generates examples:

- move toward the gem when the hazard is not threatening;
- steer away from the hazard when it is close;
- output a normalized desired movement vector.

The teacher is intentionally simple and visible. It exists only to create
input-target pairs; it is not presented as the final game AI.

**Visible result**

- gem and hazard direction lines;
- input values and output movement bars;
- target movement versus predicted movement;
- incorrect predictions flash visibly;
- loss updates during training;
- before/after replay shows behavior improvement.

**Controlled experiments**

1. Remove examples near the hazard and inspect the resulting blind spot.
2. Reduce the hidden layer and observe underfitting.
3. Add sensor noise and compare robustness.
4. Raise the learning rate until behavior becomes unstable.

**Game-development evidence**

- gem collection and hazard collision counts in fixed test rooms;
- diagnosis of one intentional data-coverage bug;
- explanation of why a training loss can improve while gameplay remains poor.

### Unit 2 completion check

Learners must:

- calculate the shape and output of each layer;
- explain why a hidden layer can create nonlinear behavior;
- describe loss as a measurable mismatch;
- trace one gradient from output toward an earlier weight;
- distinguish training from inference;
- diagnose one failure caused by data, capacity, or learning rate;
- state what must transfer from the arena collector to the final racer.

---

## Unit 3 — Learn Behavior from Reward

### Shared concepts

- policy network;
- stochastic action selection;
- trajectory;
- return;
- policy-gradient intuition;
- exploration;
- reward design;
- observation and action contracts;
- training versus deterministic deployment;
- evaluation across multiple runs.

The final projects use **RL only**. They do not use imitation learning or
supervised action labels.

### Research path: 2D point robot

**Question:** Can a policy learn navigation without correct-action labels?

Learners build a minimal continuous 2D point-robot environment in Python with:

- position and heading;
- three range sensors;
- normalized target bearing;
- one target;
- a small obstacle layout;
- three discrete actions: forward-left, forward, and forward-right;
- fixed forward speed so the policy learns steering rather than vehicle
  dynamics;
- trajectory animation in Matplotlib.

They implement a compact REINFORCE agent in PyTorch:

1. policy network forward pass;
2. stochastic action sampling;
3. trajectory collection;
4. discounted returns;
5. log-probability policy loss;
6. optimizer update.

PPO is shown only as a short comparison after learners understand REINFORCE.

**Visible result**

- animated trajectories;
- sensor rays;
- action probabilities;
- reward per episode;
- episode length;
- before/after trajectory overlays.

**Controlled experiments**

1. Run at least five seeds and report mean and spread.
2. Remove one sensor and measure the effect.
3. Compare sparse and shaped reward.
4. Compare two small hidden-layer sizes.

**Research evidence**

- fixed evaluation protocol;
- mean and variability across seeds;
- trajectory plots;
- one ablation table;
- explicit distinction between a lucky run and a reproducible result.

### Game path: arcade racer

**Question:** Can an agent combine sensors, multiple outputs, and reward to
discover how to drive a course?

Learners build a small top-down racer using minimal arcade physics:

- turn;
- accelerate;
- brake or reverse;
- collide;
- reset.

The track contains one closed loop, clear walls, and ordered checkpoints.

Recommended observations:

| Observation | Range | Purpose |
|-------------|-------|---------|
| left ray distance | `[0, 1]` | left clearance |
| center ray distance | `[0, 1]` | obstacle or curve ahead |
| right ray distance | `[0, 1]` | right clearance |
| forward speed | `[-1, 1]` | motion state |
| heading error to next checkpoint | `[-1, 1]` | progress direction |

Actions:

| Action | Range | Meaning |
|--------|-------|---------|
| steering | `[-1, 1]` | left to right |
| throttle | `[-1, 1]` | brake/reverse to accelerate |

Reward components:

- positive ordered-checkpoint progress;
- small forward-progress reward;
- collision penalty;
- penalty for prolonged immobility or driving backwards;
- lap-completion bonus;
- episode termination after collision, timeout, or prolonged immobility.

**Godot integration**

The racer uses:

- Standard Godot 4.5+;
- `NcnnAIController2D`;
- `godot-native-rl` `RaycastSensor2D` nodes;
- `NcnnSync`;
- Python `godot-rl`;
- SB3 PPO for training.

**Visible result**

- sensor rays and normalized values;
- steering and throttle bars;
- immediate reward and episode return;
- active checkpoint;
- collision and reset reason;
- replay comparison of untrained, intermediate, and trained policies.

**Controlled experiments**

1. Remove the center ray and compare failures.
2. Remove checkpoint shaping and observe exploration difficulty.
3. Change collision penalty scale and inspect learned behavior.
4. Compare the trained policy with a deliberately simple manual controller.

**Game-development evidence**

- lap-completion rate over repeated deterministic evaluations;
- collision count;
- mean lap progress;
- behavior in at least two small perturbation tests, such as shifted starting
  pose or mild sensor noise;
- diagnosis of one high-reward but visibly undesirable behavior.

### Racer as transfer assessment

The racer must not provide a finished observation, action, and reward
implementation for learners to paste. The unit provides the small scene and
API contracts, but learners assemble the policy interface from concepts learned
earlier:

| Earlier concept | Racer transfer |
|-----------------|----------------|
| enemy health and distance inputs | normalized ray, speed, and heading inputs |
| chase/retreat scalar decision | continuous steering and throttle outputs |
| arena hidden layer | policy network capacity |
| visible activation and failures | live sensor/action/reward instrumentation |
| supervised loss | policy objective driven by return |

### Unit 3 completion check

Learners must:

- identify observations, actions, policy outputs, reward, and episode boundary;
- explain how RL changes weights without action labels;
- run a repeated evaluation rather than present one successful episode;
- diagnose at least one reward or sensor failure;
- distinguish stochastic training behavior from deterministic deployment;
- explain how the final project combines concepts from Units 1 and 2.

---

## `godot-native-rl` integration

### Role in the foundations track

`godot-native-rl` becomes the standard Godot-side RL stack for Unit 3.

It provides:

- a `godot-rl`-compatible training bridge;
- GDScript controllers;
- sensors;
- reward helpers;
- native ncnn inference through a GDExtension;
- deployment without C#/.NET.

It does **not** replace Python training. Training remains a two-process system:

1. Godot simulates the environment and sends observations and rewards.
2. Python trains PPO and returns actions.

### Release prerequisites

The course release must provide tested prebuilt GDExtensions for the supported
desktop targets declared by the project:

- macOS arm64;
- Windows x86_64;
- Linux x86_64.

Each release package must include:

- addon files;
- matching native library;
- `ncnn_runner.gdextension`;
- exact supported Godot version;
- checksum or release version;
- a smoke-test scene.

Platforms without a published binary are not advertised as supported in the
foundations release.

### Existing-course boundary

This design does not change Units 0–10. The future migration project must
separately decide between:

1. a Native alternative beside existing instructions;
2. a full replacement;
3. a phase-by-phase migration.

The foundations racer becomes the reference implementation and parity fixture
for that future decision.

---

## Required ONNX stage

ONNX is a mandatory educational and validation artifact.

The deployment flow is:

1. train the PPO policy in Python;
2. save the original checkpoint;
3. export the deterministic actor to ONNX;
4. inspect the graph in Netron;
5. identify input and output names and shapes;
6. run fixed observations through PyTorch and ONNX Runtime;
7. confirm numerical parity;
8. convert ONNX to ncnn;
9. run the same observations through ncnn;
10. confirm deployment parity;
11. load the ncnn `.param` and `.bin` files in Godot.

For the racer's continuous actions, parity uses numerical closeness rather than
argmax:

- fixed evaluation observations;
- matching preprocessing;
- matching output squashing;
- maximum absolute difference target of `1e-2`.

The unit must explicitly show that:

- ONNX is the portable model graph;
- ONNX Runtime is not the final Godot runtime in this path;
- ncnn is the native deployment representation;
- observation preprocessing is part of the policy contract even though it may
  live outside the network file.

---

## Visual design requirements

Every unit must include a running visual system, not only static diagrams.

### Shared visual language

- blue: inputs;
- green: positive contribution or desired behavior;
- red: negative contribution, error, or hazard;
- yellow: activation, target, or current decision;
- muted gray: inactive values or previous state.

### Required overlays

Each practical example exposes:

- current normalized inputs;
- current network outputs;
- current selected behavior or action;
- relevant loss or reward;
- pause and single-step control where technically reasonable;
- before/after comparison.

### Accessibility

- Never encode meaning by color alone.
- Label rays, bars, and decisions textually.
- Keep all plots readable in the MkDocs dark theme.
- Provide static screenshots or short recorded sequences for readers who cannot
  run the interactive example immediately.

---

## Materials shipped per unit

Each unit provides:

1. a from-empty-project build sequence;
2. checkpoint states learners can compare against;
3. a finished reference implementation;
4. the visual instrumentation;
5. three or four controlled experiments;
6. one intentional diagnosis challenge;
7. a five-question concept check with answers;
8. a “transfer to the racer” summary;
9. a short comparison demonstration from the other path.

The reference implementation is for verification after the build, not the
starting point for the core exercise.

---

## Scope guardrails

- One scene or one compact Python project per core build.
- Primitive shapes and clear colors instead of an art pipeline.
- One primary mechanic and one success condition per project.
- No menu, persistence, procedural generation, advanced physics, or level
  system.
- No derivation longer than the visual example it explains.
- No hand-written network larger than the declared small architectures.
- No hidden framework abstraction before learners have implemented the
  corresponding basic computation.
- Comparison demonstrations must stay under ten minutes.

---

## Assessment

Performance alone is insufficient because a learner can obtain one lucky RL
run without understanding the system.

Every unit checks five capabilities:

| Capability | Evidence |
|------------|----------|
| Predict | Anticipate one weight, bias, learning-rate, sensor, or reward change |
| Build | Implement the central computation without copying a finished project |
| Diagnose | Find one intentionally introduced failure using instrumentation |
| Measure | Compare two variants with an appropriate metric |
| Transfer | Explain how the concept reappears in the final racer |

### Path-specific standards

Research learners must:

- state a hypothesis;
- hold unrelated variables constant;
- use explicit seeds;
- report mean and spread where stochasticity matters;
- distinguish training and evaluation data.

Game-development learners must:

- predict behavior before pressing Play;
- expose internal values in the game;
- test deliberate edge cases;
- compare fixed test rooms or starting states;
- explain behavior qualitatively and quantitatively.

---

## Validation and release quality gates

### Content validation

- `mkdocs build --strict` passes.
- Navigation and breadcrumbs are consistent.
- English pages are complete before translation.
- German pages either match the finalized English structure or deliberately
  use documented fallback during drafting.

### Research code validation

- fixed datasets and seeds reproduce documented plots;
- manual forward-pass outputs match PyTorch within an explicit tolerance;
- selected manual gradients match autograd within `1e-5`;
- REINFORCE evaluation uses multiple seeds;
- scripts run from the documented environment without hidden local files.

### Godot code validation

- every project opens in the documented Standard Godot version;
- Units 1–2 run without Python or a native extension;
- Unit 3 passes a no-trainer human-control smoke test;
- the training handshake works with the pinned `godot-rl` version;
- prebuilt GDExtensions load on every advertised platform;
- the racer can train from a fresh checkout;
- deterministic native inference reproduces the verified policy behavior.

### Model conversion validation

- PyTorch and ONNX outputs match on fixed observations;
- ONNX and ncnn outputs match within `1e-2` for continuous actions;
- observation normalization and action squashing are identical in all stages;
- over twenty paired deterministic racer starts, ncnn lap-completion rate
  differs from the Python policy by no more than five percentage points and
  mean checkpoint progress differs by no more than five percent.

---

## Success criteria

The track succeeds when a learner with only basic programming knowledge can:

- calculate a neuron's output from visible values;
- explain weights, bias, activation, and normalization;
- describe how hidden layers create nonlinear behavior;
- trace the purpose of loss, gradients, and backpropagation;
- distinguish training from inference;
- explain why fixed examples simplify learning backpropagation;
- explain why RL does not provide correct-action labels;
- build and evaluate the selected path's three projects;
- diagnose a visible failure using internal values;
- explain the shared pipeline across both paths;
- assemble observations, actions, reward, and policy integration for the racer;
- inspect ONNX input/output contracts;
- verify PyTorch, ONNX, and ncnn policy parity;
- deploy the racer in Standard Godot without C#/.NET.

---

## Approved decisions

| Decision | Approved choice |
|----------|-----------------|
| Course placement | Integrated foundations track, not a separate full course |
| Audience structure | Shared concepts with Research and Game paths |
| Path participation | Complete one path; view short demos of the other |
| Prerequisites | Basic programming only |
| Duration | Three units of 90–120 minutes |
| Mathematics | Intuition first, then complete equations |
| Project style | Build tiny projects from scratch |
| Teaching rhythm | Predict → Play → Build → Break → Explain |
| Visual requirement | Running visual example in every unit |
| Research tools | Python, NumPy, PyTorch, Matplotlib |
| Research capstone | 2D point robot with hand-written REINFORCE |
| Game Unit 1 | Enemy chooses chase or retreat |
| Game Unit 2 | Arena collector learns gems versus hazards |
| Game capstone | Minimal 2D arcade racer |
| Final learning method | Reward-only RL |
| Godot RL stack | `godot-native-rl` for the foundations racer |
| Existing course migration | Deferred to a separate later design |
| ONNX | Mandatory inspection and parity stage |
| Native deployment | ONNX converted to ncnn; no C#/.NET runtime |
