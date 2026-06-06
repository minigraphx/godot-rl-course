# Neural-Network Foundations Track

## Decision

Neural-network fundamentals belong in the existing Godot RL course as a
**required-for-beginners, skippable-for-experienced-learners bridge** between
the first working RL environment and the deep-RL algorithm units.

They should not become a separate full course:

- DQN, policy gradients, PPO, SAC, and ONNX all rely on neural networks.
- Explaining those algorithms without first making a neuron visible creates an
  avoidable conceptual gap.
- Godot provides an unusually good environment for turning weights, activations,
  sensors, and outputs into visible behavior.
- A separate course would repeat setup, tooling, and examples before learners
  can return to the main RL path.

The future language-model course should treat this track as a shared
prerequisite. It can add a short refresher that maps the same concepts to
embeddings, token logits, attention, and next-token prediction, but it should
not duplicate the full neuron-to-network progression.

## Placement

Insert the track after **Unit 2 — Build Your First Env** and before
**Q-Learning / DQN**.

This preserves the course's fast-start principle:

1. Learners first see an agent train successfully.
2. They build one small environment and understand the RL loop.
3. They then open the "black box" and construct the function approximator.
4. Q-Learning shows the table baseline.
5. DQN can honestly become "replace the table with the network you already
   understand."

Experienced learners may take a short diagnostic and skip directly to
Q-Learning.

## Proposed unit sequence

### Neural Foundations 1 — One Neuron, One Visible Decision

**Core question:** How can one weighted sum produce useful behavior?

Use a Godot 2D scene with a movable target and one autonomous object. The
object has two normalized inputs:

- horizontal target offset
- vertical target offset

Start with one output neuron:

$$
z = w_1x_1 + w_2x_2 + b
$$

The sign of the activated output chooses between two visibly different actions,
for example **turn left** or **turn right**.

Learners should:

- change weights and bias with UI sliders;
- see the weighted inputs, sum, activation, and action update live;
- draw or inspect the resulting decision boundary;
- compare step, sigmoid, and tanh activations;
- explain what a weight, bias, activation, and output mean without equations;
- implement the forward pass directly in GDScript;
- reproduce the same calculation in a minimal Python script.

The practical message is not "a neuron is a brain." It is: **a neuron is a
small differentiable decision function**.

### Neural Foundations 2 — Two Neurons and a Small Network

**Core question:** What becomes possible when multiple neurons learn different
features?

Extend the first scene to two output neurons:

- neuron A controls left/right steering;
- neuron B controls accelerate/brake.

Show both neurons receiving the same inputs but using different weights. Then
introduce a small hidden layer only after learners can predict the two-neuron
outputs by hand.

Learners should:

- calculate a complete forward pass for two neurons;
- understand vectors, layers, and matrix multiplication as compact notation;
- see why one linear boundary cannot solve every classification problem;
- add a hidden layer that solves a simple non-linear task;
- define a target, loss, and learning rate;
- train the tiny network with gradient descent;
- inspect how changing a weight changes the loss;
- understand backpropagation as repeated chain-rule bookkeeping, not magic.

The main exercise should train a tiny network on generated 2D points and show
the decision regions in Godot while the loss curve updates.

### Neural Foundations 3 — A Learning 2D Track Driver

**Core question:** How do sensors become steering and throttle, and how does RL
improve the network weights?

Build a deliberately small top-down Godot environment:

- one `CharacterBody2D` or simple rigid vehicle;
- one closed track with walls;
- three forward ray sensors: left, center, right;
- normalized speed;
- optional heading error to the next checkpoint;
- continuous steering output in `[-1, 1]`;
- continuous throttle output in `[-1, 1]`;
- checkpoints for progress and lap measurement.

Recommended observation vector:

| Input | Range | Purpose |
|-------|-------|---------|
| left ray distance | `[0, 1]` | left-side clearance |
| center ray distance | `[0, 1]` | obstacle / curve ahead |
| right ray distance | `[0, 1]` | right-side clearance |
| forward speed | `[-1, 1]` | motion state |
| heading error | `[-1, 1]` | optional progress guidance |

Recommended action vector:

| Output | Range | Meaning |
|--------|-------|---------|
| steering | `[-1, 1]` | left to right |
| throttle | `[-1, 1]` | brake/reverse to accelerate |

Recommended reward components:

- positive checkpoint progress;
- small positive reward for forward velocity along the track;
- collision penalty;
- penalty for driving backwards or remaining stationary;
- lap-completion bonus;
- episode end after collision, prolonged immobility, or timeout.

The unit should use three checkpoints in increasing capability:

1. **Hand-set single neuron:** steering reacts to left/right clearance but fails
   on more complex curves.
2. **Hand-set two outputs:** steering and throttle produce recognizable but
   brittle driving.
3. **PPO-trained small MLP:** the same sensor and action interface learns to
   complete the track.

This progression makes the role of RL explicit: PPO does not replace the
network; it changes the network's weights using experience and reward.

## Visual requirements

Each unit must expose the internal calculation in the Godot scene:

- sensor values next to their rays;
- edge labels showing current weights;
- neuron boxes showing weighted sum and activation;
- output bars for steering and throttle;
- color-coded positive and negative contributions;
- a pause / single-step mode;
- a toggle between manual weights and trained policy;
- a short replay comparing untrained, intermediate, and trained behavior.

TensorBoard remains the second view, but it is not sufficient for this track.
Learners must be able to connect a numerical activation to a visible action in
the scene.

## Scope boundary

This track should teach only the reusable neural-network foundations needed by
the rest of the course:

- weighted sums and biases;
- activation functions;
- layers and tensor shapes;
- forward pass;
- loss functions;
- gradient descent;
- chain rule and backpropagation;
- inference versus training;
- why normalization matters.

It should not add:

- CNN architecture details;
- transformers or attention;
- tokenization or embeddings;
- large-scale pretraining;
- distributed training;
- RLHF or DPO.

Those topics belong either in later specialist units or in the language-model
sequel.

## Relationship to the language-model sequel

The sequel should begin with a compact mapping table:

| Shared foundation | Godot RL course | Language-model sequel |
|-------------------|-----------------|-------------------------|
| input vector | sensors / observations | token embeddings |
| weighted layers | policy/value MLP | transformer blocks |
| output values | actions or Q-values | token logits |
| loss | value/policy objectives | next-token cross-entropy |
| gradient descent | improve control policy | improve token prediction |
| inference | ONNX policy in Godot | generate tokens |

This creates continuity without making the Godot course carry language-model
content or forcing the sequel to reteach basic neural computation.

## Authoring and implementation order

1. Build the reusable Godot neuron visualizer scene.
2. Write Neural Foundations 1 around the visualizer.
3. Extend the visualizer to two neurons and a hidden layer.
4. Write Neural Foundations 2 with a small trainable dataset.
5. Build the minimal track, vehicle, sensors, and checkpoints.
6. Add manual one-neuron and two-output controllers.
7. Add the `AIController` observation, action, reward, and reset contract.
8. Train and record a reproducible PPO baseline.
9. Write Neural Foundations 3 around the three capability checkpoints.
10. Wire all three units into navigation, breadcrumbs, index, and translations.

## Completion criteria

The track is complete when a beginner can:

- calculate the output of one neuron from visible sensor values;
- explain why two neurons can control two independent actions;
- describe what a hidden layer contributes;
- explain how loss, gradients, and learning rate alter weights;
- identify observations, actions, reward, and network outputs in the vehicle;
- train the provided vehicle to complete at least one lap;
- inspect the trained policy in Godot and its learning curve in TensorBoard;
- explain how DQN and PPO use neural networks differently.
