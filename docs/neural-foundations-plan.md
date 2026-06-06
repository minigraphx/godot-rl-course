# Neural-Network Foundations Track

## Decision

The course will add a three-unit neural-foundations track around a split version
of the current RL Foundations unit.

The full approved teaching design is:

[Visual Neural Foundations for Research and Game Development](superpowers/specs/2026-06-06-neural-foundations-teaching-design.md)

## Structure

Learners complete one primary path and view short comparison demonstrations
from the other:

| Unit | Research path | Game-development path |
|------|---------------|-----------------------|
| 1 · One neuron | NumPy neuron and visible decision boundary | GDScript enemy choosing chase or retreat |
| 2 · Tiny network | nonlinear classification and PyTorch verification | GDScript arena collector learning gems versus hazards |
| 3 · Reward learning | 2D point robot with hand-written REINFORCE | 2D arcade racer with PPO and `godot-native-rl` |

The intended learning order is:

1. current Unit 0 — first successful training;
2. Neural Foundations 1 — one neuron;
3. Neural Foundations 2 — tiny network and backpropagation;
4. RL Essentials — the operational RL loop extracted from current Unit 1;
5. Neural Foundations 3 — reward-only RL;
6. RL Foundations Deep Dive — MC/TD, exploration mechanisms, and algorithm
   taxonomy from current Unit 1;
7. Reward Engineering, current Unit 2, Q-Learning, and DQN.

This keeps the immediate first success, teaches neural computation early, and
delays deeper RL taxonomy until learners can connect it to a policy they
trained themselves.

Every unit follows:

> **Predict → Play → Build → Break → Explain**

Every equation has a synchronized visual showing its inputs, numerical result,
and visible effect.

## Godot stack

The foundations Game path uses Standard Godot 4.5+ and `godot-native-rl`.
Training remains in Python through the compatible `godot-rl` protocol.

The mandatory deployment path is:

> **PyTorch/SB3 → ONNX inspection and parity → ncnn conversion → native Godot
> inference**

Prebuilt GDExtensions will be included for supported platforms in the course
release. The existing course remains unchanged initially; adding or replacing
its current `godot-rl-agents` route is a separate future migration project.

## Scope

The track teaches:

- weighted sums, bias, and activations;
- normalization;
- layers and tensor shapes;
- forward passes;
- loss, gradients, and backpropagation;
- training versus inference;
- policy networks and reward-only RL;
- controlled evaluation and failure diagnosis;
- ONNX inspection and native ncnn deployment.

It does not teach CNNs, transformers, large-scale training, realistic vehicle
physics, or a from-scratch PPO implementation.

## Completion

The track is complete when learners can:

- calculate and implement a neuron;
- explain and train a tiny hidden-layer network;
- diagnose failures using visible internal values;
- evaluate stochastic learning across repeated runs;
- combine observations, actions, reward, and a policy in the racer;
- verify PyTorch, ONNX, and ncnn parity;
- deploy the trained policy without C#/.NET.
