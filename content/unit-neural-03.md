# Neural Foundations 3 — Learn from Reward

[← RL Essentials](unit-01.md) · [Course home](index.md)

!!! info "Three ways to see the computation"
    Trajectory in the point-robot world · discounted returns from one episode ·
    five-seed evaluation metrics

In Foundations 1 and 2, the network learned from labeled answers. In this unit,
the network learns from reward: it samples actions, receives consequences, and
updates the policy toward trajectories with higher return.

> **Question for both paths:** How does a policy improve when nobody gives it the
> correct action?

---

## 1 · From labeled examples to reward

Supervised learning can compare a prediction with a known target. Reinforcement
learning usually cannot. The point robot only sees:

- its current observation;
- the action it sampled;
- the reward after that action;
- whether the episode ended.

That is enough to learn, but the signal is noisier. A bad action can still appear
inside a successful episode, and a good action can appear inside a failed one.

---

## 2 · Policy, trajectory, and return

A **policy** maps observations to action probabilities. A **trajectory** is one
episode of observations, actions, and rewards. The **return** is the discounted
sum of future rewards:

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots
$$

The tested helper in `examples/neural_foundations/research/reinforce.py`
computes those returns directly. For rewards `[1.0, 2.0, 3.0]` and
`gamma = 0.5`, the returns are `[2.75, 3.5, 3.0]`.

---

## 3 · Why actions must be sampled during training

During inference, choosing the most likely action is often fine. During training,
the policy must sample actions so it can discover better trajectories. REINFORCE
keeps the log probability of each sampled action, waits for the episode return,
then nudges the policy toward actions that appeared in high-return episodes.

!!! warning "Pseudocode"
    ```text
    collect one episode with sampled actions
    compute discounted returns
    increase log probability for actions with high return
    decrease it for actions with low return
    ```

---

## 4 · Choose your path

The Research path uses a tiny Python point-robot environment and hand-written
REINFORCE loop. The Game path uses the arcade racer, PPO training, and native
Godot inference.

Complete the Research path first if you want to see every tensor in the policy
gradient update before moving to the larger game example.

---

## 5 · Research path — 2D point robot

The point robot lives in a square room. Its observation contains three normalized
wall-ray distances and the normalized bearing to the target. Its actions are:

- forward-left;
- forward;
- forward-right.

Run the deterministic tests:

```bash
conda activate godot_env
python -m unittest examples.neural_foundations.research.tests.test_point_robot -v
```

Before training, predict what the three ray values should do when the robot turns
toward a wall. Then inspect `PointRobotEnv.observation()` and verify the numbers
match the geometry.

---

## 6 · Build REINFORCE

The implementation in `examples/neural_foundations/research/reinforce.py` has
three small pieces:

- `discounted_returns()` turns episode rewards into training targets;
- `train_episode()` samples one trajectory and updates the policy;
- `evaluate()` runs fixed seeds with greedy actions and reports metrics.

Run the return test:

```bash
python -m unittest examples.neural_foundations.research.tests.test_returns -v
```

Then trace one episode by hand: record three rewards, compute their discounted
returns, and compare them with the helper.

---

## 7 · Evaluate five seeds

Generate the reference summary:

```bash
python -m examples.neural_foundations.research.reinforce \
  --save examples/neural_foundations/research/results/reinforce_five_seed.json
```

The saved JSON reports mean return, return standard deviation, success rate, and
mean episode length over seeds `0` through `4`. Treat the standard deviation as a
warning label: one lucky seed does not prove a stable policy.

---

## 8 · Ablate sensors and reward shaping

Break one part at a time and rerun the five-seed evaluation:

1. remove the target bearing from the observation;
2. reduce the collision penalty;
3. remove the progress reward and keep only success or collision;
4. shorten `max_steps`.

For each ablation, write down both the visible symptom and the metric that caught
it. You are looking for the connection between reward design, sensors, and the
behavior the policy discovers.

---

## 9 · Game path — build the arcade racer

The Game path uses `examples/neural_foundations/game/unit_03_racer/`. The scene
is intentionally primitive: a rectangular track, visible rays, checkpoint markers,
and a triangle car.

!!! note "Current native scope"
    This course copy imports the macOS arm64 native runner from the local
    `godot-native-rl` checkout. Windows and Linux binaries stay out of scope until
    the multi-platform release is available.

---

## 10 · Define observations and actions

The racer observation contains three normalized ray distances, heading error to
the next checkpoint, normalized speed, and checkpoint progress. The action space
is one continuous two-value head:

```text
drive = [steering, throttle]
```

Run the deterministic math checks:

```bash
godot --headless \
  --path examples/neural_foundations/game \
  --script res://test/test_racer_math.gd
```

---

## 11 · Design reward and episode boundaries

The reward combines:

- a small step penalty;
- ordered-checkpoint reward;
- collision penalty;
- immobility timeout.

The important habit is to test reward math before training. A wrong sign in a
reward term can look like an algorithm problem for hours.

---

## 12 · Train with PPO

The training scene uses `NcnnSync` in training mode and the same socket protocol
as `godot-rl`. Start the reference command:

```bash
conda activate godot_env
./scripts/train-foundations-racer.sh
```

The command saves a Stable-Baselines3 checkpoint and exports ONNX under
`examples/neural_foundations/game/unit_03_racer/models/`.

---

## 13 · Inspect the ONNX graph

The exported graph has input `obs` and output `out0`. Before converting it, check
that the shapes match the six observation values and two drive outputs.

---

## 14 · Verify PyTorch, ONNX, and ncnn

Run the parity verifier:

```bash
conda activate godot_env
python scripts/verify_racer_policy.py
```

The verifier uses twenty fixed observations. PyTorch and ONNX must match within
`1e-5`; ONNX and ncnn must match within `1e-2`.

---

## 15 · Run native inference in Godot

The evaluation scene uses `NcnnSync` in native inference mode and points the agent
at the converted ncnn files:

```bash
godot --path examples/neural_foundations/game \
  res://unit_03_racer/racer_eval.tscn
```

Watch the same rays and checkpoint markers you used during training. Native
inference should behave like the Python policy on the fixed starts.

---

## 16 · Diagnose reward and sensor failures

When the racer fails, inspect in this order:

1. ray normalization;
2. heading error sign;
3. steering and throttle clamp;
4. checkpoint ordering;
5. collision and timeout conditions.

The deterministic tests cover these pieces so you can separate environment bugs
from training variance.

---

## 17 · Compare the two paths

The Research path made the policy-gradient update visible. The Game path keeps the
same idea but adds Godot timing, native inference, and model export. In both cases,
reward is the teacher.

---

## 18 · Stretch goals

- Add a second track and compare success rate.
- Plot checkpoint reach time over training.
- Change the ray angles and rerun parity checks.
- Add a reward term for smooth steering and test whether behavior changes.

---

## What's next

You can now connect reward learning to the algorithm vocabulary in the deep dive:
returns, bootstrapping, exploration, on-policy training, and actor-critic methods.

[← RL Essentials](unit-01.md) · [Course home](index.md) · [→ RL Foundations Deep Dive](unit-rl-foundations-deep.md)
