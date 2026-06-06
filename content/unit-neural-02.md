# Neural Foundations 2 — A Tiny Network Learns

[← Neural Foundations 1](unit-neural-01.md) · [Course home](index.md)

!!! info "Three ways to see the computation"
    Decision regions · loss curve · gradients you can compare against PyTorch

One neuron draws one straight boundary. A tiny network can bend that boundary by
combining several neurons in a hidden layer. In this unit you will calculate one
forward pass, verify one gradient, and train a small network until the loss
falls.

> **Question for both paths:** What changes when a network has a hidden layer
> and learns from examples?

The learning loop is still **Predict → Play → Build → Break → Explain**.
Complete one primary path, then spend ten minutes viewing the other path.

---

## 1 · Why one neuron is not enough

A single neuron can only separate data with one line. That is enough for some
problems, but not for patterns like XOR:

| Input 1 | Input 2 | Target |
|---:|---:|---:|
| 0 | 0 | low |
| 0 | 1 | high |
| 1 | 0 | high |
| 1 | 1 | low |

No single straight boundary can mark the two diagonal corners as the same class.
A hidden layer solves this by creating several intermediate features, then
combining them into the final output.

---

## 2 · Two layers, one forward pass

The Research path uses a `2 → 4 → 1` network:

```text
two inputs → four hidden neurons → one output
```

The Game path uses the same idea with a `4 → 4 → 2` network:

```text
gem direction + hazard offset → four hidden neurons → movement x/y
```

The tested Python implementation lives in
`examples/neural_foundations/research/tiny_mlp.py`. The tested Godot
implementation lives in `examples/neural_foundations/game/shared/tiny_mlp.gd`.

Run the verification:

```bash
conda activate godot_env
python -m unittest examples.neural_foundations.research.tests.test_tiny_mlp -v

godot --headless \
  --path examples/neural_foundations/game \
  --script res://test/test_tiny_mlp.gd
```

Both tests use fixed weights so you can compare every intermediate value.

---

## 3 · Loss measures the mismatch

Training needs a number that says how wrong the output is. This unit uses mean
squared error:

$$
\text{loss} = \frac{1}{n}\sum_i(\hat{y}_i - y_i)^2
$$

Small loss means predictions match the target examples. Large loss means the
network is still producing the wrong behavior.

Run the Research training demo:

```bash
python examples/neural_foundations/research/train_tiny_mlp.py \
  --save artifacts/neural-foundations/tiny-mlp.png
```

The script prints the initial loss, final loss, and accuracy, then saves a
decision-region plot and loss curve.

---

## 4 · One gradient by hand

A gradient tells one weight which direction to move to reduce loss. You do not
need to memorize a long derivation yet. The important chain is:

```text
weight → hidden value → output → loss
```

The test suite compares the manual NumPy gradient against PyTorch autograd
within `1e-5`. That gives you a trusted reference while still keeping the
learner-facing implementation small and inspectable.

!!! success "Checkpoint"
    If the manual gradient and PyTorch gradient match, your bookkeeping is
    correct for this example.

---

## 5 · Backpropagation as bookkeeping

Backpropagation is not magic. It is careful bookkeeping:

1. run the forward pass and store the intermediate values;
2. measure the loss;
3. move backward from output to hidden layer;
4. calculate how much each weight contributed to the error;
5. apply a small update using the learning rate.

The learning rate matters. In the fixed Research demo, `0.05` learns reliably.
A much larger value can push the network into a bad plateau where the loss stops
improving.

---

## 6 · Choose your path

| | Research path | Game path |
|---|---|---|
| Inputs | Two normalized features | Gem direction and hazard offset |
| Output | One class score | Two movement values |
| Evidence | Loss, accuracy, decision regions | Target arrow versus predicted arrow |
| Tool | NumPy, PyTorch check, Matplotlib | Standard Godot 4 + GDScript |

Choose one primary path:

- **Research:** train the nonlinear classifier and compare gradients to PyTorch.
- **Game development:** train the arena collector from visible teacher examples.

---

## 7 · Research path — nonlinear classification

Start with the script:

```bash
python examples/neural_foundations/research/train_tiny_mlp.py
```

Before running it, predict what the loss curve should do. Then run it and
answer:

1. Did final loss fall below initial loss?
2. Did accuracy reach at least 75%?
3. Which region of the plot changed most during training?

??? success "Answer key"
    With the default seed and learning rate, the loss falls sharply and the
    fixed examples reach 100% accuracy. If you raise the learning rate too far,
    the model may stop improving even though the code is correct.

---

## 8 · Game path — arena collector

The Game path uses the same hidden-layer idea for a small collector:

- inputs describe where the gem is and whether the hazard is threatening;
- a scripted teacher provides target movement examples;
- the network predicts movement;
- the overlay compares target and predicted arrows;
- the loss display shows whether the network is learning.

Run the current scene:

```bash
godot --path examples/neural_foundations/game \
  res://unit_02_collector/unit_02_collector.tscn
```

Build the scene from primitives first. Keep the teacher visible and simple:

!!! warning "Pseudocode"
    ```text
    if hazard is close: move away from hazard
    else: move toward gem
    ```

The teacher is not the final AI. It exists to generate examples so you can
watch supervised learning before reward learning begins.

---

## 9 · Break training deliberately

Try one change at a time:

1. set the learning rate too high and watch the loss stall or jump;
2. reduce the hidden layer to one neuron and inspect underfitting;
3. remove examples near the hazard and look for the blind spot;
4. add sensor noise and compare training loss with gameplay behavior.

For each break, record the visible symptom and the metric that confirms it.

---

## 10 · Training versus inference

During training, the network changes its weights after seeing examples. During
inference, the weights stay fixed and the network only runs the forward pass.

This distinction matters for the rest of the course:

- **training:** slower, uses loss or reward, updates weights;
- **inference:** fast, uses fixed weights, chooses actions.

The final racer will use this same split: Python trains the policy, then Godot
runs the trained policy for inference.

---

## 11 · Stretch goals

- Plot three learning rates on the same loss chart.
- Add a validation split and compare train versus validation error.
- Save the trained tiny network weights and reload them.
- Add a second hazard position to the Game path and test whether behavior
  still improves.

---

## What's next

You now have the pieces inside a policy network: inputs, hidden activations,
loss, gradients, and weight updates. Next you will connect that network to the
reinforcement-learning loop: observations, actions, rewards, episodes, and
exploration.

[← Neural Foundations 1](unit-neural-01.md) · [Course home](index.md) · [→ RL Essentials](unit-01.md)
