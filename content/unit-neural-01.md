# Neural Foundations 1 — One Neuron, One Decision

[← Unit 0](unit-00.md) · [Course home](index.md)

!!! info "Three ways to see the computation"
    Running visual · current numbers · code you wrote

A trained policy can look mysterious, but every decision begins with ordinary
arithmetic. In this unit you will build one neuron, watch every term change, and
use the same calculation for a research classification and a game enemy.

> **Question for both paths:** How can two measurements become one visible
> decision?

The learning loop is **Predict → Play → Build → Break → Explain**. Complete one
primary path, then spend ten minutes viewing the other path.

---

## 1 · Predict before running

Start with this frozen neuron:

| Input | Value | Weight | Contribution |
|---|---:|---:|---:|
| \(x_1\) | 0.50 | +0.80 | +0.40 |
| \(x_2\) | -0.25 | -0.40 | +0.10 |
| Bias | — | — | +0.10 |

Before using Python or Godot, write down:

1. the weighted sum;
2. whether `tanh` returns a negative or positive value;
3. which input contributes most to the decision.

The complete calculation is:

$$
z = w_1x_1 + w_2x_2 + b
$$

$$
z = (0.8)(0.5) + (-0.4)(-0.25) + 0.1 = 0.6
$$

The activation then produces \(\tanh(0.6) \approx 0.537\).

??? success "Answer key"
    The weighted sum is `0.6`, so the activated output is positive. The first
    contribution is largest: `+0.40`, compared with `+0.10` from the second
    input and `+0.10` from the bias.

**Visible check:** the automated examples use these same numbers:

```bash
conda activate godot_env
python -m unittest \
  examples.neural_foundations.research.tests.test_neuron -v

godot --headless \
  --path examples/neural_foundations/game \
  --script res://test/test_tiny_neuron.gd
```

Both tests call the forward pass you will inspect next.

---

## 2 · Weighted inputs and bias

A neuron gives each normalized input a **weight**:

- a positive weight makes larger input values push the output upward;
- a negative weight makes them push downward;
- a larger magnitude gives that input more influence;
- the bias shifts the decision before any input contribution.

The shared implementation is deliberately small:

```python
def neuron_output(inputs, weights, bias, activation):
    weighted_sum = sum(
        value * weight for value, weight in zip(inputs, weights)
    )
    return activate(weighted_sum + bias, activation)
```

The full typed version is in
`examples/neural_foundations/research/neuron.py`. The Godot version in
`examples/neural_foundations/game/shared/tiny_neuron.gd` performs the same
loop.

### Why normalize?

Suppose temperature is recorded as `0.7` after normalization, while pressure is
accidentally left as `80`. Even a small pressure weight can dominate the
calculation:

| Feature | Input | Weight | Contribution |
|---|---:|---:|---:|
| Temperature | 0.70 | +1.20 | +0.84 |
| Raw pressure | 80.00 | -0.05 | -4.00 |

The output would mostly describe the units used to measure pressure, not the
relationship you wanted to model. Both visual examples keep inputs between
`0` and `1`, so their contributions are comparable.

**Visible check:** in the research plot, each axis spans `0–1`. In the Godot
scene, the health bar and distance overlay show the normalized values before
they enter the neuron.

---

## 3 · Activation functions

The weighted sum \(z\) can be any number. An **activation function** turns it
into the form needed by the decision.

| Activation | Output | Useful visible interpretation |
|---|---|---|
| Step | `0` or `1` | Hard class switch |
| Sigmoid | between `0` and `1` | Confidence-like score |
| Tanh | between `-1` and `1` | Direction or signed tendency |

Near the boundary, the functions tell different stories:

| \(z\) | Step | Sigmoid | Tanh |
|---:|---:|---:|---:|
| -0.10 | 0 | 0.475 | -0.100 |
| 0.00 | 1 | 0.500 | 0.000 |
| +0.10 | 1 | 0.525 | +0.100 |

The decision boundary is where \(z = 0\):

$$
w_1x_1 + w_2x_2 + b = 0
$$

Changing a weight rotates that line. Changing the bias shifts it without
rotating it.

**Visible check:** choose `step`, `sigmoid`, and `tanh` in the research plot.
The black boundary stays at \(z=0\), while the displayed output for the star
probe changes. In Godot, the sign of `tanh(z)` selects `CHASE` or `RETREAT`.

---

## 4 · Choose your path

The equation is shared; the evidence differs.

| | Research path | Game path |
|---|---|---|
| Inputs | Normalized temperature and pressure | Normalized health and player distance |
| Output | Safe or unsafe class | Chase or retreat tendency |
| Main visual | Colored points and decision boundary | Enemy color, movement, line, and health bar |
| Evidence | Accuracy and misclassified points | Expected versus actual behavior |
| Tool | Python + Matplotlib | Standard Godot 4 + GDScript |

Choose one primary path:

- **Research:** complete Section 5 and view the Godot comparison once.
- **Game development:** complete Section 6 and view the plot comparison once.

No native extension, C#, training framework, or prior machine-learning library
is needed in this unit.

---

## 5 · Research path — visible decision boundary

**Research question:** Can one neuron separate safe and unsafe experimental
conditions?

Run the interactive plot from the repository root:

```bash
conda activate godot_env
python examples/neural_foundations/research/plot_neuron.py
```

The plot gives you synchronized evidence:

- the background and point colors show predictions;
- the black line shows \(z=0\);
- red rings show incorrect predictions;
- the star marks the current numerical probe;
- the side panel shows both contributions, bias, weighted sum, activation, and
  accuracy;
- sliders expose `w₁`, `w₂`, and bias.

At the initial probe \([0.65, 0.35]\):

$$
z = (0.65)(1.2) + (0.35)(-0.9) - 0.1 = 0.365
$$

With a step activation, the prediction is class `1` (unsafe).

### Experiment 1 — reverse one weight

**Hypothesis first:** predict which colored region will change if `w₁` moves
from `+1.2` to `-1.2`. Then move only that slider and record accuracy before
and after.

| Parameter | Before | After |
|---|---:|---:|
| `w₁` | +1.2 | -1.2 |
| `w₂` | -0.9 | -0.9 |
| Bias | -0.1 | -0.1 |

??? success "Answer key"
    Increasing temperature originally pushed the score toward unsafe. After
    the sign reversal, it pushes toward safe. The boundary changes orientation,
    many high-temperature points switch class, and accuracy falls for this
    dataset.

### Experiment 2 — remove normalization

**Hypothesis first:** predict what happens if pressure values become 100 times
larger while weights stay fixed. In `plot_neuron.py`, temporarily change the
prediction input:

```python
scaled_features = FEATURES.copy()
scaled_features[:, 1] *= 100.0
```

Pass `scaled_features` to `predict`, run once, then restore the normalized
features.

??? success "Answer key"
    The pressure contribution becomes about 100 times larger and overwhelms
    temperature and bias. Most decisions follow pressure alone. This is not
    evidence that pressure is scientifically more important; it is a scale bug.

### Experiment 3 — compare activations at the boundary

Set the probe close to \(z=0\), then switch among `step`, `sigmoid`, and `tanh`
without changing any parameter. Record the displayed output.

??? success "Answer key"
    Step jumps directly between classes. Sigmoid changes smoothly around
    `0.5`; tanh changes smoothly around `0`. The decision threshold can be the
    same even though the numerical outputs differ.

### Research evidence

Save this small table in your notes:

| Run | Hypothesis | Changed parameter | Accuracy | Boundary evidence |
|---|---|---|---:|---|
| Baseline | — | — | | |
| Weight sign | | `w₁` only | | |
| Scale bug | | pressure only | | |
| Activation | | activation only | | |

One-variable-at-a-time changes make your explanation testable.

---

## 6 · Game path — chase or retreat

**Game-AI question:** Can one neuron choose whether an enemy should chase or
retreat?

Open the self-contained Standard Godot project:

```bash
godot --editor --path examples/neural_foundations/game
```

Open `unit_01_enemy/unit_01_enemy.tscn` and press **F6**. Use the arrow keys to
move the player and the slider to change enemy health.

The scene exposes the complete forward pass:

- the dashed line and distance label show the first spatial relationship;
- the health bar shows the second input;
- five calculation labels show contributions, bias, \(z\), and `tanh(z)`;
- orange movement means `CHASE`;
- blue movement means `RETREAT`;
- **Pause** freezes the enemy while values remain visible;
- **Reset** restores a repeatable starting position.

At reset, the approximate calculation is:

$$
z = (0.75)(1.4) + (0.68)(-1.2) - 0.1 \approx 0.134
$$

The positive result makes the enemy chase. The overlay's displayed weighted
sum uses the same input values, weights, and bias as
`TinyNeuron.forward()` before activation.

Select the scene root to change the exported `health_weight`,
`distance_weight`, and `bias` in the Inspector.

### Experiment 1 — reverse the health weight

**Prediction first:** with the player held at the same distance, what should a
healthy enemy do after `health_weight` changes from `+1.4` to `-1.4`?

Test low, medium, and high health. Record expected and actual behavior.

??? success "Answer key"
    More health now lowers the weighted sum. The healthiest enemy becomes more
    likely to retreat, while low health removes less from the sum. The overlay
    exposes the bad sign immediately: the health contribution is negative.

### Experiment 2 — add positive bias

Restore `health_weight` to `+1.4`. Increase bias from `-0.1` to `+0.8`, then
test the same three player positions.

??? success "Answer key"
    Positive bias shifts every situation toward chase. The enemy can remain
    aggressive at low health or long range because `+0.8` must be overcome
    before the output becomes negative. The line does not cause this behavior;
    the bias label reveals it.

### Experiment 3 — cross the threshold

Restore the defaults. Move the player slowly around the distance where the
behavior changes. Do not change health.

??? success "Answer key"
    Near \(z=0\), tiny distance changes flip the selected behavior. The enemy
    may oscillate because a single hard threshold has no memory or hysteresis.
    The smoothly changing `tanh(z)` number shows that the underlying score is
    stable even when the chosen action switches.

### Game-development evidence

Pause the scene and fill in:

| Situation | Expected | Actual | Health term | Distance term | Diagnosis |
|---|---|---|---:|---:|---|
| High health, close | | | | | |
| Low health, close | | | | | |
| High health, far | | | | | |

A rule such as `if health > 0.5 and distance < 200` can author this behavior
directly. The neuron becomes useful later because its differentiable weights
can be adjusted from examples or reward instead of hand-tuning every rule.

---

## 7 · Break it deliberately

Choose one failure from your primary path and make it obvious:

1. write a one-sentence prediction;
2. change one parameter only;
3. capture the visible result;
4. identify the dominating contribution;
5. restore the baseline and confirm recovery.

Use this diagnosis order:

| Visible symptom | First number to inspect | Likely cause |
|---|---|---|
| Almost every case has one class | Bias contribution | Bias magnitude too large |
| One feature controls everything | Weighted contributions | Missing normalization or oversized weight |
| Decision is backwards | Contribution sign | Reversed weight |
| Rapid switching near boundary | Weighted sum near zero | No margin, memory, or hysteresis |

??? question "Completion check"
    Can you calculate one output by hand, predict a weight or bias change,
    implement the forward loop, identify an unnormalized input, and explain the
    visible failure without saying only “the AI is bad”?

??? success "Answer key"
    A complete explanation names the input, weight, contribution, weighted
    sum, activation output, and visible consequence. Example: “Raw pressure
    made the second contribution `-40`, which dominated the `+0.8` temperature
    contribution, so nearly every point became class `0`.”

---

## 8 · Compare the two paths

The research boundary and enemy behavior are two views of the same forward
calculation.

| Shared role | Research visual | Game visual |
|---|---|---|
| Input \(x_1\) | Temperature position | Health bar |
| Input \(x_2\) | Pressure position | Player distance line |
| Weighted sum \(z\) | Side-panel calculation | Overlay calculation |
| Threshold | Point color | Chase/retreat switch |
| Parameter effect | Boundary rotates or shifts | Behavior changes in space |
| Error evidence | Red misclassification ring | Expected/actual mismatch |

For a researcher, the boundary summarizes many observations at once. For a
game developer, motion shows one state changing over time. Neither view changes
the neuron:

```text
normalized inputs → weighted contributions → bias → activation → decision
```

Explain the equivalence aloud: rotating a classification boundary changes
which points fall on each side; changing enemy weights changes which game
states fall on the chase or retreat side.

---

## 9 · Stretch goals

**Research — export evidence.** Run:

```bash
MPLBACKEND=Agg python \
  examples/neural_foundations/research/plot_neuron.py \
  --save neuron-boundary.png
```

Add your hypothesis and parameter table beside the saved image.

**Game development — add a decision margin.** Keep chasing until the output is
below `-0.1`, and keep retreating until it is above `+0.1`. Compare oscillation
with the original zero threshold.

**Both paths — add a third normalized input.** Choose a meaningful feature,
predict its sign, update the forward-pass test first, then update the visual.
Keep the current contribution visible.

**Both paths — test invalid shapes.** Add a test showing that inputs and
weights must have equal length. Explain why silently dropping a feature would
make the visual evidence misleading.

---

## What's next

One neuron can only draw a straight boundary through its inputs. In **Neural
Foundations 2**, you will connect a few neurons, make a nonlinear decision
region, measure error, and update weights from examples.

[← Unit 0](unit-00.md) · [Course home](index.md) · Next planned: Neural Foundations 2
