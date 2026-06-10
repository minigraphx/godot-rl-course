# Neural Foundations 1 — One Neuron, One Decision

[← Unit 0](unit-00.md) · [Course home](index.md)

!!! info "Three ways to see the computation"
    Running visual · current numbers · code you wrote

A trained policy can look mysterious, but every decision begins with ordinary
arithmetic. In this unit you will build one neuron, watch every term change, and
use the same calculation for a research classification and a jump trigger.

> **Question for both paths:** How can two measurements become one visible
> decision?

The learning loop is **Predict → Play → Build → Break → Explain**. Complete one
primary path, then spend ten minutes viewing the other path.

---

## 1 · Predict before running

Start with this **fixed-number neuron**. The names describe what each number
means before we introduce mathematical shorthand:

| Named input | Value | Weight | Contribution |
|---|---:|---:|---:|
| Speed | 0.50 | +0.80 | +0.40 |
| Closeness to edge | 0.25 | +1.20 | +0.30 |
| Bias | — | — | -0.50 |

Before using Python or Godot, write down:

1. the weighted sum;
2. whether `sigmoid` returns a value above `0.5`;
3. which input contributes most to the decision.

First add the named contributions and bias:

$$
\text{sum} =
(\text{speed}\times\text{speed weight}) +
(\text{closeness}\times\text{closeness weight}) +
\text{bias}
$$

$$
\text{sum} = (0.5)(0.8) + (0.25)(1.2) - 0.5 = 0.2
$$

The activation produces \(\operatorname{sigmoid}(0.2) \approx 0.550\). Because
`0.550 > 0.5`, the neuron **fires** and the game action is `JUMP`.

Mathematics often shortens input to \(x\), weight to \(w\), bias to \(b\), and
the sum to \(z\). Therefore the same calculation may later appear as
\(z=w_1x_1+w_2x_2+b\). These are abbreviations, not different values. Weight
always uses a lowercase \(w\) in this course.

??? success "Answer key"
    The sum is `0.2`, so the sigmoid output is about `0.550` and the neuron
    fires. Speed contributes `+0.40`, closeness contributes `+0.30`, and the
    bias subtracts `0.50`.

**Visible check:** the automated examples use these same numbers:

!!! note "Run from the course repo root"
    These commands assume your terminal is in the [course repo root](setup.md#course-repo) and that `godot` is on your PATH — see [Godot on the command line](setup.md#godot-cli).

```bash
conda activate godot_env
python -m examples.neural_foundations.research.tests.test_neuron

godot --headless \
  --path examples/neural_foundations/game \
  --script res://test/test_tiny_neuron.gd
```

Both commands print the Section 1 walkthrough (`sum = 0.2`,
`sigmoid(sum) ≈ 0.550`) and then end with `OK`. The Godot run also shows the
jumper demo's live labels (speed, closeness, sum, output, and `WAIT`/`JUMP`).

Both tests call the forward pass you will inspect next.

---

## 2 · Weighted inputs and bias

A neuron gives each normalized input a **weight**:

- a positive weight makes larger input values push the output upward;
- a negative weight makes them push downward;
- a larger magnitude gives that input more influence;
- the bias shifts the decision before any input contribution.

Build the forward pass before changing any visualization. In your primary path,
open the matching file and type the loop yourself:

- Research: `examples/neural_foundations/research/neuron.py`
- Game development: `examples/neural_foundations/game/shared/tiny_neuron.gd`

The shared implementation is deliberately small:

```python
def neuron_output(inputs, weights, bias, activation):
    weighted_sum = sum(
        value * weight for value, weight in zip(inputs, weights)
    )
    return activate(weighted_sum + bias, activation)
```

Run the tests after you finish the loop. The research and Godot versions should
return the same value for the hand-calculated example above.

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

The sum \(z\) can be any number. An **activation function** turns it
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
probe changes. In Godot, `sigmoid(z) > 0.5` fires the `JUMP` event.

---

## 4 · Choose your path

The equation is shared; the evidence differs.

| | Research path | Game path |
|---|---|---|
| Inputs | Normalized temperature and pressure | Normalized speed and closeness to the edge |
| Output | Safe or unsafe class | Wait or fire the jump event |
| Main visual | Colored points and decision boundary | Input sliders, visible arc, cliff, and lava |
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

## 6 · Game path — cliff-jump timing

**Game-AI question:** Can one neuron combine speed and distance to fire a jump
at the right moment?

Open the self-contained Standard Godot project:

```bash
godot --editor --path examples/neural_foundations/game
```

Open `unit_01_jumper/unit_01_jumper.tscn` and press **F6**.

The scene starts in **Lab mode**. Nothing moves while you investigate:

- `Speed input` controls how fast the runner would move;
- `Remaining distance` controls how far the runner is from the cliff;
- `Speed weight`, `Closeness weight`, and `Bias` are the parameters you tune;
- every contribution, the sum, and the sigmoid output remain visible;
- `WAIT` means the output is at most `0.5`;
- `JUMP` means the output is greater than `0.5`.

Distance is converted into **closeness**:

$$
\text{closeness}=1-\text{remaining distance}
$$

This makes both positive weights intuitive: more speed pushes toward jumping
earlier, and more closeness pushes toward jumping now.

### Experiment 1 — make the distance signal useful

Set speed to `0.30`. Move remaining distance from `0.80` toward `0.10`.
Adjust only `Closeness weight` until the neuron waits when far away and fires
near the edge.

??? success "What you should discover"
    A positive closeness weight makes the contribution grow as the cliff gets
    nearer. A negative weight produces the dangerous opposite behavior.

### Experiment 2 — make speed change the timing

Keep the remaining distance at `0.45`. Compare speed `0.30` and `0.90`.
Adjust only `Speed weight` until the fast runner fires while the slow runner
still waits.

??? success "What you should discover"
    A positive speed contribution moves the fast case above the threshold
    sooner. A fixed rule such as `distance < 0.2` cannot make this distinction.

### Experiment 3 — shift all decisions with bias

Use the Bias slider to move the overall trigger point. Too much positive bias
makes all situations jump. Too much negative bias makes all situations wait.
Tune it until the display reads **3 / 3 cases pass**:

| Case | Speed | Remaining distance | Expected |
|---|---:|---:|---|
| Slow and far | 0.30 | 0.80 | WAIT |
| Fast and medium | 0.90 | 0.45 | JUMP |
| Slow and near | 0.30 | 0.10 | JUMP |

### Game-development evidence

Record the parameters that pass all three cases:

| Speed weight | Closeness weight | Bias | Cases passed |
|---:|---:|---:|---:|
| | | | / 3 |

Then press **Test run** several times. The runner receives a random slow,
medium, or fast speed. Watch whether the neuron fires too early, too late, or
inside the useful timing window. You are manually doing what a learning
algorithm will automate later: observe an error, adjust parameters, and test
again.

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
| Jump always fires | Bias contribution | Bias too positive |
| Jump never fires | Sum remains below zero | Bias too negative or weights too small |

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

The research boundary and jumper behavior are two views of the same forward
calculation.

| Shared role | Research visual | Game visual |
|---|---|---|
| Input \(x_1\) | Temperature position | Speed slider |
| Input \(x_2\) | Pressure position | Closeness to edge |
| Weighted sum \(z\) | Side-panel calculation | Overlay calculation |
| Threshold | Point color | `WAIT`/`JUMP` event |
| Parameter effect | Boundary rotates or shifts | Trigger time changes |
| Error evidence | Red misclassification ring | Too early, too late, or landed |

For a researcher, the boundary summarizes many observations at once. For a
game developer, motion shows one state changing over time. Neither view changes
the neuron:

```text
normalized inputs → weighted contributions → bias → activation → decision
```

Explain the equivalence aloud: rotating a classification boundary changes
which points fall on each side; changing jumper weights changes which
speed-distance combinations fire the jump.

---

## 9 · Stretch goals

**Research — export evidence.** Run:

```bash
MPLBACKEND=Agg python \
  examples/neural_foundations/research/plot_neuron.py \
  --save neuron-boundary.png
```

Add your hypothesis and parameter table beside the saved image.

**Game development — add coyote time.** Allow the jump event for a few frames
after the runner crosses the edge. Compare how this changes late failures
without changing the neuron's calculation.

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

[← Unit 0](unit-00.md) · [Course home](index.md) · [→ Neural Foundations 2](unit-neural-02.md)
