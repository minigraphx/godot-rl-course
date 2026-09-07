# Neural Foundations 1 — One Neuron, One Decision

[← Math Foundations 2](unit-math-02.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~30 min · Hand calculation: ~15 min · Experiments in one path: ~45 min

!!! success "What you'll be able to do after this unit"
    - Compute a neuron's output by hand from inputs, weights, and a bias
    - Say what the activation function adds, and why the decision threshold sits at `0.5`
    - Read the shorthand \(z = w_1x_1 + w_2x_2 + b\) and name every letter in it
    - Predict how changing one weight or the bias moves the decision
    - Recognize an unnormalized input from its contribution alone

!!! note "Prerequisites"
    - **Unit 0 complete** — Conda, Godot, and a successful BallChase run
    - Arithmetic with decimals. No calculus, no prior machine learning
    - Basic terminal comfort

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

## 1 · What a neuron computes

A neuron does three things in order, and nothing more.

**Step 1 — weight each input.** Every input is multiplied by its own number, the
**weight**. The result is that input's **contribution** to the decision. A large
weight makes the input matter a lot; a weight near zero makes it almost
irrelevant; a negative weight makes the input argue *against* the decision.

**Step 2 — add the contributions, plus a bias.** All contributions are summed,
and one extra number is added that belongs to no input at all: the **bias**. The
bias is the neuron's default leaning before it has looked at anything. The
running total is the **weighted sum**.

**Step 3 — turn the sum into a decision.** This is the step that needs
explaining, because the weighted sum is an awkward number to act on.

### Why step 3 is needed

The weighted sum can land anywhere: `-37.2`, `0.02`, `+415.0`. But the question
being asked is not "what number is this?" — it is "should the character jump?"
The answer wanted is a confidence between *definitely not* and *definitely yes*.

An **activation function** performs that conversion. This unit uses the one
called **sigmoid**, which squashes any number, however large or small, into the
range between `0` and `1`.

<div class="diagram-scroll">

<svg class="course-diagram" viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="The sigmoid curve: it approaches 0 for large negative sums, passes through exactly 0.5 at a sum of zero, and approaches 1 for large positive sums">
  <defs>
    <marker id="arS" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 1 L10 5 L0 9 z" fill="#8892b0"/>
    </marker>
  </defs>
  <line x1="70" y1="50" x2="610" y2="50" stroke="#2e3350" stroke-width="1.1" stroke-dasharray="4 4"/>
  <line x1="70" y1="150" x2="610" y2="150" stroke="#ef8354" stroke-width="1.2" stroke-dasharray="5 5"/>
  <path d="M70 250 L616 250" fill="none" stroke="#8892b0" stroke-width="1.4" marker-end="url(#arS)"/>
  <path d="M340 268 L340 26" fill="none" stroke="#8892b0" stroke-width="1.4" marker-end="url(#arS)"/>
  <text x="606" y="270" text-anchor="end" fill="#8892b0" font-size="13">weighted sum</text>
  <text x="330" y="34" text-anchor="end" fill="#8892b0" font-size="13">output</text>
  <text x="62" y="55" text-anchor="end" fill="#8892b0" font-size="13">1.0</text>
  <text x="62" y="155" text-anchor="end" fill="#8892b0" font-size="13">0.5</text>
  <text x="62" y="255" text-anchor="end" fill="#8892b0" font-size="13">0.0</text>
  <text x="78" y="142" fill="#ef8354" font-size="13" font-weight="700">0.5 — the decision threshold</text>
  <path d="M70.0 249.5 L81.2 249.4 L92.5 249.2 L103.8 249.0 L115.0 248.7 L126.2 248.3 L137.5 247.8 L148.8 247.2 L160.0 246.4 L171.2 245.4 L182.5 244.1 L193.8 242.5 L205.0 240.5 L216.2 238.0 L227.5 234.8 L238.8 230.9 L250.0 226.2 L261.2 220.4 L272.5 213.5 L283.8 205.5 L295.0 196.2 L306.2 185.8 L317.5 174.5 L328.8 162.4 L340.0 150.0 L351.2 137.6 L362.5 125.5 L373.8 114.2 L385.0 103.8 L396.2 94.5 L407.5 86.5 L418.8 79.6 L430.0 73.8 L441.2 69.1 L452.5 65.2 L463.8 62.0 L475.0 59.5 L486.2 57.5 L497.5 55.9 L508.8 54.6 L520.0 53.6 L531.2 52.8 L542.5 52.2 L553.8 51.7 L565.0 51.3 L576.2 51.0 L587.5 50.8 L598.8 50.6 L610.0 50.5" fill="none" stroke="#4ecca3" stroke-width="2.6" stroke-linejoin="round"/>
  <circle cx="340" cy="150" r="6" fill="#4ecca3"/>
  <text x="352" y="186" fill="#8892b0" font-size="13">sum 0 → exactly 0.5</text>
  <text x="160" y="228" text-anchor="middle" fill="#8892b0" font-size="13">very negative → near 0</text>
  <text x="520" y="82" text-anchor="middle" fill="#8892b0" font-size="13">very positive → near 1</text>
</svg>

</div>

Three landmarks are worth memorizing, because every later reading of this curve
depends on them:

| Weighted sum | Sigmoid output | How to read it |
|---:|---:|---|
| -3.00 | 0.047 | almost certainly no |
| -0.50 | 0.378 | leaning no |
| 0.00 | **0.500** | perfectly undecided |
| +0.20 | 0.550 | leaning yes |
| +3.00 | 0.953 | almost certainly yes |

This is where the threshold `0.5` comes from. It is not an arbitrary cut-off:
sigmoid returns exactly `0.5` when the weighted sum is exactly `0`. So
**"output above 0.5" and "weighted sum above 0" are the same statement**. The
neuron fires when the contributions and the bias together add up to something
positive.

!!! note "The formula, for reference only"
    $$
    \operatorname{sigmoid}(z) = \frac{1}{1 + e^{-z}}
    $$

    You never have to evaluate this by hand in this course. Read the value off
    the curve, or let the code compute it. What matters is the shape: it never
    leaves the range `0` to `1`, and it crosses `0.5` at zero.

### The shorthand you will meet everywhere

Named words become long, so mathematics abbreviates them. The abbreviations are
introduced once, here, and used for the rest of the course:

| Meaning in words | Shorthand | Said aloud | Why that letter |
|---|---|---|---|
| first input, second input | \(x_1\), \(x_2\) | "x one", "x two" | \(x\) is the traditional letter for an unknown quantity |
| the weight belonging to each input | \(w_1\), \(w_2\) | "w one", "w two" | \(w\) for **w**eight — Latin \(w\), not Greek \(\omega\) |
| bias | \(b\) | "b" | \(b\) for **b**ias |
| weighted sum (the result of steps 1 and 2) | \(z\) | "z" | the conventional letter for the sum before activation |
| "add all of these up", drawn as a box in diagrams | \(\Sigma\) | "sigma" | Greek capital S, for **s**um |
| output after the activation function | — | "output" | it keeps its plain name in this course |

So the whole of steps 1 and 2 is written:

$$
z = w_1x_1 + w_2x_2 + b
$$

Four reading notes that trip people up:

- \(w_1\) and `w₁` are the **same thing**. Prose and slider labels in this
  course use `w₁`; formulas use \(w_1\). Weight is always a lowercase \(w\).
- Latin \(w\) and Greek \(\omega\) ("omega") look nearly identical,
  especially handwritten or in an italic maths font. In this course \(w\) is
  always a weight. \(\omega\) appears only much later, in
  [Hierarchical RL](unit-hierarchical.md), where it names an *option* — a
  completely unrelated idea.
- The small lowered number is a label, not a power. \(x_1\) means "the first
  input", not "x to the power of one".
- A Greek letter is read by its name, never by its shape. \(\Sigma\) is
  spoken "sigma". Later units bring \(\gamma\) ("gamma"), \(\alpha\)
  ("alpha"), and \(\varepsilon\) ("epsilon"); the
  [glossary](glossary.md) names each one.

---

## 2 · Predict before running

Now use the three steps on a **fixed-number neuron**. This is a prediction
exercise: work it out on paper *before* opening the answer or running anything.

| Named input | Value | Weight |
|---|---:|---:|
| Speed | 0.50 | +0.80 |
| Closeness to edge | 0.25 | +1.20 |
| Bias | — | -0.50 |

Write down, in this order:

1. each input's contribution — value × weight;
2. the weighted sum, contributions plus the bias;
3. whether the sigmoid output is above `0.5` — read it off the curve in
   Section 1, no calculator needed;
4. which input pushes hardest toward jumping.

??? success "Worked answer — open this after you have written yours"
    **1. Contributions**

    | Named input | Value | Weight | Contribution |
    |---|---:|---:|---:|
    | Speed | 0.50 | +0.80 | +0.40 |
    | Closeness to edge | 0.25 | +1.20 | +0.30 |
    | Bias | — | — | -0.50 |

    **2. Weighted sum**

    $$
    z =
    (\text{speed}\times\text{speed weight}) +
    (\text{closeness}\times\text{closeness weight}) +
    \text{bias}
    $$

    $$
    z = (0.5)(0.8) + (0.25)(1.2) - 0.5 = 0.2
    $$

    **3. Decision.** The sum is positive, so the output must be above `0.5`
    before you compute anything: \(\operatorname{sigmoid}(0.2) \approx 0.550\).
    The neuron **fires**, and the game action is `JUMP`.

    **4. Strongest push.** Speed contributes `+0.40`, closeness `+0.30`, and the
    bias subtracts `0.50`. Speed pushes hardest toward jumping — but note that
    the bias alone is larger than either contribution, which is why the decision
    is so close to the threshold.

    The whole calculation as one picture:

    <div class="diagram-scroll">

    <svg class="course-diagram" viewBox="0 0 800 300" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="One neuron: speed 0.50 times weight 0.80 gives plus 0.40, closeness 0.25 times weight 1.20 gives plus 0.30, summed with bias minus 0.50 gives 0.20; sigmoid gives 0.550, which is above 0.5, so the neuron fires JUMP">
      <defs>
        <marker id="arN" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 1 L10 5 L0 9 z" fill="#8892b0"/>
        </marker>
      </defs>
      <rect x="20" y="40" width="180" height="60" rx="10" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
      <text x="110" y="65" text-anchor="middle" fill="#e2e8f0" font-size="14" font-weight="700">Speed</text>
      <text x="110" y="86" text-anchor="middle" fill="#8892b0" font-size="13">0.50</text>
      <rect x="20" y="180" width="180" height="60" rx="10" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
      <text x="110" y="205" text-anchor="middle" fill="#e2e8f0" font-size="14" font-weight="700">Closeness to edge</text>
      <text x="110" y="226" text-anchor="middle" fill="#8892b0" font-size="13">0.25</text>
      <path d="M200 70 C280 70, 290 115, 350 122" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#arN)"/>
      <text x="272" y="58" text-anchor="middle" fill="#6c8ef7" font-size="12" font-weight="700">× 0.80 → +0.40</text>
      <path d="M200 210 C280 210, 290 165, 350 158" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#arN)"/>
      <text x="272" y="232" text-anchor="middle" fill="#6c8ef7" font-size="12" font-weight="700">× 1.20 → +0.30</text>
      <rect x="350" y="100" width="140" height="80" rx="10" fill="#1a1d27" stroke="#8892b0" stroke-width="1.5"/>
      <text x="420" y="132" text-anchor="middle" fill="#e2e8f0" font-size="15" font-weight="700">sum + bias</text>
      <text x="420" y="158" text-anchor="middle" fill="#8892b0" font-size="13">= 0.20</text>
      <path d="M420 242 L420 184" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#arN)"/>
      <text x="420" y="264" text-anchor="middle" fill="#8892b0" font-size="13">bias −0.50</text>
      <path d="M490 140 L540 140" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#arN)"/>
      <rect x="540" y="100" width="140" height="80" rx="10" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
      <text x="610" y="132" text-anchor="middle" fill="#e2e8f0" font-size="15" font-weight="700">sigmoid</text>
      <text x="610" y="158" text-anchor="middle" fill="#8892b0" font-size="13">≈ 0.550</text>
      <path d="M680 140 L716 140" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#arN)"/>
      <text x="755" y="136" text-anchor="middle" fill="#4ecca3" font-size="16" font-weight="700">JUMP</text>
      <text x="755" y="160" text-anchor="middle" fill="#8892b0" font-size="11">0.550 &gt; 0.5</text>
    </svg>

    </div>

**Visible check:** the automated examples use these same numbers.

!!! note "Run from the course repo root"
    These commands assume your terminal is in the [course repo root](setup.md#course-repo) and that `godot` is on your PATH — see [Godot on the command line](setup.md#godot-cli).

```bash
conda activate godot_env
python -m examples.neural_foundations.research.tests.test_neuron

godot --headless \
  --path examples/neural_foundations/game \
  --script res://test/test_tiny_neuron.gd
```

Both commands print the same walkthrough (`sum (z) = +0.200`,
`sigmoid(sum) = 0.550`) and then end with `OK`. The Godot run also shows the
jumper demo's live labels (speed, closeness, sum, output, and `WAIT`/`JUMP`).

Both tests call the forward pass you will inspect next.

---

## 3 · Weighted inputs and bias

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

## 4 · Activation functions

Sigmoid is not the only way to turn the sum \(z\) into a decision. It is the
one used for the jump trigger because a confidence between `0` and `1` is what
that decision needs, but other decisions need other output shapes.

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

## 5 · Choose your path

The equation is shared; the evidence differs.

| | Research path | Game path |
|---|---|---|
| Inputs | Normalized temperature and pressure | Normalized speed and closeness to the edge |
| Output | Safe or unsafe class | Wait or fire the jump event |
| Main visual | Colored points and decision boundary | Input sliders, visible arc, cliff, and lava |
| Evidence | Accuracy and misclassified points | Expected versus actual behavior |
| Tool | Python + Matplotlib | Standard Godot 4 + GDScript |

Choose one primary path:

- **Research:** complete Section 6 and view the Godot comparison once.
- **Game development:** complete Section 7 and view the plot comparison once.

No native extension, C#, training framework, or prior machine-learning library
is needed in this unit.

---

## 6 · Research path — visible decision boundary

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

## 7 · Game path — cliff-jump timing

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

## 8 · Break it deliberately

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

## 9 · Compare the two paths

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

## 10 · Stretch goals

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

!!! info "Self-check before you move on"
    1. What are the three steps a neuron performs, in order?
    2. What does the bias do that a weight cannot?
    3. Why is the decision threshold `0.5` and not some other number?
    4. In \(z = w_1x_1 + w_2x_2 + b\), what is each letter?
    5. A contribution comes out as `-4.00` while every other one is below `1`.
       What is the most likely cause?
    6. Changing a weight does what to the decision boundary? Changing the bias?

??? success "Self-check answers"
    1. Weight each input, add the contributions plus the bias, then pass the
       weighted sum through an activation function.
    2. The bias shifts every decision at once, regardless of the inputs. It is
       the neuron's default leaning; a weight only acts when its input is
       non-zero.
    3. Because sigmoid returns exactly `0.5` when the weighted sum is `0`.
       "Output above `0.5`" is another way of saying "weighted sum above `0`".
    4. \(x_1, x_2\) are the inputs, \(w_1, w_2\) their weights, \(b\) the
       bias, and \(z\) the weighted sum before the activation function.
    5. A missing normalization — that input is still in its raw units, so its
       contribution drowns out the others.
    6. A weight rotates the boundary; the bias shifts it without rotating it.

[← Math Foundations 2](unit-math-02.md) · [Course home](index.md) · [→ Neural Foundations 2](unit-neural-02.md)
