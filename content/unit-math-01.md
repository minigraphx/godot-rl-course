# Math Foundations 1 — Vectors and Matrices

[← Unit 0](unit-00.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~25 min · Working the examples: ~20 min

!!! tip "Skippable — come back when you need it"
    Nothing in Phase 1 or 2 breaks if you skip this unit. It exists so that when
    a later unit writes \(w^\top x\) or "the observation vector", you have
    somewhere to look it up. If the arithmetic in Neural Foundations 1 felt
    comfortable, go straight there and return when something stops making sense.

!!! success "What you'll be able to do after this unit"
    - Read an observation as a vector and say what its dimension means
    - Compute a dot product by hand and recognize it as the weighted sum you already know
    - Explain what normalizing a vector does and why the course keeps inputs in `0`–`1`
    - Read a matrix as a stack of weight rows — one row per neuron
    - Predict the output shape of a matrix–vector product, and diagnose a shape error

!!! note "Prerequisites"
    - **Unit 0 complete** — Conda and a working `godot_env`
    - Arithmetic with decimals. No calculus
    - Basic terminal comfort

!!! info "Three ways to see the numbers"
    Python + NumPy (`print(v.shape)`) · the observation array Godot sends · the
    course units where each idea comes back

Reinforcement learning is written in the language of vectors and matrices, but
the ideas underneath are small. A vector is a list of numbers. A matrix is a
list of lists. Everything in this unit is counting, multiplying, and adding —
arranged so that a computer can do thousands of them at once.

---

## 1 · A vector is an observation

When a Godot agent reports what it sees, it does not send one number. It sends
several, in a fixed order:

```text
[ distance to goal, speed, angle to target, health ]
[ 0.42,             0.80,  -0.15,           1.00   ]
```

That ordered list is a **vector**. Two properties matter:

- its **dimension** — how many numbers it holds. The vector above is
  4-dimensional;
- its **order** — position 2 is always speed. Swap two entries and the agent
  is now being told something false.

```python
import numpy as np

observation = np.array([0.42, 0.80, -0.15, 1.00])

print(observation)          # [ 0.42  0.8  -0.15  1.  ]
print(observation.shape)    # (4,)
print(observation[1])       # 0.8  — speed
```

`shape` reads `(4,)` — a single axis holding four numbers. This is the same
array that `get_obs()` produces on the Godot side, and the same one the policy
network receives as input.

!!! warning "Order is a contract"
    If `get_obs()` returns distance first and the network was trained expecting
    speed first, nothing errors. The agent simply behaves nonsensically. A
    silent reordering is one of the hardest bugs in this course to spot, which
    is why [Unit 2](unit-02.md) has you write the observation list down before
    you write the code.

---

## 2 · Adding and scaling

Two vectors of the same dimension add **element by element**:

$$
\begin{bmatrix} 2 \\ 1 \end{bmatrix} +
\begin{bmatrix} 3 \\ 4 \end{bmatrix} =
\begin{bmatrix} 5 \\ 5 \end{bmatrix}
$$

Multiplying by a single number — a **scalar** — stretches every entry by the
same factor:

$$
0.5 \times \begin{bmatrix} 4 \\ 6 \end{bmatrix} =
\begin{bmatrix} 2 \\ 3 \end{bmatrix}
$$

```python
import numpy as np

position = np.array([2.0, 1.0])
velocity = np.array([3.0, 4.0])

print(position + velocity)    # [5. 5.]
print(0.5 * velocity)         # [1.5 2. ]
```

In a game this is ordinary movement: a position plus a velocity is the next
position, and scaling the velocity by `delta` is exactly what
`position += velocity * delta` does in GDScript. The vector notation is not a
new idea — it is the same line of code, written for many numbers at once.

---

## 3 · The dot product is the weighted sum

This is the connection worth the whole unit.

In [Neural Foundations 1](unit-neural-01.md) a neuron multiplied every input by
its weight and added the results:

$$
z = w_1x_1 + w_2x_2 + b
$$

That pattern — multiply matching pairs, then add everything up — has a name.
It is the **dot product** of the input vector and the weight vector, written
\(w \cdot x\).

$$
\begin{bmatrix} 0.5 \\ 0.25 \end{bmatrix} \cdot
\begin{bmatrix} 0.8 \\ 1.2 \end{bmatrix}
= (0.5)(0.8) + (0.25)(1.2) = 0.4 + 0.3 = 0.7
$$

Those are the numbers from Neural Foundations 1. Add the bias `-0.5` and you get
`0.2` — the same weighted sum, reached the same way. Nothing new happened; the
operation just acquired a name.

<div class="diagram-scroll">

<svg class="course-diagram" viewBox="0 0 700 220" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="Two vectors side by side: inputs 0.50 and 0.25, weights 0.80 and 1.20. Matching entries are multiplied to give 0.40 and 0.30, which are added to give the single number 0.70.">
  <text x="90" y="34" text-anchor="middle" fill="#8892b0" font-size="13">inputs</text>
  <text x="240" y="34" text-anchor="middle" fill="#8892b0" font-size="13">weights</text>
  <rect x="45" y="50" width="90" height="44" rx="8" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="90" y="78" text-anchor="middle" fill="#e2e8f0" font-size="16">0.50</text>
  <rect x="45" y="106" width="90" height="44" rx="8" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="90" y="134" text-anchor="middle" fill="#e2e8f0" font-size="16">0.25</text>
  <rect x="195" y="50" width="90" height="44" rx="8" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="240" y="78" text-anchor="middle" fill="#e2e8f0" font-size="16">0.80</text>
  <rect x="195" y="106" width="90" height="44" rx="8" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="240" y="134" text-anchor="middle" fill="#e2e8f0" font-size="16">1.20</text>
  <text x="165" y="79" text-anchor="middle" fill="#8892b0" font-size="17">×</text>
  <text x="165" y="135" text-anchor="middle" fill="#8892b0" font-size="17">×</text>
  <text x="330" y="79" text-anchor="middle" fill="#8892b0" font-size="17">=</text>
  <text x="330" y="135" text-anchor="middle" fill="#8892b0" font-size="17">=</text>
  <text x="395" y="80" text-anchor="middle" fill="#e2e8f0" font-size="16">0.40</text>
  <text x="395" y="136" text-anchor="middle" fill="#e2e8f0" font-size="16">0.30</text>
  <line x1="355" y1="156" x2="435" y2="156" stroke="#8892b0" stroke-width="1.4"/>
  <text x="332" y="162" text-anchor="middle" fill="#8892b0" font-size="17">+</text>
  <text x="395" y="184" text-anchor="middle" fill="#4ecca3" font-size="18" font-weight="700">0.70</text>
  <text x="560" y="100" text-anchor="middle" fill="#8892b0" font-size="14">two vectors in,</text>
  <text x="560" y="122" text-anchor="middle" fill="#8892b0" font-size="14">one single number out</text>
</svg>

</div>

```python
import numpy as np

inputs = np.array([0.5, 0.25])
weights = np.array([0.8, 1.2])
bias = -0.5

print(inputs @ weights)                    # 0.7 — the dot product
print(round(inputs @ weights + bias, 3))   # 0.2 — Neural Foundations 1, section 2
```

The `@` operator is Python's dot product. `np.dot(inputs, weights)` does the
same thing.

!!! note "Why the code rounds"
    Without `round`, that second line prints `0.19999999999999996`. Decimal
    fractions have no exact binary representation, so tiny errors accumulate.
    This is normal and harmless here, but it is the reason you compare floats
    with a tolerance rather than with `==`.

!!! info "Why it matters that this has a name"
    Every layer of every network in this course is dot products. When a later
    unit writes \(\pi_\theta(a \mid s)\) and shows a matrix multiplication, it
    is doing what you just did by hand — many times, in parallel.

**Both vectors must have the same dimension.** Two inputs need exactly two
weights. This is not a convention; there is simply no fourth number to pair
with a third weight.

---

## 4 · Length and normalization

The **length** of a vector — its **norm**, written \(\lVert v \rVert\) — comes
from Pythagoras:

$$
\left\lVert \begin{bmatrix} 3 \\ 4 \end{bmatrix} \right\rVert
= \sqrt{3^2 + 4^2} = \sqrt{25} = 5
$$

**Normalizing** means dividing a vector by its own length. The result points in
the same direction but has length `1`:

```python
import numpy as np

v = np.array([3.0, 4.0])
length = np.linalg.norm(v)

print(length)       # 5.0
print(v / length)   # [0.6 0.8]  — same direction, length 1
```

This is the same idea as the "Why normalize?" section of
[Neural Foundations 1](unit-neural-01.md): raw pressure at `80` drowned out a
temperature at `0.7`, because the contribution of an input depends on its
magnitude. Keeping every observation in `0`–`1` means no single input can
dominate through its units alone.

!!! tip "Direction versus magnitude"
    Normalizing separates *where a vector points* from *how big it is*. In a
    game that is often exactly the split you want: the direction toward the
    player is the useful signal, while the raw distance in pixels depends on
    your screen resolution.

---

## 5 · A matrix is a layer of neurons

One neuron holds one weight vector. A **layer** holds several neurons, each with
its own weight vector. Stack those vectors as rows and you have a **matrix**:

$$
W =
\begin{bmatrix}
0.8 & 1.2 \\
-0.5 & 0.9 \\
0.3 & 0.3
\end{bmatrix}
$$

Three rows, two columns: **three neurons, each reading two inputs**. Row 1 is
the first neuron's weights, row 2 the second neuron's, row 3 the third's.

Multiplying this matrix by an input vector computes all three dot products at
once:

```python
import numpy as np

W = np.array([
    [0.8, 1.2],     # neuron 1 weights
    [-0.5, 0.9],    # neuron 2 weights
    [0.3, 0.3],     # neuron 3 weights
])
inputs = np.array([0.5, 0.25])

print(W @ inputs)     # [ 0.7  -0.025  0.225 ]
print(W.shape)        # (3, 2)  — 3 neurons, 2 inputs each
```

The first entry, `0.7`, is the dot product from section 3. The other two are the
same computation with the other neurons' weights — neuron 2 comes out negative,
because its first weight is negative and it is fed a positive input. **A matrix–vector product is
just several dot products stacked.**

That is the whole of what "the network does a forward pass" means: a matrix
multiplication per layer, an activation function after each, repeated.

---

## 6 · Shapes must line up

The single most common error when working with these arrays is a **shape**
mismatch. The rule:

> To compute `W @ x`, the number of **columns** of `W` must equal the
> **dimension** of `x`. The result has as many entries as `W` has **rows**.

$$
\underbrace{(3 \times 2)}_{W} \cdot \underbrace{(2)}_{x} \rightarrow
\underbrace{(3)}_{\text{result}}
$$

The inner numbers must match; the outer ones survive.

| Shapes | Works? | Result |
|---|---|---|
| `(3, 2) @ (2,)` | yes | `(3,)` |
| `(3, 2) @ (3,)` | no | inner `2` and `3` disagree |
| `(4, 8) @ (8,)` | yes | `(4,)` |
| `(2,) @ (2,)` | yes | a single number — the dot product |

Read the error message rather than guessing:

```text
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0,
with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 2)
```

The useful part is at the end: `size 3 is different from 2`. Something arrived
with 3 entries where 2 were expected — nearly always an observation vector whose
length no longer matches the network the policy was built for.

| Symptom | First thing to check |
|---|---|
| `size N is different from M` | Did `get_obs()` gain or lose an entry? |
| Result has the wrong number of entries | Are you multiplying by rows when you meant columns? |
| Works in Python, wrong behavior in Godot | Is the observation order the same on both sides? |

---

## 7 · Stretch goals

- **Cosine similarity.** Normalize two observation vectors and take their dot
  product. The result runs from `-1` to `1` and measures how similar two
  situations are, independent of their magnitudes. Compute it for two states
  of your own choosing and check whether the number matches your intuition.
- **Count a real network.** Neural Foundations 2 builds a small network. Write
  down the shape of each weight matrix and verify that the shapes chain
  together from input to output.
- **Transpose.** `W.T` flips rows and columns. Predict the shape of `W.T` for
  the `(3, 2)` matrix above, then check. Say in one sentence what the rows of
  `W.T` now mean.

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 4.0, 6.0])

cosine = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(cosine)   # 1.0 — b is just a scaled up, so the direction is identical
```

---

## What's next

You can now read the shapes that the rest of the course prints at you. In
**Math Foundations 2** the other half arrives: probability and expected value,
which is what a reward curve is actually averaging and what a policy actually
returns.

!!! info "Self-check before you move on"
    1. What does the dimension of an observation vector correspond to in Godot?
    2. Compute \([2, 3] \cdot [4, 1]\) by hand.
    3. Which section of Neural Foundations 1 was secretly a dot product?
    4. What does normalizing a vector preserve, and what does it discard?
    5. A matrix has shape `(5, 3)`. How many neurons is that, reading how many inputs?
    6. `(5, 3) @ (5,)` — does it work? Why or why not?

??? success "Self-check answers"
    1. The number of values `get_obs()` returns — one entry per thing the agent
       is told about the world.
    2. \((2)(4) + (3)(1) = 8 + 3 = 11\).
    3. The weighted sum, \(z = w_1x_1 + w_2x_2 + b\) — the dot product of the
       input and weight vectors, plus the bias.
    4. It preserves the direction and discards the magnitude; the result always
       has length `1`.
    5. Five neurons, each reading three inputs.
    6. No. The inner dimensions must match: the matrix has `3` columns but the
       vector has `5` entries. `(5, 3) @ (3,)` would work and give `(5,)`.

[← Unit 0](unit-00.md) · [Course home](index.md) · [→ Math Foundations 2](unit-math-02.md)
