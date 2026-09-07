# Math Foundations 2 — Probability and Expectation

[← Math Foundations 1](unit-math-01.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~25 min · Working the examples: ~20 min

!!! tip "Skippable — come back when you need it"
    Like the previous unit, nothing breaks if you skip this. But the moment a
    learning curve confuses you, or a unit writes \(\mathbb{E}[\cdot]\), this is
    the page that explains what you are looking at.

!!! success "What you'll be able to do after this unit"
    - Say why a policy returns probabilities rather than one action
    - Compute an expected value by hand and explain what it predicts
    - Explain what `rollout/ep_rew_mean` in TensorBoard actually measures
    - Say why one episode tells you almost nothing, using the word *variance*
    - Read \(P(s' \mid s, a)\) and \(\pi(a \mid s)\) out loud and say what each means

!!! note "Prerequisites"
    - **Math Foundations 1** helps but is not required
    - Arithmetic with decimals and percentages
    - Basic terminal comfort

!!! info "Three ways to see the numbers"
    Python simulation (`np.mean` over many runs) · the TensorBoard reward curve
    · the units where each symbol comes back

Reinforcement learning is built on uncertainty. The environment may respond
differently to the same action, the policy deliberately acts randomly while
exploring, and rewards arrive scattered across time. Probability is not a
complication bolted onto RL — it is the material RL is made of.

---

## 1 · Where the randomness comes from

Three separate sources, and it helps to keep them apart:

| Source | What varies | Where you meet it |
|---|---|---|
| The environment | The same action leads to different next states | Physics jitter, spawn positions |
| The policy | The agent deliberately picks different actions | Exploration, [RL Essentials](unit-01.md) §5 |
| The reward | The same state pays out differently | Randomized goals, chance events |

A deterministic game can still produce a stochastic learning problem, because
the *policy* is stochastic. Early in training, the agent is sampling actions on
purpose — that is what exploration is.

---

## 2 · A distribution is a policy

A **probability distribution** assigns a number to every possible outcome. Two
rules, and only two:

- every probability is between `0` and `1`;
- they add up to exactly `1`.

That is precisely the shape of a policy's output. Given a state, the network
does not emit "jump". It emits how likely each action is:

<div class="diagram-scroll">

<svg class="course-diagram" viewBox="0 0 620 250" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="A bar chart of a policy over three actions: left with probability 0.20, jump with 0.65, right with 0.15. The three bars add up to 1.0.">
  <line x1="90" y1="200" x2="560" y2="200" stroke="#8892b0" stroke-width="1.4"/>
  <line x1="90" y1="200" x2="90" y2="40" stroke="#8892b0" stroke-width="1.4"/>
  <text x="82" y="46" text-anchor="end" fill="#8892b0" font-size="12">1.0</text>
  <text x="82" y="126" text-anchor="end" fill="#8892b0" font-size="12">0.5</text>
  <text x="82" y="205" text-anchor="end" fill="#8892b0" font-size="12">0.0</text>
  <rect x="140" y="168" width="90" height="32" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="185" y="222" text-anchor="middle" fill="#8892b0" font-size="13">left</text>
  <text x="185" y="158" text-anchor="middle" fill="#6c8ef7" font-size="14" font-weight="700">0.20</text>
  <rect x="270" y="96" width="90" height="104" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="315" y="222" text-anchor="middle" fill="#8892b0" font-size="13">jump</text>
  <text x="315" y="86" text-anchor="middle" fill="#4ecca3" font-size="14" font-weight="700">0.65</text>
  <rect x="400" y="176" width="90" height="24" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="445" y="222" text-anchor="middle" fill="#8892b0" font-size="13">right</text>
  <text x="445" y="166" text-anchor="middle" fill="#6c8ef7" font-size="14" font-weight="700">0.15</text>
  <text x="325" y="248" text-anchor="middle" fill="#8892b0" font-size="13">0.20 + 0.65 + 0.15 = 1.00</text>
</svg>

</div>

Most of the time this agent jumps. Sometimes it does not — and that leftover
`0.35` is what lets it discover that jumping is occasionally wrong.

```python
import numpy as np

actions = ["left", "jump", "right"]
policy = np.array([0.20, 0.65, 0.15])

print(policy.sum())                              # 1.0
rng = np.random.default_rng(seed=0)
print(rng.choice(actions, p=policy))             # one sampled action
print(rng.choice(actions, size=10, p=policy))    # ten of them — mostly "jump"
```

Run that last line a few times. The agent's behavior is not "jump"; it is this
distribution. **The policy is the distribution, not the action it happens to
produce.**

!!! info "Reading the notation"
    \(\pi(a \mid s)\) is spoken "pi of a given s": the probability that the
    policy \(\pi\) picks action \(a\) when the state is \(s\). The vertical bar
    means *given*, not division.

---

## 3 · Expected value is what RL maximizes

The **expected value** of a random quantity is the average you would get if you
could run it forever. You compute it by weighting each outcome by its
probability and adding them up:

$$
\mathbb{E}[X] = \sum_i p_i \, x_i
$$

A fair six-sided die:

$$
\mathbb{E}[X] = \tfrac{1}{6}(1) + \tfrac{1}{6}(2) + \dots + \tfrac{1}{6}(6) = 3.5
$$

Note that `3.5` is not a face on the die. **An expected value need not be a
possible outcome.** It is a prediction about the long-run average, not about
the next roll.

```python
import numpy as np

faces = np.array([1, 2, 3, 4, 5, 6])
probabilities = np.full(6, 1 / 6)

print(probabilities @ faces)     # 3.5 — exactly, by the formula

rng = np.random.default_rng(seed=0)
print(rng.integers(1, 7, size=10).mean())        # 2.8     — far off
print(rng.integers(1, 7, size=100_000).mean())   # 3.49805 — close
```

Two things worth noticing in that snippet. First, `probabilities @ faces` is a
**dot product** — the same operation as [Math Foundations 1](unit-math-01.md)
§3. An expected value is a weighted sum where the weights are probabilities.
Second, ten samples land nowhere near the true answer while a hundred thousand
land right on it. That gap is the subject of the next section.

!!! info "This is the reward hypothesis, in symbols"
    [RL Essentials](unit-01.md) states that goals can be expressed as maximizing
    expected cumulative reward. Now the phrase can be read literally: the agent
    maximizes \(\mathbb{E}[G]\), the expected **return**. And since the return
    discounts future rewards by \(\gamma\), the return itself is a weighted sum
    — a dot product between a reward sequence and a vector of \(\gamma\) powers.

---

## 4 · Why one episode tells you nothing

Two distributions can share an expected value and behave completely differently.
**Variance** measures how far outcomes typically scatter from the mean.

| Agent | Episode returns | Mean | Spread |
|---|---|---:|---|
| A | 10, 10, 10, 10 | 10 | none |
| B | 0, 20, 0, 20 | 10 | large |

Judging either agent from a single episode is worthless: agent B looks like a
disaster half the time and a triumph the other half, while being exactly as good
as A on average.

```python
import numpy as np

rng = np.random.default_rng(seed=1)
returns = rng.choice([0.0, 20.0], size=4)

print(returns)          # [ 0. 20. 20. 20.] — one short run
print(returns.mean())   # 15.0 — but the true expected return is 10.0
print(rng.choice([0.0, 20.0], size=5000).mean())   # 9.916 — much closer
```

Four samples reported `15.0` for a quantity whose true value is `10.0` — a 50 %
overestimate, from data that looks perfectly reasonable.

This is the arithmetic behind the warning in
[RL Essentials](unit-01.md) §5 — that a policy should never be judged from one
early episode. It is not caution for its own sake. A single sample from a
high-variance distribution genuinely carries almost no information.

!!! tip "What TensorBoard is actually showing you"
    `rollout/ep_rew_mean` is a **sample mean**: the average return over the last
    batch of episodes. It is an *estimate* of the expected return, not the
    expected return itself. That is why the curve is jagged even when the policy
    is steadily improving, and why the useful question is always "what is the
    trend across many points?" rather than "did it go up since last time?"

More samples, better estimate. That is the law of large numbers, and it is the
reason RL training runs are measured in hundreds of thousands of steps.

---

## 5 · Conditional probability is the environment

**Conditional probability** asks: given that something is already true, how
likely is something else? It is written \(P(B \mid A)\), spoken "P of B given
A".

RL's two central symbols are both conditional probabilities:

| Symbol | Spoken | Meaning |
|---|---|---|
| \(P(s' \mid s, a)\) | "P of s-prime given s and a" | The **environment**: land in state \(s'\) after taking action \(a\) in state \(s\) |
| \(\pi(a \mid s)\) | "pi of a given s" | The **policy**: choose action \(a\) when in state \(s\) |

Together they describe the whole loop from [RL Essentials](unit-01.md): the
policy decides given the state, the environment responds given the state and
action. In this course \(P(s' \mid s, a)\) is your Godot scene — the physics
engine *is* the transition function, and you never write it down explicitly.

\(s'\) is spoken "s prime" and just means "the next s".

```python
import numpy as np

rng = np.random.default_rng(seed=2)

# The environment: "thrust" usually goes up, but sometimes drifts.
outcomes = ["higher", "same", "lower"]
p_given_thrust = [0.70, 0.20, 0.10]

sample = rng.choice(outcomes, size=10_000, p=p_given_thrust)
print((sample == "higher").mean())    # 0.6982 — recovers P(higher | thrust)
```

Simulating and counting is how you would *measure* a conditional probability you
cannot read off directly. It is also, in essence, what an RL algorithm does: it
never sees \(P(s' \mid s, a)\), it only collects samples from it.

---

## 6 · Stretch goals

- **Estimate your own environment's variance.** Run a trained BallChase policy
  for 20 episodes and record each return. Compute the mean and, by hand, how
  far a typical episode falls from it. Then decide how many episodes you would
  need before trusting a comparison between two reward designs.
- **Break the distribution.** Take the policy `[0.20, 0.65, 0.15]` and change one
  entry so the three no longer sum to `1`. Predict what `rng.choice` does before
  running it.
- **Discounting as a dot product.** For rewards `[1, 1, 1, 1]` and
  \(\gamma = 0.9\), build the vector of \(\gamma\) powers and take the dot
  product. Compare the result to \(\gamma = 0.5\) and say in one sentence what
  changed about the agent's priorities.

```python
import numpy as np

rewards = np.array([1.0, 1.0, 1.0, 1.0])

for gamma in (0.9, 0.5):
    discounts = gamma ** np.arange(len(rewards))
    print(gamma, discounts.round(3), discounts @ rewards)
```

---

## What's next

You now have both halves of the notation the rest of the course uses: vectors
and dot products for how a network computes, probability and expectation for
what it is trying to maximize. **Neural Foundations 1** starts building — one
neuron, every number visible.

!!! info "Self-check before you move on"
    1. Name the three sources of randomness in an RL problem.
    2. What two rules must every probability distribution satisfy?
    3. Compute the expected value of a reward that pays `10` with probability
       `0.3` and `0` otherwise.
    4. Why can an expected value be a number that never actually occurs?
    5. What exactly does `rollout/ep_rew_mean` estimate?
    6. Read \(\pi(a \mid s)\) out loud and say what it means.

??? success "Self-check answers"
    1. The environment, the policy, and the reward.
    2. Every probability lies between `0` and `1`, and they sum to exactly `1`.
    3. \((0.3)(10) + (0.7)(0) = 3\).
    4. Because it is a probability-weighted average of the outcomes, not one of
       them — like `3.5` for a six-sided die.
    5. The expected return, estimated as a sample mean over the most recent
       batch of episodes. It is an estimate, which is why the curve is jagged.
    6. "Pi of a given s" — the probability that the policy chooses action \(a\)
       when the state is \(s\).

[← Math Foundations 1](unit-math-01.md) · [Course home](index.md) · [→ Neural Foundations 1](unit-neural-01.md)
