# PPO Deep Dive — Clipped Objective, GAE, and Hyperparameters

This is the capstone of the theory sequence. You have already walked the long path: **Q-Learning** taught you the Bellman equation and value bootstrapping; **DQN** scaled values to neural networks; **REINFORCE** flipped to policy gradients; **Actor-Critic / A2C** glued them together with a baseline. PPO is the algorithm that takes all of those ideas, fixes A2C's two great weaknesses, and ends up as the workhorse of modern applied RL — the algorithm Godot RL Agents runs by default, the one OpenAI used for Dota 2, the one you will use for every project in this course.

This unit does not show you how to *use* PPO — that's the next unit. This unit explains **why every line of the PPO pseudocode is the way it is**.

[← Actor-Critic](unit-actor-critic.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~45 min

---

!!! info "Three ways to see your AI"
    TensorBoard (`train/approx_kl`, `rollout/ep_rew_mean`, `train/entropy_loss`) · CleanRL `ppo.py` open in your editor · A Godot agent training with the exact hyperparameters you tuned in this unit

---

## Why this unit exists

Every time you ran `gdrl` so far, PPO was the algorithm doing the work. You saw `clip_range`, `gae_lambda`, `n_epochs`, `vf_coef` scroll past in the logs and you trusted the defaults. That trust ends here. By the end of this unit you should be able to:

- Read the original PPO paper (Schulman et al. 2017) without skipping any equation.
- Open CleanRL's single-file `ppo.py` and recognize every block.
- Look at a TensorBoard curve, identify what's wrong, and pick the right hyperparameter to change.
- Explain to another student, in plain English, why the `min` and `clip` in the PPO objective are there.

Theory units in this course always pair an equation with a story. PPO has a lot of equations — but only one *idea*: **let the policy improve as much as possible, but never so much that the data we used to compute the improvement becomes irrelevant**.

---

## 1 · The A2C problem that PPO solves

Let's start by remembering what A2C does, and where it hurts.

**A2C in one loop:**

```
loop:
    collect n_steps transitions using current policy π_θ
    compute advantages A_t = G_t - V(s_t)
    compute one gradient step:
        L = -log π_θ(a_t|s_t) · A_t  +  c1 · (V_θ(s_t) - G_t)²  -  c2 · H(π_θ)
    update θ via Adam
    throw away the rollout
```

That `throw away the rollout` is painful. We collected, say, 2048 transitions across 8 parallel environments — 16384 (state, action, reward) tuples. We took *one* gradient step with them, then dropped them on the floor and went back to the simulator. In a slow environment like Godot (each step requires the engine to tick, render, and round-trip through the WebSocket bridge), collecting data is by far the bottleneck. Using each transition only once is wasteful.

**The obvious fix:** take *many* gradient steps per rollout. Loop over the same data 10 times. Get 10× the policy improvement per environment step. Done, right?

**No.** And here is where the trouble starts.

### The instability problem

After the *first* gradient step, the policy is no longer π_θ — it's π_θ' (slightly different). The advantages we computed, A_t, were computed under the *old* policy. They are estimates of *"how much better than average is action a_t, assuming we're behaving like π_θ?"* — not π_θ'.

After the *second* gradient step, the policy is π_θ''. The advantages are now even more wrong.

After ten steps, the policy might be very different from the one that collected the data. The advantages we are still using to drive gradient updates are now estimates from a *completely different policy*. We are essentially flying blind, taking confident updates based on stale information. In practice this leads to one of two failure modes:

1. **Catastrophic collapse:** the policy diverges into a degenerate distribution (always picks the same action), entropy collapses, episode reward crashes to baseline.
2. **Oscillation:** the policy ping-pongs between two regions, never settling. Loss looks noisy, reward never improves.

This is the central conflict of off-policy-ish learning:

| Desire | Risk |
|--------|------|
| Reuse data many times (sample efficiency) | Policy drifts from data-collecting policy (advantages become wrong) |
| Take small, safe steps (stability) | Update rarely, waste data (sample inefficiency) |

**PPO is the answer to this exact tradeoff.** It lets you take many gradient steps on the same data, but cleverly *stops* the update for any individual transition once the policy has drifted "too far" from the one that collected it. Each transition contributes only while it is still relevant.

---

## 2 · Trust Region intuition

The mental model that unlocks PPO is the **trust region**.

> Imagine you're hiking near the edge of a cliff in dense fog. You can see one or two meters around you (the local reward landscape). You'd love to take a giant leap toward higher ground — but a giant leap risks walking off the cliff. So you commit only to small steps, within a region you *trust*.

In policy optimization terms: the data you collected gives you a good estimate of which actions are better than average **near the current policy**. Once the new policy is too different from the data-collecting policy, those estimates are no longer reliable. The "trust region" is the set of policies close enough to the old one that the old data still gives valid advantage estimates.

### TRPO — the formal version

PPO's predecessor, **TRPO** (Trust Region Policy Optimization, Schulman et al. 2015), formalized this idea with a hard mathematical constraint:

$$
\max_{\theta} \; \mathbb{E}_t \!\left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \, A_t \right] \quad \text{subject to} \quad \mathbb{E}_t [\text{KL}(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta)] \leq \delta
$$

In words: *maximize expected advantage, but the KL divergence between the new and old policy must stay below threshold δ*. KL divergence measures how different two probability distributions are; constraining it constrains the policy step size.

TRPO works beautifully — but it requires **second-order optimization** (computing and inverting an approximation to the Fisher information matrix). That's expensive, hard to implement, and doesn't play well with shared actor-critic networks.

**How TRPO solves the constraint:** conjugate gradient to compute the natural gradient direction, then backtracking line search to find the largest step that satisfies the KL bound. O(n²) in policy parameters per update step.

### PPO — the practical version

PPO (Schulman et al. 2017) asked: *can we get TRPO's stability without the second-order math?* The answer turned out to be embarrassingly simple — **clip the objective so that the gradient becomes zero once the policy has moved "far enough"**. No constraint, no KL computation, no Lagrangian. Just a `min` and a `clip` inside the loss function. First-order optimization (vanilla Adam) is all you need.

This is the one trick that took PPO from "interesting" to "standard."

| | TRPO | PPO |
|--|------|-----|
| Constraint | Hard KL ≤ δ | Soft clip on ratio r_t |
| Optimizer | Conjugate gradient + line search | Adam |
| Compute per update | O(n²) | O(n) |
| Hyperparameter | δ (KL threshold) | ε (`clip_range`), `target_kl` |
| When to use | Safety-critical tasks requiring near-monotonic improvement guarantees | Everything else |

**PPO's `target_kl`** (SB3 parameter, default None) adds an optional early stopping check: if the approximate KL between old and new policy exceeds `target_kl` during a minibatch epoch, training stops early. This gives PPO a soft version of TRPO's hard constraint at negligible cost.

---

## 3 · The probability ratio

To clip the policy step size, we first need a way to *measure* it. PPO uses the **importance sampling ratio**:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

Let's unpack this carefully.

- **π_θ** is the *current* policy — the one being updated in this gradient step.
- **π_θ_old** is the *frozen* policy that collected the rollout. Its parameters are stored once, at the start of the rollout, and never change during the update epochs.
- The numerator: probability of action a_t under the new policy.
- The denominator: probability of action a_t under the old policy.

**Interpretation of r_t:**

| r_t value | Meaning |
|-----------|---------|
| r_t = 1 | The new policy assigns the *same* probability to a_t as the old one. (Always true at the start of an update — before the first gradient step, θ = θ_old.) |
| r_t > 1 | The new policy is *more* likely to pick a_t than the old one was. The update has *increased* the probability of this action. |
| r_t < 1 | The new policy is *less* likely to pick a_t. The update has *decreased* the probability. |
| r_t = 2 | The new policy is twice as likely to pick a_t as the old one. |
| r_t = 0.1 | The new policy almost never picks a_t anymore. |

So r_t is a clean, direct measurement of *how much the policy has moved for this particular action*. It's a per-transition trust-region meter.

### The unclipped surrogate (CPI)

If we just multiply r_t by the advantage and maximize, we get the **Conservative Policy Iteration** objective (Kakade & Langford, 2002), which is also exactly what REINFORCE+importance-sampling would give:

$$
L^{\text{CPI}}(\theta) = \mathbb{E}_t \big[ r_t(\theta) \cdot A_t \big]
$$

This says: *push up the probability of actions with positive advantage, push down the probability of actions with negative advantage, weighted by how much we've already moved*. It's mathematically equivalent to REINFORCE when r_t = 1 (single gradient step), but generalizes to multiple steps via the ratio.

Compare to the REINFORCE objective from Unit 5:

$$
L^{\text{REINFORCE}}(\theta) = \mathbb{E}_t \big[ \log \pi_\theta(a_t|s_t) \cdot A_t \big]
$$

Take the gradient of both. They are *identical* at θ = θ_old, because ∇log π = (1/π)∇π, and r_t starts at 1. The ratio form just survives further from the starting point — it accumulates the policy drift correctly.

**The problem with L^CPI:** it has no brakes. If A_t is large and positive, the gradient happily pushes r_t to 5, 10, 100 — far outside any reasonable trust region. Catastrophic update territory. We need a cap.

---

## 4 · The clipped PPO objective (THE key equation)

Here it is. The one equation that defines PPO:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \Big[ \min\big( r_t(\theta) \cdot A_t, \; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \cdot A_t \big) \Big]
$$

Where:

- `clip(x, a, b)` = `max(a, min(x, b))` — i.e. confine x to the interval [a, b].
- **ε** (epsilon) is a hyperparameter, typically **0.2**. It defines the "trust region radius."
- **min(·, ·)** picks the *smaller* of the two terms.

This looks intimidating. It isn't. Let's walk through both cases.

### Case 1: A_t > 0 (the action was better than average)

We want to *increase* π_θ(a_t|s_t) — make this good action more likely. As we update, r_t grows above 1.

- **Without the clip:** L = r_t · A_t. Gradient pushes r_t up indefinitely. Even when r_t = 5, the gradient still says "push higher!" — way outside the trust region.
- **With the clip:** consider what `min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t)` does.
    - When r_t ≤ 1+ε: the unclipped term r_t · A_t is the *smaller* of the two (well, they're equal up to 1+ε). The gradient flows normally — we improve the policy.
    - When r_t > 1+ε: the clipped term `(1+ε) · A_t` is *constant in θ* (the clip flattens it). It is also *smaller* than r_t · A_t (since A_t > 0 and r_t > 1+ε). The `min` selects the clipped term. **The gradient with respect to θ is zero.**

In plain English: *"You've already increased this action's probability by 20%. That's enough. Don't push further on this single transition — the data isn't trustworthy that far out."*

### Case 2: A_t < 0 (the action was worse than average)

We want to *decrease* π_θ(a_t|s_t). As we update, r_t falls below 1.

- **Without the clip:** L = r_t · A_t with A_t negative. Lower r_t → more negative product → "better" loss (we're maximizing). Gradient pushes r_t toward 0. Catastrophic — we can effectively delete an action from the policy in one update.
- **With the clip:**
    - When r_t ≥ 1−ε: gradient flows, we decrease the action's probability normally.
    - When r_t < 1−ε: the clipped term `(1−ε) · A_t` is constant in θ. With A_t < 0, the *unclipped* r_t · A_t is *more negative* (since r_t < 1−ε and we multiply by a negative number, smaller r_t gives larger negative value)... wait, let me redo that.

    Be careful with the sign. A_t < 0, r_t < 1−ε. Then r_t · A_t > (1−ε) · A_t (multiplying a smaller positive r by a negative gives a *larger* — less negative — result). So the unclipped term is *larger*, and the `min` selects the *clipped* term `(1−ε) · A_t`. That's constant in θ. **Gradient zero again.**

In plain English: *"You've already decreased this action's probability by 20%. Stop. Any further decrease on the basis of this one transition would be reckless."*

### The ASCII picture

```
Advantage A_t > 0  (good action — we want r_t to grow)

  L_CLIP
    │
    │
    │                ─────────────  (clipped: gradient = 0)
    │              /
    │            /
    │          /  (linear region: gradient ∝ A_t > 0)
    │        /
    │      /
    │    /
    │  /
    │/_______________________________  r_t
    0     1-ε    1     1+ε

Advantage A_t < 0  (bad action — we want r_t to shrink)

  L_CLIP
    │
  ──┼────                            (clipped: gradient = 0)
    │    \
    │      \
    │        \   (linear region: gradient ∝ A_t < 0)
    │          \
    │            \
    │              \
    │________________\______________  r_t
    0     1-ε    1     1+ε
```

Two flat regions, one sloped region in the middle. The flat regions are where the clip "kicks in" and the gradient turns off — the trust region boundary. Inside [1−ε, 1+ε], PPO behaves like vanilla policy gradient with importance sampling.

### Why `min` and not just `clip`?

A subtle point that catches everyone: the `min` is there to make the bound **pessimistic**. We want the objective to be *less* attractive whenever clipping is active, never *more* attractive. Without `min`, in Case 1 the clipped term `(1+ε) · A_t` could be larger than `r_t · A_t` if r_t < 1+ε — and the agent could exploit the clip to *avoid* updates. The `min` ensures we always take the worse of the two and never get free lunches.

It's the same reason TRPO uses an upper bound on KL: optimize the pessimistic surrogate, get a guaranteed-non-worsening update.

### The clip range ε

`clip_range` is the most important PPO hyperparameter. It directly sets the trust region size.

- **ε = 0.1** (tight): very conservative updates, slow but stable. Use when training is fragile.
- **ε = 0.2** (default): the sweet spot identified in the original paper.
- **ε = 0.3+** (loose): aggressive updates, faster initially but unstable. Often hurts.
- **ε = 0.0**: identical to A2C with importance sampling — no trust region at all (but then `n_epochs` must be 1 or it diverges).

---

## 5 · Generalized Advantage Estimation (GAE)

We've been writing A_t without defining it. Where do advantages come from?

Recall from Unit Actor-Critic:

$$
A_t = G_t - V(s_t)
$$

The advantage is the *return* minus the *baseline*. But what return? You have a choice with a spectrum.

### The bias-variance tradeoff in advantage estimation

| Estimator | Formula | Bias | Variance |
|-----------|---------|------|----------|
| 1-step TD | $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ | High (depends on V being right) | Low (only one random reward) |
| 2-step TD | $r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t)$ | Medium | Medium |
| Full Monte Carlo | $G_t - V(s_t) = (\sum_{k=0}^\infty \gamma^k r_{t+k}) - V(s_t)$ | Zero (unbiased) | High (sum of many random rewards) |

There's no free lunch. Short-horizon bootstrapping is biased (we trust V too much). Long-horizon Monte Carlo is unbiased but noisy (one lucky episode can dominate). REINFORCE used full MC and suffered from variance. A2C uses 1-step TD and suffers from bias.

### GAE — the elegant compromise

**Generalized Advantage Estimation** (Schulman et al. 2015) interpolates between all of these using a single parameter **λ**:

$$
A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{k=0}^{\infty} (\gamma \lambda)^k \, \delta_{t+k}
$$

where $\delta_{t+k} = r_{t+k} + \gamma V(s_{t+k+1}) - V(s_{t+k})$ is the 1-step TD error at step t+k.

**Knobs:**

- **λ = 0**: $A_t^{\text{GAE}} = \delta_t$ — pure 1-step TD. Maximum bias, minimum variance.
- **λ = 1**: $A_t^{\text{GAE}} = \sum_k \gamma^k \delta_{t+k}$ which (telescoping) equals $G_t - V(s_t)$ — full Monte Carlo. Zero bias, maximum variance.
- **λ = 0.95**: typical default. About 95% Monte Carlo with a small bias from bootstrapping. Best of both worlds in practice.

You can think of λ as *"how much do I trust my critic V?"* — if V is accurate, set λ small and lean on bootstrapping; if V is junk (early training), set λ closer to 1 and trust the real returns.

### Recursive computation

The brilliant part: GAE can be computed in **one backwards pass** through the rollout, with a single accumulator:

```python
# Pseudo-code
advantages = zeros(T)
gae = 0
for t in reversed(range(T)):
    delta = rewards[t] + gamma * values[t+1] * not_done[t] - values[t]
    gae = delta + gamma * lam * not_done[t] * gae
    advantages[t] = gae
returns = advantages + values   # used as targets for the value function
```

This is exactly what CleanRL's `ppo.py` does in about 8 lines. The `not_done` mask zeros out the bootstrap across episode boundaries.

**This rollout, this advantage tensor, is what gets reused across all n_epochs of update.** That's the key: A_t is computed *once*, frozen, and the clip prevents the policy from drifting so far that A_t becomes meaningless.

---

## 6 · The full PPO loss

PPO trains the actor *and* the critic *and* maintains entropy, all in one combined loss:

$$
L^{\text{PPO}}(\theta) = L^{\text{CLIP}}(\theta) - c_1 \cdot L^{\text{VF}}(\theta) + c_2 \cdot L^{\text{ENT}}(\theta)
$$

(Signs depend on whether you're minimizing or maximizing. SB3 minimizes, so internally it's `-L_CLIP + c1·L_VF - c2·L_ENT`. The math is the same.)

| Term | Formula | Role | Typical coefficient |
|------|---------|------|---------------------|
| $L^{\text{CLIP}}$ | the clipped surrogate from §4 | **maximize** — improve the policy | 1.0 (implicit) |
| $L^{\text{VF}}$ | $\tfrac{1}{2} \mathbb{E}_t [(V_\theta(s_t) - V_t^{\text{target}})^2]$ | **minimize** — train the critic toward the GAE returns | $c_1 = 0.5$ |
| $L^{\text{ENT}}$ | $H(\pi_\theta(\cdot \mid s_t)) = -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)$ | **maximize** — keep exploration alive | $c_2 = 0.01$ |

The value target $V_t^{\text{target}}$ is just `advantages[t] + values[t]` from the GAE pass — i.e. the GAE-corrected return.

One backward pass on this combined loss updates the shared backbone (if any), the policy head, and the value head simultaneously. Compare to DQN, which only trains a value head; compare to REINFORCE, which has no critic at all.

---

## 7 · The PPO training loop (multiple epochs)

Here is the heart of PPO, in pseudocode:

```
initialize θ
loop forever:
    # ----- ROLLOUT PHASE -----
    θ_old ← θ                                # freeze a copy of the policy
    collect n_steps × n_envs transitions using π_{θ_old}
    for each transition store: (s_t, a_t, r_t, done_t, log π_{θ_old}(a_t|s_t), V_{θ_old}(s_t))
    compute advantages A_t with GAE  → freeze
    compute value targets V_target_t = A_t + V_{θ_old}(s_t) → freeze

    # ----- UPDATE PHASE -----
    for epoch in 1..n_epochs:                # typically 10
        shuffle the rollout indices
        for each mini-batch of size batch_size:
            log_π_new ← log π_θ(a_t | s_t)              # CURRENT θ
            r_t ← exp(log_π_new − log π_{θ_old}(a_t|s_t))
            L_CLIP ← min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t).mean()
            L_VF   ← ((V_θ(s_t) − V_target_t)²).mean() · 0.5
            L_ENT  ← entropy(π_θ(·|s_t)).mean()
            L      ← −L_CLIP + c_1 · L_VF − c_2 · L_ENT
            backward(L); clip_grad_norm_(0.5); optimizer.step()
```

Three subtleties that took the field years to settle on:

1. **`log π_{θ_old}` is stored, not recomputed.** When we collect the rollout, we save the log-prob of each action under the policy at that moment. This is now a constant during update. The ratio is `exp(new_log_prob - old_log_prob)` — numerically stable and trivial.
2. **Advantages are normalized per mini-batch.** Most implementations (CleanRL, SB3) do `A = (A - A.mean()) / (A.std() + 1e-8)` before computing L_CLIP. This stabilizes the scale of gradients across batches with very different reward magnitudes.
3. **Gradients are clipped by global norm.** `clip_grad_norm_(0.5)` prevents the rare exploding gradient from blowing up the model. Cheap insurance.

### Why this works (the punchline)

> A_t is fixed. Only r_t depends on the trainable θ. The clip stops r_t from drifting too far from 1, so the policy stays within a region where A_t is still a valid estimate of action quality. Within that region, we can afford to take many gradient steps.

That is the entire idea of PPO. Everything else is engineering.

---

## 8 · Hyperparameter dictionary

Every PPO hyperparameter, what it controls, and what changes when you turn the knob.

| Parameter | Symbol | Typical | Mathematical role |
|-----------|--------|---------|-------------------|
| `n_steps` | T | 64 – 2048 | Rollout length per environment |
| `n_epochs` | K | 10 | Passes over each rollout |
| `batch_size` | M | 64 – 256 | Mini-batch size; must divide `n_steps × n_envs` |
| `learning_rate` | α | 3e-4 | Adam step size |
| `clip_range` | ε | 0.2 | Trust region half-width |
| `gae_lambda` | λ | 0.95 | GAE interpolation (0=TD, 1=MC) |
| `gamma` | γ | 0.99 | Discount factor |
| `vf_coef` | c_1 | 0.5 | Critic loss weight |
| `ent_coef` | c_2 | 0.0 – 0.01 | Entropy bonus weight |
| `max_grad_norm` | — | 0.5 | Global gradient norm cap |
| `n_envs` | N | 4 – 16 | Parallel environments |

### What happens when you turn each knob

**`n_steps` ↑** : better Monte Carlo signal in GAE (more terms in the sum), more memory, slower wall-clock per update. **↓** : noisier advantages but more frequent policy updates. Rule of thumb: longer for sparse-reward tasks, shorter for dense-reward tasks.

**`n_epochs` ↑** : more reuse, more sample-efficient, more risk of policy drifting outside trust region (watch `approx_kl`). **↓** : closer to A2C, safer but wasteful. 10 is the magic number from the paper; values 3–20 are reasonable.

**`batch_size` ↑** : less noisy gradients per step, fewer gradient steps per epoch. **↓** : more updates, more noise. Constraint: `n_steps × n_envs` must be divisible by `batch_size`.

**`learning_rate` ↑** : faster initial learning, higher risk of policy collapse. **↓** : safer, slower. Many practitioners linearly anneal α toward 0 over training.

**`clip_range` ↑** : larger trust region, faster updates, risk of instability. **↓** : tighter trust region, slower but safer. Some implementations also anneal ε downward over training.

**`gae_lambda` ↑** : closer to Monte Carlo, lower bias, higher variance. **↓** : closer to 1-step TD, higher bias, lower variance. 0.95 is robust.

**`gamma` ↑** (e.g. 0.999): agent cares about far-future reward; harder to learn but better for long-horizon tasks. **↓** (e.g. 0.9): agent is myopic; faster learning on short-horizon tasks. Equivalent to choosing your effective planning horizon ≈ 1/(1−γ).

**`vf_coef` ↑** : prioritize critic accuracy at the cost of policy improvement. Useful if value loss is huge and dominating gradient signal. **↓** : let the policy lead; critic catches up slower.

**`ent_coef` ↑** : more exploration; policy stays high-entropy longer. Use when the agent collapses to a degenerate policy too early. **↓** : faster convergence to deterministic behavior. Setting to 0 is fine for tasks with adequate exploration from environmental randomness.

**`max_grad_norm` ↑** : less aggressive clipping. **↓** : more aggressive. 0.5 is a safe default; rarely needs tuning.

**`n_envs` ↑** : less correlated samples, more wall-clock parallelism, more memory. **↓** : more correlated rollouts, GAE estimates more biased toward single-trajectory dynamics. In Godot RL, this is the `n_parallel` flag and is set by the env wrapper, not the algo.

---

## 9 · `approx_kl` as a diagnostic

You will spend more time staring at `train/approx_kl` in TensorBoard than at any other PPO metric. It is your single best signal for "is my PPO healthy?"

The KL divergence between the old and new policy isn't free to compute exactly (you'd have to sum over all actions). PPO uses a cheap one-sample estimator:

$$
\widehat{\text{KL}}_t = (r_t - 1) - \log r_t
$$

You can also see the (older, simpler) form `−log r_t` — both approximate the same thing. Schulman's "John's blog post on KL" recommends the first form because it is always ≥ 0 and has lower variance.

**How to read it:**

| `approx_kl` range | What it means | Action |
|-------------------|---------------|--------|
| 0.005 – 0.02 | Healthy. Policy is moving but not too fast. | Do nothing. |
| < 0.001 | Policy barely moving. Maybe learning is done — or stuck. | Raise `learning_rate`, raise `clip_range`, or check if reward is still improving. |
| 0.03 – 0.05 | Borderline. Watch reward curve for instability. | Consider lowering `learning_rate` or `clip_range` slightly. |
| > 0.05 | Policy moving very fast. Likely heading for collapse. | Lower `learning_rate`, lower `clip_range`, lower `n_epochs`. |

!!! warning "When `approx_kl` explodes"
    Some implementations (SB3, CleanRL with `--target-kl`) implement **early stopping** of the update epochs: if `approx_kl` exceeds a target (e.g. 0.015), the inner update loop breaks immediately. This is a safety net for when one mini-batch happens to push the policy too far. If you see `train/n_updates` consistently below `n_epochs`, that's why.

Related metrics to track:

- `rollout/ep_rew_mean` — the actual learning signal. All else is in service of this going up.
- `train/entropy_loss` — should *decrease in magnitude slowly*. Sudden collapse = exploration failure.
- `train/explained_variance` — how well the critic predicts returns. Should creep toward 1.0; values below 0 mean the critic is worse than predicting the mean.
- `train/clip_fraction` — fraction of samples where the clip activated. 0.1–0.3 is typical. Near 0 = clip never engaging (try larger LR). Near 1 = clip always engaging (LR or clip too aggressive).

---

## 10 · CleanRL reference implementation

!!! tip "Read `ppo.py` while reading this section"
    Open `https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py` in another tab. The entire algorithm is ~300 lines, single file, no abstractions. Every PPO concept in this unit is right there in plain PyTorch.

CleanRL's `ppo.py` is the *definitive* pedagogical PPO implementation. Stable-Baselines3's PPO is more featureful but spread across a class hierarchy; CleanRL keeps everything readable.

### Reading guide

| Section in `ppo.py` | What to look for | Maps to this unit |
|---------------------|------------------|-------------------|
| Lines ~80–100 (the `Agent` class) | Shared MLP backbone, separate actor and critic heads | Actor-Critic recap |
| Lines ~100–140 (rollout storage tensors `obs`, `actions`, `logprobs`, `rewards`, `dones`, `values`) | The rollout buffer | §7 — what gets frozen |
| Lines ~140–170 (the env step loop) | Stepping environments, storing transitions | §1 — collecting on-policy data |
| Lines ~170–195 (the GAE backward pass) | The exact recursion from §5 | §5 — GAE |
| Lines ~195–250 (the update phase: `for epoch in range(args.update_epochs)`) | The inner loops — epochs, then mini-batches | §7 |
| Inside the mini-batch loop: `ratio = (newlogprob - mb_logprobs).exp()` | This is r_t | §3 |
| `pg_loss1 = -mb_advantages * ratio`<br>`pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-clip, 1+clip)`<br>`pg_loss = torch.max(pg_loss1, pg_loss2).mean()` | **The clip!** (Note: `max` because they negated, equivalent to our `min` of positive form) | §4 |
| `v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()` | Critic loss | §6 |
| `entropy_loss = entropy.mean()` | Entropy bonus | §6 |
| `loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef` | The combined PPO loss | §6 |
| `approx_kl = ((ratio - 1) - logratio).mean()` | The KL diagnostic | §9 |

After this unit, you should be able to *read* `ppo.py` top-to-bottom without confusion. Take 30 minutes and do it.

---

## 11 · PPO in Godot RL

Godot RL Agents wraps your Godot environment in a Gymnasium-compatible interface (`StableBaselinesGodotEnv`) and hands it to Stable-Baselines3's PPO implementation. Everything you learned here applies directly.

A typical training command:

```bash
gdrl --env_path=builds/MyEnv.x86_64 \
     --n_steps=512 \
     --batch_size=256 \
     --n_epochs=10 \
     --gamma=0.99 \
     --gae_lambda=0.95 \
     --clip_range=0.2 \
     --ent_coef=0.005 \
     --learning_rate=3e-4 \
     --total_timesteps=1000000
```

Every flag is a knob from §8. Reading it left to right:

- `--n_steps=512` — collect 512 steps per environment before each update.
- `--batch_size=256` — split each rollout into mini-batches of 256.
- `--n_epochs=10` — 10 passes over each rollout (the PPO data-reuse magic).
- `--gamma=0.99` — discount factor, ~100-step effective horizon.
- `--gae_lambda=0.95` — slight bootstrapping bias, lots of variance reduction.
- `--clip_range=0.2` — standard trust region.
- `--ent_coef=0.005` — slight exploration bonus (a bit below the SB3 default of 0; Godot environments tend to need a touch of encouragement).
- `--learning_rate=3e-4` — Adam default.

### Tuning workflow when training looks bad

The TensorBoard-driven debugging loop, in order:

1. **`rollout/ep_rew_mean` flat and `train/approx_kl` very low** → policy isn't moving. Raise `learning_rate` 3×, or raise `clip_range` to 0.3.
2. **`approx_kl` spiking above 0.05, reward unstable** → policy moving too fast. Lower `clip_range` to 0.1 or `learning_rate` to 1e-4.
3. **`train/entropy_loss` collapses to ~0 early** → policy went deterministic before exploring enough. Raise `ent_coef` to 0.02.
4. **`train/explained_variance` stays near 0** → critic isn't learning. Raise `vf_coef` to 1.0 or extend `n_steps`.
5. **`train/clip_fraction` near 0** → clip never engages; you're effectively doing A2C with many epochs. Possibly `clip_range` too loose for the LR.

### The big picture

The `StableBaselinesGodotEnv` wrapper is just a Gymnasium environment. **PPO has no idea Godot is on the other side.** It sees observations and rewards arriving, actions going out, just like CartPole. Everything you learned about PPO in this unit applies identically whether the env is CartPole, Atari, MuJoCo, or your Godot platformer.

---

## 12 · PPO vs alternatives summary

| Method | Sample efficiency | Stability | Continuous actions | Discrete actions | Memory |
|--------|-------------------|-----------|--------------------|------------------|--------|
| REINFORCE (Unit 5) | Low | Low | Yes | Yes | Low |
| A2C (Unit Actor-Critic) | Medium | Medium | Yes | Yes | Low |
| **PPO (this unit)** | **High** | **High** | **Yes** | **Yes** | **Medium** |
| DQN (Unit DQN) | High | High | No | Yes (only) | High (replay buffer) |
| SAC | Very high | High | Yes (only) | Awkward | High (replay buffer) |
| TRPO | High | Very high | Yes | Yes | High (Hessian) |

PPO's winning combination: **good sample efficiency, good stability, works on both action types, simple first-order implementation, no replay buffer needed**. No other algorithm hits all five. That's why it became the default.

---

## What's next

You now understand every line of PPO. The next unit, **PPO in Practice**, drops the equations and takes you back to the keyboard: training a real Godot agent with PPO, watching the metrics from §9 evolve in TensorBoard, and developing intuition for which knobs to turn when.

After that, the course moves from theory to engineering: curriculum design, reward shaping, sim-to-real, and deployment. PPO will be in the background of every remaining unit.

---

## Stretch goals

If you want to go deeper, in increasing order of effort:

1. **Read the original PPO paper.** Schulman et al. 2017, "Proximal Policy Optimization Algorithms" — 8 pages, very readable. Every equation in the paper maps to a section of this unit. Notice that the paper proposes *two* variants: the clipped objective (what we covered, used everywhere) and an adaptive KL penalty (mostly forgotten).
2. **Implement PPO from scratch.** Fork CleanRL's `ppo.py`, train on CartPole-v1 (should solve in <1 minute) and LunarLander-v2 (~10 minutes on CPU). Do *not* use SB3; the point is to type every line yourself.
3. **Run a clip-range ablation.** Train the same Godot environment three times with `--clip_range=0.1`, `--clip_range=0.2`, `--clip_range=0.4`. Plot `ep_rew_mean`, `approx_kl`, and `clip_fraction` for all three on the same TensorBoard. Write a one-paragraph explanation of what you see, citing §4 and §9.
4. **Reproduce the GAE λ-sweep figure** (Figure 1 of the GAE paper, Schulman et al. 2015). Train PPO with λ ∈ {0.0, 0.5, 0.9, 0.95, 0.99, 1.0} and observe how variance vs bias plays out.
5. **Try `target_kl`-based early stopping.** Add a check that breaks the inner update loop if `approx_kl > 0.015`. Compare training stability with and without.

If you finish (2), you officially understand PPO better than 95% of people who use it. That is the real goal of this unit.

---

[← Actor-Critic](unit-actor-critic.md) · [Course home](index.md) · [→ PPO in Practice](unit-04.md)
