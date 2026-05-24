# Policy Gradients — REINFORCE & the Policy Gradient Theorem

[← Deep Q-Learning](unit-03.md) · [Course home](index.md)

---

!!! info "Three ways to see your AI"
    Python training output (reward per episode) · matplotlib training curve · CartPole rendering (`gym.make(..., render_mode="human")`)

So far you have trained agents that learn **values** — Q-Learning estimates Q(s,a) in a table, DQN approximates it with a neural net. In both cases the policy is *derived*: you pick `argmax_a Q(s,a)`. This unit takes the opposite path. Instead of learning values and squeezing a policy out of them, we will **parametrize the policy itself** and optimize it directly with gradient ascent.

This is the foundation that PPO, A2C, SAC, and almost every modern deep RL algorithm is built on. Once you understand REINFORCE, the rest are stability tricks layered on top.

---

## 1 · Value-based vs policy-based: the fundamental choice

You already know **value-based RL**:

- DQN learns `Q(s,a)` — the expected return of taking action `a` in state `s`.
- The policy is implicit: `π(s) = argmax_a Q(s,a)`.
- Exploration is bolted on (ε-greedy).

**Policy-based RL** flips the picture:

- You learn `π(a|s)` directly — a parametrized function (usually a neural net) that maps states to a *probability distribution* over actions.
- No `argmax`. You sample actions from the distribution.
- Exploration is built in: the policy is naturally stochastic until it sharpens.

### When policy-based wins

- **Continuous action spaces.** If `a ∈ ℝ` (e.g. a steering angle in [-1, 1]), `argmax_a Q(s,a)` requires solving an optimization problem *inside every step*. Policy networks just output a mean and std — sample, done.
- **Stochastic policies are required.** Consider rock-paper-scissors. Any deterministic policy is exploitable: if you always play rock, your opponent always plays paper. The Nash-optimal policy is uniformly random. A Q-table cannot represent this; a policy network can.
- **High-dimensional action spaces.** With 50 joints on a humanoid, enumerating actions is impossible. A policy network outputs 50 means and 50 stds — straightforward.

### When value-based wins

- **Discrete actions with a small action set.** DQN is often more sample-efficient here.
- **Deterministic environments with one optimal action per state.** A greedy value method nails this fast.
- **Off-policy learning from a replay buffer.** Pure policy gradient is on-policy — you must use fresh samples from the *current* policy.

| | Value-based (DQN) | Policy-based (REINFORCE) |
|--|--|--|
| Learns | `Q(s,a)` | `π(a|s)` |
| Policy | argmax (deterministic) | Sample (stochastic) |
| Action space | Discrete | Discrete **or continuous** |
| Exploration | ε-greedy add-on | Built into the policy |
| Data efficiency | Replay buffer (off-policy) | Fresh samples (on-policy) |

---

## 2 · Parametrizing the policy

A policy is a function `π_θ(a|s)` with parameters `θ` (the neural net weights). It takes a state and returns a probability distribution.

### Discrete actions — softmax head

For `N` discrete actions, the network outputs `N` logits. A softmax converts them to probabilities:

```
π_θ(a_i | s) = exp(z_i) / Σ_j exp(z_j)
```

In plain English: the logit `z_i` is the "score" for action `i`; softmax turns scores into a proper probability distribution that sums to 1.

### Continuous actions — Gaussian head

For a continuous action `a ∈ ℝ^d`, the network outputs a mean `μ_θ(s)` and (often) a log standard deviation `log σ_θ`. You sample from a Gaussian:

```
a ~ N(μ_θ(s), σ_θ²)
```

In plain English: the network predicts roughly where the good action is (`μ`), and how confident it is (`σ`). Smaller `σ` = sharper, more deterministic.

### The objective

We want to maximize the **expected discounted return** under our policy:

```
J(θ) = E_{τ ~ π_θ} [ Σ_t γ^t r_t ]
```

Term by term:
- `τ` is a trajectory `(s_0, a_0, r_0, s_1, a_1, r_1, …)` sampled by running the policy.
- `γ ∈ [0, 1)` is the discount factor — future rewards are worth less.
- `r_t` is the reward at step `t`.
- The expectation is over the randomness in the policy *and* the environment.

We want `θ* = argmax_θ J(θ)`. With a differentiable `J`, we would just do gradient ascent: `θ ← θ + α ∇_θ J(θ)`. The problem — addressed in the next section — is that `J` involves the environment, which is **not** differentiable.

---

## 3 · The policy gradient theorem

### The problem

`J(θ)` depends on `θ` through two channels:

1. The action probabilities `π_θ(a|s)` — differentiable.
2. The trajectory of states the environment produces in response — **not** differentiable. You cannot backpropagate through `env.step()`.

So how do we compute `∇_θ J(θ)`?

### The trick — log-derivative (score function)

A small identity from calculus:

```
∇_θ π_θ(a|s) = π_θ(a|s) · ∇_θ log π_θ(a|s)
```

This follows from `∇ log f = ∇f / f`. Multiply both sides by `f` and you get the line above. It looks innocent but it is the key to the whole field.

Applying this inside the expectation and doing some algebra (skipped — see Sutton & Barto Ch. 13), you arrive at the **policy gradient theorem**:

```
∇_θ J(θ) = E_{τ ~ π_θ} [ Σ_t ∇_θ log π_θ(a_t | s_t) · G_t ]
```

Term by term:
- `∇_θ log π_θ(a_t | s_t)` — how to nudge `θ` to make the action `a_t` more likely. PyTorch computes this for you via autograd on `log_prob`.
- `G_t = Σ_{k≥t} γ^{k-t} r_k` — the **return from step t onwards** (also called *returns-to-go*).
- The expectation `E_{τ ~ π_θ}` — we estimate it by sampling episodes.

### In plain English

> Increase the probability of actions that led to high returns; decrease the probability of actions that led to low returns. The size of the nudge is proportional to the return.

Critically, the environment's non-differentiability is gone. We never differentiate through `env.step()` — we only differentiate `log π_θ`, which is just a neural net forward pass. The return `G_t` is a **scalar weight**, treated as a constant during backprop.

### Why this is estimable

We can approximate the expectation with a single (or a few) sampled trajectories:

```
∇_θ J(θ) ≈ Σ_t ∇_θ log π_θ(a_t | s_t) · G_t
```

Run an episode, collect `(s_t, a_t, G_t)` for every step, sum up the gradients, take a step. That is REINFORCE.

---

## 4 · REINFORCE algorithm

Williams, 1992. The simplest possible policy gradient method.

### Algorithm

1. Initialize policy network `π_θ` randomly.
2. **Roll out** one complete episode using `π_θ`: collect `(s_0, a_0, r_0), …, (s_T, a_T, r_T)`.
3. **Compute returns-to-go**: `G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + … + γ^(T-t)·r_T` for every `t`.
4. **Compute the loss**:
   ```
   L(θ) = - Σ_t G_t · log π_θ(a_t | s_t)
   ```
   The minus sign is because PyTorch *minimizes* losses, but we want to *maximize* `J`.
5. Backpropagate and take an optimizer step.
6. Repeat from step 2 until convergence.

### Complete PyTorch implementation on CartPole-v1

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

env = gym.make("CartPole-v1")
policy = PolicyNetwork(4, 2)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
gamma = 0.99

def compute_returns(rewards, gamma):
    G, returns = 0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)

for episode in range(1000):
    obs, _ = env.reset()
    log_probs, rewards = [], []
    done = False

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        probs = policy(obs_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))

        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated

    returns = compute_returns(rewards, gamma)

    # Normalize returns (variance reduction)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Policy gradient loss (negative because we do gradient ASCENT)
    loss = -torch.stack(log_probs) @ returns

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 100 == 0:
        print(f"Episode {episode}: total reward = {sum(rewards):.0f}")
```

### What to expect

Run it. You will see something like:

```
Episode 0:   total reward = 23
Episode 100: total reward = 47
Episode 200: total reward = 89
Episode 400: total reward = 156
Episode 700: total reward = 200
```

CartPole's max reward is 200 (or 500 on the v1 episode-length cap). REINFORCE typically reaches the cap in **500–800 episodes**. But the curve is **noisy** — some episodes will randomly drop back to 30. That brings us to the central weakness of REINFORCE.

---

## 5 · The variance problem

!!! warning "REINFORCE is unbiased but has very high variance"
    The gradient estimate is correct *in expectation*, but any single sample can be wildly off. This is the single biggest issue with vanilla policy gradients.

Why is variance so high?

- A single trajectory is a **single sample** of a high-dimensional random process.
- Two episodes from the same policy can have returns that differ by 10x due purely to chance — random environment dynamics, lucky/unlucky exploration.
- The gradient is `∇log π · G`. If `G` swings wildly between episodes, so do the gradient updates.

Practical consequences:

- Training curves oscillate hard. You will see runs hit 200, then crash back to 50.
- Sample efficiency is poor — you need thousands of episodes for problems DQN solves in hundreds.
- On harder tasks (long episodes, sparse rewards), vanilla REINFORCE can fail to learn anything.

A common cheap fix is in the code above: **normalize returns** to zero mean and unit variance across the episode. This is a hack — not theoretically justified — but it helps a lot in practice. The principled fix is the next section.

### Why score function variance is high

The gradient is `∇log π(a|s) · G_t`. The return `G_t` has high variance — different episodes take wildly different paths through the environment, and the sum of rewards can swing by 10× between runs. Multiplying by `G_t` amplifies that variance into the gradient. Every update is a noisy estimate of the true gradient direction, and the noise can easily overwhelm the signal.

**Why the reparameterization trick achieves lower variance:**

SAC's actor (see [SAC unit](unit-sac.md)) faces the same problem — it needs gradients through sampled actions. Instead of the score function estimator, it uses reparameterization: write the sample as a *deterministic* function of the policy parameters and an independent noise variable:

```
z ~ N(0, 1)               ← sampled independently (z, not ε, to avoid confusion with ε-greedy)
a = μ(s) + σ(s) · z       ← deterministic function of policy parameters
```

Now `∂a/∂θ = ∂μ/∂θ + z · ∂σ/∂θ` — the gradient flows directly through `μ` and `σ`, bypassing the sampling step. This gives much lower variance because the gradient estimate doesn't depend on the return magnitude.

**Which estimator each algorithm uses:**

| Estimator | Used in | Variance |
|-----------|---------|----------|
| Score function (REINFORCE) | REINFORCE, A2C, PPO | High (needs baseline) |
| Reparameterization | SAC actor | Low |
| Neither | DQN (no policy gradient) | N/A |

**Why PPO uses score function, not reparameterization:** PPO's clipped ratio objective requires differentiating through `log π(a|s)`, not through the action sample itself. The importance-sampling ratio `π_new/π_old` works with already-collected actions — reparameterization would require re-sampling fresh actions at each gradient step and loses the reuse benefit of the replay-free rollout.

---

## 6 · Baseline variance reduction

!!! tip "Subtracting a baseline keeps the gradient unbiased"
    Here is the magic identity:
    ```
    E_π [ ∇ log π_θ(a|s) · b(s) ] = 0
    ```
    for *any* function `b(s)` that does not depend on `a`. Subtract it freely.

So we can rewrite the policy gradient as:

```
∇_θ J(θ) = E_π [ ∇_θ log π_θ(a_t|s_t) · (G_t − b(s_t)) ]
```

Term by term:
- `G_t − b(s_t)` — the return *minus a baseline*. Same gradient in expectation, but typically much smaller variance.
- `b(s_t)` — anything that depends only on the state, not the action.

### The optimal baseline

The variance-minimizing baseline is the **state-value function** `V(s) = E_π[G_t | s_t = s]`: the expected return from state `s` under the current policy.

With this baseline:

```
A(s_t, a_t) = G_t − V(s_t)
```

is called the **advantage**. In plain English:

> The advantage tells you how much better (or worse) the action you took was, compared to the average action you would normally take in this state.

The modified loss becomes:

```
L(θ) = − Σ_t A_t · log π_θ(a_t | s_t)
```

### Adding a baseline to the CartPole code

You can fit `V_φ(s)` as a second network and train it to predict the observed returns (MSE loss). The policy network and value network share nothing structurally:

```python
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

value_net = ValueNetwork(4)
value_optim = optim.Adam(value_net.parameters(), lr=1e-3)

# Inside the training loop, after collecting log_probs, rewards, obs_buffer:
obs_tensor = torch.tensor(np.array(obs_buffer), dtype=torch.float32)
returns = compute_returns(rewards, gamma)

# Critic update: fit V(s) to observed returns
values = value_net(obs_tensor)
value_loss = ((returns - values) ** 2).mean()
value_optim.zero_grad()
value_loss.backward()
value_optim.step()

# Advantage = return - baseline (detach so policy grad doesn't flow into critic)
advantages = (returns - value_net(obs_tensor).detach())
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

policy_loss = -torch.stack(log_probs) @ advantages
optimizer.zero_grad()
policy_loss.backward()
optimizer.step()
```

Run this side-by-side with vanilla REINFORCE. The baseline version will reach 200 reward faster and oscillate noticeably less. You have just built your first **actor-critic** — the policy is the actor, `V_φ` is the critic.

---

## 7 · The credit assignment problem

In a 500-step CartPole episode that ended in failure, **which step caused the failure**? The actions in the first 50 steps were probably fine; the bad one was around step 480.

Vanilla "total episode return for every step" credits all 500 actions equally for the total reward. That is bad. We want to credit each action only for the rewards that came *after* it.

This is exactly why we compute `G_t` as the **return from step t onwards** — not the total episode return. The earlier code already does this:

```python
def compute_returns(rewards, gamma):
    G, returns = 0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)
```

Reading backwards: `G_T = r_T`, `G_{T-1} = r_{T-1} + γ·r_T`, and so on. Action `a_t` is weighted by what happened *from t onwards*, not before.

This is mathematically justified: rewards received *before* an action cannot have been caused by that action, so they contribute zero gradient in expectation. Dropping them reduces variance without introducing bias.

The discount factor `γ` also helps: rewards far in the future contribute less, so the gradient focuses on near-term consequences of each action.

---

## 8 · From REINFORCE to Actor-Critic

REINFORCE works, but it has two structural limitations:

1. **It needs complete episodes.** Returns `G_t` are computed by summing rewards to the end. You can't update mid-episode.
2. **Variance is still high** even with a baseline, because `G_t` is itself a noisy Monte Carlo sample.

The fix is to replace the Monte Carlo return with a **bootstrapped estimate** from a learned value function, just like DQN's TD targets:

```
G_t ≈ r_t + γ · V_φ(s_{t+1})
```

Now the advantage is:

```
A_t = r_t + γ · V_φ(s_{t+1}) − V_φ(s_t)
```

This is the **one-step TD advantage**, and you can compute it after every single step. No need to wait for the episode to end.

Architecturally:

- **Actor** = the policy `π_θ`. Outputs actions.
- **Critic** = the value function `V_φ`. Outputs scalar value estimates.
- Actor is trained with the policy gradient using advantages from the critic.
- Critic is trained with TD regression (just like DQN, but predicting `V` not `Q`).

> Actor-Critic combines the **stability of value methods** (low-variance TD targets) with the **generality of policy methods** (works for continuous actions, learns stochastic policies).

This is the family A2C, A3C, PPO, SAC all belong to. The next unit develops it fully.

---

## 9 · Connection to PPO

You have been using PPO in earlier Godot units (Unit 1, Unit 2). It is worth pausing to see how PPO relates to what you just learned:

PPO **is** a policy gradient method. The objective at its core is still:

```
∇ log π_θ(a|s) · A(s,a)
```

What PPO adds on top:

1. **A clipped probability ratio.** Instead of `log π_θ`, PPO uses the ratio `π_θ(a|s) / π_old(a|s)` and clips it to `[1−ε, 1+ε]`. This prevents the policy from changing too much in one update, which is a major source of instability in vanilla policy gradient.
2. **Generalized Advantage Estimation (GAE).** Instead of one-step TD or full Monte Carlo, GAE interpolates between them, controlled by a parameter `λ`. It is a knob between high-bias-low-variance (TD) and low-bias-high-variance (Monte Carlo).
3. **Multiple epochs per batch.** PPO reuses each batch of data for several optimizer steps, which is only safe because of the clipping.
4. **Minibatching.** Large batches are split into minibatches for SGD.

> Mental model: **PPO is REINFORCE with a critic, a clipped update, GAE advantages, and minibatch reuse.** Everything you learned in this unit carries forward unchanged — only the stability machinery is new.

When you read the PPO loss in `stable-baselines3`, you will recognize every term.

---

## 10 · Stretch goals

Try these to deepen your understanding:

### 10.1 PixelCopter — observe variance in the wild

Replace `CartPole-v1` with `PixelCopter-PLE-v0` (from `gym-pygame` or `pygame-learning-environment`). Episodes are longer and rewards are sparser. Vanilla REINFORCE will struggle visibly — you will see exactly why we need actor-critic.

### 10.2 Entropy bonus

Add an exploration bonus to the loss:

```python
entropy = dist.entropy().mean()
loss = policy_loss - 0.01 * entropy
```

Term by term:
- `dist.entropy()` — the Shannon entropy of the current policy. Higher = more uniform = more exploration.
- The `-0.01 * entropy` term pushes the policy *away* from being too sharp too early.

Compare training with and without. With the bonus, the policy holds onto exploration longer and often finds better solutions on harder tasks. This is exactly what PPO uses (`ent_coef` parameter in stable-baselines3).

### 10.3 Plot training stats with matplotlib

Track `ep_rew_mean` and `ep_len_mean` (rolling mean over last 50 episodes) and plot them:

```python
import matplotlib.pyplot as plt
plt.plot(rolling_mean(episode_rewards, window=50))
plt.xlabel("Episode")
plt.ylabel("Mean reward (50-ep window)")
plt.title("REINFORCE on CartPole-v1")
plt.savefig("reinforce_curve.png")
```

Compare three curves on the same axes:
- Vanilla REINFORCE (with return normalization)
- REINFORCE + value baseline
- REINFORCE + value baseline + entropy bonus

You will see, with your own eyes, the variance reduction story unfold.

---

## 11 · REINFORCE and the gdrl Training Loop

Every time you run `gdrl --env_path=... --timesteps=...`, it runs PPO — which IS a policy gradient method. REINFORCE is the conceptual foundation; PPO is REINFORCE with stability tricks layered on top. The loss you minimized in Section 4 and the loss SB3 optimizes are the same equation.

### Mapping REINFORCE to what you see in gdrl

```
REINFORCE concept         →  gdrl / SB3 equivalent
─────────────────────────────────────────────────────
Episode rollout           →  n_steps rollout collection
log π(a|s)                →  policy_gradient_loss in TensorBoard
Return G_t                →  advantage estimate (with GAE)
Policy update             →  n_epochs gradient steps
Exploration via entropy   →  ent_coef parameter
```

### Reading TensorBoard through REINFORCE's lens

When you look at `train/policy_gradient_loss` in TensorBoard, you are watching REINFORCE's loss — `−Σ G_t · log π(a_t|s_t)` — being minimized. The curve is noisy early on (high variance, like REINFORCE) and smooths out as the policy sharpens and GAE provides better advantage estimates.

The `--ent_coef` flag in `gdrl` is the entropy bonus from Section 10.2 of this unit — the same `−0.01 * entropy` term you added to the CartPole loss. SB3's default is `ent_coef=0.0` for PPO but you have likely raised it (e.g. `--ent_coef 0.01`) to prevent early policy collapse.

### The key difference: G_t vs GAE

REINFORCE uses the actual `G_t` — a Monte Carlo return computed from the full episode. PPO uses a **GAE advantage** — a multi-step TD estimate (see PPO Deep Dive unit). GAE is a knob between pure MC (high variance, no bias) and pure TD (low variance, some bias). This is why PPO's `train/policy_gradient_loss` curves are visibly smoother than a vanilla REINFORCE run on the same environment.

### Practical comparison

If you trained CartPole with REINFORCE in this unit, you can compare its TensorBoard loss curves to a `gdrl` PPO run — same loss type, smoother in PPO due to GAE and clipping. The shapes should be recognizably similar: a noisy loss that trends downward as the policy improves, with the PolicyGradientLoss sign flipping when the agent starts consistently getting positive advantages.

---

## What's next

You now understand:

- Why we need policy-based methods (continuous actions, stochastic policies, scale).
- How the policy gradient theorem turns a non-differentiable RL objective into a tractable gradient via the log-derivative trick.
- How REINFORCE estimates that gradient from sampled episodes.
- Why variance is the central practical problem, and how baselines and credit assignment reduce it.
- How adding a learned value function turns REINFORCE into actor-critic — eliminating the need for complete episodes.
- Why PPO is just REINFORCE with stability tricks.

In the next unit, we build a full actor-critic — two networks training in lockstep — and watch the variance collapse compared to vanilla REINFORCE. After that, we layer on the PPO tricks one by one and finally connect everything back to Godot.

[→ Actor-Critic](unit-actor-critic.md)
