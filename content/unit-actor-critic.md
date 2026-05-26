# Actor-Critic — Combining Value Methods with Policy Gradients

REINFORCE taught the policy directly, but at the cost of waiting for whole episodes and tolerating noisy returns. DQN taught a value function, but only for discrete actions. **Actor-Critic** unifies both ideas: an **actor** picks actions like REINFORCE, while a **critic** estimates returns like DQN. This unit walks from the variance problem in REINFORCE all the way to a complete A2C implementation — the algorithmic backbone of PPO that you have been running in `gdrl` since Unit 2.

[← Policy Gradients](unit-policy-gradients.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~35 min · Training: ~20 min GPU / ~1 h CPU

---

!!! info "Three ways to see your AI"
    Python console (actor / critic loss per update) · matplotlib (actor_loss, critic_loss, ep_rew_mean) · `gym.make("CartPole-v1", render_mode="human")` for a live cart-pole window

!!! warning "Concepts before code"
    The PyTorch listing in Section 5 only makes sense after you understand the advantage function in Section 3. Read top to bottom — do not jump straight to the code.

---

## 1 · The problem REINFORCE left us with

REINFORCE works, but it pays a price for being a pure Monte Carlo method:

- **It needs complete episodes.** The update rule uses the discounted return `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …`. You cannot compute `G_t` until you have seen every reward after step `t`. Long episodes mean slow learning.
- **It has high variance.** `G_t` is the outcome of one rollout. In CartPole an unlucky gust pushes the pole over at step 17 even though the action at step 3 was perfect — REINFORCE still punishes that step 3 action. Across thousands of episodes the noise averages out, but it takes a *lot* of samples.
- **It throws away information.** DQN learned a value network `Q(s,a)` that estimates the return from *any* state without rolling out an episode. REINFORCE ignores that idea entirely.

What if we asked DQN's trick to help REINFORCE? Use a neural network to **estimate the return from a state**, so the actor does not have to wait for the episode to end. That estimator is called the **critic**.

> Mental model: REINFORCE is a student who only knows their grade *after* finals. Actor-Critic is a student who gets an estimated grade after every quiz, courtesy of a TA (the critic) who has been watching all term.

---

## 2 · Two networks, one goal

Actor-Critic uses two function approximators that work together:

| Network | Symbol | Job | Analogue |
|---|---|---|---|
| **Actor** (policy) | `π_θ(a \| s)` | Pick an action given a state | REINFORCE's policy |
| **Critic** (value) | `V_φ(s)` | Estimate the expected return from `s` | DQN's `Q`, but over states only |

- `θ` are the actor's parameters, `φ` are the critic's. Each network has its own gradient signal, but in practice they **share most of the layers** — one trunk processing observations, two small heads on top. This saves parameters and helps the actor learn from features the critic discovered (and vice versa).
- The actor is trained with a **policy gradient**, just like REINFORCE — except the noisy `G_t` is replaced by a less-noisy signal that uses the critic.
- The critic is trained with **TD learning**, exactly the bootstrapping idea from Q-Learning and DQN: predict the return, then nudge the prediction toward the observed reward plus the next state's prediction.

```
            ┌────── shared backbone ──────┐
            │  (e.g. 2× Linear+Tanh, 128) │
            └──────────────┬──────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
      actor head (logits)        critic head (V(s))
         → action                  → scalar return estimate
```

---

## 3 · The advantage function

REINFORCE's gradient pushes up actions whose return `G_t` is large. But "large" relative to what? An episode in CartPole might return +200 because the *state* was easy, not because the *action* was clever. We want to credit the action, not the state.

That is exactly the **advantage**:

```
A(s, a) = Q(s, a) - V(s)
```

In plain English:

> "How much better is action `a` than the *average* action I would have picked from state `s`?"

- **A > 0** — this action is better than average → increase its probability
- **A < 0** — this action is worse than average → decrease its probability
- **A = 0** — this action is exactly average → leave it alone

We do not have access to the true `Q(s, a)`. But we have a critic that estimates `V(s)`, and we know one real reward and the next state from interacting with the env. Bootstrapping (the same trick DQN used) gives:

```
Q(s, a) ≈ r + γ V(s')        ← one-step return
```

Substituting back:

```
A(s, a) ≈ r + γ V(s') - V(s) = δ      ← the TD error of the critic
```

**This is the headline result of the whole unit:** the critic's TD error is an (unbiased) estimate of the advantage. The same number that the critic uses to correct itself is the signal the actor uses to update its policy. One scalar per step, two networks updated.

Term by term:

| Term | What it is |
|---|---|
| `r` | The reward you actually received after taking `a` in `s` |
| `γ V(s')` | The critic's discounted prediction of everything that happens *after* the next state |
| `V(s)` | The critic's prediction of total return from `s` *before* you acted |
| `δ = r + γV(s') - V(s)` | The "surprise" — better or worse than expected? |

---

## 4 · A2C: Advantage Actor-Critic

A2C ("Advantage Actor-Critic", the synchronous cousin of A3C) is the cleanest algorithm built on the idea above. Here is the full loop:

1. **Collect an `n`-step rollout** by running the current policy in the env:
   `(s_0, a_0, r_0), (s_1, a_1, r_1), …, (s_n, a_n, r_n)`
2. **Compute advantage estimates** for every step:
   `A_t = r_t + γ V(s_{t+1}) - V(s_t)`
   (Or the multi-step generalization in Section 7.)
3. **Actor loss** — same shape as REINFORCE, but with `A_t` instead of `G_t`:
   `L_actor = - Σ_t A_t · log π_θ(a_t | s_t)`
4. **Critic loss** — squared TD error, like DQN's regression target:
   `L_critic = Σ_t (r_t + γ V(s_{t+1}) - V(s_t))²`
5. **Entropy bonus** — keep the policy from collapsing too early:
   `L_entropy = - β · H(π_θ) = β · Σ_t π_θ log π_θ`
6. **Total loss**, summed across the rollout:
   `L = L_actor + c · L_critic - β · H(π_θ)`
   with `c = 0.5` (critic weight) and `β = 0.01` (entropy coefficient) as common defaults.
7. **Backprop once** through the shared network. Adam (or RMSProp) updates both actor and critic in a single step.

Compared to REINFORCE this is wildly more efficient: we update every `n` steps instead of every episode, and we use a learned baseline (`V(s)`) instead of the raw return.

---

## 5 · Complete A2C PyTorch implementation

Below is a self-contained A2C agent that solves CartPole-v1 in a few hundred updates on CPU. It uses a shared backbone, n-step rollouts, advantage normalization, an entropy bonus, and gradient clipping — every standard trick you will see again in PPO.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128),     nn.Tanh(),
        )
        self.actor_head  = nn.Linear(128, act_dim)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.actor_head(h)
        value  = self.critic_head(h).squeeze(-1)
        return logits, value

    def get_action(self, obs):
        logits, value = self(obs)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


env = gym.make("CartPole-v1")
model = ActorCritic(obs_dim=4, act_dim=2)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

gamma       = 0.99
vf_coef     = 0.5
ent_coef    = 0.01
n_steps     = 128    # steps per update
max_updates = 500

obs, _ = env.reset()

for update in range(max_updates):
    # 1. Collect n_steps of experience
    obs_list, act_list, rew_list, val_list, logp_list, done_list = [], [], [], [], [], []

    for _ in range(n_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, log_prob, entropy, value = model.get_action(obs_t)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        obs_list.append(obs_t.squeeze(0))
        act_list.append(action)
        rew_list.append(reward)
        val_list.append(value)
        logp_list.append(log_prob)
        done_list.append(done)

        obs = next_obs if not done else env.reset()[0]

    # 2. Compute returns and advantages (reverse pass)
    returns, advantages = [], []
    G = 0.0
    for r, v, d in zip(reversed(rew_list), reversed(val_list), reversed(done_list)):
        G = r + gamma * G * (1 - d)
        adv = G - v.item()
        returns.insert(0, G)
        advantages.insert(0, adv)

    returns    = torch.tensor(returns,    dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_t  = torch.stack(obs_list)
    logp_t = torch.stack(logp_list)

    # 3. Recompute logits/values with current params (for entropy + critic loss)
    logits, values = model(obs_t)
    dist    = torch.distributions.Categorical(logits=logits)
    entropy = dist.entropy().mean()

    actor_loss  = -(advantages * logp_t).mean()
    critic_loss = (returns - values.squeeze()).pow(2).mean()
    loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy

    # 4. Single backward pass through the shared network
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    if update % 50 == 0:
        print(f"Update {update:4d} | actor_loss={actor_loss.item():+.3f} "
              f"critic_loss={critic_loss.item():.3f} entropy={entropy.item():.3f}")
```

Run it. After ~150 updates you should see `ep_rew_mean` climbing past 200 and approaching the CartPole-v1 cap of 500. Entropy will decrease from ~0.69 (the maximum for two equally-likely actions, `ln 2`) toward something like 0.3 as the policy commits.

!!! tip "Compare with REINFORCE"
    Plug the same network into your REINFORCE script from the previous unit (drop the critic head, use raw `G_t` instead of `A_t`). You will see A2C reach the same reward in roughly an order of magnitude fewer environment steps — the same observation that pushed the field from REINFORCE to A2C in the first place.

---

## 6 · Why gradient clipping?

The single line `torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)` is doing surprisingly heavy lifting.

- The advantage can occasionally be very large (a rare big reward, an unusually wrong critic prediction).
- A large advantage multiplied by a `log π` term creates a huge gradient → Adam takes a huge step → the policy distribution swings hard → on the next rollout most actions become absurd → reward collapses → recovery is slow or impossible.
- Clipping the gradient norm to 0.5 caps how far the policy can move per update.

!!! tip "Foreshadowing PPO"
    Gradient clipping is the *blunt* version of "do not let the policy move too far per update". PPO's clipped surrogate objective (next unit) is the *principled* version: instead of capping the gradient *after* computing it, PPO redefines the loss so that updates that would shift the policy too much get zero gradient automatically.

---

## 7 · N-step returns vs 1-step TD

There is a whole spectrum of ways to estimate the return for the critic and the advantage for the actor:

| Estimator | Formula | Bias | Variance |
|---|---|---|---|
| 1-step TD | `r_t + γ V(s_{t+1})` | high (uses critic's biased estimate) | low |
| n-step | `r_t + γ r_{t+1} + … + γ^{n-1} r_{t+n-1} + γ^n V(s_{t+n})` | medium | medium |
| Monte Carlo (REINFORCE) | `G_t = r_t + γ r_{t+1} + …` (to end of episode) | none (unbiased) | high |

Larger `n` uses more real rewards and less of the critic's prediction → less bias, more variance. Smaller `n` does the opposite. `n_steps=128` in the code above is a middle ground that PPO-family algorithms favour.

This is **exactly the `n_steps` parameter you tuned in Unit 4** when calling `gdrl`. Bigger `n_steps` means longer rollouts, fewer updates, more environment data per gradient step. PPO's "advantages" are computed by a generalization called **GAE (Generalized Advantage Estimation)** that smoothly interpolates between 1-step and Monte Carlo via a parameter `λ` — but the spirit is identical to what you see here.

---

## 8 · Entropy bonus for exploration

Look at the loss again:

```
L = L_actor + 0.5 · L_critic - 0.01 · H(π_θ)
```

That last term is the **entropy bonus**. Subtracting entropy from the loss is the same as adding it as a reward.

- `H(π_θ) = - Σ_a π(a|s) log π(a|s)` measures how spread-out the action distribution is.
- For two equally likely actions, `H = ln 2 ≈ 0.693`. For a deterministic policy, `H = 0`.
- Without this bonus, A2C frequently **collapses**: very early in training one action happens to look slightly better, the actor pushes its probability to 1.0, and the agent stops exploring forever.
- `ent_coef = 0.01` is the typical default. If you watch your run and see entropy crash to 0 in the first few updates while reward is still flat, raise it to `0.05`.

!!! warning "Entropy collapse looks like a stuck reward"
    A flat reward curve with very low entropy is the classic signature. The policy has committed early and is no longer trying anything new. Increase `ent_coef`, or lower the learning rate, or both.

This is the same knob as the `--ent_coef` argument in `gdrl` from Unit 4. It is not a magic number — it is the weight in the loss you just read.

---

## 9 · Shared vs separate networks

Two reasonable architectures, two trade-offs:

- **Shared backbone, two heads** (the code in Section 5):
  - Faster, fewer parameters, the actor benefits from features the critic learns.
  - Risk: the critic's huge gradients (squared error can be much larger than the policy gradient) can overwhelm the actor's. The `vf_coef = 0.5` weight exists to soften that.
- **Two separate networks**:
  - More stable, easier to tune actor and critic learning rates independently.
  - Slower, more memory, no feature sharing.

Stable-Baselines3's PPO uses **separate policy and value heads on top of a shared feature extractor** by default, which is a sensible compromise. The `policy_kwargs={"net_arch": [...]}` argument lets you switch.

Practical rule of thumb: use shared for low-dimensional observations (CartPole, simple Godot scenes); use separate when you have image inputs or wildly different scales between actor and critic.

---

## 10 · A2C vs PPO: the single remaining problem

A2C is a complete, working algorithm. So why does anyone use PPO?

- A2C performs **one gradient update per rollout**. The data is discarded immediately afterwards.
- With expensive simulators (Godot at scale, robotics, anything with images), every rollout step is precious. We would like to **reuse the same rollout for multiple gradient steps**.
- But here is the catch: after the first gradient step, the policy has shifted. The actions we took during the rollout are no longer drawn from the *current* policy — they were drawn from the *old* policy. The advantage estimates that worked for the first update become biased for the second.
- Naïvely doing multiple epochs over the rollout makes A2C unstable. The policy can diverge far from the data-generating distribution and everything breaks.

**PPO's clipped objective is the fix.** It introduces a probability ratio `r_t(θ) = π_new(a|s) / π_old(a|s)` and *clips* it to a small interval around 1.0, so updates that would push the new policy too far from the old one get zero gradient. That makes it safe to run **multiple epochs over a single rollout**, which is exactly what `n_epochs=10` does in your Unit 4 `gdrl` command.

---

## 11 · Where A2C appears in the course

You have actually been running A2C the whole time, dressed up as PPO:

- **`gdrl`** uses SB3's PPO under the hood. PPO is A2C plus a clipped surrogate objective plus multi-epoch updates plus GAE.
- When you set `n_steps=512`, you are choosing **A2C's rollout length** from Section 7.
- When you set `batch_size=256`, you are choosing the **minibatch size** that PPO uses to chop up a rollout for multiple gradient steps.
- When you set `n_epochs=10`, you are deciding **how many times to reuse the same rollout** — the thing A2C cannot do safely but PPO can.
- `ent_coef` is the `β` from Section 8.
- `vf_coef` is the `c` from Section 4.
- `clip_range` is PPO's principled replacement for the gradient clipping in Section 6.

Now when you stare at a PPO config file you can name every line.

---

## 12 · Stretch goals

For students who want to dig deeper before moving on to PPO:

- **Separate actor and critic networks.** Refactor `ActorCritic` into two classes with two optimizers. Compare training curves on CartPole. You will probably see slightly more stable but slower learning.
- **Try LunarLander-v2.** A more challenging env where A2C typically needs ~2M steps to solve. Watch the entropy curve carefully — entropy collapse is much more common here.
- **Visualize what the critic learns.** Sample a grid of observations, run them through the critic, plot `V(s)` as a heatmap (for 2D state spaces) or as a 1D curve (for cart position, pole angle). Compare to the rollout returns at those states.
- **Replace 1-step TD with GAE-λ.** Implement Generalized Advantage Estimation with `λ ∈ {0.9, 0.95, 1.0}` and watch how variance and bias trade off in practice. This is the *exact* code path that ships in SB3's PPO.
- **Plug the policy back into Godot.** Re-export the agent as ONNX and load it in a Godot scene the way Unit 5 did, but using your own A2C training script instead of `gdrl`.

---

## 13 · The Actor and Critic Inside SB3's PPO

SB3's PPO is an Actor-Critic method — it has exactly the two heads you built in Section 5. The `ActorCritic` class you wrote maps directly onto `model.policy` in a trained SB3 model.

### Inspecting the actor and critic on a trained Godot agent

```python
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import torch, numpy as np

env = StableBaselinesGodotEnv(env_path="./JumperHard.x86_64", n_parallel=1, speedup=1)
model = PPO.load("logs/sb3/jumper_baseline/best_model", env=env)

obs, _ = env.reset()
obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    # Actor: get action distribution
    dist = model.policy.get_distribution(obs_tensor)
    action_mean = dist.distribution.loc    # mean of Gaussian (continuous actions)
    action_std  = dist.distribution.scale  # std (exploration amount)

    # Critic: get value estimate
    value = model.policy.predict_values(obs_tensor)

print(f"Action mean: {action_mean.numpy()}")
print(f"Action std:  {action_std.numpy()}")
print(f"State value: {value.item():.3f}")
env.close()
```

The `action_mean` is what the actor recommends; `action_std` reflects how much uncertainty (exploration) remains — a well-trained agent has lower std. The `value` is the critic's estimate of expected return from this state.

### Mapping unit variables to SB3 internals

```
Unit variable            →  SB3 PPO equivalent
──────────────────────────────────────────────────
actor_head               →  model.policy.action_net
critic_head              →  model.policy.value_net
shared backbone          →  model.policy.mlp_extractor
advantage A_t            →  computed in rollout buffer
ent_coef                 →  model.ent_coef
vf_coef                  →  model.vf_coef
n_steps                  →  model.n_steps (rollout length)
```

### TensorBoard connection

Every loss term from Section 4's combined loss formula has a TensorBoard counterpart:

- `train/policy_gradient_loss` = L_actor from this unit — the actor improving on advantage estimates
- `train/value_loss` = L_critic from this unit — the critic minimizing squared TD error
- `train/entropy_loss` = L_entropy — the entropy bonus keeping exploration alive

### The explained variance diagnostic

`train/explained_variance` (shown in SB3's TensorBoard) is the most useful single metric for diagnosing your critic. It measures how well `V(s)` predicts the actual returns:

- **Close to 1.0** — the critic has learned a good value function. The actor is getting accurate advantage estimates, and the training signal is clean.
- **Near 0 or negative** — the critic is useless. The actor is essentially running REINFORCE with high variance — exactly the problem this unit was designed to solve. If you see this, the critic is undertrained: try a higher `vf_coef`, more `n_steps`, or a lower learning rate.

Watching `explained_variance` climb from near-zero toward 0.9+ during a Godot training run is seeing the critic learn in real time — the same process you implemented in the CartPole code above, just at scale.

---

## What's next

You now have every conceptual ingredient PPO needs. The next unit takes A2C's loss, swaps `A_t · log π_θ(a_t | s_t)` for a clipped probability ratio, allows multiple epochs over one rollout, and walks through the full PPO update — the algorithm behind every `gdrl` command you have run.

[← Policy Gradients](unit-policy-gradients.md) · [Course home](index.md) · [→ PPO Deep Dive](unit-ppo-deep.md)
