# SAC — Soft Actor-Critic for Continuous Control

PPO is the default in this course, and for good reason: it handles discrete and continuous actions, scales with parallel environments, and is the algorithm of choice for game-like Godot environments. But the moment you read modern robotics or continuous-control papers, you will hit a different name everywhere — **SAC (Soft Actor-Critic)**. It dominates continuous control benchmarks, real-robot learning, and any domain where simulation is expensive. This unit teaches SAC end to end: the theory, the architecture, the equations, and how to swap PPO for SAC in your existing Godot workflow.

[← PPO in Practice](unit-04.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~40 min · Training: ~20 min GPU / ~1.5 h CPU

---

!!! info "Three ways to see your AI"
    TensorBoard (`train/ent_coef`, `rollout/ep_rew_mean`) · Godot viz checkpoint — SAC policies often look visibly smoother than PPO on the same task · A sample-efficiency comparison plot (SAC vs PPO at equal step counts)

---

## 0 · The PPO gap for continuous control

You have spent four units learning PPO and you are about to break that loyalty. To understand *why*, look closely at what PPO does with experience.

**PPO is on-policy.** Every transition the agent collects is tagged with the policy version that generated it. PPO uses each transition for `n_epochs` gradient passes — typically 10 — and then *throws it away*. The next rollout uses a slightly different policy, so old transitions are no longer valid under the importance-sampling ratio that the clipped objective relies on.

For game-style environments this is fine: Godot can simulate 8, 16, even 32 parallel environments at 20× speed-up. You collect millions of cheap transitions per hour, and the wastefulness of discarding them does not hurt much.

**But continuous control is different.**

| Domain | Cost per transition | Parallel envs feasible? |
|--------|--------------------|------------------------|
| Godot game env (BallChase, JumperHard) | Microseconds | 8–32 parallel, easy |
| MuJoCo robot simulation | Milliseconds | A few parallel |
| Real robot arm | Seconds | One |
| Expensive physics sim (fluid, contact) | Seconds to minutes | One |

Once each transition costs real wall-clock time, throwing them away after one rollout becomes painful. You want to **reuse data**.

**DQN solved this for discrete actions** with a replay buffer: store millions of past transitions and sample mini-batches uniformly. Q-learning is off-policy by construction — the Bellman equation does not care which policy produced the data. But DQN cannot handle continuous action spaces because the `max_a Q(s,a)` operation is undefined when actions are real-valued vectors.

So the open question is:

> Can we have **continuous actions** + **a replay buffer** + **stable training**?

The answer, since 2018, is yes. The algorithm is **SAC — Soft Actor-Critic** (Haarnoja et al., 2018). Section 9 below traces the DDPG → TD3 → SAC lineage that produced it.

---

## 1 · Maximum entropy RL — the framework SAC is built on

Standard RL has one objective:

```
J(π) = E_π [ Σ_t γ^t · r_t ]
```

Maximize expected discounted return. Nothing else.

**Maximum entropy RL adds a second term** — the entropy of the policy at each visited state:

```
J(π) = E_π [ Σ_t γ^t · ( r_t + α · H(π(·|s_t)) ) ]
```

where:

- `H(π(·|s)) = -Σ_a π(a|s) · log π(a|s)` is the **policy entropy** at state `s` — a measure of how random the action distribution is
- `α` is the **temperature** — a scalar that controls how much we weight entropy versus reward
- The summation `Σ_t γ^t · ...` is the same discounted sum as standard RL

In plain English, the agent is rewarded for two things simultaneously:

1. **Get high reward** (the original RL objective)
2. **Stay as random as possible** while doing so

This sounds paradoxical until you think about what it actually does:

- **Natural exploration.** A high-entropy policy keeps trying alternatives instead of locking in early. Compare this to PPO's `ent_coef` — an entropy *bonus* tacked onto the loss. In SAC, entropy is part of the *objective itself*, not an auxiliary regularizer.
- **Avoids premature convergence.** The classic failure mode in RL is committing to a suboptimal strategy because it pays off early. Maximum entropy says: do not commit until you are *really* confident.
- **Finds multiple modes.** If two action sequences both achieve high reward, a max-entropy policy will assign probability to *both* rather than collapsing to one. This matters in robotics — there are often multiple equally good ways to grasp an object.
- **Robust to perturbations.** A randomized policy degrades gracefully if the environment changes slightly; a deterministic policy can fail catastrophically.

The temperature `α` is the dial between these two worlds. `α = 0` collapses to standard RL (deterministic optimum). `α = ∞` ignores reward entirely (uniform random policy). SAC operates somewhere in between — and, crucially, **tunes α automatically**. We will come back to that.

---

## 2 · SAC architecture — three networks (plus targets)

SAC keeps five networks in memory at once. This is more than PPO, but each plays a clear role.

**Actor `π_φ(a|s)`** — the policy

- Input: observation `s`
- Output: parameters of a Gaussian over actions — a **mean `μ_φ(s)`** and a **log standard deviation `log σ_φ(s)`**
- To sample an action: draw `ε ~ N(0, 1)` and compute `a = μ_φ(s) + σ_φ(s) · ε` (the **reparameterization trick** — see §7)
- In practice a `tanh` squash is applied so actions stay in `[-1, 1]`

**Critic 1 `Q_θ1(s, a)`** — first Q-value network

- Input: observation *and* action concatenated (or fed in as two heads)
- Output: a single scalar — the estimated Q-value

**Critic 2 `Q_θ2(s, a)`** — second, independent Q-value network

- Identical architecture to Critic 1, but with completely independent weights `θ2`
- Trained on the same target as Critic 1
- The **twin critics trick**: when computing the actor's update or the Bellman target, take `min(Q_θ1, Q_θ2)`
- Why: a single Q-network systematically *overestimates* values because of the max-like operation in the Bellman backup (this is the same overestimation problem DQN fights with target networks). Taking the minimum of two independent estimates is a cheap, effective way to reduce that bias. The idea comes from TD3 (Twin Delayed DDPG) and SAC inherits it.

**Target critics `Q̂_θ1`, `Q̂_θ2`** — slow-moving copies of the two critics

- Used only on the *right-hand side* of the Bellman equation (computing `y`, the target)
- Updated by a **soft update** every gradient step: `θ̂ ← τ · θ + (1 - τ) · θ̂`, where `τ = 0.005` is tiny
- Without target networks the critic chases a moving target and training diverges — the same lesson DQN taught us

**Replay buffer**

- FIFO queue of up to 1,000,000 past transitions `(s, a, r, s', done)`
- Every gradient step samples a mini-batch (typically 256) uniformly at random
- The replay buffer is what makes SAC *off-policy*: transitions stay valid even though the policy that generated them is long obsolete, because the Bellman update does not depend on the data-generating policy

**Diagram of one forward pass:**

```
Observation s
    │
    ▼
Actor π_φ ──► (μ, log σ) ──► sample a (reparameterization)
                                  │
                                  ▼
                       Critic 1 Q_θ1(s, a) ─┐
                                            ├─► min ─► Q̂(s, a)   ← target for actor update
                       Critic 2 Q_θ2(s, a) ─┘

Target critics Q̂_θ1, Q̂_θ2 (soft update from Q_θ1, Q_θ2 every step, τ = 0.005)
```

---

## 3 · SAC update equations — one gradient step, term by term

Every step, SAC samples a mini-batch `{(s, a, r, s', done)}` from the replay buffer and runs three updates: critic, actor, temperature. We walk through each carefully.

### 3.1 · Critic update — minimize the (soft) Bellman error

The target value `y` for each transition is:

```
y = r + γ · ( min( Q̂_θ1(s', ã'),  Q̂_θ2(s', ã') ) - α · log π_φ(ã' | s') )

    where  ã' ~ π_φ( · | s' )      (sampled from the *current* policy at s')
```

Term by term:

- `r` — immediate reward, from the buffer
- `γ` — discount factor (typically 0.99)
- `s'` — next state, from the buffer
- `ã'` — a fresh action sampled from the **current** policy at `s'`. This is the key reason SAC is off-policy: we do not need the action that was originally taken at `s'` — we ask the current policy what *it* would do
- `min( Q̂_θ1, Q̂_θ2 )` — the twin-critic trick: take the *lower* of the two target-critic estimates to reduce overestimation bias
- `- α · log π_φ(ã' | s')` — the **maximum entropy correction**. This subtracts the log-probability of the chosen action, weighted by the temperature. Because `-log π` is an unbiased estimator of entropy `H(π)`, this term effectively adds the future-entropy contribution to the Bellman target. This is what makes it a *soft* Bellman equation

The two critics are trained by minimizing mean-squared Bellman error:

```
L_critic = MSE( Q_θ1(s, a) - y )  +  MSE( Q_θ2(s, a) - y )
```

Note that `(s, a)` come from the buffer (the action that was actually taken), while `ã'` is freshly sampled. This separation is exactly what lets SAC reuse old data.

### 3.2 · Actor update — maximize Q while maximizing entropy

The actor's objective: pick actions that yield high Q-value *and* keep the policy stochastic.

```
L_actor = E_{s ~ buffer,  a ~ π_φ(·|s)}  [  α · log π_φ(a | s)  -  min( Q_θ1(s, a),  Q_θ2(s, a) )  ]
```

We *minimize* `L_actor`, which is equivalent to *maximizing* `Q - α · log π`.

- `α · log π_φ(a | s)` — penalize being too confident (low entropy). Minimizing this term *increases* entropy
- `- min(Q_θ1, Q_θ2)` — negative Q means maximizing Q
- The expectation `a ~ π_φ(·|s)` is computed using the reparameterization trick, so gradients flow from `Q(s, a)` back through `a` into the actor's weights `φ`. Without reparameterization this gradient would be blocked by the sampling step

The actor never sees the replay buffer's *actions* directly — it only sees the *states*, and rolls its own fresh actions through the reparameterized sampler.

### 3.3 · Temperature update — automatic entropy tuning

`α` itself is a learnable parameter, trained to keep policy entropy close to a target value `H_target`:

```
L_α = E_{a ~ π_φ}  [  -α  ·  ( log π_φ(a | s)  +  H_target )  ]

    where  H_target = -dim(action_space)    (a common default)
```

Intuition:

- If current entropy `H(π) = -E[log π]` is **above** `H_target` → the term `(log π + H_target)` is **negative** on average → gradient *decreases* `α` → less entropy pressure
- If current entropy is **below** `H_target` → gradient *increases* `α` → more entropy pressure
- At equilibrium, entropy hovers around `H_target` and `α` settles to whatever value sustains it

The default `H_target = -dim(action_space)` is heuristic but works well across a huge range of tasks. For a 4-dimensional continuous action space, `H_target = -4`. It scales naturally: more action dimensions → more entropy needed to stay diverse.

---

## 4 · Hands-on — SAC with Stable Baselines 3

The good news: SB3 makes SAC a drop-in replacement for PPO. You change one import and a few hyperparameters.

```python
from stable_baselines3 import SAC
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

# SAC on a continuous Godot environment (FlyBy from Unit 6)
env = StableBaselinesGodotEnv(
    env_path="./FlyBy.x86_64",
    n_parallel=1,        # SAC is off-policy: no benefit from many envs
    speedup=20,
)

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/",
    learning_rate=3e-4,
    buffer_size=1_000_000,    # replay buffer size
    learning_starts=10_000,   # collect transitions before training starts
    batch_size=256,
    tau=0.005,                # soft update coefficient
    gamma=0.99,
    train_freq=1,             # update every env step
    gradient_steps=1,         # one gradient step per env step
    ent_coef="auto",          # automatic entropy tuning (α is learned)
    target_entropy="auto",    # H_target = -dim(action_space)
)

model.learn(total_timesteps=1_000_000, tb_log_name="sac_flyby")
model.save("flyby_sac")
env.close()
```

A few notes on the hyperparameters:

| Parameter | Why this value |
|-----------|---------------|
| `learning_rate=3e-4` | Standard for SAC, identical to PPO default. The Adam optimizer handles the rest |
| `buffer_size=1_000_000` | One million transitions. Smaller (100k) is fine for short tasks; larger wastes memory |
| `learning_starts=10_000` | Collect 10k transitions with a random policy before any gradient step. Avoids garbage-quality early updates |
| `batch_size=256` | Mini-batch size for each gradient step. 256 is the universal SAC default |
| `tau=0.005` | Soft-update rate for target networks. Smaller = more stable, slower to track changes |
| `train_freq=1`, `gradient_steps=1` | One env step → one gradient step. You can do more gradient steps per env step (e.g. `gradient_steps=4`) if data collection is slow relative to GPU training — common in robotics |
| `ent_coef="auto"` | Activates the automatic α tuning described in §3.3 |
| `target_entropy="auto"` | Sets `H_target = -dim(action_space)` automatically |

**TensorBoard signals to watch:**

| Signal | What it means |
|--------|--------------|
| `train/ent_coef` | The current value of α. Should stabilize at some positive value, not collapse to 0 (means policy went deterministic too early) and not stay huge (means learning is not making the policy more confident at all) |
| `train/actor_loss` | Should be negative (we are maximizing Q minus entropy penalty) and should slowly improve as Q-values rise |
| `train/critic_loss` | Mean-squared Bellman error. Should decrease as Q-estimates stabilize. Spikes early are normal; persistent high values mean something is wrong |
| `train/ent_coef_loss` | The loss for the temperature update. Should hover near zero at equilibrium |
| `rollout/ep_rew_mean` | The main performance metric — same as PPO |

!!! warning "Don't expect parallel speed-ups"
    SAC's update is per-environment-step. Adding more parallel envs increases data collection but does **not** proportionally increase gradient steps. Set `n_parallel=1` (or 2–4 if your sim is very fast) and let the algorithm do its work.

---

## 5 · PPO vs SAC — the decision guide

This is the table to come back to whenever you start a new project.

| Aspect | PPO | SAC |
|--------|-----|-----|
| On/off-policy | On-policy (discard after use) | Off-policy (replay buffer) |
| Action space | Discrete + continuous | Continuous (vanilla); discrete variants exist (sb3_contrib DiscreteSAC, CleanRL sac_atari) |
| Replay buffer | No | Yes (typically 1M transitions) |
| Parallel envs | Yes — scales near-linearly | Minimal benefit |
| Sample efficiency | Lower — discards data | Higher — reuses transitions |
| Hyperparameter sensitivity | Medium (clip range, n_steps, lr) | Low — automatic α tuning handles most of it |
| Stability | High (clipped objective) | High (twin critics + target nets) |
| Wall-clock speed | Fast with many envs | Slower per step, but fewer steps needed |
| Memory footprint | Small (no buffer) | Large (1M transitions × obs dim) |
| Best for | Game-style envs, fast simulators, parallel rollouts | Robotics, expensive sims, precise continuous control |

!!! tip "When to choose which — the rule of thumb"
    **Use PPO when** your action space includes any discrete actions, your environment is cheap enough to run 8+ parallel copies, and you are working in a typical Godot game environment.

    **Use SAC when** the action space is purely continuous, simulation is expensive (a real robot, a heavy physics sim, anything that takes seconds per step), data collection is your bottleneck, and you want maximum sample efficiency.

!!! warning "Vanilla SAC is designed for continuous action spaces"
    SB3's `SAC` class assumes purely continuous actions. If your `AIController` exposes any discrete buttons (jump, fire, grab) alongside continuous controls, vanilla SAC will not work out of the box. Discrete variants do exist — `sb3_contrib.DiscreteSAC` and CleanRL's `sac_atari.py` — but for mixed action spaces the simplest choice is PPO or `RecurrentPPO`.

---

## 6 · The reparameterization trick — why SAC's actor update works

The actor update in §3.2 contains this expectation:

```
E_{a ~ π_φ(·|s)} [ Q(s, a) - α · log π_φ(a | s) ]
```

We want gradients of this expectation with respect to `φ`. There is a deep problem: the action `a` is *sampled* from a distribution parameterized by `φ`, and **sampling is not differentiable**. The gradient signal from `Q(s, a)` cannot flow back into `μ_φ` and `σ_φ` through a sample-step.

**The reparameterization trick** solves this by moving the randomness *outside* the network. Instead of sampling `a` directly from `N(μ_φ(s), σ_φ(s)²)`, we sample a *standard* Gaussian noise variable independently and combine it:

```
ε ~ N(0, 1)              ← sampled independently, no parameters
a = μ_φ(s) + σ_φ(s) · ε  ← deterministic function of φ and ε
```

Now `a` is a *deterministic* function of `φ` (and the random noise `ε`, which has nothing to do with `φ`). Gradients flow cleanly:

```
∂a/∂φ = ∂μ_φ/∂φ  +  ε · ∂σ_φ/∂φ
```

That gradient gets multiplied by `∂Q/∂a` to give the full actor gradient. The randomness is preserved (different `ε` samples give different `a`), but the path from `φ` to `a` is fully differentiable.

**Compare to PPO.** PPO sidesteps this entire issue with the importance-sampling ratio `r_t = π_new(a|s) / π_old(a|s)`. PPO never differentiates *through* the action sample — it just reweights *already collected* actions by the policy ratio. The reason isn't simply that PPO is on-policy (REINFORCE is also on-policy yet uses the score-function estimator). It's that PPO's update is built around the IS ratio, which only needs `log π(a|s)` evaluated at stored actions — a derivative through the *parameters*, not the sample. SAC's actor objective is `E_{a ~ π}[Q(s, a)]`, so the gradient must flow through the action `a` itself; reparameterization is what makes that path differentiable.

This is one of the deep mathematical reasons SAC and PPO look so different despite both being actor-critic methods.

---

## 7 · Automatic entropy tuning — what `ent_coef="auto"` does for you

Tuning `α` by hand is painful:

- Too high → the entropy term dominates → the policy stays random forever and never converges
- Too low → entropy collapses → the policy becomes deterministic → no exploration → stuck in a local optimum
- The right value depends on reward scale, action dimension, and how far along training is

Worse, the *ideal* `α` changes during training. Early on you want lots of entropy (exploration); later you want less (exploitation).

**Automatic entropy tuning** (Haarnoja et al. 2018, follow-up paper) treats `α` as a learnable parameter and adapts it online to maintain a *target entropy* `H_target`:

- You set `H_target` once. The default `-dim(action_space)` is a heuristic that works across most tasks.
- Each gradient step, the temperature loss `L_α` from §3.3 is minimized.
- If the policy is becoming too deterministic (entropy below target), `α` rises automatically, increasing the entropy bonus.
- If the policy is too random (entropy above target), `α` falls, letting the policy converge.

**Practical effect on a training curve:**

- Early steps: `α` is high (say 0.5–1.0) — the policy is encouraged to explore broadly
- As Q-estimates stabilize: `α` drifts down (often to 0.05–0.2) — the policy starts exploiting
- Eventually: `α` settles at whatever value sustains `H(π) ≈ H_target` for the final policy

In SB3, this is all enabled by `ent_coef="auto"` and `target_entropy="auto"`. You will see `train/ent_coef` in TensorBoard evolving during training — it should look like a smooth, slowly-decaying curve, not a step function or a collapse.

!!! tip "What if `ent_coef` collapses to 0?"
    Means the policy has gone fully deterministic far below `H_target`, and the optimizer is desperately pulling `α` down. Usually a sign that the reward signal is overwhelming entropy — try setting `target_entropy` more negative (e.g. `-2 * dim(action_space)`) to demand more stochasticity.

---

## 8 · SAC in the Godot workflow

The whole point of this course is that you train in Python and deploy in Godot. SAC fits into that pipeline with one small change.

**Training** — identical to PPO, just swap the algorithm class:

```python
# Old (PPO)
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, ...)

# New (SAC)
from stable_baselines3 import SAC
model = SAC("MlpPolicy", env, ...)
```

**ONNX export for inference** — export the *actor* only. The critics exist purely for training; at deployment you only need `π_φ(a|s)`.

```python
import torch

obs_dim = env.observation_space.shape[0]
dummy_obs = torch.zeros(1, obs_dim)

# SB3's SAC actor is exposed as model.policy.actor
torch.onnx.export(
    model.policy.actor,
    dummy_obs,
    "flyby_sac.onnx",
    input_names=["obs"],
    output_names=["action"],
    opset_version=11,
)
```

**Godot side** — exactly the same as your PPO units. Set the Sync node to `ONNX_INFERENCE`, point it at `flyby_sac.onnx`, and the agent runs deterministically in-game.

!!! warning "Parallel envs and SAC in Godot"
    When you set up the `Sync` node in your Godot project for SAC training, **set `n_parallel=1`** in the wrapper and in any launcher script. Running multiple parallel envs with SAC wastes CPU and barely improves wall-clock learning, because SAC's bottleneck is gradient steps, not data collection.

---

## 9 · The DDPG → TD3 → SAC lineage

Understanding where SAC's architecture comes from makes it less mysterious. Three algorithms, each fixing the previous one's main problem.

**DDPG (Deep Deterministic Policy Gradient):** off-policy actor-critic for continuous actions. The actor outputs a single deterministic action `π(s) → a` (no distribution). Uses a replay buffer — the first continuous-control algorithm to do so effectively. Problem: overestimates Q-values (single critic), brittle to hyperparameters, exploration requires manually injected noise.

**TD3 (Twin Delayed DDPG, Fujimoto et al. 2018)** — three fixes:

1. **Twin critics:** train two Q-networks independently, take the minimum for target computation → prevents Q-value overestimation
2. **Delayed policy updates:** update the actor every 2 critic steps → reduces variance in the actor gradient when critic estimates are still noisy
3. **Target policy smoothing:** add small Gaussian noise to the target action during critic updates → regularizes the Q-function, prevents it from exploiting sharp peaks

**SAC** takes the twin-critic trick from TD3 and adds maximum entropy RL (Section 1) to replace manual noise injection with principled stochastic exploration.

| | DDPG | TD3 | SAC |
|--|------|-----|-----|
| Policy | Deterministic | Deterministic | Stochastic |
| Exploration | Noise injection | Noise injection | Entropy maximization |
| Twin critics | No | Yes | Yes |
| Delayed actor updates | No | Yes | No (not needed) |
| Sample efficiency | Medium | High | Highest |
| Hyperparameter sensitivity | High | Medium | Low (auto-α) |

**When TD3 over SAC:** when you need a deterministic policy at inference (some real-robot deployments require repeatable actions); when the stochasticity of SAC creates issues in a specific environment. SB3 supports TD3 directly:

```python
from stable_baselines3 import TD3
model = TD3("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

---

## 10 · Stretch goals

Try these to deepen your understanding:

1. **Compare SAC vs PPO sample efficiency on FlyBy.** Train both algorithms for 1M steps with the same environment. Plot `rollout/ep_rew_mean` against environment steps for each. SAC should reach the same reward level with significantly fewer transitions — that is the headline result.

2. **Tune `ent_coef` manually.** Run SAC with `ent_coef=0.1`, `ent_coef=0.01`, `ent_coef=0.001`, and `ent_coef="auto"`. Plot `train/ent_coef` and final reward for each. The auto-tuned curve should outperform every hand-picked value — confirming that automatic tuning earns its keep.

3. **SAC on HovercraftRacing.** A harder continuous-control task. SAC needs longer (2–3M steps) but often produces noticeably smoother control than PPO — watch the viz checkpoint side by side and see if you agree.

4. **Twin-critic ablation (advanced).** SB3 does not expose a single-critic SAC out of the box, but you can fork the SAC class and disable the `min` operation. Train both versions and observe Q-value drift — the single-critic version should overestimate Q significantly, and final policy performance should suffer.

5. **Replay buffer size sweep.** Try `buffer_size=10_000`, `100_000`, `1_000_000`. Smaller buffers force the algorithm to "forget" earlier transitions; larger ones keep stale data around. Find the sweet spot for your environment.

---

## What's next

You now have two tools in the toolbox: PPO (your workhorse for game environments) and SAC (your specialist for continuous control). The next unit goes back to PPO and asks a different question — how do we scale data collection by running many environments in parallel?

**Unit 5: Parallel Training** covers `n_parallel`, how rollouts are stitched across envs, and how to think about throughput vs sample efficiency. Note that parallelism helps PPO substantially (linear scaling up to dozens of envs) but barely helps SAC — another illustration of the on-policy / off-policy divide.

Want to go deeper into how PPO actually works in code? **PPO From Scratch (CleanRL)** walks through every line of a single-file PPO implementation before you scale up.

[→ PPO From Scratch (CleanRL)](unit-cleanrl.md) · [→ Parallel Training](unit-05.md)
