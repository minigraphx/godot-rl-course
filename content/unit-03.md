# Unit 3 — CrossTheRoad & DQN

Study the official **CrossTheRoad** example — discrete 2D navigation with sparse rewards — then train it with **DQN** instead of PPO. Default workflow from here: exported binary, headless.

[← Unit 2: Lunar Lander](unit-02.md) · [Course home](index.md)

!!! note "Prerequisites"
    - **[Unit 1](unit-01.md)** — Q-values, value-based vs policy-based families, on-policy vs off-policy
    - **[Unit 2](unit-02.md)** — `AIController` interface, `get_obs()` / `set_action()`, training a PPO agent end-to-end
    - **[Q-Learning unit](unit-q-learning.md)** — tabular Q-Learning (recommended; helps Section 2 click instantly)
    - Comfort exporting a Godot project to a headless binary

!!! info "Time"
    Reading: ~45 min · Training: ~45 min GPU / ~3 h CPU

---

!!! warning "Console-first from here"
    Units 0–2 used the editor for building and debugging. From Unit 3 on, train with an exported binary and omit `--viz` for speed — then run a short **viz checkpoint** when training finishes (see Section 9).

!!! info "Three ways to see your AI"
    Godot (viz checkpoint) · TensorBoard (DQN vs Unit 2 PPO) · `AIController` reward tweaks

!!! warning "Training stalled?"
    Check in order: (1) reward sign and scale — is "good" actually positive? (2) sparse rewards — does the agent get any signal before the goal? (3) observation bugs — are sensors updating after resets? (4) TensorBoard flat but Godot looks fine — you may need longer training or a viz checkpoint. → [Full diagnostic guide](unit-debugging.md#3-no-learning-signal-ep_rew_mean-flat)

---

## 1 · DQN in one page

**Value-based** methods learn how good each action is in each state — these are called **Q-values**. **DQN (Deep Q-Network)** uses a neural network to approximate those values, with two stabilizing tricks:

- **Experience replay** — transitions are stored in a buffer and sampled randomly, breaking correlations between sequential observations
- **Target network** — a periodically-copied frozen network is used for computing targets, preventing oscillation

Use DQN when the action space is **discrete** and rewards are sparse — like crossing lanes of traffic.

| | PPO (Unit 2) | DQN (this unit) |
|--|--|--|
| Family | Policy-based | Value-based |
| Learns | Policy directly | Q-values → policy |
| Exploration | Stochastic policy + entropy | ε-greedy (random early, greedy later) |
| Best for | Dense rewards, continuous or discrete | Sparse rewards, discrete actions |

!!! tip "Exploration: ε-greedy"
    DQN explores with **ε-greedy**: random actions early, greedy Q-actions later. If the curve is flat, ε may be decaying too fast — check your training script's exploration schedule. Compare Unit 2's PPO run in TensorBoard after this unit.

---

## 2 · From Q-tables to neural networks

If you haven't read the Q-Learning unit yet, now is the time — the tabular foundations make everything here click: [Q-Learning unit](unit-q-learning.md).

**The problem with Q-tables**

Classical Q-Learning builds a lookup table: every row is a state, every column is an action, every cell holds a Q-value. That works perfectly when the state space is small. The moment you move to real environments, it breaks:

- **CrossTheRoad (this unit):** position on a grid — manageable, maybe a few thousand states
- **Atari Pong:** raw pixel frames — 210 × 160 pixels × 3 channels → approximately 10^18,000 possible states
- **Godot RayCast observations:** floating-point vectors — literally infinite states

A table with 10^18,000 rows cannot exist. We need a function approximator that *generalises* across similar states.

**Enter the Deep Q-Network**

Replace the table with a neural network:

```
Input layer:   observation  (pixels / raycasts / any vector)
Hidden layers: learned feature extraction
Output layer:  one Q-value per action
               Q(s, a_1), Q(s, a_2), ..., Q(s, a_n)
```

The network maps a state *s* to a vector of Q-values — one per action. The agent picks the action with the highest Q-value (when being greedy). The Bellman update rule is unchanged from tabular Q-Learning; we just apply it to the network's outputs instead of a table cell.

**The Atari breakthrough (2013/2015)**

DeepMind's 2013 paper "Playing Atari with Deep Reinforcement Learning" and the 2015 Nature paper that followed showed a *single* DQN architecture — the same network weights, the same algorithm — could learn to play 49 Atari games directly from raw pixel input, reaching human-level or better on many of them. The key ingredients were exactly the two tricks listed in Section 1: experience replay and a target network. Before those tricks, training was wildly unstable.

!!! info "Why does scaling Q-learning to deep networks create new problems?"
    Neural network training assumes i.i.d. (independent and identically distributed) data. RL transitions are neither — consecutive frames are nearly identical, and the target values we train toward keep shifting as the network learns. Experience replay and the target network are engineering solutions to both problems. We cover each in depth below.

---

## 3 · Experience Replay

**Why consecutive transitions are a problem**

Imagine the agent taking steps *s_t → s_{t+1} → s_{t+2}* across a road. These three observations are nearly identical — slightly different positions on the same road. If you train on them in order, each mini-batch contains only one type of experience. The network overfits to "being near position X" and forgets everything it learned about positions A, B, and C earlier.

This is **catastrophic forgetting** — the neural network version of a student who crams one topic so hard they forget the others.

**The replay buffer**

The fix is simple in concept: store every transition the agent ever experiences, then train on *random* mini-batches drawn from the whole history.

A single transition is a tuple:

```
(s, a, r, s', done)
 │   │   │   │    └── did the episode end?
 │   │   │   └─────── next state
 │   │   └─────────── reward received
 │   └─────────────── action taken
 └─────────────────── current state
```

In Python pseudocode:

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)  # circular: old entries drop off

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
```

**Key design choices:**

- **Capacity:** 10k–1M transitions. Larger buffers keep older, more diverse experiences in the pool. CrossTheRoad can use 50k; Atari-scale tasks use 1M.
- **Random sampling:** Each training step draws a random mini-batch. Transitions from 100 episodes ago mix with transitions from 5 episodes ago — no temporal correlation.
- **Circular (deque):** When the buffer is full, the oldest entry drops off. This prevents the buffer from filling with stale pre-trained experience.

!!! tip "Buffer size vs. memory"
    Storing raw pixel observations at 1M capacity costs gigabytes of RAM. For godot-rl-agents tasks, raycast observations are small floats — 100k capacity is usually fine and keeps memory manageable.

### Prioritized Experience Replay (PER)

Uniform random sampling treats every transition equally — but most transitions in a large buffer are "boring" (reward = 0, Q-target close to current estimate). **PER** samples transitions in proportion to their TD error: transitions the network got most wrong are revisited more often.

**Priority formula:**

```
p_i = |δ_i|^α + ε_per
```

- `δ_i` — TD error for transition i (large error → high priority)
- `α` — controls how much prioritization (0 = uniform, 1 = fully proportional)
- `ε_per` — small constant so every transition has non-zero chance of being sampled

Sampling by priority introduces bias (non-uniform data distribution), corrected by importance-sampling weights `w_i = (1 / N·P(i))^β` applied to the loss.

**SB3 via sb3-contrib:**

```python
from stable_baselines3 import DQN
from sb3_contrib.common.buffers import PrioritizedReplayBuffer

model = DQN(
    "MlpPolicy", env,
    replay_buffer_class=PrioritizedReplayBuffer,
    replay_buffer_kwargs={"alpha": 0.6},
    learning_starts=1000,
)
```

**When PER helps:** sparse reward tasks where a few transitions contain almost all the useful signal (first-time goal reached, first collision). **When it doesn't:** dense reward environments where nearly every transition is informative — uniform sampling is already good enough. Watch `train/td_loss` variance: high variance early that falls quickly is a sign PER is working.

---

## 4 · Target Network

**The moving target problem**

The DQN loss function is:

```
L = (r + γ · max_a' Q(s', a') − Q(s, a))²
         └── bootstrap target ──┘
```

Both `Q(s, a)` (the prediction) and `Q(s', a')` (the target) come from the *same* network. Every time we update the network to reduce this loss, *both sides move*. We're chasing a target that runs away with every step — like trying to hit a ball that moves every time you swing.

In practice this causes training to oscillate or diverge entirely.

**The solution: freeze the target**

Keep *two* copies of the network:

| Network | Role | Updated how often |
|---------|------|-------------------|
| **Q-network** (online) | Makes predictions, trained every step | Every gradient step |
| **Target network Q̂** | Provides bootstrap targets | Every C steps (hard copy) or continuous soft update |

The loss becomes:

```
L = (r + γ · max_a' Q̂(s', a') − Q(s, a))²
                   ↑ frozen target network
```

Now the target side is stable for C steps, giving the online network something fixed to converge toward.

**Hard update vs. soft update**

```python
# Hard update — copy weights every C steps (e.g., C = 1000)
if step % 1000 == 0:
    target_net.load_state_dict(q_net.state_dict())

# Soft update — blend weights every step (more stable, slower lag)
tau = 0.005
for p, pt in zip(q_net.parameters(), target_net.parameters()):
    pt.data = tau * p.data + (1 - tau) * pt.data
```

SB3's DQN uses a soft update by default (`tau=1.0` means hard update; lower values give soft). The `target_update_interval` parameter controls how often updates happen.

!!! info "Two networks, same architecture"
    Target and online network share the exact same architecture — only the weights differ. The target network does not receive gradient updates directly; it only receives periodic copies of the online network's weights.

!!! warning "The deadly triad — why DQN can diverge"
    Combining three things causes Q-learning to diverge:

    1. **Function approximation** (neural network instead of table)
    2. **Bootstrapping** (using Q̂ to estimate Q — the Bellman update)
    3. **Off-policy learning** (replay buffer contains data from old policies)

    Any two of these are fine. All three together create instability.

    DQN survives by careful engineering:

    - Target network: makes bootstrapping targets more stable (addresses the #2 + #3 interaction)
    - Replay buffer with recent data only: limits how off-policy the data gets
    - Gradient clipping: prevents function approximation from overshooting

    This is why you can't just plug any neural network into the Bellman equation — the engineering matters.

---

## 5 · Epsilon-Greedy Exploration in DQN

**The exploration-exploitation dilemma**

A purely greedy agent always picks the action with the highest current Q-value. Early in training those Q-values are random noise — greedy means randomly bad choices that never improve. Purely random is safe for learning but never converges.

ε-greedy threads the needle:

```
With probability ε:     take a random action  (explore)
With probability 1-ε:   take argmax Q(s, a)  (exploit)
```

**The decay schedule**

Start with heavy exploration, decay toward exploitation as the agent accumulates experience:

```
ε_start = 1.0      # 100% random at step 0
ε_end   = 0.05     # 5% random after decay period
decay_steps = 100_000

ε(t) = ε_end + (ε_start - ε_end) × max(0, 1 - t / decay_steps)
```

At step 0 the agent acts randomly, discovering diverse crossings. By step 100k it mostly exploits its learned Q-values, with 5% random exploration to avoid getting stuck in local optima.

**Three decay schedules**

| Schedule | Formula | When to use |
|----------|---------|-------------|
| **Linear** | `ε(t) = ε_end + (ε_start - ε_end) · max(0, 1 - t/decay_steps)` | Most tasks — predictable and easy to tune |
| **Exponential** | `ε(t) = ε_end + (ε_start - ε_end) · exp(-t / decay_rate)` | Fast early exploration; tail stays above `ε_end` |
| **Curriculum** | Step function at milestones | When task difficulty changes in discrete phases |

Exponential decay explores aggressively early and tapers slowly — useful when you need the agent to cover state space fast but want a long "refinement" phase. Curriculum schedules (e.g., keeping ε = 0.5 for 100k steps while the task is still random, then dropping to 0.05) are common in multi-stage environments.

**Practical rules:**

- `exploration_fraction` — fraction of total timesteps over which ε decays (SB3 DQN parameter). `exploration_fraction=0.1` means ε decays over the first 10% of training.
- `exploration_final_eps` — ε at the end of the decay period (= `ε_end` in the formula). Default `0.05`.
- `exploration_initial_eps` — starting ε. Default `1.0`.

```python
model = DQN("MlpPolicy", env,
    exploration_fraction=0.2,       # decay over first 20% of timesteps
    exploration_final_eps=0.05,     # 5% random at convergence
    verbose=1)
```

TensorBoard tracks `rollout/exploration_rate` — watch it decay and cross-reference with `ep_rew_mean`. The reward should start rising roughly when ε crosses 0.2–0.3 (agent begins exploiting learned Q-values). If `ep_rew_mean` is still flat at ε = 0.05, the problem is not exploration — check reward design.

**Training vs. evaluation**

- **During training:** use the ε schedule — lots of exploration early
- **During evaluation:** set ε = 0 (fully greedy) or use `deterministic=True` — you want the agent's *best* behavior, not random exploration

!!! tip "Watching ε decay in CrossTheRoad"
    The DQN agent exploring CrossTheRoad is trying random moves early — you'll see it fall off the road constantly in the first few thousand steps. As ε decays toward 0.05, the agent starts exploiting its learned Q-values and you'll see it make purposeful crossing attempts. The flat → sharp-jump curve in TensorBoard corresponds directly to ε decay meeting a sufficient amount of replay buffer experience.

### Decay schedule design

**Linear decay** — the SB3 default, controlled by two parameters:

```python
model = DQN(
    "MlpPolicy", env,
    exploration_fraction=0.1,   # fraction of total_timesteps over which ε decays
    exploration_final_eps=0.05, # ε_min at the end of the decay period
)
# ε(t) = max(ε_min, ε_start − (ε_start − ε_min) × (step / decay_steps))
```

**Exponential decay** — aggressive early exploration, slow late:

```python
epsilon = epsilon_min + (epsilon_start - epsilon_min) * math.exp(-step / decay_rate)
```

| Schedule | Shape | When to use |
|----------|-------|-------------|
| Linear | Constant rate | Most tasks — predictable and easy to tune |
| Exponential | Fast early, slow late | When you want heavy exploration in early training |
| Curriculum | Step function | When task difficulty changes (multi-room, staged environments) |

**Practical rules:** `exploration_fraction=0.1` (decay over 10% of timesteps) is often too fast for sparse tasks — try `0.3`. `exploration_final_eps=0.05` keeps the policy from becoming brittle. Monitor `rollout/exploration_rate` in TensorBoard to confirm ε is decaying at the rate you expect.

---

## 6 · Open CrossTheRoad

**Clone and import**

1. From [examples/CrossTheRoad](https://github.com/edbeeching/godot_rl_agents_examples/tree/main/examples/CrossTheRoad), open the project in Godot .NET
2. Enable the Godot RL Agents plugin (Project → Project Settings → Plugins)
3. Locate the training scene and `AIController` script

**Export for headless training**

Project → Export → add your platform preset → export binary. Then train against it:

```bash
conda activate godot_env
gdrl --env_path=./CrossTheRoad.x86_64 \
  --experiment_name=CrossTheRoad_DQN \
  --timesteps=500000 \
  --speedup=8 \
  --n_parallel=4
```

!!! info "DQN via SB3"
    The default `gdrl` command uses PPO. To use DQN, write a short training script:

    ```python
    from stable_baselines3 import DQN
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

    env = StableBaselinesGodotEnv(env_path="./CrossTheRoad.x86_64", n_parallel=4, speedup=8)
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="logs/")
    model.learn(total_timesteps=500_000)
    model.save("crosstheroad_dqn")
    env.close()
    ```

---

## 7 · Read the code

Trace these in order — same rhythm as SimpleReachGoal in Unit 2:

- **`get_obs()`** — what the agent sees (position, nearby hazards, distance to goal)
- **`get_action_space()`** — discrete moves (wait / up / down / left / right)
- **Reward logic** — sparse signal at goal + penalty for crashes; note the contrast to Unit 2's dense shaped lander rewards
- **Sync node** — number of parallel env roots in the training scene

**Key question to answer before training:** Does the agent receive *any* reward signal during a typical episode, or only at the very end? Sparse rewards require more exploration time — factor that into your timestep budget.

---

## 8 · Train headless

```bash
conda activate godot_env
tensorboard --logdir=logs &
python train_crosstheroad_dqn.py
```

Watch `ep_rew_mean` — sparse rewards may stay flat for thousands of episodes, then jump sharply as the agent discovers safe crossings. This is normal for DQN on sparse tasks; PPO on the same task would show a smoother rise.

**What to compare in TensorBoard after this unit:**

| Metric | PPO (Unit 2 lander) | DQN (CrossTheRoad) |
|--------|--------------------|--------------------|
| `ep_rew_mean` curve shape | Smooth, gradual rise | Flat → sharp jump |
| `train/entropy_loss` | Present | Not applicable |
| `train/loss` | Policy + value losses | TD loss only |

!!! check "Done when"
    CrossTheRoad has no published benchmark, so judge success two ways: (1) the **viz checkpoint** (Section 9) shows the agent reaching the far side in the majority of episodes, and (2) `ep_rew_mean` has clearly stepped up out of its early flat phase and stabilised — the characteristic DQN "flat → sharp jump" curve. A curve still flat after your full step budget points to the ε schedule or a reward-sign bug, not to needing more time.

---

## 9 · Tweak & viz checkpoint

**Viz checkpoint (~5 min)**

Re-run the trained policy with `--viz` or Play Scene in Godot. Screenshot behavior that matches (or contradicts) the TensorBoard curve — keeps headless training from feeling invisible.

**Stretch goals (pick one):**

- Increase crash penalty by 2× — does learning speed up or stall?
- Add a small reward for forward progress — compare to pure sparse setup
- Train the same env with PPO — which algorithm reaches reliable crossings first?

---

## 10 · DQN limitations

DQN is elegant but has real constraints you will hit in later units.

**Discrete actions only**

Q-values are defined over a finite set of actions: Q(s, a_1), Q(s, a_2), ..., Q(s, a_n). Taking the max is O(n) — tractable when n is 5 (CrossTheRoad) or 18 (Atari). With *continuous* actions (e.g., joint torques in JumperHard or throttle in FlyBy), n is infinite. You cannot enumerate and maximise over infinitely many actions. This is why value-based methods are mostly restricted to discrete control.

**Overestimation bias**

The `max` operator is optimistic: it tends to overestimate Q-values because it always picks the highest noisy estimate. Over time, overestimated values bootstrap into each other and Q-values become inflated. This can slow learning or destabilize late training.

**Double DQN — the fix**

Decouple action *selection* from action *evaluation*:

```
# Standard DQN (biased):
target = r + γ · max_a' Q̂(s', a')                     # target net picks AND evaluates

# Double DQN (unbiased):
a_star = argmax_a' Q(s', a')                            # online net selects best action
target = r + γ · Q̂(s', a_star)                         # target net evaluates that action
```

Using the online network to choose the action and the target network to evaluate it removes the systematic upward bias.

**Dueling DQN — the extension**

Split the network's final layers into two streams:

- **Value stream** V(s) — how good is this state regardless of action?
- **Advantage stream** A(s, a) — how much better is action a than average?
- Combine: Q(s, a) = V(s) + (A(s, a) − mean_a A(s, a))

This helps the agent learn that some states are simply bad regardless of what it does — useful for CrossTheRoad where falling into traffic is catastrophically bad no matter what move you make next.

!!! info "SB3 handles these automatically"
    Stable-Baselines3's `DQN` class supports Double DQN via `policy_kwargs={"optimize_memory_usage": False}` and Dueling networks via `policy_kwargs={"dueling": True}`. You don't need to implement them from scratch.

!!! tip "Bridge to continuous control"
    For continuous actions — Unit 6 FlyBy, JumperHard with joint torques — we need policy-based methods that output action *distributions* rather than Q-value tables. That is exactly what PPO does, and why the next unit focuses on it. See [Unit 4: JumperHard & PPO](unit-04.md).

### Noisy Networks and Rainbow DQN

**Noisy Networks** replace the final linear layers of the Q-network with `NoisyLinear` layers that inject *learned* noise into the weights. The network controls its own exploration by adjusting the noise magnitude — no ε schedule needed.

**Rainbow DQN** (Hessel et al. 2017) combines six DQN improvements into a single agent:

| Component | What it adds |
|-----------|-------------|
| Double DQN | Unbiased target Q-values |
| Dueling DQN | Separate value + advantage streams |
| Prioritized Replay (PER) | Sample important transitions more |
| Multi-step returns | n-step TD instead of 1-step |
| Distributional RL (C51) | Model full return distribution, not just mean |
| Noisy Networks | Learned exploration — no ε schedule |

The landmark result: combining all six beats any individual improvement by a large margin on Atari. The improvements are complementary, not redundant.

**Practical note for Godot tasks:** full Rainbow is not in SB3. Use individual components: Double DQN is on by default; Dueling via `policy_kwargs={"dueling": True}`; PER via sb3-contrib (see Section 3.1 above). For most Godot environments, PPO outperforms any DQN variant — DQN shines on discrete, sparse-reward tasks. If you are on a discrete task with sparse rewards and DQN is still underperforming, try PER + Dueling before reaching for Rainbow.

---

## 11 · Off-policy vs On-policy: Why it matters

This distinction is one of the most practically important in RL — it explains why DQN and PPO need very different infrastructure.

**On-policy (PPO):** training data must come from the *current* policy. After each gradient update, all collected transitions are discarded — they are now "stale" (from the old policy) and cannot be reused.

| | On-policy (PPO) | Off-policy (DQN, SAC) |
|--|--|--|
| Data source | Current policy only | Any policy (including old ones, random policy) |
| After an update | Discard all transitions | Keep all transitions in replay buffer |
| Sample efficiency | Low — each transition used once | High — each transition used many times |
| Stability | Theoretically sound, stable | Requires correction mechanisms |

**Off-policy (DQN):** data from *any* policy can be used for training. The replay buffer stores millions of transitions collected by many different policy versions — including the random policy from early training. The agent trains on random mini-batches from this entire history.

**Why this makes DQN's target network necessary:**

Because the replay buffer contains transitions collected by old policies, the Q-values you're bootstrapping from were generated by a different (often worse) policy than the one currently being trained. Without the target network stabilising the bootstrap targets, this off-policy nature would cause the loss function to chase a shifting, inconsistent target — leading to divergence.

**Practical implications:**

- DQN with a large replay buffer can be more sample-efficient than PPO for discrete-action tasks — every transition is reused hundreds of times
- PPO scales better with parallel environments (see [Unit 5: Parallel Training](unit-05.md)) — running N envs in parallel gives N times more on-policy data per second
- For continuous actions, SAC uses the same off-policy principle as DQN but extends it to continuous control — see [Unit SAC](unit-sac.md)

---

## 12 · Stretch Goals

**Sweep the ε schedule.** Re-run CrossTheRoad three times, changing only the exploration schedule: (a) fast decay — ε reaches 0.05 by 25 % of training, (b) the default, (c) slow decay — ε still at 0.3 at 75 % of training. Predict which curve climbs fastest, which plateaus highest, and which never recovers. Then check TensorBoard. The lesson is that DQN's wallclock-to-solve depends as much on the ε schedule as on the network.

**Head-to-head DQN vs PPO.** Train CrossTheRoad with PPO using the same total step budget (e.g. 500k). Plot both `ep_rew_mean` curves on the same TensorBoard. Which one reaches the goal first? Which one ends higher? Write down a one-sentence hypothesis for *why* before you run it. CrossTheRoad's discrete actions + sparse rewards favour DQN — confirm or falsify on your own machine.

**Implement Double DQN by hand.** Without SB3, write a small training loop that learns CartPole-v1 using two networks: an online Q-network for action selection and a target network for evaluation. The Double DQN target is `r + γ · Q_target(s', argmax_a Q_online(s', a))` — not `r + γ · max_a Q_target(s', a)`. Copy the online weights into the target every 500 steps.

!!! warning "Pseudocode"
    ```python
    import gymnasium as gym
    import torch
    import torch.nn as nn

    env = gym.make("CartPole-v1")
    q_online = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))
    q_target = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))
    q_target.load_state_dict(q_online.state_dict())

    # In the update step:
    next_action = q_online(next_obs).argmax(dim=-1)              # online picks
    next_q = q_target(next_obs).gather(-1, next_action.unsqueeze(-1))  # target evaluates
    td_target = reward + gamma * next_q.squeeze(-1) * (1 - done)
    loss = ((q_online(obs).gather(-1, action.unsqueeze(-1)).squeeze(-1) - td_target.detach()) ** 2).mean()
    ```

    Watch `ep_rew_mean` rise toward 500. Compare against a single-network baseline — the gap is small on CartPole, large on Atari.

---

## What's next

**Unit 4:** JumperHard — the canonical PPO benchmark, headless export, hyperparameter tuning.

!!! info "Self-check before you move on"
    Can you answer these in your own words?

    1. Why does a neural network replace the Q-table once you leave grid worlds?
    2. What does the **experience replay buffer** fix that on-policy methods don't need to worry about?
    3. What goes wrong if the target network is updated every step instead of every N steps?
    4. Why is ε-greedy the natural exploration strategy for DQN but not for PPO?
    5. Pick one Godot environment from Phase 2 — would you reach for DQN or PPO, and why?

    If you can answer all five — you're ready.

??? success "Self-check answers"
    1. Real environments have too many (or continuous) states to store one Q-value each. A **network approximates Q(s, a)** and generalises across similar states it has never seen — a table cannot.
    2. The **replay buffer** breaks the temporal correlation between consecutive transitions (and lets each transition be reused). Random minibatches behave more like i.i.d. data, which stabilises training. On-policy methods discard data after each update, so they never face this correlation/reuse problem.
    3. If the target network is updated **every step**, the bootstrap target moves with the online network — the net chases a constantly shifting target, causing oscillation or divergence. Freezing it for N steps gives a stable target to regress toward.
    4. DQN learns deterministic **Q-values** with no built-in randomness, so it needs an explicit explore/exploit knob — **ε-greedy**. PPO already has a **stochastic policy + entropy bonus**, so exploration is intrinsic and ε is unnecessary.
    5. Example — **CrossTheRoad: DQN.** Discrete actions plus sparse rewards are DQN's sweet spot, and off-policy replay is sample-efficient there. (A dense-reward or continuous-action env would point to PPO instead.)

[→ Unit 4: JumperHard & PPO](unit-04.md)
