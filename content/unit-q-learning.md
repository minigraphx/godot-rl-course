# Q-Learning — From Tables to Deep Networks

Before we replace policies with neural networks (Unit 3 / DQN), we need to understand the algorithm at the heart of value-based RL. This unit fills the gap between RL Foundations and Deep Q-Networks: it explains the **Bellman equation**, the **Q-table**, the **Q-Learning update rule**, and shows you a 50-line Python implementation that solves FrozenLake from scratch.

[← Unit 2: Lunar Lander](unit-02.md) · [Course home](index.md)

---

!!! info "Three ways to see your AI"
    Python terminal (Q-table printout) · matplotlib (training curve) · policy visualization (arrow grid)

---

## Why this unit exists

So far you have trained policies with PPO — a neural network that outputs **action probabilities** directly. That is a *policy-based* method. A whole other family of RL algorithms works differently: instead of learning *what to do*, they learn *how good every choice is*, and then act greedily on that knowledge. This is the *value-based* family, and **Q-Learning** is its most famous member.

Deep Q-Networks (DQN) — which trained the first agent to beat humans at Atari games — are nothing more than Q-Learning with a neural network in place of a table. If you understand the table version, DQN becomes a small step rather than a leap.

---

## 1 · Why value-based methods?

There are two big strategies in reinforcement learning:

| Family | What it learns | Example algorithms |
|--------|----------------|--------------------|
| **Policy-based** | π(a\|s) — a direct mapping from state to action probabilities | REINFORCE, PPO, A2C |
| **Value-based** | Q(s,a) — how good each action is in each state | Q-Learning, SARSA, DQN |

The core idea of value-based methods is beautifully simple:

> **If you know the value of every (state, action) pair, picking the best action is trivial — just take the action with the highest value.**

You never explicitly learn a policy. The policy is *implicit* in the value function: "in state s, take argmax over a of Q(s,a)".

**When value methods shine:**

- **Discrete action spaces.** Computing `argmax` over 4 actions is cheap. Computing it over a continuous action space (e.g. steering angle ∈ [−1, 1]) is not.
- **Well-defined state spaces.** Grid worlds, board games, discrete navigation tasks.
- **Sample efficiency.** Value methods can reuse old experience (off-policy learning) — see Section 4.

**When they struggle:**

- Continuous actions (use DDPG / SAC / PPO instead).
- Huge or continuous state spaces (use DQN — Section 8).

---

## 2 · The Bellman equation

The Bellman equation is *the* foundational equation of reinforcement learning. Almost everything in value-based RL is a variation of it.

**Intuition first, math second.**

Imagine you are standing in some state `s`. You take an action `a`, receive a reward `r`, and end up in a new state `s'`. The question Bellman answers is: *how valuable was that action?*

The answer has two parts:

1. The **immediate reward** you just received: `r`.
2. The **future value** of where you ended up: whatever you can collect from `s'` onward, if you play optimally from there.

That's it. The value of an action is *what you got right now* plus *what you can get later*.

### Bellman optimality equation for Q

$$
Q^*(s, a) = \mathbb{E}\big[\, r + \gamma \cdot \max_{a'} Q^*(s', a') \,\big]
$$

Term by term:

| Symbol | Meaning |
|--------|---------|
| `Q*(s,a)` | The *optimal* Q-value of taking action `a` in state `s`. The best you can possibly do. |
| `E[...]` | Expected value (average over randomness in environment transitions). |
| `r` | The immediate reward received after taking action `a` in state `s`. |
| `γ` (gamma) | The **discount factor**, between 0 and 1. How much we care about future rewards. |
| `s'` | The next state, reached after taking action `a`. |
| `max_a' Q*(s', a')` | The value of the **best** action in the next state — the *greedy* future. |

**In plain English:** *"A state-action pair is worth as much as the immediate reward, plus the discounted value of the best thing I can do next."*

### Why the discount factor?

`γ` (typically 0.9 to 0.99) does two jobs:

- **Mathematically**, it keeps the sum of future rewards finite even in infinite-horizon tasks.
- **Conceptually**, it expresses preference for sooner rewards. A reward of 1 in 100 steps is worth `γ^100 ≈ 0.37` (at γ=0.99) compared to 1 right now.

### Why the `max`?

Because Q* describes *optimal* play. Once you reach `s'`, you're going to do the best possible thing there — not some random thing. So the future value is the value of the *best* next action, not the average.

---

## 3 · The Q-table

In small, discrete environments, we can store Q-values literally in a table:

|              | Action 0 | Action 1 | Action 2 | Action 3 |
|--------------|----------|----------|----------|----------|
| **State 0**  | Q(0,0)   | Q(0,1)   | Q(0,2)   | Q(0,3)   |
| **State 1**  | Q(1,0)   | Q(1,1)   | Q(1,2)   | Q(1,3)   |
| **State 2**  | Q(2,0)   | Q(2,1)   | Q(2,2)   | Q(2,3)   |
| ...          | ...      | ...      | ...      | ...      |

Each cell `Q(s,a)` is the **expected return** from taking action `a` in state `s`, then playing optimally from then on.

**Training procedure:**

1. Initialize all cells to zero — the agent knows nothing.
2. Let the agent interact with the environment.
3. After each step, update the relevant cell using the **Q-Learning update rule** (Section 4).
4. Repeat for thousands of episodes.

### Walk-through: a 4×4 FrozenLake

FrozenLake is a 4×4 grid:

```
S F F F
F H F H
F F F H
H F F G
```

- `S` = start (state 0)
- `F` = frozen (safe to walk on)
- `H` = hole (fall in → episode ends, reward 0)
- `G` = goal (reward +1, episode ends)

States are numbered 0–15 (row-major). The 4 actions are:

| Action | Meaning |
|--------|---------|
| 0      | ← Left  |
| 1      | ↓ Down  |
| 2      | → Right |
| 3      | ↑ Up    |

After training, the Q-table for state 0 (start) might look like:

| State 0   | ← (0) | ↓ (1) | → (2) | ↑ (3) |
|-----------|-------|-------|-------|-------|
| Q-value   | 0.59  | 0.66  | 0.62  | 0.59  |

The agent learned that going **down** from the start is the best move (highest Q-value). It will pick action 1.

Repeat this for all 16 states, and you have a complete policy.

---

## 4 · The Q-Learning update rule

This is the algorithm in one line:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \cdot \big[\, r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \,\big]
$$

Term by term:

| Symbol | Meaning |
|--------|---------|
| `α` (alpha) | **Learning rate**. How aggressively to update. 0 = never learn; 1 = overwrite completely. Typical values 0.01–0.8. |
| `r + γ · max_a' Q(s',a')` | The **TD target** — our new, better estimate of what Q(s,a) *should* be. |
| `Q(s,a)` (the one being subtracted) | The current estimate. |
| `r + γ · max Q(s') − Q(s,a)` | The **TD error** — the "surprise". How wrong was our old estimate? |

**In plain English:** *"Adjust the current Q-value a small step toward our new best guess. The size of the step depends on how surprised we were."*

### Why is it called "off-policy"?

Look closely at the update. The target uses `max_a' Q(s', a')` — the value of the *greedy* next action. But the agent might not have *taken* the greedy action when it reaches `s'`; it might explore (Section 5).

This means we update toward the *optimal* policy even when *behaving* with a different (exploratory) policy. The behavior policy ≠ the target policy. That's the definition of **off-policy** learning.

The big practical benefit: we can reuse old experience. In DQN this enables **replay buffers** — store transitions, replay them many times.

### Convergence

Q-Learning has a beautiful theoretical guarantee: **given enough exploration** (every state-action pair visited infinitely often) **and a properly decayed learning rate**, Q converges to Q*. In practice you stop early when performance plateaus.

---

## 5 · Exploration vs exploitation

Here is the eternal RL dilemma:

- **Exploit**: take the action with the current highest Q-value — get reliable reward now.
- **Explore**: take a random action — maybe discover something better.

A purely greedy agent stuck on a locally-good action will *never* discover that a different path leads to bigger rewards. It thinks it knows what's best, but its knowledge is based on a tiny sample.

### ε-greedy

The simplest, most effective strategy:

```
With probability ε:      take a random action  (explore)
With probability 1 − ε:  take argmax Q(s, ·)   (exploit)
```

### ε-decay schedule

We want lots of exploration early (when Q is mostly noise) and lots of exploitation late (when Q is trustworthy). So we decay ε over time:

| Phase | ε value | Behavior |
|-------|---------|----------|
| Start | 1.0     | 100% random — pure exploration |
| Mid   | 0.3     | 30% random, 70% greedy |
| End   | 0.05    | Mostly greedy with a tiny bit of exploration |

Two common schedules:

- **Linear decay**: `ε ← ε − ε_step` each episode, clipped to `ε_min`.
- **Exponential decay**: `ε ← ε · decay_rate` each episode.

!!! tip "Why never go to ε = 0?"
    Keeping a small ε (e.g. 0.05) prevents the agent from getting permanently stuck if the environment is stochastic. It is a cheap insurance policy.

---

## 6 · Monte Carlo vs Temporal Difference

There are two fundamentally different ways to estimate value:

### Monte Carlo (MC)

Wait until the episode ends. Compute the **actual** total discounted return:

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
$$

Update Q(s,a) toward `G_t`. You used the **real** future, not an estimate.

### Temporal Difference (TD)

Don't wait. After **one step**, update using the next-step estimate:

$$
\text{target} = r + \gamma \cdot \max_{a'} Q(s', a')
$$

This is called **bootstrapping** — updating an estimate using another estimate. Q-Learning is a TD method (specifically TD(0) — single-step).

### Comparison

| Property | Monte Carlo | TD (Q-Learning) |
|----------|-------------|-----------------|
| Bias     | Unbiased (uses real returns) | Biased (uses estimates) |
| Variance | High (depends on entire trajectory) | Lower (depends on one step) |
| Episode requirement | Needs complete episodes | Works step-by-step |
| Continuing tasks | Cannot handle | Handles fine |
| Learning speed | Slow (one update per episode) | Fast (one update per step) |
| Sample efficiency | Lower | Higher (especially off-policy with replay) |

In practice TD methods dominate in deep RL because of their lower variance and step-by-step learning. Q-Learning is the canonical example.

---

## 7 · Hands-on: Q-Learning on FrozenLake

Time to write the algorithm. We will solve FrozenLake in about 50 lines of NumPy.

### Setup

```bash
pip install gymnasium numpy matplotlib
```

### The full agent

```python
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
Q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.8       # learning rate
gamma = 0.95      # discount
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.005
n_episodes = 10_000

for ep in range(n_episodes):
    state, _ = env.reset()
    done = False
    while not done:
        # Epsilon-greedy action
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-Learning update
        td_target = reward + gamma * np.max(Q[next_state]) * (not done)
        td_error  = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

        state = next_state

    epsilon = max(epsilon_min, epsilon - epsilon_decay)

print("Trained Q-table:")
print(Q.reshape(4, 4, 4))  # 4x4 grid, 4 actions
```

### What is happening, line by line?

- `Q = np.zeros(...)` — 16 states × 4 actions = a 16×4 matrix of zeros.
- The outer loop is **episodes**. We play 10 000 games.
- The inner loop is **steps within an episode**.
- The ε-greedy block picks a random action ε of the time and the best-known action otherwise.
- `td_target` is `r + γ · max Q(s')`, but multiplied by `(not done)` so terminal states have zero future value (nothing comes after the end).
- The update line is literally the Q-Learning formula from Section 4.
- After each episode, ε decays linearly toward 0.05.

### Expected output

Most cells will be zero (states never visited or with no path to the goal). Cells along the optimal path will be non-zero and roughly increasing toward the goal:

```
[[[0.59 0.66 0.62 0.59]
  [0.59 0.   0.34 0.55]
  [0.55 0.49 0.39 0.46]
  [0.34 0.   0.16 0.41]]
 ...
```

Values closer to the goal will approach `γ^k` where `k` is the number of steps remaining. The cell adjacent to the goal will have a Q-value close to 1.

### Evaluation

After training, evaluate the *greedy* policy (no exploration) for 100 episodes:

```python
successes = 0
for _ in range(100):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state])
        state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
    successes += int(reward == 1)

print(f"Success rate: {successes}%")
```

On the **non-slippery** FrozenLake you should reach **100%** success. The greedy policy from the learned Q-table is optimal.

---

## 8 · Why Q-tables don't scale

The Q-table approach works beautifully on FrozenLake. It also works on slightly bigger problems like Taxi-v3 (500 states, 6 actions = 3 000 cells). But it falls apart very quickly:

### State space explosion

Atari Pong has a screen of **210 × 160 RGB pixels**. The number of possible distinct screens is:

```
256^(210 · 160 · 3)  ≈  10^242 932
```

That is more states than atoms in the observable universe. A table is not happening.

### Continuous states

CartPole has 4 continuous state variables (position, velocity, angle, angular velocity). Each is a real number. You cannot index a table with a real number — there are infinitely many.

Naive workaround: **discretize** (bin into buckets). But:

- Coarse bins → poor policy.
- Fine bins → state-space explosion.
- You lose all generalization between similar states.

### Curse of dimensionality

The number of cells in a Q-table grows **exponentially** with the number of state dimensions. 10 binary state features = 1024 states. 20 binary features = a million. 30 = a billion.

### The solution: function approximation

Instead of storing Q(s,a) in a table, **predict** it with a neural network:

```
Q(s, a) ≈ f_θ(s)[a]
```

The network takes a state and outputs a Q-value per action. Similar states naturally get similar outputs (generalization), and the network has a *fixed* number of parameters regardless of state-space size.

That is the move from **Q-Learning** to **Deep Q-Networks**. It is the topic of [Unit 3 — Deep Q-Networks](unit-03.md). Everything you just learned still applies — only the storage mechanism changes.

---

## 9 · Connection to Godot

Why are we doing all this NumPy work in a Godot course?

Because the agents you train in Godot in later units use **the exact same ideas**.

### Mapping

| Q-Learning concept | Godot RL Agents equivalent |
|---------------------|----------------------------|
| State `s` (an integer 0–15) | Observation from `get_obs()` in `AIController` (raycasts, position, velocity) |
| Action `a` (an integer 0–3) | Discrete action from `get_action_space()` |
| Q-table cell `Q[s, a]` | One scalar output of the neural network for action `a` |
| ε-greedy exploration | Built into DQN training in `stable_baselines3.DQN` |
| Q-Learning update | Implemented inside `DQN.learn()` — you just call `.learn(total_timesteps=...)` |
| TD error / loss | The loss that the DQN optimizer minimizes |

### Example: CrossTheRoad (Unit 3)

CrossTheRoad uses DQN. Conceptually:

- The **state** is the agent's local view (grid cells / raycast distances + ego position).
- The **actions** are discrete movements (left / right / up / down / stay).
- The **Q-function** is a neural network: `Q(observation) → 5 Q-values, one per action`.
- Training is Q-Learning with a replay buffer + a target network. We will dig into those tricks in Unit 3.

When you see `model = DQN("MlpPolicy", env)` later, mentally translate it to: *"Build a function approximator for Q(s,a), and update it using the rule from Section 4."*

---

## 10 · Viz checkpoint

A Q-table is useful for analysis but not very *visual*. Here is the trick that makes it click: print the **greedy policy** as a grid of arrows.

```python
actions = ['←', '↓', '→', '↑']
policy = np.argmax(Q, axis=1).reshape(4, 4)
for row in policy:
    print(' '.join(actions[a] for a in row))
```

### Before training (random Q-table)

```
← ← ← ←
← ← ← ←
← ← ← ←
← ← ← ←
```

All zeros → `argmax` returns 0 everywhere → all arrows point left. The agent has no preferences.

### After training

```
↓ → ↓ ←
↓ ← ↓ ←
→ ↓ ↓ ←
← → → ←
```

You can read the path with your eyes: from the start (top-left), follow the arrows down and right, around the holes, to the goal (bottom-right). Cells in holes or unreachable states show garbage arrows — they don't matter because the agent never visits them under the greedy policy.

!!! tip "Sanity check"
    If your trained policy still looks random, you probably trained for too few episodes, or ε never decayed (so the agent never tried exploiting). Print ε at the end of training — it should be near `epsilon_min`.

---

## 11 · Stretch goals

If you finished the FrozenLake exercise quickly, here are three meaningful next steps. They all reuse the code above with small changes.

### A · Slippery FrozenLake

Switch one flag:

```python
env = gym.make("FrozenLake-v1", is_slippery=True)
```

Now the environment is **stochastic** — when you press "right", you have a 1/3 chance to go right and 1/3 chance to slip left or down. Q-Learning still works (the Bellman equation includes an expectation over transitions), but:

- Convergence is much slower (more episodes needed).
- The optimal success rate is well below 100% — even an optimal agent slips occasionally.
- You may need to lower `alpha` (e.g. 0.1) to avoid noisy updates.

Question to answer: *what success rate does your agent reach? How does it change if you raise `n_episodes` to 50 000?*

### B · Taxi-v3

A much bigger discrete environment:

```python
env = gym.make("Taxi-v3")
```

- 500 states (encoding taxi position, passenger location, destination).
- 6 actions (4 movements + pickup + dropoff).
- Q-table size: 500 × 6 = 3 000 cells.

The same code works — `Q = np.zeros((env.observation_space.n, env.action_space.n))` adapts automatically. Try ~25 000 episodes.

### C · Plot the training curve

Track success rate per 100 episodes during training:

```python
import matplotlib.pyplot as plt

window = []
rates  = []
for ep in range(n_episodes):
    # ... training code ...
    window.append(int(reward == 1))
    if len(window) == 100:
        rates.append(sum(window) / 100)
        window = []

plt.plot(rates)
plt.xlabel("Training batch (x100 episodes)")
plt.ylabel("Success rate")
plt.title("FrozenLake Q-Learning")
plt.show()
```

You should see a noisy but rising curve: near-zero at first, climbing as ε decays and the Q-table fills in, plateauing near 1.0 (non-slippery) or some lower value (slippery).

---

## What's next

You now understand the core machinery of value-based RL:

- The **Bellman equation** defines what optimal values look like.
- The **Q-Learning update** moves estimated values toward Bellman targets, one TD error at a time.
- **ε-greedy** balances exploration and exploitation.
- Tables work for tiny worlds; **neural networks** take over for big ones.

In the next unit you will replace the table with a neural network and meet the engineering tricks (replay buffer, target network, Huber loss) that make Deep Q-Networks stable in practice — and then apply them to a Godot scene.

[→ Deep Q-Learning](unit-03.md)
