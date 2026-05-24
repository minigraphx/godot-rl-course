# Intrinsic Motivation — Curiosity and Sparse Rewards

[← Deep Q-Learning](unit-03.md) · [Course home](index.md)

---

!!! info "Three ways to see your AI"
    Godot (exploration coverage — how much of the map does the agent visit?) · TensorBoard (`rollout/ep_rew_mean` rises earlier than the no-curiosity baseline) · RND prediction error decreasing over training as states become familiar

---

## 1 · The sparse reward problem

Standard RL training assumes the agent will occasionally stumble onto a reward by chance — then learn to repeat that lucky behaviour. Dense reward environments make this easy: every small step forward gives a non-zero signal, gradients flow, the policy updates.

Sparse rewards break this assumption.

**Example:** a maze where the only reward is a +1 at the exit, and -0 everywhere else. With a random policy, the agent might run 1 million steps without a single positive signal. No reward → no gradient → no learning. The policy stays frozen at random behaviour.

The failure mode looks like a flat `ep_rew_mean` in TensorBoard — not oscillating, just perfectly flat — because the agent is effectively blind to the task objective.

**Why standard tricks don't fix it:**

- More timesteps: the agent is still exploring blindly — a longer random walk rarely helps
- ε-greedy (DQN): still random in unexplored regions
- Entropy bonus: keeps the policy spread out, but doesn't *direct* exploration toward novel states
- Reward shaping: works, but requires hand-engineering a dense proxy reward for every new environment

What's needed is a general mechanism that motivates the agent to explore — independent of the external reward signal.

---

## 2 · Intrinsic motivation: curiosity as a reward

The core idea: give the agent an additional reward for visiting **novel states**, regardless of what the environment says.

```
r_total = r_ext + β · r_int
```

- `r_ext` — the environment's external reward (sparse, rare, task-specific)
- `r_int` — an intrinsic curiosity reward (dense, generated internally, state-novelty)
- `β` — a scaling factor that balances exploration drive vs. task performance (typical: 0.01–1.0)

`r_int` is high for states the agent hasn't visited before, and decays toward zero as states become familiar. The agent becomes intrinsically motivated to explore its environment — not because a designer told it to, but because novelty itself is rewarding.

This mirrors theories of human motivation: infants are "curious" about novel stimuli in their environment long before they understand what the stimuli are useful for.

**Key property:** intrinsic rewards work even when `r_ext = 0` for the entire first phase of training. The agent explores widely, builds an internal model of the environment, and *then* starts exploiting when external rewards appear.

---

## 3 · Random Network Distillation (RND)

RND (Burda et al. 2018) is the simplest effective curiosity method and the standard starting point for deep RL.

**The setup — two networks:**

| Network | Role | Trained? |
|---------|------|----------|
| **Target network** f: obs → embedding | Produces a fixed random embedding of any observation | No — weights are frozen at random initialisation |
| **Predictor network** g_θ: obs → embedding | Tries to match the target network's output | Yes — trained on every visited observation |

**The intrinsic reward:**

```
r_int = ||f(obs) - g_θ(obs)||²
```

This is the mean squared prediction error between the two networks' outputs for the current observation.

**Why this measures novelty:**

- **Novel state:** the predictor has never seen this observation → it has no learned mapping → high prediction error → high `r_int`
- **Familiar state:** the predictor has been trained on this observation many times → it closely matches the target → low prediction error → low `r_int`

**Why the target network is fixed — the key insight:**

The target must be *fixed* (random, frozen weights). If the target network could also update, the predictor could just "follow" the target regardless of novelty — prediction error would collapse to zero everywhere. The frozen random target creates a stable, consistent function the predictor can only match by *actually seeing the observation during training*.

**Compared to earlier curiosity methods:**

Older approaches (ICM — Intrinsic Curiosity Module, Pathak et al. 2017) used a learned forward model to predict `s_{t+1}` from `(s_t, a_t)` and measured surprise as the prediction error. This required learning both forward and inverse dynamics models — more complex and prone to the "noisy TV problem" (stochastic environments appear infinitely novel). RND sidesteps this entirely: it doesn't model transitions, just embeddings.

---

## 3.5 · ICM vs RND — when each approach wins

**ICM (Intrinsic Curiosity Module) recap:**

- **Forward model:** predicts next-state embedding from `(state, action)` — `f(s_t, a_t) → ê_{t+1}`
- **Inverse model:** predicts action from `(state, next_state)` — `g(s_t, s_{t+1}) → â_t`
- **Curiosity signal:** prediction error of the forward model in feature space (not raw pixels)

The inverse model ensures the feature space captures action-relevant information only — irrelevant background noise is not encoded.

**The noisy-TV problem** — why ICM can fail:

A TV showing random static is always "novel": the forward model cannot predict the next frame (it is stochastic). ICM assigns high curiosity to the TV forever — the agent gets stuck watching it instead of exploring. The root cause: ICM confuses *unpredictability* (stochasticity) with *novelty* (unfamiliarity).

RND avoids this because its target network is fixed: a stochastic observation maps to the same fixed random embedding every time. The predictor eventually learns it, error drops, curiosity fades. Stochastic ≠ novel under RND.

**ICM vs RND comparison:**

| | ICM | RND |
|--|-----|-----|
| What it measures | Forward model prediction error | Distance from fixed random network |
| Noisy-TV problem | Yes — stochastic envs fool it | No — stochastic obs have fixed target |
| Compute cost | Higher (trains two networks) | Lower (one predictor only) |
| Feature space | Learned via inverse model | Fixed random projection |
| Best for | Deterministic envs with structured dynamics | General use — the safe default |
| SB3 support | Manual (no built-in) | Manual (no built-in) |

**When to choose ICM:** deterministic environments where the agent controls all meaningful state changes, and you want the feature space to reflect action-relevant information (e.g., a robot arm where joint angles matter, but background lighting does not).

**Default:** use RND. It is simpler, cheaper, and immune to the noisy-TV problem.

---

## 4 · RND in practice with Stable-Baselines3

Install the required packages:

```bash
pip install stable-baselines3 sb3-contrib
```

The `RNDWrapper` below wraps any Godot environment to add RND intrinsic rewards. The core idea: intercept every `step()` call, compute `r_int`, and add it to `r_ext`.

```python
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv


class RNDModule:
    """Computes RND intrinsic rewards for a given observation dimension."""

    def __init__(self, obs_dim: int, embed_dim: int = 64, lr: float = 1e-3):
        # Fixed random target — never updated
        self.target = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )
        for param in self.target.parameters():
            param.requires_grad = False

        # Trained predictor — updated on every step
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)

    def compute_reward_and_train(self, obs_np: np.ndarray) -> float:
        """Return intrinsic reward for obs_np, and update the predictor."""
        obs_t = torch.tensor(obs_np, dtype=torch.float32)

        with torch.no_grad():
            target_embed = self.target(obs_t)

        pred_embed = self.predictor(obs_t)
        error = ((target_embed - pred_embed) ** 2).mean()

        # Train predictor to match target on this observation
        self.opt.zero_grad()
        error.backward()
        self.opt.step()

        return error.item()


class RNDGodotEnv:
    """
    Wraps a StableBaselinesGodotEnv to inject RND intrinsic rewards.

    Usage:
        env = RNDGodotEnv("./MyEnv.x86_64", beta=0.1)
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(500_000)
        env.close()
    """

    def __init__(self, env_path: str, beta: float = 0.1, n_parallel: int = 4, speedup: int = 20):
        self.env = StableBaselinesGodotEnv(
            env_path=env_path, n_parallel=n_parallel, speedup=speedup
        )
        self.beta = beta
        obs_dim = self.env.observation_space.shape[0]
        self.rnd = RNDModule(obs_dim=obs_dim)

        # Expose SB3-required attributes
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, r_ext, done, info = self.env.step(action)
        r_int = self.rnd.compute_reward_and_train(obs)
        r_total = r_ext + self.beta * r_int
        return obs, r_total, done, info

    def close(self):
        self.env.close()


# Training with RND
env = RNDGodotEnv("./MultiLevelRobot.x86_64", beta=0.1, n_parallel=4, speedup=20)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/")
model.learn(total_timesteps=1_000_000)
model.save("multilevel_rnd")
env.close()
```

!!! warning "β tuning is environment-specific"
    If β is too high, the agent ignores external rewards and explores forever — `ep_rew_mean` stays near zero even after millions of steps. If β is too low, curiosity adds no signal. Start with β = 0.1 and halve/double based on whether the agent is exploring enough or ignoring the task. Watch both `ep_rew_mean` (external task) and the RND prediction error in TensorBoard.

**Logging RND prediction error:**

Add this to your training loop to track curiosity signal strength over time:

```python
# After each update step, log mean prediction error across recent batch
# Lower error = agent is visiting more familiar states = exploration maturing
```

---

## 5 · Where curiosity helps in Godot

**Good fits:**

- **MultiLevelRobot (Unit 9):** the agent must navigate to platforms it has never visited. Random exploration rarely reaches higher platforms. RND pushes the agent to visit novel elevations.
- **FPS / RobotFPS (Unit 8):** agents stuck behind walls need to discover doors or corridors. Standard ε-greedy explores locally; RND drives global novelty.
- **Any maze-like environment** with a single terminal reward — the classic sparse reward case.

!!! warning "Don't use curiosity in dense-reward environments"
    In environments with dense, well-shaped rewards (BallChase, LunarLander, CrossTheRoad with forward progress bonus), adding curiosity introduces noise. The agent may explore irrelevant states rather than optimising the task signal. Dense reward + curiosity = slower learning, not faster. If `ep_rew_mean` rises smoothly without curiosity, leave it out.

**Diagnosis checklist — when to add curiosity:**

1. `ep_rew_mean` is flat for > 200k steps?
2. The reward function has only 1–2 terminal reward events per episode?
3. Random exploration cannot reach the reward without extended luck?

If yes to all three → try RND. Otherwise → reward shaping (see [Reward Engineering unit](unit-reward-engineering.md)) may be the better lever.

---

## 6 · Count-based exploration

The oldest exploration bonus method — provably optimal in tabular settings:

```
r_int = 1 / sqrt(N(s))
```

`N(s)` is the number of times state `s` has been visited. Newly visited states have `N=1` → bonus = 1.0. Frequently visited states have large N → bonus ≈ 0. The agent is automatically pulled toward unexplored regions.

**Where it works:** tabular settings with discrete state spaces — the Q-Learning FrozenLake example from the [Q-Learning unit](unit-q-learning.md). With a finite grid, you can maintain an exact count table and show that a UCB-style bonus achieves sample-efficient exploration with theoretical guarantees.

**UCB (Upper Confidence Bound):** extends the count-based idea to action selection:

```
a* = argmax_a [ Q(s, a) + c · sqrt(log t / N(s, a)) ]
```

The square-root term is an exploration bonus that decays as action `(s, a)` is visited more. Well-studied in bandit literature; SB3 does not implement UCB for deep RL directly, but the intuition shows up everywhere — including the PPO entropy bonus.

**Why exact counts don't scale:** continuous state spaces make every state technically unique. A robot at position (1.000, 2.000) and (1.001, 2.000) are different states — both have N=0 forever. State aggregation (discretising) partially helps but loses information.

**SimHash / locality-sensitive hashing** — an approximate solution:

```python
# SimHash: project obs onto a random matrix, take sign → binary hash
A = np.random.randn(hash_bits, obs_dim)  # fixed random matrix
def hash_obs(obs):
    return tuple((A @ obs > 0).astype(int))

counts = defaultdict(int)
def count_bonus(obs):
    h = hash_obs(obs)
    counts[h] += 1
    return 1.0 / np.sqrt(counts[h])
```

Nearby observations hash to the same bucket more often than distant ones. Count bucket visits rather than exact states. Computationally cheap; works in moderate-dimensional obs spaces.

**Exploration methods at a glance:**

| Method | Measures | Scales to deep RL? | Noisy-TV safe? |
|--------|---------|-------------------|----------------|
| ε-greedy | Nothing — random | Yes | Yes |
| Entropy bonus | Action distribution spread | Yes | Yes |
| Count-based | Exact visit count | No (continuous states) | Yes |
| SimHash | Approximate visit count | Partially | Yes |
| RND | Predictor error ≈ inverse count | Yes | Yes |
| ICM | Forward model surprise | Yes | No |

**Practical recommendation:** RND for most deep RL tasks (scalable, noisy-TV safe). Count-based/SimHash for research comparison or for environments with discrete, low-dimensional state spaces. UCB for bandit-style problems.

---

## 7 · Entropy bonus vs. curiosity

These address different levels of exploration and are complementary:

| Mechanism | Controls | Level |
|-----------|---------|-------|
| `ent_coef` in PPO | Keeps the **action distribution** spread out | Action-level diversity |
| RND curiosity bonus | Rewards visiting **novel states** | State-level diversity |

**Entropy bonus** (`ent_coef=0.01` is the SB3 default) prevents the policy from collapsing to a single deterministic action. It says: "don't always do the same thing." It costs nothing extra — PPO already computes the entropy of the policy distribution.

**Curiosity bonus** rewards discovering new regions of state space. It says: "go somewhere you haven't been before." It requires an extra network and adds training cost.

!!! tip "Use both for hard exploration tasks"
    For the hardest exploration problems, use `ent_coef` plus RND:
    
    - Entropy bonus keeps the policy varied at the action level — less likely to get locked into a local loop
    - RND directs exploration toward globally novel states
    
    Set `ent_coef=0.01–0.05` in PPO and `beta=0.05–0.2` for RND. If the agent's behaviour looks repetitive in the viz checkpoint, increase `ent_coef` first (cheap). If the agent isn't reaching new map regions, increase `beta`.

---

## 8 · Viz checkpoint for curiosity

After training with and without RND, run both policies with `--viz` (or `show_window=True` in your eval script):

**What to look for in Godot:**

- Does the curiosity-trained agent visit more of the map before finding the goal?
- Does it revisit the same corner repeatedly (low curiosity signal) or spread out systematically?
- Does the no-curiosity agent get stuck near spawn while the curiosity agent discovers remote platforms?

**What to watch in TensorBoard:**

| Metric | No curiosity | With curiosity |
|--------|-------------|----------------|
| `rollout/ep_rew_mean` | Stays flat for 500k+ steps | Starts rising at ~100–200k steps |
| RND prediction error | N/A | High early, decays as map becomes familiar |
| `rollout/ep_len_mean` | Episodes end at spawn (early death or timeout) | Longer episodes as agent explores more |

The curiosity signal should show a clear decay curve: high prediction error in the first 20% of training (everything is novel), declining as the agent builds familiarity with the environment. A flat or non-decaying error curve means the agent is not revisiting states — check whether it's dying before reaching novel regions.

---

## 9 · Stretch goals

- **Count-based comparison:** Train the FrozenLake Q-Learning example from the [Q-Learning unit](unit-q-learning.md) with and without a `1/sqrt(N)` count bonus. How many fewer steps are needed to find the optimal policy?
- **CrossTheRoad (Unit 3) test:** Apply RND to CrossTheRoad. Does curiosity help (sparse reward) or hurt (small but non-zero forward progress bonus)? Measure `ep_rew_mean` convergence speed.
- **Read the paper:** Burda et al. 2018, "Exploration by Random Network Distillation." ~10 pages, clearly written, includes Atari results. The ablation in Section 5 is particularly instructive — it shows why the fixed random target is essential.
- **β sweep:** Train MultiLevelRobot with β ∈ {0.01, 0.1, 0.5, 1.0}. Plot `ep_rew_mean` vs timesteps for each. What is the cost of β being too large?

---

## What's next

With curiosity, your agent can tackle environments that standard PPO/DQN would fail on due to sparse rewards. The next unit covers policy gradients — the theoretical foundation behind PPO and why policy-based methods scale to continuous actions.

[→ Policy Gradients](unit-policy-gradients.md)
