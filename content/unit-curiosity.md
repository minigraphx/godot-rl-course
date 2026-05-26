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

## 3.1 · ICM vs RND — when to use each

**ICM recap (Intrinsic Curiosity Module, Pathak et al. 2017):** two networks trained jointly — an *inverse model* predicts the action taken from `(s_t, s_{t+1})` (forces features to be action-relevant), and a *forward model* predicts the next-state embedding from `(s_t, a_t)`. Curiosity = forward model error in feature space.

**The noisy-TV problem:** ICM measures how *unpredictable* the next state is. A TV showing random static is always unpredictable — the agent gets stuck watching it forever because every frame is "novel." Any stochastic element in the environment (random particle effects, procedural noise) becomes a curiosity magnet.

**Why RND avoids this:** the fixed random target network produces the same output for the same observation every time. A noisy TV produces the same distribution of pixel patterns — after a few visits, the predictor matches the target for those patterns and `r_int` falls to near zero. RND measures *unfamiliarity*, not *unpredictability*.

| | ICM | RND |
|--|-----|-----|
| What it measures | Forward model prediction error | Distance from fixed random network |
| Noisy-TV problem | Yes — stochastic envs fool it | No — stochastic obs have fixed target |
| Compute | Higher (trains two networks) | Lower (trains one predictor) |
| Feature space | Learned (inverse model) | Random projection |
| Best for | Deterministic envs, action-relevant features | General use — the safe default |

**When to use ICM:** fully deterministic environments where the agent controls all state changes and you want the curiosity features to capture action-relevant structure. **Default to RND** in all other cases — it is simpler, cheaper, and handles stochastic environments correctly.

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

The oldest exploration bonus: reward the agent inversely proportional to how often it has visited a state.

```
r_int = 1 / sqrt(N(s))
```

where `N(s)` is the visit count. Never-visited states get high bonus; well-explored states get near-zero bonus. This is provably optimal in tabular settings — it drives the agent to visit every state at least O(√T) times in T steps.

**UCB (Upper Confidence Bound)** — extends the same idea to action selection in bandit problems:

```
a* = argmax_a [ Q(s,a) + c · sqrt(log t / N(s,a)) ]
```

The second term is the exploration bonus: high when an action has rarely been tried. Well-studied in theory; SB3 does not implement UCB for deep RL.

**SimHash / locality-sensitive hashing** — approximate counting for continuous spaces: hash the observation into a discrete bucket using a random projection matrix, then count bucket visits. `r_int = 1 / sqrt(N(hash(s)))`. Computationally cheap; works on high-dimensional observations.

**Why exact counting doesn't scale:** continuous state spaces make every state unique — a robot at position (1.000, 2.000) and (1.001, 2.000) are technically different states, both with N=0. SimHash aggregates nearby states into the same bucket.

| Method | Requires exact states? | Noisy-TV safe? | Typical use |
|--------|----------------------|----------------|-------------|
| Count-based (exact) | Yes | Yes | Tabular FrozenLake |
| SimHash | No (bucket counts) | Yes | Moderate-dim continuous obs |
| RND | No | Yes | General deep RL |
| ICM | No | **No** | Deterministic envs only |

**Practical recommendation:** use RND for most Godot tasks. SimHash is worth trying when the observation space is low-to-medium dimensional and you want something simpler than a neural network. Count-based comparison: train FrozenLake Q-Learning with and without `1/sqrt(N)` — measuring how many fewer steps are needed to find the optimal policy is a useful exercise (Stretch goal, Section 9).

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
