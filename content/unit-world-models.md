# World Models — Model-Based RL with DreamerV3

Every algorithm in this course is **model-free**: the agent learns from environment interaction without building an explicit model of how the world works. **World models** change this — the agent learns a compressed latent dynamics model, then plans and imagines inside it. DreamerV3 (Hafner et al. 2023) is the state of the art: one algorithm, one set of hyperparameters, that masters Atari, proprioceptive control, Minecraft, and visual Godot environments.

[← Population-Based Training](unit-pbt.md) · [Course home](index.md)

---

!!! info "Three ways to see your AI"
    TensorBoard (world model loss: reconstruction, dynamics, reward prediction) · latent space t-SNE — visualize what the model has learned to represent · imagined rollout: watch the agent plan inside its own head before acting

---

## 1 · Model-free vs model-based

Every unit so far has been model-free. The agent observes, acts, and receives rewards. No internal representation of "what happens if I do X" — just a learned policy (or Q-function) that maps observations directly to actions.

**Model-based RL adds a world model:**

```
Model-free:  obs → policy → action  (direct mapping)
Model-based: obs → encode → latent z
             latent z + action → predicted z'  (dynamics model)
             use z' to plan or train policy inside imagination
```

**Why bother?** Sample efficiency. If the model is accurate, the agent can do millions of imagined transitions at near-zero cost — no real environment needed. This matters enormously when:

- Each real environment step is expensive (real robot, physics simulation, hours of game time)
- The environment has sparse rewards (imagined trajectories can be dense with predicted rewards)
- You want test-time planning (use the model to look ahead before acting)

---

## 2 · What a world model learns

A world model is trained jointly to predict four things from a latent state `z` and action `a`:

| Component | Input | Output | Loss |
|-----------|-------|--------|------|
| **Encoder** | obs | latent z | — (learned via ELBO) |
| **Dynamics** | z_t, a_t | z_{t+1} | KL divergence |
| **Decoder** | z | reconstructed obs | MSE / CE |
| **Reward predictor** | z | predicted r | MSE |

The decoder reconstructs the original observation — this forces the latent space to capture the information the agent actually needs. The dynamics model predicts the next latent state, enabling imagined rollouts.

```
Real env:   obs_t → encoder → z_t → dynamics(z_t, a_t) → z_{t+1}
                                                           ↓
Imagination:                          z_{t+1} → decoder → reconstructed_obs
                                                 reward_predictor → r̂_{t+1}
```

---

## 3 · The RSSM — Dreamer's dynamics architecture

DreamerV3's core is the **Recurrent State Space Model (RSSM)**. It combines two state components:

| Component | Type | Purpose |
|-----------|------|---------|
| **h_t** (deterministic) | GRU hidden state | Memory across time (no randomness) |
| **z_t** (stochastic) | Categorical or Gaussian | Uncertainty about the current state |

```
h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})    # deterministic recurrent state
z_t ~ p(z | h_t, obs_t)                  # stochastic state from observation (posterior)
ẑ_{t+1} ~ p(z | h_{t+1})                # stochastic state without obs (prior — for imagination)
```

**Why both?** `h_t` provides memory across time (like an LSTM), capturing temporal structure. `z_t` provides stochasticity — the model expresses uncertainty about which exact state it is in. The combination enables imagination: set `obs = None`, use the prior `p(z | h)`, and roll out purely in latent space.

**Why imagination for policy training?** Backpropagating through imagined trajectories gives dense gradient signal even in sparse-reward environments. In a real sparse-reward maze, the agent might go 1,000 real steps with zero reward. In imagination, the reward predictor can signal "almost at the goal" every step.

---

## 4 · DreamerV3 — self-tuning hyperparameters

DreamerV3 (Hafner et al. 2023) added two key improvements that make the algorithm **self-tuning across domains**:

**Symlog transforms:** apply `symlog(x) = sign(x) · log(|x| + 1)` to all inputs and targets. This rescales extreme values without losing sign information — crucial when rewards span orders of magnitude across different environments.

**Free bits:** a minimum KL divergence floor prevents the model from ignoring the stochastic latent `z_t` early in training. The KL loss is `max(1.0, KL(posterior ∥ prior))` — the model is allowed to use up to 1 nat of latent information "for free" before the KL penalty kicks in.

```bash
pip install dreamerv3
```

```python
import dreamerv3
import gymnasium as gym

# DreamerV3 on a standard benchmark
env = gym.make("LunarLander-v2")

config = dreamerv3.configs.defaults.update({
    "logdir": "logs/dreamer_lunarlander",
    "steps": 1_000_000,
    "envs": 4,
})
config = config.update(dreamerv3.configs.small)   # smaller model for fast iteration

agent = dreamerv3.Agent(env.observation_space, env.action_space, config)
dreamerv3.train(agent, env, config)
```

---

## 5 · Godot integration

DreamerV3 accepts standard gym-interface environments. Use the SubViewport pipeline from [Visual Observations](unit-visual-observations.md):

```python
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import dreamerv3

# Use visual observations — DreamerV3 is designed for image obs
env = StableBaselinesGodotEnv(
    env_path="./FlyBy.x86_64",
    n_parallel=1,
    speedup=1,
    use_visual_observations=True,   # enable SubViewport pipeline
)

# DreamerV3 configuration for Godot
config = dreamerv3.configs.defaults.update({
    "logdir": "logs/dreamer_godot",
    "steps": 2_000_000,
    "encoder.mlp_keys": "$^",       # use only image observations
    "encoder.cnn_keys": "image",
    "decoder.mlp_keys": "$^",
    "decoder.cnn_keys": "image",
})
config = config.update(dreamerv3.configs.small)

agent = dreamerv3.Agent(env.observation_space, env.action_space, config)
dreamerv3.train(agent, env, config)
```

**Expected sample efficiency on visual tasks:** DreamerV3 typically matches PPO with 3–10× fewer real environment steps on image-based tasks. The tradeoff: each real step requires training the world model, which is computationally expensive — total wall-clock time may be similar.

---

## 6 · Latent space visualization

After training, visualize what the world model has learned to represent:

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Collect latent states from evaluation rollouts
latent_states = []
obs_labels = []  # e.g., "near_goal", "in_danger", "exploring"

# ... collect during eval loop ...

z_array = np.array(latent_states)
labels = np.array(obs_labels)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
z_2d = tsne.fit_transform(z_array)

plt.figure(figsize=(10, 8))
for label in np.unique(labels):
    mask = labels == label
    plt.scatter(z_2d[mask, 0], z_2d[mask, 1], label=label, alpha=0.6, s=10)
plt.legend()
plt.title("World model latent space (t-SNE)")
plt.savefig("latent_space.png")
```

A well-trained world model shows **semantic clustering**: states with similar task-relevant properties (near goal, in danger, etc.) cluster together in latent space, even if their pixel representations are very different.

---

## 7 · Dreamer vs Dyna — a brief lineage

**Dyna (Sutton 1991)** is the original model-based RL algorithm: learn a simple tabular dynamics model from experience, use it to generate synthetic transitions for additional Q-learning updates. Two lines of pseudocode, provably correct in tabular settings.

**Modern Dreamer** scales Dyna to deep RL with high-dimensional observations: the dynamics model is an RSSM neural network, the synthetic transitions are latent imagination rollouts, and the policy is trained entirely on imagined trajectories via actor-critic.

| | Dyna | DreamerV2/V3 |
|--|------|------------|
| Dynamics model | Tabular or shallow | RSSM (GRU + stochastic latent) |
| Imagination | Single synthetic transition | Multi-step latent rollouts |
| Policy training | Q-learning updates | Actor-critic in latent space |
| Decoder | Not needed | Reconstruction loss (observation space) |
| Scale | Tabular / low-dim | Image observations, complex tasks |

---

## 8 · When world models win — and when they don't

**World models are most useful when:**

- **Simulation is expensive:** each real step costs significant time (robot arm, fluid simulation, a closed-source game). Imagination provides cheap additional experience.
- **Sparse rewards:** the reward predictor can amplify the rare reward signal across imagined trajectories.
- **Test-time planning:** you want the agent to look ahead before acting (search in latent space).

**World models often lose to model-free when:**

- **Complex contact dynamics:** contact-rich manipulation (grasping, dexterous manipulation) is notoriously hard to model accurately. Model errors compound during imagined rollouts and produce unrealistic experiences.
- **You can collect data cheaply:** if Godot runs at 20× speedup with 8 parallel envs, PPO can collect millions of real transitions per hour. World model training overhead may not be worth it.
- **Engineering overhead:** DreamerV3 is a complex system. Debugging model errors (reconstruction failures, dynamics drift) is harder than debugging PPO.

**Rule of thumb:** if PPO with 8 parallel envs doesn't converge in 5M steps, consider world models. If it converges in 500k, you don't need them.

---

## 9 · Stretch goals

- **DreamerV3 on FlyBy visual obs:** use the SubViewport pipeline from Section 5 and run DreamerV3 on FlyBy. Compare sample efficiency (steps to reach `ep_rew_mean = 500`) vs PPO from [Unit 6](unit-06.md).
- **Imagined trajectory visualization:** implement a function that runs N imagined steps from the current latent state and plots the reconstructed observations side-by-side with real observations. How faithful is the world model?
- **Minimal Dyna on FrozenLake:** implement Dyna from scratch on the tabular FrozenLake example from the [Q-Learning unit](unit-q-learning.md). Store `(s, a) → (s', r)` transitions in a table; use them for K extra Q-learning updates per real step. Measure step efficiency vs Q-learning alone.
- **World model ablation:** train DreamerV3 with the decoder disabled (no reconstruction loss — latent space is unconstrained). Compare final performance and latent space quality.

---

## What's next

You have reached the end of the course content. Return to the [Course home](index.md) for the full unit list, or revisit the capstone project to put everything together.

[→ Course home](index.md)

---

[← Population-Based Training](unit-pbt.md) · [Course home](index.md)
