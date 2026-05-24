# Population-Based Training — Hyperparameter AutoML for RL

Hyperparameter tuning in RL is notoriously difficult: performance is **non-stationary** — a learning rate that works well in early training can be harmful later. Optuna (static search, covered in the [JumperHard unit](unit-04.md)) finds a fixed set of hyperparameters before training begins. **Population-Based Training (PBT)** adapts hyperparameters *during* training by running a population of agents simultaneously and copying knowledge between them.

[← Advanced Evaluation](unit-evaluation.md) · [Course home](index.md)

---

!!! info "Three ways to see your AI"
    TensorBoard (hyperparameter trajectories over training steps — watch `learning_rate` evolve per trial) · Ray Tune dashboard (population performance side-by-side) · final policy: compare best PBT agent vs best Optuna agent at equal compute

---

## 1 · Why grid search and Optuna fail for RL

**The non-stationarity problem:**

| Technique | What it finds | Limitation |
|-----------|-------------|-----------|
| Grid/random search | Best fixed hyperparams | Ignores time-dependence |
| Optuna (TPE/SMAC) | Best fixed hyperparams | Same — each trial uses fixed params throughout |
| PBT | Best adaptive hyperparam schedule | Higher compute, more complex |

A policy benefits from **high learning rate early** (fast exploration of parameter space) and **low learning rate late** (stable fine-tuning). A static Optuna search picks one value for the whole run — it finds the best compromise, not the best schedule.

RL compounds the issue: hyperparameters like `ent_coef` (entropy bonus) are useful early to prevent premature convergence but become harmful later once the policy is focused. No fixed value is optimal throughout.

---

## 2 · The PBT algorithm

PBT (Jaderberg et al. 2017, DeepMind) runs N agents in parallel with different random hyperparameters. Every T steps, each agent's performance is evaluated:

```
Population: [agent_1(hp_1), agent_2(hp_2), ..., agent_N(hp_N)]

Every T steps:
  1. Evaluate all agents → scores [s_1, ..., s_N]
  2. EXPLOIT: bottom 20% of agents copy weights from top 20%
  3. EXPLORE: copied agents perturb their hyperparameters (±20%)
  4. Continue training
```

**Exploit:** the worst-performing agents get a fresh start from the best-performing agents' current weights. No wasted compute on failing runs.

**Explore:** the copied hyperparameters are slightly perturbed. This prevents collapse to a single hyperparameter configuration and maintains diversity.

**Key insight:** PBT is not just hyperparameter search — it also **transfers learned weights**. A well-trained agent from the middle of training seeds a new agent with better hyperparameters. This is qualitatively different from static search.

---

## 3 · Ray Tune setup

```bash
pip install "ray[tune]" stable-baselines3
```

```python
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np

ray.init(ignore_reinit_error=True)

# Define the PBT scheduler
pbt_scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="episode_reward_mean",
    mode="max",
    perturbation_interval=5,          # exploit/explore every 5 iterations
    hyperparam_mutations={
        "lr":            lambda: np.random.uniform(1e-5, 1e-3),
        "clip_range":    lambda: np.random.uniform(0.1, 0.4),
        "ent_coef":      lambda: np.random.uniform(0.0, 0.05),
        "n_steps":       [512, 1024, 2048, 4096],
    },
)


def train_ppo(config):
    """Single trial — will be called for each member of the population."""
    env = gym.make("CartPole-v1")

    model = PPO(
        "MlpPolicy", env,
        learning_rate=config["lr"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        n_steps=config["n_steps"],
        verbose=0,
    )

    for iteration in range(100):
        model.learn(total_timesteps=10_000, reset_num_timesteps=False)

        # Evaluate
        rewards = []
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_reward += r
            if terminated or truncated:
                rewards.append(ep_reward)
                ep_reward = 0
                obs, _ = env.reset()

        # Report to Ray Tune (enables PBT decisions)
        tune.report(
            episode_reward_mean=np.mean(rewards) if rewards else 0.0,
            training_iteration=iteration,
            **config,
        )

    env.close()


# Run PBT
analysis = tune.run(
    train_ppo,
    config={
        "lr":         tune.uniform(1e-5, 1e-3),
        "clip_range": tune.uniform(0.1, 0.4),
        "ent_coef":   tune.uniform(0.0, 0.05),
        "n_steps":    tune.choice([512, 1024, 2048]),
    },
    num_samples=8,        # population size
    scheduler=pbt_scheduler,
    resources_per_trial={"cpu": 2},
    stop={"training_iteration": 100},
    local_dir="ray_results/pbt_cartpole",
)

best_config = analysis.get_best_config(metric="episode_reward_mean", mode="max")
print("Best config:", best_config)
```

---

## 4 · Hyperparameter trajectory plots

The most interesting PBT output is not just the final performance — it's **how hyperparameters evolved**:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load trial data from Ray Tune results
df = analysis.trial_dataframes   # dict of trial_id → DataFrame

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, hp in enumerate(["lr", "clip_range", "ent_coef", "n_steps"]):
    ax = axes[i // 2][i % 2]
    for trial_id, trial_df in df.items():
        if hp in trial_df.columns:
            ax.plot(trial_df["training_iteration"], trial_df[hp], alpha=0.5)
    ax.set_title(hp)
    ax.set_xlabel("training iteration")

plt.tight_layout()
plt.savefig("pbt_hyperparameter_trajectories.png")
```

**What to look for:**

- `learning_rate` typically decreases over time as agents settle into fine-tuning — confirms the non-stationarity intuition
- `ent_coef` often starts higher and drops — entropy is most useful early
- Exploited agents (weights copied from top performers) show discontinuous jumps in hyperparameters when perturbed
- Diverse trajectories = healthy exploration; all agents converging to same values = premature collapse

---

## 5 · PBT with Godot environments

For Godot environments, each trial runs a separate Godot instance. This requires careful resource management:

```python
import subprocess
import os

class GodotPBTTrial:
    def __init__(self, env_path, port_base=11000):
        self.env_path = env_path
        self.port_base = port_base

    def get_env(self, trial_id):
        port = self.port_base + trial_id
        return StableBaselinesGodotEnv(
            env_path=self.env_path,
            n_parallel=1,         # 1 env per trial — parallelism comes from population
            speedup=20,
            port=port,            # each trial gets its own port
        )
```

**Resource allocation:** with 8 population members, you need 8 Godot processes running simultaneously. On a 16-core machine: 2 CPUs per trial, speedup=20. Monitor RAM — each Godot instance uses ~500MB.

**Practical tip:** Use `n_parallel=1` per PBT trial and let the population provide parallelism, rather than `n_parallel=8` per trial. The latter requires 64 Godot instances for a population of 8, which is usually too many.

---

## 6 · What to tune with PBT

| Hyperparameter | PBT benefit | Notes |
|----------------|------------|-------|
| `learning_rate` | High — non-stationary | Warm-up then decay is a common adaptation pattern |
| `ent_coef` | High — useful early, harmful late | Often decreases spontaneously under PBT |
| `clip_range` | Medium | More stable throughout training |
| `gamma` | Low | Changing γ mid-training destabilises value estimates |
| `n_steps` | Medium | Affects gradient variance |
| Network architecture | Not recommended | Changing arch requires reinitializing weights |

---

## 7 · PBT vs Optuna — when to use each

| | Optuna | PBT |
|--|--------|-----|
| Search strategy | Sequential Bayesian (TPE) | Parallel evolutionary |
| Hyperparameter schedule | Fixed throughout run | Adaptive per trial |
| Compute model | N trials × M steps | N population × M steps (same total) |
| Best for | Short to medium runs, limited compute | Long runs, non-stationary tasks |
| Implementation complexity | Low | Medium (requires Ray Tune or custom) |
| Reproducibility | Easy | Harder (inherent stochasticity) |

**Practical recommendation:** start with Optuna. If you find that the best Optuna runs use different hyperparameters early vs late in training, PBT will improve results. If the optimal config is stable throughout, Optuna is sufficient.

---

## 8 · Stretch goals

- **Minimal PBT from scratch:** implement the exploit/explore loop with 4 processes and Python `multiprocessing`. No Ray required — each process runs a PPO loop on FrozenLake and reports its score to a shared queue. ~100 lines.
- **PBT on FlyBy:** run PBT with population size 6 on FlyBy (continuous control). Compare best PBT agent to best Optuna agent after 2M steps each.
- **Hyperparameter trajectory analysis:** for each PBT trial, plot `lr` vs `ep_rew_mean` over training. Identify the adaptation pattern — does lr decrease monotonically, or does it oscillate?
- **Population diversity metric:** compute the variance of each hyperparameter across the population at each perturbation step. Low variance = premature convergence. Design a diversity-preserving perturbation strategy.

---

## What's next

World Models take PBT's sample efficiency focus a step further: instead of adapting hyperparameters, the agent learns an internal model of the world and trains entirely inside it.

[→ World Models](unit-world-models.md)

---

[← Advanced Evaluation](unit-evaluation.md) · [Course home](index.md) · [→ World Models](unit-world-models.md)
