# Population-Based Training — Hyperparameter AutoML for RL

!!! info "Time"
    Reading: ~30 min · Training: ~45 min GPU / ~3 h CPU

[Course home](index.md)

---

!!! info "Three ways to see your AI"
    - **Ray Tune dashboard** (`ray[tune]` spins up a local web UI at `http://localhost:8265`) — watch each trial's reward curve in real time, see which hyperparameter configurations are alive vs. terminated early.
    - **TensorBoard** — compare the best PBT trial against your Optuna baseline with identical step budgets; the gap is your PBT dividend.
    - **Hyperparameter trajectory plot** — a line chart per hyperparameter showing how `learning_rate`, `ent_coef`, etc. evolve across training for each agent in the population. Static search methods produce flat lines; PBT produces dynamic curves that adapt.

---

## 1 · Why grid search fails for RL

Standard hyperparameter optimisation asks one question: **given a fixed budget of time, which static configuration performs best?** Grid search, random search, and Optuna all share this assumption. You pick a set of hyperparameters, run training to completion, measure the final score, and move on to the next configuration.

For supervised learning this works well. A learning rate of `1e-3` that reaches 94% accuracy in 100 epochs will reach roughly the same accuracy whether you evaluate at epoch 1, epoch 50, or epoch 100. The hyperparameters are stable features of a configuration.

RL breaks this assumption in three distinct ways.

**Non-stationary performance.** RL training is not monotone. An agent trained with `learning_rate=1e-3` might be ahead of `learning_rate=3e-4` at 200k steps but behind it at 1M steps because the larger learning rate destabilises later updates. Evaluating all configurations at the same final timestep misses this — the "winner" at step 1M may not have been the winner at any earlier checkpoint.

**Interaction between hyperparameters and training stage.** PPO's `clip_range` exists to prevent updates that are too large. Early in training, large updates are often beneficial — the policy is random and needs to move fast. Late in training, large updates cause instability. The ideal `clip_range` is not a constant: it should start permissive and tighten as the policy matures. Grid search picks one value and lives with it for the entire run.

**Combinatorial explosion.** A modest PPO search over `learning_rate` (5 values) × `ent_coef` (4 values) × `gamma` (3 values) × `n_steps` (3 values) × `clip_range` (3 values) produces 540 configurations. At 1M steps each on a single machine that is 540M environment steps — days of wall-clock time before you see a single result from the full grid. Random search and Bayesian optimisation reduce the number of evaluations, but they still commit each run to a static configuration for its entire lifetime.

The core problem: **the optimal hyperparameters for an RL agent change during training**, and any method that treats a configuration as fixed is leaving performance on the table.

---

## 2 · Optuna recap — static search and its ceiling

You have likely seen Optuna used for hyperparameter search: define a search space, let Optuna sample configurations with TPE (Tree-structured Parzen Estimator), and use early stopping to kill bad runs before they waste the full budget.

```python
import optuna
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)

    env = StableBaselinesGodotEnv(env_path="./FlyBy.x86_64", n_parallel=4, speedup=20)
    model = PPO(
        "MlpPolicy", env, verbose=0,
        learning_rate=lr,
        ent_coef=ent_coef,
        clip_range=clip_range,
    )
    model.learn(total_timesteps=500_000)
    mean_reward = evaluate_policy(model, env, n_eval_episodes=10)[0]
    env.close()
    return mean_reward

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(study.best_params)
```

Optuna is excellent and is the right tool for many RL problems. Its limitation is precisely what we just described: the hyperparameters `lr`, `ent_coef`, and `clip_range` are sampled once and held constant for all 500k training steps. Optuna finds the best *static* configuration. It cannot discover that `lr=1e-3` for the first 200k steps followed by `lr=1e-4` for the next 300k is better than either value alone.

!!! tip "Optuna is not obsolete"
    PBT does not replace Optuna — it addresses a different use case. The comparison in Section 8 will be explicit about when each tool wins. For most Godot tasks with cheap simulators, Optuna with 30–50 trials is the right starting point. PBT earns its overhead when training runs are long (> 2M steps) and you have the hardware to run a population in parallel.

---

## 3 · The PBT algorithm

Population-Based Training (Jaderberg et al., DeepMind, 2017) runs **N agents simultaneously**, each with its own hyperparameter configuration and policy weights. Periodically — every `T` steps — the population is evaluated and two operations are applied.

### The two operations: exploit and explore

**Exploit** — copy weights from a top performer.

Each agent is ranked by its recent reward. The bottom 20–25% of the population look at the top 20–25% and copy their neural network weights directly. The losing agent does not continue with its old weights — it inherits the winner's weights and picks up from there.

This is the key mechanism that makes PBT fundamentally different from independent parallel runs: agents can *inherit* learning progress from better-performing peers.

**Explore** — perturb the inherited hyperparameters.

After copying weights, the agent perturbs the inherited hyperparameters by multiplying each by a random factor drawn from `{0.8, 1.2}` (a common default), then clamps values back to the search space bounds. The new agent continues training with the same weights but modified hyperparameters.

This ensures that the population never collapses to a single configuration: every exploitation step is followed immediately by exploration.

### Synchronous vs asynchronous PBT

| Variant | How it works | Trade-off |
|---------|-------------|-----------|
| **Synchronous** | All agents pause at step T; exploit/explore happens; all resume | Clean comparisons; GPU sits idle during synchronisation |
| **Asynchronous** | Each agent triggers exploit/explore independently when it reaches T steps | Better hardware utilisation; comparison is approximate (not all at the same step) |

Ray Tune's `PopulationBasedTraining` scheduler uses the asynchronous variant by default, which is why it is well-suited to situations where agents have different wall-clock speeds.

### A concrete example with N=4

```
Initial population (step 0):
  Agent A: lr=1e-3,  ent=0.02,  reward=—
  Agent B: lr=3e-4,  ent=0.01,  reward=—
  Agent C: lr=5e-4,  ent=0.05,  reward=—
  Agent D: lr=2e-4,  ent=0.005, reward=—

After T=200k steps, evaluate:
  Agent A: reward=320  ← top 25%
  Agent B: reward=280
  Agent C: reward=175  ← bottom 25%
  Agent D: reward=260

Exploit: Agent C copies Agent A's weights.
Explore: Agent C perturbs lr: 1e-3 × 1.2 = 1.2e-3, ent: 0.02 × 0.8 = 0.016

New population:
  Agent A: lr=1e-3,   ent=0.02   (unchanged)
  Agent B: lr=3e-4,   ent=0.01   (unchanged)
  Agent C: lr=1.2e-3, ent=0.016  (weights from A, hyperparams perturbed)
  Agent D: lr=2e-4,   ent=0.005  (unchanged)
```

After another `T` steps, the cycle repeats. Winning hyperparameter regions get more exploration around them; losing regions are abandoned.

### Population size N

N = 4 is the minimum viable population for meaningful selection (at least one winner and one loser per cycle). N = 8 is the practical sweet spot that balances diversity against compute cost. N = 16+ is for large-scale experiments where you have a cluster available.

!!! warning "N=2 is not PBT"
    With only two agents, every cycle kills the loser and replaces it with a copy of the winner. This is equivalent to restart-with-perturbation, not population selection. Meaningful diversity emerges from N ≥ 4.

---

## 4 · Ray Tune integration

Ray Tune is the standard Python library for distributed hyperparameter search and PBT. It wraps any training function, manages trials, and handles early stopping.

### Install

```bash
pip install "ray[tune]"
```

If you are on a machine with multiple GPUs or want the full dashboard:

```bash
pip install "ray[tune]" tensorboard
```

### Wiring SB3 into Ray Tune

Ray Tune expects a training function with signature `train_fn(config: dict)`. The `config` dict holds the hyperparameters for this trial. The function should call `ray.air.session.report(metrics)` periodically so Ray can compare trials and schedule exploit/explore.

```python
import ray
from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import numpy as np


def train_pbt(config: dict):
    """One PBT trial. Ray Tune calls this in a separate process per agent."""

    env = StableBaselinesGodotEnv(
        env_path="./FlyBy.x86_64",
        n_parallel=4,
        speedup=20,
        show_window=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=config["learning_rate"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        gamma=config["gamma"],
        n_steps=config["n_steps"],
    )

    # If Ray Tune passed checkpoint weights from an exploit step, load them.
    checkpoint = air.session.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            model.set_parameters(f"{checkpoint_dir}/model")

    # Train for one reporting interval, then report metrics.
    STEPS_PER_REPORT = 50_000
    for _ in range(10):  # 10 × 50k = 500k total steps per trial
        model.learn(total_timesteps=STEPS_PER_REPORT, reset_num_timesteps=False)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, warn=False)

        # Save weights so Ray can copy them during exploit.
        with tune.checkpoint_dir(step=model.num_timesteps) as checkpoint_dir:
            model.save(f"{checkpoint_dir}/model")

        air.session.report(
            {"mean_reward": mean_reward, "timesteps": model.num_timesteps},
        )

    env.close()
```

### The PBT scheduler

```python
pbt_scheduler = PopulationBasedTraining(
    time_attr="timesteps",               # the metric Ray uses to sync the population
    metric="mean_reward",                # what to maximise
    mode="max",
    perturbation_interval=100_000,       # exploit/explore every 100k steps
    hyperparam_mutations={
        # Ray will perturb these by ×0.8 or ×1.2
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "clip_range":    tune.uniform(0.1, 0.4),
        "ent_coef":      tune.loguniform(1e-4, 0.1),
        "gamma":         tune.uniform(0.95, 0.999),
        "n_steps":       [512, 1024, 2048],
    },
)
```

### Running the population

```python
ray.init()

initial_config = {
    "learning_rate": tune.loguniform(1e-5, 1e-3),
    "clip_range":    tune.uniform(0.1, 0.4),
    "ent_coef":      tune.loguniform(1e-4, 0.1),
    "gamma":         tune.uniform(0.95, 0.999),
    "n_steps":       tune.choice([512, 1024, 2048]),
}

tuner = tune.Tuner(
    train_pbt,
    tune_config=tune.TuneConfig(
        scheduler=pbt_scheduler,
        num_samples=8,          # population size N=8
    ),
    param_space=initial_config,
    run_config=air.RunConfig(
        name="pbt_flyby",
        local_dir="./ray_results",
        stop={"timesteps": 500_000},
    ),
)

results = tuner.fit()
best = results.get_best_result(metric="mean_reward", mode="max")
print("Best config:", best.config)
print("Best reward:", best.metrics["mean_reward"])
```

!!! check "Done when"
    `tuner.fit()` runs all eight trials to the 500k-step stop — watch them appear and report `mean_reward` in the Ray Tune dashboard at `http://localhost:8265` — and prints a best config and reward at the end. The real success signal is the trajectory plot from Section 7: the hyperparameter lines show vertical steps at perturbation events, and the surviving trials drift toward lower `learning_rate` as training progresses. If every trajectory is a flat line with no jumps, the exploit step never fired — check `perturbation_interval` against your total step budget and the `metric`/`mode` settings, as described in Section 7.

### Search space for PPO and SAC

| Hyperparameter | PPO range | SAC range | Notes |
|----------------|-----------|-----------|-------|
| `learning_rate` | `[1e-5, 1e-3]` log-uniform | `[1e-5, 1e-3]` log-uniform | Most impactful parameter in both algorithms |
| `clip_range` | `[0.1, 0.4]` | N/A | PPO-specific; controls policy update size |
| `ent_coef` | `[1e-4, 0.1]` log-uniform | use `"auto"` | SAC auto-tunes this; include for PPO |
| `gamma` | `[0.95, 0.999]` | `[0.95, 0.999]` | Very task-dependent; near 1.0 for long horizons |
| `n_steps` | `{512, 1024, 2048}` | N/A | PPO rollout length; discrete set works better than continuous |
| `batch_size` | `{64, 128, 256}` | `{128, 256, 512}` | GPU-dependent; often not worth tuning first |
| `tau` | N/A | `[0.001, 0.05]` | SAC target network update rate |

!!! tip "Start with fewer parameters"
    In practice, mutating more than 3–4 hyperparameters simultaneously slows convergence — the explore step takes the agent too far from a working configuration. Start with `learning_rate` + `ent_coef` + `clip_range` (PPO) or `learning_rate` + `gamma` (SAC), and add more only if the initial run plateaus.

---

## 5 · Godot + PBT — running multiple instances

PBT requires running N agents simultaneously, each with its own Godot environment. This means N separate Godot processes on your machine (or across a cluster).

### Architecture overview

```
Ray Tune orchestrator (Python)
    │
    ├── Trial 0 ──► PPO process ──► StableBaselinesGodotEnv ──► FlyBy.x86_64 (port 11000)
    ├── Trial 1 ──► PPO process ──► StableBaselinesGodotEnv ──► FlyBy.x86_64 (port 11001)
    ├── Trial 2 ──► PPO process ──► StableBaselinesGodotEnv ──► FlyBy.x86_64 (port 11002)
    ├── Trial 3 ──► PPO process ──► StableBaselinesGodotEnv ──► FlyBy.x86_64 (port 11003)
    ├── Trial 4 ──► PPO process ──► StableBaselinesGodotEnv ──► FlyBy.x86_64 (port 11004)
    ├── Trial 5 ──► PPO process ──► StableBaselinesGodotEnv ──► FlyBy.x86_64 (port 11005)
    ├── Trial 6 ──► PPO process ──► StableBaselinesGodotEnv ──► FlyBy.x86_64 (port 11006)
    └── Trial 7 ──► PPO process ──► StableBaselinesGodotEnv ──► FlyBy.x86_64 (port 11007)
```

Each Godot process listens on a unique port. The `StableBaselinesGodotEnv` wrapper accepts a `port` argument; Ray Tune's trial index provides a natural unique ID.

```python
def train_pbt(config: dict):
    # Use Ray's trial ID to assign a unique port.
    trial_id = int(tune.get_trial_id().split("_")[-1])
    port = 11000 + trial_id

    env = StableBaselinesGodotEnv(
        env_path="./FlyBy.x86_64",
        port=port,
        n_parallel=2,       # 2 parallel envs per trial — keep total manageable
        speedup=20,
        show_window=False,
    )
    # ... rest of training function
```

### Resource allocation

Every Godot process consumes CPU. With N=8 trials and 2 parallel envs per trial, you are running 16 Godot processes simultaneously. On a 16-core machine this saturates CPU. Use Ray's resource hints to prevent oversubscription:

```python
tuner = tune.Tuner(
    tune.with_resources(train_pbt, resources={"cpu": 2}),   # 2 cores per trial
    tune_config=tune.TuneConfig(
        scheduler=pbt_scheduler,
        num_samples=8,
    ),
    # ...
)
```

!!! warning "GPU memory with PBT"
    N=8 PPO models in memory simultaneously can exhaust GPU VRAM on smaller cards. PPO's MLP policy is small (< 50 MB each), but N=8 adds up. If you see CUDA out-of-memory errors, reduce `num_samples` to 4–6, or run trials on CPU (`device="cpu"` in the PPO constructor). PBT's benefit comes from the selection mechanism, not GPU throughput.

### Headless Godot on Linux

All Godot instances should run headless (no display) to avoid fighting over the screen. Export your project with the `--headless` flag:

```bash
./FlyBy.x86_64 --headless --port 11000 &
```

The `StableBaselinesGodotEnv` wrapper handles this automatically when `show_window=False`.

---

## 6 · What to tune with PBT

Not all hyperparameters are equally worth including in the PBT mutation set. The table below reflects typical experience across Godot environments with PPO.

| Hyperparameter | Mutation priority | Why |
|----------------|-------------------|-----|
| `learning_rate` | **High** | The single most impactful parameter. The optimal value genuinely changes during training — high early, lower late. |
| `ent_coef` | **High** | Controls exploration pressure. High early in training drives exploration; lower later as the policy converges. PBT naturally discovers this schedule. |
| `clip_range` | **Medium** | Larger values allow faster learning early; smaller values give stability late. Adapting this over training is a known PBT win. |
| `gamma` | **Medium** | Task-dependent. If episodes have highly variable lengths, gamma affects credit assignment significantly. |
| `n_steps` | **Low** | A discrete choice that sets rollout length. Changing it mid-training can disrupt the advantage estimator. Safer to keep fixed unless you have a specific reason. |
| `batch_size` | **Low** | Hardware-constrained more than task-constrained. Usually not worth mutating. |
| `n_epochs` | **Low** | Rarely beneficial to change after initial tuning. |

### Practical starting point

For most Godot tasks, start with this minimal mutation set:

```python
hyperparam_mutations={
    "learning_rate": tune.loguniform(1e-5, 1e-3),
    "ent_coef":      tune.loguniform(1e-4, 0.1),
    "clip_range":    tune.uniform(0.1, 0.4),
}
```

Add `gamma` if the task involves long episodes (> 1000 steps) or if you are seeing poor credit assignment (the agent learns what to do at the end of episodes but not the beginning).

---

## 7 · Results analysis — hyperparameter trajectory plots

The defining visualisation for PBT is the **hyperparameter trajectory**: a line per agent showing how each hyperparameter evolves over training. Static methods produce flat horizontal lines; PBT produces dynamic curves.

### Extracting trajectories from Ray Tune results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the experiment results directory produced by Ray Tune
results_df = results.get_dataframe()

# PBT logs hyperparameter values alongside metrics at each reporting step
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for trial_id in results_df["trial_id"].unique():
    trial_data = results_df[results_df["trial_id"] == trial_id]

    axes[0].plot(trial_data["timesteps"], trial_data["config/learning_rate"], alpha=0.7)
    axes[1].plot(trial_data["timesteps"], trial_data["config/ent_coef"], alpha=0.7)
    axes[2].plot(trial_data["timesteps"], trial_data["config/clip_range"], alpha=0.7)

axes[0].set_ylabel("learning_rate")
axes[0].set_yscale("log")
axes[1].set_ylabel("ent_coef")
axes[1].set_yscale("log")
axes[2].set_ylabel("clip_range")
axes[2].set_xlabel("Environment steps")

fig.suptitle("PBT Hyperparameter Trajectories — FlyBy PPO (N=8)")
plt.tight_layout()
plt.savefig("pbt_trajectories.png", dpi=150)
```

### What to look for in the trajectory plot

A well-functioning PBT run shows several characteristic patterns:

- **Learning rate decay:** agents that survive longest tend to have `learning_rate` that was perturbed downward over training. This confirms PBT is discovering the well-known "start high, end low" learning rate schedule automatically, without you specifying it.
- **Entropy coef convergence:** `ent_coef` trajectories often converge toward a common region as training progresses — the survivors have discovered the right exploration level for the task.
- **Exploit events as discontinuities:** when an agent copies weights from a top performer, its hyperparameter values jump discontinuously. These jumps are visible as vertical steps in the trajectory lines and indicate that the exploit-explore mechanism is firing.
- **Pruned trials:** agents that were killed early (via Successive Halving or other mechanisms) appear as short lines that terminate before the end of training.

If all trajectories are flat and never show discontinuities, the exploit step is not triggering — check that `perturbation_interval` is small enough relative to your total training budget and that `metric` and `mode` are set correctly.

---

## 8 · Comparison: PBT vs Optuna

This is the most practically important section in the unit. The honest answer is that each method wins in different circumstances.

### Head-to-head comparison

| Dimension | Optuna | PBT |
|-----------|--------|-----|
| **Hyperparameter schedule** | Static — one value for the entire run | Dynamic — adapts throughout training |
| **Parallelism** | Embarrassingly parallel across trials | Requires communication between trials (exploit step) |
| **Overhead** | Low — standard Python, no Ray dependency | Higher — Ray cluster setup, inter-process communication |
| **Minimum useful runs** | 10–20 trials | 4–8 agents (N ≥ 4) |
| **Best for** | Short-to-medium runs (< 1M steps), fast simulators | Long runs (> 2M steps), expensive simulators |
| **Result interpretability** | Clear: best static config + importance scores | Complex: schedules, not a single config |
| **Re-usability** | Best config is fixed, easy to apply to new runs | Schedules do not transfer — must re-run PBT |
| **Early stopping** | Pruners (Median, Hyperband) kill bad trials early | Bottom 25% replaced by exploit — inherent early stopping |
| **Implementation effort** | Low — 20 lines of Optuna code | Medium — Ray Tune setup + reporting loop |

### When Optuna wins

- **Godot environments with fast simulation** (BallChase, CrossTheRoad, JumperHard running at 20× speed-up). These complete 1M steps in minutes. Optuna's 30–50 trials finish in under two hours, and the overhead of PBT's exploit/explore mechanism is not justified.
- **You want a transferable configuration.** Optuna gives you a single dict of hyperparameters you can paste into any training script. PBT gives you a schedule that was learned for one specific task and must be relearned for each new environment.
- **Limited hardware.** Running 8 parallel Godot instances requires a machine with at least 8 free cores and enough RAM for all processes simultaneously. A laptop with 4 cores is not a good PBT machine.
- **Diagnostic phase of a new environment.** When you are still figuring out whether a reward function works at all, Optuna's sequential trials are easier to debug. PBT's parallelism makes it harder to isolate which configuration caused a specific behaviour.

### When PBT wins

- **Long training runs (> 2M steps)**. The dynamic scheduling advantage compounds over long runs. At 500k steps the gap between static and dynamic hyperparameters is small; at 5M steps it can be substantial.
- **Expensive simulators where re-runs hurt.** If one training run takes 6 hours, running 30 Optuna trials is prohibitive. PBT extracts more signal from a single parallel run.
- **Tasks where the optimal learning rate is known to decay.** MuJoCo continuous control benchmarks show consistent PBT wins precisely because learning rate decay is well-motivated and PBT discovers it automatically.
- **You have Ray infrastructure already.** If your team uses Ray for other purposes (distributed data processing, RLlib), integrating PBT has near-zero additional overhead.

### Practical recommendation

!!! tip "The decision rule"
    **Start with Optuna.** Run 30–50 trials with the Optuna snippet from Section 2. This gives you a solid static baseline in a few hours on any modern laptop. If that baseline is good enough for your task, stop — you are done.

    **Graduate to PBT when** either (a) your training run is longer than 2M steps *and* you have hardware for N ≥ 4 parallel processes, or (b) you have already found a good static config with Optuna and want to squeeze out the last few percent of performance with adaptive scheduling.

    Do not skip Optuna and jump straight to PBT. The Optuna baseline is both useful (you learn what range of hyperparameters works) and necessary (PBT's initial population should be seeded from a sensible range, not a blind prior).

---

## 9 · Stretch goals

- **Reproduce the learning rate decay.** Run PBT on FlyBy with `learning_rate` as the only mutated parameter. Plot the trajectories for the top 3 surviving agents. Confirm that survivors end training with lower learning rates than they started with. Compare their final reward to an Optuna run with the best static `learning_rate` from the same search range.

- **Seed PBT from Optuna results.** Run 20 Optuna trials first. Use the top-5 configs as the initial PBT population (instead of sampling randomly from the prior). Does warm-starting the population speed up PBT convergence? Measure time-to-threshold (steps until `mean_reward > target`) for cold-start PBT vs Optuna-seeded PBT.

- **PBT on SAC.** Replace PPO with SAC in the `train_pbt` function. Mutate `learning_rate` and `gamma` (SAC's `ent_coef` is auto-tuned and should be left as `"auto"`). The key difference: SAC with a replay buffer does not reset cleanly between exploit steps — the buffer contains transitions from the old hyperparameter regime. Observe whether this causes instability, and if so, try clearing the replay buffer after each exploit step.

- **Read the original PBT paper.** Jaderberg et al. 2017, "Population Based Training of Neural Networks." Freely available on arXiv. Section 3 (the algorithm) and Section 4.2 (the Atari results) are the most relevant to this course. The headline result — PBT on Atari matches the best hand-tuned hyperparameter schedule — is what motivated the method.

- **Compare wall-clock time, not just steps.** Run Optuna (30 trials, 500k steps each, sequential) and PBT (N=8, 500k steps per agent, parallel) and time both with `time.time()`. On a machine where all 8 PBT agents fit in parallel, PBT should finish in roughly the same wall-clock time as 8 Optuna trials — not 30. Calculate the wall-clock efficiency ratio.

---

## What's next

Population-Based Training is the top of the hyperparameter optimisation stack covered in this course. You now have three levels of tool: manual tuning (trial and error), static search (Optuna), and dynamic adaptive search (PBT). The right level depends on your task, your hardware, and how much the hyperparameter schedule matters for your specific problem.

For most Godot projects, Optuna with 30 trials is the practical ceiling — fast to run, easy to interpret, and portable across environments. Reserve PBT for long runs where the dynamic scheduling dividend is large enough to justify the setup cost.

[Course home](index.md)
