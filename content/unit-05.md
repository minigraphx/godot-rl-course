# Unit 5 — Parallel Training

Same **BallChase** environment you already know — but now you open the source, add parallel env instances inside a single Godot process, and measure the throughput gain. The new skill here is **scaling**, not a new environment.

[← Unit 4: JumperHard & PPO](unit-04.md) · [Course home](index.md)

!!! note "Prerequisites"
    - **[Unit 4](unit-04.md)** — a working PPO training run + comfort with TensorBoard
    - **[Unit 2](unit-02.md)** — BallChase source open in the editor at least once
    - Comfort exporting a headless Godot binary (Unit 3 §9)
    - No new Python — this unit only changes the *training scene* and CLI flags

!!! info "Time"
    Reading: ~25 min · Training: ~20 min GPU / ~1 h CPU

---

!!! info "Three ways to see your AI"
    Godot (steps/second counter + viz checkpoint) · TensorBoard (wall-clock time vs ep_rew_mean) · training scene node count

---

## 1 · Why parallelism helps

Every RL update needs a batch of diverse transitions. A single environment generates correlated transitions (consecutive frames from the same episode). Running **N parallel environments** gives N independent trajectories simultaneously:

- **More diversity** → better gradient estimates → faster convergence
- **Higher GPU/CPU utilisation** — the trainer is no longer waiting for a single env
- **Same wall clock, more steps** — N envs don't run N× slower; Godot handles them in one process

The tradeoff: more RAM per env instance, and the training scene gets larger and harder to debug visually.

```
1 env  × 1M steps = 1M transitions, ~60 min
8 envs × 125k steps each = 1M transitions, ~8 min  (approximately)
```

---

## 2 · Open BallChase from source

1. Clone [godot_rl_agents_examples](https://github.com/edbeeching/godot_rl_agents_examples) if you haven't already
2. Open `examples/BallChase` in Godot .NET (not the hub binary this time — you'll edit the training scene)
3. Enable the Godot RL Agents plugin

---

## 3 · Add parallel env instances

Open `training_scene.tscn`. You should already have a `Sync` node and one `BallChase` env root.

**Add instances:**

1. In the scene tree, select the env root node (e.g. `BallChase`)
2. Duplicate it (**Ctrl+D** or right-click → Duplicate) 7 times → you now have 8 instances
3. Spread them out spatially so they don't overlap (select each, move with the transform gizmo)
4. All instances share the same `Sync` node — no extra config needed

**Check the Sync node:**

| Property | Recommended value |
|----------|------------------|
| Control Mode | `TRAINING` |
| Speed Up | `20` |
| Action Repeat | `1` |

Export a new binary (Project → Export) after saving the scene.

---

## 4 · Measure the throughput gain

Run three experiments — 1, 4, and 8 parallel instances — and compare wall-clock time to reach the same `ep_rew_mean`:

```bash
conda activate godot_env
tensorboard --logdir=logs &

# 1 env
gdrl --env_path=./BallChase.x86_64 \
  --experiment_name=ballchase_1env \
  --timesteps=500000 --n_parallel=1 --speedup=20

# 4 envs
gdrl --env_path=./BallChase.x86_64 \
  --experiment_name=ballchase_4env \
  --timesteps=500000 --n_parallel=4 --speedup=20

# 8 envs
gdrl --env_path=./BallChase.x86_64 \
  --experiment_name=ballchase_8env \
  --timesteps=500000 --n_parallel=8 --speedup=20
```

In TensorBoard, switch the x-axis to **wall time** (not steps) to see the real speedup.

!!! check "Done when"
    All three runs launch cleanly, train to 500k steps, and show up as separate experiments in TensorBoard. With the x-axis on **wall time**, the 8-env run reaches any given `ep_rew_mean` level clearly faster than the 1-env run. Expect a large speedup, but the exact ratio depends on your CPU core count and seed — treat the *ordering* (8 ≥ 4 > 1 in throughput) as the check, not any specific multiplier. On a machine with few cores, 8 envs can tie 4 once the cores saturate — that's your hardware ceiling (extra processes add contention, not throughput), not a broken setup.

!!! tip "n_parallel vs in-scene instances"
    `--n_parallel` launches **separate Godot processes**. In-scene instances run inside **one process**. Both increase parallelism; combining them gives maximum throughput. In-scene instances are easier to set up; `--n_parallel` scales better on multi-core machines.

---

## 5 · Eval protocol

Use the same deterministic eval loop from Unit 4 — run 20 episodes with `deterministic=True` and report mean ± std. A properly trained BallChase agent should average > 80 reward at 500k steps with 8 envs.

```python
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import numpy as np

env = StableBaselinesGodotEnv(env_path="./BallChase.x86_64", n_parallel=1, speedup=1)
model = PPO.load("logs/sb3/ballchase_8env/best_model")

rewards = []
for _ in range(20):
    obs, done, total = env.reset(), False, 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _ = env.step(action)
        total += r
    rewards.append(total)

print(f"Mean ± std: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
env.close()
```

**Viz checkpoint** — replay one eval episode with `show_window=True`. Confirm the agent chases the ball reliably.

---

## 6 · Why one seed is never enough

RL training has high variance between random seeds. Two runs with identical hyperparameters and the same number of timesteps — but different random seeds — can produce `ep_rew_mean` values that differ by 50% or more at convergence.

A single training run that "works" might be a lucky seed. A run that "fails" might be an unlucky one. If you tune hyperparameters on one seed, you may be optimising for randomness rather than algorithm quality.

**Standard practice:** run N = 3–5 seeds, report **mean ± std across seeds** (not the mean and std within a single run).

```python
import subprocess
import numpy as np
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

seeds = [0, 1, 2, 3, 4]
final_rewards = []

for seed in seeds:
    # Train with this seed
    subprocess.run([
        "gdrl",
        "--env_path=./BallChase.x86_64",
        f"--experiment_name=ballchase_seed{seed}",
        "--timesteps=500000",
        "--n_parallel=8",
        "--speedup=20",
        f"--seed={seed}",
    ])

    # Evaluate the trained model
    env = StableBaselinesGodotEnv(env_path="./BallChase.x86_64", n_parallel=1, speedup=1)
    model = PPO.load(f"logs/sb3/ballchase_seed{seed}/best_model")

    ep_rewards = []
    for _ in range(20):
        obs, done, total = env.reset(), False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _ = env.step(action)
            total += r
        ep_rewards.append(total)

    final_rewards.append(np.mean(ep_rewards))
    env.close()

print(f"Mean across seeds: {np.mean(final_rewards):.1f} ± {np.std(final_rewards):.1f}")
```

!!! tip "How many seeds?"
    For a course project: 1 seed is fine — you're learning, not publishing.
    For a comparison between two methods: 3–5 seeds minimum.
    For a paper or production decision: 10 seeds.

**Comparing two algorithms properly (PPO vs SAC):**

Use the *same* seeds for both algorithms, then compare mean ± std across those seeds:

```python
# WRONG: PPO on seeds [0,1,2], SAC on seeds [3,4,5]
# The seed sets are different — any difference might be due to seed luck

# CORRECT: both algorithms trained on seeds [0, 1, 2, 3, 4]
ppo_rewards = [train_and_eval("PPO", seed) for seed in seeds]
sac_rewards  = [train_and_eval("SAC",  seed) for seed in seeds]

print(f"PPO: {np.mean(ppo_rewards):.1f} ± {np.std(ppo_rewards):.1f}")
print(f"SAC: {np.mean(sac_rewards):.1f}  ± {np.std(sac_rewards):.1f}")
```

Paired seeds control for environment randomness — if PPO beats SAC on seed 0, 1, 2, 3, and 4, that's a much stronger conclusion than a single-seed comparison.

In TensorBoard, use the shaded area view (IQM or mean ± std) to visualise multi-seed results. The width of the shaded band tells you more about algorithm stability than the center line alone.

---

## 7 · Stretch goals (pick one)

- **Scale curve** — plot steps/sec vs N envs (1, 2, 4, 8, 16). Where does the gain flatten?
- **Batch size scaling** — when doubling `n_parallel`, also double `--batch_size`. Does it help?
- **Different env** — apply the same parallelism technique to your Lunar Lander from Unit 2

---

## What's next

**Unit 6:** Continuous 3D — FlyBy / HovercraftRacing, continuous action spaces, observation normalization for 3D sensors.

!!! info "Self-check before you move on"
    Can you answer these in your own words?

    1. Why do N parallel envs give *better* gradient estimates, not just *more* gradient estimates?
    2. What's the difference between in-process parallel envs (the training scene) and `--n_parallel` subprocesses, and when would you use each?
    3. If steps/sec scales linearly to 8 envs but flattens at 16, what is the bottleneck likely to be?
    4. Why is a single-seed reward curve never enough evidence that a change "helped"?
    5. Why does doubling `n_parallel` often need a matching bump to `batch_size`?

    If you can answer all five — you're ready.

??? success "Self-check answers"
    1. A single env emits **correlated transitions** — consecutive frames from one episode. N parallel envs sample N independent trajectories at the same time, so each batch covers more of the state space; it's that diversity, not just the extra volume, that sharpens the gradient estimate.
    2. In-scene instances are duplicated env roots inside **one Godot process** — quick to set up, all sharing the same `Sync` node. `--n_parallel` launches **separate Godot processes**, which scales better on multi-core machines. Use in-scene instances for convenience, `--n_parallel` (or both combined) for maximum throughput.
    3. A shared hardware resource — most likely **saturated CPU cores** (16 envs now contend for fewer physical cores), with RAM per instance and the single trainer process as the next suspects. Past that point, extra envs add contention, not throughput.
    4. **Seed variance** in RL is huge — identical hyperparameters can land 50% apart in `ep_rew_mean` at convergence, so a single "improved" curve may just be a lucky seed. Only mean ± std across 3–5 paired seeds separates a real effect from randomness.
    5. Doubling `n_parallel` doubles the rollout buffer; with `batch_size` fixed, PPO compensates by running **twice as many small updates** per epoch — each gradient still averages over only a thin slice of the now-more-diverse buffer, and the longer chain of small updates lets the policy drift further per rollout. Bumping `batch_size` in step keeps the update count stable and lets each gradient actually average over the extra diversity.

[→ Unit 6: Continuous 3D](unit-06.md)

!!! tip "If parallel training feels slow or unstable"
    See the [Debugging RL Training guide](unit-debugging.md#9-performance-slow-training) for throughput bottlenecks, subprocess launch issues, and VecEnv misconfigurations.
