# Unit 5 — Parallel Training

Same **BallChase** environment you already know — but now you open the source, add parallel env instances inside a single Godot process, and measure the throughput gain. The new skill here is **scaling**, not a new environment.

[← Unit 4: JumperHard & PPO](unit-04.md) · [Course home](index.md)

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

## 6 · Stretch goals

- **Scale curve** — plot steps/sec vs N envs (1, 2, 4, 8, 16). Where does the gain flatten?
- **Batch size scaling** — when doubling `n_parallel`, also double `--batch_size`. Does it help?
- **Different env** — apply the same parallelism technique to your Lunar Lander from Unit 2

---

## What's next

**Unit 6:** Continuous 3D — FlyBy / HovercraftRacing, continuous action spaces, observation normalization for 3D sensors.

[→ Unit 6: Continuous 3D](unit-06.md)
