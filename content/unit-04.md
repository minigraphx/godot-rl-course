# Unit 4 — JumperHard & PPO Benchmarking

Train the **JumperHard** example — a 3D jumping robot that serves as a standard PPO benchmark in the godot-rl-agents repo. Focus: reading PPO hyperparameters, tuning them, and knowing when training has genuinely solved the task.

[← Unit 3: CrossTheRoad & DQN](unit-03.md) · [Course home](index.md)

---

!!! info "Three ways to see your AI"
    Godot (viz checkpoint after training) · TensorBoard (track against your Unit 2 lander baseline) · `AIController` hyperparameter changes

---

## 1 · What JumperHard teaches

JumperHard is harder than the lander for two reasons:

1. **3D physics** — the robot must balance while jumping, making the state space larger and the reward landscape rougher
2. **Sparse at the margin** — a bonus only fires on hard jumps, so the agent needs dense shaped rewards *and* occasional sparse bonuses to learn

This makes it a good benchmark: if your hyperparameters can solve JumperHard, they will generalise well.

---

## 2 · PPO hyperparameters

These are the knobs you turn. Defaults work for most tasks; you only need to change one at a time.

| Parameter | Default | Effect | When to change |
|-----------|---------|--------|----------------|
| `--learning_rate` | 0.0003 | Step size of each gradient update | Lower if reward oscillates; raise if learning is very slow |
| `--n_steps` | 64 | Steps per env collected before each update | Raise (256–2048) for longer episodes; needs more memory |
| `--batch_size` | 64 | Mini-batch size (must divide `n_steps × n_envs`) | Match to `n_steps`; larger = more stable gradients |
| `--clip_range` | 0.2 | Max policy change per update (PPO trust region) | Lower (0.1) if policy gradient loss explodes |
| `--ent_coef` | 0.0001 | Entropy bonus — encourages exploration | Raise (0.01) if agent converges too early to a suboptimal policy |
| `--n_epochs` | 10 | How many times to re-use each rollout | Lower if `approx_kl` grows large |

!!! warning "Training stalled?"
    Check in order: (1) reward sign and scale, (2) sparse rewards — is there a shaped component? (3) `approx_kl` > 0.02 → reduce `--clip_range` or `--learning_rate`, (4) `ep_rew_mean` flat after 500k steps → raise `--n_steps` and `--ent_coef`.

---

## 3 · Open JumperHard

1. From [examples/JumperHard](https://github.com/edbeeching/godot_rl_agents_examples/tree/main/examples/JumperHard), open the project in Godot .NET
2. Enable the Godot RL Agents plugin
3. Read `ai_controller.gd`:
    - **`get_obs()`** — body position, velocity, nearby platform distances (RayCast3D sensors)
    - **`get_action_space()`** — continuous forces on joints, or discrete jump/move
    - **Reward** — forward progress + jump bonus + survival time
4. Export a headless binary (Project → Export → Linux/Windows/macOS)

---

## 4 · Baseline run

Run with defaults first. This gives you a reference curve to beat:

```bash
conda activate godot_env
tensorboard --logdir=logs &

gdrl --env_path=./JumperHard.x86_64 \
  --experiment_name=jumper_baseline \
  --timesteps=1_000_000 \
  --n_parallel=8 \
  --speedup=20
```

Target: `ep_rew_mean` should climb steadily and stabilise above 150–200 by 1M steps.

---

## 5 · Hyperparameter sweep (one change at a time)

Run three experiments varying one parameter each. Use distinct `--experiment_name` values so TensorBoard overlays them.

```bash
# Experiment A — larger rollout buffer
gdrl --env_path=./JumperHard.x86_64 \
  --experiment_name=jumper_nsteps512 \
  --n_steps=512 --batch_size=256 \
  --timesteps=1_000_000 --n_parallel=8 --speedup=20

# Experiment B — more exploration
gdrl --env_path=./JumperHard.x86_64 \
  --experiment_name=jumper_entropy \
  --ent_coef=0.01 \
  --timesteps=1_000_000 --n_parallel=8 --speedup=20

# Experiment C — tighter trust region
gdrl --env_path=./JumperHard.x86_64 \
  --experiment_name=jumper_clip01 \
  --clip_range=0.1 \
  --timesteps=1_000_000 --n_parallel=8 --speedup=20
```

In TensorBoard, compare `rollout/ep_rew_mean` and `train/approx_kl` across the four runs.

---

## 6 · Eval protocol

Once you have a trained model, evaluate it properly — don't just read off the training curve.

```python
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import numpy as np

env = StableBaselinesGodotEnv(
    env_path="./JumperHard.x86_64",
    n_parallel=1,
    speedup=1,
    show_window=True,       # viz checkpoint
)
model = PPO.load("logs/sb3/jumper_baseline/best_model")

episode_rewards = []
for _ in range(20):                 # 20 fixed episodes
    obs = env.reset()
    done, total = False, 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)   # no exploration
        obs, reward, done, _ = env.step(action)
        total += reward
    episode_rewards.append(total)

print(f"Mean: {np.mean(episode_rewards):.1f}  Std: {np.std(episode_rewards):.1f}")
env.close()
```

!!! info "Why deterministic=True?"
    During training, the policy samples actions stochastically. During evaluation you want the **best** action at every step — `deterministic=True` picks the action with highest probability instead of sampling. Always use it for benchmarking.

**Viz checkpoint** — run the eval script with `show_window=True` and watch 3–5 episodes. Confirm behavior matches the `ep_rew_mean` you measured.

---

## 7 · Save & load checkpoints

```bash
# Save a checkpoint every 100k steps
gdrl --env_path=./JumperHard.x86_64 \
  --experiment_name=jumper_final \
  --timesteps=2_000_000 \
  --save_model_path=jumper_ppo \
  --save_checkpoint_frequency=100000 \
  --onnx_export_path=jumper_ppo.onnx \
  --n_parallel=8 --speedup=20

# Resume if interrupted
gdrl --env_path=./JumperHard.x86_64 \
  --resume_model_path=jumper_ppo.zip \
  --experiment_name=jumper_final_resume \
  --timesteps=1_000_000 \
  --onnx_export_path=jumper_ppo.onnx
```

---

## What's next

**Unit 5:** Same BallChase environment — new skill: parallel rollout scaling with `n_parallel` and a proper evaluation protocol.

[→ Unit 5: Parallel Training](unit-05.md)
