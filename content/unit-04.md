# Unit 4 — JumperHard & PPO Benchmarking

Train the **JumperHard** example — a 3D jumping robot that serves as a standard PPO benchmark in the godot-rl-agents repo. Focus: reading PPO hyperparameters, tuning them, and knowing when training has genuinely solved the task.

[← Unit 3: CrossTheRoad & DQN](unit-03.md) · [Course home](index.md)

!!! note "Prerequisites"
    - **[Unit 2](unit-02.md)** — a working SB3 PPO run end-to-end (training + ONNX export)
    - **[Unit 3](unit-03.md)** — DQN trained on CrossTheRoad; on-policy vs off-policy distinction
    - Comfort reading TensorBoard scalars (`ep_rew_mean`, `approx_kl`, `entropy_loss`)
    - **[Actor-Critic unit](unit-actor-critic.md)** (optional but useful) — makes Section 0 click immediately

!!! info "Time"
    Reading: ~40 min · Training: ~45 min GPU / ~3 h CPU

---

!!! info "Three ways to see your AI"
    Godot (viz checkpoint after training) · TensorBoard (track against your Unit 2 lander baseline) · `AIController` hyperparameter changes

---

## 0 · Why PPO? The theory in 5 minutes

Before touching JumperHard's config files, it is worth understanding *why* PPO is the default algorithm in godot-rl-agents — and what makes it different from DQN.

**PPO is an Actor-Critic method**

Where DQN only learns Q-values (a critic), PPO learns *two* things simultaneously:

| Component | What it learns | Neural network output |
|-----------|---------------|----------------------|
| **Actor** (policy π) | Which action to take in state s | Action distribution (mean + std for continuous, logits for discrete) |
| **Critic** (value V) | How good state s is, regardless of action | A single scalar V(s) |

The actor uses the critic's estimates to reduce variance in its gradient updates. The critic improves by comparing its predictions to actual returns. They train together, bootstrapping off each other.

For the full Actor-Critic derivation see [Actor-Critic unit](unit-actor-critic.md). Here we focus on applying it.

**The PPO update loop**

```
1. Run the current policy for n_steps transitions  → collect a rollout
2. Compute advantages using GAE (Generalized Advantage Estimation)
3. Make n_epochs gradient updates on that rollout
4. Throw away the rollout and repeat from step 1
```

The key difference from DQN's replay buffer: PPO discards rollout data after n_epochs passes. DQN keeps transitions forever; PPO is on-policy and uses data only from the current policy version.

**The clipped objective — the "P" in PPO**

The core idea of Proximal Policy Optimization is a clipped surrogate objective that prevents the policy from changing too drastically in a single update:

```
L_CLIP = E[ min(
    r_t(θ) · A_t,
    clip(r_t(θ), 1-ε, 1+ε) · A_t
)]

where r_t(θ) = π_θ(a|s) / π_θ_old(a|s)   (probability ratio)
      A_t    = advantage estimate
      ε      = clip_range (default 0.2)
```

If the new policy wants to change more than `clip_range` relative to the old policy, the gradient is zeroed out — effectively enforcing a **trust region** without the expensive second-order optimisation of TRPO.

!!! tip "Why clipping matters for JumperHard"
    JumperHard has a rough reward landscape — small changes in jump timing cause large changes in reward. Without clipping, a single good rollout could push the policy so far that it forgets everything else. The clipped objective keeps updates conservative and stable.

For the full PPO derivation with proofs see [PPO Deep Dive](unit-ppo-deep.md).

---

## 1 · What JumperHard teaches

JumperHard is harder than the lander for two reasons:

1. **3D physics** — the robot must balance while jumping, making the state space larger and the reward landscape rougher
2. **Sparse at the margin** — a bonus only fires on hard jumps, so the agent needs dense shaped rewards *and* occasional sparse bonuses to learn

This makes it a good benchmark: if your hyperparameters can solve JumperHard, they will generalise well.

---

## 2 · PPO hyperparameters — with theory

These are the knobs you turn. Defaults work for most tasks; you only need to change one at a time. The theoretical reason for each parameter is listed so you understand *why* a change should help.

| Parameter | gdrl default | Theoretical role | When to change |
|-----------|-------------|-----------------|----------------|
| `--learning_rate` | 0.0003 | Step size in gradient space — controls how far the optimizer moves per update | Lower if reward oscillates; raise if learning is very slow |
| `--n_steps` | 64 | Rollout length — controls the bias-variance tradeoff in GAE advantage estimates (longer = lower bias, higher variance; see [PPO Deep Dive](unit-ppo-deep.md)) | Raise (256–2048) for longer episodes; needs more memory |
| `--batch_size` | 64 | Mini-batch size within each epoch — must divide `n_steps × n_envs` | Match to `n_steps`; larger = more stable gradients |
| `--clip_range` | 0.2 | The ε in the clipped objective — controls the trust region size; how far the policy is allowed to move per update | Lower (0.1) if policy gradient loss explodes |
| `--ent_coef` | 0.0001 | Entropy bonus coefficient — adds a term to the loss that rewards keeping the policy stochastic, encouraging exploration | Raise (0.01) if agent converges too early to a suboptimal policy |
| `--gae_lambda` | 0.95 | λ in Generalized Advantage Estimation — interpolates between pure TD (λ=0, low variance, high bias) and pure Monte Carlo (λ=1, high variance, low bias) | Rarely needs changing; 0.9–0.99 is safe range |
| `--n_epochs` | 10 | Number of gradient passes over each rollout — with clipping it is safe to reuse; too many epochs push the policy outside the trust region | Lower if `approx_kl` grows large |
| `--vf_coef` | 0.5 | Value function loss coefficient — scales how strongly the critic is trained relative to the actor | Raise (0.75–1.0) if `value_loss` plateaus while policy improves |

!!! warning "Training stalled?"
    Check in order: (1) reward sign and scale, (2) sparse rewards — is there a shaped component? (3) `approx_kl` > 0.02 → reduce `--clip_range` or `--learning_rate`, (4) `ep_rew_mean` flat after 500k steps → raise `--n_steps` and `--ent_coef`. → [Full PPO diagnostic guide](unit-debugging.md#6-training-instability-oscillating-curves)

**The learning rate is the most sensitive knob.** When in doubt, halve it and retrain. Large learning rates destabilize the clipped objective; small ones just train slowly. "Too slow" is recoverable; "diverged" is not.

**n_steps and batch_size must be co-tuned.** `batch_size` must divide evenly into `n_steps × n_parallel`. If you set `n_steps=512` with `n_parallel=8`, total rollout size is 4096 — set `batch_size` to 256 or 512.

!!! info "GAE and the bias-variance tradeoff"
    GAE with λ=0.95 is a weighted sum over n-step returns. Higher λ looks further into the future (lower bias, noisier estimates). Lower λ relies more on the value function bootstrap (smoother but biased by value function errors). For JumperHard's long jump sequences, λ=0.95 is usually correct. See [PPO Deep Dive](unit-ppo-deep.md) for the full GAE derivation.

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

## 6 · TensorBoard diagnostic guide

Reading TensorBoard correctly is the skill that separates guessing from systematic tuning. Use this table as a reference while your experiments run.

| Metric | Healthy range | If outside range — action |
|--------|--------------|--------------------------|
| `train/approx_kl` | 0.01–0.02 | > 0.05: lower `--learning_rate` or `--clip_range`; policy is moving too far per update |
| `train/entropy_loss` | Slowly, steadily decreasing | Sudden collapse to ~0: raise `--ent_coef`; policy has gone deterministic prematurely |
| `train/value_loss` | Steadily decreasing | Plateaus while reward still low: raise `--n_steps` or `--vf_coef`; critic is not getting enough signal |
| `train/policy_gradient_loss` | Small oscillations near 0 | Sudden large positive spike: lower `--learning_rate`, add gradient clipping (`max_grad_norm=0.5`) |
| `rollout/ep_rew_mean` | Rising | Flat at 1M steps: raise `--ent_coef` (more exploration), check reward shaping in `AIController` |
| `rollout/ep_len_mean` | Stable or increasing | Drops suddenly: agent is dying early — check collision reward or episode reset logic |

**Reading `approx_kl` is the single most useful skill here.** KL divergence measures how much the policy changed from old to new in each update. Healthy PPO keeps it between 0.01 and 0.02. Above 0.05 means updates are too large and you risk destabilizing the policy. SB3 logs it automatically.

!!! info "Why entropy matters for JumperHard"
    JumperHard requires the agent to explore timing strategies for jumping — a robot that commits to one jump rhythm too early will never discover better ones. Entropy loss tracks policy stochasticity: a healthy entropy loss declines *slowly*. If it collapses in the first 200k steps, the agent has committed to a suboptimal strategy before seeing enough of the state space.

!!! warning "Don't read TensorBoard too early"
    The first 50–100k steps of a PPO run are heavily influenced by initial random weights. Reward curves during this phase are noisy and not informative. Judge runs by the 200k–1M step window.

---

## 7 · Eval protocol

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

## 8 · Save & load checkpoints

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

## 9 · Locomotion reward engineering

Navigation tasks have a single objective: get from A to B. Locomotion is different — the robot must discover a **gait** (a coordinated pattern of limb movements) as an emergent side effect of maximizing forward progress. The reward function is not just measuring success; it is sculpting which gait appears.

### Locomotion-specific reward components

**Forward velocity reward (the primary signal)**

```gdscript
# Reward forward velocity — encourages the robot to move fast
var forward_vel = linear_velocity.dot(global_transform.basis.z)  # z = forward axis
_ai.reward += forward_vel / max_speed * 0.1
```

This is the core signal, but it has a dangerous failure mode: the robot learns to **fall forward**. A single large forward lurch before collapsing maximizes forward velocity for one step — then the episode ends. Fix: pair it with a survival bonus.

**Survival bonus**

```gdscript
_ai.reward += 0.005   # per physics step — gives robot reason to stay alive
```

Without this, the agent has no incentive to remain on its feet. The survival bonus is what converts "fall forward" into "keep moving forward".

**Upright posture reward**

```gdscript
# Reward staying upright — dot product of up-vector with world up
var uprightness = global_transform.basis.y.dot(Vector3.UP)   # 1.0 = upright, -1.0 = upside down
_ai.reward += max(0.0, uprightness) * 0.01
```

This discourages locomotion policies that stay low or lean heavily. Combined with survival, it pushes the agent toward an upright stance before it has to worry about efficient walking.

**Energy efficiency (minimize power consumption)**

```gdscript
# Power = torque × angular_velocity (in Watts)
# Penalizing it encourages smooth, efficient gaits
var total_power = 0.0
for joint in joints:
    total_power += abs(joint_torques[joint] * joint_angular_velocities[joint])
_ai.reward -= total_power * 0.0001
```

This is the most important shaping term for producing natural-looking gaits. Without an energy penalty, locomotion policies frequently discover "galloping" motions that are mechanically unreasonable and would not work on real hardware. The energy penalty naturally produces walking-like gaits because walking is metabolically cheap — each leg briefly supports weight, transfers momentum, and swings forward with minimal effort.

**Smoothness reward (minimize jerk)**

```gdscript
# Penalize rapid changes in velocity (mechanical wear, instability)
var jerk = (linear_velocity - _prev_linear_velocity).length() / delta
_ai.reward -= jerk * 0.00001
_prev_linear_velocity = linear_velocity
```

Jerk is the derivative of acceleration. High jerk means the robot is lurching rather than flowing. Penalizing it tends to produce smoother trajectories and reduces the visual "vibrating" that appears in early locomotion policies.

**Contact reward (for multi-legged robots)**

```gdscript
# For robots with legs: reward foot-ground contact to encourage proper gait
for foot in feet:
    if foot.is_colliding():
        _ai.reward += 0.001
```

For walking robots with discrete feet, rewarding ground contact guides the policy toward periodically planting and lifting each foot — the basis of a gait — rather than sliding along the ground or hopping on one leg.

**Fall termination**

```gdscript
# End episode if robot falls — saves training time
if global_position.y < fall_threshold:
    _ai.reward -= 1.0   # penalty for falling
    _ai.done = true
    _ai.needs_reset = true
```

Ending the episode on a fall has two benefits: it saves compute (no more steps from a failed state) and it sends a strong signal that falling is costly. The penalty magnitude should be around 10–20× the per-step survival bonus.

### Gait emergence

The gait that emerges depends directly on which reward components are active. Adding components one at a time and watching the Godot viz checkpoint at each stage is the best way to build intuition:

| Reward components | Gait that typically emerges |
|---|---|
| Forward velocity only | Falls forward (one step, then dies) |
| Velocity + survival | Shuffling, dragging along the ground |
| Velocity + survival + upright | Upright hopping or bobbing |
| + energy efficiency | Walking-like gait with reduced wasted motion |
| + smoothness | Smooth walking with less jitter and vibration |
| All of the above | Natural-looking locomotion gait |

Each row is the previous row plus one signal. This is not a coincidence — each component closes a specific loophole that the agent was exploiting in the row above it.

!!! tip "JumperHard's jump bonus is a shaped sparse component"
    JumperHard uses a jump bonus — a sparse component that fires only on "hard" jumps. This is a sparse signal layered on top of dense locomotion rewards. The dense rewards keep learning stable between jumps; the sparse bonus steers the policy toward the actual task objective. Add `print(_ai.reward)` for 10 steps to see which component dominates in a given training phase.

### Comparison with MuJoCo locomotion benchmarks

If you ever read a locomotion paper — HalfCheetah, Ant, Hopper, Walker2D — the reward function will look almost identical to what is described above:

- **`forward_reward`** → forward velocity (same as the primary signal here)
- **`healthy_reward`** or **`survive_reward`** → survival bonus (same concept)
- **`ctrl_cost`** → `sum(action² ) × weight` — this is the MuJoCo approximation of energy efficiency. Instead of measuring actual torque × angular velocity, MuJoCo penalizes the magnitude of the action vector, which is a proxy for control effort.
- **`contact_cost`** (Ant) → penalizes high contact forces, similar to safety constraints

The key difference is that MuJoCo uses action magnitude as a proxy for power, while Godot's physics engine lets you compute actual joint torques. Both approaches produce similar gait behavior in practice.

Understanding JumperHard's reward is sufficient to read any locomotion paper's reward section. The vocabulary is the same; only the coefficient values differ.

---

## Stretch Goals

### Automated hyperparameter search with Optuna

Instead of manually running one experiment at a time, use Optuna to search the hyperparameter space automatically:

```python
import optuna
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import numpy as np

def objective(trial):
    lr         = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps    = trial.suggest_categorical("n_steps", [64, 128, 256, 512])
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef   = trial.suggest_float("ent_coef", 1e-4, 0.05, log=True)
    
    env = StableBaselinesGodotEnv(env_path="./JumperHard.x86_64", n_parallel=4, speedup=20)
    model = PPO(
        "MlpPolicy", env,
        learning_rate=lr,
        n_steps=n_steps,
        clip_range=clip_range,
        ent_coef=ent_coef,
        batch_size=64,
        verbose=0,
    )
    model.learn(total_timesteps=200_000)
    
    # Quick eval
    rewards = []
    for _ in range(10):
        obs, done, total = env.reset(), False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _ = env.step(action)
            total += r
        rewards.append(total)
    env.close()
    return np.mean(rewards)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print("Best params:", study.best_params)
```

Install: `pip install optuna`

!!! tip
    Optuna is the most practical hyperparameter search tool for SB3. RL-Zoo3 (used in HF course Unit 3) uses Optuna internally.

---

## What's next

**Unit 5:** Same BallChase environment — new skill: parallel rollout scaling with `n_parallel` and a proper evaluation protocol.

!!! info "Self-check before you move on"
    Can you answer these in your own words?

    1. What does the **clip range** in PPO actually clip, and what does it prevent?
    2. How does GAE-λ trade off bias against variance, and what does λ = 1 collapse to?
    3. If `approx_kl` is consistently above 0.05, what hyperparameter would you tune first — and which way?
    4. Why does Optuna's pruner cut bad trials early, and what would happen if you didn't prune?
    5. What's the difference between "training converged" and "policy is good enough to ship"?

    If you can answer all five — you're ready.

[→ Unit 5: Parallel Training](unit-05.md)
