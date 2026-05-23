# Unit 4 — JumperHard & PPO Benchmarking

Train the **JumperHard** example — a 3D jumping robot that serves as a standard PPO benchmark in the godot-rl-agents repo. Focus: reading PPO hyperparameters, tuning them, and knowing when training has genuinely solved the task.

[← Unit 3: CrossTheRoad & DQN](unit-03.md) · [Course home](index.md)

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

| Parameter | Default | Theoretical role | When to change |
|-----------|---------|-----------------|----------------|
| `--learning_rate` | 0.0003 | Step size in gradient space — controls how far the optimizer moves per update | Lower if reward oscillates; raise if learning is very slow |
| `--n_steps` | 64 | Rollout length — controls the bias-variance tradeoff in GAE advantage estimates (longer = lower bias, higher variance; see [PPO Deep Dive](unit-ppo-deep.md)) | Raise (256–2048) for longer episodes; needs more memory |
| `--batch_size` | 64 | Mini-batch size within each epoch — must divide `n_steps × n_envs` | Match to `n_steps`; larger = more stable gradients |
| `--clip_range` | 0.2 | The ε in the clipped objective — controls the trust region size; how far the policy is allowed to move per update | Lower (0.1) if policy gradient loss explodes |
| `--ent_coef` | 0.0001 | Entropy bonus coefficient — adds a term to the loss that rewards keeping the policy stochastic, encouraging exploration | Raise (0.01) if agent converges too early to a suboptimal policy |
| `--gae_lambda` | 0.95 | λ in Generalized Advantage Estimation — interpolates between pure TD (λ=0, low variance, high bias) and pure Monte Carlo (λ=1, high variance, low bias) | Rarely needs changing; 0.9–0.99 is safe range |
| `--n_epochs` | 10 | Number of gradient passes over each rollout — with clipping it is safe to reuse; too many epochs push the policy outside the trust region | Lower if `approx_kl` grows large |
| `--vf_coef` | 0.5 | Value function loss coefficient — scales how strongly the critic is trained relative to the actor | Raise (0.75–1.0) if `value_loss` plateaus while policy improves |

!!! warning "Training stalled?"
    Check in order: (1) reward sign and scale, (2) sparse rewards — is there a shaped component? (3) `approx_kl` > 0.02 → reduce `--clip_range` or `--learning_rate`, (4) `ep_rew_mean` flat after 500k steps → raise `--n_steps` and `--ent_coef`.

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

## What's next

**Unit 5:** Same BallChase environment — new skill: parallel rollout scaling with `n_parallel` and a proper evaluation protocol.

[→ Unit 5: Parallel Training](unit-05.md)
