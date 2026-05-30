# Apply It — SAC vs PPO on JumperHard

[← SAC](unit-sac.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~10 min · Training: ~20 min GPU / ~1 h CPU per algorithm

!!! info "Three ways to see your AI"
    Godot (does SAC's policy look smoother than PPO's?) · TensorBoard (compare `ep_rew_mean` slopes side by side) · sample efficiency (how many env steps to reach the same reward?)

!!! note "Prerequisites"
    - **[Unit 4](unit-04.md)** — JumperHard runs end-to-end with PPO
    - **[SAC](unit-sac.md)** — actor-critic-with-entropy, replay buffer, off-policy intuition (especially §5 *PPO vs SAC — the decision guide*)

---

## 1 · Why swap?

You've trained JumperHard with PPO in Unit 4 and read the SAC theory in the previous unit. Time to see the difference for yourself on a Godot env you already know.

PPO is **on-policy**: it samples a batch with the current policy, takes a few gradient steps on it, throws the batch away, and samples again. Simple, robust, easy to parallelise. SAC is **off-policy**: every transition goes into a replay buffer and gets reused for many gradient updates. On JumperHard's continuous action space, SAC's sample efficiency *can* dominate PPO — but each gradient step is more expensive, and the replay buffer eats RAM.

The point of this interlude is not to pick a winner. It's to **feel** the tradeoff on a project you already have running.

---

## 2 · Edit the training script

Open the Python training script you used for JumperHard in Unit 4 (`train_jumperhard.py` or whatever you named it). The PPO version looks roughly like this:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="JumperHard.x86_64", show_window=False)
env = VecMonitor(env)

model = PPO(
    "MlpPolicy", env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.0,
    tensorboard_log="./tb_logs_ppo/",
    verbose=1,
)
model.learn(total_timesteps=200_000)
model.save("jumperhard_ppo")
```

Make a copy and swap PPO for SAC:

```python
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecMonitor
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="JumperHard.x86_64", show_window=False)
env = VecMonitor(env)

model = SAC(
    "MlpPolicy", env,
    learning_rate=3e-4,
    buffer_size=200_000,        # replay buffer capacity (transitions)
    learning_starts=5_000,      # collect random data first
    batch_size=256,
    tau=0.005,                  # soft-update rate for target nets
    gamma=0.99,
    train_freq=1,               # one gradient step per env step
    gradient_steps=1,
    ent_coef="auto",            # automatic entropy temperature
    tensorboard_log="./tb_logs_sac/",
    verbose=1,
)
model.learn(total_timesteps=200_000)
model.save("jumperhard_sac")
```

Notice what disappeared (`n_steps`, `n_epochs`, `clip_range`) and what appeared (`buffer_size`, `learning_starts`, `tau`, `train_freq`, `gradient_steps`, `ent_coef`). These are not the same algorithm with a different name — every line above maps to a different optimization story.

!!! warning "Replay buffer memory"
    `buffer_size=200_000` with a small observation vector (JumperHard is ~tens of floats) is harmless. Bump the buffer for image observations and you'll feel it: 200 k × 84×84×4 bytes ≈ 5.6 GB. The buffer is the price of off-policy.

---

## 3 · Train both

Two terminal sessions, same env config, different algorithm:

```bash
# Terminal 1
python train_jumperhard_ppo.py

# Terminal 2
python train_jumperhard_sac.py
```

Point a single TensorBoard at both logdirs so you can overlay the curves (requires TensorBoard ≥ 2.x — the version SB3 currently pins):

```bash
tensorboard --logdir_spec ppo:./tb_logs_ppo,sac:./tb_logs_sac
```

Now open `localhost:6006` and watch `rollout/ep_rew_mean` for both runs simultaneously.

---

## 4 · What you'll see

Expected behaviour on JumperHard (this is **expected**, not a measurement — your numbers will vary with seed, hardware, and SB3 version):

- **SAC's `ep_rew_mean` rises in fewer environment steps.** That is sample efficiency: SAC squeezes more out of each transition because the replay buffer lets every transition contribute to many updates.
- **PPO often wins on wall-clock time.** JumperHard's environment step is cheap, PPO's gradient step is cheap, and PPO's data path is simpler. SAC's per-step cost (gradient step + target-net update + entropy temperature update) eats its sample-efficiency advantage in wall time on this env.
- **SAC's policy can look smoother in Godot.** Continuous action distributions with automatic entropy tuning often produce less jittery control than a clipped PPO policy that's still bleeding entropy.
- **SAC is more sensitive to hyperparameters early on.** `learning_starts` too low and the critic is fitting garbage; `tau` too high and the target nets oscillate. PPO's hyperparameters are forgiving by comparison.

If you don't see SAC reach the same reward in fewer env steps, check `ent_coef` (auto-tuning may have decayed entropy too fast — try `ent_coef=0.2` fixed) and `buffer_size` (too small means the buffer is dominated by stale early-training data).

---

## 5 · When to actually reach for SAC

A short decision guide once the experiment is done:

| Situation | Pick |
|---|---|
| Continuous actions, expensive simulation (real robot, physics-heavy sim) | **SAC** — sample efficiency matters more than wall-clock |
| Cheap parallel envs, discrete or continuous actions | **PPO** — easier to scale, more forgiving |
| You need stable training out of the box with little tuning | **PPO** |
| You want to push state of the art on continuous control benchmarks | **SAC** (or TD3) |
| Tight RAM budget, can't afford a replay buffer | **PPO** |

You'll meet SAC again in [Phase 6 — Locomotion](unit-locomotion.md) and [Sim-to-Real](unit-sim-to-real.md), where its sample efficiency stops being a curiosity and becomes essential.

---

## Stretch Goals

- **Wall-clock vs steps.** Re-train both with `time` and plot wall-clock seconds vs environment steps. Does SAC's sample-efficiency advantage translate into wall-clock advantage on JumperHard? Why or why not?
- **SAC on CrossTheRoad.** Try the SAC script on the discrete-action CrossTheRoad env from Unit 3. It will fail or behave badly — figure out why before reading the SAC docs.
- **Entropy-temperature sweep.** Train SAC with `ent_coef ∈ {0.05, 0.1, 0.2, "auto"}` and compare. What does the auto-tuner converge toward on JumperHard?

---

## What's next

You've now seen PPO and SAC as **users** — picking an algorithm class and trusting the library. Next, you peel one layer off: **CleanRL** strips PPO down to ~400 lines of single-file PyTorch so you can read every gradient step. Useful when SB3 is too opaque to debug, when you need a custom loss, or when a paper's algorithm has no library implementation yet.

[→ PPO From Scratch (CleanRL)](unit-cleanrl.md)
