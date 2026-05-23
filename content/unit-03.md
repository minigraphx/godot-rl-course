# Unit 3 — CrossTheRoad & DQN

Study the official **CrossTheRoad** example — discrete 2D navigation with sparse rewards — then train it with **DQN** instead of PPO. Default workflow from here: exported binary, headless.

[← Unit 2: Lunar Lander](unit-02.md) · [Course home](index.md)

---

!!! warning "Console-first from here"
    Units 0–2 used the editor for building and debugging. From Unit 3 on, train with an exported binary and omit `--viz` for speed — then run a short **viz checkpoint** when training finishes (see [Section 5](#5-tweak-viz-checkpoint)).

!!! info "Three ways to see your AI"
    Godot (viz checkpoint) · TensorBoard (DQN vs Unit 2 PPO) · `AIController` reward tweaks

!!! warning "Training stalled?"
    Check in order: (1) reward sign and scale — is "good" actually positive? (2) sparse rewards — does the agent get any signal before the goal? (3) observation bugs — are sensors updating after resets? (4) TensorBoard flat but Godot looks fine — you may need longer training or a viz checkpoint.

---

## 1 · DQN in one page

**Value-based** methods learn how good each action is in each state — these are called **Q-values**. **DQN (Deep Q-Network)** uses a neural network to approximate those values, with two stabilizing tricks:

- **Experience replay** — transitions are stored in a buffer and sampled randomly, breaking correlations between sequential observations
- **Target network** — a periodically-copied frozen network is used for computing targets, preventing oscillation

Use DQN when the action space is **discrete** and rewards are sparse — like crossing lanes of traffic.

| | PPO (Unit 2) | DQN (this unit) |
|--|--|--|
| Family | Policy-based | Value-based |
| Learns | Policy directly | Q-values → policy |
| Exploration | Stochastic policy + entropy | ε-greedy (random early, greedy later) |
| Best for | Dense rewards, continuous or discrete | Sparse rewards, discrete actions |

!!! tip "Exploration: ε-greedy"
    DQN explores with **ε-greedy**: random actions early, greedy Q-actions later. If the curve is flat, ε may be decaying too fast — check your training script's exploration schedule. Compare Unit 2's PPO run in TensorBoard after this unit.

---

## 2 · Open CrossTheRoad

**Clone and import**

1. From [examples/CrossTheRoad](https://github.com/edbeeching/godot_rl_agents_examples/tree/main/examples/CrossTheRoad), open the project in Godot .NET
2. Enable the Godot RL Agents plugin (Project → Project Settings → Plugins)
3. Locate the training scene and `AIController` script

**Export for headless training**

Project → Export → add your platform preset → export binary. Then train against it:

```bash
conda activate godot_env
gdrl --env_path=./CrossTheRoad.x86_64 \
  --experiment_name=CrossTheRoad_DQN \
  --timesteps=500000 \
  --speedup=8 \
  --n_parallel=4
```

!!! info "DQN via SB3"
    The default `gdrl` command uses PPO. To use DQN, write a short training script:

    ```python
    from stable_baselines3 import DQN
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

    env = StableBaselinesGodotEnv(env_path="./CrossTheRoad.x86_64", n_parallel=4, speedup=8)
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="logs/")
    model.learn(total_timesteps=500_000)
    model.save("crosstheroad_dqn")
    env.close()
    ```

---

## 3 · Read the code

Trace these in order — same rhythm as SimpleReachGoal in Unit 2:

- **`get_obs()`** — what the agent sees (position, nearby hazards, distance to goal)
- **`get_action_space()`** — discrete moves (wait / up / down / left / right)
- **Reward logic** — sparse signal at goal + penalty for crashes; note the contrast to Unit 2's dense shaped lander rewards
- **Sync node** — number of parallel env roots in the training scene

**Key question to answer before training:** Does the agent receive *any* reward signal during a typical episode, or only at the very end? Sparse rewards require more exploration time — factor that into your timestep budget.

---

## 4 · Train headless

```bash
conda activate godot_env
tensorboard --logdir=logs &
python train_crosstheroad_dqn.py
```

Watch `ep_rew_mean` — sparse rewards may stay flat for thousands of episodes, then jump sharply as the agent discovers safe crossings. This is normal for DQN on sparse tasks; PPO on the same task would show a smoother rise.

**What to compare in TensorBoard after this unit:**

| Metric | PPO (Unit 2 lander) | DQN (CrossTheRoad) |
|--------|--------------------|--------------------|
| `ep_rew_mean` curve shape | Smooth, gradual rise | Flat → sharp jump |
| `train/entropy_loss` | Present | Not applicable |
| `train/loss` | Policy + value losses | TD loss only |

---

## 5 · Tweak & viz checkpoint

**Viz checkpoint (~5 min)**

Re-run the trained policy with `--viz` or Play Scene in Godot. Screenshot behavior that matches (or contradicts) the TensorBoard curve — keeps headless training from feeling invisible.

**Stretch goals (pick one):**

- Increase crash penalty by 2× — does learning speed up or stall?
- Add a small reward for forward progress — compare to pure sparse setup
- Train the same env with PPO — which algorithm reaches reliable crossings first?

---

## What's next

**Unit 4:** JumperHard — the canonical PPO benchmark, headless export, hyperparameter tuning.

[→ Unit 4: JumperHard & PPO](unit-04.md)
