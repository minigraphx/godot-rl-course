# Unit 8 — Memory & POMDPs

Train **FPS / RobotFPS** — environments where the agent can't see everything at once. Learn why memory matters, how LSTM policy networks work, and how to use **RecurrentPPO** from `sb3-contrib`.

[← Unit 7: Multi-Agent](unit-07.md) · [Course home](index.md)

---

!!! info "Three ways to see your AI"
    Godot (watch the agent hesitate when a target moves out of sight — a sign of memory working) · TensorBoard (`ep_rew_mean` with vs without RecurrentPPO) · observation design: what the agent can and cannot see

---

## 1 · Partially Observable Environments (POMDPs)

All previous units used **fully observable** environments — the agent's observation contained everything it needed to choose an action. Real environments are rarely this clean:

- An FPS character can't see through walls
- A robot in a maze doesn't know where it started
- A lander with a noisy sensor can't be sure of its exact altitude

These are **Partially Observable MDPs (POMDPs)**. The Markov property (observation = full state) no longer holds. The agent must **remember** past observations to infer the hidden state.

**Standard PPO breaks on POMDPs** — it treats each step independently. Given only the current (partial) observation, the optimal action is ambiguous.

**RecurrentPPO** adds an **LSTM** (Long Short-Term Memory) layer to the policy network. The LSTM carries a hidden state across timesteps, effectively giving the agent memory.

---

## 2 · How LSTM memory works in RecurrentPPO

```
Observation_t  →  [Shared MLP]  →  [LSTM cell]  →  [Policy head]  →  Action_t
                                        ↕
                                   hidden state h_t  →  h_{t+1}
```

The LSTM hidden state is:
- **Reset at episode boundaries** — memory doesn't leak between episodes
- **Shared across parallel envs** — each env instance has its own hidden state
- **Fixed size** — controlled by `lstm_hidden_size` (default: 256)

The agent learns to write relevant information into the hidden state (e.g. "target was last seen to the left") and read it back when needed.

---

## 3 · Install sb3-contrib

RecurrentPPO lives in `sb3-contrib`, not the base `stable-baselines3`:

```bash
conda activate godot_env
pip install sb3-contrib
```

---

## 4 · Open FPS or RobotFPS

1. From [godot_rl_agents_examples](https://github.com/edbeeching/godot_rl_agents_examples), open `examples/FPS` or `examples/RobotFPS`
2. Enable the Godot RL Agents plugin
3. Read `ai_controller.gd`:

**What the agent can see** (`get_obs()`):
```gdscript
func get_obs() -> Dictionary:
    return {"obs": [
        # RayCast readings — local perception only
        ray_forward.get_collision_distance() / max_dist,
        ray_left.get_collision_distance()    / max_dist,
        ray_right.get_collision_distance()   / max_dist,
        # No global position — agent can't see where it is on the map
        linear_velocity.x / max_speed,
        linear_velocity.z / max_speed,
        # Target visible? (0 or 1)
        float(target_in_sight),
    ]}
```

Note what is **missing**: global position, map layout, target location when out of sight. The agent must infer these from memory.

---

## 5 · Design partial observations (build your own)

If you want to add a memory requirement to an existing env:

**Remove global information:**
```gdscript
# Before (fully observable):
(global_position.x - target.global_position.x) / 100.0

# After (partial — agent must remember where it last saw the target):
float(target_in_sight) * (global_position.x - target.global_position.x) / 100.0
# When target is not in sight, this returns 0.0 — the agent loses the signal
```

**Add noise:**
```gdscript
# Noisy altitude reading
(global_position.y + randf_range(-0.5, 0.5)) / max_height
```

---

## 6 · Train with RecurrentPPO

```python
from sb3_contrib import RecurrentPPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(
    env_path="./RobotFPS.x86_64",
    n_parallel=8,
    speedup=20,
)

model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/",
    n_steps=512,
    batch_size=256,
    lstm_hidden_size=256,
    n_lstm_layers=1,
)
model.learn(total_timesteps=3_000_000)
model.save("robotfps_recurrent")
env.close()
```

```bash
conda activate godot_env
tensorboard --logdir=logs &
python train_robotfps.py
```

---

## 7 · PPO vs RecurrentPPO comparison

Run both on the same env and compare in TensorBoard:

```python
# Standard PPO baseline
from stable_baselines3 import PPO
model_ppo = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/")
model_ppo.learn(total_timesteps=3_000_000)
```

**Expected result:** PPO plateaus or fails on tasks that require memory. RecurrentPPO continues to improve. The gap is usually visible after 1–2M steps.

| Metric | PPO | RecurrentPPO |
|--------|-----|-------------|
| `ep_rew_mean` peak | Lower | Higher |
| Convergence speed | Faster early | Slower early, higher ceiling |
| RAM use | Low | Higher (LSTM states per env) |

---

## 8 · Viz checkpoint

Watch the agent with `--viz` after training:

- Does the agent **search** when the target moves out of sight, or does it freeze?
- Does it **remember** the direction it last saw the target?
- Does it handle **dead ends** (turn around) or get stuck?

A working LSTM agent will hesitate briefly when losing sight, then move in the last known direction — clear, human-readable memory behavior.

---

## 9 · Stretch goals

- **Stack frames** instead of LSTM: use a `VecFrameStack` wrapper to give the agent the last N observations as input to standard PPO
- **Longer memory** — increase `lstm_hidden_size` to 512; measure if it helps on a maze-style task
- **Build a memory task** — design an env where the agent must remember which of two doors it opened last episode

---

## What's next

**Unit 9:** Imitation learning — MultiLevelRobot, Behavioral Cloning, GAIL. Learning from expert demonstrations instead of reward signals.

[→ Unit 9: Imitation Learning](unit-09.md)
