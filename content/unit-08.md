# Unit 8 — Memory & POMDPs

Train **FPS / RobotFPS** — environments where the agent can't see everything at once. Learn why memory matters, how LSTM policy networks work, and how to use **RecurrentPPO** from `sb3-contrib`.

[← Unit 7: Multi-Agent](unit-07.md) · [Course home](index.md)

!!! note "Prerequisites"
    - **[Unit 4](unit-04.md)** — PPO end-to-end, confident reading `train/*` curves
    - **[Unit 6](unit-06.md)** — continuous action design (FPS uses continuous look + discrete fire)
    - **[Visual Observations](unit-visual-observations.md)** (optional) — only if you train from pixels
    - High-level familiarity with RNN / LSTM ideas (we re-explain in §2)

!!! info "Time"
    Reading: ~35 min · Training: ~30 min GPU / ~2 h CPU

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

!!! check "Done when"
    With both 3M-step runs in TensorBoard, `ep_rew_mean` for RecurrentPPO sits clearly above the PPO baseline, the gap opening somewhere around the 1–2M steps this section predicts. Expect noise — judge the trend over the last million steps, not single spikes. If the two curves are indistinguishable, suspect your observations before your hyperparameters: a leaked global position (§4–§5) makes the env fully observable and erases RecurrentPPO's advantage.

### Build it · Frame stacking on a masked CartPole

Frame stacking is the cheaper memory mechanism from the table above: instead of an LSTM, standard PPO simply receives the last N observations as input. This experiment runs entirely on the pinned course stack — no `sb3-contrib`, no Godot build needed. We hide CartPole's velocities (turning the MDP into a POMDP), then show that four stacked frames bring the missing information back: velocity is just a difference of consecutive positions.

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

class MaskVelocity(gym.ObservationWrapper):
    """CartPole as a POMDP: keep cart position + pole angle, hide both velocities."""
    def __init__(self, env):
        super().__init__(env)
        high = self.observation_space.high[[0, 2]]
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def observation(self, obs):
        return obs[[0, 2]].astype(np.float32)

def make_env():
    return MaskVelocity(gym.make("CartPole-v1"))

# Baseline: one masked frame per step — velocity is unrecoverable
blind = DummyVecEnv([make_env] * 8)
model_blind = PPO("MlpPolicy", blind, verbose=1, tensorboard_log="logs/")
model_blind.learn(total_timesteps=200_000, tb_log_name="ppo_blind")

# Frame stacking: 4 masked frames — velocity becomes a finite difference
stacked = VecFrameStack(DummyVecEnv([make_env] * 8), n_stack=4)
model_stacked = PPO("MlpPolicy", stacked, verbose=1, tensorboard_log="logs/")
model_stacked.learn(total_timesteps=200_000, tb_log_name="ppo_stacked")
```

!!! check "Done when"
    In TensorBoard, `ppo_stacked` climbs decisively while `ppo_blind` plateaus well below it — the same gap pattern the table above predicts for memory vs no memory, reproduced with the cheapest form of memory. Runs are noisy, so expect the ordering, not exact curves. If the two runs look identical, verify `MaskVelocity` is applied in both: the base env's `observation_space` should have shape `(2,)`, not `(4,)`.

---

## 8 · Viz checkpoint

Watch the agent with `--viz` after training:

- Does the agent **search** when the target moves out of sight, or does it freeze?
- Does it **remember** the direction it last saw the target?
- Does it handle **dead ends** (turn around) or get stuck?

A working LSTM agent will hesitate briefly when losing sight, then move in the last known direction — clear, human-readable memory behavior.

---

## 9 · Stretch goals

- **Longer memory** — increase `lstm_hidden_size` to 512; measure if it helps on a maze-style task
- **Build a memory task** — design an env where the agent must remember which of two doors it opened last episode

---

## What's next

**Self-Play:** Train agents by competing against copies of themselves — AirHockey, frozen checkpoints, league-based training, and ELO tracking.

!!! info "Self-check before you move on"
    Can you answer these in your own words?

    1. What property does a POMDP break, and what concretely does that imply for the network?
    2. What does the LSTM **hidden state** carry across time steps that a feed-forward policy can't?
    3. When would frame-stacking be enough, and when do you genuinely need RecurrentPPO?
    4. Why does RecurrentPPO use truncated BPTT instead of unrolling the full episode?
    5. Name one Godot environment in this course where you would *not* use memory, and one where you would.

    If you can answer all five — you're ready.

??? success "Self-check answers"
    1. A POMDP breaks the **Markov property** — the current observation no longer equals the full state. Concretely, a feed-forward network mapping one observation to one action faces ambiguity: the same observation can demand different actions depending on history, so the network needs a mechanism that carries past observations forward.
    2. The **hidden state** carries a learned, fixed-size summary of everything relevant seen so far — e.g. "target was last seen to the left" — written and read across timesteps. A feed-forward policy starts from scratch every step and can only react to the current observation.
    3. **Frame stacking** is enough when the missing information lives in a short, fixed window — like recovering a velocity from the last few positions. You need **RecurrentPPO** when the relevant event can lie arbitrarily far back: a target that left view many steps ago, or which corridor you entered a maze from.
    4. Unrolling the full episode would mean storing activations and backpropagating through potentially thousands of steps — memory blows up and gradients vanish or explode. **Truncated BPTT** backpropagates only through fixed-length chunks (the `n_steps=512` rollouts), keeping updates cheap and stable while the hidden state still flows forward across chunk boundaries.
    5. **No memory:** a fully observable env like JumperHard (Unit 4) — its observation already contains everything needed for the optimal action. **Memory:** FPS/RobotFPS from this unit — the target leaves view and the agent has no global position.

[→ Self-Play](unit-self-play.md)
