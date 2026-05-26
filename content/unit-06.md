# Unit 6 — Continuous 3D

Move from discrete button-presses to **continuous forces and steering**. Study **FlyBy** or **HovercraftRacing** — both expose a continuous action space — and learn how to normalize observations and actions so the neural network doesn't saturate.

[← Unit 5: Parallel Training](unit-05.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~30 min · Training: ~30 min GPU / ~2 h CPU

---

!!! info "Three ways to see your AI"
    Godot (viz checkpoint — smooth vs jerky motion tells you a lot) · TensorBoard (`ep_rew_mean` and `train/std` of action distribution) · `ai_controller.gd` normalization ranges

---

## 1 · Continuous action spaces

So far every action was **discrete**: fire engine / don't fire. Continuous actions are **real numbers** — "apply 0.73 Newtons of thrust at 12° left". This is closer to how real motors and steering work.

| | Discrete | Continuous |
|--|--|--|
| `get_action_space()` | `"action_type": "discrete", "size": N` | `"action_type": "continuous", "size": N` |
| Policy output | Softmax → one index | Gaussian distribution → N floats |
| Algorithm | DQN or PPO | **PPO only** (DQN requires discrete) |
| Gotcha | None | Values must be bounded; normalize before use |

```gdscript
# Continuous action space — 2 outputs: [thrust, steering]
func get_action_space() -> Dictionary:
    return {
        "motion": {"size": 2, "action_type": "continuous"}
    }

func set_action(action) -> void:
    var thrust   = action["motion"][0]   # roughly in [-1, 1]
    var steering = action["motion"][1]
    apply_force(Vector3.FORWARD * thrust * max_thrust)
    rotate_y(steering * max_steering_rate * get_physics_process_delta_time())
```

The policy outputs values in approximately **[−1, 1]** at the start of training — but without normalization, raw physics values (e.g. velocity in m/s = 42.7) will dominate the observations and destabilize learning.

---

## 2 · Normalization — the single most important thing

!!! warning "Normalize everything at system boundaries"
    The neural network works best when all inputs are in **[−1, 1]** or **[0, 1]**. Unnormalized observations (position in world units, velocity in pixels/s) cause slow learning or divergence.

**Observation normalization** — divide by the expected maximum:

```gdscript
func get_obs() -> Dictionary:
    var max_speed    = 20.0   # tune to your scene
    var max_dist     = 100.0
    var max_angle    = PI

    return {"obs": [
        linear_velocity.x / max_speed,
        linear_velocity.y / max_speed,
        linear_velocity.z / max_speed,
        rotation.y        / max_angle,
        angular_velocity.y / 5.0,
        raycast_forward.get_collision_distance() / max_dist,
        raycast_left.get_collision_distance()    / max_dist,
        raycast_right.get_collision_distance()   / max_dist,
    ]}
```

**Action normalization** — the policy outputs in [−1, 1]; scale to your physics units in `set_action()`:

```gdscript
func set_action(action) -> void:
    var raw_thrust   = action["motion"][0]          # [-1, 1]
    var raw_steering = action["motion"][1]          # [-1, 1]
    var thrust   = raw_thrust   * max_thrust        # scale to Newtons
    var steering = raw_steering * max_steering_rate
    ...
```

**How to check normalization is working:** Print `get_obs()` for a few steps during human-control mode. Every value should stay inside [−2, 2]. If any value routinely exceeds ±5, reduce its divisor.

---

## 3 · Open FlyBy or HovercraftRacing

Both examples are in [godot_rl_agents_examples](https://github.com/edbeeching/godot_rl_agents_examples).

- **FlyBy** — simpler, aerial navigation with raycasts
- **HovercraftRacing** — harder, racing track, mixed terrain

For your first continuous unit, start with FlyBy.

1. Open the project in Godot .NET, enable the plugin
2. Open `ai_controller.gd` — read `get_obs()`, `get_action_space()`, `set_action()`
3. Print obs values during human control to verify normalization:

```gdscript
func get_obs() -> Dictionary:
    var obs = { ... }
    if _ai.heuristic == "human":
        print(obs)
    return obs
```

4. Export a headless binary

---

## 4 · RayCast3D sensors

Continuous 3D environments often use **RayCast3D** nodes to give the agent spatial awareness without visual rendering.

```gdscript
# In _ready():
@onready var rays = $RayCastGroup.get_children()  # array of RayCast3D nodes

func get_obs() -> Dictionary:
    var ray_obs = []
    for ray in rays:
        if ray.is_colliding():
            ray_obs.append(ray.get_collision_point().distance_to(global_position) / max_ray_dist)
        else:
            ray_obs.append(1.0)   # no collision = max distance
    return {"obs": ray_obs + [linear_velocity.x / max_speed, ...]}
```

Place RayCast3D nodes as children of a `Node3D` group, pointing forward/left/right/up/down. Rotate the group with the agent. More rays = richer state; start with 5–9.

---

## 5 · Train

```bash
conda activate godot_env
tensorboard --logdir=logs &

gdrl --env_path=./FlyBy.x86_64 \
  --experiment_name=flyby_ppo \
  --timesteps=2_000_000 \
  --n_parallel=8 \
  --speedup=20 \
  --n_steps=512 \
  --batch_size=256
```

Continuous tasks typically need more timesteps than discrete ones — 1–5M is common. `n_steps=512` gives PPO longer rollouts to estimate advantage across the longer episodes.

**TensorBoard signals to watch:**

| Signal | Healthy | Problem |
|--------|---------|---------|
| `train/std` of actions | Starts ~1.0, slowly decreases | Collapses to 0 immediately → add `--ent_coef=0.005` |
| `rollout/ep_rew_mean` | Climbs by 1M steps | Still negative at 500k → check obs normalization |
| `train/approx_kl` | < 0.02 | Spikes → reduce `--learning_rate` or `--clip_range` |

---

## 6 · Viz checkpoint

After training, re-run with `--viz` or in the editor:

```bash
gdrl --env_path=./FlyBy.x86_64 \
  --resume_model_path=logs/sb3/flyby_ppo/best_model.zip \
  --inference \
  --viz
```

**What smooth vs jerky motion tells you:**

- **Smooth, deliberate** — policy has learned a stable action distribution; normalization is good
- **Jerky, oscillating** — action values are saturated (too large); reduce the scale factor in `set_action()`
- **Spinning in place** — angular velocity reward term may dominate; re-check reward shaping

---

## 7 · Build your own continuous env

Apply the continuous action pattern to a new scene:

1. Create a `RigidBody3D` vehicle or aircraft
2. Extend `AIController3D` (not `AIController2D`)
3. Implement 3D observations with RayCast3D sensors
4. Use a 2–4 dimensional continuous action space
5. Shape rewards: distance to goal + speed penalty + survival bonus

---

## 8 · VecNormalize — observation and reward normalization

While manual normalization in `get_obs()` works well, SB3 also provides **VecNormalize** — a vectorized wrapper that tracks a running mean and standard deviation across all parallel environments and normalizes on the fly.

**What VecNormalize does:**

- Maintains a running mean and std for every observation dimension across all `n_parallel` envs
- Normalizes each observation to approximately **N(0, 1)** before it reaches the policy network
- Optionally normalizes rewards by their running std (reduces variance without changing sign)
- Clips normalized values at `clip_obs` (default 10.0) to prevent outliers from dominating

**Why continuous control needs it more than discrete:**

In discrete environments, observations are often already bounded (e.g. a grid index or a boolean flag). In continuous 3D environments, raw physics values live on wildly different scales: a joint velocity might be 0.03 rad/s while a world-space position might be 847.2 m. Without normalization, the largest-magnitude observation dimension dominates the gradient update and the other dimensions are effectively ignored.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="./FlyBy.x86_64", n_parallel=8, speedup=20)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/")
model.learn(total_timesteps=2_000_000)

# CRITICAL: save the normalization stats alongside the model
env.save("flyby_vecnormalize.pkl")
model.save("flyby_ppo")
```

!!! warning "Always save VecNormalize stats"
    At inference time you must reload the same normalization statistics, otherwise the ONNX export will receive un-normalized observations and output nonsense actions.

    ```python
    env = StableBaselinesGodotEnv(env_path="./FlyBy.x86_64", n_parallel=1)
    env = VecNormalize.load("flyby_vecnormalize.pkl", env)
    env.training = False   # freeze the running stats during inference
    env.norm_reward = False

    model = PPO.load("flyby_ppo", env=env)
    ```

!!! tip "When to skip VecNormalize"
    If you already normalize every observation to **[−1, 1]** inside `get_obs()` in GDScript (the recommended approach for this course), VecNormalize is redundant. Manual GDScript normalization is more transparent and easier to debug during development. Reach for VecNormalize when you can't easily bound a value at the source — for example, a cumulative distance or an unbounded physics force.

---

## 9 · FlyBy vs HovercraftRacing — choosing your benchmark

Both examples ship with `godot_rl_agents_examples` and expose a continuous action space, but they have meaningfully different characteristics:

| | FlyBy | HovercraftRacing |
|--|-------|-----------------|
| Physics | 6-DOF free flight | Ground-constrained, high friction |
| Action space | 3D thrust + yaw | 2D thrust + steer |
| Reward | Checkpoint proximity | Race position + speed |
| Difficulty | Medium — 3D orientation is hard | Hard — tight turns, opponents |
| Recommended for | Learning continuous 3D obs | Competitive continuous control |
| Typical convergence | 1–2 M steps | 3–5 M steps |

**Which to use first:** Start with FlyBy. It has simpler physics and converges faster, so you can iterate on your observation and reward design quickly. Move to HovercraftRacing once you've confirmed your obs normalization, action scaling, and reward shaping all work correctly — the additional complexity of a race track and opponents will only add noise to debugging if the fundamentals aren't solid yet.

**Key differences in observation design:**

- *FlyBy* needs orientation relative to the next checkpoint (a 3D unit vector), linear velocity, and raycast distances. The checkpoint direction provides a clear learning signal — reward goes up as the agent heads toward it.
- *HovercraftRacing* typically exposes track-relative position, velocity components, and distance to the nearest wall or waypoint. Because the hovercraft is ground-constrained, you can drop the vertical velocity component and the up/down rays, keeping the obs vector smaller.

**Key differences in reward shaping:**

- *FlyBy*: A dense reward proportional to `max(0, prev_dist_to_checkpoint - curr_dist_to_checkpoint)` works well. Add a small time-alive bonus to discourage crashing early.
- *HovercraftRacing*: Pure distance reward can teach the agent to take tight lines but ignore opponents. Consider adding a penalty for lateral distance from the track centre and a bonus for overtaking.

!!! tip "Benchmark progression"
    FlyBy → HovercraftRacing mirrors the general pattern of starting simple and adding complexity. Resist the temptation to jump straight to the harder environment — a policy that doesn't converge in HovercraftRacing could be failing for a dozen different reasons; in FlyBy there are far fewer places to look.

---

## 10 · RayCast3D sensor design for 3D environments

The brief raycast example in section 4 covers the mechanics. This section goes deeper on design choices.

**Key `RayCast3D` parameters to set in the Inspector:**

| Parameter | Purpose | Recommended starting value |
|-----------|---------|--------------------------|
| `target_position` | Direction and max length of the ray | `Vector3(0, 0, -20)` for forward, 20 m max |
| `collision_mask` | Which physics layers to sense | Match your obstacle/wall layers; exclude the agent's own layer |
| `enabled` | Whether the ray is active | Always `true` during training; see warning below |

**Normalized distance pattern for `get_obs()`:**

```gdscript
@onready var rays = [$RayForward, $RayLeft, $RayRight, $RayUp, $RayDown]
const RAY_MAX = 20.0  # meters — must match target_position length

func _get_ray_obs() -> Array:
    var obs = []
    for ray in rays:
        if ray.is_colliding():
            obs.append(ray.get_collision_point().distance_to(global_position) / RAY_MAX)
        else:
            obs.append(1.0)  # no hit = max distance (already normalized)
    return obs
```

Call `_get_ray_obs()` inside `get_obs()` and concatenate with your velocity and orientation observations.

!!! tip "How many rays?"
    Five rays (forward, left, right, up, down) gives enough spatial awareness for FlyBy and keeps the observation vector small. More rays increase observation size linearly — a larger obs vector slows training and may require more network capacity. Run with 5 first. Only add more if the agent is hitting obstacles it couldn't see.

!!! warning "Collision mask must match between training and inference"
    If you change scene geometry or physics layer assignments after exporting the ONNX model, the rays may return different values for the same situation. Always retrain after structural scene changes. During ONNX inference in a live Godot scene, verify that collision masks on the `RayCast3D` nodes match exactly what was used during training.

---

## 11 · Stretch goals

These exercises extend the unit and are optional but highly recommended before moving on.

**VecNormalize vs manual normalization**

Train FlyBy twice — once using `VecNormalize` (section 8), once with all observations manually normalized to [−1, 1] inside `get_obs()` in GDScript. Log both runs to TensorBoard and compare:

- Which reaches a reward of +50 faster?
- Which is easier to debug when something goes wrong?
- What happens if you forget to reload the VecNormalize stats at inference time?

**Ray count ablation**

Train FlyBy with 3 rays, 5 rays, and 9 rays. Hold all other hyperparameters fixed. Measure:

- Steps to reach a stable positive reward
- Final `ep_rew_mean` at 2 M steps

At what point does adding more rays stop improving performance? Does it ever hurt?

**HovercraftRacing multi-agent**

Use the multi-agent setup introduced in Unit 7. Race two HovercraftRacing agents against each other in the same scene. Compare:

- Average lap time vs a single agent racing against a static obstacle course
- Does competition (an opponent to avoid) improve or hurt lap times?
- Do the agents develop cooperative or adversarial driving styles?

To set up the multi-agent race, duplicate the hovercraft `Node3D` (including its `AIController3D` child) and give each controller a unique `player_id`. The `godot-rl-agents` plugin handles the per-agent action dispatch automatically.

**Reward shaping challenge**

Design a reward function for FlyBy that satisfies all three of these constraints simultaneously:

1. The agent must reach the checkpoint (dense distance reward)
2. The agent must not spin faster than 90°/s (angular velocity penalty)
3. The agent must arrive in under 30 seconds (time pressure bonus)

How do you weight these three terms without one overwhelming the others? Log each reward component separately to TensorBoard — `ep_rew_mean` alone won't tell you which term is dominating.

---

## What's next

**Visual Observations:** Move from raycast sensors to raw pixels — SubViewport pipeline, NatureCNN, and frame stacking in Godot.

[→ Visual Observations](unit-visual-observations.md)
