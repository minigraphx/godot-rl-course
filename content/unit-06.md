# Unit 6 — Continuous 3D

Move from discrete button-presses to **continuous forces and steering**. Study **FlyBy** or **HovercraftRacing** — both expose a continuous action space — and learn how to normalize observations and actions so the neural network doesn't saturate.

[← Unit 5: Parallel Training](unit-05.md) · [Course home](index.md)

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

## 7 · Build your own continuous env (stretch)

Apply the continuous action pattern to a new scene:

1. Create a `RigidBody3D` vehicle or aircraft
2. Extend `AIController3D` (not `AIController2D`)
3. Implement 3D observations with RayCast3D sensors
4. Use a 2–4 dimensional continuous action space
5. Shape rewards: distance to goal + speed penalty + survival bonus

---

## What's next

**Visual Observations:** Move from raycast sensors to raw pixels — SubViewport pipeline, NatureCNN, and frame stacking in Godot.

[→ Visual Observations](unit-visual-observations.md)
