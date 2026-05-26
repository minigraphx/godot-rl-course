# Locomotion Agents — Walker, Crawler & Worm in Godot

If you've seen **AI Warehouse** or other Unity ML-Agents showcases, the demos that get the most attention are the locomotion ones: a biped learns to walk from scratch, a quadruped discovers a trot, a worm figures out how to slither. These look like magic. They are not — they are reward design and physics, and you can build the same thing in Godot.

This unit shows you how.

[← Robot Observations & Sensors](unit-robotics.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~35 min · Training: ~45 min GPU / ~3 h CPU

---

!!! info "Three ways to see your AI"
    Godot (does the agent walk forward or collapse immediately?) · TensorBoard (`rollout/ep_rew_mean` should climb slowly over 5–10M steps — locomotion is slow to learn) · Viz checkpoint after every 1M steps: watch gait style change over training

---

## 1 · What locomotion RL actually is

Locomotion RL trains a policy to **control a chain of joints** so that the body moves in a desired direction. There is no hardcoded gait — the agent discovers one from scratch, using only the reward signal.

| Unity ML-Agents example | Godot equivalent | Key challenge |
|------------------------|------------------|---------------|
| Walker (biped) | `RigidBody3D` torso + 2 legs via `Generic6DOFJoint3D` | Balance while moving forward |
| Crawler (quadruped, no trunk) | 4 independent limbs on shared base | Coordination without a central body |
| Worm | Chain of `RigidBody3D` segments, `HingeJoint3D` | Exploit ground friction with no legs |
| Ant (4 legs + trunk) | Trunk + 4 legs, 2 joints each | High-DOF joint control |

All four use the same underlying recipe: **continuous joint control + shaped locomotion reward**. Change the body geometry and joint count; keep the reward structure.

---

## 2 · Scene setup — the biped Walker

Build this scene step by step. By the end you have a trainable biped.

### 2.1 · Body parts

Create a `Node3D` scene called `Walker`. Inside it:

| Node | Type | Size (m) | Position |
|------|------|----------|----------|
| `Torso` | `RigidBody3D` | 0.4 × 0.6 × 0.2 | (0, 1.2, 0) |
| `UpperLegL` | `RigidBody3D` | 0.12 × 0.35 × 0.12 | (−0.15, 0.85, 0) |
| `LowerLegL` | `RigidBody3D` | 0.10 × 0.35 × 0.10 | (−0.15, 0.48, 0) |
| `FootL` | `RigidBody3D` | 0.20 × 0.06 × 0.10 | (−0.12, 0.27, 0) |
| `UpperLegR` | `RigidBody3D` | 0.12 × 0.35 × 0.12 | (0.15, 0.85, 0) |
| `LowerLegR` | `RigidBody3D` | 0.10 × 0.35 × 0.10 | (0.15, 0.48, 0) |
| `FootR` | `RigidBody3D` | 0.20 × 0.06 × 0.10 | (0.12, 0.27, 0) |

Give each a `CollisionShape3D` matching its size. Set mass: Torso = 8 kg, upper legs = 2 kg, lower legs = 1.5 kg, feet = 0.8 kg.

### 2.2 · Joints

Between each pair of connected parts, add a `Generic6DOFJoint3D`. Lock all translation axes. Set angular limits:

| Joint | Connects | Angular limits (rad) |
|-------|----------|---------------------|
| `HipL` | Torso ↔ UpperLegL | X: [−1.0, 1.0], Y: [−0.3, 0.3], Z: [−0.5, 0.5] |
| `KneeL` | UpperLegL ↔ LowerLegL | X: [0.0, 1.8] (only forward bend) |
| `AnkleL` | LowerLegL ↔ FootL | X: [−0.6, 0.6] |
| `HipR` | Torso ↔ UpperLegR | X: [−1.0, 1.0], Y: [−0.3, 0.3], Z: [−0.5, 0.5] |
| `KneeR` | UpperLegR ↔ LowerLegR | X: [0.0, 1.8] |
| `AnkleR` | LowerLegR ↔ FootR | X: [−0.6, 0.6] |

Enable motors on every angular axis you want the agent to control. Set `PARAM_ANGULAR_MOTOR_FORCE_LIMIT` to 40 N·m (hip), 30 N·m (knee), 15 N·m (ankle).

### 2.3 · Contact sensors

Add an `Area3D` at each foot's bottom surface. Connect `body_entered` → a flag `foot_contact` on a small script attached to that Area3D. These go into the observation and the reward.

### 2.4 · AIController and Sync

Add `AIController3D` and `Sync` nodes to the scene root. Hook up `reset()` and `get_reward()` as usual.

---

## 3 · Observation space

Follow the egocentric pattern from the [Robot Observations unit](unit-robotics.md). For the biped:

```gdscript
extends AIController3D

@onready var torso        = $Torso
@onready var upper_leg_l  = $UpperLegL
@onready var lower_leg_l  = $LowerLegL
@onready var foot_l_body  = $FootL
@onready var upper_leg_r  = $UpperLegR
@onready var lower_leg_r  = $LowerLegR
@onready var foot_r_body  = $FootR

@onready var joints = {
    "hip_l":   $HipL,
    "knee_l":  $KneeL,
    "ankle_l": $AnkleL,
    "hip_r":   $HipR,
    "knee_r":  $KneeR,
    "ankle_r": $AnkleR,
}

@onready var foot_l = $FootContactL   # Area3D flag node
@onready var foot_r = $FootContactR

const MAX_SPEED     = 5.0    # m/s
const MAX_ANG_VEL   = 4.0    # rad/s
const MAX_JOINT_VEL = 8.0    # rad/s

var target_speed := 2.0      # m/s forward — set per episode or fixed

# Returns the x-axis rotation of `child` relative to `parent` in parent-local space.
# Use this to read joint angles — NOT Generic6DOFJoint3D parameters, which are
# static limits, not the current angle.
func _joint_angle_x(child: RigidBody3D, parent: RigidBody3D) -> float:
    var rel_basis = parent.global_transform.basis.inverse() * child.global_transform.basis
    return rel_basis.get_euler().x

func get_obs() -> Dictionary:
    var obs = []

    # Torso state — egocentric
    var fwd = -torso.global_transform.basis.z   # forward direction
    var vel = torso.linear_velocity

    obs.append(vel.dot(fwd)                         / MAX_SPEED)  # forward speed
    obs.append(vel.dot(Vector3.UP)                  / MAX_SPEED)  # vertical speed
    obs.append(vel.dot(fwd.cross(Vector3.UP))        / MAX_SPEED)  # lateral drift

    obs.append(torso.rotation.x / PI)                              # pitch
    obs.append(torso.rotation.z / PI)                              # roll
    obs.append(sin(torso.rotation.y))                              # yaw sin (avoids ±π wrap)
    obs.append(cos(torso.rotation.y))                              # yaw cos

    obs.append(torso.angular_velocity.x / MAX_ANG_VEL)
    obs.append(torso.angular_velocity.y / MAX_ANG_VEL)
    obs.append(torso.angular_velocity.z / MAX_ANG_VEL)

    obs.append(torso.global_position.y / 1.5)                     # height above ground

    # Per-joint: current angle (via relative body transform) + motor target velocity
    # child/parent pairs for each joint, in the same order as the action array
    var joint_pairs = [
        [upper_leg_l, torso],         # hip_l
        [lower_leg_l, upper_leg_l],   # knee_l
        [foot_l_body, lower_leg_l],   # ankle_l
        [upper_leg_r, torso],         # hip_r
        [lower_leg_r, upper_leg_r],   # knee_r
        [foot_r_body, lower_leg_r],   # ankle_r
    ]
    var joint_names = ["hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r"]
    for i in range(joint_names.size()):
        var angle = _joint_angle_x(joint_pairs[i][0], joint_pairs[i][1])
        var vel_cmd = joints[joint_names[i]].get_param_x(
            Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY)
        obs.append(angle    / PI)
        obs.append(vel_cmd  / MAX_JOINT_VEL)

    # Foot contact (binary)
    obs.append(1.0 if foot_l.foot_contact else 0.0)
    obs.append(1.0 if foot_r.foot_contact else 0.0)

    # Target speed command (enables a single policy trained at multiple speeds)
    obs.append(target_speed / MAX_SPEED)

    return {"obs": obs}
```

**Observation count:** 11 (torso) + 12 (6 joints × 2) + 2 (feet) + 1 (target speed) = **26 dimensions.**

A Crawler (4 legs, no torso rotation about Y) adds another 8 joints = ~42 dims. A Worm with 6 segments uses ~30 dims.

!!! warning "`PARAM_ANGULAR_LOWER_LIMIT` is a static limit, not the current angle"
    A common mistake: reading `get_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_LOWER_LIMIT)` always returns the fixed joint limit you set in the inspector — it never changes during simulation. Use `_joint_angle_x()` above to get the actual current angle from the relative body transform.

---

## 4 · Action space

Ten continuous outputs — one per controlled DOF. The joint motors accept a **target velocity**; the motor force limit caps the force applied.

```gdscript
func get_action_space() -> Dictionary:
    # 10 outputs: hip_l (x,y,z), knee_l (x), ankle_l (x),
    #             hip_r (x,y,z), knee_r (x), ankle_r (x)
    return {"joints": {"size": 10, "action_type": "continuous"}}

func set_action(action) -> void:
    var a = action["joints"]
    var MAX_VEL = 6.0   # rad/s — tune to your joint force limits

    joints["hip_l"].set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, a[0] * MAX_VEL)
    joints["hip_l"].set_param_y(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, a[1] * MAX_VEL)
    joints["hip_l"].set_param_z(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, a[2] * MAX_VEL)
    joints["knee_l"].set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY,  a[3] * MAX_VEL)
    joints["ankle_l"].set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, a[4] * MAX_VEL)
    joints["hip_r"].set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY,   a[5] * MAX_VEL)
    joints["hip_r"].set_param_y(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY,   a[6] * MAX_VEL)
    joints["hip_r"].set_param_z(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY,   a[7] * MAX_VEL)
    joints["knee_r"].set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY,  a[8] * MAX_VEL)
    joints["ankle_r"].set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, a[9] * MAX_VEL)
```

---

## 5 · Reward design

Locomotion reward is a **weighted sum of four components**. Get this right and the gait emerges. Get it wrong and the agent discovers creative ways to cheat.

```gdscript
var _alive := true

func _physics_process(_delta):
    if _ai.needs_reset:
        reset()
        return
    _compute_reward()
    _check_termination()

func _compute_reward():
    var fwd     = -torso.global_transform.basis.z
    var vel     = torso.linear_velocity
    var fwd_vel = vel.dot(fwd)           # positive = moving forward

    # 1. Forward velocity — the primary drive
    var r_vel = clampf(fwd_vel, -1.0, target_speed) / target_speed

    # 2. Energy penalty — discourage flailing
    var energy = 0.0
    for j in joints.values():
        var v = j.get_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY)
        energy += v * v
    var r_energy = -energy * 0.001

    # 3. Upright bonus — torso should stay level
    var up_dot = torso.global_transform.basis.y.dot(Vector3.UP)  # 1.0 = perfectly upright
    var r_upright = (up_dot - 0.5) * 0.1

    # 4. Alive bonus — prefer longer episodes over early termination
    var r_alive = 0.002

    _ai.reward += r_vel + r_energy + r_upright + r_alive

func _check_termination():
    if torso.global_position.y < 0.5:
        _alive = false
        _ai.reward -= 1.0
        _ai.done = true
```

### Why each component exists

| Component | Without it | With it |
|-----------|-----------|---------|
| Forward velocity | Agent doesn't move | Agent moves forward |
| Energy penalty | Agent vibrates joints at max speed (looks robotic, damages joints) | Smooth, efficient gait |
| Upright bonus | Agent crawls on its face or hops sideways | Stays balanced |
| Alive bonus | Agent falls immediately to end bad episodes faster | Prefers longer episodes to accumulate rewards |

!!! warning "Coefficient ordering matters"
    Keep `r_vel` in the range [0, 1]. All other terms should be ≤ 10% of the forward velocity signal. If the energy penalty dominates, the agent stands still. If upright dominates, the agent balances without walking.

---

## 6 · Training configuration

Locomotion needs more timesteps than any other example type in this course. Start with this:

```bash
conda activate godot_env
tensorboard --logdir=logs &

gdrl --env_path=./Walker.x86_64 \
  --experiment_name=walker_ppo \
  --timesteps=10_000_000 \
  --n_parallel=16 \
  --speedup=20 \
  --n_steps=2048 \
  --batch_size=512 \
  --learning_rate=0.0003 \
  --ent_coef=0.01
```

`n_steps=2048` gives PPO long rollouts — locomotion episodes last hundreds of steps and the advantage estimate needs time to propagate backward through a full gait cycle.

### What to expect over training

| Timesteps | Typical behaviour |
|-----------|------------------|
| 0–500k | Falls immediately. `ep_len_mean` = 20–50 steps |
| 500k–2M | Discovers how to stay upright. Still no forward motion |
| 2M–5M | Begins shuffling forward. Recognizable proto-gait |
| 5M–10M | Gait stabilizes. Speed approaches target |
| 10M+ | Gait refinement, energy efficiency improves |

Locomotion is the slowest-converging task in this course. **Do not judge it at 1M steps.**

---

## 7 · Common failure modes

### The statue
Agent stands perfectly still. `ep_rew_mean` > 0 but `ep_len_mean` maxes out at the episode timeout.

**Cause:** Alive bonus + upright bonus > forward velocity reward. The agent earns more by standing than by risking a fall.

**Fix:** Increase `target_speed` in the velocity reward, or reduce `r_alive` by half.

### The spinner
Agent learns to rotate in place. Forward velocity ≈ 0 but angular velocity ≈ max.

**Cause:** No lateral drift penalty. A rotating body has zero net forward velocity but the reward function never punishes spinning.

**Fix:** Add `r_lat = -abs(vel.dot(lateral)) * 0.1` to penalize sideways motion. Also add `r_yaw = -abs(torso.angular_velocity.y) * 0.05`.

### The hopper
Agent learns a single-leg bounce — technically moves forward, but looks nothing like the ML-Agents demos.

**Cause:** Single-leg hopping is a valid local optimum. It satisfies the forward velocity reward with less joint coordination than a full gait.

**Fix:** Add a **foot alternation bonus**: reward when the left and right contact signals alternate (not both on, not both off). `r_contact = abs(float(foot_l.foot_contact) - float(foot_r.foot_contact)) * 0.05`.

### Instant collapse
Agent falls every episode, reward never rises from the alive bonus.

**Cause:** Initial pose is unstable — the body spawns with enough torque or height that gravity wins before the policy acts.

**Fix:** Spawn the torso lower (0.8 m instead of 1.2 m), or add a 0.5-second physics-settled freeze at episode start before the policy begins issuing actions.

---

## 8 · Adapting to other body types

Once the Walker works, the same structure applies to other ML-Agents-style bodies:

### Crawler (quadruped, no trunk rotation)

- Remove the torso rotation penalty from the reward (the body is close to the ground — rolling less catastrophic)
- 4 legs × 2 joints each = 8-DOF action space (8 outputs)
- Add all 4 foot contact signals to the observation
- Increase `n_parallel` to 32+ — quadruped training benefits more from data volume

### Worm (segment chain)

- 6 `RigidBody3D` capsules connected by `HingeJoint3D`, single-axis rotation per joint
- No foot contacts — replace with "height of head segment above ground"
- Forward velocity measured from the head segment
- Worm discovers a sinusoidal wave pattern around 3–5M steps — clearly visible in the viz checkpoint

### Variable-speed command

To train a **single policy** that walks at multiple speeds (like the ML-Agents Walker demo), randomize `target_speed` per episode:

```gdscript
func reset() -> void:
    target_speed = randf_range(0.5, 3.0)   # m/s — wide range forces the policy to condition on it
    _alive = true
    _ai.reset()
```

Include `target_speed / MAX_SPEED` in `get_obs()` so the policy can read the current command. The agent learns to interpolate speeds without separate policies.

---

## 9 · Viz checkpoint — what a trained Walker should look like

Run with `--viz` at 3M, 6M, and 10M steps. Look for:

**3M:** Upright but awkward. Shuffles forward. Falls occasionally. This is the hardest phase to watch — it looks like a person learning to walk. It's working.

**6M:** Recognizable alternating gait. Rarely falls on flat ground. Leg swing is visible.

**10M:** Smooth gait. Stays upright. Approaches `target_speed`. Energy use has dropped (watch `train/std` of action distribution — it narrows as the policy becomes more decisive).

**Terrain test:** Add a slight slope or a low obstacle. A 10M policy will adapt without retraining. If it falls immediately, the domain is too far outside what it trained on — add terrain variation to the training environment.

---

## 10 · Stretch goals

- **Crawl → Walk curriculum:** Start training with the torso at 0.3 m height (forces crawling). After 2M steps, raise the spawn height to 1.2 m. Does the crawl policy transfer, or must it relearn?
- **Terrain variation:** Add random height variation to the floor using a `HeightMapShape3D`. How many extra timesteps does the policy need to stay upright on uneven ground?
- **Two-agent race:** Use the multi-agent setup from [Unit 7](unit-07.md). Two Walkers compete for forward position. Does competition accelerate or slow gait quality?
- **Export to ONNX:** Follow [Unit 10](unit-10.md) to export the Walker's policy. Embed it in the Godot scene as a pure inference demo — no Python at runtime. Share the HTML5 build.

---

## What's next

Your Walker/Crawler uses a fixed, handcrafted reward. **Hindsight Experience Replay (HER)** is the technique that turns sparse, goal-conditioned tasks — "reach this target position" — learnable without dense shaping. The same articulated body becomes a reaching arm.

[→ Goal-Conditioned RL & HER](unit-her.md)
