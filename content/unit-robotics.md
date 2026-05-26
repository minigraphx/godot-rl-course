# Robot Observations, Actions & Sensors

You can train an agent to win games. But a **robot** is not a game — it has joints that wear out, sensors that drift, motors that overheat, and a power cable that pulls on it. Before you can cross the sim-to-real gap, you need to design your observation and action spaces **as if the robot already exists in the real world**, even when you're still in Godot.

This unit teaches the robotics-specific patterns you'll use in every manipulation, locomotion, and sim-to-real unit from here on.

[← Ship Your Brain](unit-10.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~30 min

---

!!! info "Three ways to see your AI"
    Godot (watch the arm reach for a target — sparse reward = sparse motion at first) · TensorBoard (track joint-limit violations per episode — should trend to zero) · Print joint states during human mode (verify your action mapping moves the right joint in the right direction)

---

## 1 · What makes robotics RL different

So far in this course, every environment has been a **game**. The agent lives inside a simulator, scores points, dies, respawns. If the policy works in Godot, you ship it.

Robotics RL is not like that.

| Concern | Game AI | Robotics RL |
|---------|---------|-------------|
| Agent body | Virtual, indestructible | Physical, breakable, expensive |
| Sensors | Perfect, instantaneous | Noisy, delayed, sometimes missing |
| Actions | Apply force directly | Send commands to motors with latency |
| Failure cost | Restart episode | Replace gearbox |
| "Done" criterion | Policy beats benchmark | Policy survives 8 hours of continuous operation |
| Reproducibility | `seed=42` | The motor wears differently every day |

**The core challenge is the sim-to-real gap** — the difference between your Godot world and the real robot. We dedicate a full unit to closing that gap. But the gap **starts here**, in this unit, with how you design your observation and action spaces.

> **Key principle.** Design your obs and action spaces *as if the robot already exists*, even if you're still in pure simulation. Every shortcut you take in sim ("just read the perfect joint angle from `Skeleton3D.get_bone_pose()`") is a debt you'll repay later when the encoder gives you a noisy value 8ms late.

---

## 2 · Proprioceptive vs exteroceptive sensing

Roboticists split sensors into two categories. You need to know both terms because every paper uses them.

### Proprioception — sensing your own body

The robot's "sense of self": where its joints are, how fast they're moving, how much torque each motor is producing.

- **Joint angles** (from encoders on each motor)
- **Joint velocities** (encoder deltas)
- **Joint torques / currents** (motor current → torque)
- **IMU**: orientation, angular velocity, linear acceleration of the base / torso
- **Motor temperatures** (yes, real policies sometimes need these)

### Exteroception — sensing the world outside

Anything that depends on what's *around* the robot:

- **Cameras** (RGB, depth, stereo)
- **LiDAR / 3D depth scanners**
- **Distance sensors** (ultrasonic, ToF)
- **Force / torque sensor at the wrist or feet**
- **Microphones, contact switches, tactile skin**

### Why this distinction matters for RL

Proprioceptive obs are **always available, fast, and accurate** on real hardware. Every servo ships with an encoder. The numbers you get are typically clean within a few hundredths of a degree.

Exteroceptive obs are **expensive, noisy, and unreliable**. Cameras need calibration. LiDAR has occlusions. Force sensors drift with temperature. Some sensors update at 30 Hz while your control loop runs at 200 Hz.

> **Rule of thumb.** Prefer proprioception for the **actor**. Use exteroception for the **critic** (asymmetric actor-critic — see the sim-to-real unit). This way the actor depends only on signals you trust at deployment, while the critic gets the privileged view it needs to learn good value estimates.

---

## 3 · Building a robot observation space in Godot

Here's the canonical pattern for a 6-DOF robot arm built on Godot's `Skeleton3D` plus a chain of `Generic6DOFJoint3D` nodes:

```gdscript
# ai_controller.gd for a robot arm
extends AIController3D

@onready var skeleton     = $RobotArm/Skeleton3D
@onready var joints       = $RobotArm.get_children().filter(
    func(n): return n is Generic6DOFJoint3D)
@onready var end_effector = $RobotArm/EndEffector

const MAX_JOINT_VEL = 3.14   # rad/s — typical servo limit
const MAX_FORCE     = 50.0   # N — load cell range
const ARM_REACH     = 0.85   # m — fully extended

var _prev_angles : Array  = []

func get_obs() -> Dictionary:
    var obs = []

    # --- Proprioceptive: joint state (angle + velocity per joint) ---
    for bone_idx in range(skeleton.get_bone_count()):
        var pose  = skeleton.get_bone_pose(bone_idx)
        var angle = pose.basis.get_euler()          # rotation as Euler angles
        obs.append(angle.x / PI)                    # normalize to [-1, 1]
        obs.append(angle.y / PI)
        obs.append(angle.z / PI)
        # Note: angular velocity requires tracking previous pose
        # (computed in _physics_process and cached)

    # --- End-effector position relative to base ---
    var ee_local = to_local(end_effector.global_position)
    obs.append(ee_local.x / ARM_REACH)
    obs.append(ee_local.y / ARM_REACH)
    obs.append(ee_local.z / ARM_REACH)

    # --- Exteroceptive: target position (goal-conditioned — see HER unit) ---
    var goal_local = to_local(goal.global_position)
    obs.append(goal_local.x / ARM_REACH)
    obs.append(goal_local.y / ARM_REACH)
    obs.append(goal_local.z / ARM_REACH)

    return {"obs": obs}
```

### Observation vector size

For a 6-DOF arm with the above template:

| Block | Components | Count |
|-------|-----------|-------|
| Joint angles (6 joints × 3 Euler) | x, y, z per joint | 18 |
| End-effector position (local) | x, y, z | 3 |
| Goal position (local) | x, y, z | 3 |
| **Total** | | **24** |

Add joint velocities and you double the joint block — typical 6-DOF arms end up around 36–48 dims.

### The normalization rule

> **Normalize everything to [-1, 1] or [0, 1].** Neural network optimizers are designed around inputs of roughly unit variance. A raw angle of `2.93 rad` and a raw position of `0.04 m` will be combined inside the network — if one is two orders of magnitude larger, gradients on the small one will vanish.

Divide by the **physical maximum** (the joint limit, the arm reach, the max servo speed), not by the largest value you happen to have seen.

---

## 4 · Joint control modes

This is the single most important decision in robotics RL, and it's almost never discussed in game-RL courses. **Your action space defines what your policy can express.**

| Mode | What it controls | Godot implementation | Real hardware |
|------|-----------------|---------------------|---------------|
| **Position control** | Target joint angle | Set `rotation` or PID toward target | Servo motors (most hobby robots) |
| **Velocity control** | Target joint speed | `PARAM_ANGULAR_MOTOR_TARGET_VELOCITY` | DC motors with encoders |
| **Torque control** | Raw force / torque | `apply_torque_impulse` on the bone body | High-end actuators (ANYdrive, Dynamixel Pro) |

### Position control

The agent outputs a **target angle**. An inner controller (PID on real hardware, a stiff motor in Godot) drives the joint toward that target.

```gdscript
# Position control — agent outputs target angles
func set_action(action) -> void:
    for i in range(joints.size()):
        var target_angle = action["joints"][i] * PI            # action ∈ [-1, 1] → [-π, π]
        var error        = target_angle - current_angles[i]
        joints[i].set_param_x(
            Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY,
            error * position_gain)
```

### Velocity control

The agent outputs a **target angular velocity**. The motor tries to maintain that speed.

```gdscript
# Velocity control — agent outputs angular velocities
func set_action(action) -> void:
    for i in range(joints.size()):
        var target_vel = action["joints"][i] * MAX_JOINT_VEL   # action ∈ [-1, 1] → [-vmax, vmax]
        joints[i].set_param_x(
            Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY,
            target_vel)
```

### Torque control

The agent outputs **raw torque** applied to the link.

```gdscript
# Torque control — agent outputs raw forces
func set_action(action) -> void:
    for i in range(joints.size()):
        var torque = action["joints"][i] * MAX_TORQUE          # action ∈ [-1, 1] → [-τmax, τmax]
        bones[i].apply_torque(Vector3(torque, 0, 0))
```

### Which mode should you choose?

!!! tip "Control mode selection"
    - **Position control** → simplest, safest, good for slow precise tasks. Pick this first for manipulation, pick-and-place, reaching.
    - **Velocity control** → natural for locomotion, differential-drive mobile robots, and any task where "how fast" matters more than "where exactly."
    - **Torque control** → maximum expressiveness, hardest to train, needed for genuinely dynamic tasks like running, jumping, throwing, catching.

A useful heuristic: **the lower the control mode, the harder the learning problem and the more dynamic the achievable behaviour.** Start at position control and only drop down a level if your task fundamentally requires it.

!!! warning "Position control can hide bugs"
    A stiff position controller will *force* the joint to track the commanded angle, even through collisions. Your policy can learn to "teleport" by commanding wild angles, because the simulator's infinite-strength motor obeys. On real hardware this trips overcurrent protection and the arm shuts down. Cap the per-step change in commanded angle (action smoothing) to keep behaviour physically achievable.

---

## 5 · Sensor noise injection

Every real sensor lies a little. Encoders quantize. IMUs drift. Load cells respond to temperature.

If you train against perfect sim values, the policy becomes addicted to precision it won't have on the real robot. The fix is to **inject noise during training** so the policy learns to act robustly under uncertainty:

```gdscript
# In get_obs() — add realistic noise to all readings
func _add_noise(value: float, std: float) -> float:
    return value + randfn(0.0, std)

# Proprioceptive noise (small — encoders are accurate)
obs.append(_add_noise(angle.x / PI, 0.005))

# IMU noise (moderate — gyros drift, accelerometers vibrate)
obs.append(_add_noise(angular_velocity.x / MAX_VEL, 0.02))

# Force / torque noise (larger — load cells are noisy and temperature-sensitive)
obs.append(_add_noise(contact_force / MAX_FORCE, 0.05))
```

### Typical noise magnitudes (start here, tune from datasheets)

| Sensor | Reasonable σ (normalized) |
|--------|--------------------------|
| Encoder angle | 0.002 – 0.01 |
| Joint velocity (encoder diff) | 0.02 – 0.05 |
| IMU angular rate | 0.01 – 0.03 |
| IMU linear acceleration | 0.03 – 0.08 |
| Force / torque sensor | 0.05 – 0.10 |
| Depth camera distance | 0.02 – 0.05 |

### Noise as the entry point to domain randomization

This is your first taste of **domain randomization** — the technique behind almost every successful sim-to-real transfer. We'll go much deeper in the sim-to-real unit, but the pattern is the same: instead of training in one perfect world, train in a *distribution* of slightly broken worlds. Vary `std` per episode and the policy learns to be robust across the whole family.

```gdscript
func reset() -> void:
    # Vary sensor noise scale each episode
    encoder_std = randf_range(0.002, 0.010)
    imu_std     = randf_range(0.010, 0.040)
    # ...
```

---

## 6 · Safety and constraint rewards

A game agent that drives off the cliff respawns. A real robot that drives off the cliff is gone.

Robotics reward functions almost always combine a **task reward** with a stack of **safety and efficiency penalties** that shape behaviour toward something a real motor can sustain:

```gdscript
func _physics_process(delta):
    if _ai.needs_reset:
        reset()
        return

    # --- Task reward (sparse — only at success) ---
    if end_effector_near_goal():
        _ai.reward += 1.0
        _ai.done = true

    # --- Joint limit penalty ---
    for i in range(joints.size()):
        var angle = get_joint_angle(i)
        var limit = joint_limits[i]
        if abs(angle) > limit * 0.9:           # warn at 90% of limit
            _ai.reward -= 0.1
        if abs(angle) > limit:                  # hard stop — would damage real hardware
            _ai.reward -= 1.0
            _ai.done = true                     # terminate the episode

    # --- Energy efficiency (minimize power = torque × velocity) ---
    var power = 0.0
    for i in range(joints.size()):
        power += abs(get_joint_torque(i) * get_joint_velocity(i))
    _ai.reward -= power * 0.0001                # tiny coefficient — don't dominate task reward

    # --- Smoothness / jerk penalty ---
    var jerk = (linear_velocity - _prev_velocity).length() / delta
    _ai.reward -= jerk * 0.00001
    _prev_velocity = linear_velocity

    # --- Survival bonus (discourage immediate failure) ---
    _ai.reward += 0.001
```

!!! warning "Joint limit violations break real hardware"
    A position command past the mechanical stop drives the motor against the end-stop at full torque. On a real arm this strips gears within seconds. **Always include a hard joint-limit penalty *and* terminate the episode** — you want the policy to treat exceeding limits as catastrophic, not as "slightly negative." If you only soft-penalize, the policy may discover it's worth the cost.

### Tuning the coefficients

The numbers above (`-0.1`, `-1.0`, `0.0001`, `0.00001`, `0.001`) are not magic. They reflect a relative ordering:

1. **Task reward** is the largest signal (`+1.0` on success).
2. **Hard safety violations** are comparable in magnitude (`-1.0`) — you want them to truly hurt.
3. **Soft warnings** are 10× smaller (`-0.1`) — they shape behaviour without dominating.
4. **Efficiency and smoothness** are 1000–10000× smaller (`1e-4`, `1e-5`) — they polish a working policy, they don't drive learning.
5. **Survival bonus** is small and positive — encourages staying alive long enough to find reward.

If your robot is twitchy, raise the smoothness coefficient. If it's slow and overly cautious, lower the energy penalty. **Tune one at a time.**

---

## 7 · The standard robotics simulation stack

You will read robotics RL papers. They will mention simulators that aren't Godot. Here's what each one is and where Godot fits:

| Tool | Primary use | Physics | GPU-parallel | Godot equivalent |
|------|-------------|---------|--------------|------------------|
| **MuJoCo** | Locomotion, manipulation research | Excellent contact | No (CPU) | Close — different solver |
| **Isaac Gym / Isaac Lab** | Massive parallel training, sim-to-real | Good | **Yes (thousands of envs on one GPU)** | `n_parallel` but CPU-bound |
| **PyBullet** | Free MuJoCo alternative, classic baselines | Good | No | Similar |
| **gymnasium-robotics** | HER benchmark envs (Fetch, Hand) | Via PyBullet / MuJoCo | No | HER unit ports these |
| **Webots** | Education, ROS integration | Good | No | Similar open-source spirit |
| **Gazebo** | ROS-native robotics | OK (older) | No | Heavier, ROS-coupled |

### Why Godot anyway?

- **Visual fidelity.** Pretty out of the box — useful when your real-world task uses cameras.
- **Game-ready.** Add humans, doors, interactive props, full UI without leaving the engine.
- **Licensing.** Free, open-source, MIT-style. No login wall, no enterprise tier.
- **Cross-platform export.** Your trained agent can ship as an HTML5 demo or a desktop game.

### Where Godot is weaker

- **Throughput.** Isaac Gym runs 4,000+ environments on a single GPU. Godot's `n_parallel` is CPU-bound and tops out around your core count.
- **Contact physics.** MuJoCo's soft-contact solver is the gold standard for manipulation. Godot Physics is fine for most tasks but you'll notice the difference on delicate contact.
- **Robotics ecosystem.** No URDF importer, no ROS bridge, fewer pre-built robot models. You build the arm yourself.

**Choose Godot when** the task is mostly geometric / visual and you want a polished sim you can hand to non-researchers. **Switch to MuJoCo / Isaac** when contact precision or massive parallelism becomes the bottleneck.

---

## 8 · Building a minimal robot arm in Godot

Let's build a 3-DOF planar arm — the simplest non-trivial manipulator. This will be your environment for the HER (Hindsight Experience Replay) unit.

### Steps

1. **Create the scene.** New `Node3D` called `RobotArm`.
2. **Add three segments.** Three `RigidBody3D` children, each with a `BoxShape3D` collider (e.g. 0.3 × 0.05 × 0.05 m). Stack them along the X axis.
3. **Add hinge joints.** Between each pair of segments, add a `HingeJoint3D`. Wire its `node_a` and `node_b` to the two segments.
4. **Set joint limits.** On each `HingeJoint3D`, enable the limit and set it to ±90° (±1.57 rad). Mark the base segment's `freeze` to true so the arm has a fixed root.
5. **Attach the controller.** Add an `AIController3D` node to the root. Set `action_space = {"joints": {"size": 3, "action_type": "continuous"}}`.
6. **Add an end-effector.** Small `Sphere3D` (radius 0.02 m) as a child of the third segment, positioned at the tip.
7. **Add a goal.** Sphere (radius 0.03 m, no collision) that respawns at a random reachable position on `reset()`.
8. **Implement `get_obs()`.** Joint angles (3) + end-effector position (3) + goal position (3) = **9-dim** obs.
9. **Implement `set_action()`.** Map `action["joints"][i] ∈ [-1, 1]` to target velocity for hinge `i` via `set_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY, ...)`.

### Reset logic

```gdscript
func reset() -> void:
    # Zero all joint angles
    for seg in segments:
        seg.angular_velocity = Vector3.ZERO
        seg.linear_velocity  = Vector3.ZERO
    # Spawn goal somewhere reachable
    var theta = randf_range(-PI/2, PI/2)
    var r     = randf_range(0.3, 0.85)
    goal.position = Vector3(cos(theta) * r, sin(theta) * r, 0.0)
    _ai.reset()
```

That's it. The HER unit will pick this scene up and add goal relabeling on top.

---

## 9 · Locomotion observation patterns

Locomotion robots (legged, wheeled, humanoid) have a different canonical observation layout. This is adapted from the MuJoCo Ant / HalfCheetah convention that you'll see in every locomotion paper:

```gdscript
# Standard locomotion obs (MuJoCo Ant-style)
func get_obs() -> Dictionary:
    var obs = []

    # --- Torso state ---
    obs.append(linear_velocity.x  / max_speed)    # forward velocity (reward signal)
    obs.append(linear_velocity.y  / max_speed)    # vertical
    obs.append(linear_velocity.z  / max_speed)    # lateral
    obs.append(rotation.x         / PI)           # pitch
    obs.append(rotation.z         / PI)           # roll
    obs.append(angular_velocity.x / max_ang_vel)  # pitch rate
    obs.append(angular_velocity.z / max_ang_vel)  # roll rate

    # --- Per-leg state (× n_legs) ---
    for leg in legs:
        obs.append(leg.hip_angle    / hip_limit)
        obs.append(leg.knee_angle   / knee_limit)
        obs.append(leg.hip_velocity / max_joint_vel)
        obs.append(leg.knee_velocity / max_joint_vel)
        obs.append(1.0 if leg.foot_in_contact else 0.0)

    # --- Height above ground (useful for fall detection) ---
    obs.append(global_position.y / max_height)

    return {"obs": obs}
```

### What's deliberately *not* in there

- **Absolute X / Z world position** — the policy should generalize across the whole floor, not memorize "go to coordinate (3.7, 0, -2.1)." Replace with desired-velocity command.
- **Absolute yaw** — same reason. Replace with target heading relative to current heading.
- **The actual position of the goal in world frame** — pass a *direction* and *distance*, locally.

This is the **egocentric** view: the policy sees the world through the robot's own frame, never through a global coordinate system. Egocentric observations are the single biggest win for generalization in locomotion.

### Yaw as sin/cos

For any angular observation that wraps around (yaw, hip angle on a non-limited joint), pass it as **two** components — `sin(θ)` and `cos(θ)` — instead of the raw radians. This removes the discontinuity at ±π that confuses the network.

```gdscript
obs.append(sin(yaw))
obs.append(cos(yaw))
```

---

## 10 · Viz checkpoint

Train the 3-DOF arm from § 8 with PPO for 100 episodes (`--n_parallel 4 --speedup 8` works fine) and watch with `--viz`. Ask yourself:

- **Does the arm reach the target?** With sparse reward the first 50 episodes will look random. Around episode 70–100 you should start seeing intentional motion. If not, the reward shaping or normalization is wrong — debug obs values first.
- **Does the energy penalty actually smooth motion?** Run two trainings — one with the energy penalty and one without. Compare visually. The penalized run should look noticeably less twitchy.
- **Are joint velocities within hardware limits?** Add a debug print in `_physics_process`:

  ```gdscript
  if Engine.get_physics_frames() % 30 == 0:
      print("joint vels: ", joints.map(func(j): return get_joint_velocity(j)))
  ```

  If any value regularly exceeds `MAX_JOINT_VEL`, the policy will demand impossible motion on real hardware. Tighten action scaling or add a velocity penalty.

- **TensorBoard check.** Plot `episode_length` and any custom `joint_limit_violations` counter. The violation count should trend down — if it stays flat, your hard penalty isn't strong enough.

---

## 11 · Stretch goals

- **Add a 4th DOF — a wrist rotation.** Retrain. How does sample efficiency change? (Spoiler: every extra DOF roughly doubles training time. Curse of dimensionality.)
- **Swap velocity control for position control.** Which mode reaches the goal in fewer environment steps? Which produces smoother motion?
- **Simulated load cell.** Add a force reading at the end-effector when it touches the goal sphere. Include it in the obs. Does the policy learn to use contact feedback to "feel" the goal, or does it ignore the extra channel?
- **Asymmetric actor-critic preview.** Give the critic the goal's world position directly, but force the actor to see only its own joint state plus a noisy distance estimate. (Full treatment in the sim-to-real unit.)
- **Randomize the arm.** Per episode, vary link lengths ±10%, masses ±20%, joint friction ±50%. Does the trained policy survive? This is your first domain-randomization experiment.

---

## What's next

You can now:

- Distinguish proprioceptive from exteroceptive sensors and choose the right ones per role
- Build a robot observation space that won't lie to you when ported to real hardware
- Pick a control mode (position / velocity / torque) appropriate to your task's dynamics
- Inject noise to harden the policy against sensor reality
- Stack safety, efficiency, and smoothness rewards without drowning the task signal
- Read robotics papers and know whether their MuJoCo / Isaac setup maps to what you have in Godot

The next unit shows how to build Walker, Crawler, and Worm agents from scratch in Godot — the locomotion demos you've seen in AI Warehouse, rebuilt with the same reward structure you now know.

[→ Locomotion Agents](unit-locomotion.md)
