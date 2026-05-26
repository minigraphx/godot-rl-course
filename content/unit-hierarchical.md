# Hierarchical RL — Decomposing Long-Horizon Tasks

[← Self-Play](unit-self-play.md) · [Course home](index.md)

---

Long-horizon tasks break flat RL. When a reward arrives only after 1 000 steps, gradient
signals become so attenuated that a standard PPO agent can spend millions of frames
thrashing near the starting position. Hierarchical Reinforcement Learning (HRL) cuts the
problem down to size: a **manager** (high-level policy) selects *subgoals*, and a
**worker** (low-level policy) achieves them. Each layer operates on its own timescale and
receives its own reward signal, which means credit assignment stays tractable.

This unit builds the concept from first principles, implements a two-level controller in
Godot, and shows you how to train the two halves together.

**Cross-unit links**

- [Goal-Conditioned RL & HER](unit-her.md) — the low-level worker IS a goal-conditioned
  policy; read that unit before section 3.
- [Multi-Agent RL](unit-07.md) — manager and worker can be framed as two cooperating agents.
- [Curiosity-Driven Exploration](unit-curiosity.md) — a cheaper alternative when the task
  has only 1–2 natural bottlenecks.
- [Reward Engineering](unit-reward-engineering.md) — always try dense shaping before
  reaching for HRL.

!!! warning "Try reward shaping first"
    Hierarchical RL introduces two policies, two learning rates, two reward signals, and
    at least one new hyperparameter (the manager step interval *k*). Before committing to
    this complexity, go back to [Reward Engineering](unit-reward-engineering.md) and ask:

    - Can I add a shaping term for reaching each intermediate room?
    - Can I add curiosity to drive exploration past the first door?

    If dense shaping plus curiosity still plateaus, *then* HRL is the right tool.

!!! info "Three ways to see your AI"
    - **Godot scene** — subgoal markers change colour in real time as the high-level policy
      switches targets; green = current subgoal, grey = inactive.
    - **TensorBoard** — two separate reward curves: `reward/high_level` (sparse, steps by
      *k*) and `reward/low_level` (dense, every step); watch them converge at different
      speeds.
    - **Hierarchical decision diagram** — a scrolling timeline strip that shows which option
      is active at each timestep, letting you spot when the manager is indecisive or the
      worker is failing to terminate cleanly.

---

## 1 · The Long-Horizon Problem

### Why flat PPO fails at 1 000+ steps

PPO's policy gradient update is:

```
∇J(θ) = E[ ∇ log π(aₜ|sₜ) · Aₜ ]
```

The advantage `Aₜ` is computed from the discounted return. With γ = 0.99 and a reward
that arrives only at step 1 000, the discount factor at the start of the episode is
`0.99^1000 ≈ 2.5 × 10⁻⁵`. That factor multiplies every gradient update for actions
taken near the beginning of the episode. In practice the policy receives almost no useful
learning signal for early decisions — it cannot tell which of the thousand actions it took
contributed to eventual success.

This is the **credit assignment** problem. It is not a PPO-specific limitation; it
affects every on-policy algorithm, and it is severe for off-policy methods too (replay
buffer entries from early in an episode contribute tiny gradients).

### Concrete example: multi-room navigation in Godot

Imagine a Godot scene with three rooms connected by narrow doors.

```
┌────────┐   door 1   ┌────────┐   door 2   ┌────────┐
│        │◄──────────►│        │◄──────────►│        │
│  Room  │            │  Room  │            │  Room  │
│   A    │            │   B    │            │   C    │
│ (start)│            │        │            │  ★GOAL │
└────────┘            └────────┘            └────────┘
```

The agent starts in Room A. The only reward is `+1` for collecting the star in Room C.
A flat PPO agent must randomly stumble through two doors — a conjunction of low-probability
events — before it ever sees a positive reward. With small doors and a large room, this
may never happen in a reasonable number of training steps.

**Typical learning curve comparison**

| Algorithm | Steps to first reward | Final success rate |
|---|---|---|
| Flat PPO (sparse) | Often never in 10 M steps | < 5 % |
| Flat PPO + curiosity | ~3 M steps | 40–60 % |
| HRL (manager + HER worker) | ~800 K steps | 85–95 % |

(Numbers are illustrative; exact values depend on room size and door width.)

### When to suspect you need HRL

Apply this checklist before reaching for hierarchical methods:

1. Episode length exceeds **500 steps** on average.
2. Reward is **sparse** — fewer than one reward event per 100 steps.
3. The task has **natural subtasks** you could describe to a human in one sentence each
   ("go to the door", "open the door", "cross to the next room").
4. Dense reward shaping and curiosity have **already been tried** and have plateaued.

If all four are true, HRL is a reasonable next step.

---

## 2 · The Options Framework

The Options framework (Sutton, Precup & Singh 1999) is the foundational formalism for HRL.
It replaces primitive actions with **macro-actions** called *options*.

### Formal definition

An option `ω` is a triple `(I, π, β)`:

| Component | Symbol | Meaning |
|---|---|---|
| Initiation set | `I ⊆ S` | States in which the option can be started |
| Intra-option policy | `π: S × A → [0,1]` | The policy executed while the option runs |
| Termination condition | `β: S → [0,1]` | Probability of terminating in each state |

The **manager** selects an option `ω` from the set `Ω`. Control then passes to `π_ω`
until termination, at which point the manager selects again.

### Semi-MDP: two timescales

The standard MDP ticks at every primitive step *t*. The Options framework creates a
**Semi-MDP** (SMDP) at the manager level, where one "step" spans the entire duration of
an option — potentially many primitive steps.

```
Primitive time:   t₀  t₁  t₂  t₃  t₄  t₅  t₆  t₇  t₈  t₉  …
                  │←── option ω₁ ──────►│←── option ω₂ ──►│ …
Manager time:     τ₀                     τ₁                 τ₂
```

The manager never sees the intermediate primitive steps; it only sees states at option
boundaries. This massively shortens the effective episode horizon at the manager level.

### Navigation options

For the three-room example, define four options:

| Option name | Initiation set | Terminates when |
|---|---|---|
| `go_to_door_1` | Room A | Agent within 30 px of door 1 |
| `go_to_door_2` | Room B | Agent within 30 px of door 2 |
| `go_to_door_3` | Room C | Agent within 30 px of goal |
| `go_to_goal` | Room C | Agent collects the star |

The manager picks which option to activate; each option's internal policy handles
moment-to-moment movement. The manager now only needs to solve a 4-step problem instead
of a 1 000-step problem.

### Hand-coded vs learned options

Options can be:

- **Hand-coded** — you define `I`, `π`, and `β` explicitly (simplest to start with).
- **Learned** — the agent discovers useful options through option-discovery algorithms
  (e.g., eigenoptions, covering options). This is an active research area and beyond the
  scope of this unit.

For the hands-on exercise we start with hand-coded options and train only the manager.

---

## 3 · Goal-Conditioned Low-Level Policy

Hand-coding options works for small, structured tasks. For larger tasks we want the
low-level policy to be *general* — able to reach any subgoal in the observation space —
so the high-level policy can propose arbitrary targets.

### The practical HRL loop

```
Every k primitive steps:
    manager observes sₜ
    manager selects subgoal gₜ  (a position or feature vector)

Every primitive step:
    worker observes [sₜ, gₜ]
    worker selects action aₜ
    worker receives intrinsic reward r_worker = -‖pos(sₜ₊₁) - gₜ‖

Every k steps:
    manager receives extrinsic reward r_manager = sum of environment rewards
    manager updates its policy
```

The worker is a **goal-conditioned policy** — it takes both the current observation and
the target subgoal as input. Training the worker with HER makes it robust to subgoals it
rarely reaches in early training.

### Link to unit-her.md

[Hindsight Experience Replay (HER)](unit-her.md) is the standard training method for the
low-level policy in practical HRL:

- For each trajectory the worker generates, HER retroactively relabels unsuccessful
  episodes with the state the worker *actually* reached as the "goal".
- This gives the worker a dense learning signal even when it misses the manager's
  intended subgoal.
- Without HER, the worker can take millions of steps to learn basic navigation; with HER,
  it typically converges in tens of thousands.

**Key implication**: train the low-level policy first using HER on a random-subgoal
curriculum, then freeze it (or continue fine-tuning it at a lower learning rate) while
training the high-level policy.

### Timescale mismatch and non-stationarity

A subtle problem arises because the low-level policy keeps improving while the high-level
policy trains. From the high-level policy's point of view, the "environment" (which
includes the worker) is non-stationary — the same subgoal `g` may be easier to reach at
step 500 K of training than at step 100 K. This is the core technical challenge of HRL
and is addressed by HIRO in section 5.

### Code sketch: the two-level training loop

```python
# Pseudocode — not Godot-specific
MANAGER_INTERVAL = 20  # k

obs = env.reset()
subgoal = manager.select_subgoal(obs)

for step in range(MAX_STEPS):
    # Worker acts
    worker_obs = concat(obs, subgoal)
    action = worker.predict(worker_obs)
    next_obs, env_reward, done, info = env.step(action)

    # Worker reward: negative distance to subgoal
    worker_reward = -distance(next_obs["position"], subgoal)
    worker.store_transition(worker_obs, action, worker_reward, next_obs)
    worker.train_step()

    # Manager acts every k steps
    if step % MANAGER_INTERVAL == 0:
        manager_reward = sum_of_env_rewards_since_last_manager_step
        manager.store_transition(obs_at_last_decision, subgoal, manager_reward, next_obs)
        subgoal = manager.select_subgoal(next_obs)
        manager.train_step()

    obs = next_obs
    if done:
        obs = env.reset()
        subgoal = manager.select_subgoal(obs)
```

---

## 4 · Godot Multi-Room Navigation Example

### Scene setup

Create a Godot 4 scene with the following node hierarchy:

```
MultiRoomEnv (Node3D)
├── Room_A (StaticBody3D + CollisionShape3D)
├── Room_B (StaticBody3D + CollisionShape3D)
├── Room_C (StaticBody3D + CollisionShape3D)
├── Door_1 (Area3D)           ← triggers room transition detection
├── Door_2 (Area3D)
├── Goal (Area3D + MeshInstance3D)  ← collectible star
├── Agent (CharacterBody3D)
│   ├── HighLevelAIController
│   └── LowLevelAIController
└── SubgoalMarker (MeshInstance3D)  ← visual indicator, changes colour
```

### Observation spaces

**High-level (manager) observation** — compact, room-scale information:

| Field | Type | Description |
|---|---|---|
| `current_room` | int (0–2) | Which room the agent is currently in |
| `door_1_pos` | Vector2 | 2D position of door 1 in world space |
| `door_2_pos` | Vector2 | 2D position of door 2 in world space |
| `goal_pos` | Vector2 | 2D position of the star |
| `steps_since_last_decision` | float (normalised) | Time pressure signal |

**Low-level (worker) observation** — precise, movement-scale information:

| Field | Type | Description |
|---|---|---|
| `agent_pos` | Vector2 | Agent position in world space |
| `agent_vel` | Vector2 | Agent velocity |
| `subgoal_pos` | Vector2 | Current subgoal position set by manager |
| `subgoal_delta` | Vector2 | `subgoal_pos - agent_pos` (redundant but helps learning) |
| `wall_distances` | float[4] | Raycast distances N/S/E/W |

### Action spaces

- **High-level**: `Discrete(4)` — indices map to `[door_1, door_2, goal, explore]`
- **Low-level**: `Box([-1,-1], [1,1])` — continuous 2D movement (x, z velocity targets)

### GDScript: two-level AIController pattern

```gdscript
# HighLevelAIController.gd
extends AIController3D

const MANAGER_INTERVAL := 20  # primitive steps between manager decisions
var _step_counter := 0
var _current_subgoal_index := 0
var _subgoal_positions: Array[Vector3] = []

func _ready() -> void:
    # Subgoal positions are set by the scene after doors are placed
    _subgoal_positions = [
        get_node("../Door_1").global_position,
        get_node("../Door_2").global_position,
        get_node("../Goal").global_position,
        Vector3.ZERO,  # "explore" — low-level falls back to curiosity
    ]

func get_obs() -> Array:
    var agent := get_node("../Agent") as CharacterBody3D
    return [
        _current_room_index(),
        get_node("../Door_1").global_position.x,
        get_node("../Door_1").global_position.z,
        get_node("../Door_2").global_position.x,
        get_node("../Door_2").global_position.z,
        get_node("../Goal").global_position.x,
        get_node("../Goal").global_position.z,
        float(_step_counter) / float(MANAGER_INTERVAL),
    ]

func get_action_space() -> Dictionary:
    return {
        "subgoal": {"size": 4, "action_type": "discrete"}
    }

func set_action(action: Dictionary) -> void:
    _step_counter += 1
    if _step_counter >= MANAGER_INTERVAL:
        _step_counter = 0
        _current_subgoal_index = action["subgoal"]
        _broadcast_subgoal(_subgoal_positions[_current_subgoal_index])

func _broadcast_subgoal(subgoal: Vector3) -> void:
    # Push subgoal to the low-level controller and update visual marker
    var low_level := get_node("../LowLevelAIController") as LowLevelAIController
    low_level.current_subgoal = subgoal
    get_node("../SubgoalMarker").global_position = subgoal
    _update_subgoal_colour(_current_subgoal_index)

func _update_subgoal_colour(idx: int) -> void:
    var colours := [Color.RED, Color.ORANGE, Color.GREEN, Color.BLUE]
    var marker := get_node("../SubgoalMarker") as MeshInstance3D
    var mat := marker.get_surface_override_material(0) as StandardMaterial3D
    if mat:
        mat.albedo_color = colours[idx]

func _current_room_index() -> int:
    var agent_pos := get_node("../Agent").global_position
    # Simplified: rooms are laid out along the X axis
    if agent_pos.x < -10.0:
        return 0
    elif agent_pos.x < 10.0:
        return 1
    else:
        return 2
```

```gdscript
# LowLevelAIController.gd
extends AIController3D

var current_subgoal := Vector3.ZERO

func get_obs() -> Array:
    var agent := get_node("../Agent") as CharacterBody3D
    var delta := current_subgoal - agent.global_position
    return [
        agent.global_position.x,
        agent.global_position.z,
        agent.velocity.x,
        agent.velocity.z,
        current_subgoal.x,
        current_subgoal.z,
        delta.x,
        delta.z,
        _raycast_distance(Vector3.LEFT),
        _raycast_distance(Vector3.RIGHT),
        _raycast_distance(Vector3.FORWARD),
        _raycast_distance(Vector3.BACK),
    ]

func get_action_space() -> Dictionary:
    return {
        "move": {"size": 2, "action_type": "continuous"}
    }

func set_action(action: Dictionary) -> void:
    var agent := get_node("../Agent") as CharacterBody3D
    var move := action["move"] as Array
    agent.velocity.x = move[0] * 5.0
    agent.velocity.z = move[1] * 5.0

func get_reward() -> float:
    # Intrinsic: negative distance to current subgoal
    var agent := get_node("../Agent") as CharacterBody3D
    var dist := agent.global_position.distance_to(current_subgoal)
    return -dist * 0.01  # scale so reward stays in [-1, 0]

func _raycast_distance(direction: Vector3) -> float:
    var space := get_world_3d().direct_space_state
    var query := PhysicsRayQueryParameters3D.create(
        get_node("../Agent").global_position,
        get_node("../Agent").global_position + direction * 10.0
    )
    var result := space.intersect_ray(query)
    if result.is_empty():
        return 1.0  # normalised: 10 m = 1.0
    return result["position"].distance_to(get_node("../Agent").global_position) / 10.0
```

### Training sequence

Train in two phases to avoid the chicken-and-egg problem (manager needs a working worker;
worker needs subgoals to practice on):

**Phase 1 — Train the low-level worker in isolation (≈ 1 M steps)**

```bash
# Use a dedicated scene that spawns random subgoal targets
gdrl train --config config/low_level_her.yaml
```

In `low_level_her.yaml`:
```yaml
trainer_type: ppo
n_envs: 8
use_her: true
her_replay_k: 4
reward_signal:
  - type: distance_to_subgoal
    weight: 1.0
```

**Phase 2 — Freeze worker, train manager (≈ 500 K steps)**

```bash
gdrl train --config config/high_level_ppo.yaml \
    --load-worker-checkpoint checkpoints/low_level_her_final.ckpt
```

Optionally fine-tune both together at a lower learning rate for the worker.

---

## 5 · HIRO — Off-Policy HRL

The two-phase training approach above is practical but suboptimal because the high-level
policy trains on a frozen worker that was not trained under the manager's actual subgoal
distribution. **HIRO** (Nachum et al. 2018, "Data-Efficient Hierarchical Reinforcement
Learning") solves this with **off-policy correction**.

### The non-stationarity problem

When the manager selects subgoal `g` at time `τ`, the low-level policy `π_worker` that
will execute `g` is the *current* policy at time `τ`. But by the time the manager's
replay buffer entry is sampled for training (much later), `π_worker` has changed. The
stored `(state, subgoal, reward, next_state)` transition was generated by an *old* worker,
but the manager's policy gradient is evaluated under the *current* worker. This mismatch
makes naive off-policy training for the manager incorrect.

### HIRO's correction

HIRO relabels stored high-level transitions by asking: "Given the sequence of states the
worker actually visited (`s_τ, s_τ₊₁, …, s_τ₊ₖ`), what subgoal `g'` would the *current*
worker have tried to reach to produce those same actions?"

Formally, it finds:

```
g* = argmax_{g'} Σₜ log π_worker(aₜ | sₜ, g' + sₜ - sτ)
```

This re-labelled `g*` replaces the original `g` in the replay buffer, making the stored
transition approximately consistent with the current worker.

### Architecture

Both levels use **SAC** (Soft Actor-Critic) for off-policy, continuous-action training:

```
High-level (manager):   SAC with subgoal space = obs space
Low-level (worker):     SAC with goal-conditioned obs + intrinsic reward
Off-policy correction:  Applied when sampling manager's replay buffer
```

### Availability in SB3

Stable-Baselines3 does not include HIRO. Options:

- **rl-games** — GPU-accelerated; some HRL support via custom callbacks.
- **stable-baselines3-contrib** — check for community-contributed HIRO; availability
  varies by version.
- **Reference implementation** — Nachum et al.'s original TensorFlow code is on GitHub
  (`tensorflow/models/research/efficient-hrl`); a PyTorch port exists as
  `HIRO-PyTorch` by various community authors.
- **For this course** — we use the two-phase PPO/HER approach (section 4), which is
  simpler and sufficient for the multi-room task.

---

## 6 · Feudal Networks (Conceptual)

Feudal Networks (Vezhnevets et al. 2017, "FeUdal Networks for Hierarchical Reinforcement
Learning") take the manager–worker idea one step further: the manager operates in a
**latent feature space** rather than the raw observation space.

### Architecture

```
Observation sₜ  →  Perception module  →  feature vector zₜ
                                              │
                              ┌───────────────┤
                              │               │
                         Manager            Worker
                    (slow, latent space)  (fast, raw obs)
                              │               │
                     direction dₜ in z-space  │
                              └───────────────►
                                     Worker reward:
                                     cos(zₜ₊c - zₜ, dₜ)
                                     (did worker move in
                                      the direction manager wanted?)
```

### Key ideas

- The **manager** outputs a *direction* in feature space — "move the feature vector in
  this direction over the next `c` steps".
- The **worker** receives a reward proportional to the cosine similarity between the
  direction it actually moved in feature space and the direction the manager requested.
- The manager never has to reason about pixel-level or position-level details; it reasons
  about high-level structure captured in the shared feature representation.

### Why useful

- The feature space is typically much lower-dimensional and smoother than raw observations,
  making the manager's learning problem easier.
- The manager's goal (a direction vector) is more robust to changes in the worker's policy
  than a specific position target, which partially addresses the non-stationarity problem.

### Practical status

Feudal Networks are conceptually elegant but difficult to tune. The shared perception
module, the timescale `c`, and the cosine reward signal all interact. For most Godot
projects, the simpler goal-conditioned + HER approach (section 3–4) or HIRO (section 5)
is preferable.

---

## 7 · When NOT to Use Hierarchical RL

HRL is a precision tool, not a default. Before choosing it, run through this checklist:

### Decision flowchart

```
Episode < 500 steps?
  YES → Use flat PPO (maybe + curiosity). Stop here.
  NO  ↓
Reward dense enough (> 1 event per 100 steps)?
  YES → Use reward shaping (see unit-reward-engineering.md). Stop here.
  NO  ↓
Fewer than 3 natural subtasks?
  YES → Use curiosity (see unit-curiosity.md) + sparse reward. Stop here.
  NO  ↓
Already tried shaping + curiosity?
  NO  → Try those first. Stop here.
  YES → Consider HRL.
```

### Specific warnings

**Debugging difficulty** — when something goes wrong you must determine:
- Is the manager selecting bad subgoals?
- Is the worker failing to reach the subgoal?
- Are the two reward signals conflicting?
- Is the manager interval `k` too short or too long?

Each question requires separate analysis. Budget extra debugging time.

**Hyperparameter explosion** — HRL adds at minimum:
- Manager learning rate (usually 10× lower than worker)
- Manager step interval `k`
- Subgoal space dimensionality
- Intrinsic reward scaling coefficient

**Local optima in the manager** — the manager can learn to select only the subgoals the
worker has already mastered, ignoring harder subgoals that would allow the agent to
progress further. Monitor option/subgoal usage frequency during training.

**Practical rule of thumb**: if the task has fewer than 3 natural subtasks that you can
describe unambiguously, flat PPO + curiosity will probably match or beat HRL in wall-clock
training time. HRL's advantage grows with task complexity and number of subtasks.

---

## 8 · Viz Checkpoint

Good visualisation is essential for debugging a two-level system. Without it you are
flying blind.

### In-scene visualisation (Godot)

Colour-code the `SubgoalMarker` node by which option or subgoal is currently active:

| Active subgoal | Marker colour |
|---|---|
| go_to_door_1 | Red |
| go_to_door_2 | Orange |
| go_to_goal | Green |
| explore | Blue |

Add a second floating label above the agent showing the current option name and the
number of steps since the manager last switched. A long-running option suggests the worker
is stuck; rapid switching suggests the manager is uncertain.

You can also draw a line from the agent to the current subgoal using `draw_line` in a
`_draw()` override on a `CanvasLayer` — this makes it immediately obvious whether the
worker is heading in the right direction.

### TensorBoard metrics

Log these metrics during training:

```python
# High-level metrics (logged every manager step)
writer.add_scalar("reward/high_level", manager_reward, global_step)
writer.add_scalar("manager/option_entropy", option_entropy, global_step)
writer.add_scalar("manager/steps_per_option", avg_option_duration, global_step)

# Low-level metrics (logged every primitive step)
writer.add_scalar("reward/low_level", worker_reward, global_step)
writer.add_scalar("worker/distance_to_subgoal", subgoal_dist, global_step)
writer.add_scalar("worker/action_std", action_std, global_step)
```

**What good HRL looks like on TensorBoard**:

- `reward/low_level` converges first (usually within the first 200 K steps).
- `reward/high_level` starts improving only after the worker is reliable.
- `manager/steps_per_option` increases over time — the manager learns to commit to
  subgoals for longer as the worker gets better at achieving them.
- `worker/action_std` decreases over time — the worker's policy becomes more decisive.

### The hierarchical decision timeline

Plot a horizontal strip chart with time on the X axis and option index on the Y axis,
colour-filled by option. A well-trained manager shows long contiguous blocks (it commits
to one subgoal, the worker achieves it, then the manager moves on). A poorly trained
manager shows rapid flickering between options.

```python
# Pseudocode for logging option timeline
timeline = []  # list of (timestep, option_index)
for step, option in enumerate(option_history):
    timeline.append((step, option))

# Visualise with matplotlib or log to W&B as a custom plot
```

---

## 9 · Stretch Goals

These exercises extend the unit for students who want to go deeper.

### Stretch 1 — Hand-coded Options

Implement the full Options framework in Godot with four hand-coded options:

- `go_north` — move to the northernmost point of the current room
- `go_south` — move to the southernmost point of the current room
- `go_east` — move through the easternmost door (if one exists)
- `go_west` — move through the westernmost door (if one exists)

Each option has a hard-coded `β` (terminate within 50 px of target) and a hard-coded
`π` (PID controller toward target). Train only the meta-policy (manager) using flat PPO
with the SMDP formulation. Compare convergence speed against the fully-learned variant.

### Stretch 2 — Flat PPO + Curiosity vs HRL

Run a controlled comparison on the multi-room task:

1. **Baseline A**: flat PPO + sparse reward only.
2. **Baseline B**: flat PPO + ICM curiosity (see [unit-curiosity.md](unit-curiosity.md)).
3. **HRL**: manager + HER worker (section 4 of this unit).

Measure:
- Steps to first successful episode
- Success rate at 1 M, 2 M, and 5 M steps
- Wall-clock training time for 5 M steps

Plot all three on the same axes and include your results in the course Discord.

### Stretch 3 — Three-Level Hierarchy

Add a third level above the manager: a **meta-manager** that selects which *room* to
explore next. The meta-manager operates on an even slower timescale (every 100 primitive
steps). Its observation is just the set of rooms visited and the current success rate for
each room. Its action is `Discrete(3)` — pick the next room to prioritise.

This creates the full three-level stack:

```
Meta-manager (every 100 steps) → which room?
    Manager (every 20 steps)   → which subgoal?
        Worker (every step)    → which primitive action?
```

Observe how the meta-manager learns to direct training effort toward rooms the agent has
not yet mastered.

---

[← Self-Play](unit-self-play.md) · [Course home](index.md) · [→ Multi-Task RL](unit-multitask.md)
