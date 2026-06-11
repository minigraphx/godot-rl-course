# Multi-Task RL — One Policy for Multiple Tasks

[← Hierarchical RL](unit-hierarchical.md) · [Course home](index.md) · [→ Imitation Learning](unit-09.md)

!!! info "Time"
    Reading: ~35 min · Training: ~30 min GPU / ~2 h CPU

---

A single robot arm that can reach, push, and pick-up-and-place. A character controller
that handles normal locomotion, obstacle avoidance, and time-pressure sprinting — all
from one set of network weights. This is the promise of multi-task reinforcement learning.

Rather than training N separate specialists, you train one generalist. Done well, the
shared representation is more data-efficient than any individual specialist, and the policy
generalises to task combinations neither specialist has seen.

Done badly, multi-task training can leave every task worse than a dedicated single-task
policy. That failure has a name — **negative transfer** — and avoiding it is the central
skill this unit teaches.

**Cross-unit links**

- [Goal-Conditioned RL & HER](unit-her.md) — goal conditioning is a special case of
  multi-task where the "task" is the goal position. Understanding it first makes the
  generalisation here cleaner.
- [Hierarchical RL](unit-hierarchical.md) — a manager + worker system is another way to
  reuse a single low-level policy across tasks; compare the two architectures.
- [Sim-to-Real Transfer](unit-sim-to-real.md) — multi-task training is one of the
  strongest tools for domain randomisation; tasks with different physics behave like
  separate domains.
- [Reward Engineering](unit-reward-engineering.md) — each task needs its own well-shaped
  reward; weak per-task reward design amplifies negative transfer.

!!! warning "Try separate policies first"
    Multi-task RL is more complex than it looks. Before reaching for it, ask:

    - Can I afford to train N separate policies? If N ≤ 5, that is usually the right
      answer.
    - Are the tasks similar enough to share a representation, or are they so different
      that one policy will always compromise the other?
    - Do I actually need to switch tasks at runtime, or can I just load a different
      checkpoint?

    If separate policies are feasible and you don't need runtime task switching, use them.
    Multi-task RL pays off when N is large, inference cost is constrained, or you need
    genuine forward transfer to tasks you haven't trained on yet.

!!! info "Three ways to see your AI"
    - **Godot scene** — a task indicator in the corner of the viewport shows which variant
      is active (coloured badge: blue = reach, orange = reach + obstacles, red = timed
      sprint); watch the same agent adapt its movement style to each badge.
    - **TensorBoard** — separate `reward/task_0`, `reward/task_1`, `reward/task_2` curves
      let you spot the moment negative transfer begins (one curve drops while another
      climbs).
    - **Per-task success rate table** — after training, run 100 evaluation episodes per
      task and display a `3 × 1` bar chart; a well-trained multi-task policy should hit
      ≥ 80 % on every bar.

---

## 1 · Beyond Goal Conditioning

Goal-conditioned RL (covered in [unit-her.md](unit-her.md)) generalises across *targets*
within one task structure. The physics, the reward function, and the required motor skills
are fixed — only the goal position changes.

Multi-task RL generalises across *tasks themselves*. Tasks may differ in:

- **Reward function** — reach the goal (dense distance reward) vs. avoid obstacles
  (penalty for collision) vs. finish within a time limit (time-based bonus).
- **Physics** — an agent trained in low gravity and high gravity simultaneously acquires
  a more robust locomotion skill than one trained in either alone.
- **Required skills** — "move forward" requires nothing special; "pick up the cube" requires
  grasping; "open the door" requires torque application. A shared policy must contain all
  three motor sub-programmes.

### Taxonomy

| Type | What varies | Same as GCRL? |
|---|---|---|
| Goal-conditioned RL | Target position / object | Yes — tasks are identical up to goal |
| Multi-task RL (same structure) | Reward function weights | Close — one MDP, varying objectives |
| Multi-task RL (different physics) | Transition dynamics | No — different MDPs |
| Multi-task RL (different skills) | Action space requirements | No — qualitatively different behaviour |

The further down this table you go, the harder multi-task becomes, and the more likely
negative transfer is to appear.

### The key structural change

In single-task RL the policy is `π(a | s)`. In multi-task RL the policy is `π(a | s, z)`
where `z` is a **task encoding** — extra information the policy uses to know which task it
is currently solving. Choosing the right form for `z` is the first architectural decision
this unit addresses.

---

## 2 · Task Representation

How you encode `z` determines how much the policy can generalise. Three main approaches:

### One-hot task ID

The simplest option. If there are N tasks, the task ID is an N-dimensional binary vector
with exactly one `1`:

```
task 0 (reach goal):          z = [1, 0, 0]
task 1 (avoid + reach):       z = [0, 1, 0]
task 2 (timed reach):         z = [0, 0, 1]
```

**Advantages**: trivially simple; no training required; the policy can, in principle, learn
completely different behaviour per task by conditioning on this signal.

**Disadvantages**: tasks are treated as categorically distinct with no notion of similarity.
A policy trained on 3 tasks cannot be queried with `z = [0.5, 0.5, 0]` to interpolate
between them. Does not generalise to unseen task IDs.

### Task parameter vector

Encode each task as its key numerical parameters:

```
task 0: z = [reward_weight_distance=1.0, obstacle_penalty=0.0, time_bonus=0.0]
task 1: z = [reward_weight_distance=1.0, obstacle_penalty=0.5, time_bonus=0.0]
task 2: z = [reward_weight_distance=1.0, obstacle_penalty=0.0, time_bonus=2.0]
```

**Advantages**: the policy can generalise to unseen parameter combinations (e.g.
`obstacle_penalty=0.3`) without retraining. Smooth interpolation between tasks is possible.

**Disadvantages**: requires you to identify and expose the task parameters, which is not
always possible.

### Natural language embedding

Encode task descriptions as dense vectors using a pre-trained language model:

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("all-MiniLM-L6-v2")

task_descriptions = [
    "reach the goal as fast as possible",
    "reach the goal while avoiding red obstacles",
    "reach the goal before the timer runs out",
]
z_vectors = encoder.encode(task_descriptions)  # shape: (3, 384)
```

**Advantages**: the most expressive encoding; enables instruction-following and
generalisation to entirely new task descriptions at inference time. Bridge to large
language models.

**Disadvantages**: high-dimensional (384+); computationally expensive; requires the
language model to remain frozen or co-train carefully.

### Comparison table

| Encoding | Dimensions | Generalises to unseen tasks | Implementation cost |
|---|---|---|---|
| One-hot ID | N (number of tasks) | No | Trivial |
| Task parameters | K (number of parameters) | Yes (interpolation) | Low |
| Language embedding | 384–768 | Yes (zero-shot text) | High |

**Course recommendation**: start with one-hot for the hands-on exercise. Graduate to task
parameters once your multi-task setup is working and you want to test generalisation.

---

## 3 · Multi-Task PPO

### Architecture: shared trunk + task-conditioned input

The standard multi-task policy uses a single neural network with the task encoding
concatenated directly to the observation:

```
input = concat(observation, task_encoding)
  ↓
shared MLP trunk (256 → 256)
  ↓
policy head → action distribution
  ↓
value head → V(s, z)
```

Both the policy head and the value head see the task encoding. Crucially, the trunk is
**shared** — the policy is forced to learn a representation that is useful across all
tasks. This is the source of both positive transfer (shared features help everyone) and
negative transfer (conflicting gradient directions hurt someone).

### Alternative: task-specific heads

For tasks with very different output requirements, replace the shared head with N
task-specific heads and use the task ID to select which head to activate:

```
shared trunk
  ↓
task_id → select head
  ├── head_0 → action distribution for task 0
  ├── head_1 → action distribution for task 1
  └── head_2 → action distribution for task 2
```

Task-specific heads reduce negative transfer at the output layer at the cost of more
parameters and the inability to interpolate between task heads.

### SB3 with a task-ID observation wrapper

Stable-Baselines3 PPO does not natively support multi-task training, but a simple
`gymnasium.ObservationWrapper` that prepends the task ID to the observation vector is
sufficient for most cases:

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

class TaskIDWrapper(gym.ObservationWrapper):
    """Prepends a one-hot task ID to the flat observation vector."""

    def __init__(self, env, task_id: int, n_tasks: int):
        super().__init__(env)
        self.task_id = task_id
        self.n_tasks = n_tasks

        # Use np.prod so this works for any flat obs shape (MLP-only wrapper).
        # For image observations, flatten first or use a different approach.
        original_shape = int(np.prod(env.observation_space.shape))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(original_shape + n_tasks,),
            dtype=np.float32,
        )

    def observation(self, obs):
        one_hot = np.zeros(self.n_tasks, dtype=np.float32)
        one_hot[self.task_id] = 1.0
        return np.concatenate([one_hot, obs])
```

### Training loop: round-robin task sampling

The simplest multi-task training strategy is to rotate through tasks in round-robin order,
collecting a rollout for each before performing a joint update:

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

N_TASKS = 3

def make_task_env(task_id: int):
    """Returns a factory function for DummyVecEnv."""
    def _make():
        # Replace with your actual per-task environment
        env = gym.make("LunarLander-v2")
        return TaskIDWrapper(env, task_id=task_id, n_tasks=N_TASKS)
    return _make

# Create one vectorised environment per task
envs = [DummyVecEnv([make_task_env(i)]) for i in range(N_TASKS)]

# Single policy — observation space must match across all tasks
# (guaranteed by the wrapper)
model = PPO(
    "MlpPolicy",
    envs[0],
    verbose=1,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    tensorboard_log="logs/multitask_ppo/",
)

# Manually implement round-robin: collect rollouts from each task env,
# then call a single gradient update
TOTAL_TIMESTEPS = 3_000_000
STEPS_PER_TASK = 2048

for iteration in range(TOTAL_TIMESTEPS // (STEPS_PER_TASK * N_TASKS)):
    for task_id, env in enumerate(envs):
        # Swap the environment and collect one rollout batch
        model.set_env(env)
        model.learn(
            total_timesteps=STEPS_PER_TASK,
            reset_num_timesteps=False,
            tb_log_name=f"task_{task_id}",
        )
```

!!! tip "Per-task TensorBoard logging"
    Log `reward/task_0`, `reward/task_1`, and `reward/task_2` as separate scalars rather
    than aggregating them. Aggregation hides negative transfer — two tasks might average
    to a healthy-looking 0.5 while one is stuck at 0.0 and the other is at 1.0.

---

## 4 · Negative Transfer

Negative transfer occurs when learning one task actively degrades performance on another.
It is the central practical challenge of multi-task RL and can be subtle to diagnose.

### Why it happens: gradient interference

Each task produces its own gradient direction in parameter space. If task A's gradient
points "north" and task B's gradient points "south-east", the combined gradient points
"north-north-east" — neither task gets the update it wants. When the angle between two
task gradients exceeds 90 degrees, the dot product is negative, meaning the tasks are in
active conflict. One task's update literally makes the other task worse.

```
Task A gradient: ▲  (wants to increase value of action "jump")
Task B gradient: ▼  (wants to decrease value of action "jump")
Combined:        →  (neither task is served correctly)
```

This conflict is most severe when:

- Tasks require qualitatively different behaviours (e.g. aggressive vs. cautious).
- Tasks have different reward scales that cause one to dominate the gradient norm.
- The policy is forced to share all parameters including the final output layer.

### How to detect it: per-task TensorBoard curves

The clearest signal is **diverging per-task reward curves**:

```
Step 0         Step 500K      Step 1M
──────────────────────────────────────
task_0: 0.2 → 0.8 → 0.9    (improving)
task_1: 0.1 → 0.3 → 0.1    (regressing after initial gain)
task_2: 0.0 → 0.0 → 0.0    (never learned)
```

Task 1 regressing while task 0 continues to improve is the canonical negative transfer
signature. Task 2 never learning at all suggests a reward scale or difficulty imbalance.

```python
# Log per-task rewards during evaluation
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs/multitask_eval")

def evaluate_per_task(model, envs, n_episodes=20):
    for task_id, env in enumerate(envs):
        total_reward = 0.0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
        mean_reward = total_reward / n_episodes
        writer.add_scalar(f"eval/reward_task_{task_id}", mean_reward, global_step)
```

### Mitigation strategies

| Strategy | Idea | When to use |
|---|---|---|
| Reward normalisation | Normalise each task's reward to zero mean, unit variance | Always — first thing to try |
| Task-specific heads | Separate output layers per task | When tasks require different action distributions |
| Gradient surgery (PCGrad) | Project conflicting gradients to remove interference | When tasks are known to conflict |
| Separate value functions | Share actor, separate critic per task | Moderate conflict; cheap to implement |
| Curriculum ordering | Introduce tasks sequentially, not all at once | When one task is significantly harder |

!!! warning "Reward scale mismatch is the most common culprit"
    Before blaming gradient interference, check reward scales. If task 0 returns rewards
    in `[-1, 1]` and task 1 returns rewards in `[-100, 100]`, the policy gradient is
    dominated entirely by task 1. Normalise all per-task rewards to the same scale before
    debugging anything else.

---

## 5 · Curriculum Over Tasks

Not all tasks are equally hard. Throwing a beginner policy at the hardest task from step 1
leads to slow convergence or complete failure. A **task curriculum** — starting easy and
introducing harder tasks as the policy matures — dramatically speeds up learning.

### Fixed manual curriculum

Define task phases explicitly:

```python
CURRICULUM = [
    # (start_step, task_ids_to_include)
    (0,           [0]),          # Phase 1: reach goal only
    (500_000,     [0, 1]),       # Phase 2: add obstacle avoidance
    (1_500_000,   [0, 1, 2]),    # Phase 3: add time pressure
]

def get_active_tasks(global_step: int) -> list[int]:
    active = [0]
    for start_step, task_ids in CURRICULUM:
        if global_step >= start_step:
            active = task_ids
    return active
```

This requires manual tuning of the transition thresholds, which is effort but gives
precise control.

### Automatic curriculum (task success threshold)

Introduce a new task only after the policy achieves a minimum success rate on the current
task set:

```python
SUCCESS_THRESHOLD = 0.75  # require 75% success before adding next task

def maybe_expand_curriculum(current_tasks, per_task_success_rates, all_tasks):
    min_success = min(per_task_success_rates[t] for t in current_tasks)
    if min_success >= SUCCESS_THRESHOLD:
        next_task_id = len(current_tasks)
        if next_task_id < len(all_tasks):
            print(f"Curriculum expanding: adding task {next_task_id}")
            return current_tasks + [next_task_id]
    return current_tasks
```

!!! tip "Automatic curriculum can stall"
    If the policy never reaches `SUCCESS_THRESHOLD` on a hard task, the curriculum
    stalls permanently. Add a maximum wait time — introduce the next task after N steps
    regardless of success rate, otherwise you may never reach the later tasks at all.

### Sampling weights over tasks

Instead of hard curriculum phases, use a soft probability distribution over tasks that
shifts as training progresses:

```python
import numpy as np

def task_sampling_weights(per_task_success: list[float], temperature: float = 1.0) -> np.ndarray:
    """
    Weight tasks inversely by their current success rate:
    harder tasks (low success) get sampled more often.
    """
    difficulties = [1.0 - s for s in per_task_success]
    # Avoid zero weights
    difficulties = [max(d, 0.05) for d in difficulties]
    weights = np.array(difficulties) ** (1.0 / temperature)
    return weights / weights.sum()

# Example: tasks at success rates [0.9, 0.4, 0.1]
weights = task_sampling_weights([0.9, 0.4, 0.1])
# → task 2 (hardest) gets sampled most often
task_id = np.random.choice(N_TASKS, p=weights)
```

This is a simple form of **prioritised task replay**, analogous to prioritised experience
replay in DQN but at the task level.

---

## 6 · Godot Multi-Task Example

This section implements a full three-task AIController in Godot 4. A single agent switches
between three variants on demand:

| Task | ID | Reward structure | Required skill |
|---|---|---|---|
| Reach goal | 0 | Dense distance reward | Efficient navigation |
| Reach goal, avoid obstacles | 1 | Distance reward − collision penalty | Navigation + obstacle reading |
| Reach goal under time pressure | 2 | Distance reward + time bonus | Fast navigation |

### Scene setup

```
MultiTaskEnv (Node3D)
├── Agent (CharacterBody3D)
│   └── MultiTaskAIController (extends AIController3D)
├── Goal (Area3D + MeshInstance3D)          ← moves each episode
├── ObstacleSpawner (Node3D)                ← spawns 0–5 obstacles per episode
├── TaskIndicator (Label3D)                 ← displays current task ID in viewport
└── TimerBar (ProgressBar — CanvasLayer)    ← visible only in task 2
```

### GDScript: MultiTaskAIController

```gdscript
# MultiTaskAIController.gd
extends AIController3D

const N_TASKS := 3
const OBS_DIM := 8       # base observation size before task ID
const MAX_EPISODE_STEPS := 200
const TIME_PRESSURE_LIMIT := 100  # steps before time bonus decays to zero (task 2)

var current_task_id := 0
var _step_count := 0
var _episode_reward := 0.0

# Obstacles cached each episode
var _obstacle_positions: Array[Vector3] = []

func _ready() -> void:
    # Task ID is set externally by the training script via set_task()
    _randomise_episode()

func set_task(task_id: int) -> void:
    current_task_id = clamp(task_id, 0, N_TASKS - 1)
    get_node("../TaskIndicator").text = ["REACH", "AVOID+REACH", "TIMED"][current_task_id]

func get_obs() -> Array:
    var agent := get_node("../Agent") as CharacterBody3D
    var goal := get_node("../Goal") as Area3D
    var to_goal: Vector3 = goal.global_position - agent.global_position

    # Base observation (8 values)
    var obstacle_delta := _nearest_obstacle_delta()
    var base_obs := [
        to_goal.x / 20.0,          # normalised delta X
        to_goal.z / 20.0,          # normalised delta Z
        agent.velocity.x / 10.0,
        agent.velocity.z / 10.0,
        obstacle_delta.x / 20.0,
        obstacle_delta.z / 20.0,
        float(_step_count) / float(MAX_EPISODE_STEPS),
        to_goal.length() / 20.0,   # distance to goal (scalar)
    ]

    # One-hot task ID prepended (3 values)
    var task_one_hot := [0.0, 0.0, 0.0]
    task_one_hot[current_task_id] = 1.0

    return task_one_hot + base_obs  # total: 11 values

func get_action_space() -> Dictionary:
    return {
        "move": {"size": 2, "action_type": "continuous"}
    }

func set_action(action: Dictionary) -> void:
    var agent := get_node("../Agent") as CharacterBody3D
    var move := action["move"] as Array
    agent.velocity.x = move[0] * 6.0
    agent.velocity.z = move[1] * 6.0
    _step_count += 1

func get_reward() -> float:
    var agent := get_node("../Agent") as CharacterBody3D
    var goal := get_node("../Goal") as Area3D
    var dist := agent.global_position.distance_to(goal.global_position)

    var reward := 0.0

    # Component shared across all tasks: dense distance reward
    reward += -dist * 0.01

    # Task-specific components
    match current_task_id:
        0:
            # Reach goal: bonus for reaching
            if dist < 1.0:
                reward += 1.0
        1:
            # Avoid obstacles: penalty per collision
            if dist < 1.0:
                reward += 1.0
            reward += -0.5 * float(_current_collision_count())
        2:
            # Time pressure: bonus decays linearly with remaining steps
            if dist < 1.0:
                var steps_remaining := MAX_EPISODE_STEPS - _step_count
                var time_bonus := float(steps_remaining) / float(TIME_PRESSURE_LIMIT)
                reward += 1.0 + clamp(time_bonus, 0.0, 2.0)

    return reward

func get_done() -> bool:
    var agent := get_node("../Agent") as CharacterBody3D
    var goal := get_node("../Goal") as Area3D
    var dist := agent.global_position.distance_to(goal.global_position)
    return dist < 1.0 or _step_count >= MAX_EPISODE_STEPS

func reset() -> void:
    _step_count = 0
    _episode_reward = 0.0
    _randomise_episode()

func _randomise_episode() -> void:
    # Randomise goal position
    get_node("../Goal").global_position = Vector3(
        randf_range(-8.0, 8.0), 0.5, randf_range(-8.0, 8.0)
    )
    # Spawn obstacles only for tasks 1 and 2
    _obstacle_positions.clear()
    if current_task_id >= 1:
        for i in range(randi_range(2, 5)):
            _obstacle_positions.append(Vector3(
                randf_range(-7.0, 7.0), 0.5, randf_range(-7.0, 7.0)
            ))
    get_node("../ObstacleSpawner").rebuild(_obstacle_positions)

func _nearest_obstacle_delta() -> Vector3:
    if _obstacle_positions.is_empty():
        return Vector3(20.0, 0.0, 20.0)  # far away — no obstacle
    var agent_pos := get_node("../Agent").global_position
    var nearest := _obstacle_positions[0]
    for pos in _obstacle_positions:
        if pos.distance_to(agent_pos) < nearest.distance_to(agent_pos):
            nearest = pos
    return nearest - agent_pos

func _current_collision_count() -> int:
    # Count obstacles within collision radius
    var agent_pos := get_node("../Agent").global_position
    var count := 0
    for pos in _obstacle_positions:
        if pos.distance_to(agent_pos) < 1.2:
            count += 1
    return count
```

### Python training script with task rotation

```python
# train_multitask.py
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

N_TASKS = 3
STEPS_PER_ROTATION = 4096  # collect this many steps per task before rotating


class TaskRotationCallback(BaseCallback):
    """Rotates the active task ID in the Godot environment every K steps."""

    def __init__(self, n_tasks: int, steps_per_rotation: int):
        super().__init__()
        self.n_tasks = n_tasks
        self.steps_per_rotation = steps_per_rotation
        self._current_task = 0
        self._steps_since_rotation = 0

    def _on_step(self) -> bool:
        self._steps_since_rotation += 1
        if self._steps_since_rotation >= self.steps_per_rotation:
            self._steps_since_rotation = 0
            self._current_task = (self._current_task + 1) % self.n_tasks
            # Signal the Godot environment to switch tasks
            # (environment must expose a set_task method or info channel)
            self.training_env.env_method("set_task", self._current_task)
            self.logger.record("curriculum/active_task", self._current_task)
        return True


env = StableBaselinesGodotEnv(
    env_path="builds/multitask_env.x86_64",
    n_parallel=4,
    speedup=10,
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    tensorboard_log="logs/multitask_ppo/",
    policy_kwargs=dict(net_arch=[256, 256]),
)

callback = TaskRotationCallback(n_tasks=N_TASKS, steps_per_rotation=STEPS_PER_ROTATION)
model.learn(total_timesteps=3_000_000, callback=callback)
model.save("models/multitask_ppo_final")
```

!!! check "Done when"
    After `train_multitask.py` finishes, run the per-task evaluation from Section 8
    (100 episodes per task): a well-trained policy hits the ≥ 80 % success bar on
    **every** task, as in the bar chart from "Three ways to see your AI". The bar is
    the unit's target, not a seed-proof guarantee — one task plateauing just short
    while all curves still climb together is a tuning matter (Section 4's weighting
    knobs), not a failed run. But if one curve drops while another keeps climbing,
    that is negative transfer, not a run that needs more steps: work through Section 4
    (check reward scales first) before training longer.

---

## 7 · Multi-Task SAC (optional on a first read)

!!! note "First pass? Skim or skip this section."
    The core path through this unit is Sections 1–6 (why goal conditioning isn't
    enough, task encodings, multi-task PPO, negative transfer, task curricula, the
    Godot example), Section 8 (per-task evaluation) and Section 10 (the multi-task
    vs. separate-policies decision). SAC is
    an off-policy alternative that buys sample efficiency — come back to it once your
    PPO run works.

PPO is on-policy: it collects fresh rollouts before each update and then discards them.
In multi-task settings, this is wasteful — experience from task 0 is discarded before
task 1 begins collecting.

**SAC** (Soft Actor-Critic) is off-policy: it stores all experience in a replay buffer and
can mix experience from all tasks in every gradient update. For multi-task training this is
a significant advantage: a single replay buffer shared across tasks gives every task's
gradient information about every other task's transitions.

### Task-conditioned SAC architecture

The critic must condition on both the observation and the task encoding:

```python
from stable_baselines3 import SAC
import gymnasium as gym
import numpy as np

# SAC with task-ID obs uses the same TaskIDWrapper as PPO
# — SAC's MultiInputPolicy or MlpPolicy handles the concatenated obs identically

model = SAC(
    "MlpPolicy",
    env,                         # TaskIDWrapper-wrapped env
    verbose=1,
    learning_rate=3e-4,
    buffer_size=1_000_000,       # shared replay buffer across all tasks
    learning_starts=10_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",             # automatic entropy tuning
    tensorboard_log="logs/multitask_sac/",
    policy_kwargs=dict(net_arch=[256, 256]),
)
```

!!! tip "SAC sample efficiency advantage"
    In a head-to-head comparison on the three-task Godot example, SAC typically achieves
    the same per-task success rate as PPO in roughly half the environment steps. The cost:
    SAC requires more memory (the replay buffer) and is harder to tune (actor-critic
    instabilities, entropy coefficient sensitivity). For a Godot environment running at
    10× speedup, the wall-clock time difference is usually small — use PPO if you are
    getting started and SAC if you need the last few percent of sample efficiency.

### Mixed-task replay sampling

By default, SB3 SAC samples from the replay buffer uniformly. For multi-task training,
it helps to sample uniformly across tasks (not uniformly over all transitions, which would
under-sample rare tasks):

!!! warning "Pseudocode — not runnable as-is"
    The class below illustrates the concept. Full SB3 integration requires subclassing
    `ReplayBuffer` and overriding `sample` with the proper SB3 buffer API.
    Do not paste this into a training script without completing those details.

```python
class MultiTaskReplayBuffer:
    """Wraps SB3 ReplayBuffer to ensure uniform task sampling."""

    def __init__(self, base_buffer, n_tasks):
        self.buffers = [base_buffer.__class__(...) for _ in range(n_tasks)]
        self.n_tasks = n_tasks

    def add(self, obs, next_obs, action, reward, done, infos):
        task_id = self._extract_task_id(obs)
        self.buffers[task_id].add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size):
        per_task = batch_size // self.n_tasks
        batches = [buf.sample(per_task) for buf in self.buffers]
        return self._concatenate_batches(batches)
```

---

## 8 · Evaluation

A multi-task policy must be evaluated per task, not as a single aggregate.

### Per-task success rate

```python
def evaluate_multitask(model, task_envs: dict, n_episodes: int = 100) -> dict:
    """
    Args:
        task_envs: {task_id: gym.Env} mapping
    Returns:
        {task_id: success_rate}
    """
    results = {}
    for task_id, env in task_envs.items():
        successes = 0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            if info.get("is_success", False):
                successes += 1
        results[task_id] = successes / n_episodes
    return results
```

### Forward transfer

**Forward transfer** measures whether learning task A helped when learning task B later.
The metric compares:

- `AUC(single-task B)` — the area under the learning curve when training on task B from
  scratch.
- `AUC(multi-task B)` — the area under the learning curve for task B when trained jointly
  with task A.

Positive forward transfer: multi-task B converges faster (higher AUC for the same number
of steps). Negative transfer: multi-task B converges slower.

```python
def forward_transfer(single_task_curve, multitask_curve) -> float:
    """Returns > 0 for positive transfer, < 0 for negative transfer."""
    auc_single = np.trapz(single_task_curve)
    auc_multi = np.trapz(multitask_curve)
    return (auc_multi - auc_single) / auc_single
```

### Zero-shot generalisation

After training on tasks 0, 1, 2, evaluate the policy on a held-out task 3 (e.g., reach
goal while avoiding obstacles AND under time pressure) with no further training:

```python
# Construct task 3 encoding from seen task parameter vectors
# (only possible with task-parameter encoding, not one-hot)
task_3_params = np.array([1.0, 0.5, 1.5])  # novel combination of seen parameters
obs_with_task = np.concatenate([task_3_params, raw_observation])
action, _ = model.predict(obs_with_task, deterministic=True)
```

One-hot encoding cannot generalise to held-out task IDs at all. Task-parameter encoding
generalises by interpolation (within the training distribution of parameters). Language
embedding generalises furthest, to semantically similar but never-seen instructions.

### Honest evaluation protocol

| Metric | What it measures | When to report it |
|---|---|---|
| Per-task success rate | Does the policy solve each trained task? | Always |
| Aggregate reward | Summary across tasks | Useful but mask negative transfer; report alongside per-task |
| Forward transfer ratio | Did multi-task help vs single-task? | When comparing vs N separate policies |
| Zero-shot success rate | Does the policy generalise to unseen tasks? | Only if task-parameter or language encoding is used |

---

## 9 · Connection to Foundation Models (optional on a first read)

!!! note "First pass? Skim or skip this section."
    Nothing here is needed for the hands-on exercise. The core path is Sections 1–6,
    Section 8 and Section 10; this section is a conceptual outlook linking the unit's
    task-conditioned policy to generalist agents such as Gato and RT-2 — read it for
    context once your own multi-task agent trains.

Multi-task RL is the conceptual precursor to the most ambitious goal in the field:
**generalist agents** that can follow instructions and solve tasks they have never
explicitly trained on.

### The scaling argument

As the number of training tasks N grows:

- N = 3 → Multi-task RL (this unit)
- N = 100 → Meta-RL: the policy learns to *adapt* to new tasks at inference time
- N = 600 → Gato (DeepMind, 2022): one transformer trained on Atari, robotics, captioning,
  question answering, and text dialogue simultaneously
- N = ∞ → Hypothetical generalist agent trained on all tasks expressible as language

The core finding from Gato: a single large transformer with task-conditioned inputs
achieves competitive performance on hundreds of diverse tasks. The multi-task policy from
this unit is architecturally identical to a tiny Gato — the same structure, just without
the scale.

### Instruction-following robots

Language embedding (section 2) bridges multi-task RL to robotics systems like:

- **RT-2** (Google DeepMind, 2023) — vision-language-action model; language commands
  condition robot manipulation policies.
- **SayCan** (Google, 2022) — LLM generates task plans; each step is executed by a
  specialised skill policy trained with RL.

Both rely on the same principle: a task-conditioned policy where the conditioning vector
is derived from natural language. The gap between the Godot exercise in this unit and these
systems is primarily scale and pre-training data, not architectural novelty.

### Practical takeaway

Multi-task RL with language embeddings in a Godot game is training a generalist agent in
miniature. The habits formed here — careful per-task evaluation, negative transfer
monitoring, curriculum design — are exactly the habits needed to work with larger systems
at research scale.

---

## 10 · When to Use Multi-Task RL vs Separate Policies

This is the question the unit opened with. Here is a structured answer.

### Use separate policies when

- N ≤ 5 tasks and you can afford to train and store N checkpoints.
- Tasks are qualitatively very different (different observation spaces, different action
  spaces) — the shared trunk provides little benefit.
- Per-task performance is more important than generalisation. Specialists outperform
  generalists on their own task almost every time.
- You want the simplest possible debugging story. Each policy's failure is isolated.

### Use multi-task RL when

- N is large (> 10) and training N separate policies is infeasible.
- Inference time or memory is constrained (one model fits where N models do not).
- You need runtime task switching without loading a new checkpoint.
- You want forward transfer: you have evidence (or strong belief) that some tasks will help
  others.
- You are building toward zero-shot generalisation to held-out tasks — this requires
  multi-task training almost by definition.

### The benchmark comparison you should always run

Before committing to multi-task training, run this comparison:

```python
# 1. Train separate PPO policies for each task
single_task_results = {}
for task_id in range(N_TASKS):
    model = PPO("MlpPolicy", make_task_env(task_id)(), verbose=0)
    model.learn(total_timesteps=1_000_000)
    single_task_results[task_id] = evaluate_task(model, task_id)

# 2. Train one multi-task PPO policy
multitask_model = PPO("MlpPolicy", multitask_env, verbose=0)
multitask_model.learn(total_timesteps=1_000_000 * N_TASKS)  # same total budget
multitask_results = evaluate_multitask(multitask_model, task_envs)

# 3. Compare per-task success rates
for task_id in range(N_TASKS):
    delta = multitask_results[task_id] - single_task_results[task_id]
    print(f"Task {task_id}: single={single_task_results[task_id]:.2f}, "
          f"multi={multitask_results[task_id]:.2f}, delta={delta:+.2f}")
```

If the multi-task policy matches or beats all single-task policies: multi-task wins.
If any task is substantially worse in multi-task: investigate negative transfer before
deploying the multi-task policy.

---

## 11 · Stretch Goals

### Stretch 1 — Gradient Surgery (PCGrad)

Implement PCGrad (Yu et al. 2020, "Gradient Surgery for Multi-Task Learning"). For each
pair of tasks with a negative gradient dot product, project one task's gradient onto the
normal plane of the other before summing:

```python
import torch

def pcgrad_update(gradients: list[torch.Tensor]) -> torch.Tensor:
    """
    gradients: list of per-task gradient vectors (one per task, flat)
    Returns: combined gradient with conflicts resolved
    """
    n_tasks = len(gradients)
    combined = torch.zeros_like(gradients[0])
    for i in range(n_tasks):
        g_i = gradients[i].clone()
        for j in range(n_tasks):
            if i == j:
                continue
            g_j = gradients[j]
            dot = torch.dot(g_i, g_j)
            if dot < 0:
                # Project g_i to remove the component in direction of g_j
                g_i -= (dot / (g_j.norm() ** 2 + 1e-8)) * g_j
        combined += g_i
    return combined
```

Plug this into the SB3 PPO training loop by subclassing `OnPolicyAlgorithm.train()`.
Compare per-task performance curves with and without PCGrad on the three-task Godot
environment. Report whether the regression curve for any task disappears.

### Stretch 2 — Held-Out Task Generalisation

Extend the Godot environment with a fourth task: reach the goal while avoiding obstacles
AND under time pressure (combining tasks 1 and 2). Train only on tasks 0–2. Evaluate
zero-shot on task 3 with:

1. One-hot encoding (cannot generalise — use a new index and observe failure).
2. Task parameter encoding (should partially generalise by interpolation).
3. Language embedding (if GPU available — should generalise best).

Record the success rates for each encoding type on the held-out task and write a one-page
analysis.

### Stretch 3 — Multi-Task SAC vs Separate SAC

Run a controlled experiment comparing wall-clock and sample efficiency:

| Condition | Setup |
|---|---|
| Separate SAC | Train one SAC per task, 1 M steps each |
| Multi-task SAC | Train one SAC on all 3 tasks, 3 M steps total |
| Multi-task SAC (shared buffer) | Same, but force uniform task sampling from replay buffer |

Measure: final success rate per task, training wall-clock time, total GPU memory.
Report which condition achieves the best per-task success rate per GPU-hour.

### Stretch 4 — Language-Conditioned Policy in Godot

Replace the one-hot task ID with a SBERT embedding of the task description. Wire
`sentence-transformers` into the Python training script, encode three task description
strings to 384-dim vectors, and pass them as the task encoding to the policy. Reduce the
embedding dimension to 8 with a learned linear projection (trained jointly with the
policy) to keep the observation vector small. Evaluate zero-shot on two new task
descriptions the policy has never seen.

---

[← Hierarchical RL](unit-hierarchical.md) · [Course home](index.md) · [→ Imitation Learning](unit-09.md)
