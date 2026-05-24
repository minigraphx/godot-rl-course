# Multi-Task RL — One Policy for Multiple Tasks

Goal-conditioned RL teaches a policy to reach different targets. Multi-task RL goes further: one policy must solve **structurally different tasks** — different reward functions, different required skills, different environment physics. This is the foundation of generalist agents and curriculum learning.

[← Hierarchical RL](unit-hierarchical.md) · [Course home](index.md)

---

!!! info "Three ways to see your AI"
    Godot (same agent switching behaviors on task change) · TensorBoard (per-task `ep_rew_mean` curves — watch for negative transfer) · task-ID observation: print what task the agent thinks it's on

---

## 1 · Beyond goal conditioning

**Goal-conditioned RL** (see [HER unit](unit-her.md)) varies the *target* within one task structure: "reach position A" vs "reach position B" — same physics, same action space, same reward shape.

**Multi-task RL** varies the *task itself*:

| Variation | Goal-conditioned | Multi-task |
|-----------|----------------|------------|
| Target location | Yes | Yes |
| Reward function | No | Yes |
| Required skills | No | Yes |
| Physics/environment | No | Sometimes |
| Task count | Many goals | A few distinct tasks |

**Why one policy?** Separate policies per task are wasteful — they share structure (the physics, the observation space, the action format) but duplicate parameters. A multi-task policy amortizes learning across tasks, and can transfer knowledge: learning "balance while walking" in one task improves "balance while carrying" in another.

---

## 2 · Task representation — how the agent knows which task to solve

The agent needs to know which task is active. Three common encodings:

**One-hot task ID** — simplest. Prepend `[0,0,1]` for task 3 to the observation vector. The policy learns a separate behavior branch per task.

**Task parameter vector** — encode interpretable task properties: `[goal_distance, obstacle_density, time_limit]`. Generalizes better; the policy interpolates between seen task configurations.

**Language embedding** — encode the task description as a language vector. The foundation-model approach; enables zero-shot generalization to unseen task descriptions. Requires a large pretrained encoder.

For this unit we use **one-hot task IDs** — straightforward with any SB3 algorithm.

---

## 3 · Godot multi-task setup

Create three variants of the same scene, differing only in the AIController's reward function and goals:

```gdscript
# In AIController.gd — task_id is set by the scene manager or training wrapper
@export var task_id: int = 0  # 0 = reach_goal, 1 = avoid_obstacles, 2 = timed_race

func get_obs() -> Array:
    var obs = [
        linear_velocity.x,
        linear_velocity.y,
        distance_to_goal,
        nearest_obstacle_dist,
        time_remaining,
    ]
    # Append one-hot task encoding
    var task_onehot = [0.0, 0.0, 0.0]
    task_onehot[task_id] = 1.0
    obs.append_array(task_onehot)
    return obs

func get_reward() -> float:
    match task_id:
        0:  # Reach goal as fast as possible
            return -0.01 + (10.0 if reached_goal else 0.0)
        1:  # Reach goal while avoiding obstacles
            return (-0.01
                    - 1.0 * int(hit_obstacle)
                    + 10.0 * int(reached_goal))
        2:  # Reach goal under a tight time limit
            return (10.0 * int(reached_goal)
                    + 2.0 * int(reached_goal and time_remaining > 5.0))
        _:
            return 0.0
```

---

## 4 · Multi-task PPO with SB3

A custom observation wrapper prepends the task one-hot to any Godot env and randomizes the task each episode:

```python
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

NUM_TASKS = 3

class MultiTaskWrapper(gym.Wrapper):
    """Randomly samples a task at the start of each episode."""

    def __init__(self, env):
        super().__init__(env)
        obs_dim = env.observation_space.shape[0]
        # Observation already includes the task one-hot from GDScript
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        # The Godot AIController randomises task_id on reset internally
        return self.env.reset(**kwargs)

env = StableBaselinesGodotEnv(
    env_path="./MultiTask.x86_64",
    n_parallel=4,
    speedup=20,
)
env = MultiTaskWrapper(env)

model = PPO(
    "MlpPolicy", env,
    n_steps=2048,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="logs/multitask/",
)
model.learn(total_timesteps=3_000_000)
model.save("multitask_ppo")
env.close()
```

**TensorBoard setup for per-task monitoring:**

Log task-specific rewards using info dicts from the env:

```python
# In AIController.gd — include task_id in the info dict
func get_info() -> Dictionary:
    return {"task_id": task_id, "reached_goal": reached_goal}
```

```python
# In training callback — split ep_rew_mean by task_id
from stable_baselines3.common.callbacks import BaseCallback

class PerTaskCallback(BaseCallback):
    def _on_step(self):
        for info in self.locals["infos"]:
            task = info.get("task_id", -1)
            rew = info.get("episode", {}).get("r", None)
            if rew is not None:
                self.logger.record_mean(f"task_{task}/ep_rew_mean", rew)
        return True
```

---

## 5 · Negative transfer — when tasks hurt each other

If learning task A changes the policy in ways that hurt task B, you have **negative transfer**. Symptoms:

- Per-task `ep_rew_mean` curves diverge — one task improves while another degrades
- Adding a new task makes all existing tasks worse
- Gradient directions from different tasks are nearly opposite (gradient interference)

**Detecting negative transfer in TensorBoard:** watch per-task reward curves. If task 1 reaches 500 while tasks 2 and 3 drop from their solo-training baseline, tasks are interfering.

**Mitigation strategies:**

| Strategy | How | When to use |
|----------|-----|-------------|
| Task-specific heads | Shared trunk, separate output layers per task | Tasks share perception, differ in control |
| Gradient projection | Project conflicting gradients to shared subspace | When you can measure gradient interference |
| Curriculum | Train one task at a time, then mix | When one task is much harder than others |
| Separate policies | Don't multi-task | When tasks are truly orthogonal |

**Simple task-specific head with SB3:**

```python
policy_kwargs = {
    "net_arch": [{"pi": [128, 64], "vf": [128, 64]}],  # shared trunk handled by SB3
}
```

For more control, subclass `ActorCriticPolicy` and add task-conditional output layers.

---

## 6 · Curriculum over tasks

Start with the easiest task variant and introduce harder ones as competence grows:

```python
class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env, num_tasks=3):
        super().__init__(env)
        self.num_tasks = num_tasks
        self.active_tasks = 1   # start with task 0 only
        self.success_rate = 0.0

    def reset(self, **kwargs):
        # Unlock next task when current tasks are mastered
        if self.success_rate > 0.7 and self.active_tasks < self.num_tasks:
            self.active_tasks += 1
            print(f"Unlocked task {self.active_tasks - 1}")
        # Uniform sample from active tasks
        task_id = np.random.randint(0, self.active_tasks)
        # Pass task_id to Godot via env info or reset kwargs
        return self.env.reset(**kwargs)
```

**Automatic curriculum** (the research approach): train on tasks where performance is intermediate — not yet mastered (boring) and not impossible (uninformative). Monitor per-task success rate; weight sampling by `0.3 < success_rate < 0.7`.

---

## 7 · Multi-task SAC (off-policy)

For continuous-control multi-task problems, SAC is often more sample-efficient than PPO because the replay buffer stores transitions from all tasks, and each transition can be reused across many gradient updates:

```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy", env,
    buffer_size=500_000,   # shared buffer for all tasks
    learning_starts=10_000,
    verbose=1,
    tensorboard_log="logs/multitask_sac/",
)
model.learn(total_timesteps=2_000_000)
```

The replay buffer stores `(obs_with_task_id, action, reward, next_obs_with_task_id, done)` — the task ID is just another part of the observation. No special handling needed.

---

## 8 · Evaluation

**Per-task success rate** is more informative than aggregate reward:

```python
# Evaluate each task separately after training
for task_id in range(NUM_TASKS):
    successes = []
    for _ in range(50):
        obs = env.reset(task_id=task_id)  # force specific task
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)
        successes.append(info["reached_goal"])
    print(f"Task {task_id} success rate: {np.mean(successes):.2f}")
```

**Forward transfer:** train on tasks 0 and 1; measure zero-shot performance on task 2. Multi-task pre-training should give a head-start over scratch training.

---

## 9 · Connection to generalist agents

One-hot task IDs scale to natural language: replace the one-hot with a text embedding, and the policy can be conditioned on unseen task descriptions at test time. This is the architecture of agents like Gato (Reed et al. 2022) and SayCan (Ahn et al. 2022) — the conceptual bridge from multi-task RL to instruction-following robots.

Multi-task RL is the technical on-ramp: learn the skill of "do what the conditioning tells you" on a small, controllable set of tasks before scaling to language-conditioned policies.

---

## 10 · Stretch goals

- **Zero-shot transfer** — train on 3 tasks; test on a 4th variant (e.g., same reward but different map layout). Measure how much the multi-task policy outperforms a single-task baseline.
- **Automatic curriculum** — implement the intermediate-performance sampling strategy. Track which tasks are sampled most; verify easier tasks are sampled less as competence grows.
- **Negative transfer ablation** — train with task 2 (timed race) included vs excluded. Measure impact on tasks 0 and 1. Does the time-pressure task cause interference?
- **Task-specific heads** — subclass `ActorCriticPolicy` to add separate output layers per task while sharing the trunk. Compare to flat one-hot conditioning.

---

## What's next

**Imitation learning:** recorded expert demonstrations as an alternative learning signal — no multi-task conditioning needed when a human can show the agent what to do.

[→ Imitation Learning](unit-09.md)

---

[← Hierarchical RL](unit-hierarchical.md) · [Course home](index.md) · [→ Imitation Learning](unit-09.md)
