# Goal-Conditioned RL & Hindsight Experience Replay

[← Locomotion Agents](unit-locomotion.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~35 min · Training: ~30 min GPU / ~2 h CPU

!!! info "Three ways to see your AI"
    - **Godot viewport** — watch the arm reach toward a randomly spawned goal marker every episode; the goal moves, but the same policy handles it.
    - **TensorBoard** — plot `rollout/success_rate` vs `timesteps` for two runs side-by-side: SAC alone (flat at zero) vs SAC+HER (climbs to ~0.9).
    - **gymnasium FetchReach render** — `env.render()` shows a 3-DOF Fetch arm in MuJoCo reaching a colored target sphere; identical algorithm, different simulator.

---

## 1 · The manipulation problem

Up to this point in the course you have built robot environments in Godot (Unit Robotics), shaped dense rewards (Unit Reward Engineering), and even bolted on intrinsic curiosity to push agents past sparse-reward walls (Unit Curiosity). Manipulation tasks now stretch every one of those tools to breaking point.

Consider the canonical robotic instruction:

> "Pick up the red cube and place it on the plate."

What makes this hard?

- **Combinatorial goal space.** The cube can start anywhere on the table; the plate can sit at thousands of valid positions. A 1 m × 1 m table sampled at 1 cm resolution already gives 10 000 cube positions × 10 000 plate positions — a hundred million distinct task instances.
- **Binary reward.** Either the cube ends up on the plate (+1) or it does not (0). There is no partial credit for "almost".
- **Exploration is hopeless.** A random-policy arm does not place objects by accident. It flails, knocks the cube off the table, and earns zero reward for every one of its 500 episode steps.
- **Reward shaping does not scale.** You could hand-craft a dense reward — `-distance(cube, plate)` — but every new goal variation (a new plate, a stacking task, a peg insertion) requires a brand new reward function.

The core insight that drives this unit:

> **We want one policy that works for ANY goal position, not a separate policy for each goal.**

A policy that has been told "reach (0.5, 0.3, 0.2)" should behave differently from the same policy told "reach (-0.4, 0.1, 0.6)" — without retraining. The trick is to make the goal part of the observation, and to find a way to learn from sparse rewards.

That trick is **Hindsight Experience Replay**.

---

## 2 · Goal-conditioned RL

The first half of the solution is structural: extend the observation vector with the desired goal.

### Standard observation

```
s = [joint_angles, joint_velocities, end_effector_pos]
```

A policy `π(a | s)` trained on this observation can only solve one task — the task it was rewarded for during training.

### Goal-conditioned observation

```
s = [joint_angles, joint_velocities, end_effector_pos, goal_pos]
                                                       ^^^^^^^^
                                                       new!
```

A policy `π(a | s, g)` trained on this observation can — in principle — solve any task expressible as a goal vector `g`. The same network weights, queried with a different goal, produce different actions. We call this a **universal policy** or, in the formal literature, a **Universal Value Function Approximator (UVFA, Schaul et al. 2015)**.

### Example: goal-conditioned observation in Godot

```gdscript
# Goal-conditioned observation in Godot
func get_obs() -> Dictionary:
    # Robot state (proprioceptive)
    var robot_obs = [
        end_effector.global_position.x / reach,
        end_effector.global_position.y / reach,
        end_effector.global_position.z / reach,
        linear_velocity.x / max_vel,
        linear_velocity.y / max_vel,
        linear_velocity.z / max_vel,
    ]

    # Goal (changes every episode)
    var goal_obs = [
        goal.global_position.x / reach,
        goal.global_position.y / reach,
        goal.global_position.z / reach,
    ]

    return {"obs": robot_obs + goal_obs}

func reset():
    # Randomize goal position each episode
    goal.global_position = Vector3(
        randf_range(-reach * 0.8, reach * 0.8),
        randf_range(0.1, reach * 0.5),
        randf_range(-reach * 0.8, reach * 0.8),
    )
    _ai.needs_reset = false
```

Two things are worth highlighting:

1. **The goal is sampled fresh every episode.** If you keep the goal fixed, the agent collapses back to a single-task learner that ignores the goal channel — there is no incentive to attend to it.
2. **Reward is purely sparse.** The reward function the agent sees is:

```gdscript
func get_reward() -> float:
    var d = end_effector.global_position.distance_to(goal.global_position)
    return 1.0 if d < 0.05 else 0.0
```

No distance shaping, no proxy term, no curriculum trickery. Just success or failure. This is where most learners die — and where HER will save us.

---

## 3 · Why sparse goal-conditioned RL fails without HER

Before reaching for HER, let us be honest about how badly vanilla RL fails on this kind of task. The math is grim.

### The volume argument

- The arm has 6 DOF × 2 (angle + velocity) = 12 obs dims, plus 3 goal dims = **15-dim observation**.
- Success is defined as end-effector within 5 cm of goal.
- If arm reach is ≈ 1 m, the success region's volume relative to the workspace is roughly `(0.05 / 1.0)^3 ≈ 0.0125 %`.
- A random policy lands in the success region roughly **once every 10 000 episodes**.
- At 500 steps per episode, you would need ≈ **5 million environment steps before seeing a single positive reward**.

### Why standard algorithms time out

- **PPO** updates from on-policy rollouts. If every rollout returns reward 0, the advantage estimates are zero and the policy gradient is zero. PPO simply does not move.
- **SAC** stores transitions in a replay buffer, but if no transition has reward > 0, the Q-targets are all zero and the critic learns the constant function `Q(s, a) = 0`. The actor then optimizes against a flat landscape and never improves.

### Why curiosity does not save us

In Unit Curiosity you saw that ICM/RND solves sparse-reward exploration tasks like Montezuma's Revenge by rewarding novelty. But manipulation has a subtly different problem: the arm needs to reach a **specific** goal, not just visit novel states. Curiosity drives the arm to wave around in interesting configurations, none of which happens to be the goal the supervisor cares about today.

We need a method that re-uses the data we already have — even when that data records failures — to teach the agent something concrete. That method is HER.

---

## 4 · Hindsight Experience Replay (HER)

> **The key insight (Andrychowicz et al., NeurIPS 2017):** even failed episodes contain useful information.

### A worked example

Imagine the arm is told to reach goal `G = (0.5, 0.3, 0.2)`. After 500 steps it ends up at `G' = (0.4, 0.25, 0.15)`. The reward function says `r = 0` — failure.

But pause for a moment. The arm **did** reach a position. It just was not the one we asked for. What if we lied to the replay buffer and told it the original instruction had actually been "reach `(0.4, 0.25, 0.15)`"? Then the very same trajectory would be a textbook success: zero reward for 499 steps, then +1 on the final step.

That is hindsight relabeling. It is the philosophical equivalent of "I meant to do that".

### The algorithm

1. **Roll out** an episode with the originally sampled goal `G`. Collect transitions:

   ```
   (s_0, a_0, r_0, s_1), (s_1, a_1, r_1, s_2), ..., (s_T, a_T, r_T, s_{T+1})
   ```

   Each `s_t` already contains `G` (because the observation is goal-conditioned).

2. **Store as-is** in the replay buffer. These transitions will mostly have `r = 0`.

3. **Hindsight relabel.** Pick a substitute goal `G' = achieved_goal(s_{T+1})` — i.e. the state the arm actually reached. Walk through the same trajectory and produce **new transitions** in which the goal channel is replaced with `G'` and the reward is **recomputed** using the same reward function:

   - Steps where the achieved goal is still far from `G'`: `r = 0`.
   - The final step (where by construction the achieved goal equals `G'`): `r = 1`.

4. **Store the relabeled transitions** alongside the originals. Train SAC/DDPG/TD3 on the combined buffer.

Every episode — success or failure — now contributes at least one positive-reward transition to learning.

### The transformation in pseudocode

```python
# Original failed trajectory stored as:
# (obs=[robot_state, goal_G], action, reward=0, next_obs)  × T times
# (obs=[robot_state, goal_G], action, reward=0, next_obs)  ← last step, still failed

# Hindsight relabeled as:
# (obs=[robot_state, goal_G'], action, reward=0, next_obs) × T-1 times
# (obs=[robot_state, goal_G'], action, reward=1, next_obs) ← last step, "success" on G'
```

### Why this works

The agent is not learning to give up. It is learning a general skill: **"to reach any position I have ever reached before."** Because the relabeled goals are drawn from the achieved-goal distribution of the agent's own trajectories, the curriculum naturally starts easy (reach the messy places the random policy stumbles into) and grows harder as the policy improves (the trajectories themselves get more deliberate, so the relabeled goals become more meaningful).

This is, in effect, an automatically generated curriculum — without any human design effort.

!!! tip "Compatible algorithms"
    HER is a **replay-buffer trick**, not a standalone algorithm. It plugs into any off-policy method: DDPG (the original paper), TD3, SAC. It does **not** combine with PPO, A2C, or any on-policy method, because on-policy algorithms throw away old trajectories.

---

## 5 · HER goal selection strategies

The relabeling step asks: *which* state should we use as the substitute goal `G'`? The original paper benchmarked four strategies:

| Strategy | How `G'` is chosen | Effect |
|----------|--------------------|--------|
| `final` | Last state of the episode | Simple, often good enough |
| `future` | Random state from *later* in the same episode | More diverse goals, usually best |
| `episode` | Random state from anywhere in the same episode | Uniform over episode |
| `random` | Random state from the entire replay buffer | Maximum diversity, slower |

For each real transition, HER typically generates **k** additional relabeled transitions (`k = 4` is the standard default). With `future` + `k = 4`, every real transition spawns four hindsight copies, each using a different randomly-chosen future state from the same episode as the substitute goal.

!!! tip "Recommended default"
    Use `goal_selection_strategy="future"` with `n_sampled_goal=4`. This is what Stable-Baselines3, RL-Zoo, and most published HER benchmarks use, and it is rarely worth tuning unless your task is unusual.

### Intuition for each strategy

- **`final`**: every relabeled goal is what the arm ended up touching at the end of the episode. Simple but biases the curriculum toward "stop where you happen to stop".
- **`future`**: for a transition at time `t`, sample some `t' > t` from the same trajectory and use `achieved_goal(s_{t'})` as the goal. This means each transition is asked: "Given the action you took at time `t`, would you have made progress toward this later state you actually reached?" The answer is yes, by construction — so the value function gets dense, consistent signal.
- **`episode`**: like `future` but allows `t' < t`. Less principled (you cannot reach a goal in the past), but sometimes works for non-temporal goal spaces.
- **`random`**: relabel using any goal from the entire buffer. Maximum diversity, but most of those goals are irrelevant to the current trajectory, so most relabeled transitions are uninformative.

---

## 6 · Hands-on: HER with gymnasium-robotics

We will warm up in Python before touching Godot. The `gymnasium-robotics` package ships the Fetch suite — a MuJoCo-based 7-DOF arm whose observation space is already a goal-conditioned dict, perfectly aligned with Stable-Baselines3's `HerReplayBuffer`.

### Install

```bash
conda activate godot_env
pip install gymnasium-robotics
```

If you have not yet installed the MuJoCo bindings (transitive dependency), follow the prompts; on macOS and Linux it is automatic.

### Full working HER training on FetchReach

```python
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

# FetchReach: 3-DOF arm, reach a target position
# obs includes: observation (25-dim), achieved_goal (3-dim), desired_goal (3-dim)
env = gym.make("FetchReach-v3")

model = SAC(
    "MultiInputPolicy",    # handles dict obs with obs + goal keys
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,       # k — relabel each transition 4 times
        goal_selection_strategy="future",
    ),
    verbose=1,
    tensorboard_log="logs/",
    learning_rate=1e-3,
    buffer_size=1_000_000,
    learning_starts=1000,
    batch_size=256,
    gamma=0.95,
    tau=0.005,
)

model.learn(total_timesteps=500_000, tb_log_name="her_fetchreach")
model.save("fetchreach_her")
```

### What to watch in TensorBoard

Open TensorBoard in another terminal:

```bash
tensorboard --logdir logs/
```

The metrics that matter:

- `rollout/success_rate` — should climb from 0 to ≈ 0.95 over training.
- `rollout/ep_rew_mean` — Fetch rewards are `-1` per step until success, so this rises from `-50` (timed-out episodes) toward `0`.
- `train/critic_loss` — should *not* be flat at zero. If it is, HER's relabeling is not producing positive rewards (check `compute_reward`).

### Expected result

FetchReach (3-DOF reach) is solved in ~200 k steps with HER. Without HER (set `replay_buffer_class=None`), SAC typically needs 2–5× more steps, and on harder tasks below it fails entirely.

!!! check "Done when"
    `rollout/success_rate` on FetchReach climbs off the floor and approaches the unit's stated ≈ 0.9–0.95 level around the ~200 k-step mark, with `rollout/ep_rew_mean` rising from `-50` toward `0` — a run in that neighbourhood is solved; don't fail it over the last few hundredths. If you also run the no-HER baseline (`replay_buffer_class=None`) on the same budget, its curve should sit visibly below — without HER, SAC needs 2–5× more steps. A `success_rate` still flat at zero after the full budget is not a "train longer" problem: first confirm `train/critic_loss` is not flat at zero (if it is, relabeling is producing no positive rewards), then work the checklist in Section 11, starting with the bit-for-bit `compute_reward` consistency check.

### A harder task: FetchPush

```python
env = gym.make("FetchPush-v3")
# Same code, just change the env — HER handles both
model.learn(total_timesteps=1_000_000)
```

FetchPush replaces the empty-air target with a cube that must be **pushed** to the goal. The arm must make contact, friction must align, and the cube must slide. Without HER this task is essentially unlearnable in under 10 M steps; with HER it solves in ≈ 1 M.

---

## 7 · HER in Godot (goal-conditioned wrapper)

Now back to your Godot robot. The Godot env from Unit Robotics emits flat float observations through the godot-rl bridge. To feed HER we need to repackage that flat vector into the dict structure that `HerReplayBuffer` expects: `{"observation", "achieved_goal", "desired_goal"}`.

### Required convention on the Godot side

In your `get_obs()`, agree to emit a flat vector with a known layout:

```
[robot_state (obs_dim)] [achieved_goal (3)] [desired_goal (3)]
```

`achieved_goal` is `end_effector.global_position` (normalized); `desired_goal` is `goal.global_position` (normalized). The wrapper below will split this for SB3.

### The wrapper

```python
# Godot → HER compatible wrapper
import gymnasium as gym
import numpy as np
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

class GoalConditionedGodotEnv(gym.Env):
    """Wraps a Godot env that returns [robot_state (n), achieved_goal (3), desired_goal (3)]."""

    def __init__(self, env_path, obs_dim=12, goal_dim=3):
        self.env = StableBaselinesGodotEnv(env_path=env_path, n_parallel=1, speedup=20)
        self.obs_dim   = obs_dim
        self.goal_dim  = goal_dim
        total_dim = obs_dim + goal_dim * 2

        self.observation_space = gym.spaces.Dict({
            "observation":    gym.spaces.Box(-np.inf, np.inf, (obs_dim,)),
            "achieved_goal":  gym.spaces.Box(-np.inf, np.inf, (goal_dim,)),
            "desired_goal":   gym.spaces.Box(-np.inf, np.inf, (goal_dim,)),
        })
        self.action_space = self.env.action_space

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (distance < 0.05).astype(np.float32) - 1.0  # -1/0 sparse reward

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        obs_dict = self._split_obs(obs)
        reward = self.compute_reward(obs_dict["achieved_goal"], obs_dict["desired_goal"], info)
        return obs_dict, reward, done, info

    def _split_obs(self, flat_obs):
        robot  = flat_obs[:self.obs_dim]
        achieved = flat_obs[self.obs_dim:self.obs_dim + self.goal_dim]
        desired  = flat_obs[self.obs_dim + self.goal_dim:]
        return {"observation": robot, "achieved_goal": achieved, "desired_goal": desired}
```

### Why `compute_reward` must be a method

This is the single most important detail when integrating HER. The `HerReplayBuffer` will call `env.compute_reward(achieved_goal, desired_goal, info)` from the **outside**, with vectorized arrays of relabeled goals — *not* with the goal the agent actually saw. Your reward must therefore:

1. Depend only on `(achieved_goal, desired_goal, info)`, never on hidden environment state.
2. Be vectorized — accept batched goal arrays and return a batched reward.
3. Match exactly what the environment returns during `step` — otherwise the relabeled transitions are inconsistent and training diverges.

### Training call

```python
env = GoalConditionedGodotEnv(env_path="builds/arm.exe", obs_dim=12, goal_dim=3)
model = SAC(
    "MultiInputPolicy", env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
    verbose=1, tensorboard_log="logs/",
    learning_rate=1e-3, buffer_size=1_000_000,
    learning_starts=1000, batch_size=256, gamma=0.95, tau=0.005,
)
model.learn(total_timesteps=500_000, tb_log_name="her_godot_arm")
```

---

## 8 · Beyond reaching: FetchPush, FetchPickAndPlace, FetchSlide

Switching environments while keeping the same training code is one of the joys of goal-conditioned RL. Here is the standard difficulty progression:

| Task | Difficulty | What's hard |
|------|------------|-------------|
| `FetchReach-v3` | Easy | Arm reaches empty space |
| `FetchPush-v3` | Medium | Must contact and push an object |
| `FetchPickAndPlace-v3` | Hard | Grasp (close gripper) + lift + place |
| `FetchSlide-v3` | Hard | Push object that slides on frictionless surface |

All four use the same HER code from §6 — only the environment name changes. Sample budgets vary:

- FetchReach: ≈ 200 k steps
- FetchPush: ≈ 1 M steps
- FetchPickAndPlace: 1–2 M steps
- FetchSlide: 1–2 M steps (often the worst, because pushing a sliding puck requires precise contact timing)

These numbers assume `n_sampled_goal=4`, `future` strategy, and a single environment. With vectorized environments (e.g. 8 parallel sims) wall-clock training time drops roughly linearly.

---

## 9 · When HER helps (and when it doesn't)

### HER works when…

- **The task is goal-conditioned.** The goal changes per episode and is part of the observation.
- **Reward is sparse and binary.** Success or failure, no shaping.
- **Reward is a function of `(achieved_goal, desired_goal)` alone.** This is what makes relabeling sound — any trajectory that reaches a state can be relabeled as a success for that state.

### HER does NOT help when…

!!! warning "HER does not help dense-reward tasks"
    If you already have a useful dense reward (e.g. `-distance(end_effector, goal)`), HER provides little extra signal — the agent is already getting gradient on every step. Worse, mixing dense reward with HER relabeling can yield inconsistent reward values for the same `(state, action, next_state)` triple across the buffer, which destabilizes the critic. **Use HER with sparse rewards, or not at all.**

Additional failure modes:

- **Trivial goal space.** If only one goal exists, hindsight relabeling collapses to the original problem.
- **Reward depends on path, not endpoint.** If the reward function includes terms like "minimum jerk" or "energy expended along the trajectory", you cannot recompute it from `(achieved_goal, desired_goal)` alone. HER's relabeled rewards would be incorrect.
- **Non-stationary goal semantics.** If "reaching position X" means different things in different episodes (e.g. because obstacles move), the relabeled transitions teach the wrong lesson.

---

## 10 · Stretch goals

1. **Solve FetchPickAndPlace.** The hardest standard Fetch task. Budget 1–2 M steps with HER. Then disable HER (`replay_buffer_class=None`) and run for the same budget; compare `success_rate` curves. The gap is usually the difference between "almost 1.0" and "exactly 0.0".
2. **Multi-goal training in Godot.** Spawn 5 candidate goal markers per episode. Pick one as the desired goal, but record the achieved end-effector position. Use HER with `future` strategy — and observe that the policy generalizes to any of the 5 markers without retraining.
3. **Read the original HER paper.** [Andrychowicz et al. 2017, "Hindsight Experience Replay"](https://arxiv.org/abs/1707.01495). Only 10 pages, and the introduction's analogy to a child learning to slide a hockey puck is one of the best motivating examples in the RL literature.
4. **Implement HER from scratch.** Subclass `ReplayBuffer` in SB3 and write your own `sample()` that injects relabeled transitions. You will appreciate the elegance after seeing how few lines it takes.
5. **Combine HER with curiosity.** For genuinely sparse + exploration-hard tasks (e.g. reach goals hidden behind doors), pair HER with RND from Unit Curiosity. Each addresses a different failure mode.

---

## 11 · Debugging an HER run

Even with the right algorithm, HER training has its own characteristic failure modes. Here is a checklist when training fails to converge.

### The success rate stays at zero

- **Cause #1: `compute_reward` is inconsistent.** Print the reward returned by `step()` and compare it to `compute_reward(achieved, desired, info)` called with the same arguments. They must match bit-for-bit. If they differ, HER's relabeled rewards do not match the agent's actual rewards, and learning silently breaks.
- **Cause #2: success threshold too tight.** A 1 cm threshold on a 1 m arm is borderline unreachable. Loosen to 5 cm during training, tighten at evaluation time.
- **Cause #3: goal sampling outside the reachable workspace.** If `randf_range` generates goals beyond the arm's kinematic reach, no relabeling can help.

### The critic loss explodes

- **Cause: reward magnitude mismatch with `gamma`.** With sparse `{-1, 0}` rewards and `gamma=0.99`, the Q-function can take values down to `-100`. Make sure your network's output scale and learning rate handle that magnitude. The recommended `gamma=0.95` in §6 caps the magnitude at `≈ -20`, which is friendlier.

### The agent learns to reach but never grasps (FetchPickAndPlace)

- **Cause: HER cannot relabel grasping.** The achieved goal is the cube's position, but grasping requires gripper-open + descend + gripper-close + lift. Hindsight relabeling tells the agent "if you had been asked to put the cube where it currently is, you succeeded" — which is trivially true if the cube never moved. Solutions: increase `n_sampled_goal` to 8, use a curriculum that initializes the cube already in the gripper for the first 100 k steps, or combine with demonstration data (DDPG+HER+Demonstrations, Nair et al. 2018).

---

## What's next

You now have an algorithm that can train a single, universal policy for any reachable target — sparse rewards and all. In the next unit you will face the final wall between simulation and the real world: **sim-to-real transfer**. A policy that solves FetchPush in MuJoCo or a Godot scene rarely works on a physical UR5 arm out of the box. We will look at domain randomization, system identification, and observation noise injection to bridge that gap.

Before moving on, make sure you can answer:

- Why does sparse goal-conditioned RL fail with vanilla SAC, and why does HER fix it?
- What three components must your environment's `compute_reward` satisfy for HER to work?
- Why is HER incompatible with PPO?
- When is HER the wrong tool, and what should you reach for instead?

[→ Sim-to-Real Transfer](unit-sim-to-real.md)
