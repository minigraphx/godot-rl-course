# Capstone — Build Your Own RL Project

[← Ship Your Brain](unit-10.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~20 min · Training: varies

---

!!! info "Three ways to see your AI"
    Your own Godot environment — an agent you designed running in a scene you built · TensorBoard tracking a reward curve you wrote · the final itch.io page or desktop build you shipped to the world.

---

You have finished every unit. You know how Q-Learning bootstraps value estimates, how PPO clips its own gradient updates, why SAC maximises entropy, how curiosity drives exploration, how imitation learning bootstraps from demonstrations, and how to physically simulate a robot arm. Now the training wheels come off.

This unit is a **scaffold**, not a tutorial. There is no starter scene, no pre-written reward function, no reference implementation to check against. You will pick a problem, design the MDP, choose the algorithm, run the training pipeline, evaluate the result, and ship it. The sections below walk you through each decision in the order you will face it.

---

## 1 · Picking an Environment

The single most common capstone mistake is picking a scope that is too large. A two-room maze learned in one hour teaches you more about RL than a half-finished open-world game. Start smaller than you think you need to, ship it, and expand from there.

### Decision tree

```
Is the task physically continuous (arms, drones, cars)?
├── Yes → Robotics / physics sim path
│         Recommended: unit-robotics.md, unit-sac.md
│         Action space: continuous (Box)
│         Consider: unit-sim-to-real.md if you want real hardware
└── No  → Game / grid path
          ├── Turn-based or discrete movement?
          │   └── Yes → Discrete actions (Discrete or MultiDiscrete)
          │             Recommended: unit-q-learning.md, unit-03.md
          └── Real-time, smooth motion?
              └── Yes → Continuous or hybrid actions
                        Recommended: unit-ppo-deep.md, unit-sac.md
```

```
How many agents are there?
├── One → Single-agent (start here)
└── Multiple → Are they cooperative, competitive, or mixed?
    ├── Competitive (zero-sum) → self-play, unit-self-play.md
    ├── Cooperative → MAPPO / shared reward, unit-self-play.md
    └── Mixed → contact the course forum before proceeding
```

```
How dense is your reward signal?
├── Agent gets feedback every step → Dense reward → standard PPO / SAC
└── Agent only finds out at the end → Sparse reward
    ├── Episode < 200 steps → try shaped reward first (unit-reward-engineering.md)
    └── Episode > 200 steps → add curiosity (unit-curiosity.md) or HER (unit-her.md)
```

### Recommended starting points by goal

| Your goal | Environment type | Action space | Algorithm | Difficulty |
|---|---|---|---|---|
| Learn the full pipeline quickly | 2D grid game | Discrete | PPO | Beginner |
| Smooth locomotion | 2D or 3D physics | Continuous Box | SAC | Intermediate |
| Adversarial AI | Any 2-player game | Discrete or continuous | PPO + self-play | Intermediate |
| Long-horizon planning | Puzzle / strategy | Discrete | PPO + curiosity | Intermediate |
| Real robot | Physics sim | Continuous Box | SAC | Advanced |
| Emergent multi-agent behaviour | Any | Discrete or continuous | MAPPO / self-play | Advanced |

!!! warning "Scope creep is the #1 capstone killer"
    Pick **one** environment, **one** agent type, and **one** success criterion before you write a line of GDScript. If your project description contains the word "and" more than twice, cut it in half. You can always add complexity after your first working policy.

---

## 2 · Observation Design

The observation vector is everything the agent knows about the world at each timestep. Get it wrong and the agent either ignores most of its input (too much info) or can never solve the task because it can't tell where it is (too little info).

### The sensor checklist

Before adding a component to your observation vector, ask these questions for each candidate value:

1. **Does it change meaningfully with agent state?** If the value is nearly constant across all episodes, remove it.
2. **Is it observable in reality?** Don't give the agent information a real entity couldn't sense (e.g. the opponent's internal goal).
3. **Is it redundant with another component?** Highly correlated inputs waste capacity and slow learning.
4. **Is it normalised?** Neural networks expect inputs in a consistent range. Everything must be in `[-1, 1]` or `[0, 1]`.

### Normalisation in GDScript

```gdscript
# BAD — raw world coordinates (can be thousands of units)
obs.append(global_position.x)
obs.append(global_position.y)

# GOOD — normalised relative position (always -1 to 1 if arena is 200 units wide)
const ARENA_HALF_WIDTH := 100.0
const ARENA_HALF_HEIGHT := 100.0
obs.append(clamp(global_position.x / ARENA_HALF_WIDTH, -1.0, 1.0))
obs.append(clamp(global_position.y / ARENA_HALF_HEIGHT, -1.0, 1.0))
```

```gdscript
# BAD — unnormalised velocity (unbounded)
obs.append(linear_velocity.x)

# GOOD — velocity normalised by maximum expected speed
const MAX_SPEED := 500.0
obs.append(clamp(linear_velocity.x / MAX_SPEED, -1.0, 1.0))
obs.append(clamp(linear_velocity.y / MAX_SPEED, -1.0, 1.0))
```

### Common observation mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Including pixel positions in a large world | Policy learns nothing, value loss explodes | Normalise to arena-relative coordinates |
| Exposing goal position but not relative vector to goal | Policy converges slowly | Add `(goal_pos - agent_pos).normalized()` as obs |
| Including time remaining in episode | Policy ignores it early, over-weights it late | Remove unless time-pressure is a key mechanic |
| Not resetting observations on episode end | Stale values from previous episode bleed in | Reset all mutable obs state in `_reset()` |
| Partial observability without memory | Policy oscillates, never converges | Add RecurrentPPO (see [unit-08](unit-08.md)) |

!!! tip "Start minimal"
    Begin with the smallest observation vector that could theoretically solve the task. Add components only when training stalls and you can identify what information is missing. A 6-element obs that works beats a 60-element obs that confuses the network.

---

## 3 · Reward Design

The reward function is the most important thing you will write. Spend more time on it than on the algorithm choice.

### The structured process

Follow these steps in order. Do not skip ahead.

**Step 1 — Define success in plain English.**
Write one sentence: "The agent succeeds when ___." This becomes your terminal reward condition. Be specific. "Scores a goal" is specific. "Plays well" is not.

**Step 2 — Write the terminal reward first.**

```gdscript
func _get_reward() -> float:
    # Terminal reward only — shaped components come later
    if _goal_reached():
        return 1.0
    if _episode_timeout():
        return -0.1  # mild penalty for timing out, not failing
    return 0.0
```

Train with only this reward. If the agent finds the goal within a reasonable number of steps (try 1–5 million), you are done. Move to step 5.

**Step 3 — Run the random-policy sanity check.**

Before training, run 100 episodes with a random policy and log the return:

```python
# Python / SB3 sanity check before training
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

env = your_env  # your GodotEnv or wrapped env
check_env(env)

# Random rollout
obs, _ = env.reset()
total_reward = 0.0
for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        obs, _ = env.reset()
print(f"Random policy return: {total_reward:.3f}")
```

If a random policy never receives a positive reward in 100 episodes, a trained policy likely won't either. Add a small shaped component before training.

**Step 4 — Add shaped components only if training stalls.**

```gdscript
func _get_reward() -> float:
    var reward := 0.0

    # Terminal reward (primary signal — keep this weight at 1.0)
    if _goal_reached():
        reward += 1.0
        return reward  # early return, don't add shaping on terminal step
    if _episode_timeout():
        reward -= 0.1
        return reward

    # Shaped components (secondary — keep weights small, << 1.0)
    var dist_to_goal: float = global_position.distance_to(_goal_position)
    var dist_delta: float = _prev_dist_to_goal - dist_to_goal  # positive = getting closer
    reward += 0.01 * dist_delta  # potential-based shaping: safe and consistent

    _prev_dist_to_goal = dist_to_goal
    return reward
```

**Step 5 — Check for reward hacking.**
After every training run, watch your agent in Godot. A high TensorBoard score with bizarre-looking behaviour means your reward has been gamed. See [unit-reward-engineering](unit-reward-engineering.md) for the full theory and more anti-gaming techniques.

!!! note "Reward scale matters"
    Terminal rewards should dominate. If your shaped components sum to more than your terminal reward over an episode, the agent optimises the shaped signal instead. A good rule of thumb: `shaped_total_per_episode ≤ 0.5 × terminal_reward`.

For the complete reward engineering reference, see [unit-reward-engineering.md](unit-reward-engineering.md).

---

## 4 · Choosing an Algorithm

Use this flowchart to pick your algorithm. Do not overthink it — algorithm choice matters far less than reward design.

```
Are your actions continuous (Box action space)?
├── Yes → SAC (preferred for robotics / smooth control)
│          or PPO (simpler, works well for game-like tasks)
│          → see unit-sac.md
└── No  → PPO with Discrete or MultiDiscrete action space
           → see unit-ppo-deep.md

Is your reward sparse (agent rarely sees a positive reward)?
├── Yes → Add curiosity (RND or ICM) on top of your chosen algorithm
│          → see unit-curiosity.md
│          or add HER if your task is goal-conditioned
│          → see unit-her.md
└── No  → Standard PPO or SAC without modification

Does the agent need memory (partial observability, sequence-dependent decisions)?
├── Yes → RecurrentPPO (LSTM-augmented PPO)
│          → see unit-08.md
└── No  → Standard PPO or SAC

Is there more than one agent?
├── Competitive (zero-sum) → PPO + self-play
│                             → see unit-self-play.md
├── Cooperative → MAPPO with shared reward
│                  → see unit-self-play.md
└── Single-agent → any of the above
```

### Quick reference

| Scenario | Algorithm | Config key |
|---|---|---|
| Discrete game, dense reward | PPO | `algorithm: ppo` |
| Continuous control, dense reward | SAC | `algorithm: sac` |
| Any task, sparse reward | PPO + curiosity | `use_curiosity: true` |
| Goal-conditioned robotics | SAC + HER | `use_her: true` |
| Partial obs / memory needed | RecurrentPPO | `algorithm: rppo` |
| Competitive multi-agent | PPO + self-play | `self_play: true` |

!!! tip "When in doubt, start with PPO"
    PPO is the most robust algorithm in this course. It handles discrete and continuous actions, is tolerant of hyperparameter choices, and trains stably on almost any reward design. Switch to SAC only if you specifically need sample efficiency on a continuous-control task.

---

## 5 · Training Pipeline

### Headless export

Always train with a headless Godot build — the renderer costs ~30 % of your wall-clock time during training. Export from the Godot editor:

`Project → Export → Linux/macOS (Headless) → Export Project`

Name the binary `game.x86_64` (Linux) or `game` (macOS). Place it in your project root.

### Launch command template

Copy and adapt this command. Adjust `n_parallel` to the number of physical CPU cores you have (not hyper-threads):

```bash
# Minimum viable training run (3 seeds, 4 parallel envs each)
for SEED in 1 2 3; do
  gdrl train \
    --config-name=ppo \
    env.path=./game.x86_64 \
    env.n_parallel=4 \
    train.timesteps=3_000_000 \
    train.seed=$SEED \
    train.checkpoint_freq=100_000 \
    hydra.run.dir=runs/seed_${SEED} \
    &
done
wait
echo "All seeds finished"
```

For SAC (single-process, no vectorised envs):

```bash
for SEED in 1 2 3; do
  gdrl train \
    --config-name=sac \
    env.path=./game.x86_64 \
    env.n_parallel=1 \
    train.timesteps=1_000_000 \
    train.seed=$SEED \
    hydra.run.dir=runs/sac_seed_${SEED} \
    &
done
wait
```

### TensorBoard setup

```bash
# In a separate terminal — keep this running throughout training
tensorboard --logdir=runs/ --port=6006
# Open http://localhost:6006 in your browser
```

Key metrics to watch:

| Metric | What it tells you |
|---|---|
| `rollout/ep_rew_mean` | Is the agent learning at all? Should trend upward |
| `rollout/ep_len_mean` | Episode length — stable once agent solves task |
| `train/value_loss` | Critic accuracy — should decrease then plateau |
| `train/entropy_loss` | Exploration — if it drops to zero, agent is stuck |
| `train/approx_kl` | PPO update size — if > 0.05 consistently, lower lr |

### Checkpoint frequency

Set `checkpoint_freq` to save every 100 000 steps for tasks under 3 M steps, and every 250 000 steps for longer runs. Checkpoints let you recover a good policy even if training diverges later.

!!! warning "Always run at least 3 seeds"
    RL training has high variance. A single seed that performs well may be an outlier. Running three seeds gives you a mean and standard deviation that are meaningful. Anything less and you cannot tell if your design works or if you got lucky.

---

## 6 · Evaluation

Training ends. Now prove your agent actually learned something.

### Baselines first

Always compare against a random policy. The random baseline is your floor. If your trained agent barely beats it, something is wrong.

```python
# Compute random baseline before any training
import numpy as np
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="./game.x86_64", show_window=False)
random_returns = []
for episode in range(50):
    obs, _ = env.reset()
    ep_ret = 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_ret += reward
        done = terminated or truncated
    random_returns.append(ep_ret)
env.close()

print(f"Random baseline — mean: {np.mean(random_returns):.3f}  std: {np.std(random_returns):.3f}")
```

### Multi-seed summary

After training completes across all seeds, compute:

```python
import numpy as np

# Load per-seed final episode returns (e.g. from TensorBoard CSV export)
seed_returns = [
    [final_ep_rets_from_seed_1],
    [final_ep_rets_from_seed_2],
    [final_ep_rets_from_seed_3],
]
all_returns = np.concatenate(seed_returns)
print(f"Trained agent — mean: {np.mean(all_returns):.3f}  std: {np.std(all_returns):.3f}")

# Interquartile mean (IQM) — more robust than plain mean for RL
q25, q75 = np.percentile(all_returns, [25, 75])
iqm = np.mean(all_returns[(all_returns >= q25) & (all_returns <= q75)])
print(f"IQM: {iqm:.3f}")
```

The **interquartile mean** (IQM) trims the top and bottom 25 % of episodes before averaging. It is more robust than the plain mean when episode returns have outliers, which is common in RL. Use it in your report.

### Visualisation checkpoint video

```bash
# Run a trained checkpoint in render mode and record with OBS or ffmpeg
gdrl enjoy \
  --config-name=ppo \
  enjoy.checkpoint=runs/seed_1/best_model.zip \
  env.path=./game.x86_64 \
  env.show_window=true \
  enjoy.n_episodes=5

# Alternatively, pipe Godot window to ffmpeg (Linux)
ffmpeg -video_size 1280x720 -framerate 30 -f x11grab -i :0.0 \
  -c:v libx264 -preset fast capstone_demo.mp4
```

### What "good" looks like

"Good" is always relative to your task. Define it before you start training:

- If success is binary (goal reached / not), a "good" agent should reach the goal in > 80 % of evaluation episodes.
- If success is a score, a "good" agent should significantly exceed the random baseline (at least 2× mean return).
- If success is qualitative (smooth locomotion, coherent strategy), record a video and watch it critically.

---

## 7 · Shipping

When you are happy with your agent, export the ONNX model and run it inside Godot without Python.

### ONNX export

```python
# Export from SB3 checkpoint to ONNX
from godot_rl.wrappers.onnx.stable_baselines_export import export_ppo_model_as_onnx

export_ppo_model_as_onnx(
    model_path="runs/seed_1/best_model.zip",
    onnx_model_path="exported/agent.onnx",
)
print("ONNX export complete → exported/agent.onnx")
```

For SAC:

```python
from godot_rl.wrappers.onnx.stable_baselines_export import export_sac_model_as_onnx

export_sac_model_as_onnx(
    model_path="runs/sac_seed_1/best_model.zip",
    onnx_model_path="exported/agent.onnx",
)
```

### Godot inference mode

In your `AIController` node, set `inference_mode = true` in the editor Inspector and point `onnx_model_path` to your exported file:

```gdscript
# AIController.gd — inference mode snippet
extends AIController3D  # or AIController2D

func _ready() -> void:
    # These are set in the Inspector — shown here for reference
    # inference_mode = true
    # onnx_model_path = "res://exported/agent.onnx"
    pass

func get_obs() -> Array:
    # Must match exactly what you returned during training
    var obs: Array = []
    obs.append(clamp(global_position.x / ARENA_HALF_WIDTH, -1.0, 1.0))
    obs.append(clamp(global_position.y / ARENA_HALF_HEIGHT, -1.0, 1.0))
    obs.append(clamp(linear_velocity.x / MAX_SPEED, -1.0, 1.0))
    obs.append(clamp(linear_velocity.y / MAX_SPEED, -1.0, 1.0))
    return obs
```

!!! warning "Observation order must be identical"
    The ONNX runtime feeds your obs vector into the network in the order you return it from `get_obs()`. If the order differs from training, the policy produces garbage. Keep `get_obs()` under version control and never reorder it between training and export.

For HTML5 export and full shipping details, see [unit-10.md](unit-10.md).

---

## 8 · Project Ideas

Pick one that interests you. The difficulty ratings assume you have completed all course units.

| Project | Difficulty | Environment type | Action space | Most relevant units |
|---|---|---|---|---|
| **Soccer AI** — one-v-one agent scores goals against an opponent | Beginner | 2D physics | Continuous | unit-04, unit-self-play |
| **Platformer NPC** — agent learns to navigate a hand-built level | Beginner | 2D physics | Discrete | unit-01, unit-ppo-deep |
| **Tower Defense** — agent places towers to survive waves | Intermediate | 2D grid | Discrete (MultiDiscrete) | unit-03, unit-reward-engineering |
| **Cooking Game** — agent combines ingredients before the timer | Intermediate | 2D grid | Discrete | unit-08, unit-curiosity |
| **Traffic Simulation** — agents at an intersection minimise waiting | Intermediate | 2D physics | Discrete | unit-self-play, unit-reward-engineering |
| **Drone Racing** — quadrotor agent completes a checkpoint course | Intermediate | 3D physics | Continuous Box | unit-sac, unit-robotics |
| **Warehouse Robot** — arm places boxes onto pallets in order | Intermediate | 3D physics | Continuous Box | unit-robotics, unit-her |
| **Hide-and-Seek** — hiders vs seekers emergent strategy | Advanced | 3D physics | Continuous | unit-self-play, unit-curiosity |
| **Chess Variant Self-Play** — agent learns a modified chess | Advanced | 2D grid | Discrete (large) | unit-self-play, unit-03 |
| **Music Rhythm Agent** — agent hits beats with precise timing | Advanced | 2D | Continuous | unit-sac, unit-reward-engineering |

!!! tip "Beginner recommendation"
    Start with **Platformer NPC** or **Soccer AI**. Both have simple, verifiable success conditions, short episodes, and dense enough rewards that a random policy occasionally succeeds — exactly the conditions where RL works reliably.

!!! note "Multi-agent projects"
    Hide-and-Seek, Traffic Simulation, and Chess Variant Self-Play all require multi-agent infrastructure. Complete [unit-self-play.md](unit-self-play.md) before attempting any of them. Do not underestimate the added complexity of multi-agent debugging.

---

## 9 · Common Failure Modes

These are issues that appear specifically on original projects — they do not appear in the guided examples because the starter code already handles them. They are not in the standard debugging guide ([unit-debugging.md](unit-debugging.md)) because they relate to design decisions, not training pathologies.

### 1 · Wrong action space type

**Symptom:** Agent seems to do the same thing every step, or actions have no visible effect.

**Diagnosis:** You defined a `Box` (continuous) action space but your game expects an integer index, or vice versa. Check the action space declaration in your `AIController` and the `set_action()` handler.

```gdscript
# WRONG — Box action when you want discrete
func get_action_space() -> Dictionary:
    return {"move": {"size": 4, "action_type": "continuous"}}

# CORRECT — Discrete action for 4 directions
func get_action_space() -> Dictionary:
    return {"move": {"size": 4, "action_type": "discrete"}}
```

### 2 · Observations not resetting on episode end

**Symptom:** Agent performs well for the first episode, then degrades or behaves erratically in subsequent episodes.

**Diagnosis:** Mutable state used in `get_obs()` (e.g. `_prev_dist_to_goal`, stacked frame buffers) is not reset when the episode ends.

```gdscript
func _reset() -> void:
    # ALWAYS reset all mutable observation state here
    _prev_dist_to_goal = global_position.distance_to(_goal_position)
    _prev_velocity = Vector2.ZERO
    _step_count = 0
    # ... any other stateful obs components
```

### 3 · Reward scale mismatch between shaped and terminal components

**Symptom:** TensorBoard shows high `ep_rew_mean` but the agent never reaches the goal — it is harvesting shaped reward indefinitely.

**Diagnosis:** Your shaped reward per episode exceeds your terminal reward. The agent rationally avoids terminating the episode.

```gdscript
# BAD — shaped reward dwarfs terminal reward over a 500-step episode
# Step reward: +0.01 × 500 steps = +5.0 total shaped
# Terminal reward: +1.0
# Agent prefers staying alive and collecting +5.0 over ending the episode

# GOOD — keep shaped total << terminal
# Step reward: +0.001 × 500 steps = +0.5 total shaped
# Terminal reward: +1.0
# Agent prefers ending the episode (efficiency) over harvesting shaped reward
```

### 4 · Forgetting to normalise continuous observations

**Symptom:** Value loss oscillates wildly, policy loss diverges, or the agent learns nothing after millions of steps despite a sane reward.

**Diagnosis:** One or more obs components are on a very different scale from the rest. The neural network weights cannot stabilise when inputs range from `0.001` to `10 000` in the same vector.

**Fix:** Audit every component returned by `get_obs()`. If any value can exceed `1.0` or drop below `-1.0` in the wild, normalise it as shown in Section 2.

### 5 · Running only one seed and declaring victory

**Symptom:** Your agent "works great" but a classmate cannot reproduce your results, or your agent performs well on one evaluation day and poorly a week later on the same checkpoint.

**Diagnosis:** You ran one seed. RL has high variance. A single lucky seed can appear to solve a task that your design does not actually solve reliably.

**Fix:** Always run three seeds minimum (Section 5). Report mean ± std or IQM. If performance varies enormously across seeds, your reward design or observation design is too sensitive — investigate before shipping.

### 6 · Treating a 3D problem as a 2D problem (or vice versa)

**Symptom:** Agent solves the task in 2D test scenarios but fails on the full 3D environment. Or: agent's `AIController` extends `AIController2D` in a 3D scene (or vice versa), causing physics queries to return wrong values.

**Diagnosis:** Check which base class your `AIController` inherits. Check that raycasts, collision shapes, and position queries use the correct coordinate system.

```gdscript
# For a 2D scene
extends AIController2D

# For a 3D scene
extends AIController3D

# Do NOT mix them — the parent class determines which physics server is queried
```

---

## 10 · Stretch Goals

You shipped a working agent. Here are three ways to go further.

### Submit to a leaderboard or itch.io

Export your game as an HTML5 build ([unit-10.md](unit-10.md)) and publish it on [itch.io](https://itch.io). Write a short description explaining what the agent learned and how you trained it. Share the link in the course Discord.

If your project fits an existing benchmark (e.g. a soccer variant), consider submitting to the Godot RL Agents community leaderboard.

### Write a 1-page technical report

Structure:
1. **Task** (2 sentences) — what is the environment and success condition?
2. **MDP design** (3–4 sentences) — observation space, action space, reward function.
3. **Algorithm** (1 sentence) — which algorithm and why?
4. **Results** (3–4 sentences) — multi-seed mean ± std, IQM, comparison to random baseline.
5. **What did not work** (2–3 sentences) — at least one failed approach. This is the most valuable section.
6. **Future work** (1–2 sentences) — what would you try next?

Writing about what failed is more valuable than reporting success. Failures contain the real lessons.

### Record a 2-minute video demo

Show:
1. The random policy (first 30 seconds) — so viewers appreciate how hard the task is.
2. A mid-training checkpoint (30 seconds) — show the agent learning.
3. The final trained policy (60 seconds) — best behaviour you observed.

Narrate each section. "The agent has learned to avoid walls but still does not know how to score" is more informative than silence.

---

!!! tip "You are now an RL practitioner"
    Every project you build from here will be faster than the last. The hard part — designing the MDP, debugging reward hacking, interpreting TensorBoard, running multi-seed evaluations — is now familiar. The algorithms and frameworks will change; this process does not.

---

[← Ship Your Brain](unit-10.md) · [Course home](index.md)
