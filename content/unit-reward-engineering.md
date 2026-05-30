# Reward Engineering — Designing Signals That Actually Work

[← Unit 1: Foundations](unit-01.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~35 min

!!! info "Three ways to see your AI"
    - **Godot viz** — the only place where reward hacking is visible. A high TensorBoard score plus a weird-looking agent in Godot means your reward is broken, not your algorithm.
    - **TensorBoard** — the shape of `ep_rew_mean` over time tells you the whole story: flat means no signal, rising means learning, oscillating means weight imbalance.
    - **Reward sanity check script** — run 100 random episodes before training. If a random policy never sees a positive reward, your agent never will either.

---

## 1 · Why Reward Engineering Matters

The reward is the **only** signal the agent gets from the world. The environment doesn't tell the agent what to do — it just hands back a number after every action. That number is the entire definition of "success" for the agent.

This means:

- A bad reward = **no learning**, **wrong behavior**, or **reward hacking**.
- The hardest part of applied RL isn't picking PPO vs SAC vs DQN — it's getting the reward right.
- You will spend more time tuning your reward than tuning your hyperparameters. By a lot.

### The three failure modes

Every broken reward falls into one of these buckets:

1. **No signal** — the agent never learns because every episode produces the same flat reward. Common with pure sparse rewards on hard tasks.
2. **Wrong signal** — the agent learns *something*, but not what you wanted. The reward you wrote doesn't match the behavior you imagined.
3. **Gaming the reward** — the agent finds an unintended shortcut. The reward is technically being maximized, but in a way that defeats the purpose.

### A real example

OpenAI trained a boat racing agent on a game where collecting fuel pickups gave small bonuses, and finishing the race gave a big bonus. The agent discovered it could **spin in circles near a respawning cluster of fuel bonuses** and rack up more points than actually racing. It never finished a single race, but its TensorBoard score was great.

This is the canonical reward-hacking story. The lesson: **your agent does not know what you meant. It only knows the number you gave it.**

---

## 2 · Dense vs Sparse Rewards

Every reward design lives on a spectrum between two extremes.

### Sparse reward

The agent gets a signal **only at the end** of an episode (or at rare key moments).

- **Clean**: captures the real objective exactly. There's no ambiguity about what you want.
- **Problem**: most episodes produce zero signal. Without any gradient to follow, the agent flounders.
- **Example**: CrossTheRoad — only `+1` when the agent reaches the goal, `-1` on death. Everything in between is zero.

```gdscript
# Pure sparse reward
if reached_goal:
    _ai.reward += 1.0
    _ai.done = true
elif died:
    _ai.reward -= 1.0
    _ai.done = true
# All other steps: reward = 0
```

### Dense reward

The agent gets a signal at **every step**.

- **Provides learning signal constantly** — there's always a direction to improve.
- **Risk**: the agent may optimize the *dense* signal instead of the real goal. The dense reward becomes the goal.
- **Examples**: distance to goal, velocity toward goal, height gained, alignment to target heading.

```gdscript
# Pure dense reward
_ai.reward += velocity_toward_goal * 0.01
_ai.reward -= distance_to_goal * 0.001
```

### The key insight

> **Dense rewards must be consistent with the sparse goal.**

If your dense reward disagrees with your sparse reward, the agent will optimize the **dense** one. Why? Because the dense one fires constantly and the sparse one fires once. In terms of total signal, dense almost always wins.

**The diagnostic test:** Can a human score high on the dense reward *while failing the sparse goal*?

- If **yes** → your dense reward is misaligned. Redesign it.
- If **no** → the dense reward is at least consistent with the goal.

Example of a misaligned dense reward: rewarding "velocity" without "velocity *toward the goal*". The agent will happily go fast in the wrong direction.

---

## 3 · Potential-Based Reward Shaping (The Theory)

There's actually a formal theory that tells you which dense rewards are safe to add. It comes from Ng, Harada, and Russell (1999).

### The theorem

Any reward of the form:

$$F(s, s') = \gamma \cdot \Phi(s') - \Phi(s)$$

**preserves the optimal policy** of the underlying MDP.

- $\Phi(s)$ is called a **potential function** — it can be any function of the state.
- If you add $F$ to your environment's reward: $r_{shaped} = r + \gamma \cdot \Phi(s') - \Phi(s)$
- The optimal policy under $r_{shaped}$ is the **same** as under $r$.

### Why this works (intuition)

When you sum the shaping term $F$ over an episode, the consecutive $\Phi$ terms telescope:

$$\sum_t F(s_t, s_{t+1}) = \gamma \Phi(s_T) - \Phi(s_0)$$

It depends only on the start and end states. It doesn't reward the agent for any particular *path* — it just adds a constant offset over the episode. So the agent's incentive structure is preserved.

### Common potentials

| Potential Φ(s)         | F = γΦ(s') − Φ(s)         | Effect                       |
|------------------------|---------------------------|------------------------------|
| `-distance_to_goal`    | Progress toward goal      | Encourages movement to goal  |
| `height`               | Height gain               | Encourages climbing          |
| `speed_toward_goal`    | Acceleration toward goal  | Encourages moving fast       |
| `-time_elapsed`        | Constant negative offset  | Encourages speed/efficiency  |

!!! tip "Potential-based shaping is the safe default"
    If you're not sure whether a dense reward is going to corrupt your agent's behavior, write it as `γ·Φ(s') - Φ(s)` instead of as a raw reward. You get the learning signal without changing what the optimal policy is.

!!! warning "Arbitrary dense rewards do NOT have this property"
    A reward like `+0.01 for being in zone A` is **not** potential-based and **can** change the optimal policy. The agent might learn to camp in zone A even if that hurts the real objective.

---

## 4 · Reward Components in GDScript

Here's the standard pattern for composing rewards in a Godot agent script:

```gdscript
extends RigidBody3D

@onready var _ai = $AIController3D
@onready var goal = get_node("../Goal")

var _prev_dist_to_goal: float = 0.0
var time_alive: float = 0.0
var max_time: float = 30.0
var max_dist: float = 50.0

func _physics_process(delta):
    if _ai.needs_reset:
        reset()
        return

    time_alive += delta

    # --- Dense shaped reward (runs every physics step) ---
    var dist_to_goal = global_position.distance_to(goal.global_position)
    var prev_dist    = _prev_dist_to_goal
    _prev_dist_to_goal = dist_to_goal

    # Progress reward: potential-based (γ·Φ(s') - Φ(s))
    # Φ(s) = -dist_to_goal, so progress = prev_dist - dist_to_goal
    var progress = (prev_dist - dist_to_goal) / max_dist
    _ai.reward += progress * 0.5

    # Survival bonus: small positive per step to discourage suicide
    _ai.reward += 0.001

    # Velocity penalty: discourages erratic jitter
    _ai.reward -= linear_velocity.length() * 0.0001

    # --- Sparse terminal reward (fires once on episode end) ---
    if dist_to_goal < 1.0:
        _ai.reward += 1.0     # goal reached
        _ai.done = true
        _ai.needs_reset = true
    elif time_alive > max_time:
        _ai.reward -= 0.5     # timeout penalty
        _ai.done = true
        _ai.needs_reset = true

func reset():
    # CRITICAL: initialize ALL reward state here, not in _ready()
    _prev_dist_to_goal = global_position.distance_to(goal.global_position)
    time_alive = 0.0
    _ai.needs_reset = false
```

### The single most-broken line in student code

> **Always initialize `_prev_dist_to_goal` in `reset()`, not in `_ready()`.**

If you initialize it in `_ready()`, then after the *first* reset, `_prev_dist_to_goal` still holds the value from the *previous* episode. On the first step of the new episode, `progress` will be a huge spike (positive or negative depending on where the agent was). The agent learns garbage from that step.

This is the most common silent bug in student reward functions. It doesn't crash. It just makes your training look noisy and slow.

---

## 5 · Reward Scale and Normalization

Neural networks learn best when their inputs and targets are in a predictable range. If your reward is sometimes `+1000` and usually `0.0001`, the value function has to span six orders of magnitude — and it won't.

### Scale guidelines

| Component                  | Magnitude          |
|----------------------------|--------------------|
| Terminal rewards (win/lose)| ±1.0 to ±10.0      |
| Per-step shaped rewards    | 0.001 to 0.01      |
| Survival bonuses           | ~0.001             |
| Per-step penalties         | 0.0001 to 0.001    |

The rule of thumb: **per-step shaped rewards should be much smaller than terminal rewards.**

If per-step >> terminal: the agent ignores the goal and farms the per-step signal. The episode return is dominated by "how many steps did I survive while collecting tiny rewards", not "did I succeed".

If per-step << terminal: you essentially have a sparse reward. Whether that's a problem depends on whether the agent can reach the terminal reward by exploration alone.

### The normalization check

After designing rewards but **before** running PPO, run 100 episodes with a **random policy** and print the statistics:

```python
# Quick reward sanity check script
import numpy as np
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="./MyEnv.x86_64", n_parallel=1, speedup=1)

rewards_per_episode = []
step_rewards = []
terminal_hits = 0

for ep in range(100):
    obs = env.reset()
    total = 0.0
    done = False
    while not done:
        action = env.action_space.sample()   # random policy
        obs, r, done, info = env.step(action)
        total += r
        step_rewards.append(r)
    rewards_per_episode.append(total)
    if total > 0:
        terminal_hits += 1

print(f"Random policy stats over 100 episodes:")
print(f"  mean episode reward: {np.mean(rewards_per_episode):.3f}")
print(f"  min episode reward:  {min(rewards_per_episode):.3f}")
print(f"  max episode reward:  {max(rewards_per_episode):.3f}")
print(f"  per-step reward range: [{min(step_rewards):.4f}, {max(step_rewards):.4f}]")
print(f"  episodes with positive terminal: {terminal_hits}/100")
env.close()
```

### What the output tells you

- **`terminal_hits == 0`** for 100 random episodes → your sparse signal is unreachable. Add a shaped reward or curriculum.
- **Per-step reward range is huge** (e.g. `[-100, 100]`) → rescale. The network can't fit this.
- **`ep_rew_mean ≈ 0`** during the first 100k–500k training steps → you have a **signal problem**, not an **algorithm problem**. Stop tweaking PPO and fix the reward.

---

## 6 · Common Failure Modes and Fixes

These are the bugs you will hit. All of them. Eventually.

### Failure 1: Reward sign wrong

- **Symptom**: `ep_rew_mean` goes *down* over training. The agent gets *worse* the more it trains.
- **Cause**: you flipped a sign somewhere. The agent is correctly maximizing the (negated) thing you intended.
- **Fix**: flip the sign on the offending reward component.
- **Debug**: print reward values during the first few steps of an episode:

```gdscript
print("step=%d reward=%.4f progress=%.4f" % [step, _ai.reward, progress])
```

### Failure 2: Agent farms the shaped reward

- **Symptom**: agent achieves high TensorBoard reward but never reaches the goal when you watch it in Godot.
- **Example**: agent oscillates back and forth near the goal to collect progress reward, never crossing the threshold.
- **Fix**: reduce the shaped component's weight; add a per-step time penalty so dawdling costs reward; ensure the terminal reward is genuinely larger than what farming can produce.

### Failure 3: Reward too sparse, agent never learns

- **Symptom**: flat `ep_rew_mean` for 500k+ steps; the random-policy sanity check showed 0 positive terminal episodes.
- **Cause**: the agent's exploration never accidentally hits the goal, so it never gets any positive signal to learn from.
- **Fix**: add a potential-based shaping term (distance to goal is the safest); or shorten the task; or use curriculum learning; or add curiosity (see Unit Curiosity).

### Failure 4: Episode reset mid-progress

- **Symptom**: agent does well for a few steps, then seems to "give up" and die quickly.
- **Cause**: `reset()` doesn't restore `_prev_dist_to_goal`. The first step of the new episode produces a huge negative spike in `progress`. The agent learns that being alive is bad and that ending the episode quickly avoids the spike.
- **Fix**: always reset **all** reward state variables in `reset()`. Make a checklist of every variable that participates in the reward calculation.

### Failure 5: Reward cliffs

- **Symptom**: agent avoids otherwise-rewarded regions of the state space. Behaviour looks bizarrely conservative.
- **Cause**: a large negative reward at a boundary (e.g. `-100` for falling off the platform). The agent's value function has a giant negative spike near that boundary and the policy refuses to approach it, even where it would be beneficial.
- **Fix**: scale terminal penalties to be 5–10× the typical per-step reward, **not** 100×. The agent should be discouraged from failure, not terrified of it.

---

## 7 · Reward Hacking and How to Prevent It

!!! warning "Reward hacking"
    **Definition**: the agent finds an unintended way to get high reward that doesn't match the real goal. It's not a bug in the agent — it's a bug in your reward.

### Famous examples

- **Boat racing agent** (OpenAI): circles near a fuel-bonus cluster instead of racing. Gets more points than racers.
- **Simulated walker** (DeepMind): "walks" by falling forward and exploiting a physics glitch that gives x-axis progress for free.
- **Gripper robot** (research lab): covers the camera lens with the gripper so the "object not grasped" detector returns false. Reward = "object grasped" was achieved, technically.
- **Cheating block-stacker**: flips the block on its side, since "top of block is high" is achieved when the block is horizontal and taller than wide.

In every case the agent is doing exactly what the reward asked for. The reward just didn't ask for the right thing.

### Prevention strategies

1. **Spec the real goal**: always have a terminal sparse reward for the *actual* objective. No matter how good the dense shaping is, the sparse goal anchors the agent to what success means.
2. **Limit the shaped components**: each shaped reward is another potential loophole. Fewer signals = fewer loopholes. Start with one shaping term and add more only if needed.
3. **Watch in Godot**: open the viz checkpoint. Reward hacking is *immediately* visible to a human — the agent looks weird, does the same thing over and over, ignores obvious paths.
4. **Randomize reward conditions**: if shaped rewards depend on a fixed feature of the level (e.g. a specific platform position), the agent can memorize and exploit it. Randomize platform layouts, spawn positions, goal locations.
5. **Adversarially test your reward**: ask yourself "what's the dumbest behavior that scores well here?" Often that's exactly what the agent will find.

---

## 8 · Multi-Objective Rewards

Real tasks usually involve trade-offs. A racing agent should be fast *but* not crash. A delivery robot should be fast *but* not bump into people.

```gdscript
# Example: racing agent — fast but not crashy
_ai.reward += speed_toward_finish * 0.3     # go fast
_ai.reward -= collision_force * 0.5         # don't crash
if finished:
    _ai.reward += 5.0                       # actually finish the race
    _ai.done = true
```

### Weight tuning

Treat reward weights as **hyperparameters**. They're as important as learning rate.

Workflow:

1. Pick a default weight set (your best guess).
2. Run a short experiment (500k steps).
3. Change **one** weight at a time and re-run.
4. Compare runs in TensorBoard.

| Weight config              | Observed behavior          |
|----------------------------|----------------------------|
| `speed=0.3, crash=-0.5`    | Fast racing, some crashes  |
| `speed=0.1, crash=-1.0`    | Slow but safe              |
| `speed=0.3, crash=0.0`     | Fast but crashes constantly|
| `speed=0.0, crash=-0.5`    | Sits still to avoid crashes|

Two things to notice:

- **`speed=0.0`** produces an agent that does nothing. Removing a positive incentive is just as destructive as miscalibrating it.
- **`crash=0.0`** produces an agent that ignores safety entirely. Negative incentives matter.

The best weights are usually found by trial and error. There is no closed-form solution. Budget experiment time for this.

---

## 9 · Curriculum Learning Teaser

Sometimes even well-designed rewards are too hard for a random-exploration agent to find.

Example: in Lunar Lander, if the landing pad is tiny and far away, a random policy never lands. With zero successful landings, there's no positive signal to learn from. Training stalls.

The solution: **start easy, get harder**.

- **Manual curriculum**: spawn the agent closer to the goal in early training. As `ep_rew_mean` improves, increase the distance.
- **Automatic curriculum**: check `ep_rew_mean` against a threshold. When the agent beats threshold A, advance to level B. When it beats B, advance to C.
- **Domain randomization**: randomize starting conditions over a range that grows over time.

Sketch in GDScript:

```gdscript
# In your environment manager
var curriculum_level: int = 0
var success_buffer: Array = []

func on_episode_end(success: bool):
    success_buffer.append(success)
    if success_buffer.size() > 100:
        success_buffer.pop_front()
    var rate = success_buffer.count(true) / float(success_buffer.size())
    if rate > 0.8 and curriculum_level < 5:
        curriculum_level += 1
        success_buffer.clear()
        print("Advancing curriculum to level ", curriculum_level)
```

This is a full topic on its own — see the stretch goals and a later unit.

---

## 10 · Safety Constraints in Reward Design

For most course projects, a "bad episode" means the agent falls over or misses the goal. On real hardware, a bad episode can mean a broken servo, a burned-out motor, or a damaged gearbox. Reward design must reflect the cost structure of the actual system being controlled.

### Hard constraints vs soft penalties

There are two ways to encode a constraint in reward:

```gdscript
# Soft penalty — discourages the behavior but does not terminate
if abs(joint_angle) > safe_range * 0.8:
    _ai.reward -= 0.1   # warns the policy to back off

# Hard constraint — terminates immediately and applies a large penalty
if abs(joint_angle) > hard_limit:
    _ai.reward -= 5.0   # strong negative signal
    _ai.done = true     # end the episode — this would break hardware
    _ai.needs_reset = true
```

Use soft penalties to shape the policy toward safe operation well before the hard boundary. Use hard constraints to terminate episodes that cross into physically dangerous territory. The two-tier structure — warn at 70%, terminate at 90% — is a practical rule of thumb used in sim-to-real transfer work.

### Common hardware safety constraints

| Constraint | Godot implementation | Why it matters |
|---|---|---|
| Joint angle limits | Check bone rotation against configured limit | Servo strip / mechanical stop damage |
| Velocity limits | Check `linear_velocity.length()` against max | Motor overheating under sustained high speed |
| Acceleration limits | Check velocity change per physics step | Gearbox shock loads from abrupt starts/stops |
| Ground contact force | Check collision normal force magnitude | Leg impact damage on hard landings |
| Workspace limits | Check `global_position` bounds | Arm or end-effector hitting a fixed surface |

### The safety-performance tradeoff

Strict safety constraints reduce learning speed because more episodes terminate early, meaning fewer transitions per episode reach later task states. Too-loose constraints allow the policy to find dangerous behaviors in simulation that would damage hardware on transfer.

A practical starting point: set hard termination at 90% of the physical limit; begin soft penalties at 70%. If the policy still finds the boundary too often, lower both thresholds or increase the soft penalty coefficient.

!!! warning "Safety constraints are not free"
    Every additional termination condition is effectively a curriculum difficulty increase. If training stalls after adding a safety constraint, check whether the survival time has dropped sharply in TensorBoard (`rollout/ep_len_mean`). If episodes are very short, the safety boundary may be tighter than the policy can reliably avoid during early training — consider curriculum-based constraint tightening.

---

## 11 · Energy Efficiency in Reward Design

Energy efficiency matters in two different contexts:

- **Real robots**: battery life, motor heat, mechanical longevity — all are directly affected by how much power the policy commands.
- **Virtual robots**: energy penalties produce more natural-looking, human-like motion by discouraging the "spastic" high-frequency joint oscillations that unconstrained policies tend to discover.

### Power-based penalty

The physically accurate approach: penalize the actual power consumed at each joint.

```gdscript
# Power = force × velocity (translational) or torque × angular_velocity (rotational)
var translational_power = applied_force.dot(linear_velocity)
var rotational_power    = applied_torque.dot(angular_velocity)
var total_power = abs(translational_power) + abs(rotational_power)
_ai.reward -= total_power * power_penalty_coeff   # typically 0.0001 to 0.001
```

The coefficient `power_penalty_coeff` is the most sensitive hyperparameter here. Too large and the policy freezes (no motion = no power = high reward). Too small and it has no effect on the gait. Start at `0.0001` and increase until gait quality improves without stopping locomotion.

### Action magnitude penalty (simpler approximation)

When joint torque data is not readily accessible, penalize the magnitude of the action vector directly:

```gdscript
# Penalize large actions regardless of outcome — reduces jerkiness
var action_magnitude = 0.0
for a in last_action.values():
    if a is Array:
        for v in a: action_magnitude += v * v
    else:
        action_magnitude += a * a
_ai.reward -= action_magnitude * 0.0005
```

This is the same as MuJoCo's `ctrl_cost` (the sum of squared action components multiplied by a cost coefficient), which appears in every standard locomotion benchmark. The intuition is that large actions require large forces, which require large currents, which consume large amounts of power. Action magnitude is a cheap and effective proxy.

### Smoothness penalty

The energy penalty reduces average power consumption. The smoothness penalty reduces peak-to-peak variation — it discourages policies that alternate between high and low torque rapidly:

```gdscript
# Penalize rapid action changes (jerk) — produces smoother policies
if _prev_action != null:
    var action_delta = 0.0
    for key in last_action:
        var curr = last_action[key] if last_action[key] is float else last_action[key][0]
        var prev = _prev_action[key] if _prev_action[key] is float else _prev_action[key][0]
        action_delta += (curr - prev) * (curr - prev)
    _ai.reward -= action_delta * 0.001
_prev_action = last_action.duplicate(true)
```

Note `duplicate(true)` — a deep copy is required here. If you store a reference to `last_action`, `_prev_action` will reflect the current action on the next step, making the delta always zero.

### When to use which

| Situation | Recommended approach |
|---|---|
| Simulated locomotion, want natural-looking gait | Action magnitude penalty (simple, effective) |
| Sim-to-real transfer | Power-based penalty (physically grounded) |
| Policy produces jitter / vibration | Smoothness penalty on action deltas |
| All three issues | Stack all three with separate tunable coefficients |

---

## 12 · Reward Engineering Checklist

Print this. Tape it to your monitor.

### Before training

- [ ] Can I explain in **one sentence** what the agent should maximize?
- [ ] Is there a **terminal reward** for the real goal (not just shaped signals)?
- [ ] Is the terminal reward **reachable** by a random policy, even rarely? (run the sanity script)
- [ ] Are per-step rewards **less than 1/10** the magnitude of terminal rewards?
- [ ] Did I initialize **all** reward state variables (`_prev_dist`, counters, timers) in `reset()`?
- [ ] Did I run **100 random episodes** and print `ep_rew_mean`, min, max?
- [ ] Are shaped rewards **potential-based** where possible?
- [ ] Did I ask "what's the dumbest way to score well here?" — and rule out that loophole?

### After 500k steps

- [ ] Is `ep_rew_mean` **rising**? (if not: reward or observation problem, not algorithm problem)
- [ ] Does the Godot viz checkpoint **match** the TensorBoard score? (if not: reward hacking)
- [ ] Is the agent doing **something reasonable**, even if suboptimal?
- [ ] Are the per-component reward magnitudes **balanced**? (log them separately in TensorBoard)

### Hardware safety (add if deploying to real robot)

- [ ] Joint limit violations terminate episodes with a hard penalty
- [ ] Soft penalty begins at 70% of hard limit; hard penalty + `done = true` at 90%
- [ ] Energy efficiency penalty is active to prevent overheating behavior
- [ ] Action magnitude or smoothness penalty is present to reduce mechanical wear

If any item fails: **stop training. Fix the reward. Restart.** Throwing more steps at a broken reward never works.

---

## 13 · Stretch Goals

If you finished the unit and want more practice:

1. **Curriculum by distance** — spawn the agent at a random distance from the goal (sampled uniformly in `[0, max_dist]`). After training, plot success rate vs spawn distance. Where does the agent fall off?
2. **Reward ablation study** — train three agents on the same task: one with only terminal reward, one with only shaped reward, one with both. Compare their learning curves in TensorBoard. Which converges fastest? Which converges to the best final policy?
3. **Debug a broken reward** — deliberately introduce a sign error in your reward function. Train for 100k steps. Identify the bug from the TensorBoard curve alone, without looking at the code. This is the most common real-world debugging scenario.
4. **Reward hacking on purpose** — design a small environment where a shaped reward *can* be hacked, and see how long it takes the agent to find the exploit. Then fix the reward so the hack disappears.

---

## What's Next

You now have the most important skill in applied RL: designing rewards that produce the behavior you actually want.

In the next unit you put this directly into practice: building **Lunar Lander** in Godot from scratch, including writing the lander's per-step reward function in GDScript. The shaping you write there in §5 is potential-based — the theory you just learned.

If you only remember one thing from this unit: **the reward is not a description of the goal. The reward is the goal.** Whatever you write down, that's what the agent will maximize. Make sure that's what you want.

[→ Unit 2: Build Lunar Lander in Godot](unit-02.md)
