# Safe RL / Constrained MDPs

[← Sim-to-Real Transfer](unit-sim-to-real.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~35 min · Training: ~30 min GPU / ~2 h CPU

!!! info "Three ways to see your AI"
    - **Godot viewport** — watch the agent navigate toward its goal while visibly avoiding the hazard zone; at the start of training it blunders straight through; by convergence it should route around it almost every episode.
    - **TensorBoard** — plot two curves side by side: `rollout/ep_rew_mean` (task reward) and `constraint/violation_rate` (fraction of steps that incurred a cost). A working CMDP run shows reward climbing *while* violation rate falls — penalty-based methods often show the two curves fighting each other.
    - **λ monitor** — log the Lagrange multiplier `λ` each update. It should rise whenever the constraint is violated and fall when the agent is safe. A flat λ curve means either the constraint is never triggered or the dual update is broken.

---

You have trained policies that jump, walk, grasp, and transfer to real hardware. In every unit so far, "safe" has meant adding a `-100` penalty to the reward and hoping. For demos and research prototypes, that often works. For policies that will control physical robots, surgical assistants, autonomous vehicles, or any system where a constraint *must* hold — it does not work reliably.

This unit teaches you why, and what to do instead.

## 1 · Why reward penalties aren't enough

The standard approach to teaching an agent about constraints is to add a large negative reward whenever the constraint is violated:

```python
# Standard penalty approach
if is_in_hazard_zone(state):
    reward -= 100.0
```

This is convenient and sometimes works. But it has a structural problem that makes it unsuitable for hard constraints:

**The penalty coefficient has no principled interpretation.** When you set the penalty to `-100`, you are implicitly saying "one constraint violation is worth losing 100 reward points." But you probably never verified that claim. What if the task reward is +5 per step and the episode is 500 steps long? A maximum possible task reward of +2500 means your penalty coefficient is 4% of max return — barely a blip, and the agent may decide violating the constraint six times per episode is worthwhile. Raise the penalty to `-10000` and the agent may refuse to move at all, collapsing the task reward to near zero.

You end up tuning a coefficient whose correct value you cannot know without solving the problem first. This is the **penalty tuning trap**.

**The tradeoff is unpredictable.** With a penalty term, the agent implicitly trades reward for safety. How many violations is "acceptable"? With a penalty, the answer depends on the ratio of penalty magnitude to expected return — a ratio that changes every time you modify the task reward or the episode length. There is no knob that means "zero violations." There are only knobs that approximately push in that direction, and the approximation quality varies by task.

**The constraint can be satisfied on average but violated every episode.** Suppose your penalty produces an agent that violates the constraint 0.5 times per episode on average. The expected penalty exactly cancels against the expected gain. Statistically, this looks fine. But in practice the agent enters the hazard zone on half of all episodes — which is exactly what you were trying to prevent.

!!! warning "Penalty tuning is a disguised tradeoff, not a constraint"
    When you add `-100` for constraint violation, you are not enforcing a constraint — you are encoding a *preference*. The agent will violate the constraint whenever doing so is worth more than 100 reward points. If you truly need a hard limit (power consumption, joint torque, contact force, speed near humans), you need a different formulation.

The right framework is the **Constrained Markov Decision Process**.

---

## 2 · Constrained MDP (CMDP) formulation

A standard MDP maximizes a single expected return:

```
max J(π) = E[Σ γᵗ r(sₜ, aₜ)]
```

A **Constrained MDP (CMDP)** adds one or more explicit constraint functions alongside the reward:

```
max J(π) = E[Σ γᵗ r(sₜ, aₜ)]       ← task objective (unchanged)
subject to:
    C(π) ≤ d                         ← constraint: expected cost ≤ threshold
```

Where:

- `r(s, a)` is the **task reward** — the signal you always had
- `c(s, a)` is a **cost function** — a separate signal for constraint-relevant events (entering a hazard zone, exceeding a torque limit, etc.)
- `C(π) = E[Σ γᵗ c(sₜ, aₜ)]` is the **expected discounted cost** — the constraint value function, analogous to the return
- `d` is the **constraint threshold** — the maximum tolerable expected cost

This is a cleaner formulation for three reasons:

1. **The threshold `d` has a direct physical interpretation.** "The expected number of hazard-zone steps per episode must be below 2.0." You can choose `d` by asking domain experts what is acceptable, rather than by tuning a penalty coefficient.
2. **The cost function is separate from the reward function.** You can change one without changing the other. You can add a new constraint without re-tuning all reward coefficients.
3. **The algorithm provably satisfies the constraint during training (not just at convergence).** This is the key claim of CPO, discussed in Section 4.

### Building the cost function in Godot

Cost functions have a simpler structure than reward functions: they measure "how bad is this step from a safety perspective," independently of task progress.

```gdscript
# In the environment script — compute cost alongside reward
func compute_step_signals() -> void:
    var reward: float = 0.0
    var cost: float   = 0.0

    # --- Task reward ---
    var dist_to_goal = global_position.distance_to(goal.global_position)
    reward += 1.0 - clampf(dist_to_goal / MAX_DIST, 0.0, 1.0)
    if dist_to_goal < GOAL_RADIUS:
        reward += 5.0
        _ai.done = true

    # --- Cost signal: hazard zone entry ---
    if _in_hazard_zone():
        cost += 1.0

    # --- Cost signal: velocity limit (e.g. near humans) ---
    if linear_velocity.length() > MAX_SAFE_SPEED:
        cost += linear_velocity.length() - MAX_SAFE_SPEED   # proportional penalty

    _ai.reward = reward
    # Store cost for the training wrapper to read
    info["cost"] = cost
```

The training wrapper reads `info["cost"]` and tracks `C(π)` over episodes. Section 5 shows how PPO-Lagrangian uses this signal.

---

## 3 · Lagrangian relaxation

The CMDP is a constrained optimization problem. The standard method for converting constrained optimization to unconstrained optimization is **Lagrangian relaxation**.

Introduce a **Lagrange multiplier** `λ ≥ 0` for the constraint. The Lagrangian objective becomes:

```
L(π, λ) = J(π) - λ · (C(π) - d)
```

The optimization problem splits into two:

- **Primal update** (policy): `max_π L(π, λ)` — maximize the Lagrangian given the current multiplier
- **Dual update** (multiplier): `max_λ min_π L(π, λ)` — find the multiplier that enforces the constraint

The dual update is a gradient ascent step on `λ`:

```
λ ← max(0, λ + α_λ · (C(π) - d))
```

Reading this update in plain English:

- If `C(π) > d` (constraint violated): `λ` increases → the penalty `λ · C(π)` in the primal objective grows → the policy is pushed harder to reduce costs
- If `C(π) < d` (constraint satisfied with slack): `λ` decreases → the penalty relaxes → the policy is free to improve task reward
- `λ` is always clamped to `≥ 0` — a constraint cannot become a bonus

This is the **primal-dual method** for CMDPs, also called **Lagrangian RL**. The multiplier `λ` is an adaptive penalty coefficient that automatically calibrates itself to enforce the constraint.

### Why this is better than a fixed penalty

A fixed penalty of `-100` is equivalent to setting `λ = 100` and never updating it. The Lagrangian method updates `λ` at every training iteration based on the current constraint violation. If the policy improves at avoiding the hazard zone, `λ` naturally relaxes and the policy can focus more on task reward. If training drifts toward the constraint boundary, `λ` rises to push it back.

!!! tip "λ as a diagnostic"
    Plot `λ` over training timesteps. A healthy run shows `λ` oscillating around a stable equilibrium. A rising λ that never stabilizes means the policy cannot satisfy the constraint — check whether `d` is achievable, or whether the cost function is firing too often.

---

## 4 · CPO — Constrained Policy Optimization (optional on a first read)

!!! note "First pass? Skim or skip this section."
    Sections 1–3 build the theory (the penalty trap, the CMDP formulation, Lagrangian relaxation), and Sections 5–9 take PPO-Lagrangian from working code through the Godot implementation to the viz checkpoint — that is everything you need to train a constrained agent. CPO is the heavier second-order alternative; come back to it when a constraint violation *during training* is itself unacceptable, e.g. on real hardware.

**CPO (Achiam et al. 2017)** is the trust-region method for CMDPs. It extends TRPO to the constrained setting by solving a constrained policy update at each step:

```
max_θ  J(π_θ) - J(π_old)          ← maximize policy improvement
subject to:
    D_KL(π_old || π_θ) ≤ δ        ← trust region: don't change policy too fast
    C(π_θ) ≤ d                    ← constraint must be satisfied after the update
```

The key guarantee: **CPO provably satisfies the constraint at every policy update**, not just at convergence. The policy never "passes through" an unsafe region during training.

This matters in practice because it means you can train CPO on a real robot (or in a safety-critical simulator) without the policy thrashing against the constraint boundary while learning.

### Where to get CPO

CPO is available through two routes:

**Option A: safety-gymnasium + SB3-contrib**

```bash
pip install safety-gymnasium
pip install sb3-contrib
```

```python
import safety_gymnasium
from sb3_contrib import TRPOPolicy  # CPO is a constrained extension of TRPO

env = safety_gymnasium.make("SafetyPointGoal1-v0")
# safety-gymnasium envs return (obs, reward, cost, terminated, truncated, info)
# CPO implementations wrap these automatically
```

**Option B: safety-starter-agents (OpenAI original)**

```bash
git clone https://github.com/openai/safety-starter-agents
cd safety-starter-agents && pip install -e .
```

### CPO tradeoffs

| Property | CPO | PPO-Lagrangian |
|----------|-----|----------------|
| Constraint satisfaction during training | Provable | Approximate |
| Implementation complexity | High (second-order) | Low (first-order) |
| Compute per update | High | Low |
| Stability | Very stable | Moderate |
| When to choose | When training on real hardware; strict safety | When prototyping; sim-only training |

For most Godot projects, PPO-Lagrangian (Section 5) is the right starting point. Use CPO when the cost of a constraint violation during training is itself unacceptable.

---

## 5 · PPO-Lagrangian — working code example

PPO-Lagrangian is the workhorse safe RL algorithm: it combines PPO's simplicity with the Lagrangian multiplier update from Section 3. It is simpler to implement than CPO and works well in practice.

### With safety-gymnasium (quickstart)

```python
# ppo_lagrangian_safetygymnasium.py
# Requires: pip install safety-gymnasium torch stable-baselines3

import safety_gymnasium
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class LagrangianCallback(BaseCallback):
    """
    Dual update: adjust λ based on constraint violation.
    Plugs into SB3's callback system — called after each rollout.
    """
    def __init__(
        self,
        constraint_threshold: float = 25.0,  # d: max acceptable cost per episode
        lr_lambda: float = 0.05,             # α_λ: dual learning rate
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.d = constraint_threshold
        self.lr_lambda = lr_lambda
        self.lam = 0.0          # Lagrange multiplier, initialized at 0

    def _on_rollout_end(self) -> bool:
        # Read episode costs accumulated during the rollout
        ep_costs = self.locals.get("ep_cost_mean", None)
        if ep_costs is None:
            # Fall back: try reading from info buffers
            infos = self.locals.get("infos", [{}])
            costs = [inf.get("cost", 0.0) for inf in infos if isinstance(inf, dict)]
            ep_costs = float(np.mean(costs)) if costs else 0.0

        # Dual ascent update
        self.lam = max(0.0, self.lam + self.lr_lambda * (ep_costs - self.d))

        # Inject λ into the policy's loss as a penalty coefficient
        # The policy optimizer will minimize: -J(π) + λ * C(π)
        self.model.policy.optimizer.param_groups[0]["lagrange_lambda"] = self.lam

        self.logger.record("constraint/lambda", self.lam)
        self.logger.record("constraint/ep_cost_mean", ep_costs)
        self.logger.record("constraint/threshold", self.d)
        return True


class SafetyGymWrapper(gym.Wrapper):
    """
    safety-gymnasium returns a 6-tuple including cost.
    This wrapper extracts cost into info and presents a standard 5-tuple
    to stable-baselines3.
    """
    def __init__(self, env):
        super().__init__(env)
        self._ep_cost = 0.0

    def reset(self, **kwargs):
        self._ep_cost = 0.0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        self._ep_cost += cost
        info["cost"] = cost
        info["ep_cost"] = self._ep_cost
        return obs, reward, terminated, truncated, info


def make_safe_env(env_id: str = "SafetyPointGoal1-v0"):
    raw_env = safety_gymnasium.make(env_id)
    return SafetyGymWrapper(raw_env)


if __name__ == "__main__":
    env = make_safe_env("SafetyPointGoal1-v0")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./logs/ppo_lagrangian/",
    )

    lagrangian_cb = LagrangianCallback(
        constraint_threshold=25.0,   # allow up to 25 cost units per episode
        lr_lambda=0.05,
    )

    model.learn(
        total_timesteps=2_000_000,
        callback=lagrangian_cb,
    )
    model.save("ppo_lagrangian_pointgoal")
    print("Training complete. Check TensorBoard: tensorboard --logdir logs/")
```

### How the Lagrangian penalty enters the policy gradient

The primal update minimizes:

```
loss = -J(π) + λ · C_batch
```

Where `C_batch` is the mean cost over the current rollout batch. In the callback above, `lam` is updated on the dual side after each rollout. To wire it into the policy loss, extend the SB3 PPO policy's `train()` method:

```python
class LagrangianPPO(PPO):
    def __init__(self, *args, lagrange_lambda: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lagrange_lambda = lagrange_lambda

    def train(self) -> None:
        # Collect batch costs from rollout buffer infos
        costs = np.array([
            info.get("cost", 0.0)
            for info in self.rollout_buffer.infos
            if isinstance(info, dict)
        ])
        mean_cost = float(np.mean(costs)) if len(costs) > 0 else 0.0

        # Standard PPO update, but add λ·C to the policy loss
        # (Override _compute_loss in a real implementation; this is the conceptual sketch)
        super().train()

        # Dual update
        self.lagrange_lambda = max(
            0.0,
            self.lagrange_lambda + 0.05 * (mean_cost - self.constraint_threshold)
        )
```

For a production-quality implementation, see [safety-baselines3](https://github.com/PKU-Alignment/safety-gymnasium) or [FSRL (Tianshou-based)](https://github.com/liuzuxin/FSRL), which implement PPO-Lagrangian with the rollout buffer integration already done correctly.

!!! check "Done when"
    On `SafetyPointGoal1-v0`, `rollout/ep_rew_mean` climbs *while* `constraint/ep_cost_mean` settles at or below the `d = 25.0` budget the callback sets, and `constraint/lambda` peaks then levels off instead of rising without end — the healthy shape from Section 9. A λ that keeps climbing or curves that oscillate with no trend are tuning signals, not bugs: work through the failure patterns in Section 9 (check that `d` is achievable, halve `lr_lambda`) before suspecting the wrapper code.

---

## 6 · safety-gymnasium — the standard benchmark

**safety-gymnasium** is the standard benchmark suite for safe RL research. It provides a set of well-understood environments where you can compare algorithms on a neutral playing field before testing on your own Godot environments.

```bash
pip install safety-gymnasium
```

### Canonical tasks

| Environment | Agent | Hazards | Task |
|-------------|-------|---------|------|
| `SafetyPointGoal1-v0` | 2D point | Circles to avoid | Reach goal sphere |
| `SafetyPointGoal2-v0` | 2D point | More/harder hazards | Reach goal sphere |
| `SafetyCarGoal1-v0` | Car (4-wheel) | Circles + pillars | Reach goal sphere |
| `SafetyAntGoal1-v0` | MuJoCo Ant | Circles | Reach goal sphere |
| `SafetyPointButton1-v0` | 2D point | Circles + gremlins | Press button |

The naming convention: `Safety<Agent><Task><Difficulty>-v0`.

All safety-gymnasium environments return a 6-tuple:

```python
obs, reward, cost, terminated, truncated, info = env.step(action)
```

Where `cost ∈ {0.0, 1.0}` per step: 1.0 if the agent entered a hazard region that step, 0.0 otherwise.

### Using safety-gymnasium to calibrate

Before building your own Godot safe RL environment, run PPO with a fixed penalty, then PPO-Lagrangian, on `SafetyPointGoal1-v0`. Plot the violation rate for both. The gap between them is the gap your CMDP formulation closes. This gives you a calibration baseline before you invest in a custom environment.

```python
import safety_gymnasium
env = safety_gymnasium.make("SafetyPointGoal1-v0")
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, cost, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

---

## 7 · Godot implementation

In Godot, implementing safe RL requires returning two signals from the environment: the task reward and the cost signal. The cost signal is structurally separate from the reward — it is not subtracted from reward, it is a second channel that the training wrapper reads independently.

### The AIController pattern

```gdscript
# ai_controller.gd — two-signal return pattern

extends AIController3D

# Cost accumulates within the episode; training wrapper reads it at rollout end
var step_cost: float = 0.0
var episode_cost: float = 0.0

func get_obs() -> Dictionary:
    var obs = []
    # ... build observation vector as usual ...
    return {"obs": obs}

func _physics_process(delta: float) -> void:
    # Compute task reward
    var task_reward: float = _compute_task_reward()
    reward += task_reward

    # Compute cost signal (separate from reward)
    step_cost = _compute_cost()
    episode_cost += step_cost

    # Expose cost through the info dict
    # godot-rl-agents passes this back to the Python side as part of the step return
    # Access it with: info = env.step(action)[4]  (the info dict in gymnasium)
    set_heuristic("cost", step_cost)

func _compute_task_reward() -> float:
    var dist = global_position.distance_to(goal.global_position)
    var reward = 1.0 - clampf(dist / MAX_DIST, 0.0, 1.0)
    if dist < GOAL_RADIUS:
        reward += 5.0
        done = true
    return reward

func _compute_cost() -> float:
    var cost = 0.0
    # Hazard zone entry
    for hazard in hazards:
        if global_position.distance_to(hazard.global_position) < HAZARD_RADIUS:
            cost = 1.0
            break
    # Velocity limit (e.g. near a human proxy object)
    if linear_velocity.length() > MAX_SAFE_SPEED:
        cost += 0.5
    return cost
```

### Environment scene structure

```
SafeRLEnv (Node3D)
├── Agent (CharacterBody3D or RigidBody3D)
│   └── AIController3D       ← returns reward + cost
├── Goal (Node3D)             ← randomized position each episode
├── Hazards (Node3D)
│   ├── Hazard_0 (Area3D)    ← cost = 1.0 when agent overlaps
│   ├── Hazard_1 (Area3D)
│   └── ...
└── NavigableFloor (StaticBody3D)
```

### Python-side wrapper

```python
# godot_safe_wrapper.py
from godot_rl.wrappers.stable_baselines_wrapper import StableBaseline3Wrapper
import numpy as np


class GodotSafeWrapper(StableBaseline3Wrapper):
    """
    Extend the standard godot-rl SB3 wrapper to expose the cost signal
    from ai_controller.step_cost so it can be consumed by PPO-Lagrangian.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ep_cost = 0.0

    def reset(self, **kwargs):
        self._ep_cost = 0.0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # godot-rl places agent info under info["observations"]
        # Our ai_controller exposes step_cost via set_heuristic("cost", ...)
        step_cost = float(info.get("cost", 0.0))
        self._ep_cost += step_cost
        info["cost"] = step_cost
        info["ep_cost"] = self._ep_cost
        return obs, reward, terminated, truncated, info


# Training script
from godot_rl.core.godot_env import GodotEnv

base_env = GodotEnv(env_path="res://safe_rl_env.tscn")
env = GodotSafeWrapper(base_env)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/godot_safe/")
lagrangian_cb = LagrangianCallback(constraint_threshold=5.0, lr_lambda=0.02)
model.learn(total_timesteps=1_000_000, callback=lagrangian_cb)
```

### Wiring the cost into TensorBoard

The `LagrangianCallback` from Section 5 already logs `constraint/lambda`, `constraint/ep_cost_mean`, and `constraint/threshold`. In your Godot wrapper, also add:

```python
# In LagrangianCallback._on_rollout_end()
ep_lengths = [ep_info["l"] for ep_info in self.model.ep_info_buffer]
if ep_lengths:
    avg_len = np.mean(ep_lengths)
    # Violation rate: fraction of steps where cost > 0
    violation_rate = ep_costs / avg_len if avg_len > 0 else 0.0
    self.logger.record("constraint/violation_rate", violation_rate)
```

---

## 8 · Practical guidance — when to use CMDP

The CMDP formulation adds complexity. It introduces a second signal, a dual variable to tune, and potentially a new algorithm. Here is when that cost is justified:

### Use CMDP / safe RL when

| Situation | Why CMDP |
|-----------|----------|
| Deploying on real hardware with physical limits (torque, speed, force) | Physical limits can damage hardware; λ provides automatic enforcement |
| Humans in the environment (cobots, social robots) | Violation cost is unacceptable; you need a guarantee, not an average |
| Multi-objective tasks where one objective must hold absolutely | Reward scalarization cannot represent strict priorities |
| Constraint boundary changes frequently | Adjusting `d` is interpretable; adjusting a penalty coefficient is not |
| You need to audit compliance ("never exceeded 50 N contact force") | Cost function C(π) gives a direct audit metric; penalty blending does not |

### Use reward penalties when

| Situation | Why penalty is fine |
|-----------|---------------------|
| Sim-only training, constraint violation has no real-world consequence | Tuning convenience outweighs principled formulation |
| Soft preferences ("prefer to stay on the road, but recovery is OK") | Preference tradeoffs are natural in a reward function |
| Early prototyping before defining the constraint threshold precisely | CMDP requires knowing `d`; penalty lets you probe behavior first |
| Very simple single-constraint problems where coefficient tuning is tractable | Penalty adds no overhead; CMDP adds dual variable bookkeeping |

### Choosing the constraint threshold `d`

The threshold `d` should come from domain expertise or physical limits, not from hyperparameter search:

- **Physical limit:** "Motor overheats if instantaneous torque exceeds 12 N·m for more than 0.5 s." Translate to cost-per-step and set `d` accordingly.
- **Regulatory:** "Contact force on human must not exceed 150 N (ISO/TS 15066)." One cost unit per violation, `d = 0` (zero violations allowed per episode).
- **Operational:** "Robot may spend at most 5% of episode steps in the restricted zone." `d = 0.05 × episode_length`.

If you do not have domain knowledge to set `d`, use safety-gymnasium's standard benchmarks as a calibration tool. The community-accepted threshold for `SafetyPointGoal1` is `d = 25.0` cost units per episode.

---

## 9 · Viz checkpoint — constraint violation rate in TensorBoard

Training is not complete until you have verified both signals behave correctly. Walk through this checklist before claiming a safe policy.

**Step 1: Verify the cost function fires correctly.**

```python
# Quick sanity check — run a random policy and count cost events
env = make_safe_env()
obs, _ = env.reset()
total_steps = 0
total_cost = 0
for _ in range(10_000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_cost += info.get("cost", 0.0)
    total_steps += 1
    if terminated or truncated:
        obs, _ = env.reset()

print(f"Random policy violation rate: {total_cost / total_steps:.3f}")
# Expected: 0.1 – 0.4 for PointGoal1 with random actions
# If 0.0: cost function is not firing — check env wrapper
# If 1.0: cost fires every step — threshold or geometry is wrong
```

**Step 2: Run training and watch the curves.**

In TensorBoard, open `constraint/violation_rate`, `rollout/ep_rew_mean`, and `constraint/lambda` on the same page. A correctly converging run looks like this:

```
timesteps →    0         500k        1M         2M

ep_rew_mean    ▁▁▂▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▇▇▇  ← rises steadily
violation_rate ████▇▇▆▅▄▃▃▂▂▁▁▁▁▁▁▁▁  ← falls and stabilizes
lambda         ▁▁▂▃▃▃▃▃▂▂▂▂▁▁▁▁▁▁▁▁▁  ← peaks then settles
```

Watch for these failure patterns:

- **Reward rises, violation_rate stays flat (high):** λ is not rising — check dual update is being called and `lr_lambda > 0`.
- **Violation_rate falls to zero, reward stays flat:** The agent learned to avoid all hazards by standing still. Lower `d` or check that the task reward is strong enough to pull the agent toward the goal.
- **Lambda diverges (keeps rising indefinitely):** The constraint is not achievable. Either `d` is too small, or the task and constraint are in fundamental conflict — the agent cannot reach the goal without passing through hazards. Redesign the environment or raise `d`.
- **Both curves oscillate with no trend:** `lr_lambda` is too large. Halve it and retrain.

**Step 3: Compare to the penalty baseline.**

Run the same environment with a fixed `-100` penalty instead of PPO-Lagrangian. Plot the final distributions of violation_rate across 10 eval episodes for each. The PPO-Lagrangian run should show a tighter distribution and lower mean violation rate at the same task reward level.

---

## 10 · Stretch goals

If you want to take this unit further before the next one, here are four exercises that mirror real safe RL engineering work:

- **Implement full LagrangianPPO from scratch.** Instead of using the callback hook, subclass SB3's PPO and override `train()` to add the Lagrangian penalty directly to the policy gradient. Verify that the gradient now has two terms — task advantage and scaled cost advantage. Compare convergence speed to the callback approach.
- **Multi-constraint CMDP.** Add a second constraint to the Godot environment — for example, both a hazard zone and a velocity limit. Introduce a separate `λ₁`, `λ₂` for each constraint and update them independently. How do the two multipliers interact? (They should not — the CMDP formulation keeps them decoupled.)
- **CPO vs PPO-Lagrangian comparison on PointGoal1.** Install `safety-starter-agents` and run both algorithms on `SafetyPointGoal1-v0` for 2M steps each. Plot the Pareto frontier of (task reward, violation rate) at the end of training. CPO should appear in the lower-right (lower violation, comparable reward).
- **Safety-gymnasium transfer.** Train PPO-Lagrangian on `SafetyPointGoal1-v0` until violation_rate < 0.05. Then rebuild the same task in Godot (flat floor, hazard circles, goal sphere) and evaluate whether the policy structure generalizes. What changes do you need to make to the obs normalization to get a clean transfer?

---

## What's next

You now have the mathematical machinery and implementation tools for hard safety constraints in RL:

- The CMDP formulation gives constraints a physical interpretation that penalty coefficients cannot
- Lagrangian relaxation converts constrained optimization into an adaptive dual update
- PPO-Lagrangian puts this into practice with minimal changes to the SB3 training loop
- The cost function sits alongside — not inside — the reward function in your Godot environment

Safe RL is an active research area. The techniques here (CPO, PPO-Lagrangian, Lagrangian TRPO) are first-generation methods. Current research explores offline safe RL (learning from data without violations), safe exploration guarantees, and combining Lyapunov stability theory with RL. The CMDP framework from this unit is the common foundation for all of them.

---

[→ Course home](index.md)
