# Safe RL / Constrained MDPs

Real deployment requires hard constraints. A robot arm must not exceed joint torque limits; a drone must stay within a geofence; an autonomous vehicle must not exceed speed limits even when that would improve task performance. Standard PPO and SAC optimize reward with no notion of constraint — they will sacrifice safety for performance if the reward function allows it. This unit teaches the formal treatment of safety in RL.

[← Sim-to-Real Transfer](unit-sim-to-real.md) · [Course home](index.md)

---

!!! info "Three ways to see your AI"
    TensorBoard (constraint violation rate alongside `ep_rew_mean`) · Godot (safety boundary visualised as a red zone the agent learns to avoid) · cost curve: watch violations drop during training until the constraint becomes binding

---

## 1 · Why reward penalties aren't enough

The naive approach: add a large negative reward for constraint violations. `r_total = r_task - 100 * constraint_violated`. Simple. But it has a fundamental problem.

**The penalty coefficient has no principled interpretation.** Suppose the task reward is 10 for reaching the goal and -100 for a safety violation. The agent learns to tolerate ~10% violations (since `10 × 0.9 - 100 × 0.1 = 9 - 10 = -1`, which is worse than zero, so it won't tolerate more — but at 5% the math inverts: `10 × 0.95 - 100 × 0.05 = 9.5 - 5 = 4.5`, which is positive). The agent will settle at a violation rate that makes the tradeoff profitable.

If you need **zero violations**, there is no finite penalty that guarantees it during training. The agent will violate when the instantaneous task reward makes it worthwhile.

**The Constrained MDP (CMDP) formulation** replaces the penalty tradeoff with a hard constraint.

---

## 2 · The CMDP formulation

A **Constrained MDP** is an MDP with an additional cost function:

```
Standard RL:
    max J(π) = E_π [Σ_t γ^t · r_t]

CMDP:
    max J(π) = E_π [Σ_t γ^t · r_t]
    subject to:
    C(π) = E_π [Σ_t γ^t · c_t] ≤ d
```

- `c_t` — cost at each timestep (0 if safe, 1 if constraint violated)
- `C(π)` — expected cumulative discounted cost under policy π
- `d` — constraint threshold (e.g., d=0 means no violations allowed; d=0.1 means average 10% violation rate)

The key difference: `C(π)` is a constraint, not a term in the objective. The policy must satisfy `C(π) ≤ d` **regardless of how much task reward it would gain by violating it**.

---

## 3 · Lagrangian relaxation

The standard algorithmic approach: convert the constrained problem to an unconstrained one using a **Lagrange multiplier** λ.

**Lagrangian objective:**

```
L(π, λ) = J(π) - λ · (C(π) - d)
```

The policy and multiplier are updated in alternating steps:

```
Primal update:   π ← argmax_π L(π, λ)     # improve policy (treat λ as fixed penalty coeff)
Dual update:     λ ← λ + α_λ · (C(π) - d)  # increase λ if constraint violated, decrease if slack
```

**Intuition:** λ acts as an adaptive penalty coefficient. If the policy violates the constraint (`C(π) > d`), λ increases, making violations more costly. If the policy has slack (`C(π) < d`), λ decreases, giving the policy more freedom. At convergence, λ settles at the value that makes the constraint exactly binding.

This is **dual ascent** — gradient ascent on λ (to maximize over the constraint violation), gradient ascent on π (to maximize task reward - penalty).

---

## 4 · PPO-Lagrangian

The simplest practical algorithm: run PPO, but add a Lagrangian penalty for constraint violations and update λ every rollout.

```python
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv


class LagrangianCallback(BaseCallback):
    """Updates the Lagrange multiplier λ based on constraint violations."""

    def __init__(self, lambda_lr=0.01, constraint_threshold=0.1, verbose=0):
        super().__init__(verbose)
        self.lambda_lr = lambda_lr       # dual learning rate
        self.d = constraint_threshold     # allowed violation rate
        self.lam = 0.0                    # Lagrange multiplier

    def _on_rollout_end(self):
        # Gather constraint violations from info dicts collected this rollout
        violations = []
        for info in self.model.ep_info_buffer:
            violations.append(info.get("constraint_violated", 0.0))

        if len(violations) == 0:
            return

        c_pi = np.mean(violations)        # estimated C(π)
        self.lam = max(0.0, self.lam + self.lambda_lr * (c_pi - self.d))

        # Inject λ as an adaptive entropy coefficient or add cost to reward
        # Simplest: re-weight the cost in the environment via a wrapper
        if hasattr(self.training_env, "set_lambda"):
            self.training_env.set_lambda(self.lam)

        if self.verbose:
            print(f"λ={self.lam:.4f}  C(π)={c_pi:.4f}  d={self.d:.4f}")
        self.logger.record("safety/lambda", self.lam)
        self.logger.record("safety/violation_rate", c_pi)

    def _on_step(self):
        return True


class SafetyWrapper:
    """Wraps a Godot env to inject λ-weighted cost into the reward."""

    def __init__(self, env, lam=0.0):
        self.env = env
        self.lam = lam
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def set_lambda(self, lam):
        self.lam = lam

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        cost = info.get("cost", 0.0)
        safe_reward = reward - self.lam * cost
        return obs, safe_reward, done, info

    def close(self):
        self.env.close()


# Training
base_env = StableBaselinesGodotEnv(
    env_path="./SafeRobot.x86_64",
    n_parallel=4,
    speedup=20,
)
env = SafetyWrapper(base_env)

lagrangian_cb = LagrangianCallback(
    lambda_lr=0.01,
    constraint_threshold=0.05,  # allow 5% violation rate at most
    verbose=1,
)

model = PPO(
    "MlpPolicy", env,
    verbose=1,
    tensorboard_log="logs/safe_rl/",
)
model.learn(total_timesteps=2_000_000, callback=lagrangian_cb)
model.save("safe_robot_ppo_lagrangian")
env.close()
```

---

## 5 · Godot implementation — two reward signals

The AIController must return both a task reward and a cost signal:

```gdscript
# In AIController.gd
var cost_this_step: float = 0.0

func get_reward() -> float:
    return task_reward_this_step   # task-only reward (no penalty)

func get_info() -> Dictionary:
    return {
        "cost": cost_this_step,
        "constraint_violated": int(cost_this_step > 0.0),
    }

func _physics_process(_delta):
    task_reward_this_step = 0.0
    cost_this_step = 0.0

    # Task reward
    if reached_goal:
        task_reward_this_step = 10.0

    # Cost signal — separate from task reward
    if joint_torque > MAX_TORQUE or outside_geofence():
        cost_this_step = 1.0

    # Small step penalty to encourage efficiency
    task_reward_this_step -= 0.01
```

Keeping cost and reward separate (not combined) lets the Lagrangian callback control the tradeoff adaptively.

---

## 6 · CPO — Constrained Policy Optimization (advanced)

**CPO** (Achiam et al. 2017) is a trust-region method that guarantees constraint satisfaction **during** training — not just at convergence. It solves the CMDP directly with a second-order constraint on the policy update.

CPO replaces the unconstrained update step with:

```
max  L_π (π')          (policy improvement objective)
s.t. KL(π' ∥ π) ≤ δ   (trust region — same as TRPO)
     C(π') ≤ d         (constraint satisfaction)
```

The resulting update is computed via conjugate gradient and line search — the same mechanism as TRPO.

**When to use CPO vs PPO-Lagrangian:**

| | PPO-Lagrangian | CPO |
|--|---------------|-----|
| Guarantee | Converges to constraint satisfaction | Satisfies constraint each update step |
| Implementation | Simple (SB3 callback) | Complex (second-order optimization) |
| Training stability | Depends on λ learning rate | More stable |
| Libraries | This unit | `safety-gymnasium`, `omnisafe` |

**safety-gymnasium** (standard benchmarks):

```bash
pip install safety-gymnasium
```

```python
import safety_gymnasium
import numpy as np
from stable_baselines3 import PPO

env = safety_gymnasium.make("SafetyPointGoal1-v0")

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/")
model.learn(total_timesteps=1_000_000)
```

safety-gymnasium tasks (PointGoal, CarGoal, AntGoal) are standard benchmarks for comparing safe RL algorithms. Start here before adapting the Lagrangian approach to your custom Godot environment.

---

## 7 · When CMDP is overkill

Safe RL adds complexity. Use the simpler penalty approach when:

- **Constraints are soft:** "try not to crash" but crashes don't have hard consequences
- **Violations are rare by design:** the environment is already well-shielded
- **You can tune the penalty coefficient offline** via a hyperparameter sweep
- **Single prototype:** fast iteration matters more than formal guarantees

Use CMDP / PPO-Lagrangian when:

- **Hard deployment constraints:** joint limits, geofences, regulatory requirements
- **Real hardware:** violations cost money or hardware damage
- **The tradeoff is unstable:** performance and safety pull in opposite directions, and a fixed penalty doesn't stabilize

---

## 8 · Viz checkpoint

After training, run the policy with `--viz` and monitor:

- Does the agent stay within the safety boundary (visible red zone in Godot)?
- Constraint violation rate should drop below `d` during training (watch `safety/violation_rate` in TensorBoard)
- `safety/lambda` should increase early (violations happening), then stabilize once the constraint is met
- Final policy: high task reward AND low violation rate — if one is achieved at the cost of the other, λ or the learning rate needs tuning

---

## 9 · Stretch goals

- **PPO-Lagrangian from scratch:** extend the [CleanRL unit](unit-cleanrl.md) single-file PPO to include a cost value function and Lagrangian update. ~50 additional lines.
- **CPO on safety-gymnasium:** install `omnisafe` and run CPO on PointGoal. Compare constraint satisfaction during training vs PPO-Lagrangian.
- **Geofence task in Godot:** design an environment where the agent must reach a goal without leaving a defined boundary (visualised as a wall or floor marker). Wire the cost signal for boundary violations.
- **Constraint tightening:** train with `d=0.2`, then fine-tune with `d=0.05`, then `d=0.0`. Observe how the policy behavior changes as the constraint tightens.

---

## What's next

You have the full robotics stack: algorithm design (PPO, SAC), engineering (reward shaping, debugging), advanced control (curiosity, HER, locomotion), hardware bridge (sim-to-real), and safety. The course continues with optional guides: advanced hyperparameter search and world models.

[→ Course home](index.md)

---

[← Sim-to-Real Transfer](unit-sim-to-real.md) · [Course home](index.md)
