# Unit 6 — Actor-Critic (A2C)

**Source:** https://huggingface.co/learn/deep-rl-course/unit6/introduction

## Learning objectives

- Explain variance problem in REINFORCE (Monte Carlo returns).
- Describe **Actor–Critic** hybrid: actor = policy, critic = value.
- Train with **A2C** in Stable-Baselines3 on robotics tasks.

## Theory topics

| Topic | Student takeaway |
|-------|------------------|
| Actor | Chooses actions (policy-based) |
| Critic | Evaluates actions (value-based) |
| Advantage | \(A(s,a) = Q(s,a) - V(s)\) — how much better than average |
| Variance reduction | Critic provides lower-variance learning signal than raw return |
| A2C | Synchronous advantage actor–critic |

## Hands-on

| Environment | Task |
|-------------|------|
| Robotic arm (Fetch-style) | Move end-effector to target |

## Integration notes (Godot course)

| Godot target | What to borrow |
|--------------|----------------|
| **Units 4–6** | Short “actor / critic” vocabulary when explaining PPO |
| Main algorithm | Course standard is **PPO**, not A2C — mention A2C as related, not primary |

**One-liner for students:** “PPO is also actor–critic style, with a clipped update for stability.”
