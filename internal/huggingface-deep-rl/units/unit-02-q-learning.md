# Unit 2 — Q-Learning

**Source:** https://huggingface.co/learn/deep-rl-course/unit2/introduction

## Learning objectives

- Understand **value-based** methods.
- Compare **Monte Carlo** vs **Temporal Difference** learning.
- Implement **Q-Learning** from scratch.
- Train on FrozenLake and Taxi.

## Theory topics

| Topic | Student takeaway |
|-------|------------------|
| Value-based RL | Estimate action-values; policy is implicit (e.g. greedy) |
| MC methods | Learn from full episode returns; high variance |
| TD methods | Bootstrap from next step; online updates |
| Bellman optimality | Recursive Q* relationship |
| Q-Learning update | \(Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]\) |
| Off-policy | Learns greedy policy while following exploratory behavior |

## Hands-on

| Environment | Challenge |
|-------------|-----------|
| `FrozenLake-v1` (non-slippery) | Navigate S → G on F tiles, avoid H holes |
| `Taxi-v3` | Pick up passenger, navigate grid |

Implementation: **NumPy from scratch** (no SB3) — builds algorithmic literacy.

## Why this unit matters for Deep RL

Q-tables fail when state space is huge (\(10^9\)–\(10^{11}\) in Atari). Unit 3 replaces table with a network → DQN.

## Integration notes (Godot course)

| Godot target | What to borrow |
|--------------|----------------|
| **Unit 3** (DQN) | TD intuition, Bellman, “greedy from Q” before neural Q |
| **CrossTheRoad** | Sparse reward parallels FrozenLake |
| Skip | Requiring students to code NumPy Q-Learning |

**Suggested callout:** “DQN is Q-Learning with a neural network and replay buffer.”
