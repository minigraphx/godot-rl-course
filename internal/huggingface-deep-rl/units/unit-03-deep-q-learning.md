# Unit 3 — Deep Q-Learning (DQN)

**Source:** https://huggingface.co/learn/deep-rl-course/unit3/introduction

## Learning objectives

- Explain why Q-tables do not scale to large / continuous state spaces.
- Use a neural network to approximate \(Q(s,a)\).
- Train on Atari via RL Baselines3 Zoo.
- Tune hyperparameters and record eval videos.

## Theory topics

| Topic | Student takeaway |
|-------|------------------|
| Function approximation | Network maps state → Q per action |
| Experience replay | Break correlation; reuse past transitions |
| Target network | Fixed Q-targets for stable training |
| ε-greedy exploration | Random actions early, greedy later |
| Loss | MSE between predicted Q and TD target |

## Hands-on

| Item | Detail |
|------|--------|
| Framework | [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) |
| Environments | Atari (e.g. Space Invaders) — pixel inputs |
| Skills | Train, evaluate, hyperparameter search, plot, video |

## Integration notes (Godot course)

| Godot target | What to borrow |
|--------------|----------------|
| **Unit 3** | ε-greedy callout, replay/target net one-paragraph explanation |
| **CrossTheRoad** | DQN algorithm choice, sparse 2D rewards |
| Skip | Atari-specific preprocessing as required work |

**Pedagogy:** Compare Unit 2 PPO (entropy) vs Unit 3 DQN (ε) in TensorBoard discussion.
