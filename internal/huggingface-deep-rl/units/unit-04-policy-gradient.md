# Unit 4 — Policy gradients (REINFORCE)

**Source:** https://huggingface.co/learn/deep-rl-course/unit4/introduction

## Learning objectives

- Contrast value-based (policy from Q) vs **policy-based** (optimize π directly).
- Derive intuition for **policy gradient** methods.
- Implement **Monte Carlo REINFORCE** in PyTorch from scratch.
- Test on CartPole-v1 and PixelCopter.

## Theory topics

| Topic | Student takeaway |
|-------|------------------|
| Policy π(a\|s) | Directly parameterized (e.g. softmax over actions) |
| Policy gradient theorem | Adjust weights to increase probability of high-return actions |
| REINFORCE | Use full-episode return; high variance |
| Credit assignment | Which actions caused good outcomes |

## Hands-on

| Environment | Notes |
|-------------|-------|
| CartPole-v1 | Classic discrete control |
| PixelCopter | Visual inputs; harder |

## Integration notes (Godot course)

| Godot target | What to borrow |
|--------------|----------------|
| **Unit 4** (JumperHard PPO) | “We use PPO, a stable policy-gradient method” — one paragraph bridge from REINFORCE |
| Skip | PixelCopter / from-scratch PyTorch as student requirement |

**Pedagogy:** REINFORCE variance motivates actor–critic (Unit 6) and PPO clipping (Unit 8).
