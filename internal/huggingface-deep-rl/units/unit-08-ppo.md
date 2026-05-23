# Unit 8 — Proximal Policy Optimization (PPO)

**Sources:**

- Part 1: https://huggingface.co/learn/deep-rl-course/unit8/introduction
- Part 2: https://huggingface.co/learn/deep-rl-course/unit8/introduction-sf

## Learning objectives

### Part 1 — PPO theory + CleanRL-style implementation

- Motivate PPO from A2C: avoid destructively large policy updates.
- **Clip** probability ratio to \([1-\epsilon, 1+\epsilon]\).
- Code PPO from scratch (PyTorch, CleanRL-style).
- Train **LunarLander-v2** — full circle from Unit 1.

### Part 2 — Sample Factory + VizDoom

- Async / high-throughput PPO with Sample Factory.
- Train on VizDoom (Health Gathering Supreme; optional Deathmatch).
- **Author:** Edward Beeching (godot-rl-agents) — cite in Extension 11.

## Theory topics (Part 1)

| Topic | Student takeaway |
|-------|------------------|
| Surrogate objective | Optimize policy with trust region intuition |
| Clipped ratio | \(r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}\) clipped |
| GAE | Generalized Advantage Estimation for credit assignment |
| Hyperparameters | `clip_coef`, `vf_coef`, `ent_coef`, `gae_lambda`, `n_steps`, minibatches, epochs |
| Advantage normalization | Stabilizes updates (`norm_adv`) |

## Hands-on (Part 1)

- Notebook: implement PPO loop, push to Hub with eval score and video.
- Reference: [PPO implementation details (ICLR blog)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

## Hands-on (Part 2)

| Item | Detail |
|------|--------|
| Library | Sample Factory |
| Environment | VizDoom — partial observability, FPS |
| Progression | Health Gathering → harder levels |

## Integration notes (Godot course)

| Godot target | What to borrow |
|--------------|----------------|
| **Unit 4** | Hyperparameter callout (LR, `n_steps`, clip range) |
| **Unit 5** | Parallel envs ↔ Sample Factory throughput idea |
| **Unit 10** | Checkpoint resume + export (not VizDoom requirement) |
| **Extension 11** | CleanRL + Sample Factory as stretch |

**Pedagogy:** Unit 1 trained PPO black-box; Unit 8 explains *why* it works — Godot course can give intuition earlier without full scratch implementation.

## Godot-specific bridge sentence (draft)

> The HF course ends by implementing PPO in Python; your course trains PPO on Godot environments and ships weights to ONNX for in-game inference.
