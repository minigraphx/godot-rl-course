# Syllabus — Hugging Face Deep RL Course

Official index: https://huggingface.co/learn/deep-rl-course/unit0/introduction

## Format per unit

| Part | Content |
|------|---------|
| Theory | Markdown lessons + quizzes |
| Hands-on | Google Colab notebooks (+ optional video) |
| Challenges | Leaderboard / AI-vs-AI (mostly deprecated) |

## Units

| Unit | Title | Core algorithms / methods | Key environments | Libraries |
|------|-------|---------------------------|------------------|-----------|
| 0 | Introduction | — | — | HF Hub, Discord, Colab |
| 1 | Introduction to Deep RL | MDP, RL taxonomy, exploration | Lunar Lander | Stable-Baselines3, Hub upload |
| 2 | Q-Learning | MC vs TD, Q-Learning, Bellman | FrozenLake-v1, Taxi-v3 | From scratch (NumPy) |
| 3 | Deep Q-Learning | DQN, replay buffer, target network | Atari (e.g. Space Invaders) | RL Baselines3 Zoo |
| 4 | Policy gradients | REINFORCE, policy gradient theorem | CartPole-v1, PixelCopter | PyTorch from scratch |
| 5 | Unity ML-Agents | Curiosity (RND), engine-based envs | Snowball target, Pyramid | Unity ML-Agents |
| 6 | Actor-Critic | A2C, advantage, actor + critic | Robotic arm (reach) | Stable-Baselines3 |
| 7 | Multi-agent RL | MARL basics, independent learners | 2v2 soccer | Unity ML-Agents |
| 8a | PPO (theory + scratch) | PPO clip, GAE, minibatch updates | Lunar Lander-v2 | CleanRL-style PyTorch |
| 8b | PPO at scale | Async PPO, throughput | VizDoom (Health Gathering, etc.) | Sample Factory |

## Bonus material (course mentions)

| Topic | Notes |
|-------|-------|
| Huggy the Dog | Post–Unit 1 bonus; stick-fetching agent (Unity ML-Agents) |
| Hugging Face Hub | Push/load models, model cards, eval metrics |

## Tools required (HF course)

- Computer + internet
- Google Colab (free tier)
- Hugging Face account (free)

## Certification (HF course only)

| Path | Requirement |
|------|-------------|
| Completion | ≥ 80% of assignments |
| Honors | 100% of assignments |
| Audit | No certificate; full access to content |

Recommended pace: ~1 unit/week, ~3–4 h/week (self-paced).

## Maintenance caveats (document for authors)

- **Unit 7 AI vs AI** — feature broken; training soccer env still works.
- **Leaderboards** — no longer operational.
- Community issue threads on Colab notebooks may contain workarounds.

## Libraries introduced across the course

See [libraries-and-tools.md](libraries-and-tools.md).
