# Unit 7 — Multi-agent reinforcement learning

**Source:** https://huggingface.co/learn/deep-rl-course/unit7/introduction

## Learning objectives

- Motivate MARL: humans and agents interact; single-agent is a special case.
- Train agents in **multi-agent** settings (2v2 soccer).
- Understand cooperation / competition basics.

## Theory topics

| Topic | Student takeaway |
|-------|------------------|
| Single-agent vs MARL | Non-stationarity: other agents change the environment |
| Independent learners | Each agent trains own policy (simple baseline) |
| Centralized training / decentralized execution | Common pattern (overview level in HF) |
| Cooperative vs competitive | Team soccer vs opponents |

## Hands-on

| Item | Detail |
|------|--------|
| Environment | 2v2 soccer (Unity ML-Agents) |
| Challenge | AI-vs-AI vs classmates — **broken / leaderboard down** |
| Still works | Train team, observe play locally |

## Maintenance

- AI-vs-AI and leaderboard **non-functional** — document for authors, not students.

## Integration notes (Godot course)

| Godot target | What to borrow |
|--------------|----------------|
| **Unit 7** | MARL motivation, independent learners, 2+ agents in scene |
| Examples | Racer, MultiAgentSimple |
| Skip | HF competition submission flow |

**Pedagogy:** Godot Unit 7 can show viz checkpoint with multiple agents without Unity.
