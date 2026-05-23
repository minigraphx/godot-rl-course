# Unit 1 — Introduction to Deep Reinforcement Learning

**Source:** https://huggingface.co/learn/deep-rl-course/unit1/introduction

## Learning objectives

- Define RL: agent learns behavior via actions and consequences.
- Master foundations before implementing agents.
- Train Lunar Lander with Stable-Baselines3.
- Upload trained agent to Hugging Face Hub.

## Theory topics (typical HF Unit 1 sequence)

| Topic | Student takeaway |
|-------|------------------|
| RL vs supervised / unsupervised | No labels; reward-only signal |
| Agent, environment, reward | Core loop |
| MDP | States, actions, transitions, rewards, discount γ |
| Policy vs value | Two families of methods |
| Exploration vs exploitation | Balance discovery and reward |
| Types of RL | Value-based, policy-based, model-based (overview) |
| Deep RL | Neural networks approximate policy or value |

## Hands-on

| Item | Detail |
|------|--------|
| Environment | `LunarLander-v2` (Gymnasium) |
| Algorithm | PPO via SB3 (students train before deriving PPO in Unit 8) |
| Deliverable | Working lander + Hub model card |

## Bonus (HF)

Huggy the Dog — fetch stick (Unity ML-Agents); optional after Unit 1.

## Integration notes (Godot course)

| Godot target | What to borrow |
|--------------|----------------|
| **Unit 1** | MDP loop diagram, glossary terms, “no answer key” intuition |
| **Unit 2** | Lunar Lander narrative — same pedagogical env, built in Godot |
| Skip | Hub upload steps as required homework |

**Alignment:** Godot Unit 1 already mirrors HF foundations; ensure glossary terms match [concepts-glossary.md](../concepts-glossary.md).
