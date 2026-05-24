# Course Plan — Godot RL Course

Internal authoring document. Updated: 2026-05-24

## Vision

One-sentence mission: teach deep RL from first principles to hardware deployment using Godot as the simulation platform, producing students who can read RL papers, train agents on custom environments, and deploy policies to real robots.

## Target student profiles

Three tracks through the course:

| Profile | Goals | Recommended path |
|---------|-------|-----------------|
| Game developer | Add intelligent NPCs to games | Phases 1–5 + Debugging guide |
| Robotics engineer | Train policies for real hardware | Phases 1–4 + 6 + Sim-to-Real |
| RL researcher | Understand algorithms deeply, read papers | All phases + PPO Deep Dive + theory units |

## Complete curriculum map

[Full table of every unit — status: Done / In Progress / Planned / Out of Scope]

Include ALL current units (done), the three being added now (self-play, visual obs, theory bridges — in progress), and planned future content.

### Done ✅

Phase 1 — Foundations:
- Unit 0: Setup & First Run
- Unit 1: RL Foundations (MC/TD, exploration taxonomy, deep RL)
- Unit 2: Build Your First Env (Lunar Lander)
- Reward Engineering (potential-based shaping, failure modes, checklist)

Phase 2 — Value-Based Methods:
- Q-Learning (Bellman, tabular, FrozenLake)
- Deep Q-Learning / DQN (CrossTheRoad, experience replay, target network)
- Intrinsic Motivation & Curiosity (RND, sparse rewards)

Phase 3 — Policy-Based Methods:
- Policy Gradients & REINFORCE (CartPole, variance problem)
- Actor-Critic / A2C (advantage, shared backbone)
- PPO Deep Dive (clipped objective, GAE, hyperparameter math)
- PPO in Practice / JumperHard (tuning, locomotion rewards)
- SAC — Soft Actor-Critic (continuous control, entropy regularization)

Phase 4 — Scale & Complexity:
- Parallel Training (n_parallel, multi-seed evaluation)
- Continuous 3D / FlyBy (normalization, RayCast3D)
- Multi-Agent (cooperative/competitive, CTDE, MARL)
- Memory & POMDPs / RobotFPS (RecurrentPPO, LSTM)

Phase 5 — Beyond Reward:
- Imitation Learning (BC, GAIL, MultiLevelRobot)
- Ship Your Brain (ONNX export, Godot inference, HTML5)

Phase 6 — Robotics:
- Robot Observations & Sensors (proprioception, joint control modes)
- Goal-Conditioned RL & HER (FetchReach/Push, Godot wrapper)
- Sim-to-Real Transfer (domain randomization, deployment checklist)

Guides:
- Debugging RL Training (diagnostic flowchart, TensorBoard reference)

### In Progress 🔄

- Self-Play (AirHockey, league, ELO) → unit-self-play.md
- Visual Observations (SubViewport, CNN, VirtualCamera) → unit-visual-observations.md
- Theory-to-Godot bridges → additions to unit-q-learning.md, unit-policy-gradients.md, unit-actor-critic.md

### Planned 📋

These topics are identified but not yet written. Sorted by priority:

**High priority:**
- Self-play unit (in progress above)
- CleanRL & Sample Factory (Extension 11 from original curriculum) — implement PPO from scratch with CleanRL, train VizDoom with Sample Factory
- Capstone project guide — how to pick an environment, design obs/reward, train, export, ship
- Advanced evaluation — interquartile mean, performance profiles, statistical significance tests (Agarwal et al. 2021)

**Medium priority:**
- Offline RL overview — train from a fixed dataset with no environment interaction (Conservative Q-Learning, Decision Transformer); important for robotics (expensive sim, real data available)
- Multi-task RL — one policy that solves multiple tasks; goal conditioning generalized
- Hierarchical RL overview — high-level policy selects subgoals; low-level policy executes; Options framework
- Population-based training — tune hyperparameters and architecture simultaneously during training

**Lower priority / stretch:**
- Safe RL / Constrained MDP — formal treatment of safety constraints as Lagrangian; CMDP; relevant for real hardware
- World models (Dreamer) — hands-on with a model-based approach; strong for Godot (visual obs + world model)
- RL for NLP (alignment sequel — separate repo)

### Out of scope for this course 🚫

- Hugging Face Hub certification/leaderboards
- Unity ML-Agents hands-on (conceptual parallel only)
- Atari preprocessing (resizing, frame stacking from pixels — covered conceptually in visual obs unit)
- LLM APIs, RLHF, DPO (alignment sequel course)
- Multi-GPU / distributed training beyond n_parallel

## Dependency graph

Which units should be completed before others:

```
Unit 0 → Unit 1 → Unit 2
                ↓
         Reward Engineering → Q-Learning → DQN → Curiosity
                                              ↓
                         Policy Gradients → Actor-Critic → PPO Deep Dive → PPO Practice → SAC
                                                                                ↓
                                                              Parallel → Visual Obs → Multi-Agent → Memory → Self-Play
                                                                                                              ↓
                                                                               Imitation → Ship Your Brain
                                                                                              ↓
                                                                  Robot Sensors → HER → Sim-to-Real
```

## Content quality standards

Every unit must have:
- [ ] Clear learning objectives (implicit in the intro paragraph)
- [ ] "Three ways to see your AI" callout
- [ ] At least one runnable code example
- [ ] At least one connection to an adjacent unit
- [ ] A viz checkpoint or evaluation section
- [ ] 2–3 stretch goals
- [ ] Breadcrumb nav (top and bottom)

Theory units additionally:
- [ ] Every equation explained term-by-term immediately after
- [ ] Plain English before formal notation
- [ ] At least one hands-on Python example (gymnasium or Godot)
- [ ] Theory-to-Godot bridge section (connecting to real gdrl training)

## Estimated completion

| Phase | Status | Estimated student time |
|-------|--------|----------------------|
| Phase 1 | Complete | ~2 weeks |
| Phase 2 | Complete | ~2 weeks |
| Phase 3 | Complete | ~3 weeks |
| Phase 4 | In progress (2 units adding) | ~2 weeks |
| Phase 5 | Complete | ~1.5 weeks |
| Phase 6 | Complete | ~2 weeks |
| Total current | | ~12.5 weeks |
| With planned additions | | ~15–18 weeks |

## Maintenance notes

- MkDocs pinned at 1.6.1 + mkdocs-material 9.7.6 — do NOT upgrade until MkDocs 2.0 stabilizes
- All unit files in `content/` — docs_dir set in mkdocs.yml
- Internal authoring docs in `docs/` and `internal/` — not published
- Gap analysis: `internal/gap-analysis.md`
- HF course reference: `internal/huggingface-deep-rl/`
