# Integration map — HF Deep RL → Godot RL course

Use this when pulling pedagogy into `content/unit-*.md`, `Unit-*.html`, or `docs/curriculum.md`.

**Scope rule:** Core Godot units 0–10 do **not** require Hugging Face Hub, Unity, or Colab. Borrow *concepts* and *debugging patterns*, not certification flows.

## Unit mapping

| HF unit | HF focus | Godot course target | Integration priority | Notes |
|---------|----------|---------------------|----------------------|-------|
| 0 | Course meta, Hub, challenges | — | Skip in student path | Author reference only |
| 1 | RL foundations, Lunar Lander, SB3, Hub | **Unit 1** (MDP), **Unit 2** (Lander) | **High** | Align glossary with [concepts-glossary.md](concepts-glossary.md); Lander narrative already shared |
| 2 | Q-Learning tabular | **Unit 3** (DQN) | **High** | Teach Bellman / TD intuition before DQN; FrozenLake analog = sparse 2D |
| 3 | DQN + Atari | **Unit 3** (CrossTheRoad) | **High** | ε-greedy, replay, target net callouts |
| 4 | REINFORCE | **Unit 4** (PPO intro) | Medium | Policy gradient intuition before PPO; no need for PixelCopter |
| 5 | Unity ML-Agents | **Unit 0–2** (Godot socket) | Medium | “Engine as env builder” — replace Unity with Godot + godot-rl-agents |
| 6 | A2C | **Unit 4–6** | Medium | Advantage / critic ideas; main course uses PPO not A2C |
| 7 | MARL | **Unit 7** | **High** | Racer, MultiAgentSimple; skip broken AI-vs-AI leaderboard |
| 8a | PPO from scratch | **Unit 4**, **Unit 10** | **High** | Hyperparameters, clip, resume; optional Extension 11 CleanRL |
| 8b | Sample Factory + VizDoom | **Extension 11** | Low in core | Beeching / godot-rl-agents lineage; headless throughput parallel to Unit 5 `n_parallel` |

## Concept → Godot unit checklist

| Concept (from HF) | Suggested Godot placement | Already planned? |
|-------------------|---------------------------|------------------|
| MDP loop, reward design | Unit 1 | Yes |
| Training stalled (sparse reward, sign, sensors) | Units 1–3 | Yes (`docs/curriculum.md`) |
| ε-greedy exploration | Unit 3 | Yes |
| PPO entropy exploration | Units 2, 4 | Partial |
| Eval protocol (seed, deterministic, mean return) | Unit 5 | Yes |
| Obs/action normalization | Unit 6 | Yes |
| BC / demos → alignment bridge | Unit 9 | Yes (sequel, not HF) |
| Checkpoint resume | Unit 10 | Yes |
| Experience replay / target network | Unit 3 | When authoring Unit 3 HTML |
| MARL independent learners | Unit 7 | When authoring Unit 7 |

## What not to integrate into Units 0–10

- HF assignment submission / certificate criteria
- Unity Editor or ML-Agents install steps
- VizDoom as required homework (optional Extension reading)
- Deprecated leaderboards and AI-vs-AI competition flow

## Sequel course (alignment) — HF overlap

HF course does **not** teach LLM RLHF. Godot Unit 9 + planned alignment sequel map to **preference / SFT** ideas, not to HF Units 0–8. See [docs/superpowers/specs/2026-05-24-llm-sequel-course-boundary-design.md](../../docs/superpowers/specs/2026-05-24-llm-sequel-course-boundary-design.md).
