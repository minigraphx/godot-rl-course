# Design: Main course scope, RL gaps, and sequel alignment course

**Date:** 2026-05-24  
**Status:** Implemented (see plan [docs/superpowers/plans/2026-05-24-sequel-bridge-and-rl-gaps.md](../plans/2026-05-24-sequel-bridge-and-rl-gaps.md))  
**Decision:** LLM usage, fine-tuning, and RLHF belong in a **separate follow-on course**, not in Units 0–10.

---

## Summary

The Godot RL course keeps a single spine: **Godot env → socket → SB3/CleanRL → ONNX in Godot**. Advanced alignment topics (SFT, preference learning, RLHF/DPO) are deferred to a prerequisite-gated sequel. The main course gains targeted RL concept coverage and a thin “where next” bridge on the course home and curriculum doc.

---

## Goals

1. **Protect scope** — No LLM labs, Hugging Face training, or RLHF units in the core 0–10 path.
2. **Close RL gaps** — Students learn debugging, eval, exploration, normalization, and hyperparameters without diluting the example ladder.
3. **Set expectations** — Graduates know a sequel exists and how Unit 9 connects conceptually.
4. **Enable a future sequel** — Clear prerequisites and topic outline for a second course (separate repo when built).

---

## Non-goals

- Integrating LLM inference into Godot or ONNX export of language models.
- In-game RLHF or human rating loops in v1 of either course.
- Drafting full sequel lesson HTML in this repo (stub outline only).
- SAC/TD3 or curriculum learning (RL sense) in core units unless added later as Extension 11+ material.

---

## Main course (this repo)

### In scope

| Topic | Placement | Delivery |
|-------|-----------|----------|
| Training stalled (sparse reward, hacking, wrong sign) | Units 1–3 | Sidebar / callout (~1 screen each) |
| Exploration (ε-greedy, policy entropy) | Unit 3 | Section or callout tied to DQN |
| Hyperparameters (what to tweak first) | Units 4–5 | JumperHard + parallel BallChase |
| Eval vs train (deterministic policy, seeds, mean return) | Units 4–5 | Short protocol before/after headless default |
| Observation/action normalization | Unit 6 | Note before continuous 3D examples |
| Checkpoint resume / continue training a policy | Unit 10 | Alongside ONNX export |
| Unit 9 bridge sentence | Unit 9 | One sentence linking BC to SFT in sequel |

### Out of scope

LLM APIs, SFT/DPO/RLHF labs, LoRA, transformer training, Hugging Face as a required dependency.

### Extensions (unchanged numbering)

Units 11–13 remain RL-focused (CleanRL, capstone, self-play). No Unit 14 for LLMs in this repo.

### “Where next” bridge (placement)

Add a subsection on **`index.html`** (after Training workflow or after unit grid) and a matching section in **`docs/curriculum.md`**:

- Title: e.g. **After this course**
- ~3–5 sentences: sequel covers language-model alignment; Unit 9 imitation is the conceptual on-ramp; same ideas (demonstrations, reward signal, policy improvement) different observation space.
- Link placeholder: `TBD — alignment course repo` until the sequel exists.
- Explicit line: **Not required** to finish Godot RL Units 0–10.

Do **not** duplicate the full bridge in `godot_rl_course_reference.html` unless reference doc is later consolidated (user chose home + curriculum only).

---

## Sequel course (future, separate product)

### Working title

*From Godot RL to Model Alignment* (rename allowed; may be standalone without “Godot” in the title).

### Prerequisites

Completion of this course’s Units 0–10, or equivalent:

- MDP loop, reward design, PPO and DQN at practitioner level
- Imitation learning (BC/GAIL) exposure
- Comfort with Python training loops and TensorBoard

### Proposed unit outline (sequel only)

| Seq | Topic | Bridge from Godot RL |
|-----|--------|----------------------|
| 1 | SFT on demonstrations | Unit 9 BC |
| 2 | Preference data & reward models | Reward function design (Units 1–2) |
| 3 | RLHF or DPO | PPO as policy optimization (Units 4+) |
| 4 | LoRA / efficient fine-tuning (optional) | — |
| 5 | Eval harnesses & safety basics (optional) | TensorBoard / eval protocol |
| 6 | Capstone: align a small LM on text (optional Godot trajectory → text later) | — |

### Sequel non-goals (v1)

- Godot socket env for RLHF
- Deploying fine-tuned LMs inside Godot via ONNX
- Full-scale frontier-model training

### Repo strategy

When development starts: **new repository** or top-level `courses/alignment/` — not new `Unit-*.html` files mixed into the core 0–10 ladder in this repo.

---

## Concept mapping (bridge content)

Use this table in the thin bridge copy (abbreviated on `index.html`):

| Godot RL course | Sequel alignment course |
|-----------------|-------------------------|
| Reward \(r_t\) | Preference / learned reward model |
| Policy \(\pi(a \mid s)\) | Language model policy \(\pi(y \mid x)\) |
| Expert demos (Unit 9) | Supervised fine-tuning (SFT) |
| PPO improvement | RLHF / DPO improvement |
| Vector observations | Token sequences |

---

## Implementation notes (main course content only)

Priority when writing Units 4–10:

1. “Training stalled?” sidebar (Units 3–4)
2. Eval protocol (Unit 5)
3. Normalization note (Unit 6)

Lower priority: SAC/TD3 mention in Extension 11 only.

---

## Success criteria

- A reader of `index.html` and `docs/curriculum.md` understands that **LLMs are not part of this course** but have a defined follow-on path.
- Units 0–10 remain example-driven with no new conda packages for transformers.
- Unit 9 explicitly points forward without requiring sequel enrollment.
- Sequel outline is documented enough to spin a new repo without re-litigating scope.

---

## Open items

| Item | Owner | When |
|------|-------|------|
| Sequel repo URL | Course author | When sequel repo is created |
| Full sequel syllabus | Course author | Separate brainstorming cycle |
| Optional `docs/sequel-alignment-curriculum.md` stub | Defer | Only if sequel planning starts within 6 months |

---

## Approvals (brainstorming)

| Decision | Choice |
|----------|--------|
| LLM/RLHF in main course? | No — separate sequel course |
| Main course strategy | B — thin bridge + RL gap fills |
| Bridge placement | A — `index.html` + `docs/curriculum.md` |
| Sequel hands-on depth | Deferred to sequel course design (not integrated capstone in Godot RL) |
