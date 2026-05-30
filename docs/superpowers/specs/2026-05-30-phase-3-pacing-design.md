# Design: Phase 3 pacing — SAC↔CleanRL interlude and reward-engineering reorder

**Date:** 2026-05-30
**Status:** Spec — awaiting implementation plan
**Issue:** [#39](https://github.com/minigraphx/godot-rl-course/issues/39)

---

## Summary

Phase 3 currently runs **3 theory units → unit-04 (JumperHard) → 2 theory units**. The worst dry stretch is the 3 pre-unit-04 theory units (~1581 lines). The post-unit-04 stretch is shorter in count but anchored by the 901-line `unit-cleanrl.md`, making it the heaviest unbroken theory load in the course.

Separately, `unit-reward-engineering` currently sits *after* `unit-02`, but `unit-02` §5 already asks the learner to write potential-based-style reward shaping — theory arrives too late.

This spec resolves both with two targeted changes:

1. Insert a short ~150-line **Apply It interlude** (`unit-sac-applied.md`) between `unit-sac` and `unit-cleanrl`. Learner swaps PPO→SAC on the existing JumperHard project.
2. Move `unit-reward-engineering` to **before** `unit-02` in Phase 1 so theory precedes use.

---

## Goals

1. **Break up the heaviest theory stretch** — give the learner a hands-on checkpoint immediately before the 901-line CleanRL unit.
2. **Fix the reward-theory-after-use ordering bug** in Phase 1.
3. **Preserve the deliberate PG → AC → PPO Deep theory buildup** — do not split or reorder those units.
4. **Reuse the existing JumperHard project** — no new Godot scene work for the interlude.

---

## Non-goals

- Splitting `unit-04` into smaller pieces.
- Reordering `unit-policy-gradients` / `unit-actor-critic` / `unit-ppo-deep` (the deliberate theory buildup).
- Adding "Try this in Godot" sidebars to other theory units (Option C from the issue) — interlude carries the practice load.
- German translation of the new interlude in this PR. The `mkdocs-static-i18n` plugin falls back to English; a follow-up issue tracks `unit-sac-applied.de.md`.
- Changes to BallChase, ICM, or any algorithm/env outside the SAC vs PPO comparison.

---

## Diagnosis (background)

### Phase 3 unit sizes (lines)

| Unit | Lines | Role |
|---|---:|---|
| unit-policy-gradients | 561 | theory |
| unit-actor-critic | 434 | theory |
| unit-ppo-deep | 586 | theory |
| **unit-04 (JumperHard)** | **438** | **Godot hands-on** |
| unit-sac | 473 | theory |
| **unit-cleanrl** | **901** | theory (1.5–2x other units) |

### Dependency graph (from each unit's Prerequisites block)

- `unit-04`: needs `unit-02` + `unit-03` only. Has its own §0 "Why PPO? The theory in 5 minutes" — deliberately self-contained on theory. **Does not require** the new theory units.
- `unit-sac`: needs `unit-04` + `unit-actor-critic` + `unit-03`.
- `unit-cleanrl`: needs `unit-ppo-deep` + `unit-actor-critic` + `unit-04`.

This means any reorder of `unit-04` either improves the pre-stretch at the cost of the post-stretch (or vice versa); total dry time is invariant under reorder alone. **Adding** a hands-on touchpoint is the only way to actually shrink the longest stretch.

### Reward-engineering ordering

- `content/unit-02.md` §5 (lines 181–190) writes potential-based-style shaping in `lander.gd`: progress, velocity damping, leg-contact bonuses.
- `content/unit-reward-engineering.md` §3 teaches the theory of potential-based shaping (`r_shaped = r + γ·Φ(s') − Φ(s)`).
- Current nav has reward-engineering *after* unit-02. Theory arrives too late.

---

## Phase 1 reorder

### New nav order

```yaml
- "Phase 1 — Foundations":
    - "Unit 0 — Setup & First Run": unit-00.md
    - "Unit 1 — RL Foundations": unit-01.md
    - "Reward Engineering": unit-reward-engineering.md   # moved up from after unit-02
    - "Unit 2 — Build Your First Env": unit-02.md
```

### Files touched

| File | Change |
|---|---|
| `mkdocs.yml` | Swap two nav lines under Phase 1. |
| `content/unit-01.md` | Bottom breadcrumb `→ Unit 2` → `→ Reward Engineering`; rewrite "What's next" prose. |
| `content/unit-reward-engineering.md` | Top breadcrumb add `← Unit 1`; bottom breadcrumb `→ Unit 2`; "What's next" prose. |
| `content/unit-reward-engineering.de.md` | Mirror EN breadcrumb + "Was kommt als Nächstes" changes. |
| `content/unit-02.md` | Top breadcrumb `← Reward Engineering`; add reward-engineering to Prerequisites box. |
| `content/unit-02.de.md` | Mirror EN breadcrumb + Voraussetzungen changes. |
| `content/index.md` | Bullet order in the Phase 1 list. |
| `content/index.de.md` | Mirror EN bullet order. |
| `docs/curriculum.md` | If Phase 1 unit order is documented there, sync. |

---

## Phase 3 interlude insert

### New nav order

```yaml
- "Phase 3 — Policy-Based Methods":
    - "Policy Gradients & REINFORCE": unit-policy-gradients.md
    - "Actor-Critic": unit-actor-critic.md
    - "PPO Deep Dive": unit-ppo-deep.md
    - "PPO in Practice (JumperHard)": unit-04.md
    - "SAC — Soft Actor-Critic": unit-sac.md
    - "Apply It — SAC vs PPO on JumperHard": unit-sac-applied.md   # NEW
    - "PPO From Scratch (CleanRL)": unit-cleanrl.md
```

### Files touched

| File | Change |
|---|---|
| `mkdocs.yml` | Insert one line under Phase 3. |
| `content/unit-sac-applied.md` | **NEW** ~150-line interlude (structure below). |
| `content/unit-sac.md` | Bottom breadcrumb `→ CleanRL` → `→ Apply It (SAC vs PPO)`; "What's next" prose. |
| `content/unit-sac.de.md` | Mirror EN breadcrumb change. |
| `content/unit-cleanrl.md` | Top breadcrumb `← SAC` → `← Apply It (SAC vs PPO)`; Prerequisites box gains the interlude. |
| `content/unit-cleanrl.de.md` | Mirror EN breadcrumb + Voraussetzungen changes. |
| `content/index.md` | Bullet in Phase 3 list. |
| `content/index.de.md` | Mirror EN bullet. |

### `unit-sac-applied.md` structure (~150 lines)

```
# Apply It — SAC vs PPO on JumperHard

[← SAC](unit-sac.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~10 min · Training: ~20 min GPU / ~1 hr CPU per algorithm

!!! info "Three ways to see your AI"
    Godot (does SAC's policy look smoother than PPO's?) ·
    TensorBoard (compare ep_rew_mean slopes side by side) ·
    Sample efficiency (how many env steps to reach the same reward?)

!!! note "Prerequisites"
    - [Unit 4](unit-04.md) — JumperHard runs end-to-end
    - [SAC](unit-sac.md) — actor-critic-with-entropy, replay buffer

---

## 1 · Why swap?

One short paragraph: on-policy (PPO) reuses each sample a few times then throws it away; off-policy (SAC) keeps a replay buffer and learns continuously. On JumperHard's continuous action space SAC's sample efficiency *can* dominate, but each gradient step is more expensive. Cross-reference the actual `unit-sac` section on on-policy vs off-policy (section number determined during implementation by inspecting `unit-sac.md`).

## 2 · Edit the training script

Concrete diff: ~25 lines of real Python (not pseudocode).
- `from stable_baselines3 import PPO` → `from stable_baselines3 import SAC`
- Swap hyperparameter set: PPO's `n_steps`/`n_epochs`/`clip_range` → SAC's `buffer_size`/`learning_starts`/`tau`/`ent_coef="auto"`.
- Note replay buffer memory cost (`buffer_size × obs_dim × 4 bytes`).

## 3 · Train both

Shell commands to run two `gdrl` sessions (same JumperHard env config, different algorithm), pointing each to a distinct logdir. How to overlay both in TensorBoard.

## 4 · What you'll see

Expected behavior — written as expected, not measured (see Risks):
- SAC reaches the same `ep_rew_mean` in fewer environment steps.
- PPO often wins on wall-clock time on JumperHard specifically because PPO's gradient step is cheap and JumperHard's env step is also cheap (no real physics simulator overhead).
- Be explicit about the tradeoff — no "SAC always wins".

## 5 · When to actually reach for SAC

Decision guide: continuous robotics-style action, expensive simulation, replay memory is acceptable. Forward-pointer to Phase 6 (`unit-locomotion`, `unit-sim-to-real`) where SAC dominates.

## Stretch Goals

- Plot wall-clock time vs steps for both — does SAC's sample efficiency translate to wall-clock?
- Try SAC on CrossTheRoad (discrete actions) — does it work? Why or why not?
- Sweep SAC's `ent_coef` on JumperHard — what does auto-tuning do?

## What's next

Bridge to CleanRL: "you've now seen PPO and SAC as users; CleanRL strips PPO down to ~400 lines of PyTorch so you can read every gradient step."

[→ PPO From Scratch (CleanRL)](unit-cleanrl.md)
```

Style compliance (per CLAUDE.md): middle-dot section headings (`## 1 · …`), breadcrumbs top+bottom, "Three ways to see your AI" callout near top, Stretch Goals near end, "What's next" closing. Real imports in code (no pseudocode admonition needed — this is application, not new theory).

---

## Verification plan

1. `conda run -n mkdocs-env mkdocs build --strict` passes for both `en` and `de` outputs with zero warnings.
2. Manual grep for orphan references after the moves:
   - `grep -rn "unit-reward-engineering" content/` — every breadcrumb / prereq points to the new position.
   - `grep -rn "unit-sac\\.md\|cleanrl\|sac-applied" content/` — SAC ↔ interlude ↔ CleanRL chain consistent.
3. `mkdocs serve` spot-check:
   - `/` and `/de/` both show Phase 1 with reward-engineering before unit-02.
   - `/` shows Phase 3 with the interlude between SAC and CleanRL.
   - `/de/unit-sac-applied/` falls back to English (per `mkdocs-static-i18n` suffix strategy).
4. PR CI green before merge.

---

## Ship plan

- Branch: `docs/phase-3-pacing` (already created off `main`).
- Commit strategy (two logical commits, squash-merged on merge):
  1. `docs: move reward-engineering before unit-02 in Phase 1` — Phase 1 reorder + all 8 affected files.
  2. `docs: add Apply-It interlude between SAC and CleanRL` — new file + Phase 3 nav + 6 affected files.
- PR title: `docs: insert SAC-vs-PPO interlude + reorder reward-engineering (closes #39)`
- Per `feedback-fix-and-merge-own-prs`: push, wait CI green, squash-merge, delete branch.
- Per `github-closes-footer-multi-issue`: single-issue footer (`closes #39`), reliable. Verify #39 closed post-merge.

---

## Risks

- **Empirical claims in interlude §4.** "What you'll see" describes expected SAC-vs-PPO behavior on JumperHard. I have no historical training data to back the specific shape of those curves. Mitigation: phrase as expected behavior with explicit caveat ("on this specific environment, with these hyperparameters, you should generally observe …"), and flag in the PR description that the comparison was not empirically measured. User can replace with measured numbers later.
- **German translations stale until follow-up.** `unit-sac-applied.de.md` does not ship in this PR. mkdocs-static-i18n falls back to English at the page level, so German nav has an English-titled bullet between SAC and CleanRL. Acceptable per `project-i18n-fallback` memory; track in follow-up issue.
- **`docs/curriculum.md` may have its own Phase ordering.** Will verify during implementation; if it duplicates the nav, it gets updated to match.

---

## Out of scope (deferred)

- BallChase / CrossTheRoad parallels for SAC — left to interlude Stretch Goals as exercises, not new units.
- ICM / RND interludes in Phase 3 — separate pacing question, not this issue.
- Algorithm comparison table across all of Phase 3 — interesting but outside this scope.
