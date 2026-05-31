# Didactic Polish — Phase 1–2 Pilot

**Date:** 2026-06-01
**Status:** Approved design, pending spec review
**Scope:** Pilot only (7 units). The course-wide sweep is a separate, later effort.

## Problem

The course is technically mature and broad (47 pages, 6 phases + Guides + Reference). A didactic review of nine representative units found the gaps are **pedagogical, not coverage**. Four issues, ranked by impact on learner outcomes:

1. **Self-checks have no answers.** Units end with `!!! info "Self-check before you move on"` asking learners to answer N questions "in your own words" and declaring "if you can answer all N — you're ready." There is no answer key, so a struggling learner cannot tell whether their answers are correct. Open feedback loop.
2. **"Success" is asserted, not measurable.** Built units mostly give directional cues ("`ep_rew_mean` trends upward") rather than a checkable target. Exception: unit-02 uses Lunar Lander `≥ 200`. Learners cannot self-certify completion.
3. **Difficulty cliff + hidden prerequisite chain at Phase 2/3.** `unit-03` (DQN) is dense — PER, target networks, deadly triad, Double/Dueling/Noisy/Rainbow interleaved as subsections. The linear breadcrumb chain (`unit-02 → unit-03`) skips the tabular Q-Learning that builds the intuition, and `unit-03`/`unit-04` only *recommend* the prerequisite units in prose.
4. **Do-vs-read ratio collapses in advanced units.** Early units are hands-on; later units (e.g. `unit-08`, `unit-ppo-deep`, and conceptual units like `unit-curiosity`) push all "build it" work into optional Stretch Goals.

## Goal

Prove a reusable edit pattern for each of the four fixes on **Phase 1–2 (7 units)**, validate the format with the author, then (separately) sweep the pattern across Phases 3–6 + Guides.

Phase 1–2 is the right pilot because it contains the DQN cliff (`unit-03`), at least one read-heavy conceptual unit (`unit-curiosity`), four units with self-checks, and four built/training units — so every one of the four templates can be demonstrated here.

## Pilot units

`unit-00`, `unit-01`, `unit-reward-engineering`, `unit-02`, `unit-q-learning`, `unit-03`, `unit-curiosity`.

## The four edit templates

### Template A — Answer keys

**Applies where** a `!!! info "Self-check before you move on"` admonition exists. In the pilot that is **4 units**: `unit-01` (7 Qs), `unit-02` (5), `unit-q-learning` (5), `unit-03` (5).

**Edit:** directly below the self-check admonition, add a collapsed answer block:

```markdown
??? success "Self-check answers"
    1. <one to three sentence answer>
    2. ...
```

- Collapsed (`???`) so it does not spoil the exercise.
- One concise, correct answer per question, grounded in that unit's own content.
- No new headings; sits inside the unit's existing closing section.

### Template B — "Done when"

**Applies where** a unit has the learner run training: `unit-00` (smoke test), `unit-02` (Lunar Lander), `unit-q-learning` (FrozenLake), `unit-03` (CrossTheRoad DQN). (`unit-curiosity` trains RND — include if a meaningful signal exists.)

**Edit:** a `!!! check "Done when"` admonition adjacent to the training/run step, stating a checkable signal:

```markdown
!!! check "Done when"
    <concrete, observable criterion the learner can verify>
```

**No fabricated numbers.** Three tiers, in priority order:
1. **Known benchmark** → cite it (Lunar Lander solved at `≥ 200` mean return).
2. **No benchmark** → relative/observable target ("mean eval return over 20 deterministic episodes clearly exceeds the random-policy / flat-baseline return").
3. **Smoke test** → process signal ("training logs advance and `ep_rew_mean` is logged without error").

Where a unit already has a soft "you're ready when…" callout (`unit-01`, `unit-02`), standardize/augment it to the measurable form rather than duplicating it.

### Template C — DQN cliff fix (one-time, targeted)

Two edits, both in/around `unit-03`:

1. **Prerequisite-check callout** at the top of `unit-03` (and the same on `unit-04`):

   ```markdown
   !!! info "Prerequisite check"
       Reading straight through? Do **Q-Learning** (tabular intuition) first —
       DQN is "Q-Learning with a neural network," and that lesson makes the
       pieces below click. (unit-04: same note pointing at Actor-Critic.)
   ```

2. **Quarantine the advanced material.** Move `unit-03`'s PER, Double DQN, Dueling, and Noisy/Rainbow subsections under one clearly-marked `## N · Going deeper (optional)` section so the core DQN lesson (replay buffer, target network, ε-decay) reads lean. Preserve all content and any internal anchors other pages link to.

### Template D — Hands-on promotion (demonstrated once in pilot)

**Pilot example:** `unit-curiosity` (read-heavy; RND/ICM conceptual). Promote one existing Stretch-Goal "build it" task into the main body with a minimal code stub and its own `!!! check "Done when"`. Demonstrates the pattern the sweep will apply to other passive units (`unit-08`, `unit-ppo-deep`).

## Cross-cutting decisions

- **German / i18n — English-first.** These units have `.de.md` pairs and the i18n set is "complete." The pilot edits English pages only; German sync is a **tracked follow-up to run before the sweep**, so the didactic format is validated once before doubling the work. This briefly diverges EN/DE on the 7 pilot pages — accepted for the pilot.
- **Validation:** every edited page must pass `mkdocs build --strict` in the `mkdocs-env` conda env (CI runs the same). No new nav entries → the four-file nav rule does not apply.
- **Conventions:** respect existing house style — section headings use the `·` middle dot; any pseudocode lives in a `!!! warning "Pseudocode"` admonition; Python imports must be real and used; prefer adding to existing closing sections over inventing new structure.
- **Anchor safety:** when moving `unit-03` subsections, keep heading text stable where other pages deep-link to it; grep for inbound links before renaming.

## Out of scope (pilot)

- Phases 3–6 + Guides + Reference edits (the sweep).
- German translations of the new blocks.
- Any technical-accuracy rewrite of existing content.
- New units, nav changes, or curriculum restructuring.
- Fixing the stale `docs/curriculum.md` (references old `.html` files) — noted, not part of this work.

## Definition of done (pilot)

- All 7 pilot units carry every template that applies to them (A on 4 units, B on the training units, C on unit-03 + unit-04 callout, D on unit-curiosity).
- `mkdocs build --strict` is green in `mkdocs-env`.
- Changes are on a branch with a diff ready for author review.
- A follow-up item exists for German sync of the pilot blocks.

## Review gate

Author reviews the **pattern** on the 7 units. On approval, the sweep across Phases 3–6 + Guides becomes its own spec → plan → implementation cycle.
