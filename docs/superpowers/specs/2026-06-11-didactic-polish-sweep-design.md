# Didactic Polish — Course-Wide Sweep (Phases 3–6 + Guides)

**Date:** 2026-06-11
**Status:** Approved pattern (pilot merged 2026-06-01); this spec scopes the sweep. Tracked by issue #68.
**Scope:** The 34 units not covered by the Phase 1–2 pilot, in **both EN and DE**.

## Background

The Phase 1–2 pilot proved four edit templates on 7 units and the author approved the format. The German sync of the pilot (#65) and the early-unit EN/DE drift cleanup (#66) are done, so EN and DE are aligned going into the sweep. Unlike the pilot, the sweep edits **EN and DE together per unit** — no deferred translation pass.

## Scope

All units listed in #68, plus two units added to the course after the issue was filed:

- **Phase 3:** unit-policy-gradients, unit-actor-critic, unit-ppo-deep, unit-04, unit-sac, **unit-sac-applied** (post-issue addition), unit-cleanrl
- **Phase 4:** unit-05, unit-06, unit-visual-observations, unit-07, unit-08, unit-self-play, unit-hierarchical
- **Phase 5:** unit-multitask, unit-09, unit-rlhf, unit-offline-rl, unit-decision-transformer, unit-10, unit-capstone
- **Phase 6:** unit-robotics, unit-locomotion, unit-diffusion-policy, unit-her, unit-sim-to-real, unit-safe-rl
- **Guides:** unit-debugging, unit-evaluation, unit-experiment-tracking, unit-gpu-envs, unit-pbt, unit-world-models, **unit-foundation-models** (post-issue addition)

## The four templates (unchanged from the pilot spec)

### Template A — Answer keys

**Applies where** a `!!! info "Self-check before you move on"` admonition exists. Inventory (2026-06-11): **13 units** — Phase 3: policy-gradients, actor-critic, ppo-deep, 04, sac, cleanrl · Phase 4: 05, 06, 07, 08 · Phase 5: 09, 10, capstone. Units without a self-check do **not** get one invented — that is content authoring, not polish.

**Edit:** a collapsed `??? success "Self-check answers"` block directly below the self-check, one concise answer per question, grounded in the unit's own content. German pages get the same block translated (`??? success "Antworten zum Selbsttest"` or the unit's existing DE self-check phrasing).

### Template B — "Done when"

**Applies where** the unit has the learner run training or a comparison experiment. A `!!! check "Done when"` adjacent to the run step. **No fabricated numbers** — same three tiers as the pilot:

1. Known benchmark → cite it (e.g. CartPole solved at 475+/500, Pendulum ≈ −200).
2. No benchmark → relative/observable target (beats random baseline, characteristic curve shape).
3. Smoke test → process signal (logs advance, metric is emitted without error).

Where a unit already has soft "success criteria" prose, standardize to the measurable form instead of duplicating.

### Template C — Load-cliff fixes

Per-unit judgment call. Two instruments, both proven in the pilot:

1. Prerequisite-emphasis callout/line at the top when a unit hard-depends on an earlier concept unit and the dependency is only softly recommended.
2. `(optional on a first read)` heading suffix + a `!!! note "First pass? Skim or skip this section."` lead on clearly advanced sections, so the core lesson reads lean. Label, don't move — preserves anchors and cross-references.

No breadcrumb-chain rerouting in the sweep unless a breadcrumb is factually stale (points at the wrong neighbor vs `mkdocs.yml` nav).

### Template D — Hands-on promotion

Per-unit judgment call for **read-heavy** units that defer all "build it" work to Stretch Goals: promote exactly one stretch exercise into the main body with a runnable stub and its own "Done when"; delete the duplicated stretch bullet. Candidates flagged at planning time: unit-ppo-deep (Phase 3), unit-08 (Phase 4). Most units keep their stretch goals untouched.

## Cross-cutting decisions

- **EN + DE together.** Every edited `unit-x.md` gets its `unit-x.de.md` edited in the same task. Translate prose and admonition titles only; keep code blocks, file paths, library names, CLI flags, and TensorBoard metric names in English.
- **Phase-sized batches, one PR per batch:** Batch 1 = spec/plan + Phase 3 · Batch 2 = Phase 4 · Batch 3 = Phase 5 · Batch 4 = Phase 6 + Guides.
- **Validation per batch:** `mkdocs build --strict` green (EN + DE), plus an EN/DE structural-alignment check (`??? success` / `!!! check` counts match between each pair).
- **House style:** middle-dot headings, pseudocode only inside `!!! warning "Pseudocode"`, real and used imports, add to existing closing sections rather than inventing structure.
- **No technical-accuracy rewrites, no new units, no nav changes.**

## Definition of done

All 34 units carry every template that applies to them, every EN/DE pair is structurally aligned, `mkdocs build --strict` passes on each batch, and issue #68 can be closed when Batch 4 merges.
