# Didactic Polish — Course-wide Sweep (Phases 3–6 + Guides, EN+DE)

**Date:** 2026-06-01
**Status:** Draft design, pending spec review
**Tracks:** GitHub issue #68
**Scope:** The 32 units NOT covered by the Phase 1–2 pilot, in **both EN and DE**.

## Problem

The Phase 1–2 didactic-polish pilot (merged 2026-06-01; German sync via #65; early-unit EN/DE drift repaired via #66) proved four reusable edit templates on 7 units and validated the format with the author. The same four pedagogical gaps — answerless self-checks, asserted-not-measurable success, load cliffs, and a collapsing do-vs-read ratio — persist across the rest of the course. This sweep applies the **already-validated** templates to the remaining 32 units.

## Goal

Apply the four pilot templates across Phases 3–6 + Guides, keeping EN and DE structurally aligned, with every batch passing `mkdocs build --strict`. The format is already author-approved, so this is execution at scale — not a new design.

## The four templates (unchanged from the pilot)

Authoritative definitions live in `docs/superpowers/specs/2026-06-01-didactic-polish-pilot-design.md`. Summary:

- **(A) Answer keys.** Below every `!!! info "Self-check before you move on"`, add a collapsed `??? success "Self-check answers"` block: one concise, correct answer per question, grounded in that unit's own content. Collapsed so it doesn't spoil the exercise.
- **(B) "Done when".** A `!!! check "Done when"` adjacent to a training/run step, stating a checkable signal. **No fabricated numbers**, three tiers in priority order: (1) known benchmark → cite it; (2) no benchmark → relative/observable target; (3) smoke test → process signal. Add **only where a concrete checkable signal exists** — purely conceptual units with no run get no Template B.
- **(C) Load-cliff fix.** Targeted, judgment-based: prerequisite-check callout at the top of a dense unit and/or marking advanced sub-sections `(optional on a first read)` so the core lesson reads lean. Preserve all content and any internal anchors other pages deep-link to. Not every unit has a cliff.
- **(D) Hands-on promotion.** For read-heavy units that defer all "build it" work to Stretch Goals, promote one such exercise into the main body with a minimal real code stub and its own `!!! check "Done when"`. Promote, don't duplicate — remove the now-redundant stretch-goal bullet. Not every unit has a stretch exercise worth promoting.

## Per-unit applicability

Template **A is determined now** (presence of a self-check is a fact). Templates **B/C/D are judgment calls finalized in each phase plan** — the table flags candidates; the plan pins the exact edit (or records "skip" with a one-line reason). All 32 units have `.de.md` pairs, so every edit is mirrored in German.

### Phase 3 — Policy-gradient family

| Unit | A (self-check) | B candidate | C candidate | D candidate |
|------|:---:|:---:|:---:|:---:|
| unit-policy-gradients | ✓ | training run? | — | maybe |
| unit-actor-critic | ✓ | training run? | — | maybe |
| unit-ppo-deep (586 ln) | ✓ | ✓ | possible (dense) | ✓ (named in pilot spec) |
| unit-04 | ✓ | ✓ | prereq line already tuned in pilot | — |
| unit-sac | ✓ | ✓ | — | maybe |
| unit-cleanrl (902 ln) | ✓ | ✓ | **likely (longest unit)** | maybe |

### Phase 4

| Unit | A | B candidate | C candidate | D candidate |
|------|:---:|:---:|:---:|:---:|
| unit-05 | ✓ | ✓ | — | maybe |
| unit-06 | ✓ | ✓ | — | maybe |
| unit-visual-observations | — | ✓ | — | maybe |
| unit-07 | ✓ | ✓ | — | maybe |
| unit-08 (206 ln) | ✓ | ✓ | — | ✓ (named in pilot spec) |
| unit-self-play | — | ✓ | possible | maybe |
| unit-hierarchical (736 ln) | — | ✓ | possible (dense) | maybe |

### Phase 5

| Unit | A | B candidate | C candidate | D candidate |
|------|:---:|:---:|:---:|:---:|
| unit-multitask (1017 ln) | — | ✓ | **likely (longest in course)** | maybe |
| unit-09 | ✓ | ✓ | — | maybe |
| unit-rlhf (650 ln) | — | maybe | possible | maybe |
| unit-offline-rl | — | maybe | — | likely (conceptual) |
| unit-decision-transformer | — | maybe | — | likely (conceptual) |
| unit-10 | ✓ | ✓ | — | maybe |
| unit-capstone (641 ln) | — | ✓ | — | — (capstone is already hands-on) |

### Phase 6

| Unit | A | B candidate | C candidate | D candidate |
|------|:---:|:---:|:---:|:---:|
| unit-robotics | — | maybe | — | maybe |
| unit-locomotion | — | ✓ | — | maybe |
| unit-diffusion-policy | — | maybe | — | likely (conceptual) |
| unit-her | — | ✓ | — | maybe |
| unit-sim-to-real | — | maybe | — | likely (conceptual) |
| unit-safe-rl (640 ln) | — | ✓ | possible | maybe |

### Guides

| Unit | A | B candidate | C candidate | D candidate |
|------|:---:|:---:|:---:|:---:|
| unit-debugging | — | process-signal? | — | maybe |
| unit-evaluation (764 ln) | — | ✓ | possible | maybe |
| unit-experiment-tracking | — | process-signal? | — | maybe |
| unit-gpu-envs | — | maybe | — | — |
| unit-pbt | — | maybe | — | maybe |
| unit-world-models | — | maybe | — | likely (conceptual) |

**Template A is confirmed for exactly 12 units:** unit-policy-gradients, unit-actor-critic, unit-ppo-deep, unit-04, unit-sac, unit-cleanrl, unit-05, unit-06, unit-07, unit-08, unit-09, unit-10.

## Cross-cutting decisions

- **EN+DE together (not English-first).** Unlike the pilot, each unit's `.md` and `.de.md` are edited in the same batch so they never drift (issue #68: "keep EN/DE in sync as each unit is edited"). The pilot already validated the format, so the deferral that justified English-first no longer applies. **Translate prose and breadcrumb labels only; leave code, file paths, library names, and metric names (e.g. `ep_rew_mean`) in English.**
- **No fabricated numbers** (Template B). Where a unit has no benchmark and no clean relative signal, skip B rather than invent a target.
- **Anchor safety** (Template C). Before renaming/moving any heading, grep for inbound deep-links (`grep -rn "#the-anchor" content/`) and update them. Prefer *labeling* a section optional over moving it when a cross-reference or inbound anchor exists.
- **House style** (build won't catch): section headings use the `·` middle dot; runnable Python in a plain ```python block with real, used imports; only true pseudocode in `!!! warning "Pseudocode"`; prefer adding to existing closing sections over inventing new structure. Every unit keeps its top+bottom breadcrumbs, "Three ways to see your AI" callout, Stretch Goals, and "What's next" closing.
- **Validation gate.** Every batch ends with `conda activate mkdocs-env && mkdocs build --strict --clean` green (rebuilds `site/` and `site/de/`; `--strict` fails on broken internal links). No new nav entries → the four-file nav rule does not apply.

## Batching & artifacts

One spec (this document) → **five phase-sized implementation plans**, each executed and merged as **its own PR**:

1. `…-sweep-phase-3.md` — 6 units (policy-gradient family)
2. `…-sweep-phase-4.md` — 7 units
3. `…-sweep-phase-5.md` — 7 units
4. `…-sweep-phase-6.md` — 6 units
5. `…-sweep-guides.md` — 6 units

Each plan: one task per unit (or per concern within a heavy unit), editing EN+DE together, `mkdocs build --strict` after each, a `docs:` commit per task, and a final full-site verification task. Each plan finalizes the B/C/D judgment calls for its units, recording skips with a one-line reason. Plans are written one at a time (writing-plans) and may be executed before later plans are written, so per-phase author feedback can shape the next batch.

## Out of scope

- Phase 1–2 (done in the pilot) and Reference pages.
- Technical-accuracy rewrites of existing content.
- New units, nav changes, or curriculum restructuring.
- Fixing stale `docs/curriculum.md` (`.html` references) — noted, not part of this work.

## Definition of done (whole sweep)

- All 32 listed units carry every template that applies to them (A on the 12 self-check units; B/C/D per the finalized plan judgments).
- EN and DE stay structurally aligned for every edited unit.
- `mkdocs build --strict` is green after every batch.
- Five PRs merged (or a documented decision to consolidate).

## Review gate

Author reviews each phase PR before the next batch is executed. Per-phase feedback (especially on answer-key accuracy and B/C/D judgment calls) feeds forward into the remaining plans.
