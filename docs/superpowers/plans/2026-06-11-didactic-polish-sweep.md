# Didactic Polish — Course-Wide Sweep — Implementation Plan

**Goal:** Apply the four pilot-proven didactic templates (answer keys, "Done when" criteria, load-cliff fixes, hands-on promotion) to the 34 post-pilot units, EN and DE together, in four phase-sized batches.

**Spec:** `docs/superpowers/specs/2026-06-11-didactic-polish-sweep-design.md`
**Pilot reference:** `docs/superpowers/specs/2026-06-01-didactic-polish-pilot-design.md` (templates), `docs/superpowers/plans/2026-06-01-didactic-polish-pilot.md` (worked examples of each template)

---

## Conventions for every batch

- The correctness gate is `mkdocs build --strict` — zero WARNING/ERROR lines, EN + DE.
- Structural-alignment check after each batch: for every edited pair, the counts of `??? success` and `!!! check` in `unit-x.md` and `unit-x.de.md` must match.
- Answer keys: collapsed `??? success`, answers grounded in the unit's own content, one to three sentences each.
- "Done when": `!!! check`, three-tier criterion rule, never a fabricated number.
- DE: translate prose/titles; keep code, paths, flags, library and metric names English.
- One PR per batch; `docs:` commit messages.

## Batch map

| Batch | Units (EN+DE each) | A | B | C/D judgment notes |
|-------|--------------------|---|---|--------------------|
| **1 — Phase 3** | policy-gradients, actor-critic, ppo-deep, 04, sac, sac-applied, cleanrl | 6 units (all but sac-applied) | per unit | ppo-deep is the Phase 3 load-cliff/read-heavy candidate (C and/or D); 04 prereq line already strengthened by pilot |
| **2 — Phase 4** | 05, 06, visual-observations, 07, 08, self-play, hierarchical | 05, 06, 07, 08 | per unit | 08 flagged read-heavy in the pilot spec (D candidate); hierarchical is the longest Phase 4 page (C candidate) |
| **3 — Phase 5** | multitask, 09, rlhf, offline-rl, decision-transformer, 10, capstone | 09, 10, capstone | per unit | multitask is 1000+ lines (C candidate); capstone is checklist-like — B applies to milestones if at all |
| **4 — Phase 6 + Guides** | robotics, locomotion, diffusion-policy, her, sim-to-real, safe-rl + debugging, evaluation, experiment-tracking, gpu-envs, pbt, world-models, foundation-models | none (no self-checks) | per unit | Guides are reference-style: B only where a guide has a runnable experiment; C sparingly; D rarely |

## Per-batch procedure

1. Branch from current `main` (reused session branch is fine).
2. For each unit in the batch, edit the EN page and its `.de.md` in the same task:
   - **A** if a self-check exists → add the answer key below it.
   - **B** if the learner runs/trains something → add "Done when" at the run step (or standardize an existing soft criterion).
   - **C** if a clearly advanced section inflates the first read → label it optional; if a hard prerequisite is only softly recommended → strengthen the line.
   - **D** only if the unit is read-heavy and a stretch goal contains a concrete buildable exercise → promote it once, with stub + "Done when", and delete the duplicate bullet.
3. `mkdocs build --strict`; structural-alignment grep; fix anything flagged.
4. Commit per unit or per template-group; open the batch PR; merge on author approval; reset branch onto main.

## Batch checklists

- [ ] Batch 1 — Phase 3 (7 units) + this spec/plan
- [ ] Batch 2 — Phase 4 (7 units)
- [ ] Batch 3 — Phase 5 (7 units)
- [ ] Batch 4 — Phase 6 (6 units) + Guides (7 units)
- [ ] Close #68 after Batch 4 merges
