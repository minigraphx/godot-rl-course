# Sequel Bridge & RL Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the approved “After this course” sequel bridge to the course home and curriculum, document RL concept gaps for existing and future units, and embed priority callouts in Units 1–3 without introducing any LLM training content.

**Architecture:** Docs-first bridge (`index.html`, `docs/curriculum.md`); reusable `.callout.warn` pedagogy blocks in HTML units per `docs/html-units.md`; authoring checklist in `docs/curriculum.md` for Units 4–10 when those HTML files are written. Sequel course remains out of repo except a pointer.

**Tech Stack:** Static HTML course pages, Markdown docs, optional MkDocs (`content/index.md`) for parity.

**Spec:** [docs/superpowers/specs/2026-05-24-llm-sequel-course-boundary-design.md](../specs/2026-05-24-llm-sequel-course-boundary-design.md)

---

## File map

| File | Action | Responsibility |
|------|--------|----------------|
| `index.html` | Modify | “After this course” section + sidebar anchor |
| `docs/curriculum.md` | Modify | Matching section + RL gaps authoring checklist |
| `docs/html-units.md` | Modify | Document new pedagogy blocks |
| `Unit-01.html` | Modify | Training-stalled callout near reward design |
| `Unit-02.html` | Modify | Training-stalled callout near reward shaping |
| `Unit-03.html` | Modify | Exploration callout + training-stalled callout |
| `docs/superpowers/specs/2026-05-24-llm-sequel-course-boundary-design.md` | Modify | Status → Implemented (after tasks complete) |
| `content/index.md` | Modify (optional) | MkDocs parity for bridge |
| `Unit-04.html`–`Unit-10.html` | Create later | Include checklist items when authored |

---

## Canonical copy (reuse verbatim)

### Bridge — short (for `index.html`)

```html
<section class="section" id="after-this-course">
  <h2 class="section-header">After this course</h2>
  <div class="callout">
    <strong>Follow-on: model alignment (separate course)</strong>
    <p style="margin:0.5rem 0 0;color:var(--muted);font-size:0.9rem">
      This course stops at vector policies shipped with ONNX. A planned sequel covers
      <strong style="color:var(--text)">language-model alignment</strong> (SFT, preferences, RLHF/DPO) —
      same ideas as here (demonstrations, reward signal, policy improvement) on token sequences instead of game observations.
      <strong style="color:var(--text)">Unit 9 imitation learning</strong> is the best on-ramp. Enrollment is optional; finishing Units 0–10 is complete on its own.
    </p>
    <p style="margin:0.75rem 0 0;color:var(--muted);font-size:0.85rem">
      Alignment course repository: <em>coming soon</em>.
    </p>
  </div>
</section>
```

### Bridge — curriculum (`docs/curriculum.md`)

```markdown
## After this course

This course teaches **Godot + godot-rl-agents + SB3/CleanRL → ONNX**. It does **not** include LLM APIs, Hugging Face training, or RLHF labs.

A separate follow-on course will cover **language-model alignment** (SFT, preference modeling, RLHF/DPO). **Unit 9** (imitation / BC) is the conceptual bridge. Finishing Units 0–10 here is a complete path; the sequel is optional.

| Godot RL | Alignment sequel |
|----------|------------------|
| Reward \(r_t\) | Preference / learned reward model |
| Policy \(\pi(a \mid s)\) | LM policy \(\pi(y \mid x)\) |
| Expert demos (Unit 9) | Supervised fine-tuning |
| PPO improvement | RLHF / DPO |

Alignment course repository: *coming soon* (separate repo when published).
```

### Callout — training stalled (Units 1–3)

```html
<div class="callout warn" id="training-stalled">
  <strong>Training stalled?</strong>
  Check in order: (1) reward sign and scale — is “good” actually positive?
  (2) sparse rewards — does the agent get any signal before the goal?
  (3) observation bugs — are sensors updating after resets?
  (4) TensorBoard flat but Godot looks fine — you may need longer training or a viz checkpoint.
</div>
```

### Callout — exploration (Unit 3, inside DQN section)

```html
<div class="callout tip">
  <strong>Exploration</strong>
  DQN explores with <strong>ε-greedy</strong>: random actions early, greedy Q-actions later.
  If the curve is flat, ε may be decaying too fast — check your training script’s exploration schedule.
  PPO explores via <strong>stochastic policy + entropy</strong>; compare Unit 2’s PPO run in TensorBoard.
</div>
```

### Authoring checklist — Units 4–10 (add to `docs/curriculum.md`)

```markdown
## RL concepts checklist (authoring Units 4–10)

When writing each unit HTML, include:

| Unit | Required content |
|------|------------------|
| 4 | Hyperparameters (learning rate, `n_steps`, clip range); link to training-stalled callout |
| 5 | **Eval protocol**: fixed seed, deterministic policy, report mean episodic return over N episodes |
| 6 | **Normalization**: clip/scale observations and actions for continuous 3D |
| 9 | One sentence: BC demos ↔ SFT in alignment sequel (optional enrollment) |
| 10 | Load checkpoint and resume training; ONNX export |

Do not add LLM, Hugging Face, or RLHF content to Units 0–10.
```

---

### Task 1: Course home bridge

**Files:**
- Modify: `index.html` (after `#units` section, before Training workflow)
- Modify: `index.html` sidebar (add nav link)

- [ ] **Step 1: Add sidebar link**

In `index.html`, inside `<nav>`, after the Units link block, add:

```html
<a href="#after-this-course"><span class="num">→</span> After this course</a>
```

- [ ] **Step 2: Insert bridge section**

After the closing `</section>` of `#units` (line ~136) and before `<section class="section">` for Training workflow, paste the **Bridge — short** HTML from Canonical copy above.

- [ ] **Step 3: Verify in browser**

Run: `open index.html` (macOS) or open file in browser.

Expected: New section visible; anchor `#after-this-course` scrolls correctly; no broken layout.

- [ ] **Step 4: Commit**

```bash
git add index.html
git commit -m "docs: add After this course sequel bridge on course home"
```

---

### Task 2: Curriculum doc bridge + checklist

**Files:**
- Modify: `docs/curriculum.md` (append before `## Docs` final section)

- [ ] **Step 1: Append bridge and checklist**

Insert **Bridge — curriculum** and **Authoring checklist — Units 4–10** blocks from Canonical copy above, placed after `## Extensions` and before `## Training workflow` (or after Training workflow — either is fine; keep Extensions → After this course → RL checklist → Training workflow order).

- [ ] **Step 2: Verify links**

Open `docs/curriculum.md` in preview; confirm table renders and no broken relative links.

- [ ] **Step 3: Commit**

```bash
git add docs/curriculum.md
git commit -m "docs: curriculum sequel bridge and RL gaps authoring checklist"
```

---

### Task 3: Document pedagogy blocks in html-units.md

**Files:**
- Modify: `docs/html-units.md`

- [ ] **Step 1: Add pedagogy list**

Under `## Pedagogy blocks (recurring)`, append:

```markdown
- **Training stalled?** — `.callout.warn#training-stalled` (Units 1–3; link from 4+)
- **After this course** — sequel alignment pointer (index + curriculum only, not per-unit)
- **Exploration** — ε-greedy / entropy tip (Unit 3+)
- **Eval protocol** — deterministic eval over N episodes (Units 5+)
- **Normalization note** — continuous obs/action scaling (Unit 6+)
```

- [ ] **Step 2: Commit**

```bash
git add docs/html-units.md
git commit -m "docs: document sequel bridge and RL gap callout conventions"
```

---

### Task 4: Unit 1 — training stalled callout

**Files:**
- Modify: `Unit-01.html` (Section 2 reward hypothesis area, ~line 402–410)

- [ ] **Step 1: Insert callout**

After the `callout info` block that contains “The reward hypothesis” (ends before “Markov property & MDPs” step), insert **Training stalled** callout from Canonical copy.

- [ ] **Step 2: Verify**

Open `Unit-01.html` in browser; callout uses yellow warn styling consistent with other units.

- [ ] **Step 3: Commit**

```bash
git add Unit-01.html
git commit -m "docs(unit-1): add training stalled troubleshooting callout"
```

---

### Task 5: Unit 2 — training stalled callout

**Files:**
- Modify: `Unit-02.html`

- [ ] **Step 1: Find reward section**

Search for `id="ref-reward"` or first major reward-shaping explanation in Phase B (Lander build).

- [ ] **Step 2: Insert callout**

Place **Training stalled** callout immediately after the first reward-function explanation paragraph (before detailed code steps).

- [ ] **Step 3: Commit**

```bash
git add Unit-02.html
git commit -m "docs(unit-2): add training stalled troubleshooting callout"
```

---

### Task 6: Unit 3 — exploration + training stalled

**Files:**
- Modify: `Unit-03.html`

- [ ] **Step 1: Exploration callout**

Inside `#s1` DQN section, after the closing `</div>` of the `.step` (after PPO comparison paragraph), insert **Exploration** callout from Canonical copy.

- [ ] **Step 2: Training stalled callout**

After the hero `callout info` (“Three ways to see your AI”), insert **Training stalled** callout.

- [ ] **Step 3: Commit**

```bash
git add Unit-03.html
git commit -m "docs(unit-3): add exploration and training stalled callouts"
```

---

### Task 7: Update spec status

**Files:**
- Modify: `docs/superpowers/specs/2026-05-24-llm-sequel-course-boundary-design.md`

- [ ] **Step 1: Change status line**

```markdown
**Status:** Implemented (see plan docs/superpowers/plans/2026-05-24-sequel-bridge-and-rl-gaps.md)
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-24-llm-sequel-course-boundary-design.md
git commit -m "docs: mark sequel boundary spec as implemented"
```

---

### Task 8 (optional): MkDocs home parity

**Files:**
- Modify: `content/index.md`

- [ ] **Step 1: Add admonition**

After `## Units` list, append:

```markdown
## After this course

!!! info "Follow-on: model alignment (separate course)"
    This course ends at ONNX vector policies. A planned sequel covers language-model alignment (SFT, preferences, RLHF/DPO). Unit 9 imitation learning is the on-ramp. Optional — Units 0–10 are complete on their own.

    Alignment course repository: *coming soon*.
```

- [ ] **Step 2: Verify**

Run: `mkdocs build`

Expected: `INFO - Documentation built in ...` with no errors.

- [ ] **Step 3: Commit**

```bash
git add content/index.md
git commit -m "docs(mkdocs): add After this course sequel bridge"
```

---

## Verification (end-to-end)

- [ ] `index.html` and `docs/curriculum.md` state LLMs are **not** in this course and point to optional sequel.
- [ ] Units 1–3 contain `#training-stalled` warn callout.
- [ ] Unit 3 DQN section mentions ε-greedy and PPO entropy comparison.
- [ ] No new conda packages, no Hugging Face, no Unit 14.
- [ ] `docs/curriculum.md` checklist exists for Units 4–10 authors.

---

## Out of scope (this plan)

- Writing `Unit-04.html`–`Unit-10.html` (use checklist when authored).
- Creating sequel repo or `docs/sequel-alignment-curriculum.md`.
- Changes to `godot_rl_course_reference.html` (per spec).
- SAC/TD3 content (Extension 11 only, later).

---

## Spec self-review

| Spec requirement | Plan task |
|------------------|-----------|
| Bridge on index + curriculum | Tasks 1–2 |
| RL gaps Units 1–3 | Tasks 4–6 |
| Gaps Units 4–10 | Checklist in Task 2; applied when units written |
| Unit 9 bridge sentence | Checklist row Unit 9 |
| Unit 10 checkpoint resume | Checklist row Unit 10 |
| No LLM in main course | Out of scope section |
| Sequel outline documented | Task 2 curriculum section |

No TBD placeholders in task steps. MkDocs task marked optional.
