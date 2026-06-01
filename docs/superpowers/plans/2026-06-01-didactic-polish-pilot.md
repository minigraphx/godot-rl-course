# Didactic Polish — Phase 1–2 Pilot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply four didactic improvements (answer keys, measurable "Done when" criteria, a DQN difficulty-cliff fix, one hands-on promotion) to the 7 Phase 1–2 units, proving a reusable pattern before a course-wide sweep.

**Architecture:** Pure Markdown edits to existing files under `content/`. No code, no nav additions. Each task edits one concern, verifies with `mkdocs build --strict`, and commits. English pages only; German `.de.md` sync is a tracked follow-up (Task 12).

**Tech Stack:** MkDocs 1.6.1 + mkdocs-material 9.7.6 + mkdocs-static-i18n 1.3.1, built in the `mkdocs-env` conda env. Material admonitions (`!!! check`, `??? success`) provide the answer-key/criteria UI.

**Spec:** `docs/superpowers/specs/2026-06-01-didactic-polish-pilot-design.md`

**Branch:** `docs/didactic-polish-pilot` (already created; spec already committed here).

---

## Conventions for every task

- **The only correctness gate is the build.** After each task's edits, run:
  ```bash
  conda activate mkdocs-env && mkdocs build --strict
  ```
  Expected: `INFO - Documentation built in N seconds` with **no** WARNING/ERROR lines. `--strict` fails on broken internal links and missing nav.
- House style (enforce, build won't catch): section headings use the middle dot `## N · Title`; runnable Python uses a plain fenced ```python block; only true pseudocode uses `!!! warning "Pseudocode"`; all imports must be real and used.
- Answer-key blocks use **collapsed** `??? success` so they don't spoil the exercise. "Done when" blocks use `!!! check`.
- Commit after each task with a `docs:` message (attribution is disabled globally — do not add a Co-Authored-By trailer).
- All paths below are relative to the repo root `/Users/andreas/dev/Godot-RL-Kurs`.

---

## File map

| File | Tasks | Change |
|------|-------|--------|
| `content/unit-01.md` | 1 | Answer key under self-check |
| `content/unit-02.md` | 2, 5, 8 | Answer key; "Done when" at training step; bottom breadcrumb → Q-Learning |
| `content/unit-q-learning.md` | 3, 6 | Answer key; "Done when" for FrozenLake |
| `content/unit-03.md` | 4, 7, 8, 9 | Answer key; "Done when"; top breadcrumb ← Q-Learning + prereq emphasis; mark advanced sections optional |
| `content/unit-04.md` | 8 | Strengthen Actor-Critic prereq line (parity) |
| `content/unit-curiosity.md` | 10 | Promote count-based exercise into main body |
| `content/unit-00.md` | — | **No change** — already has a `Success criteria` callout + first-evening "Done when" table (Template B already satisfied) |

---

## Task 1: Answer key — unit-01

**Files:**
- Modify: `content/unit-01.md` (after the self-check admonition at lines 517–528, before the bottom breadcrumb at line 530)

- [ ] **Step 1: Insert the answer block**

Insert this block immediately after line 528 (`    If you can answer all seven — you're ready.`) and before the blank line preceding `[→ Reward Engineering]`:

```markdown

??? success "Self-check answers"
    1. **State → action → reward → next state.** The agent observes a state, picks an action, receives a reward, and lands in the next state; repeat until the episode ends.
    2. A **state** fully describes the world; an **observation** is the (often partial) slice of it the agent actually receives. In most Godot envs the agent sees an observation, not the true state.
    3. The **discount rate γ** sets how much future rewards count versus immediate ones. γ→0 makes the agent myopic (only the next reward matters); γ→1 makes it weigh long-term return almost equally.
    4. A **policy π** is the agent's decision function — it maps a state to an action (or a distribution over actions). It is the "brain."
    5. **Monte Carlo** learns from complete episodes using the actual return (unbiased, high variance, must wait for the episode to end). **TD** updates after every step from a bootstrapped estimate rₜ + γV(sₜ₊₁) (lower variance, works online before the episode ends).
    6. PPO keeps a **stochastic policy** and adds an **entropy bonus** that penalises over-confident action distributions, so it keeps sampling varied actions — no explicit ε needed.
    7. Value-based: **DQN** (or Q-Learning) · policy-based: **REINFORCE** (or PPO) · Actor-Critic: **A2C** (or PPO) · model-based: **Dyna-Q** / MuZero (learns a model of transitions and rewards).
```

- [ ] **Step 2: Build**

Run: `conda activate mkdocs-env && mkdocs build --strict`
Expected: builds with no warnings/errors.

- [ ] **Step 3: Commit**

```bash
git add content/unit-01.md
git commit -m "docs: add self-check answer key to unit-01"
```

---

## Task 2: Answer key — unit-02

**Files:**
- Modify: `content/unit-02.md` (after the self-check admonition at lines 581–590, before `**What's next:**` at line 592)

- [ ] **Step 1: Insert the answer block**

Insert immediately after line 590 (`    If you can answer all five — you're ready.`):

```markdown

??? success "Self-check answers"
    1. **`AIController`** exposes the RL interface to Python — `get_obs()`, `get_action_space()`, `set_action()`, `get_reward()`. **`lander.gd`** owns the game itself — physics, thrust, collisions, reward shaping, `game_over()`, `reset()`. The controller is the bridge; the game script is the world.
    2. Neural nets train best on inputs of similar, bounded scale (≈[-1, 1]). Raw positions and velocities differ by orders of magnitude, which ill-conditions the gradients and slows or destabilises learning. Normalising keeps updates well-behaved.
    3. A **terminal** reward (land +, crash −) defines the true goal but is sparse; a **per-step shaped** reward (closer to the pad, angle/fuel penalties) gives dense guidance so the agent gets signal *before* it ever lands. You need the terminal reward to define success and the shaped reward to make it learnable.
    4. The **control mode** selects who drives the agents: *Training* sends observations to Python and applies the actions it returns; *ONNX Inference* runs the exported policy locally with no Python; *Human* lets you drive manually to test rewards and resets.
    5. **ONNX** is a portable, framework-independent model Godot can run at inference time with no Python or SB3. The `.zip` checkpoint only loads back into SB3 in Python.
```

- [ ] **Step 2: Build**

Run: `conda activate mkdocs-env && mkdocs build --strict`
Expected: no warnings/errors.

- [ ] **Step 3: Commit**

```bash
git add content/unit-02.md
git commit -m "docs: add self-check answer key to unit-02"
```

---

## Task 3: Answer key — unit-q-learning

**Files:**
- Modify: `content/unit-q-learning.md` (after the self-check admonition at lines 644–653, before the bottom breadcrumb at line 655)

- [ ] **Step 1: Insert the answer block**

Insert immediately after line 653 (`    If you can answer all five — you're ready for DQN.`):

```markdown

??? success "Self-check answers"
    1. **Q(s, a)** is the expected (discounted) return from taking action a in state s and then acting optimally thereafter — "how good is this action, here."
    2. **Q(s, a) ← Q(s, a) + α · [ r + γ · maxₐ′ Q(s′, a′) − Q(s, a) ]**. α is the **learning rate** (step size); the bracket is the **TD error** δ — the gap between the bootstrapped target r + γ·maxₐ′Q(s′,a′) and the current estimate.
    3. The update bootstraps from **maxₐ′ Q(s′, a′)** (the greedy target policy) regardless of which action exploration actually took. So it learns the *optimal* policy from data generated by a *different* behaviour policy (ε-greedy or even random) — concretely, it can learn from old or exploratory transitions.
    4. **γ = 0** collapses behaviour to pure greed for the immediate reward (myopic); **γ → 1** makes the agent value long-term cumulative return and plan many steps ahead.
    5. FrozenLake has a tiny, discrete, enumerable state space (16 cells), so every Q(s, a) fits in a table. CrossTheRoad has far more (effectively continuous) states than you can enumerate, so it needs a neural network to **generalise** across unseen states.
```

- [ ] **Step 2: Build**

Run: `conda activate mkdocs-env && mkdocs build --strict`
Expected: no warnings/errors.

- [ ] **Step 3: Commit**

```bash
git add content/unit-q-learning.md
git commit -m "docs: add self-check answer key to unit-q-learning"
```

---

## Task 4: Answer key — unit-03

**Files:**
- Modify: `content/unit-03.md` (after the self-check admonition at lines 542–551, before the bottom breadcrumb at line 553)

- [ ] **Step 1: Insert the answer block**

Insert immediately after line 551 (`    If you can answer all five — you're ready.`):

```markdown

??? success "Self-check answers"
    1. Real environments have too many (or continuous) states to store one Q-value each. A **network approximates Q(s, a)** and generalises across similar states it has never seen — a table cannot.
    2. The **replay buffer** breaks the temporal correlation between consecutive transitions (and lets each transition be reused). Random minibatches behave more like i.i.d. data, which stabilises training. On-policy methods discard data after each update, so they never face this correlation/reuse problem.
    3. If the target network is updated **every step**, the bootstrap target moves with the online network — the net chases a constantly shifting target, causing oscillation or divergence. Freezing it for N steps gives a stable target to regress toward.
    4. DQN learns deterministic **Q-values** with no built-in randomness, so it needs an explicit explore/exploit knob — **ε-greedy**. PPO already has a **stochastic policy + entropy bonus**, so exploration is intrinsic and ε is unnecessary.
    5. Example — **CrossTheRoad: DQN.** Discrete actions plus sparse rewards are DQN's sweet spot, and off-policy replay is sample-efficient there. (A dense-reward or continuous-action env would point to PPO instead.)
```

- [ ] **Step 2: Build**

Run: `conda activate mkdocs-env && mkdocs build --strict`
Expected: no warnings/errors.

- [ ] **Step 3: Commit**

```bash
git add content/unit-03.md
git commit -m "docs: add self-check answer key to unit-03"
```

---

## Task 5: "Done when" — unit-02 (surface ≥ 200 at the training step)

**Context:** unit-02 already states `ep_rew_mean ≥ 200` in the far-down "You're ready for Unit 3 when..." checklist (line 557). This task surfaces the same target *at the point of training* (Section 9, `## 9 · Run training (in the editor)`, line 318) so the learner has a checkable goal where they need it. Do **not** duplicate the checklist — add a focused callout.

**Files:**
- Modify: `content/unit-02.md` (within/after Section 9, line 318 onward — place the callout at the end of Section 9, immediately before `## 10 · Monitor training with TensorBoard` at line 378)

- [ ] **Step 1: Read Section 9 to find the exact insertion point**

Run: `sed -n '318,378p' content/unit-02.md` (via Read tool) to confirm the last line of Section 9 before the `## 10` heading.

- [ ] **Step 2: Insert the callout** at the end of Section 9 (just before the `## 10 · Monitor training with TensorBoard` line):

```markdown

!!! check "Done when"
    Mean episodic return reaches **`ep_rew_mean ≥ 200`** — the standard "solved" score for LunarLander. A healthy run climbs past 200 and holds there; watch it in the editor log or TensorBoard. Still under ~100 after 1M steps almost always means a reward-shaping or observation bug — work the *Training stalled?* checklist before training longer.
```

- [ ] **Step 3: Build**

Run: `conda activate mkdocs-env && mkdocs build --strict`
Expected: no warnings/errors.

- [ ] **Step 4: Commit**

```bash
git add content/unit-02.md
git commit -m "docs: surface measurable Done-when target at unit-02 training step"
```

---

## Task 6: "Done when" — unit-q-learning (FrozenLake success rate)

**Files:**
- Modify: `content/unit-q-learning.md` (end of `## 10 · Viz checkpoint`, immediately before `## 11 · Stretch goals` at line 481)

- [ ] **Step 1: Insert the callout** immediately before line 481 (`## 11 · Stretch goals`):

```markdown

!!! check "Done when"
    Evaluate the **greedy** policy (no exploration) over 100 episodes on non-slippery FrozenLake (`is_slippery=False`): it should reach the goal in **~100%** of them. The environment is deterministic, so an optimal Q-table solves it every time. Well below that means too few training episodes or ε never decayed — print ε at the end of training; it should sit near `epsilon_min`.
```

- [ ] **Step 2: Build**

Run: `conda activate mkdocs-env && mkdocs build --strict`
Expected: no warnings/errors.

- [ ] **Step 3: Commit**

```bash
git add content/unit-q-learning.md
git commit -m "docs: add measurable Done-when criterion to unit-q-learning"
```

---

## Task 7: "Done when" — unit-03 (relative/observable criterion)

**Context:** CrossTheRoad has no published benchmark score, so the criterion is relative + visual (spec Template B, tier 2/3). Place it right after the training step (`## 8 · Train headless`, lines 383–399), before `## 9 · Tweak & viz checkpoint` at line 403.

**Files:**
- Modify: `content/unit-03.md` (end of Section 8, immediately before `## 9 · Tweak & viz checkpoint` at line 403)

- [ ] **Step 1: Insert the callout** immediately before line 403 (`## 9 · Tweak & viz checkpoint`):

```markdown

!!! check "Done when"
    CrossTheRoad has no published benchmark, so judge success two ways: (1) the **viz checkpoint** (Section 9) shows the agent reaching the far side in the majority of episodes, and (2) `ep_rew_mean` has clearly stepped up out of its early flat phase and stabilised — the characteristic DQN "flat → sharp jump" curve. A curve still flat after your full step budget points to the ε schedule or a reward-sign bug, not to needing more time.
```

- [ ] **Step 2: Build**

Run: `conda activate mkdocs-env && mkdocs build --strict`
Expected: no warnings/errors.

- [ ] **Step 3: Commit**

```bash
git add content/unit-03.md
git commit -m "docs: add observable Done-when criterion to unit-03"
```

---

## Task 8: DQN cliff — repair the prerequisite breadcrumb chain

**Context (planning discovery):** The linear breadcrumb chain currently runs `unit-02 → unit-03`, **skipping Q-Learning**, and `unit-03`'s top breadcrumb is `← Unit 2: Lunar Lander` (also skipping it). The nav and `index.md` already order Phase 2 as Q-Learning → DQN, so the breadcrumbs are simply stale. `unit-q-learning` already links `← Unit 2` (top) and `→ Deep Q-Learning` (bottom), so inserting it into the chain needs only **two** edits, plus a prereq-emphasis tweak. This is the most direct fix for "linear reader hits DQN before the tabular intuition." (This was approved in spirit under "fix the DQN cliff"; flag it at review if a callout-only approach is preferred instead.)

**Files:**
- Modify: `content/unit-02.md` (bottom breadcrumb line 594, "What's next" prose line 592)
- Modify: `content/unit-03.md` (top breadcrumb line 5; prereq note bullet line 10)
- Modify: `content/unit-04.md` (prereq note line 11 — parity strengthening)

- [ ] **Step 1: Re-point unit-02's bottom breadcrumb to Q-Learning.**

In `content/unit-02.md`, change line 592 from:
```markdown
**What's next:** In Unit 3 you'll study **CrossTheRoad** and train with **DQN**.
```
to:
```markdown
**What's next:** Next is **Q-Learning** — the tabular intuition that makes DQN click — then **CrossTheRoad** with **DQN** in Unit 3.
```
and change line 594 from:
```markdown
[→ Unit 3: CrossTheRoad & DQN](unit-03.md)
```
to:
```markdown
[→ Q-Learning](unit-q-learning.md)
```

- [ ] **Step 2: Re-point unit-03's top breadcrumb to Q-Learning.**

In `content/unit-03.md`, change line 5 from:
```markdown
[← Unit 2: Lunar Lander](unit-02.md) · [Course home](index.md)
```
to:
```markdown
[← Q-Learning](unit-q-learning.md) · [Course home](index.md)
```

- [ ] **Step 3: Strengthen the Q-Learning prerequisite in unit-03.**

In `content/unit-03.md`, change line 10 from:
```markdown
    - **[Q-Learning unit](unit-q-learning.md)** — tabular Q-Learning (recommended; helps Section 2 click instantly)
```
to:
```markdown
    - **[Q-Learning unit](unit-q-learning.md)** — tabular Q-Learning. **Do this first if reading straight through:** DQN is "Q-Learning with a neural network," and the table version makes every trick below click.
```

- [ ] **Step 4: Parity tweak on unit-04's prerequisite.**

In `content/unit-04.md`, change line 11 from:
```markdown
    - **[Actor-Critic unit](unit-actor-critic.md)** (optional but useful) — makes Section 0 click immediately
```
to:
```markdown
    - **[Actor-Critic unit](unit-actor-critic.md)** — **do this first if reading straight through:** PPO is an Actor-Critic method, and Section 0 assumes the actor/critic split.
```

- [ ] **Step 5: Build (this is the link-integrity check — `--strict` fails on any broken target).**

Run: `conda activate mkdocs-env && mkdocs build --strict`
Expected: no warnings/errors. Confirm `unit-q-learning.md`, `unit-03.md`, `unit-04.md` resolve.

- [ ] **Step 6: Commit**

```bash
git add content/unit-02.md content/unit-03.md content/unit-04.md
git commit -m "docs: route Phase 2 breadcrumb chain through Q-Learning before DQN"
```

---

## Task 9: DQN cliff — mark unit-03's advanced material optional

**Context:** unit-03's load spike comes from advanced variants presented at the same weight as the core lesson. The advanced material is already grouped — Section 10 (`## 10 · DQN limitations`, line 417: Double/Dueling DQN) and its `### Noisy Networks and Rainbow DQN` subsection (line 460), plus PER in Section 3.1 (referenced at line 477). This task **labels** them optional without moving content (moving risks breaking the "Section 3.1 above" cross-reference and any inbound anchors).

**Files:**
- Modify: `content/unit-03.md` (Section 10 heading line 417; Rainbow subsection heading line 460)

- [ ] **Step 1: Add an "optional on a first read" lead to Section 10.**

In `content/unit-03.md`, change line 417 from:
```markdown
## 10 · DQN limitations
```
to:
```markdown
## 10 · DQN limitations and variants (optional on a first read)

!!! note "First pass? Skim or skip this section."
    Sections 1–9 are the core DQN lesson and everything you need to train CrossTheRoad. The variants below (Double, Dueling, Noisy, Rainbow) deepen your understanding but are not required to finish the unit — come back to them when you want them.
```

- [ ] **Step 2: Mark the Rainbow subsection advanced.**

In `content/unit-03.md`, change line 460 from:
```markdown
### Noisy Networks and Rainbow DQN
```
to:
```markdown
### Noisy Networks and Rainbow DQN (advanced)
```

- [ ] **Step 3: Build**

Run: `conda activate mkdocs-env && mkdocs build --strict`
Expected: no warnings/errors. (Heading text changes can shift auto-generated anchors — `--strict` will flag any internal link that pointed at the old `#10-dqn-limitations` anchor. If it fails, grep `grep -rn "10-dqn-limitations\|noisy-networks" content/` and update those targets, then rebuild.)

- [ ] **Step 4: Commit**

```bash
git add content/unit-03.md
git commit -m "docs: flag unit-03 DQN variants as optional to flatten the load cliff"
```

---

## Task 10: Hands-on promotion — unit-curiosity (count-based exercise)

**Context:** unit-curiosity is read-heavy with all "build it" work deferred to Stretch Goals. Promote the **Count-based comparison** stretch goal (line 326) into the main body as a runnable exercise with a stub and a "Done when". It reuses the FrozenLake Q-Learning loop from the (now-adjacent in the chain) Q-Learning unit. Then remove the now-duplicated stretch-goal bullet (promote, don't duplicate — DRY).

**Files:**
- Modify: `content/unit-curiosity.md` (insert exercise at the end of `## 6 · Count-based exploration`, immediately before `## 7 · Entropy bonus vs. curiosity` at line 279; remove the duplicate bullet at line 326)

- [ ] **Step 1: Insert the build-it exercise** immediately before line 279 (`## 7 · Entropy bonus vs. curiosity`):

```markdown
### Build it · Count bonus on FrozenLake

Take the FrozenLake Q-Learning loop from the [Q-Learning unit](unit-q-learning.md) and add a `1/√N` novelty bonus. The point: see a count-based intrinsic reward replace ε-greedy as the exploration driver.

```python
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
Q = np.zeros((n_states, env.action_space.n))
N = np.zeros(n_states)                       # state visit counts
alpha, gamma, beta = 0.1, 0.99, 0.1

for episode in range(5000):
    s, _ = env.reset()
    done = False
    while not done:
        a = int(np.argmax(Q[s]))             # greedy — the count bonus drives exploration
        s2, r_ext, terminated, truncated, _ = env.step(a)
        N[s2] += 1
        r_int = beta / np.sqrt(N[s2])         # intrinsic novelty bonus
        r = r_ext + r_int
        Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])
        s, done = s2, terminated or truncated

env.close()
```

!!! check "Done when"
    The count-bonus agent reaches a ~100% greedy success rate (Q-Learning unit's eval) in **noticeably fewer episodes** than the ε-greedy baseline. The `1/√N` term, not ε, is now doing the exploring — confirm by checking it works with a *fully greedy* action selection (no ε at all).
```

- [ ] **Step 2: Remove the now-duplicated stretch-goal bullet.**

In `content/unit-curiosity.md`, delete the bullet at line 326:
```markdown
- **Count-based comparison:** Train the FrozenLake Q-Learning example from the [Q-Learning unit](unit-q-learning.md) with and without a `1/sqrt(N)` count bonus. How many fewer steps are needed to find the optimal policy?
```
(Leave the other three stretch-goal bullets intact.)

- [ ] **Step 3: Build**

Run: `conda activate mkdocs-env && mkdocs build --strict`
Expected: no warnings/errors. Note the nested code fence inside the `### Build it` section — confirm the ```python block closes before the `!!! check` admonition and the page renders (open `site/unit-curiosity/index.html` or `mkdocs serve` and eyeball Section 6).

- [ ] **Step 4: Commit**

```bash
git add content/unit-curiosity.md
git commit -m "docs: promote count-based exercise into unit-curiosity main body"
```

---

## Task 11: Full-site verification

**Files:** none (verification only)

- [ ] **Step 1: Clean strict build of the whole site (English + German).**

Run: `conda activate mkdocs-env && mkdocs build --strict --clean`
Expected: `Documentation built` with zero WARNING/ERROR lines. This rebuilds `site/` and `site/de/` from scratch; `--strict` fails on any broken internal link introduced by the breadcrumb or heading edits.

- [ ] **Step 2: Spot-check the rendered changes.**

Run: `mkdocs serve` and visually confirm in a browser at `localhost:8000`:
- unit-01 / unit-02 / unit-q-learning / unit-03 — the collapsed "Self-check answers" expands and reads correctly.
- unit-02 / unit-q-learning / unit-03 — the "Done when" callouts render as check admonitions at the training step.
- Phase 2 nav-by-breadcrumb now flows unit-02 → Q-Learning → unit-03 (click the Next/Prev links).
- unit-03 Section 10 shows the "optional on a first read" note.
- unit-curiosity Section 6 "Build it" code block and "Done when" render (no broken fences).

- [ ] **Step 3: No commit** (verification only). If Step 1 or 2 surfaces a problem, fix it in the owning file, rebuild, and commit with a `docs:` message.

---

## Task 12: Create the German-sync follow-up

**Context:** Per the English-first decision, the German `.de.md` pages for the 7 pilot units now lag — they need the same answer keys, "Done when" blocks, breadcrumb-chain fix, and the curiosity exercise translated. This is deliberately deferred until the author signs off on the English pattern, and must happen **before** the course-wide sweep.

**Files:** none (issue creation)

- [ ] **Step 1: Open a tracking issue.**

```bash
gh issue create \
  --title "i18n: sync German pages with Phase 1-2 didactic-polish pilot" \
  --body "$(cat <<'EOF'
The Phase 1-2 didactic-polish pilot (branch docs/didactic-polish-pilot) edits English pages only. The German .de.md pages for these 7 units need the same changes translated:

- unit-01, unit-02, unit-q-learning, unit-03: collapsed "Self-check answers" blocks
- unit-02, unit-q-learning, unit-03: "Done when" callouts
- unit-02 / unit-03 breadcrumb chain repair (route through Q-Learning); parity tweak on unit-04 prereq
- unit-03: "optional on a first read" note on Section 10 + Rainbow heading
- unit-curiosity: "Build it · Count bonus on FrozenLake" exercise

Keep code blocks, file paths, library names, and TensorBoard metric names in English; translate prose and breadcrumb labels only. Do this before the course-wide didactic-polish sweep so EN/DE stay aligned.
EOF
)"
```
Expected: prints the new issue URL.

- [ ] **Step 2: Note the issue number** in the PR description when the pilot is opened for review.

---

## Self-review (completed during planning)

**Spec coverage:**
- Template A (answer keys, 4 units) → Tasks 1–4 ✓
- Template B ("Done when", training units) → Tasks 5 (unit-02), 6 (unit-q-learning), 7 (unit-03); unit-00 already satisfied (documented in the file map, no task needed) ✓
- Template C (DQN cliff: prereq + quarantine) → Task 8 (prereq/breadcrumb, incl. unit-04 parity) + Task 9 (mark advanced optional) ✓
- Template D (hands-on promotion, unit-curiosity) → Task 10 ✓
- Cross-cutting: English-first ✓ (all tasks edit `.md`, not `.de.md`); build-strict gate ✓ (every task + Task 11); German follow-up ✓ (Task 12); no-fabricated-numbers ✓ (unit-03 uses a relative criterion; LunarLander 200 and FrozenLake 100% are real)
- Definition of done → Task 11 (green strict build) + branch with reviewable commits ✓

**Deviation from spec, flagged for review:** Task 8 fixes the breadcrumb chain (route through Q-Learning) rather than only adding a prerequisite *callout*. This is the more direct fix for the same root cause and avoids a callout that contradicts the "Next" button. If you prefer the literal callout-only approach, drop Steps 1–2 of Task 8 and keep the prereq strengthening (Steps 3–4).

**Placeholder scan:** none — every inserted block contains final prose/code.

**Consistency:** admonition types consistent (`??? success` for answers, `!!! check` for "Done when"); commit messages all `docs:`; every edit followed by the same `mkdocs build --strict` gate.
