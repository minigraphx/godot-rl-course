# Phase 3 Pacing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Insert a short SAC-vs-PPO Apply-It interlude between `unit-sac` and `unit-cleanrl`, and move `unit-reward-engineering` to before `unit-02` in Phase 1.

**Architecture:** Pure docs change. Two logical commits on branch `docs/phase-3-pacing`. Verified by `mkdocs build --strict` on both `en` and `de` builds plus manual breadcrumb grep. No code, no Godot scenes, no German translation of the new interlude in this PR (relies on `mkdocs-static-i18n` fallback).

**Tech Stack:** MkDocs + mkdocs-material + mkdocs-static-i18n (pinned in `requirements.txt`). Conda env `mkdocs-env`. Source files under `content/`. Spec at `docs/superpowers/specs/2026-05-30-phase-3-pacing-design.md`.

**Branch:** `docs/phase-3-pacing` (already created off `main`; spec already committed).

---

## File map

**Phase 1 reorder commit (Task 1):** `mkdocs.yml`, `content/index.md`, `content/index.de.md`, `content/unit-01.md`, `content/unit-01.de.md`, `content/unit-reward-engineering.md`, `content/unit-reward-engineering.de.md`, `content/unit-02.md`, `content/unit-02.de.md`.

**Phase 3 interlude commit (Task 2):** `mkdocs.yml`, `content/index.md`, `content/index.de.md`, `content/unit-sac-applied.md` (NEW), `content/unit-sac.md`, `content/unit-sac.de.md`, `content/unit-cleanrl.md`, `content/unit-cleanrl.de.md`.

**Verify + ship (Task 3):** push branch, open PR, merge after CI green, verify #39 closed.

---

### Task 1: Phase 1 — move reward-engineering before unit-02

**Files:**
- Modify: `mkdocs.yml` (Phase 1 nav block, lines ~64–67)
- Modify: `content/index.md` (Phase 1 bullets, lines 29–32)
- Modify: `content/index.de.md` (Phase 1 bullets, lines 29–32)
- Modify: `content/unit-01.md` (bottom breadcrumb, last line)
- Modify: `content/unit-01.de.md` (bottom breadcrumb, last line)
- Modify: `content/unit-reward-engineering.md` (top + bottom breadcrumb, "What's Next" prose)
- Modify: `content/unit-reward-engineering.de.md` (top + bottom breadcrumb, "Was kommt als Nächstes" prose)
- Modify: `content/unit-02.md` (top breadcrumb, Prerequisites block lines 7–11)
- Modify: `content/unit-02.de.md` (top breadcrumb)

- [ ] **Step 1.1: Edit `mkdocs.yml` Phase 1 nav block.**

Old (lines ~64–67):
```yaml
  - "Phase 1 — Foundations":
    - "Unit 0 — Setup & First Run": unit-00.md
    - "Unit 1 — RL Foundations": unit-01.md
    - "Unit 2 — Build Your First Env": unit-02.md
    - "Reward Engineering": unit-reward-engineering.md
```

New:
```yaml
  - "Phase 1 — Foundations":
    - "Unit 0 — Setup & First Run": unit-00.md
    - "Unit 1 — RL Foundations": unit-01.md
    - "Reward Engineering": unit-reward-engineering.md
    - "Unit 2 — Build Your First Env": unit-02.md
```

- [ ] **Step 1.2: Edit `content/index.md` Phase 1 bullets.**

Old (lines 29–32):
```markdown
- [Unit 0 — Setup & First Run](unit-00.md)
- [Unit 1 — RL Foundations](unit-01.md)
- [Unit 2 — Build Your First Env](unit-02.md)
- [Reward Engineering](unit-reward-engineering.md)
```

New:
```markdown
- [Unit 0 — Setup & First Run](unit-00.md)
- [Unit 1 — RL Foundations](unit-01.md)
- [Reward Engineering](unit-reward-engineering.md)
- [Unit 2 — Build Your First Env](unit-02.md)
```

- [ ] **Step 1.3: Edit `content/index.de.md` Phase 1 bullets.**

Old (lines 29–32):
```markdown
- [Einheit 0 — Einrichtung & Erster Start](unit-00.md)
- [Einheit 1 — RL-Grundlagen](unit-01.md)
- [Einheit 2 — Erste eigene Umgebung](unit-02.md)
- [Belohnungsdesign](unit-reward-engineering.md)
```

New:
```markdown
- [Einheit 0 — Einrichtung & Erster Start](unit-00.md)
- [Einheit 1 — RL-Grundlagen](unit-01.md)
- [Belohnungsdesign](unit-reward-engineering.md)
- [Einheit 2 — Erste eigene Umgebung](unit-02.md)
```

- [ ] **Step 1.4: Edit `content/unit-01.md` bottom breadcrumb (last line).**

Old:
```markdown
[→ Unit 2: Build Lunar Lander in Godot](unit-02.md)
```

New:
```markdown
[→ Reward Engineering](unit-reward-engineering.md)
```

- [ ] **Step 1.5: Edit `content/unit-01.de.md` bottom breadcrumb (last line).**

Old:
```markdown
[→ Unit 2: Lunar Lander in Godot bauen](unit-02.md)
```

New:
```markdown
[→ Belohnungsdesign](unit-reward-engineering.md)
```

- [ ] **Step 1.6: Edit `content/unit-reward-engineering.md` top breadcrumb (line 3).**

Old:
```markdown
[← Unit 2: Lunar Lander](unit-02.md) · [Course home](index.md)
```

New:
```markdown
[← Unit 1: Foundations](unit-01.md) · [Course home](index.md)
```

- [ ] **Step 1.7: Edit `content/unit-reward-engineering.md` "What's Next" prose + bottom breadcrumb.**

Old (the existing block):
```markdown
## What's Next

You now have the most important skill in applied RL: designing rewards that produce the behavior you actually want.

In the next unit we leave reward design behind and look at the first real algorithm: **Q-Learning**. You'll learn how the agent uses the rewards you design to build a value function — a map from states to expected future reward.

If you only remember one thing from this unit: **the reward is not a description of the goal. The reward is the goal.** Whatever you write down, that's what the agent will maximize. Make sure that's what you want.

[→ Q-Learning](unit-q-learning.md)
```

New:
```markdown
## What's Next

You now have the most important skill in applied RL: designing rewards that produce the behavior you actually want.

In the next unit you put this directly into practice: building **Lunar Lander** in Godot from scratch, including writing the lander's per-step reward function in GDScript. The shaping you write there in §5 is potential-based — the theory you just learned.

If you only remember one thing from this unit: **the reward is not a description of the goal. The reward is the goal.** Whatever you write down, that's what the agent will maximize. Make sure that's what you want.

[→ Unit 2: Build Lunar Lander in Godot](unit-02.md)
```

- [ ] **Step 1.8: Edit `content/unit-reward-engineering.de.md` top breadcrumb (line 3).**

Old:
```markdown
[← Unit 2: Lunar Lander](unit-02.md) · [Kursstartseite](index.md)
```

New:
```markdown
[← Unit 1: Grundlagen](unit-01.md) · [Kursstartseite](index.md)
```

- [ ] **Step 1.9: Edit `content/unit-reward-engineering.de.md` "Was kommt als Nächstes" prose + bottom breadcrumb.**

First locate the section. It ends with `[→ Q-Learning](unit-q-learning.md)`. Replace the closing block.

Old:
```markdown
In der nächsten Unit lassen wir das Belohnungsdesign hinter uns und betrachten den ersten echten Algorithmus: **Q-Learning**. Du wirst sehen, wie der Agent aus deinen Belohnungen eine Wertfunktion baut — eine Karte von Zuständen auf erwarteten zukünftigen Return.

Wenn du nur eines aus dieser Unit mitnimmst: **die Belohnung ist keine Beschreibung des Ziels. Die Belohnung *ist* das Ziel.** Was immer du aufschreibst — das wird der Agent maximieren. Stelle sicher, dass es das ist, was du willst.

[→ Q-Learning](unit-q-learning.md)
```

New:
```markdown
In der nächsten Unit setzt du das direkt in die Praxis um: du baust **Lunar Lander** in Godot von Grund auf und schreibst die Per-Schritt-Belohnungsfunktion des Landers in GDScript. Das Shaping, das du dort in §5 schreibst, ist potential-basiert — genau die Theorie, die du gerade gelernt hast.

Wenn du nur eines aus dieser Unit mitnimmst: **die Belohnung ist keine Beschreibung des Ziels. Die Belohnung *ist* das Ziel.** Was immer du aufschreibst — das wird der Agent maximieren. Stelle sicher, dass es das ist, was du willst.

[→ Unit 2: Lunar Lander in Godot bauen](unit-02.md)
```

- [ ] **Step 1.10: Edit `content/unit-02.md` top breadcrumb (line 5).**

Old:
```markdown
[← Unit 1: Foundations](unit-01.md) · [Course home](index.md)
```

New:
```markdown
[← Reward Engineering](unit-reward-engineering.md) · [Course home](index.md)
```

- [ ] **Step 1.11: Edit `content/unit-02.md` Prerequisites block to add reward-engineering (lines 7–11).**

Old:
```markdown
!!! note "Prerequisites"
    - **[Unit 0](unit-00.md) complete** — Conda env, Godot .NET, working BallChase run
    - **[Unit 1](unit-01.md) read** — MDP loop, reward, policy, PPO at a high level
    - Comfortable editing GDScript (variables, functions, signals — no advanced features)
    - No PyTorch, no SB3 internals, no game-engine experience required
```

New:
```markdown
!!! note "Prerequisites"
    - **[Unit 0](unit-00.md) complete** — Conda env, Godot .NET, working BallChase run
    - **[Unit 1](unit-01.md) read** — MDP loop, reward, policy, PPO at a high level
    - **[Reward Engineering](unit-reward-engineering.md) read** — potential-based shaping; you'll write the shaped reward in §5
    - Comfortable editing GDScript (variables, functions, signals — no advanced features)
    - No PyTorch, no SB3 internals, no game-engine experience required
```

- [ ] **Step 1.12: Edit `content/unit-02.de.md` top breadcrumb (line 5).**

Old:
```markdown
[← Unit 1: Grundlagen](unit-01.md) · [Kursübersicht](index.md)
```

New:
```markdown
[← Belohnungsdesign](unit-reward-engineering.md) · [Kursübersicht](index.md)
```

- [ ] **Step 1.13: Add reward-engineering to `content/unit-02.de.md` Voraussetzungen.**

Locate the `!!! note "Voraussetzungen"` block (just below the top breadcrumb). Add a new bullet matching the English addition. Read the file first to capture the exact existing bullet text in German, then add a third bullet (after the Unit 1 line) with content equivalent to "**[Belohnungsdesign](unit-reward-engineering.md) gelesen** — potential-basiertes Shaping; du schreibst die geformte Belohnung in §5".

- [ ] **Step 1.14: Verify Phase 1 build.**

Run: `conda run -n mkdocs-env mkdocs build --strict 2>&1 | tail -20`

Expected: `Documentation built` for both en and de directories, exit 0, zero warnings.

- [ ] **Step 1.15: Verify no orphan references to old Phase 1 ordering.**

Run: `grep -rn "Unit 2: Lunar Lander\|→ Q-Learning](unit-q-learning" content/unit-reward-engineering.md content/unit-reward-engineering.de.md`

Expected: empty output (the old breadcrumb strings should be gone from reward-engineering files).

Run: `grep -rn "Unit 2: Build Lunar Lander\|Unit 2: Lunar Lander in Godot bauen" content/unit-01.md content/unit-01.de.md`

Expected: empty output.

- [ ] **Step 1.16: Commit Phase 1 reorder.**

```bash
git add mkdocs.yml content/index.md content/index.de.md \
        content/unit-01.md content/unit-01.de.md \
        content/unit-reward-engineering.md content/unit-reward-engineering.de.md \
        content/unit-02.md content/unit-02.de.md
git commit -m "docs: move reward-engineering before unit-02 (#39)

Theory now precedes use: unit-02 §5 writes potential-based shaping in
lander.gd, but the theory was previously taught after. New Phase 1
order: unit-00 → unit-01 → reward-engineering → unit-02 → Q-Learning.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Phase 3 — insert SAC-vs-PPO interlude

**Files:**
- Modify: `mkdocs.yml` (Phase 3 nav block, lines ~72–79)
- Modify: `content/index.md` (Phase 3 bullets)
- Modify: `content/index.de.md` (Phase 3 bullets)
- Create: `content/unit-sac-applied.md` (NEW, ~150 lines)
- Modify: `content/unit-sac.md` (bottom breadcrumb, last line)
- Modify: `content/unit-sac.de.md` (bottom breadcrumb, last line)
- Modify: `content/unit-cleanrl.md` (top breadcrumb + Prerequisites block)
- Modify: `content/unit-cleanrl.de.md` (top breadcrumb + Voraussetzungen block)

- [ ] **Step 2.1: Edit `mkdocs.yml` Phase 3 nav block.**

Old (lines ~72–79):
```yaml
  - "Phase 3 — Policy-Based Methods":
    - "Policy Gradients & REINFORCE": unit-policy-gradients.md
    - "Actor-Critic": unit-actor-critic.md
    - "PPO Deep Dive": unit-ppo-deep.md
    - "PPO in Practice (JumperHard)": unit-04.md
    - "SAC — Soft Actor-Critic": unit-sac.md
    - "PPO From Scratch (CleanRL)": unit-cleanrl.md
```

New:
```yaml
  - "Phase 3 — Policy-Based Methods":
    - "Policy Gradients & REINFORCE": unit-policy-gradients.md
    - "Actor-Critic": unit-actor-critic.md
    - "PPO Deep Dive": unit-ppo-deep.md
    - "PPO in Practice (JumperHard)": unit-04.md
    - "SAC — Soft Actor-Critic": unit-sac.md
    - "Apply It — SAC vs PPO on JumperHard": unit-sac-applied.md
    - "PPO From Scratch (CleanRL)": unit-cleanrl.md
```

- [ ] **Step 2.2: Edit `content/index.md` Phase 3 bullets.**

First locate the Phase 3 block (`**Phase 3 — Policy-Based Methods**`). Insert one new bullet between the SAC and CleanRL lines:

Old:
```markdown
- [SAC — Soft Actor-Critic](unit-sac.md)
- [PPO From Scratch (CleanRL)](unit-cleanrl.md)
```

New:
```markdown
- [SAC — Soft Actor-Critic](unit-sac.md)
- [Apply It — SAC vs PPO on JumperHard](unit-sac-applied.md)
- [PPO From Scratch (CleanRL)](unit-cleanrl.md)
```

- [ ] **Step 2.3: Edit `content/index.de.md` Phase 3 bullets.**

Read the file first to locate the German Phase 3 block. Insert one new bullet between SAC and CleanRL. The visible link text in German is: `Anwenden — SAC vs PPO auf JumperHard`.

Old (the two adjacent lines in the de Phase 3 block):
```markdown
- [SAC — Soft Actor-Critic](unit-sac.md)
- [PPO von Grund auf (CleanRL)](unit-cleanrl.md)
```

New:
```markdown
- [SAC — Soft Actor-Critic](unit-sac.md)
- [Anwenden — SAC vs PPO auf JumperHard](unit-sac-applied.md)
- [PPO von Grund auf (CleanRL)](unit-cleanrl.md)
```

If the existing German strings differ, mirror the structural insertion at the same position.

- [ ] **Step 2.4: Create `content/unit-sac-applied.md`.**

Write the full file with this content:

```markdown
# Apply It — SAC vs PPO on JumperHard

[← SAC](unit-sac.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~10 min · Training: ~20 min GPU / ~1 h CPU per algorithm

!!! info "Three ways to see your AI"
    Godot (does SAC's policy look smoother than PPO's?) · TensorBoard (compare `ep_rew_mean` slopes side by side) · sample efficiency (how many env steps to reach the same reward?)

!!! note "Prerequisites"
    - **[Unit 4](unit-04.md)** — JumperHard runs end-to-end with PPO
    - **[SAC](unit-sac.md)** — actor-critic-with-entropy, replay buffer, off-policy intuition (especially §5 *PPO vs SAC — the decision guide*)

---

## 1 · Why swap?

You've trained JumperHard with PPO in Unit 4 and read the SAC theory in the previous unit. Time to see the difference for yourself on a Godot env you already know.

PPO is **on-policy**: it samples a batch with the current policy, takes a few gradient steps on it, throws the batch away, and samples again. Simple, robust, easy to parallelise. SAC is **off-policy**: every transition goes into a replay buffer and gets reused for many gradient updates. On JumperHard's continuous action space, SAC's sample efficiency *can* dominate PPO — but each gradient step is more expensive, and the replay buffer eats RAM.

The point of this interlude is not to pick a winner. It's to **feel** the tradeoff on a project you already have running.

---

## 2 · Edit the training script

Open the Python training script you used for JumperHard in Unit 4 (`train_jumperhard.py` or whatever you named it). The PPO version looks roughly like this:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="JumperHard.x86_64", show_window=False)
env = VecMonitor(env)

model = PPO(
    "MlpPolicy", env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.0,
    tensorboard_log="./tb_logs_ppo/",
    verbose=1,
)
model.learn(total_timesteps=200_000)
model.save("jumperhard_ppo")
```

Make a copy and swap PPO for SAC:

```python
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecMonitor
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="JumperHard.x86_64", show_window=False)
env = VecMonitor(env)

model = SAC(
    "MlpPolicy", env,
    learning_rate=3e-4,
    buffer_size=200_000,        # replay buffer capacity (transitions)
    learning_starts=5_000,      # collect random data first
    batch_size=256,
    tau=0.005,                  # soft-update rate for target nets
    gamma=0.99,
    train_freq=1,               # one gradient step per env step
    gradient_steps=1,
    ent_coef="auto",            # automatic entropy temperature
    tensorboard_log="./tb_logs_sac/",
    verbose=1,
)
model.learn(total_timesteps=200_000)
model.save("jumperhard_sac")
```

Notice what disappeared (`n_steps`, `n_epochs`, `clip_range`) and what appeared (`buffer_size`, `learning_starts`, `tau`, `train_freq`, `gradient_steps`, `ent_coef`). These are not the same algorithm with a different name — every line above maps to a different optimization story.

!!! warning "Replay buffer memory"
    `buffer_size=200_000` with a small observation vector (JumperHard is ~tens of floats) is harmless. Bump the buffer for image observations and you'll feel it: 200 k × 84×84×4 bytes ≈ 5.6 GB. The buffer is the price of off-policy.

---

## 3 · Train both

Two terminal sessions, same env config, different algorithm:

```bash
# Terminal 1
python train_jumperhard_ppo.py

# Terminal 2
python train_jumperhard_sac.py
```

Point a single TensorBoard at both logdirs so you can overlay the curves:

```bash
tensorboard --logdir_spec ppo:./tb_logs_ppo,sac:./tb_logs_sac
```

Now open `localhost:6006` and watch `rollout/ep_rew_mean` for both runs simultaneously.

---

## 4 · What you'll see

Expected behaviour on JumperHard (this is **expected**, not a measurement — your numbers will vary with seed, hardware, and SB3 version):

- **SAC's `ep_rew_mean` rises in fewer environment steps.** That is sample efficiency: SAC squeezes more out of each transition because the replay buffer lets every transition contribute to many updates.
- **PPO often wins on wall-clock time.** JumperHard's environment step is cheap, PPO's gradient step is cheap, and PPO's data path is simpler. SAC's per-step cost (gradient step + target-net update + entropy temperature update) eats its sample-efficiency advantage in wall time on this env.
- **SAC's policy can look smoother in Godot.** Continuous action distributions with automatic entropy tuning often produce less jittery control than a clipped PPO policy that's still bleeding entropy.
- **SAC is more sensitive to hyperparameters early on.** `learning_starts` too low and the critic is fitting garbage; `tau` too high and the target nets oscillate. PPO's hyperparameters are forgiving by comparison.

If you don't see SAC reach the same reward in fewer env steps, check `ent_coef` (auto-tuning may have decayed entropy too fast — try `ent_coef=0.2` fixed) and `buffer_size` (too small means the buffer is dominated by stale early-training data).

---

## 5 · When to actually reach for SAC

A short decision guide once the experiment is done:

| Situation | Pick |
|---|---|
| Continuous actions, expensive simulation (real robot, physics-heavy sim) | **SAC** — sample efficiency matters more than wall-clock |
| Cheap parallel envs, discrete or continuous actions | **PPO** — easier to scale, more forgiving |
| You need stable training out of the box with little tuning | **PPO** |
| You want to push state of the art on continuous control benchmarks | **SAC** (or TD3) |
| Tight RAM budget, can't afford a replay buffer | **PPO** |

You'll meet SAC again in [Phase 6 — Locomotion](unit-locomotion.md) and [Sim-to-Real](unit-sim-to-real.md), where its sample efficiency stops being a curiosity and becomes essential.

---

## Stretch Goals

- **Wall-clock vs steps.** Re-train both with `time` and plot wall-clock seconds vs environment steps. Does SAC's sample-efficiency advantage translate into wall-clock advantage on JumperHard? Why or why not?
- **SAC on CrossTheRoad.** Try the SAC script on the discrete-action CrossTheRoad env from Unit 3. It will fail or behave badly — figure out why before reading the SAC docs.
- **Entropy-temperature sweep.** Train SAC with `ent_coef ∈ {0.05, 0.1, 0.2, "auto"}` and compare. What does the auto-tuner converge toward on JumperHard?

---

## What's next

You've now seen PPO and SAC as **users** — picking an algorithm class and trusting the library. Next, you peel one layer off: **CleanRL** strips PPO down to ~400 lines of single-file PyTorch so you can read every gradient step. Useful when SB3 is too opaque to debug, when you need a custom loss, or when a paper's algorithm has no library implementation yet.

[→ PPO From Scratch (CleanRL)](unit-cleanrl.md)
```

- [ ] **Step 2.5: Edit `content/unit-sac.md` bottom breadcrumb (last line).**

Old:
```markdown
[→ PPO From Scratch (CleanRL)](unit-cleanrl.md) · [→ Parallel Training](unit-05.md)
```

New:
```markdown
[→ Apply It — SAC vs PPO on JumperHard](unit-sac-applied.md) · [→ Parallel Training](unit-05.md)
```

- [ ] **Step 2.6: Edit `content/unit-sac.de.md` bottom breadcrumb (last line).**

Old:
```markdown
[→ PPO von Grund auf (CleanRL)](unit-cleanrl.md) · [→ Paralleles Training](unit-05.md)
```

New:
```markdown
[→ Anwenden — SAC vs PPO auf JumperHard](unit-sac-applied.md) · [→ Paralleles Training](unit-05.md)
```

- [ ] **Step 2.7: Edit `content/unit-cleanrl.md` top breadcrumb (line 3).**

Old:
```markdown
[← SAC](unit-sac.md) · [Course home](index.md)
```

New:
```markdown
[← Apply It — SAC vs PPO on JumperHard](unit-sac-applied.md) · [Course home](index.md)
```

- [ ] **Step 2.8: Edit `content/unit-cleanrl.md` Prerequisites — add the interlude as the first bullet.**

Read the file's Prerequisites block (just after the breadcrumb). Add one new bullet at the top of the block:

`- **[Apply It — SAC vs PPO on JumperHard](unit-sac-applied.md)** — you've seen the PPO vs SAC comparison in practice`

Place it above the existing PPO Deep Dive bullet. Keep all other bullets.

- [ ] **Step 2.9: Edit `content/unit-cleanrl.de.md` top breadcrumb (line 3).**

Old:
```markdown
[← SAC](unit-sac.md) · [Kursstartseite](index.md)
```

New:
```markdown
[← Anwenden — SAC vs PPO auf JumperHard](unit-sac-applied.md) · [Kursstartseite](index.md)
```

- [ ] **Step 2.10: Edit `content/unit-cleanrl.de.md` Voraussetzungen — add the interlude as the first bullet.**

Add one new bullet at the top of the `!!! note "Voraussetzungen"` block:

`- **[Anwenden — SAC vs PPO auf JumperHard](unit-sac-applied.md)** — du hast den PPO-vs-SAC-Vergleich in der Praxis gesehen`

Place it above the existing PPO Deep Dive bullet.

- [ ] **Step 2.11: Verify Phase 3 build.**

Run: `conda run -n mkdocs-env mkdocs build --strict 2>&1 | tail -20`

Expected: `Documentation built` for both `en` and `de`, exit 0, zero warnings. The de build will reference `unit-sac-applied` without a `.de.md` — mkdocs-static-i18n falls back to English at the page level, which is expected and correct (no warning is emitted for missing translation pages under suffix strategy).

- [ ] **Step 2.12: Verify breadcrumb consistency around the interlude.**

Run: `grep -n "sac-applied\|Apply It" content/unit-sac.md content/unit-sac.de.md content/unit-cleanrl.md content/unit-cleanrl.de.md`

Expected: each of the four files has exactly one reference to the new interlude (top or bottom breadcrumb or prereq), all pointing to `unit-sac-applied.md` (not `.de.md`).

Run: `grep -n "→ PPO From Scratch (CleanRL)\|→ PPO von Grund auf (CleanRL)" content/unit-sac.md content/unit-sac.de.md`

Expected: empty output (those old bottom breadcrumbs were replaced by the Apply-It link).

- [ ] **Step 2.13: Commit Phase 3 insert.**

```bash
git add mkdocs.yml content/index.md content/index.de.md \
        content/unit-sac-applied.md \
        content/unit-sac.md content/unit-sac.de.md \
        content/unit-cleanrl.md content/unit-cleanrl.de.md
git commit -m "docs: add Apply-It interlude between SAC and CleanRL (#39)

New ~150-line unit-sac-applied.md: learner swaps PPO for SAC on the
existing JumperHard project, compares ep_rew_mean curves in
TensorBoard, and meets a decision guide for when to reach for which.

Breaks up the heaviest theory stretch (unit-sac → unit-cleanrl, the
901-line CleanRL unit). German fallback: mkdocs-static-i18n serves
the English page under /de/ until a .de.md is added in a follow-up.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Push, open PR, merge after CI

**Files:** none modified — git/GitHub operations only.

- [ ] **Step 3.1: Push the branch.**

```bash
git push -u origin docs/phase-3-pacing
```

- [ ] **Step 3.2: Open the PR.**

```bash
gh pr create --title "docs: insert SAC-vs-PPO interlude + reorder reward-engineering (closes #39)" --body "$(cat <<'EOF'
## Summary

Closes #39 with two targeted changes (spec: `docs/superpowers/specs/2026-05-30-phase-3-pacing-design.md`):

1. **Phase 1 reorder** — `unit-reward-engineering` moves before `unit-02`. Theory now precedes the lander reward-shaping exercise in unit-02 §5.
2. **Phase 3 interlude** — new `unit-sac-applied.md` (~150 lines) between SAC and CleanRL. Learner swaps PPO→SAC on the existing JumperHard project, overlays TensorBoard curves, and walks away with a PPO-vs-SAC decision guide.

## Why these specific changes

The issue's "5 consecutive non-Godot units after unit-04" framing was stale — current order already has unit-04 mid-Phase 3. The real pain points were the 901-line CleanRL unit immediately after SAC (heaviest theory load) and the reward-engineering theory arriving after unit-02 needs it.

PG → Actor-Critic → PPO Deep is preserved as a deliberate theory buildup. unit-04 is not split. No new Godot scene work.

## Test plan

- [x] \`mkdocs build --strict\` passes (en + de, zero warnings)
- [x] Breadcrumbs grep clean (no orphan references to old ordering)
- [ ] Spot-check \`mkdocs serve\` — Phase 1 + Phase 3 in en and de

## i18n

No \`unit-sac-applied.de.md\` in this PR. mkdocs-static-i18n falls back to English at the page level (per the i18n fallback memory). German translation tracked as follow-up.

## Caveat

The interlude's §4 "What you'll see" describes expected SAC-vs-PPO behaviour on JumperHard, written as expected rather than measured. Replace with empirical numbers later if desired.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3.3: Wait for CI to finish.**

Run: `gh pr checks --watch` (polls until CI completes)

Expected: `build` check finishes with `SUCCESS`. If it fails, read the run log via `gh run view <run-id> --log-failed` and fix.

- [ ] **Step 3.4: Verify mergeable state.**

Run: `gh pr view --json mergeable,mergeStateStatus,statusCheckRollup --jq '{mergeable, mergeStateStatus, checks: [.statusCheckRollup[] | {name, conclusion}]}'`

Expected:
```json
{"mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN", "checks": [{"name": "build", "conclusion": "SUCCESS"}]}
```

- [ ] **Step 3.5: Squash-merge and delete branch.**

Run: `gh pr merge --squash --delete-branch`

Expected: PR state → MERGED. Local branch deleted, remote branch deleted. Local main fast-forwards on next `git fetch`.

- [ ] **Step 3.6: Verify issue #39 auto-closed.**

Run: `gh issue view 39 --json state,closedAt --jq '{state, closedAt}'`

Expected: `{"state": "CLOSED", "closedAt": "<recent timestamp>"}`.

If `state` is still `OPEN`, manually close with `gh issue close 39 --reason completed --comment "Resolved by PR #<number>."`.

- [ ] **Step 3.7: Switch local back to main + pull.**

```bash
git checkout main
git pull --ff-only
```

Expected: `main` advances to include the squashed merge commit.

---

## Self-review

**Spec coverage:**
- Phase 1 reorder (spec §"Phase 1 reorder") → Task 1 (steps 1.1–1.13). ✓
- Phase 3 interlude insert (spec §"Phase 3 interlude insert") → Task 2 (steps 2.1–2.10). ✓
- `unit-sac-applied.md` structure (spec §"`unit-sac-applied.md` structure") → step 2.4 contains the full file. ✓
- Verification plan (spec §"Verification plan") → steps 1.14–1.15, 2.11–2.12, 3.3–3.4. ✓
- Ship plan (spec §"Ship plan") → Task 3 in full. ✓
- Risk: empirical claims in §4 → file content in step 2.4 frames as "expected, not measured" with caveat. ✓
- Risk: German interlude stale → PR body explicitly notes follow-up. ✓
- `docs/curriculum.md` reference (spec §"Phase 1 reorder" final bullet) → curriculum.md table is the 0–10 numbered units only; reward-engineering and sac-applied don't appear there. No update needed. Documented here so the executing agent doesn't go looking.

**Placeholder scan:** no TBD/TODO/"similar to" patterns. Step 1.13 ("Read the file first to capture the exact existing bullet text") is a deliberate read-before-write instruction, not a placeholder.

**Type/name consistency:**
- `unit-sac-applied.md` referenced consistently across steps 2.1, 2.2, 2.3, 2.4 (file create), 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.12.
- Visible title "Apply It — SAC vs PPO on JumperHard" used consistently in EN; "Anwenden — SAC vs PPO auf JumperHard" used consistently in DE.
- Branch name `docs/phase-3-pacing` used in spec, plan header, and Task 3.
