# Unit 1 — Foundations of Reinforcement Learning

Learn the RL mental model and map it onto Godot — but don't wait until the end to *do* something. Skim the loop, tweak one BallChase reward, then read the deeper sections while training runs.

---

!!! success "What you'll be able to do after this unit"
    - Explain the RL loop — *state, action, reward, next state*
    - Define the core terms: agent, environment, observation, action space, reward, return, policy, episode
    - Tell the difference between policy-based and value-based methods, and say what "Deep" adds
    - Describe how Godot and Python cooperate to train an agent
    - Run a complete training session and read its progress

!!! note "Prerequisites"
    - **Unit 0 complete** — Conda, Godot .NET, and a successful BallChase run
    - Basic Python familiarity (running scripts, installing packages) — you will *not* write a training loop
    - Comfort using a terminal
    - **No** prior Godot, game-dev, or machine-learning experience needed

    **Time:** Fast path ~15 min skim + ~20 min reward tweak + training in background. Full read of Sections 1–5: add ~25 minutes. Nothing to build from scratch yet.

---

## ⚡ Fast path (recommended first hour) { #fast-path }

1. Read [Section 1](#1-what-is-reinforcement-learning) and [Section 2](#2-the-rl-process-loop) only (~15 min) — agent, environment, reward, loop.
2. Jump to [Section 6](#6-quick-win-ballchase-recap) — change one reward in BallChase, start training with `--viz`.
3. While it trains, read Sections 3–5 and the [glossary](#glossary).

*Prefer linear reading? Follow Sections 1–6 in order — Section 6 still includes the reward tweak.*

---

## 1 · What is Reinforcement Learning?

**Reinforcement Learning (RL)** is a way of teaching software to make good decisions: an **agent** learns *how to behave* in an **environment** by *performing actions* and *seeing the results*. There is no answer key — the agent learns purely by trial and error, guided by a single number called the **reward**.

Think of training a dog. You can't explain the rules; you let it try things and give it a treat when it does well. Over many attempts it learns the behaviour that earns the most treats. RL is that idea, made precise enough for a computer.

!!! info "How RL differs from normal machine learning"
    Supervised learning needs labelled examples ("this image is a cat"). RL has no labels — only the reward signal. The agent must *discover* good behaviour itself, and its own actions decide what data it sees next.

In this course, the agent is a **lunar lander**. Its environment is a 2D world with gravity and a landing pad. By the end of Unit 2 it will have taught itself to fire its thrusters and touch down gently — without anyone scripting how.

---

## 2 · The RL process loop

**The loop: state → action → reward → next state**

RL always runs as the same loop, repeated thousands of times. At each time step:

- The agent receives a **state Sₜ** from the environment (e.g. the lander's position and velocity).
- Based on that state, the agent takes an **action Aₜ** (e.g. fire the main engine).
- The environment moves to a **new state Sₜ₊₁**.
- The environment returns a **reward Rₜ₊₁** — a number saying how good that was.

<svg viewBox="0 0 660 200" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" style="max-width:600px;display:block;margin:1.5rem auto">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 1 L10 5 L0 9 z" fill="#8892b0"/>
    </marker>
  </defs>
  <rect x="60" y="70" width="200" height="64" rx="12" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="160" y="96" text-anchor="middle" fill="#e2e8f0" font-size="15" font-weight="700">AGENT</text>
  <text x="160" y="116" text-anchor="middle" fill="#8892b0" font-size="11">the lander's brain</text>
  <rect x="400" y="70" width="200" height="64" rx="12" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="500" y="96" text-anchor="middle" fill="#e2e8f0" font-size="15" font-weight="700">ENVIRONMENT</text>
  <text x="500" y="116" text-anchor="middle" fill="#8892b0" font-size="11">the Godot game world</text>
  <path d="M260 84 C320 84, 340 84, 400 84" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#ar)"/>
  <text x="330" y="74" text-anchor="middle" fill="#6c8ef7" font-size="11" font-weight="700">action Aₜ</text>
  <path d="M400 120 C340 120, 320 120, 260 120" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#ar)"/>
  <text x="330" y="148" text-anchor="middle" fill="#4ecca3" font-size="11" font-weight="700">state Sₜ₊₁ + reward Rₜ₊₁</text>
</svg>

This loop outputs a stream of *state, action, reward, next state*. Repeat, and the agent gradually shifts toward actions that earn more reward.

**The goal: maximise the expected return**

The agent's goal is not to grab one big reward — it is to maximise the **cumulative reward** over time, called the **expected return**.

!!! info "The reward hypothesis"
    RL rests on one bold idea: *every* goal can be expressed as maximising expected cumulative reward. "Land softly on the pad" becomes "collect the most reward" once you design the reward function — which is exactly what you'll do in Unit 2.

!!! warning "Training stalled?"
    Check in order: (1) reward sign and scale — is "good" actually positive? (2) sparse rewards — does the agent get any signal before the goal? (3) observation bugs — are sensors updating after resets? (4) TensorBoard flat but Godot looks fine — you may need longer training or a viz checkpoint.

**Markov property & MDPs**

In papers this loop is called a **Markov Decision Process (MDP)**. The one thing to remember today: the **Markov property** means the agent needs *only the current state* to choose its action — not the full history. So the state must contain everything that matters *right now*.

---

## 3 · The building blocks

**Observations & states**

Both are the information the agent gets from the environment:

- **State** — a *complete* description of the world, nothing hidden (e.g. a chess board: you see everything).
- **Observation** — a *partial* description (e.g. Super Mario: you only see the part of the level near the player).

The course uses "state" loosely for both. Our lander reports an *observation*: 8 numbers (position, velocity, angle, leg contact). The set of all possible observations is the **observation/state space**.

**Action space — discrete vs continuous**

| Type | Meaning | Example |
|------|---------|---------|
| **Discrete** | A finite list of actions | Lunar lander: *do nothing / left thruster / main engine / right thruster* — 4 actions |
| **Continuous** | Infinitely many actions | A self-driving car: steer 20.0°, 20.1°, 20.15°, … |

Our lander uses a **discrete** action space of size 4. Whether a space is discrete or continuous influences which RL algorithm you pick later.

**Rewards, return & discounting**

The reward is the *only* feedback the agent gets. The **return** is the sum of all future rewards.

But future rewards are uncertain, so we **discount** them: multiply by a **discount rate γ (gamma)**, between 0 and 1 — usually **0.95–0.99**.

- γ close to 1 → the agent cares about the *long term*.
- γ smaller → the agent cares about *immediate* reward.

!!! tip "The classic intuition: mouse, cheese, cat"
    A mouse wants cheese. Cheese near the cat is worth more — but it's risky and far off, so its reward is *discounted*. Nearby cheese is a surer bet. Discounting captures "a reward I might never reach is worth less."

**Episodic vs continuing tasks**

- **Episodic** — there's a clear start and end (an **episode**). Our lander: an episode ends when it lands, crashes, or times out, then resets.
- **Continuing** — no end; the task runs forever (e.g. a stock-trading agent).

Everything in this course is **episodic**.

---

## 4 · How an agent learns

**The policy π — the agent's brain**

The **policy π** is the function that tells the agent what action to take in a given state. Training has one job: find the **optimal policy π\*** — the one that earns the most expected return.

A policy can be **deterministic** (a state always gives the same action) or **stochastic** (it outputs a probability distribution over actions). The lander you'll train uses a stochastic policy.

**Two ways to find the optimal policy**

| Approach | What it learns | How it acts |
|----------|---------------|-------------|
| **Policy-based** | The policy directly — a state → action mapping | Ask the policy for an action |
| **Value-based** | A value function — how good each state is | Move toward the highest-value state |

Both aim at the same optimal policy π\*. Unit 2 uses **PPO**, a policy-based method. Later units explore value-based methods like DQN.

**Exploration vs exploitation**

An agent that only ever does what worked before (**exploitation**) may never discover something better. An agent that only tries random things (**exploration**) never cashes in what it learned. Good learning balances the two — explore early, exploit more as you improve.

**What "Deep" means — and one word on PPO**

The policy has to be *some* function. In **Deep** RL, that function is a **neural network**: it takes the observation as input and outputs an action. "Deep" simply means a neural network is doing the learning.

!!! info "PPO in one sentence"
    **Proximal Policy Optimization (PPO)** is a popular policy-based algorithm that improves the policy in small, safe steps so training stays stable. You won't implement it — the `stable-baselines3` library provides it, and `godot-rl` wires it up.

---

## 5 · How Godot RL Agents fits together

Godot RL Agents uses two programs that run side by side and talk over a fast local network socket. The **Godot engine** is the environment — it renders the world, runs the physics, and reports observations and rewards. **Python** is the brain — it runs PPO and decides the actions.

<svg viewBox="0 0 660 320" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" style="max-width:600px;display:block;margin:1.5rem auto">
  <defs>
    <marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 1 L10 5 L0 9 z" fill="#8892b0"/>
    </marker>
    <marker id="ar3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 1 L10 5 L0 9 z" fill="#f7c86c"/>
    </marker>
  </defs>
  <rect x="30" y="50" width="260" height="200" rx="14" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="160" y="78" text-anchor="middle" fill="#e2e8f0" font-size="14" font-weight="700">Godot Engine — Environment</text>
  <rect x="52" y="100" width="216" height="52" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="160" y="121" text-anchor="middle" fill="#e2e8f0" font-size="12">Lander scene + AIController2D</text>
  <text x="160" y="138" text-anchor="middle" fill="#8892b0" font-size="10">observations · actions · reward</text>
  <rect x="52" y="168" width="216" height="52" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="160" y="189" text-anchor="middle" fill="#e2e8f0" font-size="12">Sync node</text>
  <text x="160" y="206" text-anchor="middle" fill="#8892b0" font-size="10" font-family="monospace">godot_rl_agents plugin</text>
  <rect x="370" y="50" width="260" height="200" rx="14" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="500" y="78" text-anchor="middle" fill="#e2e8f0" font-size="14" font-weight="700">Python — Agent's Brain</text>
  <rect x="392" y="100" width="216" height="52" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="500" y="121" text-anchor="middle" fill="#e2e8f0" font-size="12">godot-rl wrapper</text>
  <text x="500" y="138" text-anchor="middle" fill="#8892b0" font-size="10" font-family="monospace">gdrl</text>
  <rect x="392" y="168" width="216" height="52" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="500" y="189" text-anchor="middle" fill="#e2e8f0" font-size="12">Stable-Baselines3</text>
  <text x="500" y="206" text-anchor="middle" fill="#8892b0" font-size="10">PPO · neural-net policy</text>
  <path d="M290 118 L370 118" fill="none" stroke="#8892b0" stroke-width="1.8" marker-end="url(#ar2)"/>
  <text x="330" y="108" text-anchor="middle" fill="#4ecca3" font-size="10" font-weight="700">obs + reward</text>
  <path d="M370 196 L290 196" fill="none" stroke="#8892b0" stroke-width="1.8" marker-end="url(#ar2)"/>
  <text x="330" y="212" text-anchor="middle" fill="#6c8ef7" font-size="10" font-weight="700">actions</text>
  <text x="330" y="160" text-anchor="middle" fill="#8892b0" font-size="9">socket · port 11008</text>
  <path d="M500 250 C500 292, 160 292, 160 252" fill="none" stroke="#f7c86c" stroke-width="1.6" stroke-dasharray="5,4" marker-end="url(#ar3)"/>
  <text x="330" y="288" text-anchor="middle" fill="#f7c86c" font-size="10" font-weight="700">after training: ONNX model runs inside Godot — no Python</text>
</svg>

**Every concept has a home in Godot**

| RL concept | Where it lives in Godot RL |
|------------|---------------------------|
| Environment | Your Godot scene (the lander, ground, landing pad) |
| Observation | `get_obs()` in the AIController script |
| Action space | `get_action_space()` in the AIController script |
| Reward | `reward` updated in your game logic |
| Episode end | The `done` / `needs_reset` flags |
| Agent / policy | The PPO neural network, in Python during training |
| Communication loop | The **Sync** node ↔ Python socket |

This is exactly what you'll create, piece by piece, in Unit 2. After training, the policy is exported to an **ONNX** file and runs directly inside Godot — no Python needed to play the finished game.

---

## 6 · Quick win: BallChase recap

!!! tip "Your first ownership moment"
    In Unit 0 you ran BallChase. Here you change the reward signal and see behavior shift — that connects vocabulary to code.

**Tweak one reward (do this before a long re-read)**

1. Clone or open [BallChase](https://github.com/edbeeching/godot_rl_agents_examples/tree/main/examples/BallChase) in Godot .NET.
2. Find the script that sets `reward` on the agent (often on the player or `AIController`).
3. Change one term — e.g. double the reward for getting closer to the ball, or add a small penalty per step.
4. Note what you expect: faster chasing, more wandering, etc.

**Re-run BallChase** *(skip this if Unit 0 already succeeded)*

```bash
conda activate godot_env
python examples/stable_baselines3_example.py \
  --env_path=examples/godot_rl_BallChase/bin/BallChase.x86_64 \
  --experiment_name=unit1-recap --timesteps=100000 --viz
```

**Watch it learn (three views)**

- **Godot** — with `--viz`, confirm movement matches your reward change.
- **TensorBoard** — `ep_rew_mean` may start lower after a harsh reward; it should trend up as the policy adapts.
- **Code** — you can point to the line you edited when explaining the MDP loop.

!!! success "Checkpoint — you're ready for Unit 2"
    You changed a reward, retrained, and saw at least two of the three views update. In **Unit 2** you'll warm up on **SimpleReachGoal**, then build Lunar Lander from scratch.

---

## Glossary { #glossary }

| Term | Meaning |
|------|---------|
| **Agent** | The learner / decision-maker — here, the lander. |
| **Environment** | The world the agent acts in — here, the Godot scene. |
| **State / observation** | The information the agent receives. A state is complete; an observation is partial. |
| **Action** | A choice the agent makes that changes the environment. |
| **Action space** | The set of all possible actions — discrete (finite) or continuous (infinite). |
| **Reward** | The single number scoring how good an action was — the agent's only feedback. |
| **Return** | The (discounted) sum of all future rewards — what the agent maximises. |
| **Discount rate γ** | 0–1 factor making future rewards count less than immediate ones. |
| **Policy π** | The agent's brain — the function mapping a state to an action. |
| **Episode** | One run from start to a terminal state (land, crash, or timeout). |
| **MDP** | Markov Decision Process — the formal name for the RL loop. |
| **Policy-based / value-based** | The two families of methods for finding the optimal policy. |
| **PPO** | Proximal Policy Optimization — the stable policy-based algorithm used in Unit 2. |
| **Deep RL** | RL where the policy is a neural network. |
| **Rollout** | A batch of experience collected before each PPO update. |
| **ONNX** | A portable model format — lets the trained policy run inside Godot without Python. |

---

## What's next

You have the vocabulary, a reward tweak under your belt, and a trained run to reference. In **Unit 2**, start with **SimpleReachGoal** (run + tweak), then build Lunar Lander from scratch.

!!! info "Self-check before you move on"
    Can you answer these in your own words? (1) What are the four parts of the RL loop? (2) What's the difference between an observation and a state? (3) What does the discount rate γ control? (4) What is a policy? If yes — you're ready.

[→ Unit 2: Build Lunar Lander in Godot](unit-02.md)
