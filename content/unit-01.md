# Unit 1 — Foundations of Reinforcement Learning

Learn the RL mental model and map it onto Godot — but don't wait until the end to *do* something. Skim the loop, tweak one BallChase reward, then read the deeper sections while training runs.

---

!!! success "What you'll be able to do after this unit"
    - Explain the RL loop — *state, action, reward, next state*
    - Define the core terms: agent, environment, observation, action space, reward, return, policy, episode
    - Tell the difference between policy-based and value-based methods, and say what "Deep" adds
    - Distinguish Monte Carlo from Temporal Difference learning, and know which one PPO uses
    - Explain the exploration–exploitation trade-off and name the mechanisms PPO and DQN use
    - Place Q-Learning, DQN, REINFORCE, PPO, and MuZero in a single taxonomy table
    - Describe how Godot and Python cooperate to train an agent
    - Run a complete training session and read its progress

!!! note "Prerequisites"
    - **Unit 0 complete** — Conda, Godot .NET, and a successful BallChase run
    - Basic Python familiarity (running scripts, installing packages) — you will *not* write a training loop
    - Comfort using a terminal
    - **No** prior Godot, game-dev, or machine-learning experience needed

    **Time:** Fast path ~15 min skim + ~20 min reward tweak + training in background. Full read of Sections 1–6: add ~40 minutes. Nothing to build from scratch yet.

---

## ⚡ Fast path (recommended first hour) { #fast-path }

1. Read [Section 1](#1-what-is-reinforcement-learning) and [Section 2](#2-the-rl-process-loop) only (~15 min) — agent, environment, reward, loop.
2. Jump to [Section 7](#7-quick-win-ballchase-recap) — change one reward in BallChase, start training with `--viz`.
3. While it trains, read Sections 3–6 and the [glossary](#glossary).

*Prefer linear reading? Follow Sections 1–7 in order — Section 7 still includes the reward tweak.*

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

<div class="diagram-scroll">

<svg class="course-diagram" viewBox="0 0 660 200" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="RL loop: agent and environment exchange actions, states, and rewards">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 1 L10 5 L0 9 z" fill="#8892b0"/>
    </marker>
  </defs>
  <rect x="60" y="70" width="200" height="64" rx="12" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="160" y="96" text-anchor="middle" fill="#e2e8f0" font-size="18" font-weight="700">AGENT</text>
  <text x="160" y="116" text-anchor="middle" fill="#8892b0" font-size="14">the lander's brain</text>
  <rect x="400" y="70" width="200" height="64" rx="12" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="500" y="96" text-anchor="middle" fill="#e2e8f0" font-size="18" font-weight="700">ENVIRONMENT</text>
  <text x="500" y="116" text-anchor="middle" fill="#8892b0" font-size="14">the Godot game world</text>
  <path d="M260 84 C320 84, 340 84, 400 84" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#ar)"/>
  <text x="330" y="74" text-anchor="middle" fill="#6c8ef7" font-size="14" font-weight="700">action Aₜ</text>
  <path d="M400 120 C340 120, 320 120, 260 120" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#ar)"/>
  <text x="330" y="148" text-anchor="middle" fill="#4ecca3" font-size="14" font-weight="700">state Sₜ₊₁ + reward Rₜ₊₁</text>
</svg>

</div>

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

## 3 · Monte Carlo vs Temporal Difference

Once the agent collects experience, it has to *learn from it* — that is, update its estimate of how good each state is. There are two fundamentally different strategies for doing this.

### Monte Carlo (MC)

The idea: **wait until the episode is finished**, then use the real return to update.

At the end of an episode, you know the actual total reward collected from step *t* onward:

```
Gₜ = rₜ + γ rₜ₊₁ + γ² rₜ₊₂ + … + γᵀ rᵀ
```

You then push your value estimate V(sₜ) a bit toward that real return Gₜ.

**Pros:**
- **Unbiased** — you used the real return, no approximation.
- Easy to understand and implement.

**Cons:**
- Must wait for the **complete episode** before any update — slow.
- **High variance** — a single lucky or unlucky episode can swing your estimates wildly.
- Can't work in **continuing tasks** (no episode end).

### Temporal Difference (TD)

The idea: **update after every single step** using a *bootstrapped* estimate — that is, use your *current* value estimate for the next state instead of waiting for the real return.

The **TD target** is:

```
rₜ + γ V(sₜ₊₁)
```

The **TD error** δₜ measures how wrong your current estimate was:

```
δₜ = rₜ + γ V(sₜ₊₁) − V(sₜ)
```

You then nudge V(sₜ) by a small step in the direction of δₜ.

**Pros:**
- Works in **continuing tasks** — no episode end needed.
- **Online learning** — updates flow in as the agent acts.
- **Lower variance** than MC because you don't wait for a full noisy trajectory.

**Cons:**
- **Biased** — you're bootstrapping on an imperfect V estimate. Early in training that estimate is wrong, so the target is wrong too.

### The intuition

!!! info "MC vs TD in plain English"
    **Monte Carlo** is like waiting until the chess game is over and then updating your mental model of every position you played through.

    **Temporal Difference** is like updating your assessment of your position *after every single move*, based on how good the position looks *right now*.

    Neither is always better — the right choice depends on task structure, episode length, and how much variance you can tolerate.

### Which one does this course use?

| Algorithm | Family | Used in |
|-----------|--------|---------|
| PPO (RecurrentPPO) | Multi-step TD | Units 2, 5 |
| Q-Learning | Single-step TD | Unit 3 (theory) |
| DQN | Single-step TD | Unit 3 |
| REINFORCE | Monte Carlo | Background reading |

PPO collects a fixed-length **rollout** (a window of steps, not a full episode) and then updates — that makes it a multi-step TD method. You'll see `n_steps` in the PPO config; that's the rollout length.

!!! tip "Reading TensorBoard"
    The `ep_rew_mean` curve is the averaged episodic return — a Monte Carlo quantity. But the *learning signal* inside PPO is TD-based. When `ep_rew_mean` climbs slowly at first and then accelerates, you're watching TD bootstrapping gradually improve as the value estimate gets better. We explore this in depth in the [Q-Learning unit](unit-03.md).

---

## 4 · The building blocks

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

## 5 · How an agent learns

**The policy π — the agent's brain**

The **policy π** is the function that tells the agent what action to take in a given state. Training has one job: find the **optimal policy π\*** — the one that earns the most expected return.

A policy can be **deterministic** (a state always gives the same action) or **stochastic** (it outputs a probability distribution over actions). The lander you'll train uses a stochastic policy.

---

### Exploration vs Exploitation

Before the agent can exploit a good policy, it must first *discover* what is good — and that requires exploration. This tension is one of the most fundamental challenges in RL.

**The dilemma in one sentence:** exploit known-good actions and collect reward *now*, or explore unknown actions that might be even better.

#### ε-greedy exploration

The classical solution, used in **DQN** (Unit 3):

- With probability **ε**, take a *random* action (explore).
- With probability **1 − ε**, take the *best known* action (exploit).

| ε value | Behaviour |
|---------|-----------|
| 1.0 | Pure random exploration — the agent ignores everything it has learned |
| 0.5 | Half random, half greedy |
| 0.05 | Mostly exploiting, with a small chance of trying something new |
| 0.0 | Pure exploitation — greedy, never explores |

In practice, ε is **annealed** (decayed) during training: start at 1.0, finish at 0.05. This lets the agent explore the state space freely early on, then commit to the policy it has learned.

#### Entropy-based exploration (PPO)

PPO does *not* use ε-greedy. Instead, it works with **stochastic policies** — the network outputs a probability distribution over actions, not a single action.

The **entropy** of this distribution measures how spread out it is:

- **High entropy** = the distribution is flat = the agent explores many actions equally.
- **Low entropy** = the distribution is peaked = the agent is confident and commits to one action.

PPO adds an **entropy bonus** to the loss function. This penalises the policy for collapsing too early to a narrow distribution, keeping exploration alive throughout training.

```
PPO loss = policy gradient − c_entropy × entropy
```

The coefficient `c_entropy` (often called `ent_coef` in stable-baselines3) controls how hard you push for exploration. You'll tune this in later units.

#### Curiosity-driven exploration (bonus concept)

A third family of methods gives the agent an **intrinsic reward** for visiting novel states — states it hasn't seen before or can't predict well. This is especially useful when the extrinsic reward is very sparse (e.g. a puzzle game where reward only arrives at the very end). We don't use curiosity in this course, but it's worth knowing the idea exists.

!!! info "Which method is used where"
    - **DQN** (Unit 3) — ε-greedy with linear ε decay
    - **PPO** (Units 2, 5) — entropy bonus via `ent_coef`
    - **Curiosity / RND** — advanced, not covered in this course

---

**Two ways to find the optimal policy**

| Approach | What it learns | How it acts |
|----------|---------------|-------------|
| **Policy-based** | The policy directly — a state → action mapping | Ask the policy for an action |
| **Value-based** | A value function — how good each state is | Move toward the highest-value state |

Both aim at the same optimal policy π\*. Unit 2 uses **PPO**, a policy-based method. Later units explore value-based methods like DQN.

---

### Value-Based vs Policy-Based vs Actor-Critic: the full taxonomy

In practice, RL algorithms fall into four families. Knowing the map helps you read papers, choose algorithms, and understand TensorBoard metrics.

| Family | What it learns | Representative algorithms | When to use |
|--------|---------------|--------------------------|-------------|
| **Value-based** | Q(s,a) or V(s) — how good each state/action is | Q-Learning, DQN, Double DQN | Discrete actions; sample-efficient; good for simple environments |
| **Policy-based** | π_θ(a\|s) directly — the policy as a neural network | REINFORCE, TRPO | Continuous or stochastic action spaces; no value function needed |
| **Actor-Critic** | Both π and V simultaneously — the actor acts, the critic evaluates | A2C, PPO, SAC | Best of both families; lower variance than pure policy-based; the standard today |
| **Model-based** | A dynamics model p(s'\|s,a) — predict what happens next | Dyna, World Models, MuZero | Highly sample-efficient; useful when simulation is expensive |

**This course** focuses on **Actor-Critic** (PPO) because it is the industry standard for game AI. Unit 3 also teaches **DQN** (value-based) so you can feel the contrast. Model-based methods are covered briefly in the advanced section.

!!! info "Why Actor-Critic?"
    A pure policy-based method (REINFORCE) has high variance — each update is based on a single noisy trajectory. A pure value-based method (DQN) doesn't work easily with continuous actions. Actor-Critic gets low variance (from the critic's value estimate) *and* handles any action space (from the actor's policy). PPO is the most popular Actor-Critic algorithm in games because it is also stable and sample-efficient.

---

**What "Deep" means in Deep RL**

Traditional RL maintained a **lookup table**: one row per state, one column per action, storing Q-values or V-values. This works fine for small problems — a chess endgame with a few thousand positions, or a simple grid world.

It fails completely once the state space is large. A Godot game with 8 continuous sensor values has *infinite* possible states. You can't have a row for each one.

**Deep RL** replaces the table with a **neural network**:

- **Input:** the raw observation vector (8 numbers for the lander, pixel arrays for vision-based agents, raycast distances for a robot).
- **Output:** action probabilities (for an actor/policy network) *or* a value estimate (for a critic/value network).

The network *generalises* — it learns to give similar outputs for similar observations, which is exactly what a table cannot do.

**Why "deep"?** Because the network has multiple hidden layers (it's "deep" in the neural-network sense). A shallow one-layer network can't represent the complex non-linear functions that most game policies require.

!!! info "The trade-off you buy into with Deep RL"
    | Tabular RL | Deep RL |
    |------------|---------|
    | Exact, provably converges | Approximate, may diverge |
    | Only tiny state spaces | Any state space (pixels, sensors) |
    | No hyperparameters | Learning rate, architecture, ent_coef, … |
    | Instant updates | Needs thousands of gradient steps |

    For any game environment beyond toy problems, Deep RL is the only practical option.

!!! tip "What this means for debugging"
    When training stalls, it's often not the RL algorithm that's wrong — it's the neural network failing to learn a useful representation. Check: (1) observation scale (inputs should be roughly −1 to 1), (2) reward scale (very large rewards cause gradient explosion), (3) network size (too small = underfitting, too large = slow and unstable).

**One word on PPO**

!!! info "PPO in one sentence"
    **Proximal Policy Optimization (PPO)** is a popular Actor-Critic algorithm that improves the policy in small, safe steps so training stays stable. You won't implement it — the `stable-baselines3` library provides it, and `godot-rl` wires it up.

---

## 6 · How Godot RL Agents fits together

Godot RL Agents uses two programs that run side by side and talk over a fast local network socket. The **Godot engine** is the environment — it renders the world, runs the physics, and reports observations and rewards. **Python** is the brain — it runs PPO and decides the actions.

<div class="diagram-scroll">

<svg class="course-diagram" viewBox="0 0 720 360" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="Godot environment and Python training process connected by socket">
  <defs>
    <marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 1 L10 5 L0 9 z" fill="#8892b0"/>
    </marker>
    <marker id="ar3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 1 L10 5 L0 9 z" fill="#f7c86c"/>
    </marker>
  </defs>
  <!-- Godot -->
  <rect x="20" y="16" width="290" height="238" rx="14" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="165" y="44" text-anchor="middle" fill="#e2e8f0" font-size="16" font-weight="700">Godot Engine</text>
  <text x="165" y="62" text-anchor="middle" fill="#8892b0" font-size="13">Environment</text>
  <rect x="40" y="76" width="250" height="66" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="165" y="102" text-anchor="middle" fill="#e2e8f0" font-size="14">Lander scene + AIController2D</text>
  <text x="165" y="124" text-anchor="middle" fill="#8892b0" font-size="12">observations · actions · reward</text>
  <rect x="40" y="154" width="250" height="66" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="165" y="180" text-anchor="middle" fill="#e2e8f0" font-size="14">Sync node</text>
  <text x="165" y="202" text-anchor="middle" fill="#8892b0" font-size="12" font-family="monospace">godot_rl_agents</text>
  <!-- Python -->
  <rect x="410" y="16" width="290" height="238" rx="14" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="555" y="44" text-anchor="middle" fill="#e2e8f0" font-size="16" font-weight="700">Python</text>
  <text x="555" y="62" text-anchor="middle" fill="#8892b0" font-size="13">Agent's brain</text>
  <rect x="430" y="76" width="250" height="66" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="555" y="102" text-anchor="middle" fill="#e2e8f0" font-size="14">godot-rl wrapper</text>
  <text x="555" y="124" text-anchor="middle" fill="#8892b0" font-size="12" font-family="monospace">gdrl</text>
  <rect x="430" y="154" width="250" height="66" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="555" y="180" text-anchor="middle" fill="#e2e8f0" font-size="14">Stable-Baselines3</text>
  <text x="555" y="202" text-anchor="middle" fill="#8892b0" font-size="12">PPO · neural-net policy</text>
  <!-- Socket traffic (wider gutter) -->
  <path d="M310 112 L410 112" fill="none" stroke="#8892b0" stroke-width="1.8" marker-end="url(#ar2)"/>
  <text x="360" y="98" text-anchor="middle" fill="#4ecca3" font-size="13" font-weight="700">obs + reward</text>
  <path d="M410 192 L310 192" fill="none" stroke="#8892b0" stroke-width="1.8" marker-end="url(#ar2)"/>
  <text x="360" y="208" text-anchor="middle" fill="#6c8ef7" font-size="13" font-weight="700">actions</text>
  <text x="360" y="144" text-anchor="middle" fill="#8892b0" font-size="12">socket</text>
  <text x="360" y="160" text-anchor="middle" fill="#8892b0" font-size="12">port 11008</text>
  <!-- ONNX export (below panels) -->
  <path d="M555 262 C555 288, 165 288, 165 262" fill="none" stroke="#f7c86c" stroke-width="1.6" stroke-dasharray="5,4" marker-end="url(#ar3)"/>
  <text x="360" y="318" text-anchor="middle" fill="#f7c86c" font-size="13" font-weight="700">After training: ONNX model runs in Godot</text>
  <text x="360" y="338" text-anchor="middle" fill="#f7c86c" font-size="12">no Python process required</text>
</svg>

</div>

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

## 7 · Quick win: BallChase recap

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
| **Monte Carlo (MC)** | Learning from complete episodes; unbiased but high variance and slow. |
| **Temporal Difference (TD)** | Learning after every step using bootstrapped estimates; lower variance, works online. |
| **TD error δ** | The difference between the TD target and current value estimate: rₜ + γ V(sₜ₊₁) − V(sₜ). |
| **Bootstrapping** | Using your current estimate of future value as a target, instead of waiting for the real return. |
| **Exploration** | Trying actions you don't know well, to discover better strategies. |
| **Exploitation** | Using what you already know to collect reward. |
| **ε-greedy** | Exploration strategy: random action with probability ε, best-known action otherwise. |
| **Entropy bonus** | PPO's exploration mechanism: penalises overly confident policies to keep exploration alive. |
| **Policy-based** | Methods that learn the policy π directly (e.g. REINFORCE, PPO). |
| **Value-based** | Methods that learn Q(s,a) or V(s) and derive behaviour from it (e.g. DQN). |
| **Actor-Critic** | Methods that learn both π and V simultaneously (e.g. A2C, PPO). |
| **PPO** | Proximal Policy Optimization — the stable Actor-Critic algorithm used in Unit 2. |
| **Deep RL** | RL where the policy or value function is a neural network. |
| **Rollout** | A batch of experience collected before each PPO update. |
| **ONNX** | A portable model format — lets the trained policy run inside Godot without Python. |

---

## What's next

You have the vocabulary, a reward tweak under your belt, and a trained run to reference. In **Unit 2**, start with **SimpleReachGoal** (run + tweak), then build Lunar Lander from scratch.

!!! info "Self-check before you move on"
    Can you answer these in your own words?

    1. What are the four parts of the RL loop?
    2. What's the difference between an observation and a state?
    3. What does the discount rate γ control?
    4. What is a policy?
    5. What is the key difference between Monte Carlo and TD learning?
    6. How does PPO encourage exploration without ε-greedy?
    7. Name one algorithm from each of the four RL families (value-based, policy-based, Actor-Critic, model-based).

    If you can answer all seven — you're ready.

[→ Unit 2: Build Lunar Lander in Godot](unit-02.md)
