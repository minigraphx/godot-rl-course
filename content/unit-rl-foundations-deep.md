# RL Foundations Deep Dive

Previous planned: Neural Foundations 3 · [Course home](index.md)

!!! info "Time"
    Reading: ~45 min · Best read after you have a Foundations 3 point-robot or racer run to inspect

!!! success "What you'll be able to do after this unit"
    - Distinguish Monte Carlo from Temporal Difference learning
    - Explain bootstrapping and TD error from a trajectory
    - Compare ε-greedy, entropy, and curiosity-driven exploration
    - Place common algorithms in value-based, policy-based, Actor-Critic, and model-based families
    - Explain why on-policy and off-policy methods reuse data differently

!!! note "Prerequisites"
    - **RL Essentials complete** — observations, actions, rewards, episodes, return, and policy
    - **Neural Foundations 3 complete** — you have a point-robot trajectory or racer learning curve to reference

!!! info "Three ways to see your AI"
    Trajectory traces · learning curves · algorithm families behind the training run

This page gives names to the deeper mechanics you just observed. Keep one
Foundations 3 artifact open: either a point-robot trajectory with step rewards or
a racer learning curve with episode returns.

---

## 1 · Read your Foundations 3 run

Before theory, collect evidence from your run:

- one trajectory or replay;
- the reward at several steps;
- the episode return;
- the learning curve over many episodes;
- one failure case.

Ask two questions:

1. Did this episode improve because the policy made better local choices?
2. Did training improve because the algorithm learned from whole trajectories,
   step-by-step estimates, or both?

The rest of this page gives you the vocabulary to answer those questions.

---

## 2 · Monte Carlo versus Temporal Difference

**Monte Carlo** learning waits until an episode ends, then learns from the actual
return. In the Foundations 3 point robot, that means a state near the start only
gets its training target after the robot reaches the goal, crashes, or times out.

For rewards from step `t` onward:

```text
G_t = r_t + γr_t+1 + γ²r_t+2 + ...
```

Monte Carlo uses that completed `G_t` as the target.

**Temporal Difference** learning updates after a step by combining the immediate
reward with the current estimate of the next state:

```text
TD target = r_t + γV(s_t+1)
```

On a racer learning curve, TD-style updates help the value estimate improve
before every possible lap outcome has been seen.

| Method | Learns from | Strength | Weakness |
|--------|-------------|----------|----------|
| Monte Carlo | completed episodes | unbiased target | waits for episode end, high variance |
| Temporal Difference | each transition | online updates, lower variance | biased by current estimates |

---

## 3 · Bootstrapping and TD error

**Bootstrapping** means using your current estimate as part of the new target.
Instead of waiting for the true full return, TD asks: "reward now, plus what I
currently think the next state is worth."

The **TD error** measures the surprise:

```text
δ_t = r_t + γV(s_t+1) - V(s_t)
```

Use the point-robot trajectory to make this concrete:

- if a forward step moves closer to the goal, `r_t` may be positive;
- if the next observation points toward open space, `V(s_t+1)` may be higher;
- if the old value `V(s_t)` was too pessimistic, `δ_t` becomes positive.

A positive TD error says: "this state was better than expected." A negative TD
error says: "this state was worse than expected."

!!! tip "Reading a curve"
    When a racer learning curve climbs slowly and then accelerates, one reason
    can be improved bootstrapping: better value estimates make later TD targets
    less noisy.

---

## 4 · Exploration mechanisms

Exploration is visible in Foundations 3 whenever the point robot samples a poor
turn or the racer tries a steering action that does not match the current best
behavior. Different algorithms encourage that exploration in different ways.

### ε-greedy

**ε-greedy** is common in value-based methods such as DQN:

- with probability `ε`, take a random action;
- with probability `1 - ε`, take the best-known action.

Early training uses large `ε` so the agent explores freely. Later training
decays `ε` so the agent exploits more often. On a point-robot trajectory, this
would look like many random headings at first and fewer random turns later.

### Entropy

PPO uses a stochastic policy instead of ε-greedy. The policy outputs a
distribution over actions, and **entropy** measures how spread out that
distribution is.

- high entropy: many actions are still plausible;
- low entropy: the policy is confident.

An entropy bonus discourages the racer policy from becoming too certain too
early, which can prevent it from locking into a bad steering habit.

### Curiosity

Curiosity-driven methods add internal reward for novel states. This can help
when the real reward is sparse. In a point-robot maze, curiosity might reward
visiting new corridors even before the goal is reached.

Curiosity is useful to know, but this course focuses on reward design and PPO
before adding intrinsic rewards.

---

## 5 · Value, policy, and Actor-Critic methods

RL algorithms differ by what they learn. Use your Foundations 3 run as the
anchor: the policy chooses the robot or racer action, while a value estimate can
help judge whether the current observation is promising.

| Family | What it learns | Examples | How to recognize it |
|--------|----------------|----------|---------------------|
| Value-based | `Q(s, a)` or `V(s)` | Q-Learning, DQN | choose actions from learned values |
| Policy-based | `π(a|s)` directly | REINFORCE, TRPO | learn the action distribution itself |
| Actor-Critic | policy and value together | A2C, PPO, SAC | actor acts, critic evaluates |
| Model-based | transition or world model | Dyna, World Models, MuZero | predict what happens next |

**Actor-Critic** is the family you will see most often in this course. The
actor is the policy that drives the racer. The critic estimates how good the
current observation is, helping reduce the noise of policy updates.

!!! info "Why Actor-Critic?"
    Pure policy-gradient methods can have high variance because each trajectory
    is noisy. A critic gives the policy a better baseline, which is especially
    helpful when a racer has many mediocre starts before it completes a lap.

---

## 6 · On-policy versus off-policy learning

An **on-policy** method learns from data collected by the current policy. PPO is
on-policy: the racer rollout is useful because it came from the policy being
updated now.

An **off-policy** method can learn from data collected by older or different
policies. DQN is off-policy: it can store old transitions in a replay buffer and
learn from them later.

| Question | On-policy | Off-policy |
|----------|-----------|------------|
| Can it reuse old experience heavily? | limited | yes |
| Is the data close to the current behavior? | yes | not always |
| Common example | PPO | DQN |

For the point robot, on-policy learning means a batch of wandering trajectories
may be discarded after one update. Off-policy learning could keep those old
transitions and revisit them.

The trade-off is practical: on-policy methods are often stable and simple to
reason about, while off-policy methods can be more sample-efficient.

---

## 7 · Model-free versus model-based learning

A **model-free** method learns what to do without learning a separate simulator
of the world. PPO and DQN are model-free. They do not try to predict the next
racer observation before acting; they learn actions or values from experience.

A **model-based** method learns or uses a model of environment dynamics:

```text
current observation + action → predicted next observation and reward
```

In a point-robot task, a model-based agent might learn that turning right near a
wall predicts a collision. It can then plan with that prediction before taking
the action in the real environment.

Model-based learning can be sample-efficient, but it adds a new failure mode:
the learned model can be wrong. If the model predicts the racer will clear a
corner but the real physics disagrees, planning can amplify the mistake.

---

## 8 · Algorithm map for the rest of the course

| Algorithm | Family | Policy/data style | Where it connects |
|-----------|--------|-------------------|-------------------|
| REINFORCE | policy-based | on-policy Monte Carlo | Foundations 3 Research path |
| PPO | Actor-Critic | on-policy, clipped updates | Godot RL training path |
| Q-Learning | value-based | off-policy TD | Q-learning theory unit |
| DQN | deep value-based | off-policy TD with replay | deep Q-learning unit |
| SAC | Actor-Critic | off-policy entropy-regularized | advanced continuous-control comparison |
| MuZero | model-based | learned model plus planning | advanced background |

Return to your Foundations 3 curve when reading this table. The curve is not
"PPO magic"; it is the visible result of policy updates, value estimates,
exploration, and reward design interacting over many episodes.

---

## 9 · Taxonomy self-test

Answer from memory, then expand the section below:

1. Which method waits for the episode to finish: Monte Carlo or Temporal Difference?
2. What does the TD error compare?
3. Why does PPO use entropy instead of ε-greedy?
4. What makes PPO Actor-Critic?
5. Why can off-policy methods reuse more old data?
6. What does a model-based method learn that PPO does not?

??? success "Answers"
    1. **Monte Carlo** waits for the episode to finish and uses the actual return.
    2. TD error compares the current value estimate with `reward + discounted
       next-state value`.
    3. PPO already samples from a stochastic policy, so entropy keeps that
       distribution broad enough to explore.
    4. PPO learns both an actor policy and a critic value estimate.
    5. Off-policy methods can learn from data generated by older or different
       policies, often through replay buffers.
    6. A model-based method learns or uses predicted dynamics: next observations
       and rewards.

---

## 10 · Stretch goals

- Pick one Foundations 3 trajectory and label each step with reward, return, and
  whether the action looked exploratory.
- Plot two racer learning curves with different entropy settings and compare
  early exploration.
- Implement discounted returns for a short reward list and compare the values for
  `γ = 0.5`, `0.9`, and `0.99`.
- Build a tiny table that classifies REINFORCE, PPO, DQN, SAC, and MuZero by
  family, on/off-policy status, and model-free/model-based status.

---

## What's next

You now have the algorithm map behind the visual runs. Next, return to the main
course sequence and use that map when reward design, Q-learning, DQN, PPO
configuration, and native inference start to overlap.

Previous planned: Neural Foundations 3 · [Course home](index.md) · [→ Reward Engineering](unit-reward-engineering.md)
