# RL Essentials — From Network to Learning Agent

[← Neural Foundations 2](unit-neural-02.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~25 min · Quick reward tweak: ~20 min · Training can run in the background

!!! success "What you'll be able to do after this unit"
    - Explain what reinforcement learning adds to the network you just built
    - Name the loop pieces: observation, action, reward, next observation
    - Describe episodes, return, discounting, policy, and exploration
    - Point to where Godot and Python each sit during training
    - Change one reward and predict how the learning curve should respond

!!! note "Prerequisites"
    - **Unit 0 complete** — Conda, Godot, and a successful BallChase run
    - **Neural Foundations 1–2 complete** — you have seen inputs, weights, loss, gradients, and inference
    - Basic terminal comfort

!!! info "Three ways to see your AI"
    Godot editor (the agent moving live) · TensorBoard (`rollout/ep_rew_mean` changing) · Code (the reward line you edit)

You have already built a small network that turns numbers into decisions. RL adds
one missing piece: the network no longer learns from correct answers. It learns
from actions, consequences, and reward.

---

## 1 · What reinforcement learning adds

**Reinforcement Learning (RL)** teaches software to make good decisions by
letting an **agent** act inside an **environment** and scoring what happens with
a **reward**.

Supervised learning says: "this input should produce this target." RL says:
"try an action, observe what happened, and use the reward to make better
decisions next time."

That difference matters for games. A designer may not know the perfect action in
every position, but they can often describe what good behavior earns:

- get closer to the goal;
- avoid hazards;
- finish quickly;
- stay alive;
- collect useful objects.

The reward is not the final behavior. It is the training signal the policy uses
to discover behavior.

!!! info "The reward hypothesis"
    RL rests on one bold idea: goals can be expressed as maximizing expected
    cumulative reward. "Land softly" becomes numbers for safe speed, upright
    angle, leg contact, and not crashing.

---

## 2 · Observation → action → reward → next observation

At each training step, the same loop repeats:

1. Godot sends the current **observation** to Python.
2. The policy chooses an **action**.
3. Godot applies that action in the scene.
4. Godot returns a **reward** and the **next observation**.

<div class="diagram-scroll">

<svg class="course-diagram" viewBox="0 0 660 200" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="RL loop: agent and environment exchange actions, observations, and rewards">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 1 L10 5 L0 9 z" fill="#8892b0"/>
    </marker>
  </defs>
  <rect x="60" y="70" width="200" height="64" rx="12" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="160" y="96" text-anchor="middle" fill="#e2e8f0" font-size="18" font-weight="700">POLICY</text>
  <text x="160" y="116" text-anchor="middle" fill="#8892b0" font-size="14">the network</text>
  <rect x="400" y="70" width="200" height="64" rx="12" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="500" y="96" text-anchor="middle" fill="#e2e8f0" font-size="18" font-weight="700">GODOT</text>
  <text x="500" y="116" text-anchor="middle" fill="#8892b0" font-size="14">the environment</text>
  <path d="M260 84 C320 84, 340 84, 400 84" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#ar)"/>
  <text x="330" y="74" text-anchor="middle" fill="#6c8ef7" font-size="14" font-weight="700">action</text>
  <path d="M400 120 C340 120, 320 120, 260 120" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#ar)"/>
  <text x="330" y="148" text-anchor="middle" fill="#4ecca3" font-size="14" font-weight="700">observation + reward</text>
</svg>

</div>

**Observation versus state**

- **State** means the full truth of the world.
- **Observation** means the numbers the agent actually receives.

In most games the agent receives observations, not the full state. A lander may
know its velocity and angle but not every internal physics value. A racer may
know ray distances and heading error but not the whole track map.

**Action space**

| Type | Meaning | Example |
|------|---------|---------|
| Discrete | Choose from a fixed list | fire left thruster, main thruster, or nothing |
| Continuous | Choose values from a range | steering and throttle between -1 and 1 |

The action space decides what shape the policy output must have.

---

## 3 · Episodes, return, and discounting

An **episode** is one attempt from reset to a terminal condition. A lander
episode ends when it lands, crashes, or times out. A racer episode might end
after a lap, a collision, or an immobility timeout.

The **reward** is one step of feedback. The **return** is the sum of future
rewards the agent is trying to maximize.

Future rewards are usually **discounted** by a value called gamma (`γ`):

- `γ` close to 1 means the agent cares strongly about later outcomes;
- lower `γ` means the agent focuses more on immediate reward.

!!! tip "The mouse, cheese, and cat intuition"
    Cheese near the cat may be valuable, but it is risky and far away. Discounting
    captures the idea that a future reward you might never reach is worth less
    than a reward you can reliably get now.

Everything in this course starts as an episodic task. That makes learning curves
easier to read because each run has a clear beginning and end.

---

## 4 · The policy is the network you built

The **policy** is the agent's decision function. It maps an observation to an
action, or to a probability distribution over actions.

In Neural Foundations 2, your network learned:

```text
inputs → hidden activations → outputs
```

In RL, those outputs become actions:

```text
observation → policy network → action
```

Training changes the policy weights. Inference only runs the forward pass. That
same split will matter later when a policy is trained in Python and exported to
run inside Godot.

!!! info "Policy in one sentence"
    A policy is the trained behavior of the agent, stored as network weights.

---

## 5 · Exploration in one picture

Before an agent can use a good behavior, it must discover one. That is
**exploration**.

The trade-off is simple:

- **exploit** actions that already look good;
- **explore** actions that might reveal something better.

Early in training, random-looking movement is normal. The policy is collecting
experience. Later, movement should become more consistent as reward pushes the
network toward better actions.

!!! warning "Do not judge a policy from one early episode"
    A useful policy often looks foolish at first. Watch trends across many
    episodes, especially the average return curve, not one lucky or unlucky run.

You will study detailed exploration mechanisms in the deep dive after Foundations
3. For now, the practical question is: **does the agent keep trying enough actions
to find rewarded behavior?**

---

## 6 · Godot and Python during training

Godot RL Agents runs two programs side by side:

- **Godot** is the environment: physics, observations, actions, rewards, resets;
- **Python** is the trainer: it runs the RL algorithm and updates the policy.

<div class="diagram-scroll">

<svg class="course-diagram" viewBox="0 0 720 330" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="Godot environment and Python training process connected by socket">
  <defs>
    <marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 1 L10 5 L0 9 z" fill="#8892b0"/>
    </marker>
  </defs>
  <rect x="20" y="30" width="290" height="210" rx="14" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="165" y="58" text-anchor="middle" fill="#e2e8f0" font-size="16" font-weight="700">Godot</text>
  <text x="165" y="78" text-anchor="middle" fill="#8892b0" font-size="13">environment</text>
  <rect x="40" y="105" width="250" height="50" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="165" y="135" text-anchor="middle" fill="#e2e8f0" font-size="14">scene + AIController</text>
  <rect x="40" y="170" width="250" height="50" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="165" y="200" text-anchor="middle" fill="#e2e8f0" font-size="14">reward + reset logic</text>
  <rect x="410" y="30" width="290" height="210" rx="14" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="555" y="58" text-anchor="middle" fill="#e2e8f0" font-size="16" font-weight="700">Python</text>
  <text x="555" y="78" text-anchor="middle" fill="#8892b0" font-size="13">trainer</text>
  <rect x="430" y="105" width="250" height="50" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="555" y="135" text-anchor="middle" fill="#e2e8f0" font-size="14">godot-rl wrapper</text>
  <rect x="430" y="170" width="250" height="50" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="555" y="200" text-anchor="middle" fill="#e2e8f0" font-size="14">Stable-Baselines3 PPO</text>
  <path d="M310 120 L410 120" fill="none" stroke="#8892b0" stroke-width="1.8" marker-end="url(#ar2)"/>
  <text x="360" y="106" text-anchor="middle" fill="#4ecca3" font-size="13" font-weight="700">obs + reward</text>
  <path d="M410 190 L310 190" fill="none" stroke="#8892b0" stroke-width="1.8" marker-end="url(#ar2)"/>
  <text x="360" y="208" text-anchor="middle" fill="#6c8ef7" font-size="13" font-weight="700">actions</text>
  <text x="360" y="260" text-anchor="middle" fill="#8892b0" font-size="13">local socket, usually port 11008</text>
</svg>

</div>

| RL concept | Where it lives in Godot RL |
|------------|----------------------------|
| Environment | Your Godot scene |
| Observation | `get_obs()` in the AIController script |
| Action space | `get_action_space()` in the AIController script |
| Reward | Reward variables updated by your game logic |
| Episode end | `done`, `needs_reset`, or equivalent reset flags |
| Policy | The neural network trained in Python |

---

## 7 · Quick win — change one reward

!!! tip "Your first ownership moment"
    In Unit 0 you ran BallChase. Here you change the reward signal and watch the
    training behavior respond.

1. Clone or open [BallChase](https://github.com/edbeeching/godot_rl_agents_examples/tree/main/examples/BallChase) in Godot.
2. Find the script that updates `reward` for the agent.
3. Change one term, such as doubling the reward for getting closer to the ball
   or adding a small penalty per step.
4. Predict the result before training: faster chasing, more wandering, shorter
   episodes, or slower learning.

Run a short visual training session:

```bash
conda activate godot_env
python examples/stable_baselines3_example.py \
  --env_path=examples/godot_rl_BallChase/bin/BallChase.x86_64 \
  --experiment_name=unit1-reward-tweak --timesteps=100000 --viz
```

Watch three views:

- **Godot:** does movement match your reward change?
- **TensorBoard:** does `ep_rew_mean` trend upward after enough episodes?
- **Code:** can you explain how the edited reward changes the loop?

---

## 8 · Done when

You are ready to continue when you can:

- explain observation → action → reward → next observation without notes;
- describe why the policy is the same kind of network you built in Foundations 2;
- point to Godot's role and Python's role during training;
- make one BallChase reward edit and predict the likely symptom;
- read a reward curve as a trend, not as one episode.

!!! warning "Training stalled?"
    Check in order: reward sign and scale, sparse rewards, observation bugs,
    resets, and whether the run simply needs more episodes.

---

## 9 · Stretch goals

- Write a random-policy loop for `CartPole-v1` and print the episode return.
- Run two BallChase reward tweaks: one stronger reward and one weaker reward.
  Compare the learning curves.
- Sketch the observation vector and action space for a game idea of your own.

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, _ = env.reset()
total_reward = 0.0

for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"Episode return: {total_reward}")
env.close()
```

---

## What's next

You now have the operational RL vocabulary: observations, actions, rewards,
episodes, returns, policies, exploration, and the Godot/Python training loop.
Next, Foundations 3 turns those pieces into a small reward-learning project so
you can watch a policy improve from trajectories.

!!! info "Self-check before you move on"
    1. What are the four parts of the RL loop?
    2. What is the difference between an observation and a state?
    3. What does the discount rate `γ` control?
    4. What is a policy?
    5. Why does an agent need exploration?
    6. What does Python do during Godot RL training?
    7. What does Godot do during Godot RL training?

??? success "Self-check answers"
    1. Observation → action → reward → next observation.
    2. A state is the full world description; an observation is the part the
       agent receives.
    3. `γ` controls how much future rewards count compared with immediate
       rewards.
    4. A policy is the decision function, usually a neural network, that maps
       observations to actions.
    5. Exploration lets the agent discover behaviors it does not already know
       are useful.
    6. Python runs the training algorithm and updates the policy weights.
    7. Godot simulates the environment, applies actions, computes rewards, and
       resets episodes.

[← Neural Foundations 2](unit-neural-02.md) · [Course home](index.md) · Next planned: Neural Foundations 3
