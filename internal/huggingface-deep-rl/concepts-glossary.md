# Concepts glossary — HF Deep RL course

Terms students encounter in the HF course, with Godot-course framing where useful.

| Term | Definition (student-facing) | HF units | Godot course touchpoint |
|------|----------------------------|----------|-------------------------|
| **Agent** | Learner that chooses actions | 1+ | All units — `AIController` |
| **Environment** | Everything outside the agent; returns state and reward | 1+ | Godot scene / physics |
| **State / observation** | Information the agent sees at step *t* | 1+ | Sensors, `get_obs()` |
| **Action** | What the agent does | 1+ | Discrete or continuous action space |
| **Reward** | Scalar feedback per step | 1+ | Reward functions in GDScript |
| **Return** | Discounted sum of future rewards | 1, 2, 4 | Episodic score in TensorBoard |
| **Episode** | Trajectory from reset to terminal | 1+ | `reset()` / `done` |
| **Policy π(a\|s)** | Strategy mapping states to actions | 1, 4+ | PPO policy network |
| **Value function V(s)** | Expected return from state *s* | 1, 2, 6 | Critic in actor–critic methods |
| **Q-function Q(s,a)** | Expected return after taking *a* in *s* | 2, 3 | DQN |
| **MDP** | Formal model: states, actions, transitions, rewards | 1 | Unit 1 glossary |
| **Markov property** | Future depends only on current state | 1 | Sensor design |
| **Exploration vs exploitation** | Try new actions vs use what works | 1, 2, 3 | ε-greedy (DQN), entropy (PPO) |
| **Value-based method** | Learn values, derive policy (e.g. greedy) | 2, 3 | DQN (Unit 3) |
| **Policy-based method** | Optimize policy directly | 4+ | PPO (Units 2, 4+) |
| **Model-based RL** | Learn environment model (brief mention in taxonomy) | 1 | Not core in Godot course |
| **Monte Carlo (MC)** | Update from full episode returns | 2 | Contrast with TD |
| **Temporal Difference (TD)** | Bootstrap from next-step estimate | 2 | Q-Learning, DQN targets |
| **Bellman equation** | Recursive relationship for value/Q | 2 | Foundation for DQN target |
| **Q-table** | Tabular Q(s,a) for small state spaces | 2 | Replaced by neural net in DQN |
| **Q-Learning** | Off-policy TD control with max over next actions | 2 | Precursor to DQN |
| **ε-greedy** | Random action with probability ε | 3 | Unit 3 exploration callout |
| **Experience replay** | Store transitions, sample mini-batches | 3 | Stabilizes DQN |
| **Target network** | Slow-copy Q-net for stable targets | 3 | DQN hands-on |
| **Policy gradient** | Gradient ascent on expected return | 4 | Leads to PPO |
| **REINFORCE** | Monte Carlo policy gradient | 4 | High variance; motivates critic |
| **Actor** | Policy network | 6+ | PPO actor |
| **Critic** | Value / advantage estimator | 6+ | Reduces variance |
| **Advantage A(s,a)** | How much better *a* is vs average | 6, 8 | GAE in PPO |
| **A2C** | Advantage Actor-Critic (sync) | 6 | Optional mention vs PPO |
| **PPO** | Clipped surrogate objective, stable updates | 8 | Primary algorithm Units 2, 4–8 |
| **GAE** | Generalized Advantage Estimation | 8 | PPO implementation detail |
| **Clip range ε** | Limits policy ratio per update | 8 | Unit 4 hyperparameters |
| **Entropy bonus** | Encourages exploration in PPO | 8 | Compare to ε-greedy |
| **MARL** | Multiple agents learning/interacting | 7 | Unit 7 |
| **Curiosity (RND)** | Intrinsic reward for novel states | 5 | Optional advanced topic |

## Method taxonomy (HF Unit 1)

```
RL methods
├── Value-based      → Q-Learning, DQN
├── Policy-based     → REINFORCE, PPO
├── Actor-Critic     → A2C (hybrid)
└── Model-based      → (overview only in HF)
```

**Deep RL** = function approximation (neural nets) for value or policy.
