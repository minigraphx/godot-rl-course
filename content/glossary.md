# Glossary & Cheat Sheet

[Course home](index.md)

Quick reference for all acronyms, equations, and algorithm parameters used in this course.

---

## Part 1 — Glossary (A–Z)

**A2C (Advantage Actor-Critic)** — Synchronous variant of A3C that runs multiple parallel environments and updates a shared actor and critic after each rollout. *See:* [unit-actor-critic](unit-actor-critic.md) §3

**Actor** — The neural network that outputs a probability distribution (or deterministic action) over the action space; the "policy" half of Actor-Critic methods. *See:* [unit-actor-critic](unit-actor-critic.md) §1

**Advantage (A)** — How much better an action is compared to the average: `A(s,a) = Q(s,a) − V(s)`. Positive advantage → action is better than expected; negative → worse. *See:* [unit-actor-critic](unit-actor-critic.md) §2

**Agent** — The entity that observes the environment, selects actions, and receives rewards. In this course, a Godot node wired to an `AIController`. *See:* [unit-00](unit-00.md) §1

**Alpha (α in SAC)** — Entropy temperature coefficient in SAC; controls how much the agent is rewarded for acting randomly. Can be fixed or learned automatically. *See:* [unit-sac](unit-sac.md) §3

**Atari** — Suite of Atari 2600 game environments (via ALE/Gymnasium) used as the canonical benchmark for discrete-action deep RL; DQN was first demonstrated here. *See:* [unit-q-learning](unit-q-learning.md) §4

**Baseline** — A reference value subtracted from the return to reduce variance in policy gradient updates without changing the expected gradient. The value function V(s) is the standard baseline. *See:* [unit-policy-gradients](unit-policy-gradients.md) §3

**Batch normalization** — Normalizes layer activations across the mini-batch at each gradient step; stabilizes training but interacts poorly with recurrent architectures and small batches common in RL. *See:* [unit-ppo-deep](unit-ppo-deep.md) §5

**Behavioral Cloning (BC)** — Supervised imitation learning: given a dataset of expert `(state, action)` pairs, train a policy to mimic them via cross-entropy or MSE loss. Simple and fast, but suffers from distribution shift when the agent visits states the expert never demonstrated. *See:* [unit-09](unit-09.md) §1

**Bellman equation** — Recursive definition of the value function: `V(s) = E[r + γ V(s')]`. The backbone of all value-based RL algorithms. *See:* [unit-q-learning](unit-q-learning.md) §2

**Beta (β in PBT/PER)** — In PER, controls how much importance-sampling corrects the bias introduced by non-uniform sampling (annealed 0→1 during training). In PBT, sometimes used as a schedule parameter. *See:* [unit-pbt](unit-pbt.md) §2

**Buffer (replay)** — A fixed-size FIFO memory that stores `(s, a, r, s', done)` transitions; off-policy algorithms sample mini-batches from it to break temporal correlation. *See:* [unit-q-learning](unit-q-learning.md) §5

**Clipping (PPO)** — The `clip(ratio, 1−ε, 1+ε)` operation in the PPO objective that prevents the updated policy from deviating too far from the behavior policy in a single gradient step. *See:* [unit-ppo-deep](unit-ppo-deep.md) §2

**CMDP (Constrained MDP)** — An MDP extended with constraint functions `C_i(s,a)` that must stay below thresholds; the framework underlying safe RL. *See:* [unit-safe-rl](unit-safe-rl.md) §1

**CNN (Convolutional Neural Network)** — Feature extractor designed for grid-structured inputs (images, tilemaps); uses weight-shared convolutional filters before a flat feature vector. *See:* [unit-visual-observations](unit-visual-observations.md) §2

**Conjugate gradient** — Iterative solver used in TRPO to compute the natural gradient step without forming the full Fisher information matrix, keeping computation tractable. *See:* [unit-ppo-deep](unit-ppo-deep.md) §1

**Credit assignment** — The problem of determining which past actions are responsible for a delayed reward. Temporal-difference methods (TD, Q-learning) address it by bootstrapping; eligibility traces and n-step returns extend the attribution window. Long-horizon tasks with sparse rewards make credit assignment especially hard. *See:* [unit-q-learning](unit-q-learning.md) §2

**Critic** — The neural network that estimates V(s) or Q(s,a); provides the baseline or target for the actor's gradient. *See:* [unit-actor-critic](unit-actor-critic.md) §1

**Curriculum learning** — Progressively increasing task difficulty during training (easy → hard) so the agent always trains near the edge of its current ability. *See:* [unit-multitask](unit-multitask.md) §3

**D4RL** — Offline RL benchmark suite (datasets for Deep Data-Driven RL) providing fixed transition datasets from domains like HalfCheetah, AntMaze, and Kitchen. *See:* [unit-offline-rl](unit-offline-rl.md) §2

**Dead neurons** — Neurons stuck at zero output (usually ReLU neurons that never activate); can silently collapse network capacity, especially after large gradient steps. *See:* [unit-debugging](unit-debugging.md) §4

**Deterministic policy** — A policy that maps every state to a single action `a = μ(s)` rather than a distribution; used by DDPG and TD3. *See:* [unit-sac](unit-sac.md) §1

**DDPG (Deep Deterministic Policy Gradient)** — Off-policy actor-critic for continuous actions using a deterministic policy and experience replay; predecessor to TD3 and SAC. *See:* [unit-sac](unit-sac.md) §2

**DPO (Direct Preference Optimization)** — Alignment technique that trains a policy directly from human preference pairs without a separate reward model; related to RLHF. *See:* [unit-reward-engineering](unit-reward-engineering.md) §5

**DQN (Deep Q-Network)** — Combines Q-learning with a deep neural network, experience replay, and a target network to stabilize training on high-dimensional inputs. *See:* [unit-03](unit-03.md) §1

**Discount factor (γ)** — A scalar in [0, 1) that down-weights future rewards in the return `G_t = Σ γ^k r_{t+k}`. γ=0 makes the agent myopic (only cares about immediate reward); γ→1 makes it far-sighted. Typical values: 0.99 for episodic tasks, 0.999 for long-horizon continuous tasks. *See:* [unit-00](unit-00.md) §2

**Distributional RL** — Family of RL algorithms that model the full distribution of returns Z(s,a) rather than just its expectation Q(s,a). C51 (the original) represents Z as a categorical distribution over 51 atoms. Improves stability because the agent learns *how variable* an outcome is, not just its mean. Used in Rainbow DQN. *See:* [unit-03](unit-03.md) §7

**Dreamer / DreamerV3** — Model-based RL agents that learn a compact world model in latent space and plan by imagining rollouts entirely inside that model; state-of-the-art on many benchmarks. *See:* [unit-world-models](unit-world-models.md) §4

**Dueling DQN** — DQN variant that decomposes Q into separate value and advantage streams `Q(s,a) = V(s) + A(s,a) − mean(A)`; improves stability when many actions have similar value. *See:* [unit-03](unit-03.md) §4

**Entropy (policy entropy)** — `H(π) = −Σ π(a|s) log π(a|s)`; measures how random a policy is. SAC maximizes entropy explicitly; PPO uses it as a regularizer via `ent_coef`. *See:* [unit-sac](unit-sac.md) §3

**Episode** — A complete sequence of transitions from environment reset to termination (done=True). Episodic return G is the sum of rewards over one episode. *See:* [unit-00](unit-00.md) §2

**ε-greedy** — Exploration strategy that takes a random action with probability ε and the greedy action otherwise; ε is typically annealed from 1.0 to 0.05 during training. *See:* [unit-q-learning](unit-q-learning.md) §3

**Experience replay** — Storing past transitions in a buffer and sampling random mini-batches to train on, breaking temporal correlations and enabling data reuse for off-policy algorithms. *See:* [unit-q-learning](unit-q-learning.md) §5

**Feature extractor** — The front-end network (MLP, CNN, custom GDScript sensor → Python) that converts raw observations into a fixed-size embedding fed to the policy/value heads. *See:* [unit-visual-observations](unit-visual-observations.md) §2

**Function approximation** — Using a parametric model (e.g., a neural network) to represent the value function V(s), Q(s,a), or policy π(a|s) when the state space is too large for a table. Deep RL is function approximation with deep nets; linear function approximation with tile coding or radial basis features predates it. *See:* [unit-q-learning](unit-q-learning.md) §3

**FlyBy** — One of the built-in godot-rl-agents example environments; a drone navigates a 3-D obstacle course, commonly used for testing visual observations and 3-D locomotion policies. *See:* [unit-locomotion](unit-locomotion.md) §2

**GAE (Generalized Advantage Estimation)** — Exponentially-weighted sum of TD errors used to estimate advantages, controlled by λ ∈ [0,1]: λ=0 gives one-step TD; λ=1 gives full Monte Carlo returns. *See:* [unit-ppo-deep](unit-ppo-deep.md) §3

**GDScript** — Python-like scripting language built into Godot; used to implement `AIController`, observation collection, reward functions, and scene logic in this course. *See:* [unit-00](unit-00.md) §1

**Goal-conditioned RL** — RL formulation where the goal g is part of the observation; the agent learns a single policy `π(a | s, g)` that generalizes across many goals. *See:* [unit-her](unit-her.md) §1

**HER (Hindsight Experience Replay)** — Off-policy trick that replays failed episodes with substitute goals equal to what the agent actually achieved, turning failures into learning signal for sparse-reward manipulation tasks. *See:* [unit-her](unit-her.md) §2

**Hierarchical RL** — Decomposes a task into high-level sub-goal selection (manager) and low-level primitive execution (worker), enabling long-horizon planning without credit-assignment over thousands of steps. *See:* [unit-hierarchical](unit-hierarchical.md) §1

**ICM (Intrinsic Curiosity Module)** — Adds a prediction error between a forward model's predicted next-state embedding and the actual next-state embedding as an intrinsic reward bonus, driving exploration. *See:* [unit-curiosity](unit-curiosity.md) §3

**Imitation Learning** — Learning a policy by observing an expert's behaviour, without relying primarily on an environment reward signal. Covers Behavioral Cloning (supervised), DAgger (iterative), GAIL (adversarial), and IRL (reward recovery). *See:* [unit-09](unit-09.md) §1

**Importance sampling** — Technique for reusing transitions generated by an old policy (behavior policy) to estimate gradients under the current policy, via the ratio `π_θ(a|s) / π_θ_old(a|s)`. *See:* [unit-ppo-deep](unit-ppo-deep.md) §2

**Intrinsic motivation** — Reward signal generated internally by the agent (curiosity, novelty, empowerment) rather than from the environment; used to explore sparse-reward tasks. *See:* [unit-curiosity](unit-curiosity.md) §1

**IRL (Inverse RL)** — Learning a reward function from expert demonstrations (the inverse of RL); used as a foundation for imitation and alignment methods. *See:* [unit-reward-engineering](unit-reward-engineering.md) §4

**KL divergence** — `KL(p‖q) = Σ p(x) log(p(x)/q(x))`; measures how different distribution q is from reference p; used in TRPO as a hard constraint and in PPO as a monitoring metric. *See:* [unit-ppo-deep](unit-ppo-deep.md) §2

**LayerNorm** — Normalizes activations across feature dimensions (not batch); more stable than BatchNorm for small-batch RL; default in many modern actor-critic implementations. *See:* [unit-ppo-deep](unit-ppo-deep.md) §5

**Learning rate** — Step size for the gradient update `θ ← θ − α ∇L`; one of the most impactful hyperparameters. Typical range in SB3: `1e-4` to `3e-4`. *See:* [unit-ppo-deep](unit-ppo-deep.md) §6

**Locomotion** — Task category where the agent must control a body to move efficiently (walk, run, jump, fly); continuous-action, dense-reward, and physically simulated. *See:* [unit-locomotion](unit-locomotion.md) §1

**MAML (Model-Agnostic Meta-Learning)** — Meta-learning algorithm that learns an initialization from which a new task can be solved in a few gradient steps; used in multi-task and sim-to-real transfer. *See:* [unit-multitask](unit-multitask.md) §4

**MDP (Markov Decision Process)** — Formal framework for RL: tuple (S, A, P, R, γ) where S=states, A=actions, P=transition probabilities, R=reward function, γ=discount factor. *See:* [unit-00](unit-00.md) §2

**MuJoCo** — Physics engine (now free via DeepMind) widely used for continuous-control benchmarks (HalfCheetah, Ant, Humanoid); integrated into Gymnasium as `gymnasium[mujoco]`. *See:* [unit-locomotion](unit-locomotion.md) §3

**n_steps** — Number of environment steps collected per rollout per parallel environment before a gradient update; key PPO hyperparameter controlling the trade-off between bias and variance of advantage estimates. *See:* [unit-ppo-deep](unit-ppo-deep.md) §6

**NatureCNN** — CNN architecture from the original DQN Nature paper (3 conv layers → flat → 512-dim FC); SB3's default `CnnPolicy` feature extractor for image observations. *See:* [unit-visual-observations](unit-visual-observations.md) §3

**Noisy Networks** — Replaces fixed weights in a DQN's FC layers with learned Gaussian noise parameters to achieve state-dependent exploration without ε-greedy. *See:* [unit-03](unit-03.md) §5

**Observation** — The information the agent receives from the environment at each step (sensor readings, positions, velocities, image pixels, etc.); partial or full depending on the environment. *See:* [unit-00](unit-00.md) §3

**Off-policy** — Learning style where the policy being improved (target policy) differs from the policy that collected the data (behavior policy); enables replay buffer reuse. SAC, DQN, TD3 are off-policy. *See:* [unit-sac](unit-sac.md) §1

**On-policy** — Learning style where each gradient update uses only fresh data collected by the current policy, then discards it. PPO, A2C, REINFORCE are on-policy. *See:* [unit-policy-gradients](unit-policy-gradients.md) §1

**ONNX (Open Neural Network Exchange)** — Model serialization format used to export trained SB3 policies for deployment inside Godot's ONNX bridge without a Python runtime. *See:* [unit-10](unit-10.md) §2

**PBT (Population-Based Training)** — Automated hyperparameter search that runs a population of agents in parallel, periodically copying weights from better-performing agents to worse ones and perturbing their hyperparameters. *See:* [unit-pbt](unit-pbt.md) §1

**PER (Prioritized Experience Replay)** — Replay buffer variant that samples transitions proportional to their TD error (high error → more frequently replayed), corrected by importance-sampling weights. *See:* [unit-03](unit-03.md) §6

**Policy** — A mapping from states to actions, either stochastic `π(a|s)` or deterministic `a=μ(s)`; the primary object RL algorithms optimize. *See:* [unit-00](unit-00.md) §2

**Policy gradient** — Family of algorithms that directly maximize expected return by following the gradient `∇_θ J(θ) = E[∇_θ log π_θ(a|s) · A(s,a)]`. *See:* [unit-policy-gradients](unit-policy-gradients.md) §2

**PPO (Proximal Policy Optimization)** — On-policy Actor-Critic algorithm that clips the probability ratio to prevent destructively large updates; default algorithm in godot-rl-agents. *See:* [unit-ppo-deep](unit-ppo-deep.md) §2

**Q-value** — Action-value function `Q(s,a)` = expected discounted return when taking action a in state s and following policy π thereafter. *See:* [unit-q-learning](unit-q-learning.md) §1

**Rainbow DQN** — DQN variant that combines six improvements (PER, dueling, noisy nets, n-step returns, distributional RL, double DQN) into one agent achieving state-of-the-art Atari performance. *See:* [unit-03](unit-03.md) §7

**Recurrence (LSTM / GRU)** — Architectures that maintain a hidden state across timesteps, giving the policy memory. Essential for POMDPs (partially observable environments) where the current observation is insufficient to determine the state. LSTM uses separate cell and hidden states with forget/input/output gates; GRU merges them into update/reset gates for fewer parameters. *See:* [unit-08](unit-08.md) §2

**REINFORCE** — Monte Carlo policy gradient: collect full episodes, compute returns G_t, then update `θ ← θ + α ∇_θ log π_θ(a_t|s_t) · G_t`; high variance but conceptually simple. *See:* [unit-policy-gradients](unit-policy-gradients.md) §2

**Reparameterization trick** — Expresses a stochastic sample `a ~ π(·|s)` as a deterministic function of a noise variable `ε ~ N(0,1)`: `a = μ(s) + σ(s) · ε`; allows gradients to flow through the sampling operation. Used in SAC. *See:* [unit-sac](unit-sac.md) §4

**Replay buffer** — See *Buffer (replay)*. *See:* [unit-q-learning](unit-q-learning.md) §5

**Return (G)** — Discounted sum of future rewards from timestep t: `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …`; the quantity RL algorithms ultimately maximize. *See:* [unit-00](unit-00.md) §2

**RLHF (Reinforcement Learning from Human Feedback)** — Training paradigm that learns a reward model from human preference labels and then fine-tunes a policy with RL; used in LLM alignment (ChatGPT, Claude). *See:* [unit-reward-engineering](unit-reward-engineering.md) §5

**RND (Random Network Distillation)** — Intrinsic curiosity method: the agent trains a predictor network to match a fixed random target network; states the agent has visited often have lower prediction error, so novel states yield a higher intrinsic reward. *See:* [unit-curiosity](unit-curiosity.md) §4

**Rollout** — One complete data-collection phase: the current policy interacts with the environment for `n_steps` steps across all parallel environments, producing a batch of transitions for the gradient update. *See:* [unit-ppo-deep](unit-ppo-deep.md) §3

**SAC (Soft Actor-Critic)** — Off-policy Actor-Critic that maximizes a combined objective of expected return and policy entropy, making it naturally exploratory and robust to hyperparameters; dominant in continuous control. *See:* [unit-sac](unit-sac.md) §2

**SB3 (Stable-Baselines3)** — Python library providing clean, tested implementations of PPO, SAC, DQN, A2C, DDPG, TD3, HER, and more; the training backend used in this course. *See:* [unit-01](unit-01.md) §2

**Score function estimator** — Another name for the REINFORCE / log-derivative trick: `∇_θ E[f(x)] = E[f(x) ∇_θ log p_θ(x)]`; the mathematical core of policy gradient methods. *See:* [unit-policy-gradients](unit-policy-gradients.md) §2

**Self-play** — Training paradigm where agents play against copies of themselves (or a pool of past versions), generating an automatic curriculum of increasingly skilled opponents. *See:* [unit-self-play](unit-self-play.md) §1

**Sim-to-real transfer** — The challenge of transferring a policy trained in simulation to a real physical robot without performance collapse; addressed via domain randomization, adaptive policies, and careful sensor matching. *See:* [unit-sim-to-real](unit-sim-to-real.md) §1

**Sparse reward** — Reward structure where non-zero rewards are extremely rare (e.g., +1 only at task completion), making standard gradient-based RL methods ineffective without exploration bonuses or shaping. *See:* [unit-curiosity](unit-curiosity.md) §1

**State** — A complete description of the environment at a point in time; in practice, agents often receive a partial observation rather than the true state. *See:* [unit-00](unit-00.md) §2

**Target network** — A periodically-updated copy of the Q-network (or critic) whose weights are held fixed while the online network trains against it, preventing feedback loops that destabilize DQN. *See:* [unit-03](unit-03.md) §2

**TD error (δ)** — Temporal-difference error: `δ = r + γ V(s') − V(s)`; the mismatch between a bootstrapped target and the current value estimate; drives both value learning and policy gradients. *See:* [unit-q-learning](unit-q-learning.md) §2

**TD3 (Twin Delayed Deep Deterministic)** — Off-policy continuous-action algorithm that fixes DDPG instability with twin critics (takes the minimum Q), delayed policy updates, and target policy smoothing. *See:* [unit-sac](unit-sac.md) §2

**Temperature (α)** — See *Alpha (α in SAC)*. *See:* [unit-sac](unit-sac.md) §3

**Timestep** — A single environment step: the agent receives observation o_t, produces action a_t, the environment transitions to o_{t+1} and emits reward r_t. Training budgets are measured in total timesteps. *See:* [unit-00](unit-00.md) §2

**Trajectory optimization** — Planning approach that directly searches for the sequence of actions `(a_0, a_1, …, a_T)` that maximizes return, using gradient-based or sampling-based methods. Unlike RL, it typically requires a differentiable model of the environment. *See:* [unit-world-models](unit-world-models.md) §3

**TRPO (Trust Region Policy Optimization)** — On-policy algorithm that constrains each policy update to stay within a KL-divergence trust region; theoretically motivated but computationally expensive. PPO is the practical successor. *See:* [unit-ppo-deep](unit-ppo-deep.md) §1

**Value function (V)** — `V(s)` = expected discounted return from state s under policy π; estimated by the Critic in Actor-Critic methods; used as a baseline to reduce variance. *See:* [unit-actor-critic](unit-actor-critic.md) §2

**VecEnv** — SB3's vectorized environment abstraction that runs N parallel environment copies in a single process or across subprocesses, providing batched steps for faster data collection. *See:* [unit-02](unit-02.md) §3

**VecNormalize** — SB3 wrapper around VecEnv that maintains running statistics to normalize observations and rewards online during training; essential for stable continuous-control training. *See:* [unit-sac](unit-sac.md) §6

**World model** — A learned neural model of environment dynamics: given (s_t, a_t), predicts (s_{t+1}, r_t). Enables planning, imagination, and data-efficient learning. *See:* [unit-world-models](unit-world-models.md) §1

---

## Part 2 — Algorithm Cheat Sheet

### PPO — Proximal Policy Optimization

| Field | Detail |
|-------|--------|
| **Summary** | On-policy Actor-Critic that clips the policy-ratio update to prevent large steps |
| **On/Off-policy** | On-policy |
| **Action space** | Discrete & Continuous |
| **Key SB3 hyperparameters** | `learning_rate=3e-4`, `n_steps=2048`, `batch_size=64`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.0`, `vf_coef=0.5` |
| **TensorBoard metrics** | `train/approx_kl` (keep < 0.02), `train/clip_fraction` (keep < 0.1), `train/entropy_loss`, `rollout/ep_rew_mean` |
| **SB3 class** | `from stable_baselines3 import PPO` |
| **Best for** | Godot game environments, discrete or continuous, parallel envs |

---

### DQN — Deep Q-Network

| Field | Detail |
|-------|--------|
| **Summary** | Off-policy value-based algorithm using a replay buffer and target network for discrete actions |
| **On/Off-policy** | Off-policy |
| **Action space** | Discrete only |
| **Key SB3 hyperparameters** | `learning_rate=1e-4`, `buffer_size=1_000_000`, `learning_starts=50_000`, `batch_size=32`, `tau=1.0`, `gamma=0.99`, `train_freq=4`, `target_update_interval=10_000`, `exploration_fraction=0.1`, `exploration_final_eps=0.05` |
| **TensorBoard metrics** | `train/loss`, `rollout/ep_rew_mean`, `train/exploration_rate` |
| **SB3 class** | `from stable_baselines3 import DQN` |
| **Best for** | Discrete-action environments; image observations with CnnPolicy |

---

### SAC — Soft Actor-Critic

| Field | Detail |
|-------|--------|
| **Summary** | Off-policy Actor-Critic that maximizes return + entropy; sample-efficient and robust |
| **On/Off-policy** | Off-policy |
| **Action space** | Continuous only |
| **Key SB3 hyperparameters** | `learning_rate=3e-4`, `buffer_size=1_000_000`, `learning_starts=100`, `batch_size=256`, `tau=0.005`, `gamma=0.99`, `train_freq=1`, `ent_coef='auto'`, `target_entropy='auto'` |
| **TensorBoard metrics** | `train/ent_coef`, `train/actor_loss`, `train/critic_loss`, `rollout/ep_rew_mean` |
| **SB3 class** | `from stable_baselines3 import SAC` |
| **Best for** | Continuous control (robot arms, locomotion); expensive simulations |

---

### A2C — Advantage Actor-Critic

| Field | Detail |
|-------|--------|
| **Summary** | Synchronous on-policy Actor-Critic; simpler than PPO, useful as a learning baseline |
| **On/Off-policy** | On-policy |
| **Action space** | Discrete & Continuous |
| **Key SB3 hyperparameters** | `learning_rate=7e-4`, `n_steps=5`, `gamma=0.99`, `gae_lambda=1.0`, `ent_coef=0.0`, `vf_coef=0.5`, `max_grad_norm=0.5`, `rms_prop_eps=1e-5` |
| **TensorBoard metrics** | `train/entropy_loss`, `train/value_loss`, `rollout/ep_rew_mean` |
| **SB3 class** | `from stable_baselines3 import A2C` |
| **Best for** | Quick experiments; environments where PPO's longer rollout budget is wasteful |

---

### DDPG — Deep Deterministic Policy Gradient

| Field | Detail |
|-------|--------|
| **Summary** | Off-policy Actor-Critic with deterministic policy; foundational continuous-control algorithm, now largely superseded by TD3/SAC |
| **On/Off-policy** | Off-policy |
| **Action space** | Continuous only |
| **Key SB3 hyperparameters** | `learning_rate=1e-3`, `buffer_size=1_000_000`, `learning_starts=100`, `batch_size=100`, `tau=0.005`, `gamma=0.99`, `train_freq=1` |
| **TensorBoard metrics** | `train/actor_loss`, `train/critic_loss`, `rollout/ep_rew_mean` |
| **SB3 class** | `from stable_baselines3 import DDPG` |
| **Best for** | Continuous control; historical comparison with TD3/SAC |

---

### TD3 — Twin Delayed Deep Deterministic

| Field | Detail |
|-------|--------|
| **Summary** | DDPG + twin critics + delayed policy update + target noise; significantly more stable than DDPG |
| **On/Off-policy** | Off-policy |
| **Action space** | Continuous only |
| **Key SB3 hyperparameters** | `learning_rate=1e-3`, `buffer_size=1_000_000`, `learning_starts=100`, `batch_size=100`, `tau=0.005`, `gamma=0.99`, `train_freq=1`, `policy_delay=2`, `target_policy_noise=0.2`, `target_noise_clip=0.5` |
| **TensorBoard metrics** | `train/actor_loss`, `train/critic_loss`, `rollout/ep_rew_mean` |
| **SB3 class** | `from stable_baselines3 import TD3` |
| **Best for** | Continuous control when SAC's stochastic policy is undesirable |

---

### REINFORCE

| Field | Detail |
|-------|--------|
| **Summary** | Vanilla Monte Carlo policy gradient; collect full episodes, update with discounted returns |
| **On/Off-policy** | On-policy |
| **Action space** | Discrete & Continuous |
| **Key hyperparameters** | `learning_rate`, `gamma`; no SB3 class — implemented manually in this course |
| **TensorBoard metrics** | `rollout/ep_rew_mean`, `train/policy_gradient_loss` |
| **SB3 class** | None (custom implementation) |
| **Best for** | Understanding policy gradients from scratch before moving to PPO |

---

### BC — Behavioral Cloning

| Field | Detail |
|-------|--------|
| **Summary** | Supervised imitation: minimize cross-entropy between expert actions and predicted actions; no environment interaction needed |
| **On/Off-policy** | Offline (no RL) |
| **Action space** | Discrete & Continuous |
| **Key hyperparameters** | `learning_rate=1e-3`, `batch_size=64`, `n_epochs` over the demo dataset |
| **TensorBoard metrics** | `train/bc_loss`, evaluation success rate |
| **SB3 class** | `from imitation.algorithms import bc` (imitation library) |
| **Best for** | Bootstrapping a policy from demonstrations before RL fine-tuning |

---

### DT — Decision Transformer

| Field | Detail |
|-------|--------|
| **Summary** | Frames offline RL as sequence modeling: a Transformer predicts the next action conditioned on past (return-to-go, state, action) tokens |
| **On/Off-policy** | Offline |
| **Action space** | Discrete & Continuous |
| **Key hyperparameters** | `context_length`, `n_layer`, `n_head`, `learning_rate`, target return (RTG) at inference |
| **TensorBoard metrics** | `train/action_loss`, offline evaluation return |
| **SB3 class** | None (HuggingFace `decision_transformer` or custom) |
| **Best for** | Offline datasets with wide return coverage; goal-conditioned inference via RTG conditioning |

---

### HER — Hindsight Experience Replay

| Field | Detail |
|-------|--------|
| **Summary** | Replay buffer wrapper that relabels failed episode goals with achieved goals, enabling learning from sparse-reward manipulation tasks |
| **On/Off-policy** | Off-policy (wraps SAC/TD3/DDPG) |
| **Action space** | Discrete & Continuous |
| **Key SB3 hyperparameters** | `n_sampled_goal=4`, `goal_selection_strategy='future'`; underlying algorithm hyperparameters unchanged |
| **TensorBoard metrics** | `rollout/success_rate`, `rollout/ep_rew_mean` |
| **SB3 class** | `from stable_baselines3 import HerReplayBuffer` |
| **Best for** | Sparse-reward goal-conditioned tasks (robot manipulation, navigation) |

---

### RND — Random Network Distillation

| Field | Detail |
|-------|--------|
| **Summary** | Curiosity method: intrinsic reward = prediction error of a network learning to match a fixed random encoder; scales to image observations |
| **On/Off-policy** | On-policy (typically paired with PPO) |
| **Action space** | Discrete & Continuous |
| **Key hyperparameters** | `int_coef` (intrinsic reward scale), `ext_coef` (extrinsic reward scale), feature embedding size |
| **TensorBoard metrics** | `train/intrinsic_reward_mean`, `train/rnd_loss`, `rollout/ep_rew_mean` |
| **SB3 class** | None (custom or sb3-contrib) |
| **Best for** | Hard-exploration sparse-reward environments where ICM is insufficient |

---

## Part 3 — Core Equations Reference

### Bellman Equation

```
V(s) = E_a~π [ r(s,a) + γ · V(s') ]
```

The value of a state equals the expected immediate reward plus discounted value of the next state. Every bootstrapped value update in RL is an approximation of this fixed-point equation.

---

### Policy Gradient Theorem (REINFORCE)

```
∇_θ J(θ) = E_{τ~π_θ} [ Σ_t ∇_θ log π_θ(a_t | s_t) · G_t ]
```

The gradient of expected return equals the expected log-probability gradient weighted by the return. Unbiased but high-variance; subtracting a baseline (V(s)) gives the advantage-based form used in A2C/PPO.

---

### PPO Clipped Objective

```
L_CLIP(θ) = E_t [ min(
    r_t(θ) · A_t ,
    clip(r_t(θ), 1-ε, 1+ε) · A_t
) ]

where r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
```

Take the minimum of the unclipped and clipped surrogate; this pessimistically ignores improvements beyond the trust region and prevents the ratio from drifting too far from 1, ensuring data remains on-policy.

---

### SAC Entropy-Augmented Objective

```
J(π) = E_{τ~π} [ Σ_t ( r(s_t, a_t) + α · H(π(· | s_t)) ) ]

H(π(· | s)) = -E_{a~π} [ log π(a | s) ]
```

SAC maximizes both task reward and policy entropy. The temperature α trades off exploitation (high reward) against exploration (high entropy). When α is learned, it is adjusted to match a target entropy.

---

### GAE — Generalized Advantage Estimation

```
A_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}

δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
```

A geometrically-weighted sum of TD errors with decay (γλ). λ=0 collapses to the one-step TD advantage (low variance, high bias); λ=1 collapses to full Monte Carlo returns (high variance, low bias). SB3 PPO default: `gae_lambda=0.95`.

---

### TD Error (δ)

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
```

The temporal-difference error is the "surprise" signal: how much better or worse the observed reward plus bootstrapped future value is compared to the current estimate. Minimizing δ² is the critic's training objective.

---

### KL Divergence

```
KL(p ‖ q) = Σ_x p(x) · log( p(x) / q(x) )
```

Measures how much information is lost when distribution q is used to approximate p. In RL: measures the distance between the old and new policy after a gradient update; TRPO enforces KL ≤ δ as a hard constraint; PPO monitors `approx_kl` as a soft signal.

---

### RND Intrinsic Reward

```
r_i(s_t) = ‖ f̂(s_t ; θ_pred) - f(s_t ; θ_fixed) ‖²
```

The intrinsic reward is the squared L2 distance between the predictor network output and the fixed random target network output. Novel states the agent has not visited have high prediction error (high bonus); frequently visited states have low error (low bonus), driving the agent onward.
