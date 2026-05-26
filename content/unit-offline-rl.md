# Offline RL — Learning from Fixed Datasets

[← RLHF & Preference Learning](unit-rlhf.md) · [Course home](index.md)

---

!!! info "Three ways to see your AI"
    - **d3rlpy training loss curve** — watch `critic_loss` and `actor_loss` converge across 100k steps without a single environment interaction; a curve that flattens cleanly means the offline objective is working.
    - **Behavior comparison** — run random policy, BC policy (from [unit-09.md](unit-09.md)), CQL policy, and IQL policy on the same MultiLevelRobot task; record episode returns for each; the ordering tells you exactly how much offline RL added over pure imitation.
    - **Dataset coverage visualization** — project your offline dataset's (observation, action) pairs into 2D with t-SNE or PCA; sparse regions reveal the gaps CQL cannot bridge and IQL must avoid querying.

---

## 1 · Why offline RL?

Online RL is the workhorse of every earlier unit in this course. Given enough environment steps — often in the millions — an agent can discover almost any behavior. That assumption quietly hides a massive cost: the environment must be cheap to run, fast enough to simulate millions of steps, and safe enough to let the agent fail repeatedly while it learns.

Real hardware breaks all three constraints at once.

A robot arm that crashes into a table costs money and time to reset. A surgical assistant that explores freely could injure a patient. A self-driving car cannot "try a few random actions" on a public road. Even when the environment is a simulator, training a new policy from scratch every time you change the reward function wastes months of compute.

**Offline RL** — also called batch RL or data-driven RL — attacks the problem from the other direction. Instead of interacting with an environment, you are given a fixed dataset:

```
D = { (s₀, a₀, r₀, s₀'), (s₁, a₁, r₁, s₁'), … , (sₙ, aₙ, rₙ, sₙ') }
```

This dataset was collected by **any** behavior policy — a human player, a rule-based controller, an earlier RL agent, or a mix of all three. Your job is to extract the best possible policy from that fixed batch of experience, with **zero additional environment interaction**.

### Three canonical use cases

**1. Robot learning from human demonstrations.**
A human teleoperates the robot for several hours. That data becomes the training corpus. This is precisely the setting you explored in [unit-09.md](unit-09.md) with behavioral cloning — offline RL goes further by incorporating the reward signal to improve *beyond* what the demonstrator did.

**2. Fine-tuning a pre-trained policy.**
You shipped a policy in production (see [unit-10.md](unit-10.md)). After deployment you collected real-world logs with reward labels. Offline RL lets you improve the policy from those logs without reverting to simulation.

**3. Safety-critical domains where exploration is dangerous.**
Medical devices, power grids, financial systems. In these settings the offline constraint is not a limitation but a requirement.

### Comparison table

| Property | Online RL | Offline RL | Imitation Learning (BC) |
|---|---|---|---|
| Needs live environment | Yes | No | No |
| Uses reward signal | Yes | Yes | No |
| Can exceed demonstrator | Yes | Partially | No |
| Data requirement | Many env steps | Fixed dataset | Expert demonstrations |
| Main failure mode | Sample inefficiency | Distributional shift | Compounding errors |
| Canonical algorithms | PPO, SAC, TD3 | CQL, IQL, TD3+BC | BC, GAIL, DAgger |
| Typical Godot entry point | `--train` mode | JSON dataset export | HUMAN control mode |

The alignment connection is direct: RLHF (Reinforcement Learning from Human Feedback), the technique behind ChatGPT and Claude, is an offline RL problem. Preference data is collected from humans, then a reward model is trained on it, then a language policy is fine-tuned offline. The mathematical machinery from this unit applies.

---

## 2 · The distributional shift problem

Offline RL sounds straightforward: apply Q-learning to the dataset and extract the greedy policy. The problem is that standard Q-learning on offline data **diverges spectacularly**.

### Why naive Q-learning fails

Q-learning uses the Bellman backup:

```
Q(s, a) ← r + γ · max_{a'} Q(s', a')
```

The `max` over actions in the next state is the source of the problem. During online training that maximum is usually close to what the agent actually does, because the policy and the data are tightly coupled. In offline training, the maximizing action `a'` may be something the dataset **never contains**.

Q-values for out-of-distribution (OOD) actions start at their random initialization. Through bootstrapping, the Bellman backup can propagate those random estimates into the Q-values for in-dataset actions, inflating Q everywhere. The policy greedily selects these inflated OOD actions. Because there is no environment to correct the mistake, the error compounds without bound.

### Concrete example

Your MultiLevelRobot dataset contains many transitions where the robot steps **left** at moderate speed and earns reward 1.0 for staying on the platform. The dataset never contains the action "step left at maximum speed" because the human demonstrator always moved carefully.

The Q-network, lacking any negative signal for maximum-speed-left, may assign it Q = +5. The greedy policy chooses that action. In the real environment, maximum speed causes the robot to slide off the edge and earn -10. But offline training has no environment to observe this — the policy continues to select the OOD action, and the Q-value for it continues to rise.

### The extrapolation error diagram

```
                     Dataset coverage
     ┌──────────────────────────────────────┐
     │                                      │
     │   ████████████████████████████████   │  ← transitions in D
     │   ████████████████████████████████   │
     │   ████████████████████████████████   │
     │                                      │
     │        ← OOD gap →                   │
     │                             ●        │  ← OOD action (never seen)
     │                             ↑        │
     │                       Q = +5 (wrong) │
     │                       R = -10 (real) │
     └──────────────────────────────────────┘
                      action space
```

The OOD action is not penalized by the dataset because it is never observed. The Q-network extrapolates optimistically into the gap, and the policy exploits that extrapolation.

All practical offline RL algorithms are essentially mechanisms for **constraining the learned policy to stay within the support of the dataset**, preventing exploitation of Q-value extrapolation.

---

## 3 · Conservative Q-Learning (CQL)

**Conservative Q-Learning** (Kumar et al., 2020) is the most widely adopted offline RL algorithm and the most direct fix for extrapolation error.

### Key idea

CQL adds a regularization term to the standard Bellman objective. It penalizes high Q-values for actions not well-represented in the dataset, and rewards high Q-values for actions that appear in the data. The policy can only benefit from high Q-values if those values are earned by in-dataset actions.

### CQL objective (intuition)

```
L_CQL = L_Bellman                          (standard TD loss)
       + α · E_{s~D}[ log Σ_a exp Q(s,a) ]   (penalize Q for all actions)
       − α · E_{(s,a)~D}[ Q(s, a) ]          (reward Q for dataset actions)
```

The first penalty term pushes Q down across the full action space. The second reward term pulls Q back up for actions actually in the dataset. The net effect: Q-values for OOD actions are compressed; Q-values for dataset actions are preserved. The policy stays within the data distribution.

### Implementation with d3rlpy

Install the library:

```bash
pip install d3rlpy
```

Collect your dataset from Godot using HUMAN control mode (details in section 4), then:

```python
import d3rlpy
import numpy as np

# Load a dataset collected from Godot via HUMAN control mode
# (exported as a JSON of transitions)
dataset = d3rlpy.dataset.MDPDataset(
    observations=np.array(obs_list),
    actions=np.array(act_list),
    rewards=np.array(rew_list),
    terminals=np.array(done_list),
)

cql = d3rlpy.algos.CQLConfig(
    actor_learning_rate=1e-4,
    critic_learning_rate=3e-4,
    alpha_learning_rate=1e-4,
    batch_size=256,
).create(device="cpu")

cql.fit(
    dataset,
    n_steps=100_000,
    evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
)
cql.save_model("multilevel_cql.d3")
```

### What to watch during training

- `critic_loss` should decrease and stabilize. If it grows monotonically, the conservative penalty `alpha` may be too small.
- `actor_loss` oscillates then settles. Large spikes suggest the policy is trying to exploit OOD actions.
- The `environment` evaluator reports episode return after each evaluation. A slow upward trend is healthy; a flat line after many steps means the dataset coverage is insufficient for the task.

### CQL hyperparameter notes

| Parameter | Default | Effect |
|---|---|---|
| `alpha_learning_rate` | 1e-4 | Controls how aggressively CQL adapts the conservative penalty |
| `batch_size` | 256 | Larger batches stabilize the conservative penalty estimate |
| `actor_learning_rate` | 1e-4 | Lower than online SAC — the actor must stay conservative |

---

## 4 · Collecting the offline dataset from Godot

The dataset format that d3rlpy expects is exactly the transition format produced by godot-rl-agents in HUMAN control mode. This is the same mode you used for demonstrations in [unit-09.md](unit-09.md).

### Enabling HUMAN control mode

In your Godot project's training script, set the control mode to `HUMAN`. The agent wrapper will record every transition as the human plays. Export the recorded buffer to JSON at the end of each session.

```
godot-rl-agents \
    --env path/to/MultiLevelRobot.x86_64 \
    --mode human \
    --export_demo demonstrations.json \
    --n_parallel 1
```

### Converting to d3rlpy format

```python
import json
import numpy as np
import d3rlpy

def load_godot_demos(json_path: str) -> d3rlpy.dataset.MDPDataset:
    with open(json_path, "r") as f:
        raw = json.load(f)

    obs_list = []
    act_list = []
    rew_list = []
    done_list = []

    for episode in raw["episodes"]:
        for transition in episode["transitions"]:
            obs_list.append(transition["obs"])
            act_list.append(transition["action"])
            rew_list.append(transition["reward"])
            done_list.append(float(transition["done"]))

    return d3rlpy.dataset.MDPDataset(
        observations=np.array(obs_list, dtype=np.float32),
        actions=np.array(act_list, dtype=np.float32),
        rewards=np.array(rew_list, dtype=np.float32),
        terminals=np.array(done_list, dtype=np.float32),
    )

dataset = load_godot_demos("demonstrations.json")
print(f"Loaded {len(dataset.episodes)} episodes, "
      f"{sum(len(e) for e in dataset.episodes)} transitions")
```

### How much data to collect

| Dataset size | Expected outcome |
|---|---|
| < 200 episodes | CQL may diverge — coverage too sparse |
| 200–500 episodes | CQL trains but performance is limited by coverage |
| 500–1000 episodes | Good starting point; IQL performs well here |
| 1000+ episodes | CQL reaches near-BC or better; IQL approaches online SAC |

**Quality matters more than quantity.** A dataset of 300 high-quality expert demonstrations typically outperforms 3000 random-walk transitions for CQL. The reason is coverage: CQL can only improve policies within the support of the dataset. If the dataset never demonstrates the critical platform-crossing maneuver, no amount of additional random data will teach it.

!!! warning "Dataset coverage determines the ceiling"
    If you never demonstrated a behavior during recording, CQL cannot learn it. Before training, inspect your dataset's coverage (section 9). If a critical skill is missing, record more demonstrations that specifically target that gap.

---

## 5 · Implicit Q-Learning (IQL)

**Implicit Q-Learning** (Kostrikov et al., 2021) takes a gentler approach to the OOD problem. Instead of penalizing OOD actions explicitly, IQL avoids querying them entirely.

### Key idea

Standard Q-learning takes a max over actions to compute the Bellman target, which requires evaluating the Q-function at potentially OOD points. IQL replaces this with **expectile regression** on the value function, which can be computed using only in-dataset actions.

The intuition: instead of asking "what is the best possible action?", IQL asks "what is the best action that appears in the data?". This sidesteps the extrapolation problem without needing an explicit conservative penalty.

### IQL components

- **Value function V(s):** Trained with expectile regression. The expectile parameter `τ` controls how optimistic the value estimate is — higher `τ` extracts more value from the dataset.
- **Q-function Q(s, a):** Trained with standard TD using V(s') as the target, so it never evaluates OOD actions.
- **Actor:** Trained with advantage-weighted regression (AWR), weighting behavioral cloning loss by `exp(A(s,a))` where `A = Q - V`. High-advantage actions are imitated more strongly.

### d3rlpy code for IQL

```python
iql = d3rlpy.algos.IQLConfig(
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    expectile=0.7,
    weight_temp=3.0,
    batch_size=256,
).create(device="cpu")

iql.fit(dataset, n_steps=100_000)
iql.save_model("multilevel_iql.d3")
```

### IQL hyperparameter notes

| Parameter | Meaning | Tuning |
|---|---|---|
| `expectile` | How optimistic V is (0.5 = mean, 1.0 = max) | Start at 0.7; raise for higher-quality datasets |
| `weight_temp` | Temperature for advantage weighting | Higher = more aggressive policy extraction |

### When to prefer IQL over CQL

| Situation | Recommendation |
|---|---|
| Continuous action spaces (joint velocities, wheel torques) | IQL — avoids discretization artifacts |
| Large datasets (> 1000 episodes) | IQL — expectile regression scales well |
| Mixed-quality datasets (expert + random) | IQL — advantage weighting naturally up-weights expert transitions |
| Small, high-quality expert datasets | CQL — conservative penalty is more effective with sparse coverage |
| Discrete action spaces (platformer controls) | Either; CQL slightly preferred |

For MultiLevelRobot with its continuous joint control, IQL is typically the stronger baseline. Run both and compare.

---

## 6 · Decision Transformer (conceptual)

CQL and IQL both inherit the Q-learning paradigm — they still use Bellman backups, still require a critic, still depend on the reward signal being consistent across the dataset. **Decision Transformer** (Chen et al., 2021) throws the Bellman equation away entirely.

### Framing RL as sequence modeling

A trajectory is a sequence:

```
τ = (R̂₀, s₀, a₀, R̂₁, s₁, a₁, … , R̂ₙ, sₙ, aₙ)
```

where R̂ₜ is the **return-to-go** — the sum of future rewards from timestep t onward. Decision Transformer trains a causal Transformer (GPT-style) to predict the next action given the history of returns-to-go, states, and actions.

Training is pure supervised learning: the model is given the ground-truth sequence from the dataset and asked to predict each action. No Bellman backup, no bootstrapping, no critic.

### Inference: conditioning on desired return

At test time, you set R̂₀ to whatever performance you want — say, the maximum return you ever saw in the dataset. The model conditions on that target and generates actions that, according to its learned sequence model, would produce that return. As rewards accumulate, you subtract them from R̂ to update the return-to-go.

```
R̂ₜ₊₁ = R̂ₜ − rₜ
```

The model is essentially being asked: "If I need this much more reward, what should I do next?"

### Key insight

Decision Transformer does not suffer from distributional shift in the Q-learning sense because it never estimates values for OOD actions. It is a conditional generative model over action sequences. If the desired return is within the support of the training data, it interpolates. If it exceeds the best trajectory in the dataset, it extrapolates — which may or may not work depending on how well the Transformer generalizes.

### HuggingFace integration

The `huggingface/decision-transformer` model card includes a reference implementation that slots into the standard HuggingFace Trainer. For large datasets (tens of thousands of episodes), pre-training a Decision Transformer and fine-tuning it on task-specific data is an increasingly popular pattern — directly analogous to pre-training a language model on web text and fine-tuning on downstream tasks.

### When to use Decision Transformer

- Very large datasets where the sequence modeling context helps (> 10k episodes)
- Pre-training + fine-tuning workflows
- Situations where you want to condition on different target return levels at inference time without retraining
- Research contexts where you want to avoid reward hacking — the model learns to produce the return you ask for, not to exploit a poorly specified reward

For most Godot projects with 1000–5000 demonstrations, CQL or IQL is simpler and faster to train. Decision Transformer becomes compelling when you have access to large diverse datasets or want to combine offline pre-training with language-conditioned behavior.

---

## 7 · Online fine-tuning after offline pre-training

The most practical deployment pattern is a two-stage pipeline:

**Stage 1 — Offline pre-training:** Use CQL or IQL on your dataset. The result is a policy that performs roughly as well as the demonstrators, with some improvement from the reward signal.

**Stage 2 — Online fine-tuning:** Resume training with an online algorithm (SAC or PPO). The offline policy provides a warm start that avoids the catastrophic early exploration failures that plague online RL from scratch.

This combination addresses both problems: offline pre-training gives a policy safely within the data distribution; online fine-tuning corrects the distributional shift between the dataset and the actual deployment environment.

### Why the combination is powerful

- Offline pre-training collapses the "random exploration" phase from millions of steps to zero.
- Online fine-tuning accesses the real environment and corrects gaps in dataset coverage.
- The combined approach often exceeds what either offline-only or online-only training achieves.

### Code sketch: d3rlpy offline → SB3 online

```python
import d3rlpy
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

# ── Stage 1: offline pre-train with CQL ──────────────────────────────────────
cql = d3rlpy.algos.CQLConfig(batch_size=256).create(device="cpu")
cql.fit(dataset, n_steps=100_000)

# Extract actor weights from d3rlpy (PyTorch state dict)
offline_actor_state = cql.impl.policy.state_dict()

# ── Stage 2: online fine-tune with SAC ───────────────────────────────────────
env = make_vec_env("MultiLevelRobot-v0", n_envs=4)
model = SAC("MlpPolicy", env, verbose=1, learning_starts=1_000)

# Transfer offline actor weights into SB3 policy network
# Note: layer names must match — verify architecture parity first
model.policy.actor.load_state_dict(offline_actor_state, strict=False)

# Continue training online
model.learn(total_timesteps=500_000)
model.save("multilevel_offline_then_online")
```

!!! note "Architecture parity"
    The offline actor (d3rlpy) and online actor (SB3) must share the same network architecture for weight transfer to work. Configure both with identical hidden layer sizes (e.g., `[256, 256]`) before training. If architectures differ, you can still initialize the online policy from BC warm-start rather than from CQL actor weights.

### Expected gains

In typical Godot environments, offline pre-training + 500k online steps outperforms 2M online steps from scratch on sparse-reward tasks. The gap is smaller on dense-reward tasks where online RL can explore efficiently.

---

## 8 · Decision guide

Use this flowchart to choose your training strategy.

```
Do you have a fixed dataset of transitions?
│
├── No → Use online RL (PPO, SAC — see earlier units)
│
└── Yes
    │
    ├── Does the dataset contain reward labels?
    │   │
    │   ├── No → Use Behavioral Cloning (unit-09.md)
    │   │
    │   └── Yes
    │       │
    │       ├── Is the dataset from a safe, cheap-to-run environment?
    │       │   │
    │       │   └── Yes → Consider offline pre-train → online fine-tune (section 7)
    │       │
    │       ├── Is your action space continuous?
    │       │   │
    │       │   ├── Yes, large dataset (> 1000 eps) → IQL (section 5)
    │       │   └── Yes, small dataset (< 500 eps)  → CQL (section 3)
    │       │
    │       ├── Discrete action space → CQL (section 3)
    │       │
    │       └── Very large dataset, want pre-train workflow → Decision Transformer (section 6)
```

### When offline RL is the right choice

Use offline RL when:

- You have a dataset but **cannot run the environment** (real hardware, deprecated simulator, production system).
- **Safety prohibits exploration** — medical devices, safety-critical control, financial systems.
- You want to **pre-train before deployment** to avoid the random exploration phase.
- You have **logged production data** with reward signals and want to improve an existing policy.
- You are working on **goal-conditioned tasks** with hindsight relabeling — offline HER combines HER (see [unit-her.md](unit-her.md)) with offline RL to dramatically increase the effective dataset size by relabeling transitions with achieved goals.

Use online RL when you have a fast, cheap simulator and no pre-existing data. Use behavioral cloning when you have expert demonstrations but no reliable reward signal.

---

## 9 · Viz checkpoint

Before declaring your offline policy ready for deployment, run this four-way comparison on the MultiLevelRobot task.

### Setup

Train all four policies on the same fixed dataset:

1. **Random policy** — no training, pure random action sampling
2. **BC policy** — behavioral cloning from unit-09.md, same dataset
3. **CQL policy** — Conservative Q-Learning, 100k steps
4. **IQL policy** — Implicit Q-Learning, 100k steps

Evaluate each for 50 episodes. Record mean episode return and success rate (reaching the final platform).

### Expected ordering

```
Random < BC ≤ CQL ≈ IQL
```

CQL and IQL should both exceed BC, because they incorporate the reward signal. Whether CQL or IQL wins depends on your dataset size and action space. If CQL is below BC, the conservative penalty is too aggressive or the dataset is too sparse — reduce `alpha_learning_rate` or collect more data.

### What to look for in behavior

| Policy | Characteristic behavior |
|---|---|
| Random | Immediate falls, no directional bias |
| BC | Mimics demonstrator path, fails on novel platform configurations |
| CQL | Follows demonstrated paths but adapts slightly to reward; cautious |
| IQL | Smoother, more confident than CQL on platforms it has seen; similar caution on OOD layouts |

### Dataset coverage visualization

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

obs = dataset.observations
acts = dataset.actions

# Concatenate obs and action for joint coverage visualization
joint = np.concatenate([obs, acts], axis=1)
pca = PCA(n_components=2)
reduced = pca.fit_transform(joint)

plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.3, s=2, label="Dataset transitions")
plt.title("Dataset coverage (PCA of obs+action)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend()
plt.savefig("dataset_coverage.png", dpi=150)
```

Sparse regions in this plot are where CQL and IQL are most likely to fail. If you see large empty areas that correspond to critical task states, target those states during additional data collection.

---

## 10 · Stretch goals

**Data efficiency curve.**
Collect 500 and 5000 demonstrations separately using HUMAN control mode. Train CQL on each dataset for 100k steps. Plot mean episode return vs dataset size. The curve should show diminishing returns — offline RL does not scale linearly with data. Identify the knee in the curve: the point where additional data stops meaningfully improving performance.

**D4RL benchmark baseline.**
Install the D4RL benchmark environments:

```bash
pip install d4rl
```

Train IQL on `hopper-medium-v2` or `halfcheetah-medium-v2` — standard offline RL benchmarks with published results. Compare your IQM (interquartile mean) score to the numbers in the IQL paper (Kostrikov et al., 2021). If your numbers are within 5%, your implementation is correct.

**Offline → online improvement measurement.**
Take your best CQL policy. Fine-tune it online with SAC for 500k steps (section 7). Measure the episode return before and after online fine-tuning. Report the percentage improvement. For MultiLevelRobot with sparse rewards, expect 15–40% improvement from online fine-tuning on top of the offline baseline.

**Offline HER.**
Combine offline RL with hindsight experience replay from [unit-her.md](unit-her.md). After collecting your demonstration dataset, relabel each transition with the achieved goal as if it were the intended goal. This multiplies the effective dataset size by the number of goals in each trajectory. Train CQL on the relabeled dataset and compare against CQL on the original labels.

---

[← RLHF & Preference Learning](unit-rlhf.md) · [Course home](index.md) · [→ Decision Transformer](unit-decision-transformer.md)
