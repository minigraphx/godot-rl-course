# World Models — Model-Based RL with Dreamer

!!! info "Time"
    Reading: ~40 min · Training: ~45 min GPU / ~3 h CPU

[Course home](index.md)

---

!!! info "Three ways to see your AI"
    **Godot** (watch the agent act confidently even after only a fraction of the environment steps a PPO baseline would need — the world model has been doing the heavy lifting in imagination) · **TensorBoard** (`train/reward_mean` in dreamer logs alongside world-model reconstruction loss — both should decrease; if reconstruction loss stays high, the model has not learned to represent the scene) · **Latent space plot** (project learned latent states with t-SNE or PCA — distinct clusters for "near wall", "near goal", "open space" show the model has discovered structure the reward function never explicitly defined)

---

## 1 · Model-free vs model-based

Every algorithm in this course so far has been **model-free**: the agent collects transitions from the real environment, computes a gradient, and updates its policy. The environment is a black box — the agent never tries to predict what it will do next. This works, but it is expensive. A PPO agent solving a moderately hard Godot task routinely needs millions of environment steps to converge.

**Model-based RL** takes a different bet: spend compute on learning a model of the environment, then use that model to generate synthetic experience. If the model is accurate, the agent can plan and learn inside it — requiring far fewer interactions with the real environment.

| Property | Model-free (PPO, SAC) | Model-based (Dreamer) |
|---|---|---|
| Learns environment model | No | Yes — encoder, dynamics, decoder, reward |
| Real env steps to converge | High (millions) | Low (tens–hundreds of thousands) |
| Compute per step | Low | High — world model training |
| Total wall-clock time | Can be comparable | Slower per-step, fewer steps |
| Handles sparse rewards | Poorly | Better — can imagine sparse-reward trajectories |
| Engineering complexity | Low | High — more moving parts |
| Model errors | N/A | Can cause "model hallucinations" — agent exploits inaccurate model |

### When does the sample-efficiency gain matter?

The trade-off favours model-based methods when **environment steps are expensive**:

- Simulation is slow (physics-heavy Godot scene, 1 parallel instance)
- You are training on real hardware where each step costs time or wear
- Reward is so sparse that model-free methods explore randomly for millions of steps

When environment steps are cheap — lightweight simulation, 32 parallel instances — model-free methods often converge faster in wall-clock time despite needing more steps. Measure in wall-clock hours, not just timestep count.

!!! warning "More moving parts means more failure modes"
    A PPO training run can fail in a handful of ways (bad learning rate, reward scale). A Dreamer training run can fail in many more: poor reconstruction, dynamics model inaccuracy, KL collapse, imagination horizon too long. Reach for model-based RL when you have a clear reason — sparse rewards, expensive simulation, or a desire to plan. Don't use it as a default upgrade over PPO.

---

## 2 · What a world model learns

A world model is a collection of learned functions that together let the agent simulate the environment internally. The four components are trained jointly from real experience:

```
Encoder:           obs_t         →  z_t          (compress raw observation to latent state)
Dynamics model:    z_t + a_t     →  z_{t+1}      (predict next latent state)
Decoder:           z_t           →  obs_t         (reconstruct observation — used as training signal)
Reward predictor:  z_t           →  r_t           (predict reward from latent state)
```

### Why work in latent space?

Operating directly on pixel observations (64×64 = 12,288 values) is expensive. The encoder compresses each frame to a latent vector `z` of perhaps 32 dimensions. The dynamics model then propagates these small vectors through imagined future steps — 15 steps in imagined time costs the dynamics model 15 tiny forward passes rather than 15 full environment renders.

The decoder reconstructs the original observation from `z`. Its job is not used at inference time — but during training, the reconstruction error is the supervision signal that forces `z` to capture everything necessary to explain the observation. Without the decoder, there is no guarantee the encoder learns anything meaningful.

```
Real trajectory:
  obs_0 ─[Encoder]─▶ z_0 ─[Dynamics + a_0]─▶ z_1 ─[Dynamics + a_1]─▶ z_2
                       │                          │                          │
                   [Decoder]                  [Decoder]                 [Decoder]
                       ▼                          ▼                          ▼
                   recon_0                    recon_1                    recon_2

Training losses:
  reconstruction:  ||obs_t - recon_t||²  (per pixel, or per feature)
  reward:          ||r_t - reward_pred_t||²
  dynamics (KL):   KL( posterior(z_t | obs_t) || prior(z_t | z_{t-1}, a_{t-1}) )
                   ↑ DreamerV1 form, shown for intuition.
                   DreamerV3 uses a balanced KL with free bits:
                   KL_loss = max(KL, free_bits) — see §4 for details.
```

All four components share gradients — improving the dynamics model improves the encoder because better representations lead to lower prediction error downstream.

!!! info "The posterior vs prior distinction"
    During training the encoder sees the real next observation and can compute a *posterior* distribution over `z_{t+1}`. At inference time (imagination), no real observation is available, so the dynamics model must use only its *prior* based on the previous latent state and action. The KL loss forces these two distributions to stay close — if they diverge, the imagined future will be inconsistent with what the encoder would have produced from a real observation.

---

## 3 · Dreamer architecture — RSSM

Dreamer (Hafner et al. 2019, refined in DreamerV2 and DreamerV3) introduces the **Recurrent State Space Model (RSSM)** as its world model backbone. The RSSM is the key architectural insight that separates Dreamer from simpler model-based approaches.

### The RSSM latent state: h and z together

A pure stochastic latent `z` (like a VAE) forgets history between steps — it only sees the current frame. A pure recurrent state `h` (like an LSTM hidden state) has no explicit uncertainty representation. RSSM combines both:

```
h_t  =  GRU(h_{t-1}, z_{t-1}, a_{t-1})   — deterministic recurrent hidden state
z_t  ~  posterior(z | h_t, obs_t)          — stochastic latent (during training)
     ~  prior(z | h_t)                     — stochastic latent (during imagination)

Full state:  s_t = concat(h_t, z_t)
```

| Component | Type | Role |
|---|---|---|
| `h_t` | Deterministic (GRU) | Carries long-term memory across steps — what happened before |
| `z_t` | Stochastic (diagonal Gaussian or categorical) | Represents current uncertainty — what is ambiguous right now |
| `s_t = [h_t, z_t]` | Combined | Full state passed to actor, critic, and reward predictor |

**Why both?** The GRU hidden state `h` accumulates information across many steps — essential for partially observable tasks where a single frame is not enough. The stochastic variable `z` allows the model to represent genuine ambiguity: multiple futures that are all consistent with past observations. DreamerV2 switched from Gaussian `z` to categorical (straight-through gradients), which improves training stability.

### Policy training entirely in imagination

The actor and critic are **never updated from real environment data**. They are trained entirely inside imagined rollouts:

```
1. Encode a real observation → s_0 = [h_0, z_0]
2. Roll out H = 15 steps in imagination:
     s_1, s_2, ..., s_H  using dynamics model + actor actions
3. Compute imagined rewards:  r̂_1, ..., r̂_H  using reward predictor
4. Compute λ-return (advantage):  Vλ = r̂ + γ · V(s_{t+1})
5. Update actor to maximise Vλ
6. Update critic to predict Vλ
```

Real environment steps are only used to train the world model (encoder, dynamics, decoder, reward predictor). Once the world model is accurate, the actor and critic can improve by generating millions of imagined trajectories — each one essentially free.

This is why Dreamer achieves high sample efficiency: the agent does "homework" in imagination between real interactions, rather than waiting for the next real environment step to get a gradient.

```
Outer loop (real env):
  collect 50 real steps → add to replay buffer → train world model for K steps

Inner loop (imagination):
  sample starting states from replay → roll out H steps in imagination
  → update actor and critic from imagined returns
```

---

## 4 · DreamerV3

DreamerV3 (Hafner et al. 2023) is the current production version of Dreamer. Its headline result: a single set of hyperparameters that works across domains as different as Atari, DeepMind Control Suite (DMC), Minecraft, and robot manipulation — without per-domain tuning.

### Setup

```bash
pip install dreamerv3
```

DreamerV3 requires JAX. On a machine with a CUDA GPU:

```bash
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install dreamerv3
```

On CPU (slower but functional for experimentation):

```bash
pip install jax dreamerv3
```

### Running DreamerV3 on a standard benchmark

```python
import dreamerv3
from dreamerv3 import embodied

# Standard DreamerV3 config — works out-of-the-box on DMC tasks
config = embodied.Config(dreamerv3.configs["defaults"])
config = config.update(dreamerv3.configs["medium"])  # medium compute tier

config = config.update({
    "logdir": "logs/dreamer_cheetah",
    "run.train_ratio": 32,      # imagination steps per real step
    "run.log_every": 300,
    "batch_size": 16,
    "jax.prealloc": False,
})

# DMC benchmark: HalfCheetah-v2
import gymnasium as gym
env = gym.make("dm_control/cheetah-run-v0")

# Wrap for dreamerv3
env = dreamerv3.wrap_env(env, config)

agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
replay = embodied.replay.Uniform(config.replay_size, config.replay_online)
embodied.run.train(agent, env, replay, config)
```

### Self-tuning hyperparameters — what DreamerV3 changed

Earlier Dreamer versions required per-domain tuning. DreamerV3 introduces two mechanisms that make a single hyperparameter set stable across very different scales of reward:

**Symlog transforms.** All scalar predictions (reward, value, return) pass through:

```
symlog(x) = sign(x) · log(|x| + 1)
```

This compresses large values and expands small ones symmetrically around zero. A reward of +1000 and a reward of +1 both get sensible gradient magnitudes. Without symlog, large rewards overwhelm small ones and gradients explode; tiny rewards produce vanishing gradients.

**Free bits.** The KL loss is clamped: `KL_loss = max(KL, free_bits)`. This prevents the model from collapsing the stochastic latent `z` to a deterministic point (posterior collapse). The model is allowed up to `free_bits` nats of KL "for free" — it only pays a penalty beyond that threshold.

| DreamerV2 | DreamerV3 |
|---|---|
| Gaussian z | Categorical z (more stable gradients) |
| Manual reward scaling per domain | Symlog transforms — self-normalizing |
| KL tuning per domain | Free bits — automatic posterior regulation |
| Separate configs for Atari, DMC | Single config for all domains |

---

## 5 · Godot integration

DreamerV3 accepts any Gymnasium-compatible environment. The SubViewport pipeline from [Visual Observations](unit-visual-observations.md) exposes Godot as a standard pixel-observation gym env — DreamerV3 can train on it directly.

### Connecting Godot to DreamerV3

```python
import dreamerv3
from dreamerv3 import embodied
import numpy as np
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

# 1. Load the Godot environment (pixel observations, see unit-visual-observations.md)
#    Your Godot scene must use SubViewport + 64x64 pixel obs (Section 2 of that unit)
godot_env = StableBaselinesGodotEnv(
    env_path="./VisualAgent.x86_64",
    n_parallel=1,          # DreamerV3 typically runs one env, trains in imagination
    speedup=4,
    show_window=False,
)

# 2. Reshape obs from flat array to (H, W, C) — DreamerV3 expects channels-last
import gymnasium as gym

class GodotDreamerWrapper(gym.ObservationWrapper):
    """Reshape flat Godot pixel obs [H*W*C] → (H, W, C) for DreamerV3."""

    def __init__(self, env, height=64, width=64, channels=3):
        super().__init__(env)
        self.h, self.w, self.c = height, width, channels
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(height, width, channels),  # channels-last for DreamerV3
            dtype=np.float32,
        )

    def observation(self, obs):
        return obs.reshape(self.h, self.w, self.c)

env = GodotDreamerWrapper(godot_env, height=64, width=64, channels=3)

# 3. Configure DreamerV3 for a pixel-observation task
config = embodied.Config(dreamerv3.configs["defaults"])
config = config.update(dreamerv3.configs["small"])  # smaller model for 64x64 Godot scenes
config = config.update({
    "logdir": "logs/dreamer_godot",
    "run.train_ratio": 16,   # 16 imagination steps per real step (lower for fast Godot sim)
    "encoder.mlp_keys":  "$^",   # no MLP encoder (pure pixel obs)
    "decoder.mlp_keys":  "$^",
    "encoder.cnn_keys":  "image",
    "decoder.cnn_keys":  "image",
    "batch_size": 16,
    "batch_length": 64,
})

# 4. Train
env = dreamerv3.wrap_env(env, config)
agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
replay = embodied.replay.Uniform(config.replay_size, config.replay_online)
embodied.run.train(agent, env, replay, config)
```

!!! tip "Start with a fast Godot scene"
    DreamerV3 collects real steps only to train the world model — it doesn't need millions. But the world model training itself is compute-intensive. During development, use a simple 2D scene where you can verify the reconstruction loss decreases before committing to a complex 3D environment. A scene where `reconstruction_loss` does not decrease after 10k steps indicates the encoder is failing, not the policy.

### Hybrid: pixel obs + proprioceptive state

For most Godot tasks a hybrid observation outperforms pure pixels. Pass both image and state through the world model — DreamerV3 handles dict observations natively:

```python
# Godot GDScript (ai_controller.gd)
func get_obs() -> Dictionary:
    return {
        "image": _capture_frame(),     # flat pixel array
        "state": [velocity.x, velocity.z, dist_to_goal, heading_to_goal],
    }

# Python config: tell DreamerV3 which keys are images vs vectors
config = config.update({
    "encoder.cnn_keys": "image",   # convolutional encoder for image key
    "encoder.mlp_keys": "state",   # MLP encoder for state key
    "decoder.cnn_keys": "image",
    "decoder.mlp_keys": "state",
})
```

The world model learns to reconstruct both streams. State vectors are typically easier to reconstruct than pixels, which acts as an additional consistency constraint on the latent space.

---

## 6 · When world models win

World models provide the most advantage in specific conditions. Before committing to Dreamer, check whether your task matches these patterns:

**Sparse rewards.** When rewards appear only a few times per episode, model-free methods explore randomly waiting to stumble on a reward. The world model can imagine trajectories that reach the reward state — once it has been encountered at least a few times — and train the policy on those imagined successes repeatedly. You get effective learning from very few real reward events.

**Expensive simulation.** If each environment step takes seconds (complex physics, real hardware), spending compute on the world model pays off. Ten imagined steps cost a tiny fraction of one real step.

**Partial observability.** The RSSM's recurrent hidden state `h` explicitly tracks history. A standard MLP policy acting on a single frame has no memory; Dreamer builds a compact history representation automatically.

**Planning and look-ahead.** Once you have a world model, you can run explicit planning algorithms (MCTS, CEM) inside it. Dreamer uses the actor for planning by default, but the world model is available for more deliberate search if needed.

---

## 7 · When world models lose

Being clear-eyed about when not to use model-based RL saves you weeks of debugging:

**Contact-rich manipulation and complex physics.** Rigid-body contact is notoriously hard to model accurately. A world model that is even slightly wrong about object contact dynamics will produce imagined trajectories that the real environment never generates. The policy trains on hallucinated physics and fails on the real scene. Model-free SAC is often faster to a working result for contact-rich tasks.

**When data collection is cheap.** If you can run 32 parallel Godot instances, you collect millions of real steps in hours. The sample-efficiency advantage of Dreamer shrinks — and the engineering overhead (world model bugs, reconstruction debugging) stays constant. At scale, PPO often wins on total wall-clock time despite needing more steps.

**Highly stochastic environments.** Dreamer's dynamics model learns to predict the mean next state. In environments with strong stochasticity (random obstacles, procedurally generated layouts), prediction error is irreducible. The world model stays inaccurate; policy learning in imagination diverges from real-environment performance.

**Short-horizon tasks.** If an episode is 50 steps and dense rewards are available every step, model-free methods converge quickly. The world model's advantage — compressing long-horizon credit assignment — does not apply.

| Situation | Recommendation |
|---|---|
| Sparse reward, few env steps available | Dreamer — strong fit |
| Dense reward, cheap simulation, many parallel envs | PPO/SAC — simpler and often faster |
| Contact-rich physics | SAC model-free — models are inaccurate |
| Partial observability with long horizon | Dreamer — RSSM memory helps |
| Very stochastic environment | Avoid model-based — prediction error is irreducible |
| Prototype / early experiment | PPO first — baseline first, then upgrade if needed |

!!! tip "Baseline first"
    Always train a model-free baseline (PPO or SAC) before switching to Dreamer. The baseline tells you: (1) is the task solvable at all? (2) roughly how many real steps does it take? (3) is Dreamer's sample-efficiency gain worth the engineering cost? A Dreamer run that performs worse than your PPO baseline is a signal that the world model is inaccurate — not that the task is hard.

---

## 8 · Latent space visualization

The latent space `z` (or the combined `[h, z]`) is the world model's internal representation of the environment. Visualizing it tells you what the model has learned — and exposes failure modes before you invest in long training runs.

### t-SNE projection

t-SNE projects high-dimensional latent vectors into 2D while preserving local neighbourhood structure. Distinct clusters indicate the model has discovered meaningful state categories:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_latent_space(agent, replay_buffer, n_samples=2000):
    """
    Extract latent states from a trained Dreamer agent and project to 2D.

    Parameters
    ----------
    agent       : trained DreamerV3 agent (provides encode method)
    replay_buffer : collected transitions (obs, action, reward, ...)
    n_samples   : number of transitions to sample for the plot
    """
    # Sample transitions from replay
    batch = replay_buffer.sample(n_samples)
    obs   = batch["image"]           # (N, H, W, C)
    rewards = batch["reward"]        # (N,)

    # Encode observations to latent vectors
    # In DreamerV3 the latent is (h, z); use h alone for a cleaner plot
    latents = agent.encode(obs)      # (N, latent_dim)

    # Project to 2D with t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    latents_2d = tsne.fit_transform(latents)

    # Colour by reward — reveals which latent regions lead to reward
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        latents_2d[:, 0], latents_2d[:, 1],
        c=rewards, cmap="RdYlGn", alpha=0.6, s=8,
    )
    plt.colorbar(scatter, label="reward")
    plt.title("Dreamer Latent Space (t-SNE) — coloured by reward")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig("latent_tsne.png", dpi=150)
    plt.show()


# Alternative: PCA for a faster (but linear) projection
from sklearn.decomposition import PCA

def visualize_latent_pca(latents, labels, label_name="reward"):
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    explained = pca.explained_variance_ratio_.sum() * 100

    plt.figure(figsize=(9, 7))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1],
                c=labels, cmap="plasma", alpha=0.5, s=6)
    plt.title(f"Latent Space PCA — {explained:.1f}% variance explained")
    plt.colorbar(label=label_name)
    plt.tight_layout()
    plt.savefig("latent_pca.png", dpi=150)
```

### What to look for

**Good signs:**

- Clear clusters that correspond to interpretable states (near goal, near wall, open space)
- High-reward states form a compact region — the model "knows" what reward looks like in latent space
- PCA explains > 50% variance in the first two components — the latent space is structured, not random

**Warning signs:**

- All points form a single blob — the encoder has collapsed; latents are not informative
- Reward is scattered randomly across the space — the reward predictor is not using the latent structure
- t-SNE shows a ring or fractal structure — the dynamics model is producing periodic or degenerate trajectories

!!! info "Colouring options"
    Colouring by reward reveals whether reward-relevant information is encoded. Colouring by episode time step reveals whether the model tracks temporal progression. Colouring by a hand-labelled semantic variable (e.g., "agent is near wall" from your Godot scene) reveals whether specific spatial concepts are represented — even if the model was never explicitly told about them.

---

## 9 · Dreamer vs Dyna — a brief history

Dreamer did not invent the idea of learning a world model and using it for policy updates. Understanding the lineage helps calibrate what is genuinely new.

**Dyna (Sutton, 1991)** is the original model-based RL framework. The core idea is simple: learn a tabular model `P(s', r | s, a)` from real transitions, then use it to generate synthetic transitions for Q-learning updates. Each real transition is supplemented by `k` model-generated transitions, multiplying effective data k-fold.

```
Dyna-Q algorithm:
  for each real step:
    observe (s, a, r, s')
    update Q(s, a) from real transition
    update model: P(s', r | s, a) ← observed
    for k imagined steps:
      sample (ŝ, â) from previously visited states
      ŝ', r̂ = model(ŝ, â)
      update Q(ŝ, â) from imagined transition
```

Dyna works well in tabular settings. It fails in deep RL because: (1) a small neural network model cannot represent complex observation spaces; (2) Q-learning on OOD imagined transitions leads to value overestimation (the same distributional shift problem as offline RL — see [Offline RL unit](unit-offline-rl.md)); (3) imagination is single-step, losing long-horizon structure.

**Modern Dreamer vs Dyna:**

| Dimension | Dyna (1991) | DreamerV3 (2023) |
|---|---|---|
| World model | Tabular P(s', r \| s, a) | RSSM: encoder + GRU + stochastic z + decoder |
| Imagination depth | 1 step | H = 15 steps (full rollouts) |
| Policy update | Q-learning on imagined transitions | Actor-critic entirely in imagination |
| Observation | Tabular (discrete states) | Raw pixels or vectors |
| Latent space | None (model operates in obs space) | Compact z — model operates in latent space |
| Gradient flow | Not differentiable | Differentiable — backprop through imagination |

The key leap from Dyna to Dreamer is **imagining in latent space over long horizons with differentiable backpropagation**. Instead of single-step sampling in observation space, Dreamer runs 15-step imagined trajectories through a differentiable dynamics model and backpropagates the actor gradient through all 15 steps. This lets the actor learn from the long-range consequences of its actions entirely within imagination.

### Trajectory in latent space (conceptual)

```
Real env:     s_0 → s_1 → s_2  (3 real steps, each costing a full render)

Imagination:
  encode(obs_0) → z_0
  dynamics(z_0, a_0) → z_1
  dynamics(z_1, a_1) → z_2
  ...
  dynamics(z_13, a_13) → z_14   (15 steps, each a cheap GRU forward pass)

reward_predictor(z_0), ..., reward_predictor(z_14)
→ compute λ-return
→ backprop through all 15 dynamics steps
→ update actor weights
```

Fifteen imagined steps cost a small fraction of one real render. The actor receives gradient signal from 15 future time steps for the price of a single real interaction.

---

## 10 · Stretch goals

Work through these after reading the main unit. Each isolates one aspect of world-model RL to build intuition you cannot get from theory alone.

**Reconstruction sanity check.** Train DreamerV3 on a simple Godot scene for 10k steps. Save a batch of observations and their reconstructions. View them side by side. Can you recognize the scene in the reconstruction? If the reconstructions are blurry blobs, the encoder has not converged — the dynamics model is training on meaningless latents. Fix the reconstruction before training the policy.

**Imagination vs reality comparison.** After a full training run, roll out 15 imagined steps from a real starting state. Then roll out 15 real steps from the same state with the same actions. Plot reward (real vs imagined) per step. How large is the divergence after 5 steps? After 15? This tells you the effective planning horizon — beyond which the model's predictions are too inaccurate to trust.

**Dreamer vs PPO sample efficiency.** Train both on the same Godot task. Plot `ep_rew_mean` vs number of **real environment steps** (not wall-clock time). Dreamer should reach a given performance level in fewer steps. Now plot vs wall-clock time. Which algorithm reaches the same performance first? The answer depends on your hardware and scene complexity.

**Latent interpolation.** Encode two observations — one near a wall, one near the goal. Linearly interpolate between the two latent vectors (z = α·z_wall + (1-α)·z_goal for α ∈ [0, 1]) and decode each interpolated latent. Do the decoded images show a smooth spatial transition? Smooth interpolation indicates the latent space is semantically structured; discontinuous jumps indicate a collapsed or fragmented representation.

**Dyna from scratch.** Implement the basic Dyna-Q algorithm (5 imagined steps per real step) on the FrozenLake environment from the [Q-Learning unit](unit-q-learning.md). Compare convergence speed to standard Q-learning. This builds direct intuition for where modern Dreamer came from.

---

## What's next?

World models represent the frontier of sample efficiency in deep RL. The ideas here — learning compressed representations, imagining futures, planning in latent space — are active research areas and form the backbone of some of the most capable RL systems built to date.

For production Godot projects, the recommended path remains: **PPO baseline → curiosity if sparse rewards → world models if simulation is expensive or planning is needed**. Each step adds power and complexity; move to the next only when you have evidence the simpler approach has plateaued.

If you want to keep widening the lens, the next unit — **Foundation Models for Control (VLA)** — surveys RT-2, Octo, OpenVLA, and π0: an entirely different bet on how to train general-purpose embodied agents.

[→ Foundation Models for Control](unit-foundation-models.md) · [Course home](index.md)
