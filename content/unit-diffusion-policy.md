# Diffusion Policy — Multimodal Action Generation for Robot Control

You have spent the course building Gaussian policies. PPO outputs a Gaussian over actions. SAC outputs a squashed Gaussian over actions. Even the behaviour-cloned imitation policy you trained back in Unit 9 was a Gaussian regressor. This unit is about the moment that abstraction breaks — and what to do when it does. **Diffusion Policy** (Chi et al., 2023) replaces the Gaussian head with a full denoising diffusion model, lifting the single-peak limitation that has been silently hurting every continuous-control agent you have trained so far. It is the dominant policy class in modern manipulation research, and the rest of this unit shows you why.

[← SAC](unit-sac.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~40 min · Training: ~30 min GPU / ~2 h CPU

---

!!! info "Three ways to see your AI"
    A 2-D bimodal toy task where SAC collapses to the mean and Diffusion Policy recovers both modes · The DDIM denoising trajectory rendered live in Godot (action vector animating from noise to a clean grasp) · A side-by-side video of action chunks — jittery SAC control vs. smooth diffusion-policy chunks executing on a 7-DoF arm

---

## 0 · The multimodality problem that SAC can't solve

Every policy you have trained in this course outputs a Gaussian distribution `N(μ(s), σ(s))`. PPO learns `μ` and `σ` jointly. SAC learns them under the maximum-entropy objective. Behaviour cloning fits them to demonstrations with maximum likelihood. All three share the same fundamental limitation: **a Gaussian has exactly one peak.**

That is fine when the optimal action at state `s` is a single point. But many real tasks have multiple equally good solutions:

- **Pick up a cup.** You can grasp it from the left, or from the right. Both grips score the same reward. Both appear in human demonstrations.
- **Navigate around a pillar.** Going left and going right both reach the goal. A coin flip is a valid policy.
- **Choose a Go opening.** Multiple stylistically different openings achieve the same long-term winrate.
- **Reach a target with a 7-DoF arm.** A redundant manipulator has an infinite-dimensional null space of joint configurations that all produce the same end-effector pose.

In every one of these cases, the action distribution that you *should* learn is multimodal — two or more separated peaks of equally good actions. A Gaussian, asked to fit such a distribution, will collapse onto the average of the modes. That average is often catastrophically bad.

```python
# Bimodal task: go to x=+1 OR x=-1 (both equally rewarded, x=0 hits a wall)
# Demo data: half the trajectories go to +1, half go to -1.
#
# Gaussian BC policy:
#   maximum-likelihood fit -> mu = 0, sigma ~ 1.0
#   at execution: samples cluster around x=0 -> hits the wall every time
#
# Diffusion policy:
#   the denoiser learns a bimodal score field with two attractors at +/-1
#   at execution: each rollout samples from either mode -> never hits the wall
```

You have already seen the symptoms of this in earlier units. The `unit-09` (imitation learning) demos that mixed two human teachers often produced an agent that did neither thing well — that was Gaussian mode-averaging. The dexterous grasping example in `unit-robotics` plateaued at ~60% success — that was mode-averaging across left/right grasps. And the locomotion stability fix in `unit-locomotion` that needed a curriculum to bias demos toward a *single* gait — that was a workaround for the same Gaussian limitation.

Diffusion Policy removes the limitation directly. The policy head no longer outputs `(μ, σ)`. Instead it outputs an entire learned probability distribution over the action space, sampled via iterative denoising. Whatever shape the demonstration data has — unimodal, bimodal, ring-shaped, banana-shaped — the diffusion model can represent it.

The price you pay is computational: instead of a single forward pass, you run `T` denoising steps per action. The rest of this unit is about (a) understanding *why* denoising can represent arbitrary distributions, (b) implementing the architecture, and (c) keeping `T` small enough that the policy runs at 60 Hz in a Godot loop.

---

## 1 · Diffusion models — from images to actions

Diffusion models were invented for image generation (Sohl-Dickstein 2015; Ho et al. 2020, DDPM). The same machinery transfers to action generation with no conceptual change — we just swap the data dimension from `H×W×3` to `act_dim` and add an observation as conditioning input.

The core idea has two halves: a fixed **forward process** that destroys data by adding noise, and a learned **reverse process** that recovers it.

**Forward process (no learning).** Start with a clean data point `x_0` (a real action vector from a demonstration). Repeatedly add a small amount of Gaussian noise over `T` steps until at `t=T` the sample is indistinguishable from pure noise. The whole trajectory is closed-form:

```
# Forward process — add noise in closed form
# beta_t is a fixed noise schedule, alpha_t = 1 - beta_t,
# alpha_bar_t = product_{s<=t} alpha_s (cumulative product)

x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)

# At t=0:  x_0 = the real action (no noise)
# At t=T:  x_T ~ N(0, I)  (pure noise, x_0 has decayed away)
```

You never need to learn this — it is a deterministic recipe.

**Reverse process (learned).** Train a neural network `eps_theta(x_t, t, s)` to predict the noise that was added at step `t`, conditioned on the observation `s`. Given that prediction we can take one denoising step back toward `x_0`:

```
# Reverse process — one denoising step
# Subtract predicted noise to estimate x_0, then re-noise to step t-1
x_{t-1} = mu_theta(x_t, t, s) + sigma_t * z,   z ~ N(0, I)

# Training objective: predict the noise, simple regression
L = E_{x_0, t, eps} [ || eps - eps_theta(x_t, t, s) ||^2 ]
# where x_t is built from x_0 by the forward formula above
# and s is the conditioning observation
```

**At inference.** Start from `x_T ~ N(0, I)`, run the reverse process for `T` steps, and out comes a sample from the conditional distribution `p(action | obs)`. No mean, no variance head, no Gaussian assumption — just iterative denoising.

Why does this beat a Gaussian for action generation? The reverse process is a sequence of `T` learned conditional transitions. Composed, they can represent **arbitrarily complex distributions**, including the bimodal cup-grasp distribution from Section 0. The denoiser does not have to "choose a mode" — it learns a vector field (the gradient of `log p`, equivalent to a score function) that has two attractors, and which attractor a particular rollout converges to depends on the random initial noise `x_T`. Flip the coin at the start of denoising, and you get a left grasp or a right grasp.

!!! info "Score-matching intuition"
    The trained noise predictor `eps_theta` is, up to a scale factor, an estimator of `-grad_x log p(x|s)` — the score function of the data distribution. Iterative denoising is gradient ascent on log-density with controlled noise injection. This is why diffusion can represent any distribution: scores can be arbitrarily shaped, while Gaussians are shape-constrained by their two parameters.

---

## 2 · Diffusion Policy architecture

Chi et al. (2023) took DDPM and conditioned it on observations, producing **Diffusion Policy**. They proposed two architectural variants depending on observation modality.

| Variant | Observation type | Conditioning mechanism | Best for |
|---------|------------------|------------------------|----------|
| CNN U-Net | Images (RGB or depth) | FiLM (feature-wise linear modulation) on U-Net features | Robot manipulation from cameras |
| Transformer | Low-dim state vector | Cross-attention from obs tokens to action tokens | State-based control, fast inference |
| **MLP (course choice)** | Low-dim state vector | Concatenation of `(noisy_action, obs, time_emb)` | Godot agents with vector obs |

The MLP variant is the simplest and matches the course's standard observation setup (the `AIController.get_obs()` dictionaries we have been using since RL Essentials). It is also fast enough to run inside the Godot loop at 60 Hz once we add DDIM sampling in Section 4.

Here is a complete minimal implementation. It is small enough to read in one sitting and works as a drop-in policy head for any continuous-action Godot env.

```python
import torch
import torch.nn as nn
import numpy as np


class SinusoidalPosEmb(nn.Module):
    """Time-step embedding for the diffusion denoiser.

    Maps a scalar denoising step t in [0, T-1] to a `dim`-vector using
    the same sinusoidal positional encoding as in Transformers.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class DiffusionMLP(nn.Module):
    """Noise prediction network conditioned on observation and diffusion timestep.

    The model learns eps_theta(x_t, t, obs) -> predicted noise.
    Sampling iterates the learned reverse process to produce actions.
    """
    def __init__(self, obs_dim: int, act_dim: int,
                 d_model: int = 256, n_diffusion_steps: int = 100):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_steps = n_diffusion_steps

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.Mish(),
            nn.Linear(d_model * 2, d_model),
        )
        self.net = nn.Sequential(
            nn.Linear(act_dim + obs_dim + d_model, d_model),
            nn.Mish(),
            nn.Linear(d_model, d_model),
            nn.Mish(),
            nn.Linear(d_model, d_model),
            nn.Mish(),
            nn.Linear(d_model, act_dim),
        )

        # DDPM noise schedule (linear betas, the original DDPM choice)
        betas = torch.linspace(1e-4, 0.02, n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('sqrt_alphas_bar', alphas_bar.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_bar', (1 - alphas_bar).sqrt())

    def forward(self, noisy_action, t, obs):
        """Predict noise eps given noisy action x_t, timestep t, observation."""
        t_emb = self.time_emb(t)
        x = torch.cat([noisy_action, obs, t_emb], dim=-1)
        return self.net(x)

    def loss(self, action, obs):
        """DDPM training loss — predict the noise added at a random timestep."""
        B = action.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=action.device)
        eps = torch.randn_like(action)
        noisy = (self.sqrt_alphas_bar[t, None] * action
                 + self.sqrt_one_minus_alphas_bar[t, None] * eps)
        pred_eps = self.forward(noisy, t, obs)
        return ((eps - pred_eps) ** 2).mean()

    @torch.no_grad()
    def sample(self, obs, n_ddim_steps: int = 10):
        """DDIM fast sampling — generate an action conditioned on obs.

        DDIM picks a sparse subsequence of timesteps (e.g. 10 out of 100)
        and uses the deterministic update rule from Song et al. 2021.
        """
        B = obs.shape[0]
        x = torch.randn(B, self.act_dim, device=obs.device)

        timesteps = torch.linspace(self.n_steps - 1, 0, n_ddim_steps,
                                   dtype=torch.long, device=obs.device)

        for i, t in enumerate(timesteps):
            t_batch = t.expand(B)
            pred_eps = self.forward(x, t_batch, obs)

            alpha_bar = self.alphas_bar[t]
            pred_x0 = (x - (1 - alpha_bar).sqrt() * pred_eps) / alpha_bar.sqrt()
            pred_x0 = pred_x0.clamp(-1, 1)  # actions are normalised to [-1, 1]

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_bar_next = self.alphas_bar[t_next]
                x = (alpha_bar_next.sqrt() * pred_x0
                     + (1 - alpha_bar_next).sqrt() * pred_eps)
            else:
                x = pred_x0

        return x
```

That is the whole thing. About 80 lines for a working multimodal continuous-action policy.

!!! tip "Mish over ReLU"
    Notice the activation: `nn.Mish()`, not `nn.ReLU()`. Smooth activations matter more for diffusion models than for classifiers — the denoiser is repeatedly composed with itself at inference, so kinks in the activation function show up as visible artefacts in the sampled actions. Mish, GELU, and SiLU all work. Vanilla ReLU produces visibly worse samples.

!!! warning "Normalise your actions to [-1, 1]"
    The noise schedule above assumes data with roughly unit variance. If your Godot env outputs actions in `[-100, 100]` (e.g. raw joint torques), the forward process will not destroy the signal at `t=T` and training will fail silently. Always normalise action and observation channels into a comparable range before feeding the diffusion model.

---

## 3 · Action chunking

The second key contribution of Chi et al. (2023) is **action chunking**: predict `K` consecutive future actions in one shot instead of just the next one.

In Gaussian policy land we have always treated each timestep as a fresh decision: observe `s_t`, sample `a_t`, repeat. That is fine when the policy is fast and stateless, but it has a cost: consecutive actions are sampled independently, which produces visible jitter in the control signal. On a robot arm or hovercraft you can see it — the joints twitch at high frequency even when the task does not require it.

Action chunking changes the policy interface:

```python
# Old (Gaussian / SAC / PPO):
#   action = policy(obs)            # shape (act_dim,)
#   env.step(action)
#
# New (Diffusion Policy with chunking K=8):
#   actions = policy(obs)           # shape (K, act_dim) — joint sample
#   for k in range(K):
#       env.step(actions[k])        # execute the chunk open-loop
#   # then re-plan with a fresh obs
```

Three benefits fall out immediately:

1. **Temporal coherence.** The `K` actions are sampled *jointly* from the diffusion model, so they are mutually consistent by construction. No more independent-sample jitter.
2. **Implicit planning horizon.** The model has to reason about the next `K` steps to produce a coherent chunk, so it builds an internal predictive horizon for free.
3. **Smoother control signals.** This matters for any physical or pseudo-physical actuator. Robot joints, hovercraft thrusters, and Godot's `CharacterBody3D` all hate high-frequency action noise.

The cost is **slower adaptation to sudden changes**. If something unexpected happens at step `k=3` of an 8-step chunk, the agent will continue executing the stale chunk until `k=8`. There are mitigations — receding-horizon execution (only execute the first `K_exec < K` actions, then re-plan) is the common one in the original paper.

| `K` (chunk size) | Jitter | Reactivity | Inference frequency |
|------------------|--------|------------|---------------------|
| 1 | High (independent samples) | Maximum | Every step |
| 4 | Low | Good | Every 4 steps |
| 8 | Very low (paper default) | Moderate | Every 8 steps |
| 16 | Extremely low | Poor (looks pre-recorded) | Every 16 steps |

To wire chunking into the architecture above, change `act_dim` to `K * act_dim` and reshape the output of `sample()` to `(B, K, act_dim)`. Everything else stays the same.

---

## 4 · Inference cost and DDIM speedup

DDPM as published uses `T=100` (or even `T=1000`) denoising steps. That is 100 forward passes through the denoiser per action. At ~0.16 ms per forward pass on an RTX 3060, the full sampling loop costs ~16 ms — exactly your entire 60 Hz budget, with nothing left for the Godot physics tick. Unworkable.

**DDIM (Song et al., 2021)** is the standard fix. It reinterprets the reverse process as a non-Markovian deterministic update, which allows you to skip timesteps without retraining the model. You train once with `T=100`, then at inference you sample on a sparse subsequence of, say, 10 timesteps.

```
# DDPM reverse step (stochastic):
#   x_{t-1} = mu_theta(x_t, t) + sigma_t * z     (z ~ N(0, I))
#   must visit every t in {T-1, T-2, ..., 1, 0}
#
# DDIM reverse step (deterministic, can skip):
#   pred_x0 = (x_t - sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)
#   x_{t_next} = sqrt(alpha_bar_{t_next}) * pred_x0
#              + sqrt(1 - alpha_bar_{t_next}) * eps_theta
#   can visit any sparse subsequence {tau_0 > tau_1 > ... > tau_S}
```

The `sample()` method in Section 2 already uses DDIM — that is what the `n_ddim_steps` argument does. The practical guidance below is what you actually care about as a course participant deploying this in Godot.

| `T` (denoising steps) | Latency (RTX 3060) | Action quality | Verdict for 60 Hz Godot |
|-----------------------|--------------------|----------------|--------------------------|
| 100 (full DDPM)       | ~16 ms             | Excellent      | Misses the frame budget |
| 50 (DDIM)             | ~8 ms              | Excellent      | Half the budget — risky with chunking |
| 20 (DDIM)             | ~3 ms              | Very good      | Safe, the recommended default |
| 10 (DDIM)             | ~1.5 ms            | Good           | Plenty of headroom |
| 5  (DDIM)             | ~0.8 ms            | Acceptable on smooth tasks | Use for dense control loops |
| 1  (consistency model)| ~0.2 ms            | Task-dependent | Requires distillation, advanced |

**Rule of thumb for this course.** Start with `n_ddim_steps=20`. If the policy is too slow for your env (check the `policy_time_ms` log line your inference server prints), drop to 10. If you see action quality degrade, go back up. The 60 Hz Godot budget is 16 ms total; you want diffusion to consume at most a third of it so the physics tick, the env step, and the rendering each get a fair share.

!!! tip "Action chunking and DDIM compound"
    A chunk of `K=8` actions at `T=10` denoising steps means **one DDIM call per 8 environment steps**. Your effective per-step inference cost is `1.5 / 8 ≈ 0.2 ms` — cheaper than a Gaussian policy in many cases, despite the iterative sampling.

---

## 5 · Comparison with SAC and PPO

Diffusion Policy is not a strict upgrade. It changes the rules of the game enough that the right comparison is task-by-task. The table below summarises the trade-offs you should weigh.

|                          | PPO                          | SAC                                  | Diffusion Policy                          |
|--------------------------|------------------------------|--------------------------------------|--------------------------------------------|
| Policy distribution      | Gaussian                     | Squashed Gaussian (tanh)             | Arbitrary (multimodal, any shape)          |
| Training paradigm        | On-policy RL                 | Off-policy RL                        | Supervised (offline behaviour cloning)     |
| Data requirement         | Online rollouts              | Online rollouts + replay buffer      | Offline dataset of demonstrations          |
| Inference cost           | O(1) forward pass            | O(1) forward pass                    | O(T_ddim) forward passes (T_ddim = 5–20)   |
| Sample efficiency        | Medium                       | High                                 | N/A — depends on demonstration quality     |
| Captures multiple modes  | No                           | No (single Gaussian)                 | Yes (the whole point)                      |
| Handles discrete actions | Yes                          | Discrete SAC variant exists          | Awkward — needs softmax + Gumbel tricks    |
| Training stability       | Sensitive to hyperparameters | More stable than PPO once tuned      | Very stable (just supervised regression)   |
| Best for                 | General RL on game envs      | Continuous control, sample-efficient | Manipulation, multimodal tasks, demos      |
| Godot real-time?         | Yes                          | Yes                                  | Yes, with DDIM `T <= 10` and chunking      |

The headline distinction: **PPO and SAC learn from reward; Diffusion Policy learns from demonstrations.** That is not a small difference. If you do not have demonstrations and cannot collect them, Diffusion Policy is not a viable option — it is a behaviour-cloning method, not an RL method.

There is active research combining diffusion policies with reward signals (Q-Score Matching, IDQL, Diffusion-QL) so that you can fine-tune a diffusion BC policy with RL. Those methods are out of scope here; you would build on top of `unit-offline-rl` to attempt one.

---

## 6 · Training a Diffusion Policy on Godot demonstrations

End-to-end workflow assuming you have completed `unit-09` (imitation learning) and have a working demonstration pipeline.

**Step 1 — Collect expert demonstrations.** Either replay a trained PPO/SAC agent and log `(obs, action)` pairs, or use the keyboard-driven `HumanController` from Unit 9 to record human demos. Save as a HDF5 file with two arrays of shape `(N, obs_dim)` and `(N, act_dim)`. Aim for 50k–500k transitions for non-trivial tasks.

```python
# replay_pretrained.py — collect demos from a trained PPO agent
import h5py
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="builds/PickAndPlace.x86_64", n_parallel=1)
agent = PPO.load("checkpoints/pickandplace_ppo_5M.zip")

obs_log, act_log = [], []
obs, _ = env.reset()
for _ in range(100_000):
    action, _ = agent.predict(obs, deterministic=False)
    obs_log.append(obs.copy()); act_log.append(action.copy())
    obs, _, term, trunc, _ = env.step(action)
    if term.any() or trunc.any():
        obs, _ = env.reset()

with h5py.File("demos.h5", "w") as f:
    f.create_dataset("obs", data=np.array(obs_log))
    f.create_dataset("action", data=np.array(act_log))
```

**Step 2 — Train Diffusion Policy with supervised regression.** No reward, no value function, no replay buffer — just minimise the noise-prediction loss from Section 2.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

obs = torch.tensor(np.array(obs_log), dtype=torch.float32)
act = torch.tensor(np.array(act_log), dtype=torch.float32)
loader = DataLoader(TensorDataset(obs, act), batch_size=256, shuffle=True)

model = DiffusionMLP(obs_dim=obs.shape[1], act_dim=act.shape[1]).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)

for epoch in range(200):
    for o, a in loader:
        loss = model.loss(a.cuda(), o.cuda())
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch {epoch}: noise-MSE = {loss.item():.4f}")
```

Training is fast — usually under an hour on a single GPU for the dataset sizes above. The loss curve should drop smoothly. There are no exploration/exploitation dynamics to worry about, because there is no exploration: you are just fitting a conditional density.

**Step 3 — Deploy in Godot.** This is where Diffusion Policy is awkward compared to PPO/SAC. The denoising loop is a Python `for` loop, and ONNX (which `godot-rl-agents` uses for in-process inference) cannot trace Python control flow directly.

Three options, in increasing order of complexity:

a. **Export only the single-step denoiser to ONNX, run the loop in GDScript.** Smallest export, but you have to re-implement the DDIM update rule in GDScript — error-prone.

b. **TorchScript-compile the full `sample()` method and run it from a Python inference server.** Cleanest option. The Godot env talks to the server via the existing `godot_rl` socket protocol.

c. **Use ONNX loop operators.** Technically possible, very fragile across runtimes. Not recommended for this course.

The recommended path is (b):

```python
# export_diffusion_policy.py
import torch

model = DiffusionMLP(obs_dim=24, act_dim=7).cuda().eval()
model.load_state_dict(torch.load("diffusion_policy.pt"))

# TorchScript can scriptify the full sampling loop including the for-loop.
scripted = torch.jit.script(model)
torch.jit.save(scripted, "diffusion_policy_scripted.pt")

# Inference server reloads and serves predictions:
#   scripted = torch.jit.load("diffusion_policy_scripted.pt").cuda().eval()
#   action = scripted.sample(obs_tensor, n_ddim_steps=10)
# Godot connects via the standard godot_rl tcp socket.
```

!!! warning "Watch the action normalisation at deploy time"
    The diffusion model was trained on actions in `[-1, 1]`. Your Godot env almost certainly expects actions in some other range (joint angles in radians, thruster forces in newtons, etc.). The inference server is responsible for the inverse normalisation: `env_action = unnormalise(sampled_action)`. Forgetting this is the single most common deployment bug.

---

## 7 · When diffusion wins vs when to stick with SAC

The question you will face the moment you finish this unit is: *should I rewrite my project to use Diffusion Policy?* The honest answer is "probably not, but here is when you should."

**Use Diffusion Policy when:**

- Your demonstrations contain **multiple valid strategies** for the same situation. The Gaussian collapse problem is real and Diffusion Policy directly solves it.
- You are doing **manipulation or dexterous control** where smoothness of the action signal matters. Action chunking gives you free temporal coherence.
- You are **learning from human demonstrations** (rather than from reward). Humans are inherently multimodal — different sessions, different days, different moods. Diffusion Policy can absorb that variability.
- You have a **fixed offline dataset** and no online environment access. Pure BC on a Gaussian policy is mode-averaging; Diffusion Policy is not.

**Stick with SAC (or PPO) when:**

- You are doing **online RL** and can collect fresh experience. Diffusion Policy is a BC method; it does not optimise reward directly.
- The action distribution at every state is genuinely **unimodal**. A 1-D throttle, a steering angle on a smooth track, most game-like envs — a Gaussian is fine and 100× faster.
- Your **latency budget is below ~3 ms per action** and DDIM with `T=5` still does not fit. Some high-frequency control loops (haptic devices, drone autopilots) live in this regime.
- You need **discrete actions**. Diffusion Policy for discrete outputs requires categorical-diffusion or Gumbel tricks and is rarely worth it.
- You need **strong sample efficiency from a small online budget** (the SAC sweet spot).

A pragmatic mixed strategy that is increasingly common in robotics: train a Diffusion Policy on demonstrations to bootstrap, then fine-tune with a small amount of online RL (Diffusion-QL / IDQL) to optimise reward beyond the demonstrator's level.

---

## 8 · Stretch goals

These exercises are designed to make the abstract claims of this unit concrete.

1. **Build the bimodal toy task.** Make a 2-D point-mass env in Godot with two valid goals (`x=+1` and `x=-1`), both giving identical reward when reached. Collect 1k demonstrations split 50/50 between the goals. Train SAC and a Diffusion Policy on the same data. Visualise the action distribution at the start state by sampling 1000 actions from each. SAC should produce a tight blob around `x=0`; Diffusion Policy should produce two well-separated clusters.

2. **Keyboard-demo a manipulation task.** Modify a `Reacher` or `PickAndPlace` env's `AIController` to also accept keyboard input as the action source, and log human demos as you play. Collect ~1000 successful episodes. Train Behaviour Cloning (Gaussian regression), and Diffusion Policy, on the same data. Compare success rate, smoothness (variance of action differences), and the visual quality of the resulting motion in the Godot viewer.

3. **DDIM step-count ablation.** Take one trained Diffusion Policy and evaluate it at `T_ddim ∈ {5, 10, 20, 50, 100}`. Plot task success rate and per-step inference latency on the same chart. The curve should rise steeply from `T=5` to `T=10`, plateau by `T=20`, and inference cost should scale linearly. Use the plot to choose the operating point for your project.

4. **Action chunking ablation on locomotion.** Train Diffusion Policy with chunk sizes `K ∈ {1, 4, 8, 16}` on a quadruped locomotion task from `unit-locomotion`. Measure (a) episode return, (b) action smoothness (RMS of consecutive-action differences), and (c) recovery time from a push perturbation. You should see smoothness improve monotonically with `K` while recovery time gets worse.

5. **(Advanced) Score visualisation.** For a 2-D action env, plot the learned score field `eps_theta(x, t=0, obs=fixed)` as a vector field on the action plane. Overlay the demonstration data. The vectors should point from low-density regions toward the data manifold. This is the most direct visual confirmation that diffusion models are estimating gradients of log-density.

---

## What's next

You now have a way to represent action distributions that no Gaussian could. Diffusion Policy is the third pillar of modern continuous control alongside SAC and PPO — the right tool when your data is multimodal and you have demonstrations rather than a reward signal.

The next places to take this:

- **Combining diffusion with reward.** `unit-offline-rl` covers the offline RL methods (CQL, IQL) that allow Q-learning on fixed datasets. The diffusion-RL hybrids (Diffusion-QL, IDQL) layer Q-learning on top of a diffusion BC backbone — read those papers next.
- **Closing the demo loop.** `unit-curiosity` and `unit-hierarchical` give you tools for exploring without a reward signal — useful when you want to generate diverse demonstrations programmatically rather than gathering them from humans.
- **Deployment hardening.** `unit-sim-to-real` discusses the domain gaps that matter once you move a policy off the simulator. Diffusion policies inherit all of those problems and add their own (action distribution shift if obs normalisation drifts).

[← SAC](unit-sac.md) · [Course home](index.md)
