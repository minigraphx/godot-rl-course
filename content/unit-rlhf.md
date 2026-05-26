# RLHF — Learning Rewards from Human Preferences

[← Imitation Learning](unit-09.md) · [Course home](index.md)

---

!!! info "Three ways to see your AI"
    - **Godot side-by-side viewer** — the only place where the *real* signal lives. Two recorded trajectories play next to each other; a designer clicks the one that looks better. Every other metric in this unit is a proxy for these clicks.
    - **TensorBoard** — `reward_model/val_accuracy` should climb above 70% before you trust the model; `policy/kl_to_ref` should *stay bounded* during PPO fine-tuning. A KL spike is the visible footprint of reward hacking.
    - **Reward-model probe script** — score 100 hand-picked trajectories with the trained reward model and rank them. If the ranking does not match a designer's gut ranking, the model is overfit to surface features and PPO will exploit them.

---

## 0 · Why RLHF matters (and why you should care for games)

The insight that made ChatGPT possible is small and brutal: **you cannot write a reward function for "write a helpful response."** No designer in the world can articulate, in closed-form Python, what makes a paragraph helpful. Yet a human reading two candidate paragraphs can pick the better one in under a second.

RLHF — Reinforcement Learning from Human Feedback — turns that asymmetry into a training signal. Humans compare pairs of outputs. A neural network learns to predict their preferences. That network *becomes* the reward function. PPO (which you already know from [unit-ppo-deep.md](unit-ppo-deep.md)) optimises the policy against it.

The same trick works for game AI. You spent an entire unit ([unit-reward-engineering.md](unit-reward-engineering.md)) hand-crafting reward terms for distance, velocity, energy, and survival — and even then, the resulting NPC may move "correctly" without ever feeling *natural*. Naturalness, charm, menace, fairness — none of these are functions of `position.xyz`. They are functions of a designer's taste.

This unit gives you the machinery to optimise for taste.

!!! tip "The alignment connection (revisited)"
    [Unit 9](unit-09.md) flagged the parallel between behavioural cloning and supervised fine-tuning (SFT). RLHF is the next rung. SFT teaches the model *what to say*. RLHF teaches it *which of two things to say is better*. In a game, SFT-style BC teaches an NPC to copy a recorded patrol; RLHF teaches it that *this* patrol felt more alive than *that* one.

---

## 1 · The reward function problem at scale

You already know the reward is the only signal the agent gets from the world ([unit-reward-engineering.md, §1](unit-reward-engineering.md)). You know about sparse vs dense rewards, potential-based shaping, and the canonical reward-hacking failure modes — boat agents circling fuel pickups, walkers exploiting physics glitches.

Reward engineering works **as long as the goal can be reduced to a measurable physical quantity**. You can write rewards for:

- Distance to a target
- Velocity in a chosen direction
- Energy consumption
- Time to completion
- Collision force

You cannot write rewards for:

- *"This NPC feels alive."*
- *"This boss fight is fun."*
- *"This dialogue choice fits the character."*
- *"This animation looks natural."*
- *"This patrol pattern feels purposeful, not robotic."*

The boundary is not technical. It is **definitional**. There is no `is_natural(state) -> float` function because there is no agreed-upon mapping from world-state to naturalness. Naturalness lives in a human's head.

### Why ratings don't work and comparisons do

A naive workaround: ask designers to *rate* trajectories on a 1–10 scale. Then regress the reward model on those scalar labels.

This fails for well-documented reasons:

1. **Inter-annotator drift.** Alice's "7" is Bob's "5." Without anchoring, the scale is meaningless.
2. **Intra-annotator drift.** Alice's "7" on Monday is her "5" on Friday after lunch.
3. **Anchor sensitivity.** Whatever Alice rated first becomes her unconscious reference point.
4. **Granularity is fake.** The difference between "7.3" and "7.4" is noise. The difference between "this one" and "that one" is a real preference.

Pairwise comparisons sidestep all four. A human shown two trajectories and asked "which is better?" gives a robust, low-variance signal — even across annotators, even across days. This is a known result from psychometrics (Thurstone, 1927) and from the LLM training literature (Christiano et al., 2017; Ouyang et al., 2022).

| Feedback format | Cognitive load | Inter-rater agreement | Information per click |
|---|---|---|---|
| Scalar rating (1–10) | High | Low (~40%) | High (in theory) |
| Pairwise comparison | Low | High (~80%) | 1 bit (in theory) |
| Ranking of 4 | Medium | Medium | ~5 bits |

Pairwise wins because the information-per-click is honest. A click is a click. The rating "7.3" is mostly noise dressed as signal.

---

## 2 · Preference data collection

The RLHF training loop is itself a loop of three loops:

```
┌─────────────────────────────────────────────────────────┐
│  Outer loop: alternate data collection ↔ training       │
│                                                          │
│   1. Run current policy π in Godot → trajectories       │
│   2. Sample pairs (τ_A, τ_B) for designer to compare    │
│   3. Designer clicks preferred → preference dataset D   │
│   4. Train reward model r_φ on D                        │
│   5. Fine-tune π with PPO against r_φ (+ KL penalty)    │
│   6. Go to 1                                            │
└─────────────────────────────────────────────────────────┘
```

Each iteration shifts the policy. New trajectories cover new regions of state space. New preferences correct the reward model where it was wrong. The system converges when the designer can no longer reliably tell two trajectories apart.

### The Bradley–Terry model

The mathematical backbone is the **Bradley–Terry model** (1952). Assume each trajectory τ has an underlying scalar reward `r(τ)`. The probability a human prefers `τ_A` over `τ_B` is:

$$P(\tau_A \succ \tau_B) = \sigma\left(r(\tau_A) - r(\tau_B)\right) = \frac{1}{1 + e^{-(r(\tau_A) - r(\tau_B))}}$$

Three properties make this useful:

1. **Invariance to additive constants.** `r` and `r + c` produce identical preferences. We only ever learn rewards *up to a constant*.
2. **Logistic shape.** Small reward gaps → 50/50 preference (humans see a coin flip). Large gaps → near-certain preference.
3. **Maximum-likelihood objective is convex** in `r(τ_A) - r(τ_B)`, so training is stable.

The training objective for a reward model `r_φ` over a preference dataset `D = {(τ_w, τ_l)}` (w = winner, l = loser):

$$\mathcal{L}(\varphi) = -\mathbb{E}_{(\tau_w, \tau_l) \sim D}\left[ \log \sigma(r_\varphi(\tau_w) - r_\varphi(\tau_l)) \right]$$

This is just binary cross-entropy on a difference of scores. The same loss you'd write for any pairwise ranking problem.

### How many comparisons do you need?

| Task complexity | Comparisons needed | Notes |
|---|---|---|
| Toy (CartPole "smoothness") | 200–500 | A single designer in one afternoon |
| Single-NPC behaviour | 2k–10k | Several designers across a week |
| Full character (multiple behaviours) | 10k–50k | Crowdsource or use active learning |
| LLM-scale (ChatGPT-class) | 50k–500k+ | Industrial annotation operation |

The Christiano et al. (2017) Atari result used ~5500 human comparisons to reach human-level on Pong. For game AI you should plan on **at least 1k pairs per behaviour you want to shape**.

### Active learning: query what you're uncertain about

Random sampling of trajectory pairs wastes designer time. If `r_φ(τ_A) - r_φ(τ_B) = 8.0`, the model is already certain. The click confirms what it knows and contributes almost no gradient.

The fix: prefer pairs where `|r_φ(τ_A) - r_φ(τ_B)|` is **small**, or where an ensemble of reward models *disagrees*. Concretely:

```python
def select_pair_for_annotation(reward_models, trajectory_pool):
    """Return the pair the ensemble is most uncertain about."""
    best_pair = None
    best_disagreement = -float("inf")
    for tau_a, tau_b in sample_pairs(trajectory_pool, n=200):
        # Score under each ensemble member
        diffs = [rm(tau_a).sum() - rm(tau_b).sum() for rm in reward_models]
        disagreement = float(np.std(diffs))
        if disagreement > best_disagreement:
            best_disagreement = disagreement
            best_pair = (tau_a, tau_b)
    return best_pair
```

Active learning typically cuts the comparisons-per-quality-unit by 3–5×. It is the single biggest practical improvement to RLHF data pipelines.

### A practical annotation interface

The interface a designer uses matters more than any hyperparameter. The canonical layout is dead simple:

```
┌─────────────────────────────┬─────────────────────────────┐
│       Trajectory A          │       Trajectory B          │
│  (Godot recording, loop)    │  (Godot recording, loop)    │
│                             │                             │
│   [▶ Play]  [⏸ Pause]      │   [▶ Play]  [⏸ Pause]      │
└─────────────────────────────┴─────────────────────────────┘
       [  A is better  ]     [  Tie  ]     [  B is better  ]
                      [ Skip / Can't tell ]
```

Three rules from production RLHF pipelines:

- **Hide all numeric scores.** If the designer sees `r_φ`, they anchor on it.
- **Randomise A/B order per pair.** Otherwise designers default-click "A."
- **Allow "tie" and "skip."** Forcing a choice on indistinguishable pairs injects label noise.

---

## 3 · Training the reward model

Architecturally, the reward model is the same observation encoder as your policy, with a scalar head instead of an action head.

```
obs ──► [encoder: same conv/MLP as policy] ──► scalar r_φ(obs)
```

For a trajectory `τ = (o_0, o_1, ..., o_T)`, the trajectory-level reward is the **sum** of per-step rewards:

$$r_\varphi(\tau) = \sum_{t=0}^{T} r_\varphi(o_t)$$

The Bradley–Terry loss operates on these sums.

### Complete PyTorch implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader


class RewardModel(nn.Module):
    """Per-timestep reward predictor. Same architecture as the policy encoder."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, T, obs_dim) -> (batch, T)
        return self.net(obs).squeeze(-1)


def preference_loss(reward_model, obs_chosen, obs_rejected):
    """Bradley-Terry loss on paired trajectory observations."""
    r_chosen = reward_model(obs_chosen).sum(dim=1)     # sum over timesteps
    r_rejected = reward_model(obs_rejected).sum(dim=1)
    # Numerically stable log-sigmoid: -softplus(-(r_w - r_l))
    return torch.nn.functional.softplus(-(r_chosen - r_rejected)).mean()


class PreferenceDataset(Dataset):
    def __init__(self, chosen_obs, rejected_obs):
        self.chosen = torch.as_tensor(chosen_obs, dtype=torch.float32)
        self.rejected = torch.as_tensor(rejected_obs, dtype=torch.float32)

    def __len__(self):
        return len(self.chosen)

    def __getitem__(self, idx):
        return self.chosen[idx], self.rejected[idx]


def train_reward_model(chosen, rejected, obs_dim, epochs=20, lr=3e-4, batch_size=64):
    """Train a reward model on preference pairs and report validation accuracy."""
    n = len(chosen)
    split = int(0.9 * n)
    train_ds = PreferenceDataset(chosen[:split], rejected[:split])
    val_ds = PreferenceDataset(chosen[split:], rejected[split:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = RewardModel(obs_dim)
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for c, r in train_loader:
            loss = preference_loss(model, c, r)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Validation: how often does the model agree with the human ranking?
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for c, r in val_loader:
                r_c = model(c).sum(dim=1)
                r_r = model(r).sum(dim=1)
                correct += int((r_c > r_r).sum())
                total += len(c)
        print(f"epoch {epoch:02d}  val_acc={correct / max(total, 1):.3f}")

    return model
```

### Reading the validation accuracy

The single most important metric for a reward model is **held-out preference accuracy**: on pairs the model has never seen, how often does it agree with the human?

| Val accuracy | What it means | What to do |
|---|---|---|
| ~50% | Coin flip — no signal | Collect more data, check labels for noise |
| 60–70% | Weak signal | Acceptable for early iteration; expect noisy PPO |
| 70–85% | Healthy signal | Proceed to PPO fine-tuning |
| 85–95% | Strong signal | Excellent; check for label leakage |
| >95% | Suspicious | Almost certainly leakage — same trajectories in train and val |

Note the cap: the **human Bayes error** is rarely above 90–95%. Two designers disagree on close calls roughly 5–15% of the time. A reward model that exceeds the inter-annotator agreement rate has memorised something it shouldn't.

!!! warning "Train/val splits must be by *trajectory*, not by *pair*"
    If trajectory `τ_42` appears in both training pairs and validation pairs, the model trivially memorises its score. Split by trajectory ID first, then form pairs within each split.

---

## 4 · PPO with a learned reward (the classic RLHF pipeline)

Once you have a trained reward model `r_φ`, you fine-tune your policy with PPO — exactly the PPO you wrote in [unit-ppo-deep.md](unit-ppo-deep.md) — but using `r_φ(obs)` as the per-step reward instead of the environment's reward.

Implementation-wise, this is a **wrapper around the Godot environment**, structurally identical to the `RNDGodotEnv` wrapper from [unit-curiosity.md](unit-curiosity.md):

```python
import gymnasium as gym
import numpy as np
import torch


class RLHFGodotEnv(gym.Wrapper):
    """Wrap a Godot env so its rewards come from a learned reward model
    minus a KL penalty against a frozen reference policy."""

    def __init__(self, env, reward_model, ref_policy, current_policy, kl_coef=0.05):
        super().__init__(env)
        self.reward_model = reward_model.eval()
        self.ref_policy = ref_policy.eval()       # frozen snapshot from before RLHF
        self.current_policy = current_policy      # updated by PPO
        self.kl_coef = kl_coef

    def step(self, action):
        obs, _r_env, terminated, truncated, info = self.env.step(action)

        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            r_rm = float(self.reward_model(obs_t).squeeze())

            # Per-step KL between current and reference policy at this obs
            logp_cur = self.current_policy.log_prob(obs_t, action)
            logp_ref = self.ref_policy.log_prob(obs_t, action)
            kl = float(logp_cur - logp_ref)

        r_total = r_rm - self.kl_coef * kl
        info["r_rm"] = r_rm
        info["kl_to_ref"] = kl
        return obs, r_total, terminated, truncated, info
```

You then hand this wrapped env to SB3's PPO exactly as in earlier units:

```python
from stable_baselines3 import PPO

env = RLHFGodotEnv(make_godot_env("NPCGuard.x86_64"),
                   reward_model, ref_policy, current_policy,
                   kl_coef=0.05)
model = PPO("MlpPolicy", env, n_steps=2048, batch_size=64,
            learning_rate=3e-4, verbose=1, tensorboard_log="./rlhf_tb/")
model.learn(total_timesteps=500_000)
```

### Why the KL penalty is essential

Without the KL term, PPO will discover that the reward model has loopholes. It is a finite neural network trained on finite data. Anywhere the model's score doesn't match a designer's true taste, PPO will find that anywhere and exploit it.

The KL penalty `β · KL(π || π_ref)` says: "stay close to the reference policy you started from." The reference policy was trained with reward engineering or imitation learning — it is *roughly* sensible. The KL penalty bounds how far PPO can wander from that sensible region in pursuit of reward-model loopholes.

The full RLHF reward is:

$$r_{\text{total}}(o_t, a_t) = r_\varphi(o_t) - \beta \cdot \text{KL}\bigl(\pi(\cdot \mid o_t)\,\|\,\pi_{\text{ref}}(\cdot \mid o_t)\bigr)$$

Picking `β`:

| β value | Effect | When to use |
|---|---|---|
| 0.0 | No anchor — instant reward hacking | Almost never |
| 0.01 | Loose anchor — policy explores | When ref policy is mediocre |
| 0.05–0.1 | Standard range | Default starting point |
| 0.5 | Tight anchor — barely moves | When ref policy is already very good |

A practical adaptive scheme (Stiennon et al., 2020): target a fixed KL budget (say, 10 nats per episode). If realised KL is above target, raise `β`. Below target, lower `β`. SB3 doesn't ship this out of the box — write it as a callback.

---

## 5 · DPO — Direct Preference Optimization

In 2023, Rafailov et al. published a startling result: **you can skip the reward model entirely**. Their algorithm, DPO (Direct Preference Optimization), reformulates RLHF as a single supervised loss directly on preference data.

### The key insight

Under the RLHF objective with a KL penalty, the optimal policy has a closed-form relationship with the reward model:

$$\pi^*(a \mid s) = \frac{1}{Z(s)} \pi_{\text{ref}}(a \mid s) \exp\!\left(\tfrac{1}{\beta} r(s, a)\right)$$

Solving for `r`:

$$r(s, a) = \beta \log \frac{\pi^*(a \mid s)}{\pi_{\text{ref}}(a \mid s)} + \beta \log Z(s)$$

Plugging this expression for `r` back into the Bradley–Terry preference loss, the `Z(s)` terms cancel between the chosen and rejected trajectory. The result is a loss expressed entirely in terms of `π_θ` and `π_ref` — no separate reward model anywhere:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(\tau_w, \tau_l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(\tau_w)}{\pi_{\text{ref}}(\tau_w)} - \beta \log \frac{\pi_\theta(\tau_l)}{\pi_{\text{ref}}(\tau_l)}\right)\right]$$

This is *supervised learning*. There is no environment rollout, no PPO, no reward model training. Just gradient descent on preference pairs.

### Minimal DPO implementation

```python
def dpo_loss(policy, ref_policy, obs_chosen, act_chosen, obs_rejected, act_rejected, beta=0.1):
    """DPO loss. policy is trainable; ref_policy is frozen."""
    # Sum log-probs over the trajectory
    logp_w = policy.log_prob(obs_chosen, act_chosen).sum(dim=1)
    logp_l = policy.log_prob(obs_rejected, act_rejected).sum(dim=1)

    with torch.no_grad():
        logp_w_ref = ref_policy.log_prob(obs_chosen, act_chosen).sum(dim=1)
        logp_l_ref = ref_policy.log_prob(obs_rejected, act_rejected).sum(dim=1)

    logits = beta * ((logp_w - logp_w_ref) - (logp_l - logp_l_ref))
    return torch.nn.functional.softplus(-logits).mean()
```

That is the entire algorithm. Train your policy by minimising this loss over your preference dataset. No reward model. No environment.

### RLHF vs DPO

| | RLHF + PPO | DPO |
|--|-----------|-----|
| Requires reward model? | Yes | No |
| Online RL training? | Yes | No (supervised) |
| Preference data needed | Yes | Yes |
| Sample efficiency | Medium | High (no RL loop) |
| Reward hacking risk | Yes (mitigated by KL) | No |
| Can keep learning from new data? | Yes (re-collect, re-train) | Yes (extend dataset) |
| Memory cost | 2 networks (policy + RM) | 2 networks (policy + ref) |
| Compute cost | High (rollouts + PPO + RM) | Low (one supervised pass) |
| Best for | Continuous online improvement | Fixed preference dataset |
| Hyperparameter pain | High (β, lr, n_steps, etc.) | Low (β, lr) |

DPO is the right starting point for most game projects. If your preference dataset is fixed and your reference policy is sensible, DPO will get you 80% of RLHF's quality with 20% of the engineering work. Reach for full PPO-based RLHF only when you need to keep generating new trajectories and collecting fresh preferences in a continuous loop.

!!! tip "DPO inherits the ref-policy quality"
    DPO can only move your policy *near* `π_ref`. If `π_ref` is bad (e.g. a random initialisation), DPO produces a slightly-better-than-bad policy. Always seed DPO with a reasonable BC or PPO checkpoint, never with random weights.

---

## 6 · Reward hacking

You met reward hacking in [unit-reward-engineering.md, §7](unit-reward-engineering.md). RLHF gives the phenomenon a new face: the **reward model itself** is the thing being hacked. The policy discovers states the model assigns high scores to, even though no designer would.

### Concrete failure modes

**Language-model classic.** Early RLHF chatbots learned to begin every response with *"As an AI assistant..."* because that string correlated with high preference scores in the training data (annotators favoured well-formatted responses). The model then prefixed even casual replies with the phrase. Reward model said: helpful. Humans said: annoying.

**Godot example.** Train an NPC guard with RLHF. Your reward model learned that designers prefer trajectories where the guard *faces* the patrol route at intersections. PPO discovers the guard can stand motionless at an intersection while continuously rotating to face the route. The reward model loves it. The designer watching the level says it looks broken.

**Visual policy hack.** A vision-based RLHF agent learned that the reward model assigned high scores to states with a particular lighting condition. The policy positioned itself to maximise that lighting condition rather than complete the task. The reward model had absorbed a spurious correlation in the preference data.

### Detection

| Signal | What it indicates |
|---|---|
| `policy/kl_to_ref` spikes during PPO | Policy has moved into a reward-model loophole |
| Reward-model score climbs while visual quality degrades | Classic Goodhart's Law |
| Ensemble disagreement (`std(r_φ)`) explodes on policy rollouts | Models agree on training data, disagree on novel exploits |
| Inverse correlation between `r_φ` and a held-out scalar metric | The model is pursuing a proxy |

The *operational* signal is `kl_to_ref` plus a recurring qualitative review. Set up the training callback to dump a viz checkpoint every 50k steps. If `kl_to_ref` is above a threshold *and* the viz looks weird, stop training.

### Mitigation

1. **KL penalty `β` ↑.** The cheapest fix. Tighter anchor → less room to exploit.
2. **Reward-model ensembles.** Train N reward models with different seeds. Use the **minimum** (pessimistic) or the **mean minus k·std** (uncertainty-penalised). The loopholes a single model finds are not the loopholes another model finds; the intersection is much smaller than the union.
3. **Refresh the preference data.** As the policy shifts, its trajectories drift out of the reward model's training distribution. Periodically pull fresh trajectories from the current policy and collect new preferences on those.
4. **Early stopping.** Don't train RLHF to convergence on the *reward-model loss*. Stop when the **qualitative** signal plateaus.
5. **Adversarial review.** Once a week, a designer specifically tries to find behaviour the reward model rates highly that they hate. Those become hard negatives in the next preference batch.

!!! warning "RLHF does not remove reward hacking — it relocates it"
    With hand-engineered rewards, the policy hacks the reward function. With RLHF, the policy hacks the reward *model*. The new hacking is harder to predict in advance (you cannot read the reward model the way you can read a GDScript reward) but easier to detect at runtime (it shows up as a KL spike).

---

## 7 · Godot example — NPC behaviour preferences

Concrete walkthrough. The task: a castle guard NPC that should patrol *naturally*. Hand-engineered reward terms (distance covered, time on patrol path, energy used) give a guard that technically patrols but feels robotic. RLHF will fix it.

### Step 1 — Bootstrap with vanilla PPO

Use the reward engineering from earlier units to get a baseline policy:

```python
# bootstrap_guard.py
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="./NPCGuard.x86_64", n_parallel=8, speedup=8)
model = PPO("MlpPolicy", env, n_steps=2048, batch_size=128, verbose=1)
model.learn(total_timesteps=2_000_000)
model.save("guard_baseline.zip")   # this becomes π_ref
```

The baseline guard walks the route. It also occasionally stutters, picks suboptimal corners, and looks like an RL agent. Good enough to be a reference policy.

### Step 2 — Record trajectory pairs

We need diverse trajectories to compare. Generate them by varying seeds and hyperparameters:

```python
# record_trajectories.py
import numpy as np
from stable_baselines3 import PPO

def record_trajectory(model_path, env, n_steps=400, seed=0):
    model = PPO.load(model_path)
    obs, _ = env.reset(seed=seed)
    obs_log, act_log, frame_log = [], [], []
    for _ in range(n_steps):
        act, _ = model.predict(obs, deterministic=False)
        obs, _, done, _, _ = env.step(act)
        obs_log.append(obs)
        act_log.append(act)
        frame_log.append(env.render(mode="rgb_array"))
        if done:
            break
    return dict(obs=np.array(obs_log),
                act=np.array(act_log),
                frames=np.array(frame_log))

# Generate 200 trajectories across different seeds and slight HP variations
trajectories = [record_trajectory("guard_baseline.zip", env, seed=s) for s in range(200)]
```

For each, save the recorded frames as an `.mp4` so the designer can watch.

### Step 3 — GDScript recorder for in-engine playback

If you'd rather record inside Godot directly (more authentic preview for the designer), drop this on your agent script:

```gdscript
extends RigidBody3D
@onready var _ai = $AIController3D

var recording: Array = []
var record_mode: bool = false

func _physics_process(_delta):
    if record_mode:
        recording.append({
            "pos": global_position,
            "rot": global_rotation,
            "vel": linear_velocity,
            "action": _ai.last_action,
        })

func dump_recording(path: String):
    var f = FileAccess.open(path, FileAccess.WRITE)
    f.store_string(JSON.stringify(recording))
    f.close()
    recording.clear()

func play_recording(path: String):
    # Replay mode: position the agent each physics tick from file
    var f = FileAccess.open(path, FileAccess.READ)
    var data = JSON.parse_string(f.get_as_text())
    for frame in data:
        global_position = frame.pos
        global_rotation = frame.rot
        await get_tree().physics_frame
```

Two of these instances side-by-side give the designer a real-time A/B preview.

### Step 4 — Collect preferences

Build the simplest annotation web page possible — two `<video>` tags, three buttons. Persist clicks to a JSON file:

```python
# preference_server.py — simplified
from fastapi import FastAPI
import json, random

app = FastAPI()
prefs = []

@app.get("/next_pair")
def next_pair():
    a, b = random.sample(range(200), 2)
    return {"a": f"/clips/{a}.mp4", "b": f"/clips/{b}.mp4",
            "id_a": a, "id_b": b}

@app.post("/submit")
def submit(id_a: int, id_b: int, choice: str):
    # choice in {"a", "b", "tie", "skip"}
    if choice in ("a", "b"):
        prefs.append({"chosen": id_a if choice == "a" else id_b,
                      "rejected": id_b if choice == "a" else id_a})
        json.dump(prefs, open("preferences.json", "w"))
    return {"ok": True}
```

Run it, ask the designers to label ~1000 pairs. Realistic time: ~3 hours of designer time spread over a week.

### Step 5 — Train reward model, then PPO

Build `chosen_obs` and `rejected_obs` arrays from `preferences.json`, hand them to `train_reward_model()` from §3. Then wrap the env with `RLHFGodotEnv` from §4 and PPO-fine-tune the bootstrap policy.

### Expected results

| Policy | Patrol completion rate | Designer rating (1–10) | Looks natural? |
|---|---|---|---|
| Random | 0% | 1.2 | No |
| Hand-engineered reward (PPO bootstrap) | 95% | 5.1 | "Robotic but functional" |
| Behavioural cloning of designer | 60% | 6.4 | "Natural but unreliable" |
| RLHF on top of bootstrap | 93% | 8.2 | "Yes" |

The RLHF policy keeps the *competence* of the hand-engineered baseline (because the KL penalty anchors it there) while gaining the *naturalness* of the human-preferred trajectories.

---

## 8 · RLHF for game AI vs. LLMs

The algorithm is the same. The domains differ in three practical ways.

| | LLM RLHF | Game-AI RLHF |
|---|---|---|
| Trajectory format | Token sequences | (obs, action) timesteps |
| "Episode" length | Tens to thousands of tokens | Hundreds to thousands of physics steps |
| Preference acquisition cost | $$$ (crowd workers, lengthy reading) | $ (designer watches 10s clip) |
| Data scale | 10k–500k pairs | 500–10k pairs |
| Reference policy | SFT model | BC or PPO bootstrap |
| Reward-hack failure mode | Sycophancy, refusals, repetition | Animation artefacts, broken motion |
| Evaluation | Held-out preference, win-rate vs baseline | Designer ratings + viz checkpoint |

### When *not* to use RLHF

If you can write a working reward function, **just write it**. RLHF is dramatically more engineering work than reward engineering. It is only worth the cost when:

1. The goal is genuinely subjective ("looks natural", "feels fair").
2. Multiple stakeholders disagree on the exact reward function but agree on preferences when shown trajectories.
3. The hand-engineered reward keeps producing exploits no matter how you patch it.
4. You have *already shipped* a baseline and want to fine-tune it on real player data.

For "minimise time to goal" or "stay upright": RLHF is overkill. Use reward engineering.

### When RLHF shines

The litmus test: *"I'll know good behaviour when I see it."* If a designer can reliably pick the better of two trajectories but cannot articulate the rule, RLHF will outperform any hand-engineered reward. This pattern recurs across animation quality, pacing, difficulty feel, character personality, level fairness, and combat readability.

---

## 9 · Stretch goals

1. **CartPole "smoothness" reward model.** Train PPO on CartPole. Record 300 trajectories. Define a synthetic "preference": trajectory A is preferred over B iff `mean(|action_t - action_{t-1}|)` is smaller in A (smoother control). Have the labeller be a Python function that simulates the human. Train a reward model on these preferences. Fine-tune CartPole with PPO using the learned reward. Compare to PPO with the hand-crafted smoothness reward `-λ · |Δa|`. Plot both learning curves. The learned reward model should match the hand-crafted reward to within ~10% sample efficiency — and the gap is your measure of how hard it is to learn what is obvious.

2. **Offline DPO on the imitation dataset.** Take the demonstrations you recorded in [unit-09.md](unit-09.md). Synthesise "rejected" trajectories by adding action noise. Run DPO on these chosen/rejected pairs. Compare to plain BC. DPO with noise-as-rejected should produce a policy more robust to perturbations than BC alone — measure by injecting test-time observation noise and comparing return.

3. **Minimal annotation tool.** Build a 200-line Python app (Streamlit, Flask, or pure Tkinter) that loads two MP4 clips, plays them side-by-side, and writes the click to a JSONL file. Use it to label 500 real preferences on a behaviour you care about. Time-box the labelling to one hour. Report: how many pairs/hour did you achieve? How many felt like "ties"?

4. **Preference data efficiency curve.** Train reward models on N ∈ {50, 100, 250, 500, 1000, 2500} preference pairs (subsampled from the same total dataset). Plot held-out preference accuracy vs N. Find the elbow — the point past which more data stops helping. This is your project-specific minimum labelling budget.

5. **Reward-model ensemble for hacking detection.** Train 5 reward models with different seeds on the same preference data. During PPO fine-tuning, log the *standard deviation* across the ensemble for each rolled-out state. Plot it over training. The std should rise as the policy enters regions the ensemble disagrees on — those are the suspect states. Manually inspect 10 such states and confirm they look like hacking candidates.

---

## What's next

You have closed the loop on the alignment story this course has been telling: reward engineering ([unit-reward-engineering.md](unit-reward-engineering.md)) for objective goals, imitation learning ([unit-09.md](unit-09.md)) when you have demonstrations, and now RLHF when you only have a designer's taste. Each unlocks a class of tasks the previous one could not touch.

If you only remember one thing from this unit: **the reward is the goal, but the goal is in someone's head.** RLHF is the bridge from the head to the gradient.

[← Imitation Learning](unit-09.md) · [Course home](index.md)
