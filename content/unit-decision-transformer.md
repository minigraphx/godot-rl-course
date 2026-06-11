# Decision Transformer — RL as Sequence Modeling

[← Offline RL](unit-offline-rl.md) · [Course home](index.md) · [→ Ship Your Brain](unit-10.md)

!!! info "Time"
    Reading: ~40 min · Training: ~20 min GPU / ~1.5 h CPU

---

!!! info "Three ways to see your AI"
    - **Return-conditioned rollouts** — train a single Decision Transformer once, then run it at `target_return = 50, 100, 150, 200` and watch one model produce four qualitatively different policies. The return-to-go acts as a steering wheel for skill level.
    - **Attention heatmap** — extract the causal-attention weights from the final transformer layer and visualize which past (R, s, a) tokens the model leans on. Long-horizon Godot tasks should show attention spikes at branch-points (door openings, platform jumps).
    - **Skill ladder plot** — collect three datasets from CartPole (random, mediocre, expert), train one DT on the union, then plot `episode_return` versus `target_return`. A diagonal line means return conditioning works; a flat line means the model collapsed to the dataset average.

---

## 0 · The big idea: RL as sequence prediction

Everything in this course so far has used the same machinery: a value function or policy trained with Bellman backups, experience replay, target networks, advantage estimation, and a careful dance of hyperparameters. PPO has its clip ratio and GAE lambda. SAC has its entropy temperature. CQL (covered in [unit-offline-rl.md](unit-offline-rl.md)) has its conservative coefficient `alpha`. IQL has its expectile `tau` and weight temperature.

**Decision Transformer** (Chen et al., 2021) throws all of it out.

The pitch is one sentence: *predict the next action given a context window of past `(return-to-go, state, action)` tuples, trained like a language model on offline data*. There is no Bellman backup. No target network. No replay buffer. No critic. No exploration schedule. No conservative penalty. The training loop is the standard supervised loop you would use for GPT-2.

This works far better than it has any right to. On the standard offline RL benchmarks, Decision Transformer matches or beats CQL and IQL on most tasks despite being a strictly simpler algorithm. And because the architecture and training recipe are identical to a language model, the entire LLM ecosystem — distributed training, mixed-precision kernels, attention optimizations, FlashAttention, fine-tuning libraries — applies directly.

For a game-AI course this matters twice over. First, your Godot agents can now use the same compute infrastructure your team already maintains for text models. Second, the conditioning trick — telling the model "produce this much reward" — turns one trained model into a whole family of policies at different skill levels, which is exactly what you want for adaptive game difficulty.

---

## 1 · Return-to-go conditioning

The technical core of Decision Transformer is one design choice: instead of conditioning the policy on the state alone, condition it on `(state, desired_future_return)`.

### Inference loop

```
At inference:
  - Set R_1 = desired_total_reward            (e.g., 200 for CartPole)
  - Generate action a_1 conditioned on (R_1, s_1)
  - Execute a_1, observe r_1, s_2
  - Set R_2 = R_1 - r_1                       (countdown)
  - Generate a_2 conditioned on (R_2, s_2, a_1, R_1, s_1)
  - Set R_3 = R_2 - r_2
  - Generate a_3 conditioned on (R_3, s_3, a_2, R_2, s_2, a_1, R_1, s_1)
  - ...
```

The return-to-go starts at the target and counts down as rewards are earned. The model sees a running ledger of "how much more reward I still owe."

### Why this works

If the offline dataset contains high-return trajectories labeled with high `R_1`, the model learns the conditional distribution `p(action | state, return-to-go)`. At inference, asking for a specific return is just conditioning the generative model on a particular value of one of its inputs.

This is exactly the same trick used in conditional image generation ("generate an image of a cat") or instruction-tuned LLMs ("answer in formal English"). The conditioning variable steers the generation toward the slice of training data that matches.

### The token sequence

The trajectory is flattened into a single sequence of tokens:

```
[R_1, s_1, a_1, R_2, s_2, a_2, R_3, s_3, a_3, ...]
```

Each timestep contributes exactly three tokens in a fixed order. A context window of `K = 20` timesteps therefore becomes `3K = 60` tokens. The transformer treats this as one long sequence and uses causal masking so each token only attends to earlier tokens.

!!! tip "Return-to-go, not reward"
    The model is conditioned on the **sum of future rewards**, not the per-step reward. Per-step rewards are noisy and local; the return-to-go is a global signal that uniquely identifies "this is a successful trajectory" vs "this is a mediocre trajectory." This is the same insight that makes Monte Carlo returns more informative than one-step rewards in policy gradient methods.

---

## 2 · Architecture

Decision Transformer is structurally identical to GPT-2 with three small modifications: separate input embeddings for each modality, an extra timestep positional encoding, and an action prediction head.

### Components

- **Causal transformer (GPT-style)** — the same `TransformerEncoder` with a causal mask you would use for language modeling.
- **Input embeddings** — three separate linear layers: one for return-to-go (1 → `d_model`), one for state (`obs_dim` → `d_model`), one for action (`act_dim` → `d_model`).
- **Timestep positional encoding** — an `nn.Embedding(max_timesteps, d_model)` added to all three modality embeddings at each timestep. This is in addition to (or replacing) the standard sinusoidal position encoding, because the same timestep `t` appears at three different token positions in the flattened sequence.
- **Context length K** — typically 20 to 100 timesteps. Short contexts work surprisingly well; the model does not need the full episode history.
- **Output head** — a linear layer that takes the hidden state at each state-token position and predicts the action that should follow.
- **Loss** — mean squared error for continuous actions, cross-entropy for discrete actions.

### Architecture diagram

```
[R_1] [s_1] [a_1] [R_2] [s_2] [a_2] [R_3] [s_3] [?]
  |     |     |     |     |     |     |     |     |
  embed embed embed embed embed embed embed embed embed
  |     |     |     |     |     |     |     |     |
  +-------------------------------------------------+
  |         Causal Transformer (GPT-style)          |
  |              (masked self-attention)            |
  +-------------------------------------------------+
                                                    |
                                              action head
                                              -> predicted a_3
```

The key thing to internalize: the action prediction at position `s_t` only attends to tokens at positions `<= s_t` in the flattened sequence — meaning it sees all prior `(R, s, a)` triples plus the current `R_t` and `s_t`, but not the current `a_t`. This is what causal masking enforces.

!!! warning "Don't leak the action"
    A common bug is to accidentally include `a_t` in the context used to predict `a_t`. With the interleaved layout `[R_t, s_t, a_t]`, you must predict `a_t` from the hidden state at the `s_t` position, not the `a_t` position. If you predict from the `a_t` position, the model trivially copies the input and reaches near-zero loss while learning nothing useful.

---

## 3 · Training on offline data

The training recipe is supervised learning end to end. No environment in the loop.

### Pipeline

1. **Collect an offline dataset** — recorded trajectories with `(obs, action, reward)` at every step. Same format as for CQL/IQL.
2. **Compute returns-to-go** — for each trajectory, walk backward and compute `R_t = sum of rewards from step t to end`.
3. **Slice into context windows** — chunk each trajectory into overlapping windows of length `K`.
4. **Train with supervised loss** — feed `(R, s, a)` windows through the transformer, compute MSE (continuous) or cross-entropy (discrete) on the predicted actions.

There is no RL training loop after dataset collection. Training is `for batch in loader: loss.backward(); optimizer.step()` — identical to fine-tuning a language model.

### Full minimal implementation

```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataset(Dataset):
    """Dataset of (returns-to-go, states, actions) windows."""

    def __init__(self, trajectories, context_len=20, scale=1000.0):
        # trajectories: list of dicts with 'obs', 'actions', 'rewards'
        self.context_len = context_len
        self.scale = scale
        self.windows = []  # each entry: (rtg, states, actions, timesteps)

        for traj in trajectories:
            obs = np.asarray(traj["obs"], dtype=np.float32)
            acts = np.asarray(traj["actions"], dtype=np.float32)
            rews = np.asarray(traj["rewards"], dtype=np.float32)
            T = len(rews)

            # Returns-to-go: R_t = sum of rewards from t to end
            rtg = np.zeros(T, dtype=np.float32)
            running = 0.0
            for t in reversed(range(T)):
                running += rews[t]
                rtg[t] = running
            rtg /= self.scale  # normalize for stable training

            # Slice into windows of length context_len (pad if shorter)
            for start in range(0, T):
                end = min(start + context_len, T)
                L = end - start
                pad = context_len - L

                states = np.concatenate(
                    [np.zeros((pad, obs.shape[1]), dtype=np.float32), obs[start:end]]
                )
                actions = np.concatenate(
                    [np.zeros((pad, acts.shape[1]), dtype=np.float32), acts[start:end]]
                )
                rtgs = np.concatenate([np.zeros(pad, dtype=np.float32), rtg[start:end]])
                timesteps = np.concatenate(
                    [np.zeros(pad, dtype=np.int64), np.arange(start, end, dtype=np.int64)]
                )
                self.windows.append((rtgs, states, actions, timesteps))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        rtg, states, actions, timesteps = self.windows[idx]
        return (
            torch.from_numpy(rtg),
            torch.from_numpy(states),
            torch.from_numpy(actions),
            torch.from_numpy(timesteps),
        )


class DecisionTransformer(nn.Module):
    def __init__(self, obs_dim, act_dim, d_model=128, n_heads=4, n_layers=3, context_len=20):
        super().__init__()
        self.context_len = context_len

        # Embeddings for each modality
        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_state = nn.Linear(obs_dim, d_model)
        self.embed_action = nn.Linear(act_dim, d_model)
        self.embed_timestep = nn.Embedding(1000, d_model)

        # GPT-style transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.predict_action = nn.Linear(d_model, act_dim)

    def forward(self, rtg, states, actions, timesteps):
        # rtg:        (B, T)
        # states:     (B, T, obs_dim)
        # actions:    (B, T, act_dim)
        # timesteps:  (B, T)
        B, T = states.shape[:2]

        t_emb = self.embed_timestep(timesteps)
        rtg_emb = self.embed_rtg(rtg.unsqueeze(-1)) + t_emb
        state_emb = self.embed_state(states) + t_emb
        action_emb = self.embed_action(actions) + t_emb

        # Interleave per timestep: (B, T, 3, d_model) -> (B, 3T, d_model)
        x = torch.stack([rtg_emb, state_emb, action_emb], dim=2)
        x = x.reshape(B, 3 * T, -1)
        x = self.ln(x)

        # Causal mask
        mask = torch.triu(torch.ones(3 * T, 3 * T, device=x.device), diagonal=1).bool()
        h = self.transformer(x, mask=mask)

        # Predict action from state positions (every 3rd token starting at index 1)
        state_hiddens = h[:, 1::3]
        return self.predict_action(state_hiddens)


def train_decision_transformer(dataset, obs_dim, act_dim, n_epochs=100, device="cpu"):
    model = DecisionTransformer(obs_dim, act_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(n_epochs):
        total_loss = 0.0
        for rtg, states, actions, timesteps in loader:
            rtg, states, actions, timesteps = (
                rtg.to(device),
                states.to(device),
                actions.to(device),
                timesteps.to(device),
            )
            pred_actions = model(rtg, states, actions, timesteps)
            loss = ((pred_actions - actions) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total_loss / len(loader):.4f}")
    return model
```

### What to watch during training

- **Loss curve** — MSE loss should decrease smoothly. A flat curve from epoch 1 means embeddings are not learning; check the timestep embedding range against your trajectory lengths.
- **Gradient norm** — the `clip_grad_norm_(..., 0.25)` is critical. Transformer gradients spike if context windows include badly normalized returns-to-go (hence the `scale=1000.0`).
- **Validation rollouts** — every 10 epochs, evaluate in the env with `target_return = max_return_in_dataset`. The trend should be upward.

!!! tip "Scale the return-to-go"
    Always divide returns by a `scale` constant (typical: 1000 for CartPole/Atari, environment-specific for Godot). Raw returns like 1500 fed into a linear embedding produce huge activations that destabilize attention softmaxes. Pick `scale` so the typical return-to-go after dividing is in `[0, 1]`.

---

## 4 · Inference: steering with return-to-go

Inference is autoregressive. At each step you append the latest `(R, s, a)` to the running context, trim to the last `K` timesteps, and ask the model for the next action.

```python
import numpy as np
import torch


def evaluate_dt(model, env, target_return=200, context_len=20, scale=1000.0, max_steps=1000):
    """Run Decision Transformer in an environment."""
    obs, _ = env.reset()

    # Running context buffers (lists, trimmed each step)
    rtg_buf = [target_return / scale]
    state_buf = [obs]
    action_buf = [np.zeros(env.action_space.shape, dtype=np.float32)]
    timestep_buf = [0]

    total_reward = 0.0
    model.eval()

    for t in range(max_steps):
        start = max(0, t - context_len + 1)

        rtg = torch.tensor(rtg_buf[start:], dtype=torch.float32).unsqueeze(0)
        states = torch.tensor(np.array(state_buf[start:]), dtype=torch.float32).unsqueeze(0)
        actions = torch.tensor(np.array(action_buf[start:]), dtype=torch.float32).unsqueeze(0)
        timesteps = torch.tensor(timestep_buf[start:], dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            pred = model(rtg, states, actions, timesteps)
        action = pred[0, -1].cpu().numpy()  # last predicted action

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        rtg_buf.append(rtg_buf[-1] - reward / scale)
        state_buf.append(obs)
        action_buf.append(action)
        timestep_buf.append(t + 1)

        if terminated or truncated:
            break

    return total_reward
```

### Steering experiment

Train the DT once. Then run `evaluate_dt(model, env, target_return=R)` for a grid of target returns:

```python
for target in [50, 100, 150, 200, 250]:
    returns = [evaluate_dt(model, env, target_return=target) for _ in range(10)]
    print(f"target={target:>3}  achieved={np.mean(returns):.1f} ± {np.std(returns):.1f}")
```

Expected output for a well-trained DT on CartPole-v1 (max return 500):

```
target= 50   achieved= 53.2 ± 8.1
target=100   achieved=104.7 ± 11.4
target=150   achieved=147.9 ± 14.0
target=200   achieved=198.5 ± 18.2
target=250   achieved=241.6 ± 22.7
```

One model produces five qualitatively different policies. This is impossible with standard policy gradient methods — you would have to retrain SAC/PPO from scratch for each target.

!!! check "Done when"
    There is no fixed score to hit — judge the run by this unit's own signature result: (1) the training loss from Section 3 decreases smoothly over the epochs, and (2) the steering grid above shows achieved return rising with `target_return` — asking for more visibly produces more, roughly tracking the target across the grid. A flat steering curve (every target mapped to the same achieved return) is the failure signature: check the `scale` constant and context length first, then whether your dataset actually contains varied returns (Section 7) — it does not mean you need more epochs.

!!! info "Adaptive difficulty"
    The steering experiment is a direct demonstration of how to build adaptive game difficulty into a single trained model. A NPC controlled by a Decision Transformer can be set to "easy" (low target return), "medium," or "expert" without any retraining or per-difficulty checkpoints.

---

## 5 · DT vs. classic offline RL

Decision Transformer is the easiest offline RL algorithm to *implement* but it does not dominate CQL and IQL on every problem. The differences are real and the choice should be driven by your dataset.

| | BC | CQL | IQL | Decision Transformer |
|---|---|---|---|---|
| Training | Supervised | Bellman + conservatism | Bellman + in-sample | Supervised (no Bellman) |
| Hyperparameters | Few | Many (alpha, beta, target updates) | Moderate (tau, beta) | Few (context_len, scale, lr) |
| Stitching (combine suboptimal trajectories) | No | Yes | Yes | Limited (context-only) |
| Return conditioning | No | No | No | Yes |
| Architecture | MLP | MLP | MLP | Transformer |
| Inference cost | Low | Low | Low | Higher (attention over K steps) |
| GPU friendly | Modest | Modest | Modest | Very (LLM stack applies) |
| Best for | Expert data | Sparse-coverage data | Mixed-quality data | Data with varied returns |

### The stitching caveat

The single most important limitation of Decision Transformer: **it cannot stitch**.

"Stitching" means combining pieces of two suboptimal trajectories to produce a better one. CQL and IQL stitch because the Bellman backup propagates value information across states regardless of which trajectory the data came from. If trajectory A reaches state `s*` and trajectory B starts near `s*` and reaches a high-reward terminal, the Q-function learns that `s*` is valuable and `Q(s, a -> s*)` becomes high.

Decision Transformer does not have this mechanism. It is a conditional density model over sequences. If no trajectory in the dataset achieves `target_return`, DT has no learning signal for "how to actually achieve that return" — it can only interpolate within the support of trajectories it has seen.

### Practical consequence

If your offline dataset is a mix of short, suboptimal episodes, CQL/IQL will reliably extract a stitched policy that exceeds the best individual trajectory. DT will plateau near the best trajectory's return.

If your offline dataset contains a diverse mix of returns — including some near-optimal trajectories — DT shines because the return-to-go acts as a clean conditioning signal and you can ask for the high-return slice at inference time.

| Your dataset | First algorithm to try |
|---|---|
| Many suboptimal trajectories, few/no expert | CQL or IQL (need stitching) |
| Mix from low to high return | Decision Transformer (return conditioning works) |
| Mostly expert | BC, then DT for steerability |
| Tiny (< 200 episodes) | CQL (DT will overfit) |
| Huge (> 10k episodes), want pre-train + fine-tune | DT (LLM-style scaling) |

---

## 6 · Trajectory Transformer (brief, optional on a first read)

!!! note "First pass? Skim or skip this section."
    The hands-on path runs through Sections 0–5 (the big idea, return-to-go conditioning, architecture, training, steering, and when to prefer DT) plus Section 7 (the Godot dataset and comparison). Trajectory Transformer is a sibling algorithm worth knowing by name — nothing later in the unit depends on it.

**Trajectory Transformer** (Janner et al., 2021) is the closely-related sibling of DT and worth knowing by name.

The key differences:

- Tokens are `(state, action, reward)` triples — **rewards are tokens**, not conditioning variables.
- At inference, the model performs **beam search** over the joint sequence to plan future trajectories that maximize predicted reward.
- This makes it a *planning* algorithm, not just a *policy*. It can reason about counterfactual futures.

Trade-offs versus Decision Transformer:

| | Decision Transformer | Trajectory Transformer |
|---|---|---|
| Inference | One forward pass per step | Beam search over horizon (expensive) |
| Return signal | Conditioning input | Predicted as a token |
| Planning capability | None | Yes (search) |
| Suitable for real-time game AI | Yes | Borderline (beam search latency) |
| Reference | Chen et al., 2021 | Janner et al., 2021 |

For Godot agents that must produce an action every physics tick (60 Hz typical), DT is the right choice. Trajectory Transformer becomes interesting when you have a turn-based game or a planning horizon long enough to amortize beam search cost.

---

## 7 · Godot integration

Decision Transformer needs the same offline dataset as CQL/IQL — recorded `(obs, action, reward)` trajectories. The cleanest way to produce one in this course is to record rollouts from a trained PPO agent (see [unit-07.md](unit-07.md)) and dump them to a list of dicts.

### Collecting trajectories from a trained PPO agent

```python
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from stable_baselines3 import PPO
import numpy as np


def collect_trajectories(env_path, model_path, n_trajectories=1000):
    """Collect trajectory data from a trained PPO agent."""
    env = StableBaselinesGodotEnv(env_path=env_path, n_parallel=1)
    model = PPO.load(model_path)

    trajectories = []
    for ep in range(n_trajectories):
        obs, _ = env.reset()
        traj = {"obs": [], "actions": [], "rewards": []}
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            traj["obs"].append(np.asarray(obs, dtype=np.float32))
            traj["actions"].append(np.asarray(action, dtype=np.float32))
            obs, reward, terminated, truncated, _ = env.step(action)
            traj["rewards"].append(float(reward))
            done = terminated or truncated
        trajectories.append(traj)

    env.close()
    return trajectories
```

### Mixing in suboptimal policies

A DT trained only on expert PPO rollouts will produce a flat steering curve — every target return gets mapped to the same expert behavior. To exercise return conditioning you need *varied* returns in the dataset. The cleanest way:

1. Save PPO checkpoints at 25%, 50%, 75%, 100% of training.
2. Collect 250 trajectories from each checkpoint.
3. Concatenate into a single 1000-trajectory dataset.

This produces a dataset with returns spread from "mediocre" to "expert," which is exactly the regime where DT's conditioning shines.

```python
checkpoints = ["ppo_25.zip", "ppo_50.zip", "ppo_75.zip", "ppo_100.zip"]
all_trajectories = []
for ckpt in checkpoints:
    all_trajectories.extend(collect_trajectories("MultiLevelRobot.x86_64", ckpt, n_trajectories=250))

dataset = TrajectoryDataset(all_trajectories, context_len=20, scale=1000.0)
obs_dim = all_trajectories[0]["obs"][0].shape[0]
act_dim = all_trajectories[0]["actions"][0].shape[0]
model = train_decision_transformer(dataset, obs_dim, act_dim, n_epochs=100)
```

### Headline comparison

After training, run all three on the Godot env and compare:

| Policy | How it was produced | Expected episode return |
|---|---|---|
| Original PPO (expert checkpoint) | Standard online training | High (baseline) |
| BC trained on the mixed dataset | Supervised on `(s, a)` only | Medium — collapses to dataset mean |
| DT at `target_return = max` | This unit's recipe | Close to expert PPO |
| DT at `target_return = median` | Same model, different conditioning | Close to mediocre checkpoint |

The DT at `target_return = max` should approach the PPO expert. The DT at `target_return = median` should approach the mediocre checkpoint. **One model, full skill ladder.**

---

## 8 · Connection to LLMs (optional on a first read)

!!! note "First pass? Skim or skip this section."
    Like Section 6, this is background for the curious — the hands-on path runs through Sections 0–5 (the big idea, return-to-go conditioning, architecture, training, steering, and when to prefer DT) plus Section 7 (the Godot dataset and comparison). Come back when you want the LLM-ecosystem context behind the "RL as sequence modeling" framing.

A trained Decision Transformer and a trained GPT-style language model are the same algorithm running on different tokens. Both are causal transformers trained with next-token prediction on offline corpora. The only differences are token type (action vs word piece) and modality of input embeddings (linear projections of `(R, s, a)` vs a learned vocabulary embedding).

This is not a metaphor. The concrete consequences:

- **Same training infrastructure** — DeepSpeed, FSDP, mixed-precision training, FlashAttention, gradient checkpointing all apply to DT with no modification.
- **Same fine-tuning recipes** — LoRA, prefix tuning, RLHF-style preference fine-tuning all work on DT.
- **Same scaling laws** — larger DTs trained on larger trajectory corpora keep improving, mirroring the Chinchilla-style scaling curves familiar from language models.

Two important follow-up papers extend the analogy:

- **Prompt Decision Transformer** (Xu et al., 2022) — prepends a small set of "demonstration tokens" describing the task. One model handles many tasks; switching tasks is changing the prompt, not retraining. This is the RL analog of system prompts in chat LLMs.
- **Hyper-Decision Transformer** (Xu et al., 2023) — uses a hypernetwork to generate task-specific weights from a task description. The same architectural pattern is now used in multi-task LLM serving.

For the game-AI angle: a single Decision Transformer can play many games at many skill levels by varying both the prompt (task) and the return-to-go (skill). This is exactly the substrate you need for a unified NPC brain that handles every minigame in a large RPG with one checkpoint.

---

## 9 · Stretch goals

**Skill-ladder verification on CartPole.**
Collect three CartPole datasets at distinct skill levels: a random policy (return ~20), a partially trained DQN (return ~100), and a fully trained DQN (return ~500). Train one DT on the union. Run the steering experiment from Section 4 with `target_return in {25, 50, 100, 200, 400, 500}`. Plot `achieved_return` vs `target_return`. A near-diagonal line means return conditioning is working; a flat line means the model has collapsed and you need to investigate (likely the scale constant or context length).

**Head-to-head with CQL and IQL on a Godot env.**
Take the four-PPO-checkpoint dataset from Section 7. Train three policies on it: CQL (with d3rlpy), IQL (with d3rlpy), and DT (this unit). Evaluate all three on the MultiLevelRobot task for 50 episodes each. Report mean episode return and success rate. Expected outcome: DT competitive with or slightly behind IQL on this task; well ahead of CQL if dataset coverage is rich.

**Attention visualization.**
Modify the `DecisionTransformer.forward` method to also return the attention weights from the final layer (use `torch.nn.functional.scaled_dot_product_attention` with `return_attention=True`, or swap in a custom attention module). For one rollout episode, plot the attention map (`3K x 3K`) at each step. Annotate which past tokens the model attends to at branch points in the environment (door openings, platform transitions). You should see attention spikes lining up with semantically important past states.

**Minimal Prompt Decision Transformer.**
Add a single "task token" at the front of every sequence — a learned `nn.Parameter` of shape `(n_tasks, d_model)` indexed by task ID. Train one DT on data from two distinct Godot environments (e.g., MultiLevelRobot and a different scene). At inference, switch tasks by changing the task ID. Verify that one model handles both. This is a 20-line modification that replicates the core idea of Prompt DT.

**Phase transitions in steering.**
Run the steering experiment with a fine-grained grid: `target_return in [0, 10, 20, 30, ..., 500]`. Plot the result. Is the curve smooth, or are there step-like phase transitions where small changes in `target_return` produce large jumps in behavior? Phase transitions usually indicate that your training dataset is multimodal (a few discrete clusters of trajectory quality) rather than continuously distributed. The fix is more dataset diversity.

---

[← Offline RL](unit-offline-rl.md) · [Course home](index.md) · [→ Ship Your Brain](unit-10.md)
