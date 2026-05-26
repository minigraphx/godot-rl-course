# PPO From Scratch — CleanRL and the Implementation Layer

[← SAC](unit-sac.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~50 min · Training: ~30 min GPU / ~2 h CPU

---

!!! info "Three ways to see your AI"
    CartPole training curve peaking at 500 in under 200k steps · the actual loss curves (`policy_loss`, `value_loss`, `entropy_loss`) all in one TensorBoard run · the CleanRL source file open in your editor — every line readable, no abstraction hiding anything

---

## 0 · Why read the code?

You have spent units reading about PPO. You have run SB3's PPO on a dozen Godot environments. You know what `clip_range`, `gae_lambda`, and `vf_coef` do in theory. But there is a gap between *knowing the equation* and *knowing the implementation*, and that gap matters the moment things go wrong.

**SB3's PPO spans 15+ files.** Here is a partial list:

- `on_policy_algorithm.py` — the base training loop
- `ppo.py` — the PPO-specific loss
- `policies.py` — the network architecture
- `buffers.py` — rollout storage
- `type_aliases.py`, `utils.py`, `callbacks.py` — supporting machinery

None of those files is long. Each is clean and well-documented. But to understand *how* a PPO update happens from rollout start to gradient step completion, you have to read all of them, trace the call graph, understand which class inherits from what, and follow the data through six different type transformations. It takes a day.

**CleanRL's PPO is a single `ppo.py` file, roughly 300 lines.** Read it once — 30 minutes — and you understand the whole algorithm. Every line is either data collection, advantage computation, or a loss term you can directly map to an equation.

This unit bridges the theory you built in [unit-ppo-deep.md](unit-ppo-deep.md) to a real, running, hackable implementation. After this unit you should be able to:

- Run CleanRL's PPO on CartPole and read the TensorBoard output.
- Walk a colleague through every block of `ppo.py` line by line.
- Explain the backwards GAE loop without notes.
- Patch CleanRL to wrap a Godot environment.
- Make three concrete algorithmic modifications and observe their effects.

---

## 1 · CleanRL setup

### Installation

```bash
pip install cleanrl[gym,atari]
# Or, for the full dev environment:
git clone https://github.com/vwxyzjn/cleanrl.git
cd cleanrl
pip install -e ".[gym,atari]"
```

The `[gym,atari]` extras pull in `gymnasium`, `stable-baselines3` (for vectorized env utilities), and `tensorboard`. If you already have a course virtual environment, install into it — there are no conflicts with your existing SB3 installation.

### Running the CartPole baseline

```bash
python cleanrl/ppo.py \
    --env-id CartPole-v1 \
    --total-timesteps 500000 \
    --learning-rate 2.5e-4 \
    --num-envs 4 \
    --num-steps 128 \
    --num-minibatches 4 \
    --update-epochs 4
```

You will see output like:

```
global_step=4096,  episodic_return=23.0
global_step=8192,  episodic_return=67.0
global_step=40960, episodic_return=317.0
global_step=81920, episodic_return=500.0
```

And TensorBoard at `http://localhost:6006` (run `tensorboard --logdir runs/`).

### Mapping CleanRL TensorBoard to SB3 TensorBoard

You already know SB3's metrics from previous units. Here is the direct correspondence:

| SB3 metric | CleanRL metric | Notes |
|---|---|---|
| `train/policy_gradient_loss` | `losses/policy_loss` | Same quantity, different key |
| `train/value_loss` | `losses/value_loss` | Mean squared error on value predictions |
| `train/entropy_loss` | `losses/entropy` | Negative entropy (SB3 negates it) |
| `train/approx_kl` | `losses/approx_kl` | KL proxy; watch for spikes above 0.02 |
| `train/clip_fraction` | `losses/clipfrac` | Fraction of transitions where clipping fired |
| `rollout/ep_rew_mean` | `charts/episodic_return` | The one you always watch first |
| `rollout/ep_len_mean` | `charts/episodic_length` | Useful for tasks with variable episode length |
| `time/fps` | `charts/SPS` | Steps per second throughput |

!!! tip "SPS vs FPS"
    CleanRL logs `SPS` (samples per second) rather than `FPS`. In Godot environments where each step is slower, you will see SPS drop significantly — that's expected. The number to optimize is `charts/episodic_return` per wall-clock minute, not per step.

---

## 2 · The PPO file walkthrough

What follows is a section-by-section walkthrough of CleanRL's `ppo.py`. The code shown is very close to the real file but lightly simplified for readability. All the important logic is preserved.

### 2.1 Argument parsing and seeding

```python
import argparse, random, time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id",           type=str,   default="CartPole-v1")
    parser.add_argument("--total-timesteps",  type=int,   default=500_000)
    parser.add_argument("--learning-rate",    type=float, default=2.5e-4)
    parser.add_argument("--num-envs",         type=int,   default=4)
    parser.add_argument("--num-steps",        type=int,   default=128)
    parser.add_argument("--num-minibatches",  type=int,   default=4)
    parser.add_argument("--update-epochs",    type=int,   default=4)
    parser.add_argument("--clip-coef",        type=float, default=0.2)
    parser.add_argument("--ent-coef",         type=float, default=0.01)
    parser.add_argument("--vf-coef",          type=float, default=0.5)
    parser.add_argument("--max-grad-norm",    type=float, default=0.5)
    parser.add_argument("--gae-lambda",       type=float, default=0.95)
    parser.add_argument("--gamma",            type=float, default=0.99)
    parser.add_argument("--seed",             type=int,   default=1)
    return parser.parse_args()

args = parse_args()
batch_size     = args.num_envs * args.num_steps        # total transitions per update
minibatch_size = batch_size // args.num_minibatches    # transitions per mini-batch
num_updates    = args.total_timesteps // batch_size    # how many update cycles

# Seeding — reproducibility matters for debugging
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
```

**What to notice:** `batch_size` is `num_envs × num_steps`. With 4 envs and 128 steps you get 512 transitions per rollout. Those 512 transitions get shuffled and split into 4 mini-batches of 128. The update loop runs over all 4 mini-batches, 4 times (`update_epochs`), for a total of 16 gradient steps per rollout. This maps exactly to SB3's `n_steps=128`, `n_epochs=4`, `batch_size=128`.

### 2.2 Environment setup

```python
def make_env(env_id, seed, idx, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env
    return thunk

envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, args.seed, i, run_name) for i in range(args.num_envs)]
)
```

`RecordEpisodeStatistics` wraps each environment so that `info["episode"]["r"]` and `info["episode"]["l"]` are populated at the end of every episode. CleanRL reads those to log `episodic_return` and `episodic_length`. This is the same pattern we use in the Godot wrapper section later.

`SyncVectorEnv` runs the environments one after another in a single process. The alternative, `AsyncVectorEnv`, runs them in parallel subprocesses — faster, but harder to debug. Start with `Sync`.

### 2.3 Network architecture

```python
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization, a CleanRL signature choice."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = envs.single_action_space.n  # discrete case

        # Critic network: outputs a single scalar V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Actor network: outputs logits over actions
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist   = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)
```

**Key design decisions:**

- **Separate actor and critic heads, shared nothing.** Unlike many implementations that share the trunk MLP, CleanRL's default keeps them independent. This avoids gradient interference between the policy and value objectives. See [unit-ppo-deep.md §8](unit-ppo-deep.md) for the trade-off discussion.
- **Orthogonal initialization** with `std=np.sqrt(2)` for hidden layers, `std=0.01` for the actor output (keeps initial action probabilities near-uniform), `std=1.0` for the critic output (no shrinkage on the value scale).
- **`Tanh` activations.** CleanRL uses Tanh, not ReLU, for discrete-action tasks. Tanh is bounded, which plays better with orthogonal init.

!!! note "Continuous actions"
    For continuous control, the actor outputs a mean and log-std (either as a parameter or as a network output). CleanRL has `ppo_continuous_action.py` with this exact change. The rest of the algorithm is identical.

### 2.4 Rollout storage

```python
# Pre-allocate buffers — fill them in the collection loop
obs     = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape)
actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape)
logprobs= torch.zeros((args.num_steps, args.num_envs))
rewards = torch.zeros((args.num_steps, args.num_envs))
dones   = torch.zeros((args.num_steps, args.num_envs))
values  = torch.zeros((args.num_steps, args.num_envs))
```

Shape: `(num_steps, num_envs, ...)`. Time goes along axis 0, environments along axis 1. After the rollout, we flatten both into a single `(batch_size, ...)` tensor for the update loop.

### 2.5 Rollout collection loop

```python
next_obs  = torch.Tensor(envs.reset()[0])   # (num_envs, obs_dim)
next_done = torch.zeros(args.num_envs)

for update in range(1, num_updates + 1):

    # --- ROLLOUT PHASE ---
    for step in range(args.num_steps):
        obs[step]  = next_obs
        dones[step]= next_done

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)

        actions[step]  = action
        logprobs[step] = logprob
        values[step]   = value.flatten()

        next_obs_np, reward, terminated, truncated, infos = envs.step(action.numpy())
        done = np.logical_or(terminated, truncated)

        rewards[step] = torch.tensor(reward)
        next_obs      = torch.Tensor(next_obs_np)
        next_done     = torch.Tensor(done)

        # Log completed episodes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    writer.add_scalar("charts/episodic_return",
                                      info["episode"]["r"], global_step)
```

The rollout loop is straightforward: for each step, store the current state, sample an action, step the environment, store the reward. Notice that `torch.no_grad()` wraps the forward pass — we are just collecting data, not computing gradients yet. The gradients come later in the update loop.

---

## 3 · GAE deep dive

This is where most students get confused. Read this section slowly.

### Why GAE?

After collecting a rollout, we need advantage estimates A_t for each transition. The simplest option is the **TD error**:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
```

This is low-variance but high-bias — we only look one step ahead. The full **Monte Carlo return** is unbiased but high-variance — we sum rewards all the way to the end of the episode, which is noisy.

**GAE (Generalized Advantage Estimation)** from [unit-ppo-deep.md §6](unit-ppo-deep.md) interpolates between them with parameter λ:

```
A_t^GAE = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

When λ=0 this collapses to pure TD error. When λ=1 it approximates the full Monte Carlo advantage. λ=0.95 is the default — close to Monte Carlo, but with variance tamed.

### The backwards loop

The equation above is a sum over future TD errors. Computing it forward would require you to have all future values available before you can compute A_t. The trick: **the sum has a recursive structure**.

```
A_t^GAE = δ_t + γλ · A_{t+1}^GAE
```

Read right-to-left: the advantage at step t is the TD error at step t, plus the discounted advantage at step t+1. This means you can compute all advantages in one backwards sweep — start at the last step and work backwards to the first.

### The actual code

```python
with torch.no_grad():
    next_value = agent.get_value(next_obs).reshape(1, -1)  # V(s_T)
    advantages = torch.zeros_like(rewards)  # (num_steps, num_envs)
    last_gae_lam = 0.0

    # Iterate BACKWARDS: step T-1 down to 0
    for t in reversed(range(args.num_steps)):

        if t == args.num_steps - 1:
            # At the last stored step, the "next" observation is next_obs
            nextnonterminal = 1.0 - next_done        # 0 if episode just ended
            nextvalues      = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]    # 0 if episode ended at t+1
            nextvalues      = values[t + 1]

        # TD error for this step
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]

        # GAE recursion: A_t = δ_t + γλ · (1 - done_{t+1}) · A_{t+1}
        advantages[t] = last_gae_lam = (
            delta + args.gamma * args.gae_lambda * nextnonterminal * last_gae_lam
        )

    # Returns = advantages + values (used as targets for value loss)
    returns = advantages + values
```

**Line-by-line explanation:**

1. `nextnonterminal` — when an episode ends, the future value is zero (there is no next state). Multiplying by `(1 - done)` zeroes out the bootstrap value at episode boundaries. This is the most common bug when implementing GAE yourself: forgetting to mask out values at terminal states.

2. `delta` — the TD error: reward at step t, plus discounted next value (if not terminal), minus current value estimate.

3. `last_gae_lam` — this is A_{t+1}^GAE from the recursion. On the first (backwards) iteration it is zero (there's nothing after step T). Each iteration updates it to the current A_t, which becomes A_{t+1} for the next (backwards) iteration.

4. `returns = advantages + values` — the value target. Since `advantage = return - value`, we have `return = advantage + value`. These returns become the target for the value loss: we want the critic to predict the return, not just the TD error.

!!! warning "The most common GAE bug"
    Forgetting `nextnonterminal` at episode boundaries. If you bootstrap from V(s_{t+1}) at a terminal state, you add value that doesn't exist — the episode is over. Always zero out the bootstrap value when `done[t+1]` is True.

### λ=0 vs λ=0.95: a numerical illustration

Consider a 4-step episode with rewards `[0, 0, 0, 1]` and all values estimated at 0 (a fresh, untrained network). γ=0.99, V(s_4)=0.

**TD error (λ=0):**
```
δ_3 = 1 + 0.99·0 - 0 = 1.0
δ_2 = 0 + 0.99·0 - 0 = 0.0
δ_1 = 0 + 0.99·0 - 0 = 0.0
δ_0 = 0 + 0.99·0 - 0 = 0.0

A_3=1.0,  A_2=0.0,  A_1=0.0,  A_0=0.0
```
Only the last step gets a gradient signal. The early states learn nothing about the eventual reward.

**GAE (λ=0.95):**
```
A_3 = 1.0
A_2 = 0.0 + 0.99·0.95·1.0 = 0.940
A_1 = 0.0 + 0.99·0.95·0.940 = 0.884
A_0 = 0.0 + 0.99·0.95·0.884 = 0.831
```
All four steps have a gradient signal. The early states learn that they were on a path to reward.

This is why λ matters so much in sparse-reward environments — like Godot agents that only get a reward when they reach the goal. With λ=0, the agent must get lucky and be at the exact step that precedes the reward for the update to do anything. With λ=0.95, the credit propagates backwards through the entire trajectory.

---

## 4 · The PPO update loop

```python
# Flatten the rollout dimensions: (num_steps, num_envs) → (batch_size,)
b_obs       = obs.reshape((-1,) + envs.single_observation_space.shape)
b_logprobs  = logprobs.reshape(-1)
b_actions   = actions.reshape((-1,) + envs.single_action_space.shape)
b_advantages= advantages.reshape(-1)
b_returns   = returns.reshape(-1)
b_values    = values.reshape(-1)

# Normalize advantages within the mini-batch (reduces variance)
b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

clipfracs = []

for epoch in range(args.update_epochs):
    # Shuffle indices to decorrelate mini-batches
    b_inds = np.random.permutation(batch_size)

    for start in range(0, batch_size, minibatch_size):
        end  = start + minibatch_size
        mb_inds = b_inds[start:end]

        # Forward pass with current (updating) policy
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
            b_obs[mb_inds], b_actions[mb_inds]
        )

        # Importance sampling ratio: π_new(a|s) / π_old(a|s)
        logratio   = newlogprob - b_logprobs[mb_inds]
        ratio      = logratio.exp()

        # Approximate KL (cheap proxy — no need for exact KL)
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

        mb_advantages = b_advantages[mb_inds]

        # --- CLIPPED POLICY LOSS (§4 of unit-ppo-deep.md) ---
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

        # --- VALUE LOSS ---
        newvalue = newvalue.view(-1)
        v_loss   = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

        # --- ENTROPY BONUS (encourages exploration) ---
        entropy_loss = entropy.mean()

        # --- COMBINED LOSS ---
        loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

# Log to TensorBoard
writer.add_scalar("losses/policy_loss",  pg_loss.item(),     global_step)
writer.add_scalar("losses/value_loss",   v_loss.item(),      global_step)
writer.add_scalar("losses/entropy",      entropy_loss.item(),global_step)
writer.add_scalar("losses/approx_kl",    approx_kl.item(),   global_step)
writer.add_scalar("losses/clipfrac",     np.mean(clipfracs), global_step)
```

**Mapping every term to unit-ppo-deep.md:**

| Code | Equation | Unit section |
|---|---|---|
| `ratio = (newlogprob - b_logprobs).exp()` | r_t(θ) = π_θ / π_{θ_old} | §3 |
| `pg_loss1 = -mb_advantages * ratio` | L^CPI (unclipped) | §3 |
| `pg_loss2 = -mb_advantages * clamp(ratio, ...)` | L^CLIP term 2 | §4 |
| `pg_loss = max(pg_loss1, pg_loss2)` | min(·, ·) of L^CLIP | §4 |
| `v_loss = 0.5 * MSE(newvalue, returns)` | Value loss term | §7 |
| `entropy_loss = entropy.mean()` | Entropy bonus H(π) | §7 |
| `clip_grad_norm_(...)` | Gradient clipping | §8 |

!!! note "Negation convention"
    CleanRL negates `pg_loss` to turn maximization into minimization (standard PyTorch idiom). The entropy term is subtracted (`- ent_coef * entropy_loss`) because we want to *maximize* entropy, which means minimizing *negative* entropy. If you see `entropy_loss` go down in TensorBoard, entropy is going up — that's good in the early training phase.

---

## 5 · Connecting to Godot

CleanRL does not have a Godot wrapper, but `godot_rl_agents` exposes a gymnasium-compatible environment that plugs directly into CleanRL's `make_env` factory.

### Wrapping the Godot environment

```python
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import gymnasium as gym

def make_godot_env(env_path, seed, idx, run_name, port=11008):
    """
    Factory returning a thunk that constructs a single Godot environment.
    CleanRL's SyncVectorEnv expects a list of such thunks.
    """
    def thunk():
        env = StableBaselinesGodotEnv(
            env_path=env_path,
            show_window=(idx == 0),    # only show window for the first env
            port=port + idx,           # each env needs its own port
            seed=seed + idx,
        )
        # Wrap with RecordEpisodeStatistics so CleanRL can log episodic_return
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def build_godot_envs(env_path, num_envs, seed, run_name):
    """
    Build a vectorized Godot env compatible with CleanRL's rollout loop.
    """
    envs = gym.vector.SyncVectorEnv([
        make_godot_env(env_path, seed, i, run_name, port=11008)
        for i in range(num_envs)
    ])
    # Sanity check: CleanRL expects Box or Discrete
    assert isinstance(
        envs.single_action_space,
        (gym.spaces.Discrete, gym.spaces.Box)
    ), f"Unexpected action space: {envs.single_action_space}"
    return envs


# Usage — drop this in place of the CartPole make_env call
if __name__ == "__main__":
    args = parse_args()
    run_name = f"godot__{args.env_id}__{args.seed}__{int(time.time())}"
    writer   = SummaryWriter(f"runs/{run_name}")

    envs  = build_godot_envs(
        env_path="path/to/your_game.x86_64",
        num_envs=args.num_envs,
        seed=args.seed,
        run_name=run_name,
    )
    agent = Agent(envs).to(device)
    # ... rest of the CleanRL training loop unchanged
```

!!! warning "Port conflicts"
    Each Godot instance needs its own port. With `num_envs=4`, you need ports 11008–11011 open. If you see `ConnectionRefusedError` on startup, check that no previous Godot instances are still running (`pkill -f your_game.x86_64`).

!!! tip "Start with num_envs=1"
    Debugging multi-env Godot setups is painful. Get `num_envs=1` working first, verify rewards are flowing, then scale up. The CleanRL training loop works identically with one environment — it's just slower.

### Observation and action space notes

`StableBaselinesGodotEnv` exposes:
- `observation_space` as a `gym.spaces.Box` (flat float array by default).
- `action_space` as either `Discrete` (for discrete actions) or `Box` (for continuous).

For continuous actions, swap `Agent` for the continuous-action variant: replace `Categorical(logits=self.actor(x))` with a `Normal(mean, std)` distribution, and update the log-probability and entropy computations accordingly. CleanRL's `ppo_continuous_action.py` has the complete implementation.

---

## 6 · Modifying PPO

One of the main reasons to use CleanRL instead of SB3 is that every modification is localized to a few lines in one file. Here are three concrete hacks.

### Hack 1 — Gradient norm diagnostics

**What:** Log the gradient norm to TensorBoard after every update step. High gradient norms indicate the loss landscape is steep — a sign of instability.

**Before (in the update loop):**
```python
nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
optimizer.step()
```

**After:**
```python
grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
optimizer.step()
writer.add_scalar("diagnostics/grad_norm", grad_norm.item(), global_step)
```

`clip_grad_norm_` returns the total norm of the gradients *before* clipping. Log it. If `grad_norm` regularly exceeds `max_grad_norm` by 10×, your loss is exploding — reduce learning rate or increase `max_grad_norm`.

**What to watch in TensorBoard:** `diagnostics/grad_norm` should be stable after warm-up, typically between 0.1 and 2.0. A spike to 50+ followed by a reward collapse is a clear gradient explosion signal.

### Hack 2 — Early stopping on approx_kl

**What:** SB3 has a `target_kl` parameter that stops the update loop early if the KL divergence exceeds a threshold. CleanRL does not do this by default. Add it.

**Before (epoch loop start):**
```python
for epoch in range(args.update_epochs):
    b_inds = np.random.permutation(batch_size)
    for start in range(0, batch_size, minibatch_size):
        ...
```

**After:**
```python
target_kl = 0.01   # add this to parse_args() as --target-kl

for epoch in range(args.update_epochs):
    b_inds = np.random.permutation(batch_size)
    for start in range(0, batch_size, minibatch_size):
        end    = start + minibatch_size
        mb_inds = b_inds[start:end]

        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
            b_obs[mb_inds], b_actions[mb_inds]
        )
        logratio = newlogprob - b_logprobs[mb_inds]
        ratio    = logratio.exp()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()

        # NEW: break inner loop if KL is too large
        if approx_kl > target_kl:
            break
        ...

    # NEW: also break outer epoch loop
    else:
        continue
    break
```

**What to watch:** `losses/approx_kl` should hover around 0.005–0.015. If early stopping fires consistently in epoch 2 of 4, your learning rate may be too high for the current environment. Try reducing `learning_rate` by 2×.

### Hack 3 — Running reward normalization

**What:** Normalize rewards by a running mean and standard deviation before they enter the GAE computation. This stabilizes training when reward magnitudes vary across environments or training phases.

**Before imports/setup:**
```python
class RunningMeanStd:
    """Welford online algorithm for mean and variance."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x)
        batch_mean = x.mean(axis=0)
        batch_var  = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta     = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean  = self.mean + delta * batch_count / tot_count
        m_a       = self.var   * self.count
        m_b       = batch_var  * batch_count
        m2        = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.mean  = new_mean
        self.var   = m2 / tot_count
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

# Instantiate once before the training loop
reward_rms = RunningMeanStd(shape=())
```

**In the rollout loop, after storing rewards:**
```python
rewards[step] = torch.tensor(reward)

# NEW: update running stats and normalize
reward_rms.update(reward)
rewards[step] = (rewards[step] - reward_rms.mean) / reward_rms.std
```

**What to watch:** `charts/episodic_return` should become more stable early in training. The raw episode return is still logged (via `RecordEpisodeStatistics`, which sees rewards *before* normalization). If training was already stable, normalization may not help — skip it and avoid the extra complexity.

!!! tip "Reward normalization vs advantage normalization"
    CleanRL already normalizes advantages (`b_advantages = (b_advantages - mean) / std`). Reward normalization is an additional, earlier normalization that affects the scale of the *targets* for the value loss. They address different problems: advantage normalization controls gradient scale; reward normalization controls value function scale in environments with very large or very small rewards.

---

## 7 · Sample Factory — maximum throughput

For large-scale training — millions of steps in environments that are slow per step — you eventually hit the ceiling of SB3 and CleanRL. Sample Factory is the answer.

### The key difference: asynchronous rollout collection

CleanRL and SB3 collect rollouts **synchronously**: collect N steps → compute gradients → collect N more steps → repeat. The GPU sits idle during collection; the CPU sits idle during the GPU update. Throughput is limited by whichever is slower.

Sample Factory uses **asynchronous rollout workers**: multiple processes collect experience simultaneously and push it into a shared replay buffer. The learner process consumes from the buffer continuously. The GPU is never idle waiting for rollout workers, and workers are never idle waiting for the learner. This architecture is 10–100× faster than SB3 for CPU-heavy environments.

### Installation and quick start

```bash
pip install sample-factory
```

```bash
# CartPole baseline — note the different CLI style
python -m sf_examples.gym.train_gym_env \
    --env=CartPole-v1 \
    --experiment=cartpole_sf \
    --train_for_env_steps=2_000_000 \
    --num_workers=8 \
    --num_envs_per_worker=2
```

### Godot + Sample Factory

`godot_rl_agents` includes a Sample Factory wrapper. See `godot_rl/wrappers/sample_factory_wrapper.py` in the repo. The one-liner:

```bash
python -m godot_rl.train \
    --backend=sample_factory \
    --env_path=path/to/game.x86_64 \
    --experiment=my_experiment \
    --num_workers=4
```

!!! note "When to reach for Sample Factory"
    If your Godot environment takes more than 50ms per step (physics-heavy, many agents, complex observations), Sample Factory's async workers will give you a meaningful speedup. For simple 2D games running at 60Hz with 4–8 parallel envs, SB3 or CleanRL is sufficient and easier to debug.

---

## 8 · Implementing PPO from scratch (stretch)

This section is for students who want to write the algorithm, not just read it. The skeleton below runs on `CartPole-v1` when the TODOs are filled in. Each TODO maps to a section of this unit and of [unit-ppo-deep.md](unit-ppo-deep.md).

```python
"""
PPO from scratch — skeleton for CartPole-v1.
Fill in every TODO. The file should run as-is when complete.
"""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

# ---------- Hyperparameters ----------
ENV_ID        = "CartPole-v1"
TOTAL_STEPS   = 200_000
NUM_ENVS      = 4
NUM_STEPS     = 128        # steps per rollout per env
UPDATE_EPOCHS = 4
MINIBATCH_SIZE= 64
LR            = 2.5e-4
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_COEF     = 0.2
ENT_COEF      = 0.01
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
BATCH_SIZE    = NUM_ENVS * NUM_STEPS

# ---------- Network ----------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # TODO: define self.actor (obs_dim → act_dim logits)
        # TODO: define self.critic (obs_dim → 1 scalar)
        raise NotImplementedError

    def get_action_and_value(self, obs, action=None):
        # TODO: forward pass through actor → Categorical distribution
        # TODO: sample action if not provided
        # TODO: compute log_prob, entropy, value
        # TODO: return action, log_prob, entropy, value
        raise NotImplementedError

    def get_value(self, obs):
        # TODO: return critic(obs)
        raise NotImplementedError


# ---------- Rollout buffer ----------
def collect_rollout(envs, agent, device):
    """Collect NUM_STEPS steps from NUM_ENVS environments."""
    obs_shape = envs.single_observation_space.shape
    obs     = torch.zeros(NUM_STEPS, NUM_ENVS, *obs_shape)
    actions = torch.zeros(NUM_STEPS, NUM_ENVS, dtype=torch.long)
    logprobs= torch.zeros(NUM_STEPS, NUM_ENVS)
    rewards = torch.zeros(NUM_STEPS, NUM_ENVS)
    dones   = torch.zeros(NUM_STEPS, NUM_ENVS)
    values  = torch.zeros(NUM_STEPS, NUM_ENVS)

    next_obs  = torch.tensor(envs.reset()[0], dtype=torch.float32)
    next_done = torch.zeros(NUM_ENVS)

    for step in range(NUM_STEPS):
        # TODO: store next_obs and next_done
        # TODO: get action, logprob, _, value from agent (no_grad)
        # TODO: step envs, store reward and done
        # TODO: update next_obs and next_done
        raise NotImplementedError

    return obs, actions, logprobs, rewards, dones, values, next_obs, next_done


# ---------- GAE ----------
def compute_gae(rewards, values, dones, next_obs, next_done, agent):
    """
    Compute GAE advantages and returns.
    Returns: advantages (NUM_STEPS, NUM_ENVS), returns (NUM_STEPS, NUM_ENVS)
    """
    advantages = torch.zeros_like(rewards)
    last_gae   = 0.0

    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        # TODO: loop backwards from NUM_STEPS-1 to 0
        # TODO: compute nextnonterminal (mask episode boundaries)
        # TODO: compute delta = r_t + gamma * V(s_{t+1}) * nextnonterminal - V(s_t)
        # TODO: compute advantages[t] using GAE recursion
        raise NotImplementedError

    returns = advantages + values
    return advantages, returns


# ---------- Update step ----------
def ppo_update(agent, optimizer, obs, actions, logprobs, advantages, returns, values):
    """Run UPDATE_EPOCHS passes over the rollout."""
    b_obs       = obs.reshape(-1, *obs.shape[2:])
    b_actions   = actions.reshape(-1)
    b_logprobs  = logprobs.reshape(-1)
    b_advantages= advantages.reshape(-1)
    b_returns   = returns.reshape(-1)

    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

    for epoch in range(UPDATE_EPOCHS):
        inds = np.random.permutation(BATCH_SIZE)
        for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
            mb = inds[start:start + MINIBATCH_SIZE]

            # TODO: forward pass → newlogprob, entropy, newvalue
            # TODO: compute ratio = exp(newlogprob - b_logprobs[mb])
            # TODO: compute clipped policy loss (pg_loss)
            # TODO: compute value loss (MSE against b_returns[mb])
            # TODO: compute entropy bonus
            # TODO: combine: loss = pg_loss - ent_coef*entropy + vf_coef*v_loss
            # TODO: backward, clip grad norm, optimizer step
            raise NotImplementedError


# ---------- Main ----------
if __name__ == "__main__":
    envs  = gym.vector.SyncVectorEnv([
        lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(ENV_ID))
        for _ in range(NUM_ENVS)
    ])
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = envs.single_action_space.n

    agent     = ActorCritic(obs_dim, act_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    for update in range(TOTAL_STEPS // BATCH_SIZE):
        rollout_data = collect_rollout(envs, agent, device="cpu")
        obs, actions, logprobs, rewards, dones, values, next_obs, next_done = rollout_data

        advantages, returns = compute_gae(rewards, values, dones, next_obs, next_done, agent)
        ppo_update(agent, optimizer, obs, actions, logprobs, advantages, returns, values)

        if update % 10 == 0:
            print(f"update {update}/{TOTAL_STEPS // BATCH_SIZE}")

    envs.close()
```

A correct implementation converges CartPole-v1 to 500 in roughly 150–200k steps. If it diverges or never improves, the GAE backwards loop or the ratio computation is the most likely culprit.

---

## 9 · What you now know

| Implementation concept | Role in PPO | SB3 parameter |
|---|---|---|
| `ratio = exp(new_logprob - old_logprob)` | Importance weight measuring policy drift | — (computed internally) |
| `clip_coef` (ε) | Trust region radius — how far the policy can move per update | `clip_range` |
| `ent_coef` | Weight on entropy bonus — controls exploration pressure | `ent_coef` |
| `vf_coef` | Weight on value loss relative to policy loss | `vf_coef` |
| `max_grad_norm` | Gradient clipping threshold — prevents single-step explosions | `max_grad_norm` |
| `num_steps` (T) | Rollout length per environment per update | `n_steps` |
| `update_epochs` | How many passes over the rollout data | `n_epochs` |
| `gae_lambda` (λ) | Bias-variance trade-off in advantage estimation | `gae_lambda` |
| `gamma` (γ) | Discount factor — how much future rewards are discounted | `gamma` |
| `minibatch_size` | Mini-batch size for each gradient step | `batch_size` |
| `approx_kl` | Proxy for KL(π_new ∥ π_old) — monitor for training stability | `target_kl` (stopping) |
| `clipfrac` | Fraction of transitions where ratio was clipped | — (logged only) |

---

## 10 · Stretch goals

These are open-ended. There are no provided solutions — use CleanRL's source and the papers as your guide.

**PPO with LSTM.** Replace the MLP backbone in `Agent` with an LSTM. The rollout loop needs to carry hidden states between steps, and mini-batch construction must respect sequence order (no random shuffling). CleanRL has `ppo_atari_lstm.py` as a reference.

**Curiosity bonus.** Add an intrinsic reward to the CleanRL rollout loop. After `rewards[step] = torch.tensor(reward)`, compute an intrinsic bonus using a Random Network Distillation (RND) module and add it to the stored reward. See [unit-curiosity.md](unit-curiosity.md) for the theory. Watch `charts/episodic_return` split into extrinsic and intrinsic components.

**Godot headless training.** Run a Godot environment without a display using `--headless` (Godot 4) or `--no-window` (Godot 3) flag passed via `env_path` arguments in `StableBaselinesGodotEnv`. Profile whether `SyncVectorEnv` with 8 headless Godot instances is faster than 4 windowed — the answer depends on your machine's CPU core count.

---

[← SAC](unit-sac.md) · [Course home](index.md) · [→ Parallel Training](unit-05.md)
