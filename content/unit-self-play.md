# Self-Play — Training Against Yourself

[← Memory & POMDPs](unit-08.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~30 min · Training: ~30 min GPU / ~2 h CPU

!!! info "Three ways to see your AI"
    - **Godot viewer**: watch two AI paddles, controlled by the same brain, play AirHockey against each other — strategies emerge from nothing.
    - **TensorBoard**: track ELO rating over training. Reward alone is meaningless in self-play (it averages to zero in a zero-sum game); ELO is the proxy for absolute skill.
    - **Checkpoint folder**: a growing archive of past selves. Each `.zip` is a frozen opponent your current policy must learn to beat.

---

## 1 · What self-play is and why it works

In standard reinforcement learning, the environment is **fixed**. A maze has the same walls every episode. A cart-pole always has the same physics. As your policy improves, it learns to exploit that fixed environment — and once it has, learning stalls. The difficulty ceiling is whatever the human designer baked into the simulator.

Self-play removes the ceiling.

**The core idea:** the opponent IS your current policy. When you take an action against an opponent that is a copy (or recent snapshot) of yourself, the opponent's difficulty automatically tracks your own skill level. A beginner plays a beginner. A pro plays a pro. The match is always close to your edge — which is exactly where learning happens fastest.

This makes self-play, by its nature, an **adaptive curriculum**. You never need to hand-tune a difficulty schedule, scale up enemy stats, or design level progressions. The opponent gets harder because you got better, and the only way you got better is by playing a slightly weaker version of the harder opponent you just became.

### The famous result

The defining moment for self-play came from DeepMind. **Silver et al., 2017 (AlphaGo Zero)** showed that starting from completely random play, competing only against itself, with zero human game data, a policy could reach **superhuman performance in Go** — a game humans had studied for 2,500 years. The earlier AlphaGo had been bootstrapped with millions of human games. AlphaGo Zero had none, and crushed it.

That result reframed what reinforcement learning could be. The lesson: with enough compute, the right algorithm, and self-play as the curriculum, an agent can discover strategies that exceed the best humans have ever produced.

### Three landmark systems

- **AlphaGo Zero** (DeepMind, 2017) — Go from scratch in 3 days of self-play. Discovered known human openings and then surpassed them. Crucially: **never saw a human game**.
- **OpenAI Five** (OpenAI, 2018) — Dota 2 at professional level. Trained on the equivalent of **180 years of self-play per day** across thousands of GPUs. Defeated the world champion team OG in 2019.
- **AlphaStar** (DeepMind, 2019) — StarCraft II Grandmaster level. Introduced **league-based self-play**: instead of one opponent, a whole pool of past versions, with "exploiter" agents specifically trained to find weaknesses. The first AI to reach top-0.2% in a complex real-time strategy game.

All three have the same shape: random initialization → self-play → emergent strategy that exceeds human design. Nothing else in RL has the same "compound interest" property.

---

## 2 · Non-stationarity and why self-play helps

Recall from [Unit 7 (Multi-Agent)](unit-07.md): when multiple agents learn simultaneously, each one's environment is **non-stationary**. The transition dynamics `P(s' | s, a)` depend on the *other* agents' policies — and those policies keep changing. From any single agent's perspective, the world keeps shifting under its feet. Convergence guarantees from single-agent RL no longer apply.

Self-play tames this in a specific, useful way:

- The opponent is a **copy of you**, but a **snapshot** — it changes slowly, only when a new checkpoint is saved.
- Between checkpoints, the opponent's policy is frozen. The environment, from the perspective of the learning agent, is **temporarily stationary**.
- After a checkpoint update, the opponent jumps — but only to a slightly better version of itself.

In other words, self-play replaces the chaos of simultaneous learning with a sequence of **stationary intervals separated by small jumps**. The training distribution is always roughly "current skill ± one checkpoint." Stable enough to learn from, fresh enough to keep improving.

Contrast this with naive multi-agent: if both agents update every gradient step, they may chase each other around in policy space and never converge. Self-play sidesteps that by introducing the **deliberate lag** between the learner and the opponent.

---

## 3 · Three self-play training modes

There is no single "self-play algorithm." There is a family of training schemes, each making a different tradeoff between **stability**, **diversity of opponents**, and **implementation complexity**.

### Simple self-play (latest checkpoint)

The most common variant: always play against the **latest** checkpoint of yourself. After every training step, the opponent is essentially the same network as the learner (or a copy made moments ago).

- **Pros**: trivial to implement — just call `model.predict()` on both paddles.
- **Cons**: highly **unstable**. The classic failure mode is **policy oscillation**: the network learns a strategy that beats the current self, the opponent (which is the current self) updates to counter it, the new self counters that, and the network forgets the original strategy. Round-trips through policy space without net progress.

!!! warning "The oscillation trap"
    Simple self-play often shows a hallmark pathology: rock-paper-scissors dynamics. The policy learns to "rush left," then learns to "block left rushes," then forgets the left rush. ELO bounces. Watch your replay videos — if the playstyle from checkpoint 5 looks nothing like checkpoint 10 but also nothing like checkpoint 15, oscillation is real. The fix is one of the next two modes.

### Frozen-opponent self-play

Save a checkpoint every N episodes (or every N gradient steps). For the next training window, **freeze that checkpoint as the opponent** and play only against it. After N more steps, save a new checkpoint and freeze again.

- **Pros**: opponent does not change during a window → the environment is fully stationary inside each window → standard PPO works without modification.
- **Cons**: the policy may converge to **a counter-strategy specific to that one opponent**. When the opponent is updated to a new freeze, performance can dip while the policy retools.

### League-based self-play (the AlphaStar approach)

Maintain a **pool of past checkpoints** — the "league." During training, sample an opponent from the pool for each episode. The pool grows as training progresses, optionally bounded with eviction of weakest members.

- **Pros**: highest stability and highest diversity. The current policy must beat **all past versions**, which prevents forgetting. The pool naturally contains members with different exploitable weaknesses, so the learner must generalize.
- **Cons**: more bookkeeping, more disk space, and you have to choose a sampling strategy (uniform? prioritized toward similar skill? toward exploiters?).

### Mode comparison

| Mode | Stability | Diversity | Complexity | Best for |
|------|-----------|-----------|------------|---------|
| Simple (latest) | Low | Low | Trivial | Prototyping, sanity checks |
| Frozen checkpoint | Medium | Low | Easy | Small games, single-strategy domains |
| League | High | High | Moderate | Complex games, robust policies, production |

!!! tip "Default to league-based for anything serious"
    For research or production training runs, start with league-based self-play. The extra ~50 lines of bookkeeping pay for themselves in stability. Use simple self-play only for quick prototypes or to confirm your environment works at all.

---

## 4 · Open AirHockey

Our environment for this unit is the **AirHockey** example from [godot_rl_agents_examples](https://github.com/edbeeching/godot_rl_agents_examples), in `examples/AirHockey`. It is a clean, minimal two-player zero-sum game — perfect for self-play.

```bash
git clone https://github.com/edbeeching/godot_rl_agents_examples.git
cd godot_rl_agents_examples/examples/AirHockey
```

### The game

- Two paddles (agents), one puck, one goal at each end of the table.
- Each agent observes its own state, the puck state, and the opponent's state.
- Each agent's action is a 2D continuous force vector applied to its paddle.

### The reward — pure zero-sum

Reward is **zero-sum**: when agent 0 scores, agent 0 gets `+1` and agent 1 gets `-1`. When agent 1 scores, the signs flip. The sum of rewards across the two agents is always zero (modulo any small shaping).

```text
r_agent_0 = -1 × r_agent_1
```

This is exactly the setting self-play is built for. There is no notion of "absolute" reward improving — the average return for either agent across a long training run is approximately zero. What changes is *skill*, measured by who is winning, which you cannot read off the reward alone. (That is why Section 6 introduces ELO.)

### Read the controller

Open `ai_controller.gd` in the AirHockey scene. The `get_obs()` function returns roughly:

- own paddle position `(x, y)`
- own paddle velocity `(vx, vy)`
- opponent paddle position `(x, y)`
- opponent paddle velocity `(vx, vy)`
- puck position `(x, y)`
- puck velocity `(vx, vy)`

That is the full state from the agent's view. Both agents share the same observation *shape* but see the world from opposite sides — by convention, the example mirrors the coordinate frame so both agents experience "my goal is behind me, opponent's goal is in front."

### Two binaries vs one binary with two agents

You have two options:

- **One binary, two agents inside**: a single Godot process exposes two `AIController` nodes. The Python side receives two observation batches per step and sends two action batches. This is what we will use — it is what the example ships with.
- **Two binaries**: spawn two separate Godot processes, each controlling one paddle, communicating through a shared puck server. Simpler to reason about but heavier and rarely needed for self-play.

Export the binary as you did in Unit 4:

```bash
# inside the Godot editor, Project → Export → Linux/macOS/Windows
# output: AirHockey.x86_64 (or .app / .exe)
```

---

## 5 · Implement self-play in Python (SB3)

We will use stable-baselines3 with the `StableBaselinesGodotEnv` wrapper from `godot-rl`. Two patterns follow: a single-policy symmetric setup (simplest) and a frozen-opponent setup (more stable).

### Approach A — Single policy, two instances (symmetric self-play)

The simplest pattern. One PPO model controls *both* paddles. The model sees both observations per step, outputs both actions, and gets both rewards. Because the game is symmetric and the policy is shared, every gradient step trains the same network from both sides of the table.

```python
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import os, copy

env = StableBaselinesGodotEnv(
    env_path="./AirHockey.x86_64",
    n_parallel=8,
    speedup=20,
)

# Single PPO model — controls BOTH paddles with same policy
model = PPO(
    "MlpPolicy", env,
    verbose=1,
    tensorboard_log="logs/",
    n_steps=512,
    batch_size=256,
)

checkpoint_interval = 50_000   # save opponent checkpoint every 50k steps
checkpoint_dir = "self_play_checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)

# Simple self-play: save checkpoints, use latest as opponent
for iteration in range(20):
    model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
    checkpoint_path = f"{checkpoint_dir}checkpoint_{iteration}.zip"
    model.save(checkpoint_path)
    print(f"Iteration {iteration}: saved {checkpoint_path}")

model.save("airhockey_selfplay_final")
env.close()
```

This is "simple self-play" from Section 3: the opponent is always the current policy (because it literally is the current policy — same weights). You are saving checkpoints for archival and for evaluation, not for use during training.

It works for AirHockey because the game is short, the action space is small, and oscillation manifests slowly. For richer games it would be unstable.

### Approach B — Frozen opponent (more stable)

To get a stationary opponent, load a **past** checkpoint as the opponent policy. The learner's paddle uses the current model; the opponent's paddle uses a frozen model loaded from disk.

```python
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import os, numpy as np

env = StableBaselinesGodotEnv(
    env_path="./AirHockey.x86_64",
    n_parallel=8,
    speedup=20,
)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/",
            n_steps=512, batch_size=256)

# Bootstrap: do a few iterations of simple self-play to seed a checkpoint
model.learn(total_timesteps=50_000, reset_num_timesteps=False)
model.save("self_play_checkpoints/checkpoint_0.zip")

opponent_model = PPO.load("self_play_checkpoints/checkpoint_0.zip")

def frozen_opponent_step(obs_batch):
    """Split obs into (learner, opponent); learner acts via `model`,
       opponent acts via `opponent_model`. Re-combine actions."""
    # obs_batch shape: (2 * n_parallel, obs_dim) — interleaved agent_0, agent_1
    learner_obs   = obs_batch[0::2]
    opponent_obs  = obs_batch[1::2]
    learner_act, _   = model.predict(learner_obs,   deterministic=False)
    opponent_act, _  = opponent_model.predict(opponent_obs, deterministic=True)
    combined = np.empty((learner_act.shape[0] * 2, learner_act.shape[1]),
                        dtype=learner_act.dtype)
    combined[0::2] = learner_act
    combined[1::2] = opponent_act
    return combined

# In a real loop you'd subclass the env or callback to inject this;
# for clarity we sketch the outer training schedule:
for iteration in range(1, 20):
    # Train against the *currently frozen* opponent
    model.learn(total_timesteps=50_000, reset_num_timesteps=False)
    ckpt = f"self_play_checkpoints/checkpoint_{iteration}.zip"
    model.save(ckpt)
    # Promote the new checkpoint to opponent — the lag is intentional
    opponent_model = PPO.load(ckpt)
    print(f"Iteration {iteration}: opponent promoted to {ckpt}")

env.close()
```

The trick: the opponent's policy is **frozen for the entire 50k-step window**. During that window, the environment is stationary; PPO has a well-defined target to optimize against. At the end of the window, the opponent is upgraded — a discrete, predictable shift.

In practice you would implement `frozen_opponent_step` either by wrapping the environment (so the wrapper sends opponent actions before forwarding observations to PPO) or by exposing only the learner agent to PPO and stepping the Godot env manually. Both patterns are fine; the wrapper approach is cleaner.

---

## 6 · ELO rating for self-play progress

Here is the problem you will hit five minutes after starting your first self-play run: **how do you know it's improving?** In a zero-sum game with a balanced opponent, your mean reward is zero — by definition. You cannot just plot `ep_rew_mean` and watch it climb.

The standard answer borrowed from competitive games: **ELO rating**.

### ELO in 60 seconds

- Every player starts at a baseline rating (commonly **1000**).
- After each match, ratings update based on the outcome **and** the expected outcome.
- Beating someone rated much higher than you gains a lot of rating. Beating someone much weaker gains very little. Losing to someone weaker costs a lot.
- The expected win probability is determined by the rating gap, using the logistic curve:

```text
expected_win = 1 / (1 + 10 ^ ((opponent_elo - your_elo) / 400))
```

A 400-point gap means the higher-rated player wins ~91% of the time.

### Implementation

```python
def update_elo(winner_elo, loser_elo, k=32):
    """Standard ELO update. Returns (new_winner_elo, new_loser_elo)."""
    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    new_winner = winner_elo + k * (1 - expected_winner)
    new_loser  = loser_elo  + k * (0 - (1 - expected_winner))
    return new_winner, new_loser

# After each episode:
# if agent_0 won: agent_0_elo, agent_1_elo = update_elo(agent_0_elo, agent_1_elo)
# else:           agent_1_elo, agent_0_elo = update_elo(agent_1_elo, agent_0_elo)
```

The `k` factor controls how fast ratings change. `k=32` is standard for chess. For self-play with frequent matches, `k=16` or even `k=8` reduces noise.

### Logging to TensorBoard

```python
from stable_baselines3.common.callbacks import BaseCallback

class EloLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.agent_elo = 1000.0
        self.opponent_elo = 1000.0

    def _on_step(self) -> bool:
        # Detect end-of-episode and the winner from info dicts
        for info in self.locals.get("infos", []):
            if "winner" in info:  # set by your env at episode end
                if info["winner"] == 0:
                    self.agent_elo, self.opponent_elo = update_elo(
                        self.agent_elo, self.opponent_elo)
                else:
                    self.opponent_elo, self.agent_elo = update_elo(
                        self.opponent_elo, self.agent_elo)
                self.logger.record("self_play/agent_elo", self.agent_elo)
                self.logger.record("self_play/opponent_elo", self.opponent_elo)
        return True
```

What you want to see on TensorBoard:

- `self_play/agent_elo` rising steadily over training.
- A widening gap between the agent's ELO and a *fixed* baseline opponent's ELO (one of your early checkpoints, never updated).
- For league training: a ladder where later checkpoints sit higher than earlier ones.

If ELO is **flat**, training is not working — even though loss may look fine and reward sits near zero (as it should).

---

## 7 · League training implementation sketch (optional on a first read)

!!! note "First pass? Skim or skip this section."
    The core path through this unit is Sections 1–6 (what self-play is, the AirHockey environment, the three training modes and their two implementation patterns, ELO) plus Section 8 (the viz checkpoint). League training is the production-grade upgrade; Stretch 1 brings you back here when you are ready to build it in full.

League training is the gold standard. Below is a minimal `League` class — fewer than 50 lines — that gets you most of the way to AlphaStar-style training.

```python
import random
from stable_baselines3 import PPO

class League:
    def __init__(self, initial_model_path, max_size=10):
        self.members = [initial_model_path]
        self.max_size = max_size
        self.elos = {initial_model_path: 1000.0}

    def add(self, checkpoint_path, current_elo):
        self.members.append(checkpoint_path)
        self.elos[checkpoint_path] = current_elo
        if len(self.members) > self.max_size:
            # Remove weakest member
            weakest = min(self.members, key=lambda p: self.elos[p])
            self.members.remove(weakest)
            del self.elos[weakest]

    def sample_opponent(self, strategy="uniform"):
        if strategy == "uniform":
            return random.choice(self.members)
        elif strategy == "prioritized":
            # More likely to sample opponents close to current skill
            # (implementation varies)
            return random.choice(self.members[-3:])  # recent checkpoints

    def load_opponent(self, path):
        return PPO.load(path)

    def update_elo(self, opponent_path, agent_won, agent_elo, k=16):
        opp_elo = self.elos[opponent_path]
        if agent_won:
            new_agent, new_opp = update_elo(agent_elo, opp_elo, k)
        else:
            new_opp, new_agent = update_elo(opp_elo, agent_elo, k)
        self.elos[opponent_path] = new_opp
        return new_agent
```

### How the outer training loop uses it

```python
league = League("self_play_checkpoints/checkpoint_0.zip", max_size=10)
agent_elo = 1000.0

for iteration in range(1, 50):
    # Sample a fresh opponent for this training window
    opp_path = league.sample_opponent(strategy="uniform")
    opponent_model = league.load_opponent(opp_path)
    print(f"Iteration {iteration}: opponent = {opp_path} "
          f"(elo {league.elos[opp_path]:.0f})")

    # Train against this frozen opponent for 50k steps
    model.learn(total_timesteps=50_000, reset_num_timesteps=False)

    # Evaluate to update ELO (e.g. 50 deterministic matches)
    wins = evaluate_against(model, opponent_model, n_matches=50)
    agent_elo = league.update_elo(opp_path,
                                   agent_won=(wins > 25),
                                   agent_elo=agent_elo)

    # Add a fresh checkpoint to the league
    new_ckpt = f"self_play_checkpoints/checkpoint_{iteration}.zip"
    model.save(new_ckpt)
    league.add(new_ckpt, agent_elo)
```

### Sampling strategies

- **Uniform**: each league member equally likely. Maximally diverse, but the learner spends a lot of time crushing easy opponents.
- **Prioritized (recent)**: bias toward recent checkpoints. Faster early-stage learning; risks forgetting older opponents.
- **Prioritized (close-elo)**: weight opponents by similarity in ELO. Always plays close matches → max information per game. This is what AlphaStar's "main agents" use.
- **Exploiter sampling** (advanced): keep a few "exploiter" agents specifically trained to beat the current main agent. Sample them often. Forces the main agent to learn robustness.

For this unit, uniform sampling is sufficient. Build from there in the stretch goals.

---

## 8 · Viz checkpoint for self-play

After training, load the final model and run with `--viz`:

```bash
python train_selfplay.py --viz --resume_model_path airhockey_selfplay_final.zip
```

Watch what the AI does. Self-play has very specific behavioral signatures.

### What to look for

- **Does it track the puck or just stand still?** Standing still — or worse, retreating to a corner and never moving — means training failed. The agent learned that doing nothing minimizes the chance of getting scored on, and the reward shaping did not punish passivity enough.
- **Does it block shots or only attack?** If the AI only attacks, the reward likely doesn't penalize being scored on enough relative to scoring. If it only blocks, the opposite. Healthy self-play AIs do both, often switching modes based on puck position.
- **Does it exhibit recognizable strategies?** Watch for: positioning angles to cover the goal, faking moves to draw the opponent, banking shots off the walls, intercepting at the centerline. Emergent strategy is the payoff. **Emergent strategy is the reason you ran self-play.**

### The signature of self-play working

The single most diagnostic test: **load checkpoints from different generations and watch them play each other.**

- Generation 0 (random init): paddles drift, occasional accidental hits.
- Generation 5: paddles track the puck reliably, simple block-and-hit.
- Generation 10: positional play emerges, paddles cover the goal.
- Generation 20: deliberate angles, fakes, recoveries.

If checkpoint 5 plays identically to checkpoint 10 plays identically to checkpoint 20 — training stalled. If each generation **plays noticeably differently and beats the previous generation more often than it loses** — self-play is working.

This is the visual analog of ELO rising. If you only have time for one diagnostic, run a round-robin tournament between checkpoints and tabulate win rates.

!!! check "Done when"
    Do **not** wait for `ep_rew_mean` to climb — in a zero-sum self-play run it hovers near zero by design, even when training is going well, because both sides improve together. You are done when the unit's own diagnostics agree: `self_play/agent_elo` trends upward in TensorBoard while a fixed early checkpoint's ELO stays flat or falls (zero-sum updates actively push a losing baseline down), and a late checkpoint beats an early one clearly more often than it loses — single games are coin flips, so use Section 8's round-robin tally over enough games for the win rate to be meaningful, not a best-of-three. Identical-looking play across generations means stalled training, not a finished run.

---

## 9 · Self-play for cooperative tasks

Self-play is usually introduced as a competitive technique, but it works equally well for cooperative tasks. The mechanism is the same: train against a frozen copy of yourself.

### Cooperative example

Imagine two robots that must coordinate to carry a heavy object through a doorway. Neither can do it alone. They must move in sync, lift together, navigate together.

One shared policy controls both robots. During training:

- The **learner's robot** uses the live policy.
- The **partner robot** uses a frozen checkpoint of the same policy.

Both robots receive the same shared reward (object delivered → both win). The learner improves; periodically, the partner is updated to a newer checkpoint.

Why frozen rather than live? Because of the same non-stationarity argument from Section 2. With both robots learning live, the coordination signal is noisy — your partner moves differently every gradient step. With a frozen partner, you have something stable to coordinate **with**.

### Difference from multi-agent shared policy

This is subtle but important:

- **Multi-agent shared policy (Unit 7)**: one network controls all agents, all gradient updates happen on rollouts from all agents simultaneously. The "partner" updates every step.
- **Cooperative self-play**: one network is the learner, a frozen copy is the partner. The partner only updates when you promote a new checkpoint.

The first is faster but unstable for complex coordination. The second is slower but produces more robust cooperative policies, because the learner must coordinate with multiple historical versions of itself across training.

---

## 10 · Connection to RLHF and alignment

Self-play is more than a game-playing trick — it is a foundational pattern in modern alignment research.

### Constitutional AI (Anthropic, 2022)

The model critiques its own outputs against a set of principles, then revises them. The critic and the writer are the **same model**, applied in two roles. This is self-play in the textual domain: the model improves by playing against a copy of itself acting as judge.

The training loop:

1. Model writes a response.
2. Same model (or a frozen copy) critiques the response per a constitution.
3. Model revises the response based on the critique.
4. RLHF-style fine-tuning rewards revisions that align better with the constitution.

### Debate (Irving, Christiano, Amodei, 2018)

Two AI debaters argue opposite sides of a question. A human judge picks the winner. Neither debater knows in advance which side is "correct"; both must construct the strongest case they can. The pressure of the adversary forces honest, well-supported argument — in theory, more so than a single AI answering directly.

The structural parallel to AirHockey is exact: two policies (possibly the same network in two roles) compete in a zero-sum game; the score is decided by an external judge (the human, or eventually another AI). Improvement is driven by self-play.

### Why this matters for the course

The alignment-focused sequel course covers RLHF, Constitutional AI, and debate directly. **Self-play is the conceptual on-ramp** to all of them. Once you understand what it means to train against a frozen copy of yourself, the leap to training a language model that critiques itself is small.

---

## 11 · Stretch goals

Pick one (or more) of these to deepen your understanding of self-play.

### Stretch 1: Build a league

Implement the `League` class from Section 7 in full. Train 10 generations on AirHockey with uniform sampling. After each generation, run a round-robin tournament inside the league, computing ELO for every member. Plot ELO curves over time.

You should see: later generations have monotonically rising ELO; the original random checkpoint sinks to the bottom; the curves fan out into a clear hierarchy.

### Stretch 2: Asymmetric self-play

Modify the AirHockey environment so the two paddles have **different observation spaces**:

- Paddle A sees the full observation (own state, opponent state, puck state).
- Paddle B sees only its own state and the puck — **not the opponent**.

Train with shared-policy self-play. Which paddle dominates? More interestingly: what does the **weaker paddle (B)** learn to compensate? Often it learns conservative defensive play — covering the goal without trying to predict opponent moves. This is a tiny taste of partial-observability dynamics from [Unit 8](unit-08.md).

### Stretch 3: Self-play on Racer

The Racer environment from Unit 7 supports competitive multi-agent racing. Currently you trained it with independent parallel agents (each car has its own policy). Re-implement using **league-based self-play**: one shared policy, sampling opponents from a league of past checkpoints.

Compare:

- Does the self-play version learn faster than independent training?
- Does it produce more diverse driving strategies?
- Is the final policy more robust against held-out test opponents (e.g., a checkpoint from independent training)?

This is the closest thing to a research-grade self-play experiment you can run on a single GPU in an evening.

---

## What's next

Self-play closes the loop on Phase 4. You have now seen:

- **Multi-agent** ([Unit 7](unit-07.md)) — multiple policies, possibly cooperative or competitive.
- **Memory & POMDPs** ([Unit 8](unit-08.md)) — handling partial observability with recurrence.
- **Self-play** (this unit) — using yourself as the curriculum.

**Hierarchical RL:** Long-horizon tasks where flat PPO stalls — multi-room navigation, multi-step assembly — can be decomposed with a high-level policy that sets subgoals and a low-level policy that achieves them.

[→ Hierarchical RL](unit-hierarchical.md)

---

[← Memory & POMDPs](unit-08.md) · [Course home](index.md) · [→ Hierarchical RL](unit-hierarchical.md)
