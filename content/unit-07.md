# Unit 7 — Multi-Agent

Train multiple agents simultaneously in the same environment — some cooperating, some competing. Study **Racer** (mixed discrete+continuous actions) and **MultiAgentSimple** (shared vs independent policies).

[← Unit 6: Continuous 3D](unit-06.md) · [Course home](index.md)

---

!!! info "Three ways to see your AI"
    Godot (watch agents interact — cooperation / competition is obvious visually) · TensorBoard (per-agent reward curves) · `AIController` policy sharing

---

## 1 · Multi-agent fundamentals

In single-agent RL, one policy controls one agent. Multi-agent extends this in two ways:

**Cooperative** — agents share a reward signal and work together (e.g. coordinating to push a block). Training often uses a **shared policy**: all agents run the same network, reducing sample requirements.

**Competitive** — agents have opposing rewards (e.g. racing). Training uses **independent policies**: each agent learns its own strategy, potentially via **self-play**.

**Mixed** — most real environments mix both (teammates on opposing teams).

| Setup | Policy count | Reward | Example |
|-------|-------------|--------|---------|
| Cooperative | 1 shared | Sum or mean | MultiAgentSimple |
| Competitive | N independent | Per-agent | Racer |
| Self-play | 1 (plays itself) | Win/loss | AirHockey |

---

## 1b · MARL: The non-stationarity problem

!!! warning "Non-stationarity breaks single-agent theory"
    Standard RL theory assumes the environment is **stationary** — the same policy produces the same distribution of transitions over time. In multi-agent RL this assumption is violated by design.

In single-agent RL, the Markov assumption holds: the environment dynamics $P(s'|s,a)$ do not change. The agent can safely update its policy against a fixed target.

**In MARL, other agents ARE part of the environment — and they're learning too.**

From agent A's perspective, agent B is a non-stationary component of the environment. As B updates its policy, the transition distribution that A experiences changes, even if the underlying game rules do not. This means:

- The optimal policy for A changes every time B updates its policy
- Q-values estimated by A become stale as soon as B takes a gradient step
- Basic policy gradient convergence guarantees no longer apply in general

**Consequence for Q-Learning:** The standard Q-Learning convergence proof requires a stationary MDP. With multiple learning agents, the MDP is non-stationary from any individual agent's point of view. Q-values can oscillate rather than converge.

**Consequence for policy gradients:** The policy gradient theorem assumes the value function is computed under a fixed environment. With other learning agents present, the value function target shifts continuously.

**In practice:** Non-stationarity does not always prevent learning. It often still works well, especially with:

- **Shared policies** — if all agents run the same network, "other agents" and "self" are the same entity; updates are consistent
- **Self-play** — playing against yourself introduces a controlled form of non-stationarity that can be managed
- **Large populations** — with many agents, the aggregate behavior changes slowly; any single agent sees near-stationarity
- **Cooperative tasks with sparse interaction** — agents that rarely influence each other experience little non-stationarity in practice

Understanding this problem motivates the more sophisticated training paradigms described below.

---

## 2 · How godot-rl handles multiple agents

Each agent has its own `AIController` node. The `Sync` node discovers all `AIController` nodes in the scene and routes observations/actions to each one individually.

```
TrainingScene (Node2D)
  ├─ Sync
  ├─ Agent_0
  │   ├─ ... physics nodes ...
  │   └─ AIController    ← unique per agent
  ├─ Agent_1
  │   ├─ ... physics nodes ...
  │   └─ AIController
  └─ Agent_2
      ...
```

All `AIController` nodes must implement the same `get_obs()` and `get_action_space()` interface. The Sync node collects observations from all, sends them to Python as a batched observation, and routes actions back.

**Shared policy** — the default: Python treats all agents as a single vectorized env. All agents see the same network weights.

**Independent policies** — instantiate separate `StableBaselinesGodotEnv` wrappers, one per agent group, each with its own model.

---

## 2b · Independent Learners (IL)

The simplest approach to MARL: each agent runs its own RL algorithm and treats all other agents as part of the (non-stationary) environment. No coordination or communication between agents is required during training.

**Pros:**

- Simple to implement — standard single-agent algorithms work out of the box
- Scales to many agents without architectural changes
- No inter-agent communication channel required at training or inference time

**Cons:**

- Theoretically unsound due to non-stationarity (see section 1b)
- May fail on tasks requiring tight coordination — agents cannot account for each other's learning
- Convergence is not guaranteed; training can be unstable

**When IL works in practice:**

- **Competitive / racing settings** — agents are opponents; emergent competition drives learning even without coordination
- **Large-population tasks** — with many agents, no single agent dominates the non-stationarity
- **Emergent behavior research** — IL agents often develop surprisingly complex social behaviors despite the theoretical limitations

In godot-rl, running separate `StableBaselinesGodotEnv` instances with independent `PPO` models is the simplest form of IL.

---

## 2c · Centralized Training / Decentralized Execution (CTDE)

!!! tip "CTDE: the practical gold standard for cooperative MARL"
    Train with access to global information. Execute with only local observations. You get the best of both worlds.

**The core idea:**

During **training**, agents share observations and actions with a centralized critic. The critic can see the full global state — all agents' positions, velocities, and actions — which gives it much better value estimates than any single agent could compute alone.

During **execution** (inference), each agent acts only on its own local observation. No communication channel between agents is needed at runtime.

**Why this is powerful:**

- The centralized critic solves the non-stationarity problem: it conditions on all agents' policies simultaneously, so the value target is stable
- The decentralized actor makes deployment practical: each agent runs its own network copy independently, just like a single-agent system

**Canonical algorithms:**

| Algorithm | Base | Notes |
|-----------|------|-------|
| MADDPG | DDPG | Centralized Q-function per agent; continuous actions |
| MAPPO | PPO | Shared or per-agent critic with global observations |

**In Godot terms:**

During Python training, the `Sync` node batches all agent observations together and passes them to the Python side. A CTDE implementation would feed this joint observation to the critic while feeding only per-agent slices to each actor.

At inference, you export one ONNX model per actor (or one shared model). Each `AIController` runs its own copy locally — no Python process, no inter-agent communication. This maps perfectly to CTDE's decentralized execution phase.

> For agents that additionally need memory across timesteps, see [Unit 8 — Memory & POMDPs](unit-08.md), which covers RecurrentPPO and LSTM policies — a common combination with CTDE in partially-observable multi-agent environments.

---

## 3 · Mixed actions — Racer

**Racer** uses a **mixed** action space: discrete gear selection + continuous steering and throttle.

```gdscript
func get_action_space() -> Dictionary:
    return {
        "steering":  {"size": 1, "action_type": "continuous"},   # [-1, 1]
        "throttle":  {"size": 1, "action_type": "continuous"},   # [-1, 1]
        "gear":      {"size": 3, "action_type": "discrete"},     # 0=brake 1=neutral 2=drive
    }

func set_action(action) -> void:
    var steer    = action["steering"][0]
    var throttle = action["throttle"][0]
    var gear     = int(action["gear"])
    vehicle.steering = steer * max_steering
    vehicle.engine_force = (gear == 2) ? throttle * max_force : 0.0
    vehicle.brake = (gear == 0) ? abs(throttle) * max_brake : 0.0
```

SB3's `PPO` with `MultiInputPolicy` handles mixed spaces automatically — no changes needed on the Python side.

---

## 4 · Open the examples

**Racer:**

1. Clone [godot_rl_agents_examples](https://github.com/edbeeching/godot_rl_agents_examples) → `examples/Racer`
2. Open in Godot .NET, enable plugin
3. Read `ai_controller.gd`: note the mixed `get_action_space()`, the per-lap reward, and how agents are placed around the track
4. Count AIController nodes in the training scene — this is your implicit batch size

**MultiAgentSimple:**

1. `examples/MultiAgentSimple` — two agents, one ball, cooperative push
2. Note: both `AIController` nodes have identical `get_obs()` and `get_action_space()` → shared policy works
3. Reward: positive when ball reaches target, shared between both agents

---

## 5 · Train

**Racer (competitive, independent agents):**

```bash
conda activate godot_env

# Export binary first, then:
gdrl --env_path=./Racer.x86_64 \
  --experiment_name=racer_ppo \
  --timesteps=3_000_000 \
  --n_parallel=4 \
  --speedup=20 \
  --n_steps=512 \
  --batch_size=256
```

**MultiAgentSimple (cooperative, shared policy):**

```bash
gdrl --env_path=./MultiAgentSimple.x86_64 \
  --experiment_name=multiagent_coop \
  --timesteps=1_000_000 \
  --n_parallel=8 \
  --speedup=20
```

---

## 6 · Reward shaping in multi-agent settings

Reward design is harder in multi-agent settings than in single-agent ones. The wrong reward structure can silently produce agents that appear to train but learn degenerate strategies.

### Cooperative settings

**Shared reward (easy to implement, hard to optimize):**
All agents receive the same team reward signal. Simple to set up, but creates the **free-rider problem**: one agent learns a useful behavior, and others discover that doing nothing still earns positive reward. Over time, the active agent may stop contributing too.

**Shaped individual rewards (harder to implement, better coordination):**
Give each agent a reward proportional to its personal contribution to the team outcome. A common heuristic:

```
individual_bonus = team_reward × (agent_contribution / total_contribution)
```

This requires measuring per-agent contribution, which is environment-specific (distance moved, contacts with ball, damage dealt, etc.).

**Fix for free-riding:** Add an individual action penalty — small negative reward for agents that remain idle when the team reward is positive. This breaks the "do nothing, share reward" equilibrium.

### Competitive settings

**Zero-sum:** One agent's reward is the exact negative of the other's (`r_A = -r_B`). Theoretically clean, but can produce overly conservative play if both agents learn to avoid losing rather than trying to win.

**Independent rewards:** Each agent gets its own reward signal based on its own performance metrics, without explicitly penalizing the opponent. Less theoretically grounded but often easier to tune.

### Mixed cooperative-competitive settings

Add a weighted combination of team and individual reward terms:

```
r_agent = α × team_reward + (1 - α) × individual_reward
```

Tune `α` during training. Start with `α = 1.0` (fully shared) to establish baseline cooperative behavior, then reduce `α` to encourage individual specialization.

---

## 7 · Reading multi-agent TensorBoard curves

With a shared policy across N agents, `ep_rew_mean` reports the mean across all agents and all episodes. Look for:

- **Cooperative:** reward should rise together — if it plateaus early, one agent may be "free-riding" (not contributing). Add an individual action penalty to break this.
- **Competitive:** reward for one agent rising while another falls is expected. Check that neither agent converges to a trivial strategy (e.g. standing still).

---

## 7b · Self-play

Self-play is a training technique where an agent's opponent is a copy of the agent's own policy. It avoids the need for hand-crafted opponents and scales naturally as the agent improves.

**Simple self-play:** Always play against the latest policy checkpoint. The agent trains against an opponent that is exactly as good as itself. This can cause **strategic oscillation**: the agent learns to beat its current self, but the opponent (now updated) has learned the same counter-strategy, and the cycle repeats without clear progress.

**League-based self-play (AlphaStar-style):** Maintain a pool of past checkpoints. Sample opponents from the pool according to a priority schedule (recent checkpoints more often, historical checkpoints occasionally). This prevents oscillation by ensuring the agent remains robust against a variety of opponent strategies, not just the current one.

**Practical self-play in Godot:**

1. Duplicate the training scene to have two agent slots
2. Load the current checkpoint as the "frozen opponent" policy
3. Train the "learner" policy against the frozen opponent
4. Every N episodes (or when win rate exceeds a threshold), copy the learner's weights to the opponent slot
5. Repeat

```bash
# Pseudocode — actual implementation depends on wrapper
gdrl --env_path=./AirHockey.x86_64 \
  --experiment_name=selfplay_v1 \
  --timesteps=5_000_000 \
  --opponent_policy=checkpoints/selfplay_v1_latest.zip \
  --self_play_swap_freq=50000
```

**Connection to non-stationarity:** Self-play introduces a controlled, scheduled form of non-stationarity. Because the opponent policy only changes at explicit swap steps (not every gradient update), the training MDP is approximately stationary between swaps. This is why simple self-play often works in practice despite the theoretical concerns in section 1b.

This connects directly to the **stretch goal** in section 8: implement league-based self-play on Racer.

---

## 8 · Viz checkpoint

Watch 3–5 episodes in the Godot editor:

- **Cooperative:** Do both agents move toward the goal, or does one stand idle?
- **Competitive/racing:** Do agents avoid each other or crash repeatedly?
- **Mixed actions:** Is steering smooth or does it oscillate? Check normalization if jerky.

---

## 9 · Stretch goals

- **Self-play on Racer** — load the latest checkpoint as the opponent; retrain against it iteratively (see section 7b for the full procedure)
- **Add a third agent** — duplicate an `AIController` node and retrain; observe how shared-policy quality changes
- **Individual reward** — modify MultiAgentSimple to give each agent its own reward based on its personal contribution (see section 6 for reward shaping strategies)
- **League self-play** — maintain a pool of 5 past checkpoints; sample opponents by recency weighting

---

## What's next

**Unit 8:** Memory & POMDPs — FPS / RobotFPS, RecurrentPPO, LSTM policy networks for partially-observable environments. RecurrentPPO is also the standard choice when combining memory with CTDE in cooperative multi-agent tasks.

[→ Unit 8: Memory & POMDPs](unit-08.md)
