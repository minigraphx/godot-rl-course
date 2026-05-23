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

## 6 · Reading multi-agent TensorBoard curves

With a shared policy across N agents, `ep_rew_mean` reports the mean across all agents and all episodes. Look for:

- **Cooperative:** reward should rise together — if it plateaus early, one agent may be "free-riding" (not contributing). Add an individual action penalty to break this.
- **Competitive:** reward for one agent rising while another falls is expected. Check that neither agent converges to a trivial strategy (e.g. standing still).

---

## 7 · Viz checkpoint

Watch 3–5 episodes in the Godot editor:

- **Cooperative:** Do both agents move toward the goal, or does one stand idle?
- **Competitive/racing:** Do agents avoid each other or crash repeatedly?
- **Mixed actions:** Is steering smooth or does it oscillate? Check normalization if jerky.

---

## 8 · Stretch goals

- **Self-play on Racer** — load the latest checkpoint as the opponent; retrain against it iteratively
- **Add a third agent** — duplicate an `AIController` node and retrain; observe how shared-policy quality changes
- **Individual reward** — modify MultiAgentSimple to give each agent its own reward based on its personal contribution

---

## What's next

**Unit 8:** Memory & POMDPs — FPS / RobotFPS, RecurrentPPO, LSTM policy networks for partially-observable environments.

[→ Unit 8: Memory & POMDPs](unit-08.md)
