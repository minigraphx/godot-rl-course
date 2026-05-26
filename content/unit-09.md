# Unit 9 — Imitation Learning

Skip the reward engineering entirely. Record an expert playing the game, then train a policy to **copy that behavior**. Study **MultiLevelRobot**, record demonstrations, run **Behavioral Cloning (BC)**, and optionally extend to **GAIL**.

[← Multi-Task RL](unit-multitask.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~30 min · Training: ~20 min GPU / ~1.5 h CPU

---

!!! info "Three ways to see your AI"
    Godot (record yourself, watch the clone) · TensorBoard (`train/loss` for BC; `ep_rew_mean` for GAIL) · demonstration replay: step through your own recorded actions

---

## 1 · Why imitation learning?

Reward shaping takes iteration. Imitation learning sidesteps the reward design problem entirely: instead of telling the agent *what to maximize*, you show it *what to do*.

Two main approaches:

| Method | What it learns from | Algorithm | Reward needed? |
|--------|--------------------|-----------|----|
| **Behavioral Cloning (BC)** | Expert trajectories | Supervised classification/regression | No |
| **GAIL** | Expert trajectories | Adversarial (discriminator + generator) | No (intrinsic) |
| **DAgger** | Interactive expert corrections | Iterative supervised | Partially |

**BC** is the simplest: treat each (observation, action) pair as a supervised example, train a policy to predict the expert's action. Fast to train, brittle to distribution shift.

**GAIL** trains a discriminator to distinguish expert from agent trajectories, using its output as an intrinsic reward. Slower but more robust.

!!! note "The alignment connection"
    BC on expert demonstrations (learning to clone human actions from transitions) is the direct analogue of **supervised fine-tuning (SFT)** in language model alignment — the conceptual on-ramp to the sequel course on RL from human feedback.

---

## 2 · MultiLevelRobot

**MultiLevelRobot** is a 3D platformer robot that must navigate across platforms of varying height. Reward engineering is tricky (small platforms, long fall distances). It's an ideal imitation learning candidate: a human can demonstrate the route easily; the agent cannot discover it efficiently by random exploration.

1. Clone [godot_rl_agents_examples](https://github.com/edbeeching/godot_rl_agents_examples) → `examples/MultiLevelRobot`
2. Open in Godot .NET, enable plugin
3. Read `ai_controller.gd`: observations include body velocity, ground raycasts, and platform distances; actions are continuous (jump force + lateral movement)

---

## 3 · Record expert demonstrations

Godot RL Agents supports a **human heuristic mode**: set `Control Mode` on the Sync node to `HUMAN`, then play the game yourself. The sync node records every (obs, action) pair.

**Step-by-step:**

1. Open `training_scene.tscn`
2. Select the `Sync` node → set `Control Mode` to `HUMAN`
3. Run the scene in the Godot editor
4. Play through several complete episodes (aim for 20–50 successful runs)
5. Stop the scene — demonstrations are saved to `demonstrations.json` (or the path set in Sync properties)

```gdscript
# No code changes needed — the Sync node handles recording automatically
# Check Sync node properties for:
#   record_demonstrations = true
#   demonstrations_path = "res://demonstrations.json"
```

!!! tip "Quality over quantity"
    20 high-quality demonstrations (reaching the goal every time) beat 200 mediocre ones. Re-record if you fell off platforms more than once per run.

---

## 4 · Install the imitation library

```bash
conda activate godot_env
pip install imitation
```

`imitation` provides BC, GAIL, DAgger, and AIRL on top of Stable Baselines 3.

---

## 5 · Behavioral Cloning

```python
import numpy as np
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from imitation.algorithms import bc
from imitation.data import rollout
import json

# Load the environment (needed to define obs/action spaces)
env = StableBaselinesGodotEnv(
    env_path="./MultiLevelRobot.x86_64",
    n_parallel=1,
    speedup=1,
)

# Load recorded demonstrations
with open("demonstrations.json") as f:
    demo_data = json.load(f)

# Convert to imitation Transitions format
obs      = np.array([d["obs"]    for d in demo_data])
acts     = np.array([d["action"] for d in demo_data])
dones    = np.array([d["done"]   for d in demo_data])
next_obs = np.roll(obs, -1, axis=0)

transitions = rollout.Transitions(
    obs=obs[:-1],
    acts=acts[:-1],
    infos=np.array([{}] * (len(obs) - 1)),
    next_obs=next_obs[:-1],
    dones=dones[:-1],
)

# Build a PPO policy to clone into
policy = PPO("MlpPolicy", env, verbose=0)

# Behavioral Cloning trainer
trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    policy=policy.policy,
    rng=np.random.default_rng(42),
)

trainer.train(n_epochs=50)
policy.policy = trainer.policy
policy.save("multilevel_bc")
env.close()
```

```bash
conda activate godot_env
python train_bc.py
```

**What to expect:** Loss drops quickly in the first 10 epochs. After 50 epochs, the policy mimics the expert's movement but may fail on slightly different platform layouts (distribution shift).

---

## 6 · Fine-tune with PPO after BC

BC gives a strong starting point. A short PPO fine-tuning run often fixes distribution shift:

```python
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(
    env_path="./MultiLevelRobot.x86_64",
    n_parallel=8,
    speedup=20,
)

# Load BC-initialized policy and continue with PPO
model = PPO.load("multilevel_bc", env=env)
model.learn(total_timesteps=500_000, tensorboard_log="logs/")
model.save("multilevel_bc_finetune")
env.close()
```

In TensorBoard, `ep_rew_mean` should start significantly higher than a random-init PPO run — the BC policy gives the agent a head start into the useful part of the state space.

---

## 7 · GAIL (optional)

GAIL trains a discriminator alongside the policy. The discriminator predicts "is this trajectory from the expert or the agent?"; the agent receives a reward for fooling it.

```python
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3 import PPO

env = StableBaselinesGodotEnv(env_path="./MultiLevelRobot.x86_64", n_parallel=4, speedup=20)

learner = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/")
reward_net = BasicRewardNet(env.observation_space, env.action_space)

gail_trainer = GAIL(
    demonstrations=transitions,   # from section 5
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

gail_trainer.train(total_timesteps=1_000_000)
learner.save("multilevel_gail")
env.close()
```

GAIL is slower but generalises better. Use it if BC fine-tuning still fails on novel platform arrangements.

---

## 8 · Viz checkpoint

Re-run with the Godot editor (Sync → `ONNX_INFERENCE` after export, or open the scene and load the model manually):

- Does the robot follow the route you demonstrated?
- Does it recover when it slides off-path slightly (GAIL / fine-tune)? Or does it freeze (pure BC)?
- Compare: run the BC-only model, the fine-tuned model, and a scratch PPO baseline side-by-side

A good BC agent looks "human-like" — it hesitates at the same spots you hesitated, takes the same path. If it drifts into an untrained state, GAIL fails gracefully; pure BC fails catastrophically.

---

## 9 · Stretch goals

- **DAgger** — iterative imitation: run the policy, let the expert label the new states it visits, retrain. Fixes distribution shift systematically. Available in `imitation.algorithms.dagger`.
- **Compare data efficiency** — how many demonstrations does BC need to match 500k PPO steps?
- **Mixed reward** — combine GAIL's intrinsic reward with a sparse environment reward (reaching the goal) to keep the policy on-task

---

## What's next

**RLHF & Preference Learning:** You've learned to clone behaviour from demonstrations. What if you only have a designer's *taste* — no explicit reward, just pairwise preferences? RLHF turns human judgement into a reward model that guides policy fine-tuning.

[→ RLHF & Preference Learning](unit-rlhf.md)
