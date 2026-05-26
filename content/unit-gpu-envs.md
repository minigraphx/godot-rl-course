# GPU-Accelerated RL Environments — Research-Scale Alternatives to Godot

!!! info "Time"
    Reading: ~20 min

[Course home](index.md)

---

!!! info "Three ways to see the difference"
    - **Godot** — open Task Manager while training BallChase with 16 parallel envs. Watch all CPU cores peg at 100 % while the GPU sits nearly idle.
    - **TensorBoard** — notice your `rollout/ep_rew_mean` curve is smooth but slow. That smoothness costs wall-clock time.
    - **This unit** — a map of every major GPU env framework, when each one is worth the setup pain, and a working EnvPool + SB3 example you can run today.

---

## 1 · Why CPU environments are a bottleneck (and when they aren't)

Every Godot training step touches more machinery than the neural network update does.

A single environment step in Godot involves:

1. The Python training process calling `env.step(action)` over a WebSocket connection
2. The Godot engine running its physics tick (GodotPhysics or Jolt)
3. Godot rendering the scene, even in headless mode
4. The observation being serialised, sent back over the WebSocket, and deserialised in Python
5. Your Python code batching observations and forwarding them to the GPU for inference
6. The GPU returning actions, which travel back over WebSocket to Godot

The GPU is busy for step 5. Everything else is CPU + IPC latency.

**With godot-rl-agents' 20× time-speedup and 16 parallel environments you can reach roughly 30–50 k steps per second on a modern desktop.** That sounds fast until you check what research-scale training actually requires.

| Benchmark | Approximate steps to convergence |
|-----------|----------------------------------|
| BallChase (this course) | 500 k – 2 M |
| Humanoid walk (MuJoCo) | 10 M – 50 M |
| Dexterous hand manipulation | 100 M – 500 M |
| NeurIPS locomotion baselines | 1 B – 10 B |

At 40 k steps/sec, 1 B steps takes roughly **7 hours of wall-clock time**. At 5 M steps/sec (Brax on a single GPU), the same 1 B steps takes about **3 minutes**.

!!! warning "But this course does not need that"
    Every task in this course converges comfortably in the left-hand column: well under 10 M steps. At 40 k steps/sec that is at most a few hours, often under 30 minutes. **CPU environments are the right tool for this course.** This unit exists so you know where to go when you outgrow them.

### Decision table: do you need GPU environments?

| Situation | Recommendation |
|-----------|---------------|
| Training this course's tasks | **Godot CPU — you are fine** |
| Total budget < 50 M timesteps | **Godot CPU — you are fine** |
| Comparing results to a SOTA paper | Maybe — check what framework the paper used |
| Running ablations across 20+ hyperparameter configs | Consider EnvPool for the parallelism boost |
| Writing a NeurIPS submission | Almost certainly yes |
| Shipping an agent that lives inside a Godot game | **Godot — nothing else makes sense** |

---

## 2 · The four main options

### Isaac Gym / Isaac Lab (NVIDIA)

Isaac Gym simulates thousands of rigid-body environments simultaneously on a single GPU by keeping all physics state in GPU memory and never copying it back to the CPU between steps. Isaac Lab is the newer, modular successor built on Isaac Sim.

**Best for:** robot manipulation, legged locomotion, any task that maps cleanly onto rigid-body physics — similar to this course's Phase 6 robotics units.

**Hardware requirement:** NVIDIA GPU (RTX or data-centre class). AMD GPUs are not supported.

```bash
# Isaac Gym requires a manual download from NVIDIA's developer portal.
# After downloading isaacgym-1.0rc4.tar.gz:
pip install isaacgym/
# Isaac Lab (open-source, actively maintained):
pip install isaaclab
```

A minimal PPO training loop with Isaac Lab looks like this:

```python
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from stable_baselines3 import PPO

# Isaac Lab environments return GPU tensors directly.
# The VecEnv wrapper converts them to numpy for SB3.
from isaaclab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper

cfg = ManagerBasedRLEnvCfg()          # task-specific config
cfg.scene.num_envs = 2048             # 2048 envs on one GPU
isaac_env = ManagerBasedRLEnv(cfg=cfg)
env = Sb3VecEnvWrapper(isaac_env)

model = PPO("MlpPolicy", env, verbose=1, n_steps=32, batch_size=512)
model.learn(total_timesteps=100_000_000)
```

!!! warning "Limitations"
    - Setup is involved: specific CUDA version, specific driver, specific Python version. Expect to spend an afternoon on installation.
    - Headless rendering is limited — you can log metrics but visual debugging is harder.
    - Isaac Gym (the older API) is being deprecated in favour of Isaac Lab. Prefer Isaac Lab for new projects.

---

### Brax (Google JAX)

Brax implements a full differentiable physics engine in JAX. Because JAX JIT-compiles computation graphs for GPU and TPU, you can run 4 096 + parallel environments entirely on the accelerator with zero CPU involvement between steps.

**Best for:** algorithm research where end-to-end differentiability matters, or TPU-scale experiments on Google Cloud.

```bash
pip install brax
```

A PPO-style training loop with Brax:

```python
import jax
import jax.numpy as jnp
from brax import envs
from brax.training.agents.ppo import train as ppo_train

# "ant" runs 4096 envs in parallel on GPU/TPU — all in JAX.
make_inference_fn, params, metrics = ppo_train(
    environment=envs.get_environment("ant"),
    num_timesteps=50_000_000,
    num_evals=10,
    reward_scaling=10,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=4096,
    batch_size=2048,
    seed=0,
)
```

!!! info "Brax and SB3"
    Brax ships its own PPO and SAC implementations that are tightly coupled to JAX. You can wrap a Brax environment for SB3 via `brax.io.torch`, but you lose most of the speed advantage because data must move between JAX (GPU) and PyTorch (GPU) each step. For Brax, the native training loops are the intended path.

!!! warning "Limitations"
    - Physics accuracy is simplified compared to MuJoCo. Locomotion behaviours learned in Brax may not transfer well to hardware.
    - JAX has a learning curve if you are a pure PyTorch user.
    - Differentiability is powerful but mostly matters for gradient-through-sim research, not standard model-free RL.

---

### EnvPool

EnvPool is not a GPU physics engine. It is a **C++ multi-threaded environment pool** that replaces Python's `multiprocessing`-based `SubprocVecEnv`. The speedup comes from eliminating Python GIL overhead and inter-process communication serialisation, not from moving physics to GPU.

**Best for:** Atari-scale research, classic MuJoCo, and DMControl — anywhere you want more CPU parallelism without the complexity of a full GPU physics stack.

```bash
pip install envpool
```

EnvPool is the easiest upgrade from the course's existing SB3 workflow. See Section 6 for a full working example.

!!! info "SB3 compatibility"
    EnvPool environments implement the Gymnasium `VectorEnv` interface. A thin wrapper (shown in Section 6) makes them fully compatible with SB3's `learn()` call.

---

### MJX (MuJoCo XLA)

MJX compiles the full MuJoCo physics engine to XLA (the same intermediate representation JAX uses), enabling MuJoCo-quality rigid-body simulation on GPU and TPU. Unlike Isaac Gym, it does not require NVIDIA hardware — it runs on any XLA backend.

**Best for:** research that needs MuJoCo's simulation accuracy (joint limits, contact dynamics, tendon constraints) at GPU scale.

```bash
pip install mujoco mjx
```

```python
import jax
import mujoco
import mujoco.mjx as mjx

model = mujoco.MjModel.from_xml_path("humanoid.xml")
mx = mjx.put_model(model)       # upload model to GPU

# vmap over a batch of 2048 initial states
batch_step = jax.vmap(mjx.step, in_axes=(None, 0))
```

!!! info "MJX and SB3"
    Like Brax, MJX is a JAX library. The standard path is to pair it with a JAX-native RL library (e.g. Brax's training utilities, or Rlax). SB3 wrappers exist in the community but are experimental.

---

## 3 · Performance comparison

All figures below are **approximate** and depend heavily on task complexity, observation size, network architecture, and specific GPU model. Treat them as order-of-magnitude estimates, not benchmarks.

| Framework | Steps/sec (RTX 3090, approx.) | Typical parallelism | Physics quality | SB3 compatible? |
|-----------|-------------------------------|---------------------|-----------------|-----------------|
| Godot (this course) | ~50 k | 8–32 (CPU cores) | Game physics (GodotPhysics / Jolt) | Yes, native |
| EnvPool — Atari | ~500 k | 64–1 024 | Emulated (ALE) | Yes, thin wrapper |
| EnvPool — MuJoCo | ~200 k | 64–256 | Full MuJoCo | Yes, thin wrapper |
| Isaac Gym / Isaac Lab | ~1 M | 2 048+ | Rigid body (PhysX) | Partial (official wrapper) |
| MJX | ~5 M | 2 048+ | Full MuJoCo | Partial (community wrappers) |
| Brax | ~10 M | 4 096+ | Simplified (Spring-based) | Partial (custom loop recommended) |

!!! warning "Read these numbers carefully"
    The Brax and MJX figures assume the entire training loop (environment + policy + gradient update) runs on-device in JAX. Wrapping them in SB3 typically drops performance by 2–10× due to GPU-to-CPU data movement.

---

## 4 · When Godot wins

GPU env frameworks are genuinely faster at producing training samples. That does not mean they are better for every use case. Godot has structural advantages that no physics engine rewrite can replace.

**The env IS the game.** If your agent will be deployed inside a Godot game, training in Godot eliminates the sim-to-real gap entirely. The physics, the rendering, the level geometry — everything the agent will see at runtime is exactly what it trained on. No domain randomisation, no policy transfer, no surprise.

**Rich visual scenes with art assets.** Brax and Isaac Gym render simple geometric primitives. If your task involves a painted dungeon, a stylised platformer, or a custom-rigged character, Godot is the only option that lets you train on the actual art.

**Iterative design loop.** A level designer can modify a Godot scene and restart training in minutes. GPU physics frameworks require modifying URDF/MJCF files, recompiling, and often rewriting reward functions to match the new geometry.

**Everything in this course.** None of this course's tasks requires more than a few million steps. GPU envs would not save meaningful time and would add significant setup complexity.

!!! tip "Rule of thumb"
    If the goal is to ship something that runs in a Godot game, use Godot. If the goal is to publish a paper comparing your algorithm to MuJoCo baselines, use the same framework the baselines used.

---

## 5 · Sim-to-Godot transfer

Sometimes the right strategy is to train quickly on a fast surrogate environment, then fine-tune or evaluate in Godot. This is a variant of sim-to-real transfer applied to game engines.

**What typically carries over:**

- High-level motor skills learned on similar kinematics (e.g. MuJoCo HalfCheetah → Godot biped locomotion)
- Reward-shaping intuitions — a curriculum that worked on Brax Ant often transfers to a Godot quadruped
- Hyperparameter starting points — PPO clip range, GAE lambda, and learning rate schedules are surprisingly stable across physics engines

**What typically does not carry over:**

- Fine-grained contact dynamics — MuJoCo and Godot model friction and contact resolution differently
- Observation scale and range — sensor readings mean different things in different engines; you will need a calibration pass
- Visual observations — a policy trained on MuJoCo's simple rendering will not generalise to Godot's shaders and lighting without retraining the visual encoder

!!! info "Cross-reference"
    The broader topic of handling the gap between training and deployment environments is covered in [unit-sim-to-real.md](unit-sim-to-real.md), including domain randomisation, system identification, and real-hardware deployment checklists.

---

## 6 · EnvPool + SB3 quick start (the easiest upgrade)

EnvPool is the most practical upgrade from this course's SB3 workflow. If you ever need to run Atari or classic MuJoCo at research scale, this is the path of least resistance.

**Why is EnvPool faster than `SubprocVecEnv`?**

SB3's `SubprocVecEnv` spawns N separate Python processes and communicates with them over `multiprocessing.Pipe`. Each `step()` call serialises the action array with `pickle`, sends it through the OS pipe, deserialises it in the worker process, runs the step, re-serialises the observation, and sends it back. For 64 environments that is 128 pickle round-trips per training step.

EnvPool does all of this in a single C++ thread pool. There is no pickle, no OS pipe, and no Python GIL contention between environments. The result is roughly **10× higher throughput for the same number of parallel environments**.

```python
import envpool
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn


class EnvPoolVecEnvWrapper(VecEnvWrapper):
    """Minimal wrapper to make an EnvPool VectorEnv compatible with SB3.

    EnvPool returns (obs, rew, terminated, truncated, info) in the Gymnasium
    step API. SB3 expects (obs, rew, done, info) in the old Gym API.
    This wrapper handles the conversion.
    """

    def reset(self) -> GymObs:
        obs, _ = self.venv.reset()
        return obs

    def step_wait(self) -> GymStepReturn:
        obs, rew, terminated, truncated, info = self.venv.step_wait()
        done = np.logical_or(terminated, truncated)
        # SB3 reads episode stats from info["terminal_observation"]
        # when done is True. EnvPool populates this automatically.
        return obs, rew, done, info


# ── Create 64 parallel Atari environments ─────────────────────────────────────
# EnvPool allocates a C++ thread pool — no subprocess overhead.
env = envpool.make(
    "Pong-v5",
    env_type="gymnasium",
    num_envs=64,
    seed=42,
    episodic_life=True,   # standard Atari training flag
    reward_clip=True,
)
env = EnvPoolVecEnvWrapper(env)

# ── Train with PPO ─────────────────────────────────────────────────────────────
# n_steps * num_envs = rollout buffer size = 128 * 64 = 8192 transitions
# This matches CleanRL's Atari PPO defaults closely.
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.01,
    learning_rate=2.5e-4,
    tensorboard_log="./runs/pong_envpool",
)
model.learn(total_timesteps=10_000_000)
model.save("ppo_pong_envpool")
```

!!! tip "Throughput sanity check"
    After the first rollout completes, SB3 logs `time/fps` in TensorBoard. With 64 EnvPool envs on a modern desktop CPU you should see **400 k – 600 k steps/sec** for Atari. With `SubprocVecEnv` and 64 processes the same hardware typically reaches 40 k – 80 k steps/sec — the EnvPool advantage is real.

!!! warning "EnvPool game list"
    EnvPool supports Atari (via ALE), classic MuJoCo (v4 and earlier), DMControl, and a handful of other environments. It does **not** support custom environments or Godot. For custom tasks, stick with `SubprocVecEnv` or look at Isaac Lab / Brax.

---

*Next steps:* if you are curious about how transfer learning between physics engines works in practice, read [unit-sim-to-real.md](unit-sim-to-real.md). If you want to understand the PPO implementation running inside all of the frameworks discussed here, see [unit-cleanrl.md](unit-cleanrl.md).
