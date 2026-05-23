# Unit 2 — Convert LunarLander to Godot RL Agents

Two wins in one unit: **Phase A** — run and tweak **SimpleReachGoal** (raycasts, discrete actions). **Phase B** — build Lunar Lander from scratch using the same `AIController` patterns. Train with Stable-Baselines3 via **godot-rl**.

[← Unit 1: Foundations](unit-01.md) · [Course home](index.md)

---

!!! info "Three ways to see your AI"
    Godot (`--viz` or editor) · TensorBoard · reward/obs lines in `ai_controller.gd` / `lander.gd`

---

## Phase A — SimpleReachGoal warm-up (~30–45 min) { #phase-a }

**Run before you build**

Get a second quick win before the Lunar Lander build. Use the hub binary or clone [SimpleReachGoal](https://github.com/edbeeching/godot_rl_agents_examples/tree/main/examples/SimpleReachGoal).

```bash
python -c "from godot_rl.env_from_hub import env_from_hub; env_from_hub('edbeeching/godot_rl_SimpleReachGoal')"

python examples/stable_baselines3_example.py \
  --env_path=examples/godot_rl_SimpleReachGoal/bin/SimpleReachGoal.x86_64 \
  --experiment_name=unit2-warmup --timesteps=100000 --viz
```

**Study → tweak → retrain**

1. Open the project in Godot; trace `AIController`, raycasts, and the training scene.
2. Change one sensor distance or reward term (same idea as Unit 1).
3. Retrain briefly; compare Godot behavior and TensorBoard to your prediction.

Then continue to Phase B — you are copying these patterns, not discovering them from zero.

---

## Gymnasium → Godot concept map

| Gymnasium / SB3 | Godot RL Agents equivalent |
|-----------------|---------------------------|
| `gym.make("LunarLander-v2")` | Godot scene with `RigidBody2D` lander |
| `env.observation_space` (8 floats) | `get_obs()` returning `{"obs": [...]}` |
| `env.action_space` (4 discrete) | `get_action_space()` returning discrete size 4 |
| Per-step reward logic in Python | Reward shaping in `lander.gd` + `_ai_controller.reward` |
| `make_vec_env(n_envs=16)` | N copies of env root in `training_scene.tscn` |
| `PPO("MlpPolicy", env)` | `PPO("MultiInputPolicy", StableBaselinesGodotEnv(...))` |
| `model.learn(1_000_000)` | `gdrl --timesteps=1_000_000` |
| Saved `.zip` model | ONNX model path in Sync node inspector |

!!! info "Phase B — build"
    If you completed [Phase A](#phase-a), you already traced SimpleReachGoal. This map shows how those Gymnasium concepts become Godot nodes for the Lunar Lander you create next.

!!! info "Setup already done?"
    If you completed Unit 0, skip Sections 1–3 and start at [Section 4 — Build the lander scene](#4-build-the-lander-scene-landertscn).

---

## 1 · Install the required tools

If you completed Unit 0 your tools are already installed — skip to [Section 4](#4-build-the-lander-scene-landertscn).

Otherwise follow [Setup](setup.md) for Godot .NET, Miniconda, `godot_env`, and the plugin, then come back here.

---

## 2 · Create the Godot project

- Open the Godot **.NET editor**, click **New Project**
- Name it e.g. `LunarLanderGodot`
- Choose a folder, select **Renderer: Compatibility** (fastest for training)
- Click **Create & Edit**

---

## 3 · Install the godot-rl Godot plugin

See [Setup → Godot plugin](setup.md#godot-plugin-godot-rl-agents) for installation and enabling steps. Verify that Add Node shows `Sync` and `AIController2D` before continuing.

---

## 4 · Build the lander scene (`lander.tscn`)

**Node hierarchy to create**

```
Lander            ← RigidBody2D  (root, script: lander.gd)
  ├─ CollisionShape2D  (capsule or polygon shape)
  ├─ Sprite2D          (your lander artwork, or placeholder)
  ├─ LeftLeg           (Area2D — detects ground contact)
  │   └─ CollisionShape2D
  ├─ RightLeg          (Area2D — detects ground contact)
  │   └─ CollisionShape2D
  └─ AIController2D    (script: ai_controller.gd)
```

!!! tip "Use a 2D project"
    This matches the original LunarLander-v2 physics most closely. A 3D version is also possible — just extend `AIController3D` instead.

**Add ground / landing pad nodes**

- Create a `StaticBody2D` for the ground with a `CollisionShape2D`
- Place two small `StaticBody2D` pads as the landing targets
- Add a `Camera2D` following the lander (optional, nice for watching training)
- Export a `@export var landing_pad_position: Vector2` from the lander scene root so it can be read in the reward function

**Connect leg contact signals**

Select each `Area2D` leg node, go to **Node → Signals**, connect `body_entered` and `body_exited` to `lander.gd` to set the `left_leg_contact` / `right_leg_contact` booleans.

---

## 5 · Write `lander.gd` (physics + reward)

!!! warning "Training stalled?"
    Check in order: (1) reward sign and scale — is "good" actually positive? (2) sparse rewards — does the agent get any signal before the goal? (3) observation bugs — are sensors updating after resets? (4) TensorBoard flat but Godot looks fine — you may need longer training or a viz checkpoint.

```gdscript
extends RigidBody2D

# ── Exported config ──────────────────────────────────
@export var landing_pad_position : Vector2 = Vector2(0, 300)
@export var main_thrust           : float  = 800.0
@export var side_thrust           : float  = 300.0

# ── State ────────────────────────────────────────────
var left_leg_contact  : bool = false
var right_leg_contact : bool = false
var _thrust           : Vector2 = Vector2.ZERO
var _firing_main      : bool = false
var _firing_side      : bool = false

@onready var _ai : AIController2D = $AIController2D

# ── Setup ─────────────────────────────────────────────
func _ready() -> void:
    _ai.init(self)
    reset()

# ── Reset ─────────────────────────────────────────────
func reset() -> void:
    position = Vector2(
        landing_pad_position.x + randf_range(-150, 150),
        50.0
    )
    rotation           = 0.0
    linear_velocity    = Vector2(randf_range(-30,30), randf_range(-10,10))
    angular_velocity   = 0.0
    left_leg_contact   = false
    right_leg_contact  = false
    _thrust            = Vector2.ZERO

# ── Thrust API called by AIController ────────────────
func set_thrust(direction: Vector2, is_main: bool) -> void:
    _thrust      = direction
    _firing_main = is_main
    _firing_side = (direction != Vector2.ZERO) and not is_main

# ── Physics loop ─────────────────────────────────────
func _physics_process(_delta: float) -> void:
    if _ai.needs_reset:
        _ai.reset()
        reset()
        return

    if _ai.heuristic == "human":
        _read_human_input()

    if _thrust != Vector2.ZERO:
        apply_central_force(_thrust)

    # ── Per-step reward shaping (mirrors LunarLander-v2) ──
    var dist = global_position.distance_to(landing_pad_position)
    _ai.reward -= dist * 0.003
    _ai.reward -= abs(linear_velocity.x) * 0.001
    _ai.reward -= abs(linear_velocity.y) * 0.001
    _ai.reward -= abs(rotation)          * 0.002
    if left_leg_contact:  _ai.reward += 0.01
    if right_leg_contact: _ai.reward += 0.01
    if _firing_main: _ai.reward -= 0.30
    if _firing_side: _ai.reward -= 0.03

    _thrust = Vector2.ZERO

# ── Human input ───────────────────────────────────────
func _read_human_input() -> void:
    if Input.is_action_pressed("ui_up"):
        set_thrust(Vector2.UP * main_thrust, true)
    elif Input.is_action_pressed("ui_left"):
        set_thrust(Vector2.LEFT * side_thrust, false)
    elif Input.is_action_pressed("ui_right"):
        set_thrust(Vector2.RIGHT * side_thrust, false)

# ── Terminal outcomes ─────────────────────────────────
func game_over(terminal_reward: float) -> void:
    _ai.reward     += terminal_reward
    _ai.done        = true
    _ai.needs_reset = true

# ── Leg contact callbacks (connect in editor) ─────────
func _on_left_leg_body_entered(_body):  left_leg_contact  = true
func _on_left_leg_body_exited(_body):   left_leg_contact  = false
func _on_right_leg_body_entered(_body): right_leg_contact = true
func _on_right_leg_body_exited(_body):  right_leg_contact = false
```

!!! info "Crash & land detection"
    Call `game_over(-100.0)` when the body hits the ground too fast (check `linear_velocity.y` in `body_entered` on a ground `Area2D`) and `game_over(+100.0)` when both legs touch and speed is low.

---

## 6 · Write `ai_controller.gd` (RL interface)

```gdscript
extends AIController2D

# ── Observation space (8 floats, matches LunarLander-v2) ─
func get_obs() -> Dictionary:
    var lander = get_parent() as RigidBody2D
    var pad    = lander.landing_pad_position

    return {"obs": [
        (lander.global_position.x - pad.x) / 300.0,
        (lander.global_position.y - pad.y) / 300.0,
        lander.linear_velocity.x            / 200.0,
        lander.linear_velocity.y            / 200.0,
        lander.rotation                     / PI,
        lander.angular_velocity             / 5.0,
        float(lander.left_leg_contact),
        float(lander.right_leg_contact),
    ]}

# ── Action space: 4 discrete ──────────────────────────
func get_action_space() -> Dictionary:
    return {
        "engine": {"size": 4, "action_type": "discrete"}
    }

# ── Apply action ──────────────────────────────────────
func set_action(action) -> void:
    var lander = get_parent()
    match int(action["engine"]):
        0: lander.set_thrust(Vector2.ZERO,                       false)
        1: lander.set_thrust(Vector2.LEFT  * lander.side_thrust, false)
        2: lander.set_thrust(Vector2.UP    * lander.main_thrust, true)
        3: lander.set_thrust(Vector2.RIGHT * lander.side_thrust, false)

# ── Reward passthrough ────────────────────────────────
func get_reward() -> float:
    return reward
```

!!! info "Episode timeout & reset are automatic"
    The base `AIController2D` counts steps and flips `needs_reset` once its exported `reset_after` value is exceeded — no custom `_physics_process` is needed here. `lander.gd` watches `needs_reset` and respawns the lander.

!!! tip "Human control lives in lander.gd"
    godot-rl's `AIController` has no `get_user_input()` hook. Keyboard input is read in `lander.gd`, which checks `_ai.heuristic == "human"` while Sync is in HUMAN mode.

---

## 7 · Build the training scene (`training_scene.tscn`)

**Scene structure**

```
TrainingScene    (Node2D, root)
  ├─ Sync        (add via Add Node → search "Sync")
  ├─ Env_0       (instance of your lander.tscn)
  ├─ Env_1       (another instance)
  ├─ Env_2
  └─ …Env_N      (duplicate as many as you want, e.g. 8–16)
```

!!! tip "More envs = faster training"
    Each instance runs in parallel inside the same Godot process and all of them feed the trainer. 8–16 instances is a good starting point for CPU training.

**Configure the Sync node**

Select the **Sync** node and set these properties in the Inspector:

| Property | Value for training |
|----------|-------------------|
| Control Mode | `TRAINING` |
| Speed Up | `20` (or higher on fast hardware) |
| Action Repeat | `1` |
| ONNX Model Path | *leave empty* |

**Spread out the env instances**

Move each `Env_N` instance to a different position so they don't overlap visually. They don't need to be visible but it helps when debugging.

---

## 8 · Test with human control first

!!! warning "Do not skip this step"
    Playing manually is the fastest way to catch bugs in your observation normalization and reward shaping before wasting hours on broken training.

1. In the Sync node Inspector, set **Control Mode** to `HUMAN`.
2. Press **F6** (Run current scene). Use arrow keys to fly the lander. Verify in the Godot Output panel that:
    - Rewards accumulate (positive near pad, negative far away)
    - The `done` / `needs_reset` flags trigger correctly on crash or landing
    - The episode resets and the lander respawns
3. Set the Sync node **Control Mode** back to `TRAINING` before training.

---

## 9 · Run training (in the editor)

!!! info "In-editor training is the recommended path"
    Python and Godot talk over a local socket, so you can train directly from the editor — no exported binary required. This is the simplest, most portable approach, and on macOS it avoids the architecture-dependent issues that come with exported binaries. For large-scale parallel training with an exported binary, see the optional [Section 12](#12-export-a-game-binary-optional).

**Step 1 — activate the Conda environment**

```bash
conda activate godot_env
```

**Step 2 — start the gdrl training listener**

`gdrl` is installed with `pip` — there is no script to download:

```bash
gdrl --experiment_name=ppo-lunarlander-godot \
     --timesteps=1_000_000 \
     --save_model_path=lander_ppo \
     --onnx_export_path=lander_ppo.onnx
```

The console pauses on `waiting for remote GODOT connection on port 11008` — this is expected.

**Step 3 — press Play in Godot to connect**

Switch to the Godot editor, open `training_scene.tscn`, and press **F6** (Play Scene). Godot connects to the waiting `gdrl` process and SB3 starts printing a metrics table.

**What to expect — `ep_rew_mean` milestones**

| ep_rew_mean | What it means |
|-------------|---------------|
| < 0 | Agent is still crashing / drifting — normal early on |
| 50–150 | Basic stability learned, working on landing |
| ≥ 200 | Consistently landing — a solved Lunar Lander |

!!! note "Freeze during training is normal"
    With the default SB3 PPO setup, Godot briefly freezes while model weights are updated (the Python side blocks the socket). This is not a crash.

**Resume training if interrupted**

`--save_model_path=lander_ppo` writes `lander_ppo.zip`. To resume:

```bash
gdrl --resume_model_path=lander_ppo.zip \
     --timesteps=500_000 \
     --onnx_export_path=lander_ppo.onnx
```

**Save checkpoints automatically**

```bash
# Add this flag to save a checkpoint every 50 000 steps:
--save_checkpoint_frequency=50000
```

Checkpoints are saved under `logs/sb3/<experiment_name>/` and can be loaded with `--resume_model_path`.

---

## 10 · Monitor training with TensorBoard

```bash
# In a second terminal:
conda activate godot_env
tensorboard --logdir=logs/sb3
```

Then open `http://localhost:6006`.

**Key metrics**

| Metric | Meaning | Goal |
|--------|---------|------|
| `rollout/ep_rew_mean` | Mean episode reward | ≥ 200 |
| `rollout/ep_len_mean` | Mean episode length | Stabilises |
| `train/policy_gradient_loss` | PPO policy loss | Decreasing trend |
| `train/entropy_loss` | Exploration entropy | Slowly decreasing |
| `train/approx_kl` | Policy change per update | Stays small (< 0.02) |

**Tuning tips if training is stuck**

- Reward never rises → check observation normalization; print `get_obs()` and make sure no value is always 0 or saturated
- Reward oscillates → reduce learning rate: add `--learning_rate=0.0001`
- Training is slow → raise the Sync node's **Speed Up**, or add more `Env` instances. For true `--n_parallel` scaling, export a binary (Section 12)
- Policy gradient loss explodes → reduce `--clip_range` to `0.1`

---

## 11 · Export ONNX & run inference in Godot

**The ONNX file is exported automatically**

Because you passed `--onnx_export_path=lander_ppo.onnx`, the file is written when training finishes (or when you stop it with **Ctrl+C**).

**Import the model into the project**

Drag `lander_ppo.onnx` straight into Godot's **FileSystem** dock, placing it at `res://lander_ppo.onnx`.

**Create an inference scene (`onnx_inference_scene.tscn`)**

Duplicate `training_scene.tscn`. In the copy:

- Keep only **one** env instance (delete the extras)
- Select the **Sync** node and set:

| Property | Value |
|----------|-------|
| Control Mode | `ONNX Inference` |
| ONNX Model Path | `res://lander_ppo.onnx` |
| Speed Up | `1` (or lower — slowed down looks nice) |

**Run the inference scene**

Open `onnx_inference_scene.tscn` and press **F6** — *no Python needed*. Watch the trained agent land autonomously.

!!! info "No Python at runtime"
    The ONNX model runs entirely inside Godot via the plugin's compiled C# OnnxRuntime layer. The game no longer needs Python or an active Conda environment — you can ship it as a standalone game.

**Verify the agent is working** — add a temporary debug print to `lander.gd`:

```gdscript
func game_over(terminal_reward: float) -> void:
    if terminal_reward > 0:
        print("✅ Landed! reward=", terminal_reward)
    else:
        print("💥 Crashed or timed out")
    ...
```

A well-trained agent should land successfully in the majority of episodes.

---

## 12 · Export a game binary (optional)

!!! warning "Optional — only for large-scale parallel training"
    In-editor training (Section 9) is the recommended path for Units 0–2. Export a binary only if you want to run several Godot processes at once with `--n_parallel` for higher throughput — typically on a Linux training box or in the cloud. On macOS, exported binaries are architecture-dependent and fiddly, so prefer in-editor training there.

1. Set `training_scene.tscn` as the **Main Scene** in **Project → Project Settings → Application → Run**
2. **Project → Export…** → **Add…** → choose your platform (Linux/Windows/macOS)
3. Download export templates if prompted (**Manage Export Templates → Download**)
4. Click **Export Project…**, choose a folder e.g. `build/LunarLander`
5. Make the binary executable (Linux/macOS):

```bash
chmod +x build/LunarLander/LunarLander.x86_64
```

**Train against the binary with `--env_path`**

```bash
gdrl --env_path=build/LunarLander/LunarLander.x86_64 \
     --n_parallel=4 \
     --speedup=20 \
     --experiment_name=ppo-lunarlander-godot \
     --timesteps=1_000_000 \
     --onnx_export_path=lander_ppo.onnx
```

---

## Reference: Observation space

| Index | Value | Normalization | Range |
|-------|-------|---------------|-------|
| 0 | Horizontal offset from pad | `/ 300.0` | −1 … 1 |
| 1 | Vertical offset from pad | `/ 300.0` | −1 … 1 |
| 2 | Horizontal velocity | `/ 200.0` | −1 … 1 |
| 3 | Vertical velocity | `/ 200.0` | −1 … 1 |
| 4 | Rotation angle | `/ PI` | −1 … 1 |
| 5 | Angular velocity | `/ 5.0` | −1 … 1 |
| 6 | Left leg ground contact | boolean → float | 0 or 1 |
| 7 | Right leg ground contact | boolean → float | 0 or 1 |

!!! tip "Normalization matters"
    All values should be roughly in [−1, 1]. If a value can go much larger, the neural network saturates and training stalls. Adjust the divisors to match your scene scale.

## Reference: Action space

| Action index | Engine fired | Force applied |
|-------------|-------------|---------------|
| 0 | None (do nothing) | `Vector2.ZERO` |
| 1 | Left orientation thruster | `Vector2.LEFT * side_thrust` |
| 2 | Main engine (upward) | `Vector2.UP * main_thrust` |
| 3 | Right orientation thruster | `Vector2.RIGHT * side_thrust` |

## Reference: Reward function

| Event | Reward delta | Type |
|-------|-------------|------|
| Distance to pad | `− dist × 0.003` per step | Shaped |
| Horizontal speed | `− \|vx\| × 0.001` per step | Shaped |
| Vertical speed | `− \|vy\| × 0.001` per step | Shaped |
| Tilt angle | `− \|θ\| × 0.002` per step | Shaped |
| Left leg contact | `+ 0.01` per step | Shaped |
| Right leg contact | `+ 0.01` per step | Shaped |
| Main engine firing | `− 0.30` per step | Shaped |
| Side engine firing | `− 0.03` per step | Shaped |
| Safe landing | `+ 100.0` | Terminal |
| Crash | `− 100.0` | Terminal |
| Timeout | `0.0` | Terminal |

## Reference: `gdrl` command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--env_path` | None | Path to an exported binary. Omit for in-editor training. |
| `--timesteps` | 1 000 000 | Total environment steps. |
| `--n_parallel` | 1 | Binary instances to launch. Requires `--env_path`. |
| `--speedup` | 1 | Godot physics speed multiplier (binary mode). |
| `--experiment_name` | experiment | Name shown in TensorBoard. |
| `--experiment_dir` | logs/sb3 | TensorBoard log directory. |
| `--n_steps` | 64 | Steps per env per PPO rollout. |
| `--batch_size` | 64 | PPO minibatch size. Must divide `n_steps × n_envs`. |
| `--learning_rate` | 0.0003 | Adam learning rate. |
| `--ent_coef` | 0.0001 | Entropy bonus (encourages exploration). |
| `--clip_range` | 0.2 | PPO clipping range. |
| `--onnx_export_path` | None | Export ONNX model after training. |
| `--save_model_path` | None | Save SB3 `.zip` checkpoint. |
| `--save_checkpoint_frequency` | None | Save checkpoint every N steps. |
| `--resume_model_path` | None | Load existing `.zip` to resume training. |
| `--viz` | false | Show Godot window when training against a binary. |
| `--linear_lr_schedule` | false | Decay LR linearly to 0 over training. |
| `--inference` | false | Run inference (no training) from a loaded model. |

---

## Checklist

!!! success "You're ready for Unit 3 when..."
    1. Godot 4 **.NET edition** installed, plus the .NET SDK
    2. `godot_env` created with Python 3.10; `godot-rl[sb3]` and `tensorboard` installed
    3. New Godot project created; godot-rl plugin installed and enabled
    4. `lander.tscn` built: `RigidBody2D` + two leg `Area2D` + `AIController2D`
    5. `lander.gd` written: physics, thrust, reward shaping, `game_over()`, `reset()`
    6. `ai_controller.gd` written: `get_obs()`, `get_action_space()`, `set_action()`, `get_reward()`
    7. `training_scene.tscn` built: Sync node + 8–16 env instances
    8. Tested manually in HUMAN mode — reward and reset work correctly
    9. Trained in-editor: `ep_rew_mean ≥ 200` reached
    10. TensorBoard opened; training curves look healthy
    11. `lander_ppo.onnx` imported and loaded in the Sync node; inference scene runs correctly
    12. *(Optional)* Game binary exported for `--n_parallel` training (Section 12)

**What's next:** In Unit 3 you'll study **CrossTheRoad** and train with **DQN**.

[→ Unit 3: CrossTheRoad & DQN](unit-03.md)
