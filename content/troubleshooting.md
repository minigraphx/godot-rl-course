# Troubleshooting & FAQ

[Course home](index.md)

This page collects common errors, warnings, and questions you may encounter while setting up and training RL agents with godot-rl-agents, stable-baselines3, and Python. Each entry explains the root cause and the fix. **Check here first before opening a GitHub issue.**

---

## Setup & Installation

### Plugin not found / "godot-rl-agents plugin failed to load"

**Cause:** The plugin C# project has not been built, or the .NET SDK is not installed.

**Fix:**
```bash
# Ensure you have the .NET SDK installed (dotnet --version should work)
# Then rebuild the plugin in Godot
# Project → Tools → C# → Build Project
# Wait for MSBuild to complete; watch the bottom-right notification panel
```

Restart Godot after the build completes.

**See also:** [Unit 0](unit-00.md) § 3

---

### Godot 4.0 / 4.1 / 4.2 / 4.3 compatibility error

**Cause:** godot-rl-agents has been built and tested against specific Godot .NET versions; using a version from outside the supported range may cause C# or NuGet package mismatches.

**Fix:**
```bash
# Check the plugin's README or pyproject.toml for the recommended Godot version
# Download and use that exact version from godotengine.org
# (e.g., Godot 4.3 .NET is recommended; 4.0 is no longer supported)
```

If you must use a different version, check the [godot-rl-agents](https://github.com/edbeeching/godot-rl-agents) repository for known issues.

---

### `gdrl: command not found`

**Cause:** The Python `godot-rl-agents` package was not installed, or your conda environment is not activated.

**Fix:**
```bash
# Activate your conda environment
conda activate godot_env

# Install the package
pip install godot-rl-agents stable-baselines3

# Verify
gdrl --version
```

---

### `ModuleNotFoundError: No module named 'stable_baselines3'`

**Cause:** stable-baselines3 is not installed in your active Python environment.

**Fix:**
```bash
conda activate godot_env
pip install stable-baselines3
```

If you get a permission error, use `pip install --user stable-baselines3` or check that you own `/usr/local/lib/python3.x/`.

---

### `ModuleNotFoundError: No module named 'godot_rl'`

**Cause:** The godot-rl-agents Python package is not installed in your active conda environment.

**Fix:**
```bash
conda activate godot_env
pip install godot-rl-agents
```

Verify:
```bash
python -c "import godot_rl; print(godot_rl.__version__)"
```

---

### Python version incompatibility (godot-rl-agents requires Python 3.8+)

**Cause:** Your Python environment is older than 3.8, or you are using Python 2.

**Fix:**
```bash
# Check your Python version
python --version

# If < 3.8, create a new conda environment with Python 3.10 or 3.11
conda create -n godot_env python=3.10
conda activate godot_env
pip install godot-rl-agents stable-baselines3
```

---

### `onnxruntime` not installed (ONNX inference fails)

**Cause:** The `onnxruntime` package is required for exporting and running policies as ONNX, but was not installed with stable-baselines3.

**Fix:**
```bash
conda activate godot_env
pip install onnxruntime
# or for GPU support
pip install onnxruntime-gpu
```

---

## Training fails to start

### WebSocket connection refused / `ConnectionRefusedError: [Errno 111]`

**Cause:** The Python training script is trying to connect to Godot on port 11008 (default), but Godot is not running, not listening, or is on a different port.

**Fix:**
```bash
# 1. Start Godot with the training flag or use the visualizer
gdrl --load_path=examples/BallChase --viz

# 2. Then, in another terminal, run your training script
conda activate godot_env
python train.py --env_path=./godot_binary

# 3. If using a custom port, ensure both sides match
# In Python: env = GodotEnv(..., port=12000)
# In Godot: AIController.port = 12000
```

Check that no other process is blocking port 11008:
```bash
lsof -i :11008
```

---

### "Action space mismatch" / `ValueError: Action space size mismatch`

**Cause:** The number of actions your AIController returns (`n_actions`) does not match the `n_actions` argument passed to your SB3 algorithm.

**Fix:**
```python
# In AIController.cs (or your Godot script)
# Count the exact actions you return
public override int GetActionSpaceSize() {
    return 2;  // e.g., 2 for move_x, move_y
}

# In your Python training script, match exactly
model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,
    ...
)
```

Print both to be sure:
```python
print("Python action space:", env.action_space.shape)
print("Godot n_actions:", env._get_obs()['n_actions'])  # or query your AIController
```

---

### `KeyError: 'obs'` when starting training

**Cause:** Your observation dictionary from Godot does not include the required `'obs'` key, or is using the wrong key name.

**Fix:**
```python
# In your AIController, ensure you return a dictionary with 'obs' key
public override Dictionary<string, object> GetObservation() {
    return new Dictionary<string, object> {
        { "obs", new float[] { position.X, position.Y, ... } }
    };
}

# In Python, access it as
obs = env.reset()
print(obs.keys())  # should include 'obs'
```

If you are returning a flat array instead of a dict, wrap it:
```python
return {"obs": observation_array}
```

---

### Training freezes after first rollout (silent Godot crash)

**Cause:** The Godot process crashed or hung, but the Python side did not receive the error. This commonly happens due to an exception in `_process()` or `_physics_process()`.

**Fix:**
```bash
# 1. Run Godot in the foreground to see stderr/logs
gdrl --load_path=examples/BallChase --viz 2>&1 | tee godot.log

# 2. Check the Godot log for exceptions
# Common culprits: accessing null nodes, division by zero, infinite loops in reward calculation

# 3. Add debug prints in AIController
GD.Print($"Step {_step_count}: obs={obs}, action={action}, reward={reward}");
```

---

### `RuntimeError: Expected all tensors to be on the same device` (PyTorch error)

**Cause:** Your observation or reward is a PyTorch tensor on CPU, but PyTorch expects all tensors on the same device (CPU or GPU).

**Fix:**
```python
# Ensure all observations and rewards are NumPy arrays or the same tensor device
def _get_obs(self):
    obs = np.array([...], dtype=np.float32)  # Use NumPy, not torch
    return obs

# Or, if using PyTorch tensors, ensure they are all on the same device
obs = obs.to(device)  # Move to correct device
```

---

## Training runs but doesn't learn

### NaN loss after a few thousand steps

**Cause:** The learning rate is too high, the reward is unbounded, or pixel observations are not normalized. This causes exploding gradients.

**Fix:**
```python
# Reduce the learning rate
model = PPO("MlpPolicy", env, learning_rate=1e-4)  # Instead of 1e-3 or higher

# Ensure rewards are bounded (e.g., -1 to 1)
reward = np.clip(reward, -1.0, 1.0)

# Normalize pixel observations to [0, 1] or [-1, 1]
observation = observation.astype(np.float32) / 255.0
```

---

### `ep_rew_mean` stays flat at 0 or negative forever

**Cause:** Your reward function returns 0 (or a very small value) every step, so the agent has no learning signal. Check the sign — negative rewards discourage the agent.

**Fix:**
```csharp
// In AIController: reward should be non-zero when the agent makes progress
float reward = 0.0f;
if (Vector2.Distance(transform.GlobalPosition, target.GlobalPosition) < 1.0f) {
    reward = 1.0f;  // Goal reached
} else {
    reward = -0.01f;  // Small step cost to encourage progress
}

return reward;
```

Print rewards to TensorBoard to debug:
```python
# In your training script, log rewards per episode
episode_rewards.append(total_reward)
logger.record("custom/episode_reward", total_reward)
```

---

### `ep_rew_mean` oscillates wildly and never improves

**Cause:** The learning rate is too high, or `n_steps` is too small for PPO to accumulate enough experience before updating.

**Fix:**
```python
# Increase n_steps (experience per update)
model = PPO(
    "MlpPolicy",
    env,
    n_steps=4096,     # Instead of 2048
    learning_rate=3e-4,  # Reduce if still unstable
    ent_coef=0.01,    # Increase entropy coefficient to encourage exploration
)
```

Train longer and check TensorBoard for trends over 100k+ steps, not just the first 10k.

---

### `RuntimeError: CUDA out of memory` (GPU error)

**Cause:** Your batch size is too large for the GPU memory, or you are running multiple environments in parallel without enough VRAM.

**Fix:**
```python
# Reduce the buffer size and batch size
model = PPO(
    "MlpPolicy",
    env,
    n_steps=1024,    # Smaller rollout buffer
    batch_size=64,   # Smaller training batch
)

# Or, use CPU only
env = DummyVecEnv([lambda: GodotEnv(..., use_discrete_actions=False)])  # CPU training
```

---

### Agent spins in place / does nothing (action ignored)

**Cause:** The action space is incorrectly mapped in Godot, or actions are not being applied to the physics body.

**Fix:**
```csharp
// In AIController: ensure actions are applied each frame
public override void SetAction(float[] action) {
    // action[0] and action[1] are continuous or discrete actions
    velocity.X = action[0] * max_speed;
    velocity.Y = action[1] * max_speed;
    _body.Velocity = velocity;
    _body.MoveAndSlide();
}

// Check that you call MoveAndSlide() in _physics_process
public override void _PhysicsProcess(float delta) {
    SetAction(_last_action);
    _body.Velocity = velocity;
    _body.MoveAndSlide();
}
```

---

### Entropy collapses to near-zero early (policy becomes deterministic)

**Cause:** The entropy coefficient `ent_coef` is too low, or the policy has converged to a deterministic behavior before learning the task.

**Fix:**
```python
# Increase entropy coefficient to encourage exploration
model = PPO(
    "MlpPolicy",
    env,
    ent_coef=0.1,   # Increase from default 0.0
    learning_rate=3e-4,
)

# Or, use a schedule to decay entropy over time
from stable_baselines3.common.callbacks import EvalCallback
model = PPO(
    "MlpPolicy",
    env,
    ent_coef=0.05,
    policy_kwargs={"net_arch": [128, 128]},  # Larger network = more exploration needed
)
```

---

## Behavior & Reward debugging

### TensorBoard reward goes up but agent looks wrong in Godot

**Cause:** The reward function is not aligned with the desired behavior. The agent is maximizing the numerical reward without solving the task (reward hacking).

**Fix:**
```csharp
// Example: agent may "cheat" by moving quickly rather than reaching the goal
// BAD: only reward for velocity
float reward = velocity.Length() * 0.1f;

// BETTER: reward progress toward goal AND reaching goal
float distance_to_goal = Vector2.Distance(transform.GlobalPosition, goal.GlobalPosition);
float reward = -distance_to_goal * 0.1f;  // Progress penalty
if (distance_to_goal < 1.0f) {
    reward += 1.0f;  // Goal bonus
}
```

Watch the agent in Godot while training and compare to the TensorBoard curve. They should match in spirit.

---

### Agent ignores obstacles / targets that are clearly visible

**Cause:** The observation does not include or update information about obstacles/targets, or observations are cached and not refreshed after each step.

**Fix:**
```csharp
// In AIController: ensure observations are updated every frame
public override Dictionary<string, object> GetObservation() {
    // Observations MUST be recomputed each call, not cached
    var obs = new float[] {
        transform.GlobalPosition.X,
        transform.GlobalPosition.Y,
        target.GlobalPosition.X,
        target.GlobalPosition.Y,
        // ... include obstacle positions
    };
    return new Dictionary<string, object> { { "obs", obs } };
}

// Call in _PhysicsProcess, not _Process (to stay in sync with physics)
public override void _PhysicsProcess(float delta) {
    var obs = GetObservation();
    // Reset position next frame if goal reached
}
```

---

### Agent learns to exploit reward instead of solving the task

**Cause:** Reward hacking — the agent found a way to maximize the numerical reward without actually achieving the intended goal.

**Fix:**
```csharp
// Example: agent may stay in place and get stuck if you only reward reaching goal
// Add a small step cost to push the agent forward
float reward = -0.01f;  // Step penalty

// Penalize if agent gets stuck
if (stuck_timer > max_stuck_time) {
    reward -= 0.5f;
}

// Reward goal
if (distance < goal_radius) {
    reward += 1.0f;
}

return reward;
```

Manually verify that the highest-reward actions match your intuition about the task.

---

### `rollout/ep_len_mean` always equals `max_episode_steps` (agent never reaches goal)

**Cause:** The done condition is never triggered — either the agent never succeeds, or you are not returning done=True when the episode should end.

**Fix:**
```csharp
// In AIController: check and return done
public override bool IsDone() {
    // Done when goal is reached
    bool goal_reached = Vector2.Distance(transform.GlobalPosition, goal.GlobalPosition) < 1.0f;
    
    // Or when out of bounds
    bool out_of_bounds = transform.GlobalPosition.Length() > max_distance;
    
    // Or when max steps exceeded (framework handles this, but you can override)
    bool timeout = step_count >= max_steps;
    
    return goal_reached || out_of_bounds || timeout;
}
```

Debug by printing:
```python
# In Python
done_reasons = {"goal": 0, "timeout": 0, "oob": 0}
for episode in range(100):
    obs, info = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            if info.get("goal_reached"):
                done_reasons["goal"] += 1
            else:
                done_reasons["timeout"] += 1
            break
print(done_reasons)
```

---

## ONNX export & deployment

### ONNX export succeeds but Godot inference produces garbage actions

**Cause:** The ONNX export did not include the output tanh squashing, or the actor was exported alone without the log-std layer (SAC).

**Fix:**
```python
# For PPO, export the full policy (includes tanh output layer)
model.policy.actor.to("cpu")
model.policy.actor.eval()

# Use the stable_baselines3 export utility
from stable_baselines3.common.policies import BasePolicy
input_dict = {"obs": np.zeros((1, obs_space.shape[0]), dtype=np.float32)}
model.policy.actor.to("cpu")

# Export with opset 11 or 15 to ensure compatibility
import onnx
onnx_model, _ = convert_pytorch(model.policy.actor, input_dict, opset_version=11)
onnx.save(onnx_model, "policy.onnx")
```

Test in Python before deploying to Godot:
```python
import onnxruntime as ort
sess = ort.InferenceSession("policy.onnx")
action = sess.run(None, {"obs": obs})
print(action[0].shape, action[0].min(), action[0].max())  # Should be [-1, 1] for tanh
```

---

### ONNX model in Godot returns wrong tensor shape

**Cause:** The `input_names` or `output_names` do not match the exported ONNX model, or the input/output order is swapped.

**Fix:**
```csharp
// In Godot C#, when loading ONNX:
var model = new OnnxModel();
model.LoadModel("res://policy.onnx");

// Get the input/output node names from the exported model
// Use a tool like Netron to inspect the ONNX file:
// - Input node: "obs" (float32, shape [1, n_obs])
// - Output node: "output_0" (float32, shape [1, n_actions])

var input_dict = new Dictionary<string, Tensor> {
    { "obs", Tensor.FromArray<float>(obs_array) }
};
var outputs = model.Run(input_dict);
var actions = outputs["output_0"];  // or "output_name" from your export
```

Inspect your ONNX file with Netron (online tool at [netron.app](https://netron.app)) to verify node names and shapes.

---

### ONNX model inference is slow in Godot

**Cause:** The model is too large (e.g., CNN policy for visual obs), or you are using a GPU-incompatible runtime.

**Fix:**
```python
# Use MlpPolicy for real-time inference, not CnnPolicy
model = PPO("MlpPolicy", env, ...)  # Good for real-time
# Avoid: PPO("CnnPolicy", env, ...)  # Slow in game engine

# For visual observations, either:
# 1. Train with CnnPolicy in Python, then distill to MlpPolicy
# 2. Use very small image sizes (e.g., 32x32 instead of 84x84)
```

Typical inference time for MlpPolicy on Godot: 1–5 ms. For CnnPolicy: 50–200 ms.

---

### `onnxruntime.InvalidGraph: Load model from ... failed`

**Cause:** The ONNX opset version is incompatible with your onnxruntime, or the model was not exported correctly.

**Fix:**
```python
# Export with opset version 11 or 15 (both widely supported)
from stable_baselines3.common.policies import BasePolicy
# When exporting, specify opset_version=11

# Or, update your onnxruntime
pip install --upgrade onnxruntime
```

Also check that the ONNX file is valid:
```bash
python -c "import onnx; model = onnx.load('policy.onnx'); onnx.checker.check_model(model); print('Valid')"
```

---

## Still stuck?

If your issue is not listed here, check the following:

1. **[GitHub Issues](https://github.com/edbeeching/godot-rl-agents/issues)** — Search for your error message in the godot-rl-agents repository.
2. **[Godot Docs](https://docs.godotengine.org/)** — For Godot-specific errors (physics, signals, etc.).
3. **[SB3 Docs](https://stable-baselines3.readthedocs.io/)** — For stable-baselines3 algorithm details and hyperparameters.
4. **Course units** — Each unit includes detailed examples and common pitfalls in its own Troubleshooting sections.
5. **Logging & debugging** — Add print statements to both Godot and Python, and use TensorBoard to visualize training metrics.

**Before opening an issue, please provide:**
- Your Python version (`python --version`)
- Your Godot version
- The exact error message (full traceback)
- Your training script (sanitized)
- A minimal reproducible example (if possible)
