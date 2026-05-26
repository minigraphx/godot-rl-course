# Visual Observations — Teaching Agents to See

[← Continuous 3D](unit-06.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~30 min · Training: ~45 min GPU / ~3 h CPU

---

!!! info "Three ways to see your AI"
    **Godot** (add a `TextureRect` showing the SubViewport texture — you can watch what the agent literally sees) · **TensorBoard** (expect curves that are 5–10× slower than your raycast baseline — that is normal) · **print pixel stats** (`min`, `max`, `mean` per channel after normalization — all values should sit in [0, 1] and the mean should be well above 0)

---

## 1 · Why visual observations?

Every unit so far has used **raycasts** to observe the world: a handful of rays shoot out from the agent, measure distance to the nearest surface, and return a compact float array. Raycasts are fast, easy to reason about, and work well for structured scenes. So why would you ever replace them with camera pixels?

### What raycasts give you

Eight raycasts produce eight floats. That is 8 values, each carrying clear physical meaning ("wall 2.3 m ahead"). A neural network learns from this within tens of thousands of steps.

### What visual observations give you

A 64×64 RGB image produces 12,288 values. Those values are raw pixel colours with no inherent semantics — the network must learn, from scratch, which patterns matter.

| | Raycast | Visual (64×64 RGB) |
|---|---|---|
| Observation size | ~8 values | 12,288 values |
| Feature engineering | You decide what to cast | None — CNN learns it |
| Interpretability | Easy to print and inspect | Very hard to debug |
| Typical training steps | 500k–1M | 3M–10M |
| Works on cluttered/varied scenes | Only if you designed the right rays | Yes |

### Advantages of visual observations

**No feature engineering.** You do not decide what to measure. The convolutional neural network (CNN) discovers which visual features predict reward. This is both the power and the cost: you gain generality, you pay in compute.

**Works when raycasts cannot.** Imagine a scene with randomly scattered objects of many different shapes. Designing a raycast configuration that captures every relevant object is laborious. A camera sees everything in its field of view automatically.

**Transfer potential.** A CNN trained on one visually similar environment often adapts more quickly to a new one. The low-level visual features (edges, colours, textures) are reusable in a way that handcrafted raycast configurations are not.

### Disadvantages of visual observations

**Much larger observation space.** 12,288 floats vs 8. Every network layer that processes this observation is proportionally larger, slower, and hungrier for GPU memory.

**Requires a CNN feature extractor.** Your MLP policy cannot directly process a 64×64 image sensibly. You must prepend a convolutional network. Stable-Baselines3 provides one (`NatureCNN`), but it adds thousands of parameters and slows each update.

**Harder to debug.** With raycasts you can `print(get_obs())` and immediately understand the numbers. With pixels, debugging means watching the SubViewport live in Godot or visualizing feature maps — both require more tooling.

**Needs far more training steps.** Expect 3–10× more environment steps than an equivalent raycast task. A task that converges in 500k steps with raycasts may need 3–5 million with visual observations.

### Rule of thumb

> Use raycasts if you can. Use visual observations when raycasts cannot capture what matters — or when generalisation across varied visual scenes is a project goal.

---

## 2 · Godot SubViewport → observation pipeline

The core idea: a `SubViewport` node renders a separate camera feed inside Godot. Your `AIController3D` reads that rendered texture, flattens the pixel values to a float array, and returns them as the agent observation.

### Scene setup

1. Add a `SubViewport` node as a child of the agent root node.
2. Set `SubViewport` size to **64×64** (or 84×84 for richer detail — but training is slower).
3. Set `SubViewport.render_target_update_mode` to `ALWAYS` so it renders every frame.
4. Add a `Camera3D` **inside** the SubViewport. Position it at the agent's "eye" location — usually slightly above and forward of the collision shape centre.
5. The SubViewport renders independently of the main camera. The agent "sees" through this inner camera.

!!! warning "Check the viewport renders before you train"
    A SubViewport that has `render_target_update_mode` left at the default (`ONCE`) will only render the first frame and then freeze. Your agent will train on a static image and learn nothing useful. Set the mode to `ALWAYS` before writing a single line of training code.

### GDScript: capturing pixels as observations

```gdscript
# ai_controller.gd — capture camera frames as observations
extends AIController3D

@onready var viewport = $SubViewport
@onready var camera   = $SubViewport/Camera3D

const IMG_WIDTH  = 64
const IMG_HEIGHT = 64

func get_obs() -> Dictionary:
    # Get the rendered frame from SubViewport
    var img: Image = viewport.get_texture().get_image()
    img.resize(IMG_WIDTH, IMG_HEIGHT, Image.INTERPOLATE_BILINEAR)
    img.convert(Image.FORMAT_RGB8)

    # Flatten to float array, normalize to [0, 1]
    var obs = []
    for y in range(IMG_HEIGHT):
        for x in range(IMG_WIDTH):
            var pixel = img.get_pixel(x, y)
            obs.append(pixel.r)
            obs.append(pixel.g)
            obs.append(pixel.b)

    return {"obs": obs}

func get_obs_size() -> int:
    return IMG_WIDTH * IMG_HEIGHT * 3  # 64×64×3 = 12,288
```

`Image.get_pixel()` returns a `Color` whose `.r`, `.g`, `.b` channels are already in `[0.0, 1.0]` when the format is `FORMAT_RGB8`. No further normalization is required.

The loop visits pixels in row-major order: all pixels of row 0 first, then row 1, and so on. The channel order is R, G, B for each pixel. Keep this order consistent with however you reshape the array on the Python side (see Section 4).

---

## 3 · Grayscale and frame stacking

Two standard techniques reduce the observation size and give the agent temporal awareness without adding a recurrent network.

### Grayscale

Converting to luminance drops from three channels to one — a 3× reduction in observation size:

```gdscript
# Grayscale: convert to luminance before flattening
func get_obs() -> Dictionary:
    var img: Image = viewport.get_texture().get_image()
    img.resize(IMG_WIDTH, IMG_HEIGHT, Image.INTERPOLATE_BILINEAR)
    img.convert(Image.FORMAT_L8)  # convert to luminance (grayscale)

    var obs = []
    for y in range(IMG_HEIGHT):
        for x in range(IMG_WIDTH):
            var pixel = img.get_pixel(x, y)
            obs.append(pixel.r)  # luminance stored in .r when FORMAT_L8

    return {"obs": obs}

func get_obs_size() -> int:
    return IMG_WIDTH * IMG_HEIGHT  # 64×64 = 4,096 values
```

!!! tip "Start with grayscale"
    Unless colour is genuinely important for the task (e.g., distinguishing a red enemy from a green ally), train with grayscale first. It trains 3× faster and is easier to debug. Add colour back only if grayscale clearly fails.

### Frame stacking

A single frame tells the agent where objects are. It does not tell the agent how fast they are moving or which direction. **Frame stacking** concatenates the last N frames into one observation, giving the policy implicit velocity information without a recurrent network.

```gdscript
# Frame stack: keep the last 4 grayscale frames
const STACK_SIZE = 4
var frame_buffer: Array = []

func get_obs() -> Dictionary:
    var frame = _capture_grayscale_frame()
    frame_buffer.push_back(frame)
    if frame_buffer.size() > STACK_SIZE:
        frame_buffer.pop_front()

    # Pad with zero frames at the start of an episode
    while frame_buffer.size() < STACK_SIZE:
        frame_buffer.push_front(Array.filled(IMG_WIDTH * IMG_HEIGHT, 0.0))

    # Concatenate all frames into one flat array
    var obs = []
    for f in frame_buffer:
        obs.append_array(f)
    return {"obs": obs}  # 4 × 64 × 64 = 16,384 values

func get_obs_size() -> int:
    return STACK_SIZE * IMG_WIDTH * IMG_HEIGHT

func _capture_grayscale_frame() -> Array:
    var img: Image = viewport.get_texture().get_image()
    img.resize(IMG_WIDTH, IMG_HEIGHT, Image.INTERPOLATE_BILINEAR)
    img.convert(Image.FORMAT_L8)
    var frame = []
    for y in range(IMG_HEIGHT):
        for x in range(IMG_WIDTH):
            frame.append(img.get_pixel(x, y).r)
    return frame
```

Make sure you call `frame_buffer.clear()` (or reset the buffer) inside `on_episode_end()` — otherwise frames from the previous episode contaminate the next one.

**Why frame stacking works — the Atari DQN intuition.** In the original Atari DQN paper (Mnih et al., 2015), four stacked grayscale frames of Pong let the network infer both the ball's position *and* its velocity. One frame alone is not enough to know whether the ball is moving left or right. Four consecutive frames make direction obvious even to a simple CNN.

---

## 4 · CNN feature extractors in SB3

Stable-Baselines3 expects image observations shaped as **(channels, height, width)** — the PyTorch convention, also called "channels first". Your Godot code returns a flat 1D array. You must reshape it before the policy sees it.

The cleanest approach is a `gym.ObservationWrapper` that reshapes on the Python side:

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv


class VisualObsWrapper(gym.ObservationWrapper):
    """Reshape flat obs [H*W*C] → (C, H, W) tensor for CNN."""

    def __init__(self, env, height=64, width=64, channels=1):
        super().__init__(env)
        self.h, self.w, self.c = height, width, channels
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(channels, height, width),
            dtype=np.float32,
        )

    def observation(self, obs):
        # obs is shape (H*W*C,); reshape and move channels first
        return obs.reshape(self.c, self.h, self.w)


# Build the environment
base_env = StableBaselinesGodotEnv(
    env_path="./VirtualCamera.x86_64",
    n_parallel=4,
    speedup=10,
)
env = VisualObsWrapper(base_env, height=64, width=64, channels=1)  # grayscale

# "CnnPolicy" uses NatureCNN under the hood
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/",
    n_steps=512,        # smaller rollout — large obs eats memory faster
    batch_size=64,      # smaller batch for the same reason
    learning_rate=1e-4, # lower lr — CNN has far more parameters than MLP
)

model.learn(total_timesteps=3_000_000)  # visual tasks need 3–5× more steps
model.save("virtualcamera_cnn")
env.close()
```

Key changes compared to a raycast training script:

| Parameter | Raycast typical | Visual obs typical |
|---|---|---|
| `policy` | `"MlpPolicy"` | `"CnnPolicy"` |
| `n_steps` | 2048 | 512 |
| `batch_size` | 64–256 | 32–64 |
| `learning_rate` | 3e-4 | 1e-4 |
| `total_timesteps` | 500k–1M | 3M–10M |

---

## 5 · The NatureCNN architecture

SB3's built-in CNN is the architecture from the DeepMind Atari DQN Nature paper (2015). It was designed for 84×84 greyscale frames and has become the standard starting point for visual RL:

```
Input: (C, 84, 84)
Conv2d(C,  32, kernel=8, stride=4) → (32, 20, 20)   ReLU
Conv2d(32, 64, kernel=4, stride=2) → (64,  9,  9)   ReLU
Conv2d(64, 64, kernel=3, stride=1) → (64,  7,  7)   ReLU
Flatten                            → (3136,)
Linear(3136, 512)                  → (512,)          ReLU
```

Output: 512-dimensional feature vector, passed to the actor head (policy) and the critic head (value function).

**For 64×64 input**, the spatial dimensions after the three convolutions are smaller than for 84×84. The flattened size is different, but SB3 computes it automatically. You can reduce the final feature dimension to match:

```python
from stable_baselines3.common.torch_layers import NatureCNN

policy_kwargs = dict(
    features_extractor_class=NatureCNN,
    features_extractor_kwargs=dict(features_dim=256),  # smaller for 64×64
)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
```

The `features_dim` kwarg sets the size of the final linear layer (512 → 256 here). A smaller feature vector reduces the actor/critic head size and speeds up training slightly.

---

## 6 · Custom CNN for your environment

`NatureCNN` is sized for Atari. If your task has simple visuals — flat colours, distinct shapes, not much fine detail — a much lighter CNN trains faster and generalises just as well:

```python
class SimpleCNN(BaseFeaturesExtractor):
    """Lightweight CNN for simple visual tasks."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flattened size from the conv layers dynamically
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flat = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=SimpleCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
```

The `with torch.no_grad()` block computes the output size of the conv stack by running a dummy tensor through it. This avoids hard-coding magic numbers that would break whenever you change image resolution.

When to use `SimpleCNN` over `NatureCNN`:

- Image resolution is 32×32 or 64×64 (not 84×84)
- Scene has simple geometry (few objects, flat textures)
- You want faster iteration during early experiments
- Memory is constrained (e.g., training on CPU)

---

## 7 · VirtualCamera example from godot_rl_agents_examples

The official `godot_rl_agents_examples` repository includes a ready-made visual observation scene. Use it as your reference implementation before building your own.

### Setup

```bash
git clone https://github.com/edbeeching/godot_rl_agents_examples
```

Navigate to `examples/VirtualCamera/`. The scene contains:

- An agent with a `SubViewport` and `Camera3D` inside it
- `ai_controller.gd` that implements `get_obs()` returning flattened pixel values
- A continuous action space for steering and throttle

### Viewing what the agent sees

Add a debug display overlay to the scene while you are developing:

```gdscript
# Debug: show the SubViewport texture in the running scene
func _ready():
    var display = TextureRect.new()
    display.texture = $SubViewport.get_texture()
    display.set_anchors_and_offsets_preset(Control.PRESET_TOP_LEFT)
    display.size = Vector2(128, 128)  # scaled up for visibility
    get_tree().root.add_child(display)
```

This renders the SubViewport feed in the corner of the main window. Before you start training, confirm:

- The image updates every frame (not frozen)
- The camera angle makes sense — the agent should see the track/obstacles ahead
- The image is not entirely black or entirely white

Remove or hide the debug display before exporting the binary used for parallel training. Rendering extra TextureRect nodes in 32 parallel instances wastes GPU time.

---

## 8 · Training tips for visual observations

| Issue | Cause | Fix |
|---|---|---|
| Very slow training | Large obs (64×64×3 = 12k values) | Use grayscale (÷3) + resize to 32×32 to iterate quickly first |
| Policy does not learn | Random pixel noise dominates gradient | Normalize to [0, 1]; verify SubViewport renders correctly |
| High RAM usage | Large buffer with pixel observations | Reduce `buffer_size` to 100k; use `optimize_memory_usage=True` in DQN |
| CNN not using colour | All channels look identical | Check image format; consider switching to explicit grayscale |
| `approx_kl` explodes | Learning rate too high for CNN | Lower `learning_rate` to 1e-4; CNNs have far more parameters than MLPs |
| Loss goes NaN | Unnormalized pixels (values 0–255) | Ensure `img.get_pixel().r` path is used, not raw byte values |
| Agent ignores objects at edge | Field of view too narrow | Widen camera FOV; reposition camera further back |

!!! warning "Verify the SubViewport renders before training"
    A frozen SubViewport (wrong `render_target_update_mode`) will give the agent identical observations every step. The policy will converge to a random or trivial behaviour and the loss will not decrease. Always confirm the SubViewport is live before launching a long training run.

!!! tip "Grayscale first, colour second"
    Start every new visual task with grayscale at 32×32 or 64×64. Verify the agent learns something. Then scale up resolution or add colour only if you have evidence that colour information helps. Each increase in resolution multiplies training time.

### BatchNorm vs LayerNorm vs VecNormalize

Normalization layers are standard in supervised deep learning but their use in RL is nuanced — the non-stationarity of RL data creates specific problems.

**BatchNorm** computes statistics across the batch during training and uses running statistics during inference. In RL, the data distribution shifts as the policy improves; batch statistics computed on early-training data become stale. This can cause instability and is largely abandoned in modern deep RL networks.

**LayerNorm** normalizes per-sample, not per-batch. No running statistics to go stale; compatible with non-stationary data. Preferred when normalization is needed inside a deep RL network.

**VecNormalize** (SB3 wrapper) normalizes the *observation* before it enters the network — not inside the network. Operates at the environment level, not the layer level.

| Layer | Used in | Note |
|-------|---------|------|
| No normalization | SB3 default MLP | Works fine for low-dim obs |
| BatchNorm | Early DQN papers | Largely abandoned in modern RL |
| LayerNorm | Transformer-based policies, large networks | Preferred when normalization is needed |
| VecNormalize | SB3 wrapper | Obs-level normalization, not layer-level |

**Practical rule:** for MLP policies on low-dim observations, don't add normalization — use `VecNormalize` instead. For CNN policies on images: add LayerNorm after conv layers if training is unstable. Example:

```python
class NatureCNNWithNorm(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.LayerNorm([32, 20, 20]),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
```

---

## 9 · Hybrid: visual + proprioceptive observations

In practice the most effective setup is often **not** pure visual observations. You give the CNN the camera feed *and* pass raw proprioceptive data (velocity, position, heading) directly to the policy MLP — bypassing the CNN entirely. The two streams are concatenated before the actor and critic heads.

Godot side — return a dictionary with two keys:

```gdscript
func get_obs() -> Dictionary:
    var pixels = _capture_grayscale_frame()   # flat array, 64×64 values
    var state  = [
        linear_velocity.x,
        linear_velocity.y,
        linear_velocity.z,
        rotation.y,
        distance_to_target,
        on_floor_bool,
    ]
    return {
        "image": pixels,
        "state": state,
    }
```

Python side — SB3's `MultiInputPolicy` handles dict observation spaces automatically:

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

# The Godot env must expose a dict obs space with keys "image" and "state"
env = StableBaselinesGodotEnv(env_path="./HybridAgent.x86_64", n_parallel=4)

# MultiInputPolicy: CNN for image keys, MLP for vector keys, outputs concatenated
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=2_000_000)
```

SB3 inspects the observation space dict at construction time. Any key whose space is `Box` with three dimensions is treated as an image and routed through the CNN. Any 1D `Box` key is treated as a flat vector and routed through a small MLP. The outputs are concatenated and fed to the shared actor/critic trunk.

**Why hybrids often outperform pure visual.** The CNN must infer velocity and heading from pixel differences — a hard task even with frame stacking. Giving those numbers directly as state eliminates that challenge. The CNN then focuses purely on detecting obstacles and targets, which it is well suited for.

---

## 10 · Viz checkpoint

After training, always run a visual inspection with `--viz`:

```bash
python train.py --env_path ./VirtualCamera.x86_64 --viz
```

In the running scene:

1. Add a `TextureRect` showing the SubViewport texture (see Section 7) so you can watch what the agent literally sees.
2. Observe: does the agent respond to objects as they enter the camera field of view? You should see evasive or goal-directed behaviour start when a relevant object appears in the SubViewport feed.
3. Compare behaviour: if you trained a raycast agent on the same task, run both side by side. The visual agent often moves more smoothly because the CNN generalises over partial views — the agent begins reacting before a raycast would even hit the object.

What to look for:

- **Good sign**: the agent turns toward (or away from) objects as they appear in the top portion of its SubViewport feed.
- **Bad sign**: the agent ignores objects that are clearly visible in the SubViewport — this suggests the CNN is not converging or the normalization is wrong.
- **Bad sign**: the agent behaves identically in all situations — the SubViewport may be frozen.

---

## 11 · Stretch goals

Work through these after completing the main task. Each isolates one variable to build intuition about visual RL:

**Resolution sweep.** Train the same task at 16×16, 32×32, and 64×64. Plot final reward (after equal training steps) vs resolution. At what resolution does increasing detail stop helping? How much does training time scale with resolution?

**Grayscale vs colour.** Take a task where colour provides useful information (e.g., red enemy vs green ally). Train once with grayscale and once with RGB. Measure sample efficiency (steps to reach a target reward). How large is the gap?

**Frame stacking ablation.** Create an environment where an object moves across the scene. Train with 1 frame, 2 frames, and 4 frames stacked. When does stacking help? Does a stationary target task benefit from stacking at all?

**Transfer experiment.** Train to convergence on one visual environment. Save the model. Fine-tune on a visually similar environment (same task, different textures or lighting). How many steps does fine-tuning require compared to training from scratch? Compare the CNN feature weights before and after fine-tuning.

---

## What's next?

You have now taught an agent to see. The remaining challenge is coordination: what happens when multiple agents share the same environment?

**Unit 7 — Multi-Agent** introduces cooperative and competitive scenarios where several agents act simultaneously. You will learn how godot-rl-agents handles multi-agent environments, how the reward structure changes, and how independent PPO (each agent trains its own policy) compares to shared-weight policies.

[→ Multi-Agent](unit-07.md)
