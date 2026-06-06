# Neural Foundations Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the approved visual neural-foundations sequence in the order Unit 0 → Foundations 1–2 → RL Essentials → Foundations 3 → RL Deep Dive, with audience-specific Python and Godot examples and a C#-free native deployment path.

**Architecture:** Keep one conceptual lesson per foundations unit, with a Research section and a Game Development section that share learning outcomes but use different projects. Build reference examples under `examples/neural_foundations/`; keep Units 1–2 independent of native extensions, then integrate `godot-native-rl`, ONNX verification, and ncnn only in Foundations 3. Split current `content/unit-01.md` only when all replacement pages exist, so the published navigation never contains an incomplete sequence.

**Tech Stack:** MkDocs Material, Markdown, Python 3.10, NumPy, PyTorch, Matplotlib, Godot 4.5+, GDScript, `godot-rl`, Stable-Baselines3 PPO, ONNX Runtime, `godot-native-rl`, ncnn.

**Spec:** [docs/superpowers/specs/2026-06-06-neural-foundations-teaching-design.md](../specs/2026-06-06-neural-foundations-teaching-design.md)

---

## Delivery decomposition

Each task produces a reviewable, working artifact. Use one commit per task.

| Task | Deliverable | Published immediately |
|------|-------------|-----------------------|
| 1 | Test and example scaffolding | No |
| 2 | Foundations 1 content and examples | No |
| 3 | Foundations 2 content and examples | No |
| 4 | RL Essentials / Deep Dive split | No |
| 5 | Foundations 3 Research path | No |
| 6 | Foundations 3 Game path and native deployment | No |
| 7 | Navigation, translations, complete-course QA | Yes |

The pages remain outside `mkdocs.yml` navigation until Task 7. They can still be
built and link-checked directly.

---

## File map

### Course pages

| File | Responsibility |
|------|----------------|
| `content/unit-neural-01.md` | Shared neuron lesson with Research and Game paths |
| `content/unit-neural-02.md` | Shared hidden-layer/backprop lesson with both paths |
| `content/unit-01.md` | Shortened RL Essentials bridge |
| `content/unit-neural-03.md` | Reward-learning capstone with both paths |
| `content/unit-rl-foundations-deep.md` | MC/TD, exploration, taxonomy, on/off policy |
| matching `*.de.md` files | German versions added in Task 7 |

### Research examples

| File | Responsibility |
|------|----------------|
| `examples/neural_foundations/research/neuron.py` | Weighted sum, activations, predictions |
| `examples/neural_foundations/research/plot_neuron.py` | Unit 1 live decision-boundary visualization |
| `examples/neural_foundations/research/tiny_mlp.py` | Manual `2 → 4 → 1` forward/backward implementation |
| `examples/neural_foundations/research/train_tiny_mlp.py` | Unit 2 training and Matplotlib visualization |
| `examples/neural_foundations/research/point_robot.py` | Unit 3 environment |
| `examples/neural_foundations/research/reinforce.py` | Hand-written REINFORCE trainer |
| `examples/neural_foundations/research/tests/` | Deterministic unit tests |

### Godot examples

| File | Responsibility |
|------|----------------|
| `examples/neural_foundations/game/project.godot` | One minimal project hosting all three scenes |
| `examples/neural_foundations/game/shared/tiny_neuron.gd` | Scalar neuron used by Unit 1 |
| `examples/neural_foundations/game/shared/tiny_mlp.gd` | `3 → 4 → 2` MLP and backprop used by Unit 2 |
| `examples/neural_foundations/game/unit_01_enemy/` | Enemy decision scene and scripts |
| `examples/neural_foundations/game/unit_02_collector/` | Arena collector scene and scripts |
| `examples/neural_foundations/game/unit_03_racer/` | Racer, controller, rewards, training and eval scenes |
| `examples/neural_foundations/game/test/` | Headless GDScript tests |

### Integration and validation

| File | Responsibility |
|------|----------------|
| `requirements-course.txt` | Add pinned Matplotlib, ONNX, ONNX Runtime dependencies |
| `scripts/test-neural-foundations.sh` | Run Python tests, Godot tests, and docs build |
| `scripts/verify_racer_policy.py` | PyTorch → ONNX → ncnn fixed-observation parity |
| `mkdocs.yml` | Final Phase 1 navigation |
| `content/index.md` / `content/index.de.md` | Final course ordering |

---

### Task 1: Add foundations test and example scaffolding

**Files:**
- Create: `examples/__init__.py`
- Create: `examples/neural_foundations/__init__.py`
- Create: `examples/neural_foundations/research/__init__.py`
- Create: `examples/neural_foundations/research/tests/__init__.py`
- Create: `examples/neural_foundations/game/project.godot`
- Create: `examples/neural_foundations/game/test/harness.gd`
- Create: `scripts/test-neural-foundations.sh`
- Modify: `requirements-course.txt`

- [ ] **Step 1: Pin visualization and export dependencies**

Append to `requirements-course.txt`:

```text

# Neural Foundations visualisation and model export
matplotlib==3.9.0
onnx==1.16.1
onnxruntime==1.18.1
```

- [ ] **Step 2: Create the minimal Godot project**

Create `examples/neural_foundations/game/project.godot`:

```ini
config_version=5

[application]
config/name="Neural Foundations Examples"

[display]
window/size/viewport_width=960
window/size/viewport_height=540

[rendering]
renderer/rendering_method="gl_compatibility"
renderer/rendering_method.mobile="gl_compatibility"
```

- [ ] **Step 3: Create the GDScript test harness**

Create `examples/neural_foundations/game/test/harness.gd`:

```gdscript
extends RefCounted

var failures := 0

func assert_close(actual: float, expected: float, tolerance: float, label: String) -> void:
	if absf(actual - expected) > tolerance:
		failures += 1
		push_error("%s: expected %f, got %f" % [label, expected, actual])

func assert_true(condition: bool, label: String) -> void:
	if not condition:
		failures += 1
		push_error(label)
```

- [ ] **Step 4: Create the combined validation script**

Create `scripts/test-neural-foundations.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
GODOT_BIN="${GODOT_BIN:-godot}"

"$PYTHON_BIN" -m unittest discover \
  -s examples/neural_foundations/research/tests \
  -p 'test_*.py'

shopt -s nullglob
for test_script in examples/neural_foundations/game/test/test_*.gd; do
  "$GODOT_BIN" --headless \
    --path examples/neural_foundations/game \
    --script "res://test/$(basename "$test_script")"
done

conda run -n mkdocs-env mkdocs build --strict
```

Run:

```bash
chmod +x scripts/test-neural-foundations.sh
```

- [ ] **Step 5: Verify the empty test suites and docs build**

Run:

```bash
conda run -n godot_env env \
  GODOT_BIN="${GODOT_BIN:-godot}" \
  ./scripts/test-neural-foundations.sh
```

Expected: Python reports `Ran 0 tests`; no Godot test files are executed; MkDocs
finishes with `Documentation built`.

- [ ] **Step 6: Commit**

```bash
git add requirements-course.txt examples/neural_foundations scripts/test-neural-foundations.sh
git commit -m "chore: scaffold neural foundations examples"
```

---

### Task 2: Build and teach Foundations 1

**Files:**
- Create: `examples/neural_foundations/research/neuron.py`
- Create: `examples/neural_foundations/research/plot_neuron.py`
- Create: `examples/neural_foundations/research/tests/test_neuron.py`
- Create: `examples/neural_foundations/game/shared/tiny_neuron.gd`
- Create: `examples/neural_foundations/game/unit_01_enemy/enemy_agent.gd`
- Create: `examples/neural_foundations/game/unit_01_enemy/unit_01_enemy.tscn`
- Create: `examples/neural_foundations/game/test/test_tiny_neuron.gd`
- Create: `content/unit-neural-01.md`

- [ ] **Step 1: Write the failing Python neuron tests**

Create `examples/neural_foundations/research/tests/test_neuron.py`:

```python
import unittest

from examples.neural_foundations.research.neuron import activate, neuron_output


class NeuronTests(unittest.TestCase):
    def test_weighted_sum_with_bias(self):
        self.assertAlmostEqual(
            neuron_output([0.5, -0.25], [0.8, -0.4], 0.1, "identity"),
            0.6,
        )

    def test_tanh_is_bounded(self):
        value = activate(10.0, "tanh")
        self.assertGreater(value, 0.99)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
```

The package marker files were created in Task 1, so the import works from the
repository root without modifying `PYTHONPATH`.

- [ ] **Step 2: Run the Python test and verify failure**

Run:

```bash
conda run -n godot_env python -m unittest \
  examples.neural_foundations.research.tests.test_neuron -v
```

Expected: `ModuleNotFoundError` for `neuron`.

- [ ] **Step 3: Implement the neuron**

Create `examples/neural_foundations/research/neuron.py`:

```python
from collections.abc import Sequence
import math


def activate(value: float, name: str) -> float:
    if name == "identity":
        return value
    if name == "step":
        return 1.0 if value >= 0.0 else 0.0
    if name == "sigmoid":
        return 1.0 / (1.0 + math.exp(-value))
    if name == "tanh":
        return math.tanh(value)
    raise ValueError(f"Unknown activation: {name}")


def neuron_output(
    inputs: Sequence[float],
    weights: Sequence[float],
    bias: float,
    activation: str,
) -> float:
    if len(inputs) != len(weights):
        raise ValueError("inputs and weights must have equal length")
    weighted_sum = sum(value * weight for value, weight in zip(inputs, weights))
    return activate(weighted_sum + bias, activation)
```

- [ ] **Step 4: Run the Python tests**

Run the command from Step 2.

Expected: two tests pass.

- [ ] **Step 5: Write the failing GDScript neuron test**

Create `examples/neural_foundations/game/test/test_tiny_neuron.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const TinyNeuron = preload("res://shared/tiny_neuron.gd")

func _init() -> void:
	var harness := Harness.new()
	var neuron := TinyNeuron.new()
	neuron.weights = PackedFloat32Array([0.8, -0.4])
	neuron.bias = 0.1
	harness.assert_close(
		neuron.forward(PackedFloat32Array([0.5, -0.25])),
		tanh(0.6),
		0.000001,
		"forward pass"
	)
	quit(harness.failures)
```

- [ ] **Step 6: Implement the GDScript neuron**

Create `examples/neural_foundations/game/shared/tiny_neuron.gd`:

```gdscript
class_name TinyNeuron
extends RefCounted

var weights := PackedFloat32Array([0.0, 0.0])
var bias := 0.0

func forward(inputs: PackedFloat32Array) -> float:
	assert(inputs.size() == weights.size())
	var weighted_sum := bias
	for index in range(inputs.size()):
		weighted_sum += inputs[index] * weights[index]
	return tanh(weighted_sum)
```

- [ ] **Step 7: Run the Godot test**

Run:

```bash
godot --headless \
  --path examples/neural_foundations/game \
  --script res://test/test_tiny_neuron.gd
```

Expected: exit code `0`.

- [ ] **Step 8: Build the two visual examples**

Implement:

- `plot_neuron.py`: two normalized features, colored predictions, displayed
  decision boundary, and Matplotlib sliders for `w1`, `w2`, and bias.
- `enemy_agent.gd`: normalized health and player distance, visible weighted
  contributions, chase/retreat behavior, and exported weights.
- `unit_01_enemy.tscn`: primitive player/enemy shapes, health bar, distance
  line, calculation labels, pause, and reset.

The displayed weighted sum must match `TinyNeuron.forward()` before activation.

- [ ] **Step 9: Write the lesson**

Create `content/unit-neural-01.md` with this exact section order:

```markdown
# Neural Foundations 1 — One Neuron, One Decision

[← Unit 0](unit-00.md) · [Course home](index.md)

!!! info "Three ways to see the computation"
    Running visual · current numbers · code you wrote

## 1 · Predict before running
## 2 · Weighted inputs and bias
## 3 · Activation functions
## 4 · Choose your path
## 5 · Research path — visible decision boundary
## 6 · Game path — chase or retreat
## 7 · Break it deliberately
## 8 · Compare the two paths
## 9 · Stretch goals
## What's next

[← Unit 0](unit-00.md) · [Course home](index.md) · [→ Neural Foundations 2](unit-neural-02.md)
```

Include the three approved experiments and collapsed answer keys. Do not link
the page from `mkdocs.yml` yet.

- [ ] **Step 10: Validate and commit**

Run:

```bash
./scripts/test-neural-foundations.sh
```

Expected: Python tests pass, Godot tests pass, MkDocs builds.

Commit:

```bash
git add content/unit-neural-01.md examples/neural_foundations
git commit -m "feat: add visual one-neuron foundations unit"
```

---

### Task 3: Build and teach Foundations 2

**Files:**
- Create: `examples/neural_foundations/research/tiny_mlp.py`
- Create: `examples/neural_foundations/research/train_tiny_mlp.py`
- Create: `examples/neural_foundations/research/tests/test_tiny_mlp.py`
- Create: `examples/neural_foundations/game/shared/tiny_mlp.gd`
- Create: `examples/neural_foundations/game/unit_02_collector/collector_agent.gd`
- Create: `examples/neural_foundations/game/unit_02_collector/teacher.gd`
- Create: `examples/neural_foundations/game/unit_02_collector/unit_02_collector.tscn`
- Create: `examples/neural_foundations/game/test/test_tiny_mlp.gd`
- Create: `content/unit-neural-02.md`

- [ ] **Step 1: Test a deterministic manual forward pass**

The Python and GDScript tests must use identical fixed weights and assert:

```text
inputs: [0.5, -0.25]
hidden weights: [[0.2, -0.4], [0.7, 0.1]]
hidden bias: [0.0, 0.1]
output weights: [[0.6, -0.3]]
output bias: [0.05]
activation: tanh
```

Both implementations must match a NumPy reference within `1e-6`.

- [ ] **Step 2: Test one gradient against PyTorch autograd**

Use one fixed sample and mean-squared error. Assert every manually calculated
gradient differs from PyTorch autograd by less than `1e-5`.

- [ ] **Step 3: Implement the manual Python MLP**

`tiny_mlp.py` must expose:

```python
class TinyMLP:
    def forward(self, inputs: np.ndarray) -> np.ndarray: ...
    def loss(self, prediction: np.ndarray, target: np.ndarray) -> float: ...
    def backward(self, inputs: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]: ...
    def apply_gradients(self, gradients: dict[str, np.ndarray], learning_rate: float) -> None: ...
```

Use only NumPy inside this class. PyTorch appears only in the verification test
and comparison section.

- [ ] **Step 4: Implement the GDScript `3 → 4 → 2` MLP**

`tiny_mlp.gd` must expose:

```gdscript
func forward(inputs: PackedFloat32Array) -> PackedFloat32Array
func mse_loss(prediction: PackedFloat32Array, target: PackedFloat32Array) -> float
func train_sample(
	inputs: PackedFloat32Array,
	target: PackedFloat32Array,
	learning_rate: float
) -> float
```

The implementation uses `tanh` in both layers and explicit loops so learners
can inspect every update.

- [ ] **Step 5: Build the visual experiments**

- Research: fixed two-moons data, decision regions, loss curve, selected
  gradient sign, train/validation split, seed comparison.
- Game: one-screen arena, gem and hazard direction vectors, scripted teacher,
  target versus predicted movement arrows, loss display, collision and
  collection counters.

- [ ] **Step 6: Write `content/unit-neural-02.md`**

Required section order:

```markdown
# Neural Foundations 2 — A Tiny Network Learns
## 1 · Why one neuron is not enough
## 2 · Two layers, one forward pass
## 3 · Loss measures the mismatch
## 4 · One gradient by hand
## 5 · Backpropagation as bookkeeping
## 6 · Choose your path
## 7 · Research path — nonlinear classification
## 8 · Game path — arena collector
## 9 · Break training deliberately
## 10 · Training versus inference
## 11 · Stretch goals
## What's next
```

The bottom breadcrumb points to `unit-01.md`, which will become RL Essentials
in Task 4.

- [ ] **Step 7: Validate and commit**

Run:

```bash
./scripts/test-neural-foundations.sh
```

Expected: forward and gradient tests pass in both languages; MkDocs builds.

Commit:

```bash
git add content/unit-neural-02.md examples/neural_foundations
git commit -m "feat: add tiny-network foundations unit"
```

---

### Task 4: Split current Unit 1 into Essentials and Deep Dive

**Files:**
- Modify: `content/unit-01.md`
- Create: `content/unit-rl-foundations-deep.md`
- Modify: `content/unit-neural-02.md`
- Modify later in Task 7: German pages and global links

- [ ] **Step 1: Copy the current file before editing**

Run:

```bash
cp content/unit-01.md /tmp/unit-01-before-foundations-split.md
```

- [ ] **Step 2: Rewrite `content/unit-01.md` as RL Essentials**

Retain and adapt:

- current Sections 1 and 2;
- observation/action/reward/return/discount/episode material from Section 4;
- policy and intuitive exploration material from Section 5;
- the Godot/Python loop from Section 6;
- the reward tweak from Section 7.

Remove from this page:

- MC versus TD;
- detailed ε-greedy and entropy mechanics;
- curiosity;
- algorithm taxonomy;
- on/off policy comparisons.

Required final headings:

```markdown
# RL Essentials — From Network to Learning Agent
## 1 · What reinforcement learning adds
## 2 · Observation → action → reward → next observation
## 3 · Episodes, return, and discounting
## 4 · The policy is the network you built
## 5 · Exploration in one picture
## 6 · Godot and Python during training
## 7 · Quick win — change one reward
## 8 · Done when
## 9 · Stretch goals
## What's next
```

Bottom breadcrumb points to `unit-neural-03.md`.

- [ ] **Step 3: Create the deep-dive page**

Create `content/unit-rl-foundations-deep.md` from the removed material and use:

```markdown
# RL Foundations Deep Dive
## 1 · Read your Foundations 3 run
## 2 · Monte Carlo versus Temporal Difference
## 3 · Bootstrapping and TD error
## 4 · Exploration mechanisms
## 5 · Value, policy, and Actor-Critic methods
## 6 · On-policy versus off-policy learning
## 7 · Model-free versus model-based learning
## 8 · Algorithm map for the rest of the course
## 9 · Taxonomy self-test
## 10 · Stretch goals
## What's next
```

Every theory section must reference either the point-robot trajectory or racer
learning curve from Foundations 3.

- [ ] **Step 4: Check content preservation**

Run:

```bash
for phrase in \
  "Monte Carlo" \
  "Temporal Difference" \
  "ε-greedy" \
  "Actor-Critic" \
  "model-based"; do
  rg -n "$phrase" content/unit-01.md content/unit-rl-foundations-deep.md
done
```

Expected: each advanced phrase appears in the deep-dive page; Unit 1 contains
only brief intuitive mentions where needed.

- [ ] **Step 5: Build and commit**

Run:

```bash
conda run -n mkdocs-env mkdocs build --strict
```

Commit:

```bash
git add content/unit-01.md content/unit-neural-02.md \
  content/unit-rl-foundations-deep.md
git commit -m "docs: split RL essentials from foundations deep dive"
```

---

### Task 5: Build Foundations 3 Research path

**Files:**
- Create: `examples/neural_foundations/research/point_robot.py`
- Create: `examples/neural_foundations/research/reinforce.py`
- Create: `examples/neural_foundations/research/tests/test_point_robot.py`
- Create: `examples/neural_foundations/research/tests/test_returns.py`
- Create: `content/unit-neural-03.md`

- [ ] **Step 1: Test deterministic environment transitions**

Tests must fix the seed and assert:

- forward action changes position by the declared speed;
- left/right actions change heading symmetrically;
- ray distances stay in `[0, 1]`;
- collision ends the episode;
- reaching the target sets success;
- reset restores the declared start state.

- [ ] **Step 2: Test discounted returns**

For rewards `[1.0, 2.0, 3.0]` and `gamma=0.5`, assert:

```text
[2.75, 3.5, 3.0]
```

- [ ] **Step 3: Implement the environment**

`PointRobotEnv` must expose:

```python
def reset(self, seed: int) -> np.ndarray
def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]
def observation(self) -> np.ndarray
def render(self, axis) -> None
```

The observation contains three normalized ray distances and normalized target
bearing. Actions are forward-left, forward, and forward-right at fixed speed.

- [ ] **Step 4: Implement REINFORCE**

`reinforce.py` must expose:

```python
def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor
def train_episode(env: PointRobotEnv, policy: PolicyNetwork, optimizer, gamma: float) -> float
def evaluate(env_factory, policy: PolicyNetwork, seeds: list[int]) -> dict[str, float]
```

The evaluation result includes mean return, standard deviation, success rate,
and mean episode length.

- [ ] **Step 5: Add the Research section to `unit-neural-03.md`**

Required shared and Research headings:

```markdown
# Neural Foundations 3 — Learn from Reward
## 1 · From labeled examples to reward
## 2 · Policy, trajectory, and return
## 3 · Why actions must be sampled during training
## 4 · Choose your path
## 5 · Research path — 2D point robot
## 6 · Build REINFORCE
## 7 · Evaluate five seeds
## 8 · Ablate sensors and reward shaping
```

- [ ] **Step 6: Validate and commit**

Run the complete validation script and confirm five-seed evaluation produces a
JSON or Markdown summary saved under
`examples/neural_foundations/research/results/`.

Commit:

```bash
git add content/unit-neural-03.md examples/neural_foundations
git commit -m "feat: add REINFORCE point-robot foundations path"
```

---

### Task 6: Build Foundations 3 Game path and native deployment

**Dependency gate:** Start only after a public `godot-native-rl` release
contains the tested prebuilt libraries for macOS arm64, Windows x86_64, and
Linux x86_64. The course release must pin one immutable tag. That release must
also pass the protocol compatibility suite against the course pin
`godot-rl==0.5.0` / wire protocol `0.3`; do not upgrade the course dependency
as part of this task.

**Files:**
- Create: `examples/neural_foundations/game/addons/godot_native_rl/`
- Create: `examples/neural_foundations/game/ncnn_runner.gdextension`
- Create: `examples/neural_foundations/game/unit_03_racer/racer_agent.gd`
- Create: `examples/neural_foundations/game/unit_03_racer/racer.gd`
- Create: `examples/neural_foundations/game/unit_03_racer/racer_train.tscn`
- Create: `examples/neural_foundations/game/unit_03_racer/racer_eval.tscn`
- Create: `examples/neural_foundations/game/test/test_racer_math.gd`
- Create: `examples/neural_foundations/game/test/run_protocol_smoke.py`
- Create: `scripts/train-foundations-racer.sh`
- Create: `scripts/verify_racer_policy.py`
- Modify: `content/unit-neural-03.md`

- [ ] **Step 1: Import the pinned native release**

Copy the release's addon, `.gdextension`, and platform libraries into the game
example. Record the exact release tag and SHA-256 checksums in:

```text
examples/neural_foundations/game/GODOT_NATIVE_RL_VERSION
```

The file format is:

```text
tag=<immutable-release-tag>
macos_arm64_sha256=<checksum>
windows_x86_64_sha256=<checksum>
linux_x86_64_sha256=<checksum>
```

- [ ] **Step 2: Verify the pinned training protocol**

Run a release-provided protocol smoke scene against the existing `godot_env`.
The smoke test must complete:

1. Python handshake;
2. `env_info` exchange;
3. reset;
4. one observation;
5. one two-value continuous action;
6. one reward;
7. one terminal episode and second reset;
8. clean close.

Run:

```bash
conda run -n godot_env python \
  -c "import importlib.metadata; assert importlib.metadata.version('godot-rl') == '0.5.0'"

conda run -n godot_env python \
  examples/neural_foundations/game/test/run_protocol_smoke.py
```

Expected: exit code `0` and a final `protocol smoke: PASS`. A version warning is
not sufficient; all eight exchanges must pass.

- [ ] **Step 3: Test pure racer calculations**

`test_racer_math.gd` must cover:

- ray normalization;
- heading-error normalization;
- steering and throttle clamping;
- ordered-checkpoint reward;
- collision penalty;
- immobility timeout;
- reset state.

- [ ] **Step 4: Implement the racer agent contract**

`racer_agent.gd` extends the pinned `NcnnAIController2D` and implements:

```gdscript
func get_obs() -> Dictionary
func get_reward() -> float
func get_action_space() -> Dictionary
func set_action(action) -> void
func reset() -> void
func get_info() -> Dictionary
```

The action space is:

```gdscript
{
	"drive": {
		"size": 2,
		"action_type": "continuous",
		"squash": true
	}
}
```

- [ ] **Step 5: Build training and evaluation scenes**

Training scene requirements:

- `NcnnSync` in `TRAINING` mode;
- visible ray sensors;
- visible steering, throttle, step reward, episode return, and checkpoint;
- one closed track;
- no art assets beyond primitives.

Evaluation scene requirements:

- `NcnnSync` in `NCNN_INFERENCE` mode;
- fixed start-seed list;
- twenty paired starts;
- metrics written to JSON.

- [ ] **Step 6: Create the training command**

`scripts/train-foundations-racer.sh` must:

1. activate or invoke `godot_env`;
2. start SB3 PPO with the pinned `godot-rl` package;
3. launch the headless training scene;
4. save a `.zip` checkpoint;
5. export ONNX;
6. stop cleanly if either process fails.

- [ ] **Step 7: Implement three-stage parity verification**

`verify_racer_policy.py` must:

1. load twenty fixed observations;
2. run the PyTorch actor;
3. run ONNX Runtime;
4. assert maximum absolute error `≤ 1e-5`;
5. run the converted ncnn model;
6. assert maximum absolute error from ONNX `≤ 1e-2`;
7. print input/output names and shapes;
8. write a machine-readable report.

- [ ] **Step 8: Complete the Game section of `unit-neural-03.md`**

Add:

```markdown
## 9 · Game path — build the arcade racer
## 10 · Define observations and actions
## 11 · Design reward and episode boundaries
## 12 · Train with PPO
## 13 · Inspect the ONNX graph
## 14 · Verify PyTorch, ONNX, and ncnn
## 15 · Run native inference in Godot
## 16 · Diagnose reward and sensor failures
## 17 · Compare the two paths
## 18 · Stretch goals
## What's next
```

Bottom breadcrumb points to `unit-rl-foundations-deep.md`.

- [ ] **Step 9: Validate and commit**

Required commands:

```bash
./scripts/test-neural-foundations.sh
./scripts/train-foundations-racer.sh
conda run -n godot_env python scripts/verify_racer_policy.py
```

Expected:

- all deterministic tests pass;
- the racer completes at least one lap in the reference run;
- ONNX parity is within `1e-5`;
- ncnn parity is within `1e-2`;
- twenty-start native success rate differs from Python by no more than five
  percentage points.

Commit:

```bash
git add content/unit-neural-03.md examples/neural_foundations \
  scripts/train-foundations-racer.sh scripts/verify_racer_policy.py
git commit -m "feat: add native PPO racer foundations path"
```

---

### Task 7: Publish the complete sequence

**Files:**
- Modify: `mkdocs.yml`
- Modify: `content/index.md`
- Modify: `content/index.de.md`
- Create: `content/unit-neural-01.de.md`
- Create: `content/unit-neural-02.de.md`
- Modify: `content/unit-01.de.md`
- Create: `content/unit-neural-03.de.md`
- Create: `content/unit-rl-foundations-deep.de.md`
- Modify: `content/unit-00.md`
- Modify: `content/unit-00.de.md`
- Modify: `content/unit-reward-engineering.md`
- Modify: `content/unit-reward-engineering.de.md`
- Modify: prerequisite links across `content/*.md`
- Modify: `docs/example-progression.md`
- Modify: `README.md`

- [ ] **Step 1: Set final Phase 1 navigation**

Use this order in `mkdocs.yml`:

```yaml
  - "Phase 1 — Foundations":
    - "Unit 0 — Setup & First Run": unit-00.md
    - "Neural Foundations 1 — One Neuron": unit-neural-01.md
    - "Neural Foundations 2 — Tiny Networks": unit-neural-02.md
    - "RL Essentials": unit-01.md
    - "Neural Foundations 3 — Learn from Reward": unit-neural-03.md
    - "RL Foundations Deep Dive": unit-rl-foundations-deep.md
    - "Reward Engineering": unit-reward-engineering.md
    - "Unit 2 — Build Your First Env": unit-02.md
```

- [ ] **Step 2: Repair the complete breadcrumb chain**

Final chain:

```text
unit-00
→ unit-neural-01
→ unit-neural-02
→ unit-01
→ unit-neural-03
→ unit-rl-foundations-deep
→ unit-reward-engineering
→ unit-02
```

Update top and bottom breadcrumbs plus every “What's next” section.

- [ ] **Step 3: Translate the five affected pages**

Create German pages with matching heading structure. Preserve code, paths,
package names, metrics, ONNX names, and ncnn names in English.

- [ ] **Step 4: Audit prerequisite links**

Run:

```bash
rg -n "unit-01\\.md" content --glob '*.md'
```

Rules:

- operational RL-loop prerequisites continue pointing to `unit-01.md`;
- MC/TD, taxonomy, on/off-policy, and detailed exploration prerequisites point
  to `unit-rl-foundations-deep.md`;
- neural-network prerequisites point to the relevant neural foundations unit.

- [ ] **Step 5: Run complete validation**

Run:

```bash
./scripts/test-neural-foundations.sh
conda run -n mkdocs-env mkdocs build --strict
```

Manually inspect:

- English and German nav;
- all eight Phase 1 pages;
- mobile-width plot readability;
- equations in dark theme;
- fallback behavior for downloadable example files.

- [ ] **Step 6: Mark the design implemented**

Change the spec status to:

```markdown
**Status:** Implemented
```

Add links to the reference examples and final validation commands.

- [ ] **Step 7: Commit**

```bash
git add mkdocs.yml content docs README.md examples scripts requirements-course.txt
git commit -m "feat: publish visual neural foundations track"
```

---

## Program completion gate

Do not declare the track complete until:

- all three units contain running visual examples for both paths;
- Unit 1 is split into RL Essentials and RL Foundations Deep Dive;
- Foundations 3 sits between those two pages;
- all English and German breadcrumbs form one continuous chain;
- Python manual gradients match PyTorch;
- the point robot reports multi-seed results;
- the racer trains through `godot-native-rl`;
- ONNX is inspected and numerically verified;
- ncnn inference passes numerical and behavioral parity;
- `mkdocs build --strict` passes;
- the complete test script passes from a fresh checkout.
