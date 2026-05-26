# Sim-to-Real Transfer — Domain Randomization and the Reality Gap

[← Goal-Conditioned RL & HER](unit-her.md) · [Course home](index.md)

!!! info "Three ways to see your AI"
    - **Godot** — test your policy at the *extremes* of your domain randomization (max friction, min mass, max noise). If it survives there, it has a shot on hardware.
    - **TensorBoard** — watch `ep_rew_mean` and reward *variance* across seeds. Low variance under DR is your proxy for sim-to-real robustness.
    - **Deployment checklist** — a physical robot is unforgiving. Before you flash firmware, walk through Section 7 line by line.

---

You have built quadrupeds, manipulators, and goal-conditioned graspers in Godot. You have trained them with PPO, SAC, and HER. They work beautifully in simulation.

Now you want them to work on a real robot.

This unit is the bridge. It is the most practically important — and most humbling — chapter of applied robotics RL.

## 1 · The reality gap

The single most important concept in applied robotics RL is this:

> A policy trained in simulation often fails *completely* when deployed on real hardware.

Not "performs slightly worse." Not "needs a little fine-tuning." It falls over. It crushes the object. It diverges into a NaN spiral and the emergency stop fires.

The classic cautionary tale is OpenAI's Dexterous Hand. They trained a 24-DOF Shadow Hand to manipulate a Rubik's cube using **roughly 10,000 CPU cores** and the equivalent of **about 100 years of simulated experience**. When they deployed the first version on the real hand, it failed. The simulation — despite being one of the most carefully engineered robotics simulators ever built — was *not physically accurate enough*. Contact dynamics, tendon stiffness, and finger friction were each a little wrong, and the policy's confidence was misplaced.

### Where the gap comes from

The reality gap is not a single bug — it's a stack of mismatches between two physical models (the simulator and reality):

- **Physics inaccuracy.** Simulators integrate rigid-body dynamics with discrete time steps. Real friction is stick-slip. Real contacts deform. Motor stiffness depends on temperature. Sim approximates all of this.
- **Observation noise.** Real sensors have noise floors, calibration drift, occasional dropouts, and quantization. A simulated IMU returns the ground-truth orientation; a real IMU returns *something near* the orientation, plus white noise, plus a slow bias.
- **Actuator lag.** Real motors have latency: 50–200 ms between commanded torque and produced torque, depending on the controller, the bus (CAN, EtherCAT, USB), and the motor electronics. Simulators have *zero* latency by default.
- **Unmodeled dynamics.** Cable routing pulls on links. Thermal expansion shifts encoder zero points. Backlash in gearboxes. Wear changes friction over months.
- **Visual gap.** If your policy uses cameras, sim textures and lighting will never match real photons.

### A concrete consequence

Imagine you train a quadruped to walk in Godot. The floor `PhysicsMaterial.friction` is 0.9. The policy learns a confident, springy gait that pushes hard against the floor on every step.

You deploy it on a polished concrete lab floor. Real friction: 0.6. The first push slips. The robot tries to recover, slips again, and goes down on its chest before you can hit the kill switch.

Sim reward: 980/1000. Real reward: zero, plus a bent shin plate.

The point is not "make the simulator perfect." Perfect simulators do not exist, and the closer you push toward "perfect" the steeper the diminishing returns. The point is the next section.

### Why "just buy a better simulator" doesn't work

It's tempting to think the answer is a better physics engine — softer contact models, FEM-based deformables, GPU-accelerated rigid-body solvers. These help. But three structural reasons make "more accurate sim" insufficient on its own:

1. **You cannot measure reality precisely enough.** Even with a perfect simulator, you must feed it the *actual* mass, friction, motor parameters of *your specific* robot. Those values drift across units, across temperatures, and across the lifetime of the robot. The sim is only as accurate as the parameters you put in.
2. **Some physics is genuinely hard.** Stick-slip friction, cable hysteresis, tendon-driven coupling, soft-tissue contact — these are active research problems in numerical simulation, not solved engineering.
3. **More accuracy means slower sim.** A 100×-more-accurate physics engine that runs 100× slower trains 100× fewer episodes per wall-clock hour. The data-efficiency hit usually outweighs the accuracy gain.

Domain randomization sidesteps all three. You don't need exact parameters — you need *enough* parameters in your distribution. You don't need to model stick-slip exactly — you need to randomize the friction parameters enough to cover both regimes. You don't need a slower sim — randomization runs at full speed.

---

## 2 · Domain randomization — the core solution

The key insight, articulated by OpenAI in 2017 and now standard practice across robotics labs:

> Instead of trying to perfectly model reality, **randomize simulation parameters so widely that reality is just another sample from your training distribution.**

If you train on floor friction uniformly sampled from `[0.3, 1.5]`, a real floor with friction `0.6` is no longer out-of-distribution. It's a fairly ordinary training episode. The policy has *already* learned to walk on it.

This reframes the problem completely. You stop chasing the elusive "accurate simulator" and instead chase a *robust policy* that survives across a wide range of physics.

### What to randomize

| Parameter | Godot property | Typical range |
|-----------|---------------|---------------|
| Link mass | `RigidBody3D.mass` | ×0.7 to ×1.3 |
| Joint friction | `PhysicsMaterial.friction` | 0.2 to 1.0 |
| Joint damping | `Generic6DOFJoint3D` linear/angular damping | ×0.5 to ×2.0 |
| Motor strength | Scale applied torque | ×0.8 to ×1.2 |
| Observation noise std | `randfn(0, σ)` on each obs | 0.01 to 0.05 |
| Action delay (latency) | Buffer last N actions | 0 to 3 steps |
| Ground friction | Floor `PhysicsMaterial.friction` | 0.3 to 1.5 |
| External perturbations | Random force impulse | Every 50–200 steps |

Three rules of thumb:

1. **Randomize wider than you think.** If real motor strength is `1.0`, randomize on `[0.7, 1.3]`, not `[0.95, 1.05]`. The cost of "too wide" is slower training; the cost of "too narrow" is total deployment failure.
2. **Randomize at episode reset, not per-step.** The agent should treat each episode as a *fixed instance* of physics. Re-sampling every step makes the world non-Markovian and destabilizes learning.
3. **Sample from sensible distributions.** Uniform is fine for most things. Log-uniform for parameters that span orders of magnitude (stiffness, damping). Gaussian *only* for sensor noise.

### A worked sizing example

Suppose you have a quadruped where each leg link has a measured mass of 0.5 kg. Your scale's resolution is ±10 g, so already there's a 2% measurement uncertainty. Across a production batch of robots you observe link masses ranging from 0.46 kg to 0.54 kg (±8% spread). Temperature changes the effective inertia slightly, motor controllers re-zero between sessions, and you cannot guarantee which of your six identical robots you'll be deploying on.

A sensible mass range is *not* `[0.49, 0.51]` — that's a measurement precision range. It's not `[0.46, 0.54]` either — that's the observed unit-to-unit spread. It's `[0.35, 0.65]`: about ±30% around nominal. The extra margin absorbs unknown unknowns (a payload you didn't anticipate, a battery swap, accumulated dust). The training cost of this extra margin is modest; the cost of *not* having it is a policy that fails on unit #7.

---

## 3 · Implementing domain randomization in Godot

Here is a complete pattern. The environment root script holds the randomization logic and exposes its current sample to the agent's `get_obs` / `set_action`.

```gdscript
# In the environment root script — randomize at every episode reset
extends Node3D

@onready var robot_body  = $Robot/Body
@onready var leg_joints  = $Robot.find_children("*Joint*", "Generic6DOFJoint3D")
@onready var floor_mat   = preload("res://materials/floor.tres")

# Domain randomization ranges
const MASS_RANGE      = Vector2(0.7, 1.3)     # multiplier on base mass
const FRICTION_RANGE  = Vector2(0.3, 1.2)
const DAMPING_RANGE   = Vector2(0.5, 2.0)
const FORCE_RANGE     = Vector2(0.8, 1.2)     # motor strength multiplier
const NOISE_STD_RANGE = Vector2(0.005, 0.03)  # observation noise σ

var obs_noise_std: float = 0.01
var motor_strength: float = 1.0

func randomize_domain():
    # Randomize body mass
    robot_body.mass = base_mass * randf_range(MASS_RANGE.x, MASS_RANGE.y)

    # Randomize joint damping
    for joint in leg_joints:
        var damping = base_damping * randf_range(DAMPING_RANGE.x, DAMPING_RANGE.y)
        joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_DAMPING, damping)

    # Randomize floor friction
    var new_mat = floor_mat.duplicate()
    new_mat.friction = randf_range(FRICTION_RANGE.x, FRICTION_RANGE.y)
    $Floor.physics_material_override = new_mat

    # Store for use in get_obs() and set_action()
    obs_noise_std  = randf_range(NOISE_STD_RANGE.x, NOISE_STD_RANGE.y)
    motor_strength = randf_range(FORCE_RANGE.x, FORCE_RANGE.y)

func reset():
    randomize_domain()
    # ... rest of reset
```

And on the agent side:

```gdscript
# In ai_controller.gd — use the randomized noise
func get_obs() -> Dictionary:
    var obs = []
    for value in raw_observations():
        obs.append(value + randfn(0.0, env.obs_noise_std))
    return {"obs": obs}

func set_action(action) -> void:
    for i in range(joints.size()):
        var torque = action["joints"][i] * MAX_TORQUE * env.motor_strength
        apply_joint_torque(i, torque)
```

### Implementation notes

- **`duplicate()` the material.** Editing `floor_mat.friction` directly mutates the shared resource — all parallel environments would see the same friction. `duplicate()` gives each episode its own copy. This is the same immutability lesson you learned writing Python: never mutate shared state.
- **Store base values once.** Cache `base_mass`, `base_damping` in `_ready()`. Otherwise you compound randomization across resets and the parameter distribution drifts.
- **Expose the sample to the agent.** The agent script needs `env.obs_noise_std` and `env.motor_strength`. Pass the environment node in via `@export var env: Node3D` or fetch it from the parent.
- **Log the sample for debugging.** When a training run goes sideways, the first question is "which DR samples did the worst episodes have?" Push `obs_noise_std`, `motor_strength`, and friction into the info dict so they appear in TensorBoard / the replay buffer metadata.
- **Re-seed `RandomNumberGenerator` per environment.** If you run N parallel Godot environments, default RNG state can correlate them. Construct a fresh `RandomNumberGenerator` per env and seed it from the env index, or from `Time.get_unix_time_from_system()`. Otherwise all parallel envs sample the same friction every reset and your "diversity" is fake.

---

## 4 · Observation noise as the minimum viable DR

If you do nothing else from this unit, do this: **always add observation noise.**

```gdscript
func get_obs() -> Dictionary:
    var obs = []
    for value in raw_observations():
        obs.append(value + randfn(0.0, env.obs_noise_std))
    return {"obs": obs}
```

Three reasons:

1. **It's free.** One line of GDScript. No physics changes. No new training infrastructure.
2. **It forces robustness.** The policy must learn features that survive small sensor perturbations. Brittle features (e.g., "joint angle is exactly 1.4523 rad") get punished out of existence during training.
3. **It's a prerequisite, not a substitute.** If your policy fails with `σ=0.02` noise in sim, it will *certainly* fail on real hardware, because real sensor noise is typically larger than that.

The reverse is not true — passing the noise test in sim does not guarantee real-world success — but failing it guarantees failure. Use it as a cheap filter.

### How much noise?

A reasonable starting heuristic is to set `σ` to roughly the magnitude of expected sensor noise on real hardware, expressed in the same units as the observation:

- Joint encoders (14-bit absolute): noise floor ~`0.001` rad. Train with `σ = 0.005` to add margin.
- IMU orientation (consumer-grade MEMS): drift + noise ~`0.02` rad. Train with `σ = 0.03`.
- IMU gyro: noise ~`0.01` rad/s. Train with `σ = 0.02`.
- Motor current sensing: ~5% of full-scale. Apply *multiplicative* noise of σ=0.05, not additive.

Then randomize `σ` itself per episode within `[0.5×, 2×]` of these values. A policy that has only ever seen one noise level can become brittle to it; randomizing the noise *level* forces robustness to the noise *characteristics*, not just the noise magnitude.

### Where to inject noise

Add noise to the observation *as the agent sees it*, not to the underlying state. The simulator's true state should remain clean — only the observation channel is corrupted. This matters because:

1. Reward computation should use the *true* state, not the noisy observation. Otherwise reward becomes a moving target.
2. Termination conditions ("robot fell over") should use true state. A noisy IMU should not falsely trigger episode termination.
3. The privileged critic (Section 6) explicitly *needs* the clean state.

!!! warning "Never deploy without testing DR extremes"
    Before you flash a policy onto hardware, run it in sim with **every randomized parameter pushed to its extreme value**. Max friction *and* min motor strength *and* max noise *and* max latency, all at once. If episode reward collapses there, you do not have a deployable policy — you have a policy that lives inside the comfortable middle of your distribution. Real hardware will *not* hand you the middle.

---

## 5 · Action delay / latency simulation

Real motors do not respond instantly. There is a delay between "command sent" and "torque produced":

- USB-based hobby servos: 20–50 ms
- CAN-bus quadruped motors (e.g., T-Motor, MIT Cheetah): 5–15 ms
- ROS2 control loops with networking: 50–200 ms
- Hydraulic actuators (Boston Dynamics class): 10–40 ms

At a 60 Hz physics step (16.7 ms per step), a 50 ms latency is **3 steps of delay** — and a policy that has never seen delay will issue corrective actions *before* its previous corrections have even taken effect. The result is oscillation and instability.

Simulating latency is trivial:

```gdscript
# Simulate actuator latency by buffering actions
var action_buffer: Array = []
const LATENCY_STEPS = 2  # randomize between 1-3 during DR

func set_action(action) -> void:
    action_buffer.append(action)
    if action_buffer.size() > LATENCY_STEPS:
        var delayed_action = action_buffer.pop_front()
        _apply_torques(delayed_action)
    # If buffer not full yet, apply zero torque (startup behavior)
```

A few subtle points about this implementation:

- **Startup behavior matters.** During the first `LATENCY_STEPS` steps of an episode, the buffer isn't full yet. Applying zero torque is one option; holding the initial command is another. Real systems usually hold the last commanded value, so prefer that for fidelity.
- **The agent's observation is from "now," but the action it sees applied is from "then."** If you train without latency the agent learns to react synchronously. After adding latency the agent must learn to *predict* — to issue commands that account for the delay between command and effect. This is genuinely harder and takes more samples to learn, but it's what real hardware requires.
- **Don't apply latency to the reward signal.** Reward is computed on the simulator's true state at the current step. Only the *action application* is delayed.

For full DR, randomize `LATENCY_STEPS` per episode:

```gdscript
var latency_steps: int = 0

func randomize_domain():
    # ... other randomizations ...
    latency_steps = randi_range(0, 3)
```

This is one of the highest-leverage DR techniques and one of the least frequently implemented. Most published sim-to-real failures involve a policy that was trained against zero latency and could not handle real-motor lag. Adding it costs a handful of lines and dramatically improves transfer.

---

## 6 · Asymmetric actor-critic

A technique pioneered by OpenAI's Dactyl team and used heavily by ETH Zurich's ANYmal group:

- The **actor** (deployed on the real robot) sees only observations that are available on real hardware: joint encoders, IMU, motor currents, maybe a camera.
- The **critic** (only exists during training) sees *privileged* simulation state: exact object positions, true physics parameters, ground-truth contact forces, the full domain-randomization sample.

During training, the critic produces *much better* value estimates because it can "cheat" — it sees the full simulator state. Better value estimates mean better policy gradients (via the advantage), which means the actor learns faster and to higher performance.

During deployment, you discard the critic. The actor was trained against high-quality gradients, but it only ever needed the limited observation set.

### Plain English

> Train a coach (critic) who can see everything — wind speed, the opposing team's playbook, the exact muscle fatigue of your runner. Train the player (actor) who sees only what they'd see in a real game. The coach gives the player better feedback during practice. On game day, the player runs alone.

### Sketch implementation

```python
# In Python training — custom policy with asymmetric inputs
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn

class AsymmetricPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override critic to accept larger observation
        privileged_obs_dim = 64  # sim state only available in training
        self.mlp_extractor.value_net = nn.Sequential(
            nn.Linear(self.features_dim + privileged_obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )

    def evaluate_actions(self, obs, privileged_obs, actions):
        # Concatenate privileged info to critic input
        features = self.extract_features(obs)
        critic_input = torch.cat([features, privileged_obs], dim=-1)
        values = self.value_net(critic_input)
        # ... rest of evaluation
```

On the Godot side, expose two observation dictionaries from the agent: `obs` (real-hardware-equivalent) and `privileged_obs` (sim ground truth). The training loop reads both; the deployed ONNX policy reads only the first.

!!! tip "Asymmetric actor-critic is a free upgrade"
    If your environment exposes useful privileged info (true object pose, exact contact forces, full DR sample), adding it to the critic costs almost nothing and frequently improves training stability *and* final performance. The policy you deploy is unchanged — same input dimensions, same network — but it was trained better.

---

## 7 · Real-world deployment checklist

Print this. Tape it next to the kill switch. Walk through it line by line *before* the first hardware test.

- [ ] Domain randomization covers the real hardware's parameter range (measure physical properties — weigh links, measure friction, time the actuators)
- [ ] Observation noise added to all sensors in sim
- [ ] Action delay matches real motor latency (measure with oscilloscope or timestamp logs)
- [ ] Joint limit violations terminate episodes in sim (protect hardware — a policy that learns to ram into joint stops in sim will destroy gearboxes in reality)
- [ ] Energy efficiency reward prevents overheating behavior (penalize `sum(|torque|)` or `sum(torque²)`)
- [ ] Policy tested at sim DR extremes (min mass, max friction, max noise) — if it fails here, it will fail on hardware
- [ ] ONNX export tested: inference latency per step is well under the physics timestep (target ≤ 5 ms for a 60 Hz loop)
- [ ] Fallback behavior defined: what happens when the policy outputs NaN or out-of-range values? Clamp + log + (after N violations) cut power
- [ ] Smooth-startup behavior: first few seconds of deployment use damped commands, not full policy output
- [ ] Emergency stop verified independent of the policy — the kill switch cuts motor power directly, not via software

### A note on measurement

Before you randomize, *measure*. The most common DR failure is "I randomized friction on `[0.3, 1.5]` but the real floor was 1.8." Take a fish scale, a stopwatch, and a multimeter to the lab:

- Weigh each link of the robot. Set `MASS_RANGE` to `[0.7 × measured, 1.3 × measured]`.
- Drag the robot's foot across the floor with a known normal force and a fish scale. Compute coefficient of friction. Then expand the range by ±30%.
- Send a step torque command and record the encoder. The 10–90% rise time is your latency.
- Read the motor datasheet for `kt` (torque constant). Compare to commanded torque under load. The mismatch is your motor-strength factor — randomize around it.

Measurement is unglamorous and irreplaceable.

### The first hardware test: do less, watch closely

When you finally put a policy on real hardware:

1. **Tether the robot.** Suspend a legged robot from a gantry, or mount a manipulator behind a safety cage. Catch the first fall instead of repairing it.
2. **Run at reduced action gain.** Multiply policy outputs by 0.3 for the first session. Verify the qualitative behavior matches sim before unlocking full torque.
3. **Record everything.** Joint positions, commanded torques, IMU, motor currents, and a video. The first hardware run is your reference for every future debugging session.
4. **Compare distributions, not single trajectories.** Run 20 episodes in sim with the same initial conditions and DR samples spanning your training distribution. Run 20 episodes on hardware. Plot the distribution of step rewards, contact patterns, and torque profiles. If the hardware distribution is *outside* the sim distribution, your DR range is too narrow — go back and widen.

---

## 8 · Famous sim-to-real successes and lessons

### OpenAI Dexterous Hand / Dactyl (2019)

- **Problem.** A 24-DOF Shadow Hand solving the Rubik's Cube — one of the hardest contact-rich manipulation tasks attempted in robotics.
- **Solution.** Massive domain randomization over a thousand-plus parameters (object mass, friction, tendon stiffness, motor latency, observation noise, lighting and camera parameters for the vision pipeline), combined with an asymmetric actor-critic and an LSTM-based policy that could *implicitly identify* the current dynamics from a short history.
- **Result.** Policy trained in sim for the equivalent of ~13,000 years of experience transferred directly to the real hand, solving cubes that had been physically disturbed mid-solve.
- **Lesson.** At scale, **DR range matters more than sim accuracy.** They did not have a perfect simulator. They had a simulator that produced training distributions so wide that reality lived comfortably inside.

### ETH Zurich ANYmal (2019)

- **Problem.** Quadrupedal locomotion across unstructured natural terrain — stairs, rocks, mud, sand.
- **Solution.** DR over terrain height maps, motor strength, payload, and external pushes. Privileged critic that received ground-truth height maps and contact forces; the deployed actor saw only proprioception (joint angles, IMU). A teacher-student distillation pipeline let the actor "compress" knowledge of privileged information into observable features.
- **Result.** The real robot walked up flights of stairs and over obstacles it had never seen in training.
- **Lesson.** **Terrain randomization is critical for locomotion.** If you train on a flat plane, you get a flat-plane gait. Vary the mesh height map every episode (Perlin noise, scattered boxes, randomized step heights) and the policy generalizes.

### Berkeley Agility Cassie (2022)

- **Problem.** Bipedal running and jumping on varied terrain — a much harder balance problem than quadrupedal locomotion.
- **Solution.** Comparatively *minimal* DR. The team invested heavily in simulator accuracy: precise motor models, careful identification of physical parameters, hand-tuned reward shaping.
- **Result.** Cassie ran a 5K on real hardware using a sim-trained policy.
- **Lesson.** **DR vs. sim accuracy is a tradeoff.** When the system has tight stability margins (bipedal balance), a more accurate sim with narrower DR can outperform a wider DR with a sloppier sim. Use DR — but do not let it become an excuse for never improving your physics.

### Boston Dynamics Spot (commercial, 2020+)

- **Problem.** Industrial-grade quadruped that must be reliable across thousands of deployed units.
- **Solution.** Public details are limited, but the consensus is a hybrid: classical model-predictive control for the core gait, with learned policies layered on top for higher-level behaviors. Sim-to-real is one tool in a larger toolbox.
- **Lesson.** **Production robotics is rarely pure end-to-end RL.** Sim-to-real is the right tool for some sub-problems and the wrong tool for others. Know which is which.

---

## 9 · Progressive domain randomization (curriculum)

Wide DR is hard to learn from cold. A policy that has never walked at all cannot simultaneously learn locomotion *and* generalize across friction in `[0.3, 1.5]`.

Solution: start narrow, expand gradually as the policy improves. This is just curriculum learning applied to physics parameters.

```python
# Schedule DR range to grow with ep_rew_mean
def update_dr_range(env, ep_rew_mean, target_reward=800.0):
    base_noise = 0.005
    max_noise  = 0.05
    # Increase randomization as agent improves
    progress = min(ep_rew_mean / target_reward, 1.0)
    env.set_noise_std(base_noise + progress * (max_noise - base_noise))
```

A more general pattern:

```python
def dr_schedule(progress: float) -> dict:
    """progress in [0, 1] — interpolates DR ranges from narrow to wide."""
    def interp(narrow, wide):
        return narrow + (wide - narrow) * progress

    return {
        "mass_range":     (interp(0.95, 0.7), interp(1.05, 1.3)),
        "friction_range": (interp(0.85, 0.3), interp(0.95, 1.5)),
        "noise_std_max":  interp(0.005, 0.05),
        "latency_max":    int(interp(0, 3)),
    }
```

Push these values into the Godot environment via the `set_obs` channel that `godot-rl-agents` already gives you. The agent script reads them on every reset.

Two warnings about progressive DR:

- **Do not collapse the range when reward drops.** If you expand DR and the agent struggles, do *not* immediately narrow back — that just oscillates. Hold the range constant until the agent recovers.
- **Always test at the final range before claiming success.** A policy that achieves high reward at progress=0.5 is not deployable. Train until `ep_rew_mean` is high at progress=1.0.

### When to advance the curriculum

Two common triggers, both useful:

- **Reward-based advancement.** Advance `progress` whenever `ep_rew_mean` exceeds a threshold (e.g., 80% of the maximum achievable at the current DR level). This is robust to schedule mistuning — the curriculum waits for the agent rather than forcing it forward.
- **Step-based advancement.** Linearly grow `progress` from 0 to 1 over the first N million environment steps. Simpler, but vulnerable to bad timing: if N is too small the agent never catches up; if N is too large you waste compute.

A hybrid is best: linear schedule with a *gate* that pauses progression whenever `ep_rew_mean` drops below a recent moving average. That way the curriculum advances steadily by default but backs off automatically when training stalls.

---

## 10 · Stretch goals

If you want to take this unit further before the next one, here are three concrete exercises that mirror real-world sim-to-real work:

- **Robustness sweep on JumperHard.** Take your trained `JumperHard` policy from the earlier units. Evaluate it with *zero* DR. Then evaluate it with heavy DR. Measure episode reward variance at 5 distinct physics parameter values (mass × {0.7, 0.85, 1.0, 1.15, 1.3}). Plot the variance. This is exactly the kind of robustness curve a real robotics team produces before signing off on hardware deployment.
- **Action delay study.** Implement the action-delay buffer from Section 5. Train two policies — one with zero latency, one with `LATENCY_STEPS = 2` randomized per episode. Measure (a) learning speed (steps to `ep_rew_mean = 500`) and (b) final reward variance when evaluated at latencies 0, 1, 2, 3. The zero-latency policy will likely train faster and transfer worse.
- **Read the OpenAI Dactyl paper.** *"Learning Dexterous In-Hand Manipulation"* (Andrychowicz et al. 2019). The domain-randomization section (and the appendix listing every randomized parameter) is the gold-standard reference. Most of what you read will be directly applicable to your Godot environments.

---

## What's next

You now have the full applied-RL stack for robotics: algorithms (PPO, SAC), engineering (reward shaping, debugging), advanced techniques (curiosity, HER), and the bridge to hardware (sim-to-real).

The remaining frontier is no longer technical — it's *alignment*. Once your policies leave the simulator and act in the real world, the questions become:

- How do you specify what you actually want when the reward function is incomplete?
- How do you keep the policy aligned with human intent across distribution shifts you didn't anticipate?
- How do you build oversight into the deployment loop so you catch failures before they hurt someone?

The next unit covers **Safe RL and Constrained MDPs** — how to enforce hard constraints (joint limits, geofences, regulatory requirements) during training, so the policy cannot trade safety for performance. Until then: measure your robot, randomize widely, and never deploy without walking the checklist.

---

[← Goal-Conditioned RL & HER](unit-her.md) · [Course home](index.md) · [→ Safe RL / Constrained MDPs](unit-safe-rl.md)
