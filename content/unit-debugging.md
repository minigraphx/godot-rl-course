# Debugging RL Training — Symptoms, Diagnosis, and Fixes

[Course home](index.md) · [Reference](reference.md)

!!! info "Time"
    Reading: ~25 min

!!! info "How to use this guide"
    This is a diagnostic reference, not a tutorial. Start at the **top-level flowchart** below, follow the branch that matches what you are seeing, and jump to the section it points to. Each section is structured the same way: **symptom → diagnosis → concrete fix**. The "Three ways to see your AI" principle applies here too — cross-check **Godot behavior**, **TensorBoard curves**, and **code inspection** before changing anything. If two of the three disagree, that disagreement is itself the clue.

## 1 · How to read this guide

- Start at the top-level diagnostic flowchart in Section 2.
- Follow the symptom that matches your situation.
- Each downstream section gives **symptom**, **diagnosis**, and a **concrete fix** — in that order.
- Always confirm the diagnosis with all three lenses before applying a fix:
    - **Godot behavior** — what is the agent actually doing on screen?
    - **TensorBoard** — what do the curves say?
    - **Code inspection** — does `get_obs()`, `set_action()`, and the reward code do what you think it does?
- If you skip ahead and "just try things", you will fix the wrong problem and lose hours. Diagnose first.

## 2 · Top-level diagnostic flowchart

```
Is ep_rew_mean changing at all?
        |
   No (flat) ─────────────────────────────→ Section 3: No learning signal
        |
   Yes (changing)
        |
   Going DOWN? ───────────────────────────→ Section 4: Reward sign / scale
        |
   Going up but plateauing early? ────────→ Section 5: Policy converged too early
        |
   Oscillating wildly? ────────────────────→ Section 6: Training instability
        |
   TensorBoard looks good but Godot looks wrong? → Section 7: Reward hacking / eval mismatch
        |
   Works in training but not in export? ──→ Section 8: Inference / ONNX issues
        |
   Very slow (steps/sec low)? ─────────────→ Section 9: Performance
```

Open TensorBoard side-by-side with Godot before starting. Most "mystery" failures resolve once you can see both at once.

## 3 · No learning signal (ep_rew_mean flat)

**Symptom:** `rollout/ep_rew_mean` stays near `0` for 500k+ steps. The training log shows steps accumulating but no reward signal. Episode length may also be suspiciously constant.

Work through the checklist **in order**. Do not skip ahead — earlier checks are cheaper to perform and rule out the most common causes.

### 3a. Reward is always zero

**Symptom:** Reward never fires, even on success.

**Diagnosis:** Print the reward every physics step for the first 100 steps:

```gdscript
func _physics_process(delta):
    print(_ai.reward)
```

Is it ever non-zero? If not, your reward condition never fires. The most common cause is that the `done` flag is never set, so the episode never terminates and the terminal reward never executes.

**Fix:** Verify both lines exist in `game_over()`:

```gdscript
func game_over():
    _ai.reward += 1.0
    _ai.done = true
    _ai.needs_reset = true
```

### 3b. Reward sign is wrong but non-zero

If you see non-zero reward but the wrong sign, see **Section 4**.

### 3c. Observations are all zeros or constant

**Symptom:** The policy can't learn because every observation looks identical to the network.

**Diagnosis:** Print `get_obs()` for 10 steps while moving the agent manually:

```gdscript
if _ai.heuristic == "human":
    print(get_obs())
```

Are all values `0.0`? `RayCast3D` nodes may not be enabled, may be in the wrong collision layer, or may not have been added to the scene tree before `_ready()` ran. Do the values change as the agent moves? If not, the sensors are broken.

**Fix:** Enable rays explicitly (`enabled = true`), check `collision_mask`, and force `force_raycast_update()` before reading hits if you read them inside `get_obs()`.

### 3d. Action space mismatch

**Symptom:** `get_action_space()` declares one shape, `set_action()` reads another. The policy outputs garbage relative to what the env expects.

**Diagnosis:** Read both functions side-by-side.

```gdscript
func get_action_space():
    return {"move": {"size": 4, "action_type": "continuous"}}

func set_action(action):
    velocity.x = action["move"][0]
    velocity.z = action["move"][1]
    # bug: only reads 2 of 4 declared dimensions
```

**Fix:** `size` must match exactly how many elements you read in `set_action()`. For discrete actions, `size` is the number of choices. For continuous, `size` is the number of floats.

### 3e. Episode too short

**Symptom:** Episodes end in < 5 steps. The agent dies or resets before encountering any meaningful reward.

**Diagnosis:** Print episode length at reset:

```gdscript
func reset():
    print("ep_len:", _ai.episode_steps)
    _ai.episode_steps = 0
```

**Fix:** Add a small survival bonus so even short, uneventful episodes still produce gradient signal:

```gdscript
_ai.reward += 0.001  # per-step survival bonus
```

### 3f. Task is genuinely too hard for random exploration

**Symptom:** A random policy never reaches the goal within 10k episodes. There is no gradient for PPO to follow because every episode looks the same.

**Fix:** Pick one or more of:

- Add a **shaped reward** based on distance progress (`prev_dist - current_dist`).
- **Start the agent closer to the goal** at first, then gradually move it back (manual curriculum).
- Use **intrinsic curiosity** (see the Curiosity unit) so novel states are themselves rewarding.

## 4 · Reward going down (agent getting worse)

**Symptom:** `ep_rew_mean` starts non-zero but trends down over time. Eventually the agent looks worse than random.

### 4a. Reward sign flipped

**Symptom:** Agent learns to do the *opposite* of what you want.

**Diagnosis:** Read each `_ai.reward += ...` line and ask: "Is this value positive when the agent is doing the right thing?"

A classic mistake:

```gdscript
_ai.reward += -dist_to_goal   # WRONG: becomes more negative as agent approaches!
```

The reward is negative everywhere, so the agent learns to end the episode as fast as possible to stop accumulating penalty — by walking into a wall.

**Fix:** Use **progress**, not raw distance:

```gdscript
_ai.reward += (prev_dist_to_goal - current_dist_to_goal)
prev_dist_to_goal = current_dist_to_goal
```

### 4b. Reward scale explosion

**Symptom:** One component of the reward dwarfs all the others. The agent optimizes only the dominant term.

**Diagnosis:** Print the magnitudes of each reward component separately for 1000 steps:

```gdscript
print({"goal": r_goal, "progress": r_progress, "speed": r_speed})
```

If `r_speed` reaches `100.0` per step while terminal reward is `+1.0`, the terminal signal is invisible to the optimizer.

**Fix:** Divide every per-step reward component by its expected maximum so that **all per-step rewards stay below ~0.1**. The terminal reward should dominate.

### 4c. Wrong algorithm for the action space

**Symptom:** Loss looks bizarre; agent never improves; training never starts.

**Diagnosis:** DQN only supports discrete actions. PPO and SAC support continuous (and PPO also supports discrete).

**Fix:**

- Continuous action space → **PPO** or **SAC**
- Discrete action space → **DQN** or **PPO**

### 4d. Learning rate too high

**Symptom:** Reward decays after rising briefly; `train/approx_kl` in TensorBoard is consistently > `0.05`.

**Fix:** Halve the learning rate:

```bash
--learning_rate=1e-4   # was 3e-4
```

## 5 · Policy converged too early (plateau)

**Symptom:** `ep_rew_mean` rises quickly to some value, then stops improving. Godot shows the agent doing *something* reasonable but not solving the full task.

### 5a. Entropy collapsed

**Symptom:** `train/entropy_loss` in TensorBoard suddenly drops near zero. Exploration died and the policy locked onto its first decent behavior.

**Fix:** Raise the entropy coefficient to keep the policy stochastic for longer:

```bash
--ent_coef=0.01    # try 0.01 to 0.05
```

### 5b. Local optimum in reward landscape

**Symptom:** The agent has found a shortcut behavior that scores well but isn't the full task.

**Examples:**

- The agent stays in the safe spawn zone, collecting survival bonuses, instead of seeking the goal.
- The agent spins in place to maximize a velocity-based reward without going anywhere.

**Fix:** Reduce the weight of whatever auxiliary reward is being gamed. Make sure the **terminal goal reward dominates** the cumulative auxiliary rewards over a typical episode. Watch the Godot visualization to identify the shortcut behavior precisely.

### 5c. n_steps too short

**Symptom:** Short rollouts mean the agent only ever sees the first few steps of an episode, and can't learn long-horizon credit assignment.

**Fix:** Raise `--n_steps`:

```bash
--n_steps=2048   # was 64 or 128
```

### 5d. Not enough training

**Symptom:** `ep_rew_mean` was still climbing when you stopped.

**Fix:** Double the timesteps. Always inspect the **slope** of the reward curve before deciding training is done. A flat curve for 200k+ steps is plateau; a curve still bending upward is not.

## 6 · Training instability (oscillating curves)

**Symptom:** `ep_rew_mean` oscillates wildly, or rises and then collapses to zero or negative.

!!! warning "NaN losses are emergencies"
    If you see `nan` in the training log even once, **stop training immediately**. NaN means your model weights are corrupted. Resuming from a NaN checkpoint will not recover. Go to 6c.

### 6a. approx_kl too large

**Symptom:** `train/approx_kl` consistently > `0.05`. The policy is changing too fast between updates and overshooting.

**Fix:** Reduce one of:

```bash
--clip_range=0.1        # was 0.2
--learning_rate=1e-4    # was 3e-4
```

### 6b. Value function diverging

**Symptom:** `train/value_loss` grows over time instead of falling.

**Fix:** Lower the value-function coefficient and recheck reward scale (Section 4b):

```bash
--vf_coef=0.25   # was 0.5
```

### 6c. Gradient explosion

**Symptom:** Training log shows `NaN` loss, or `inf` gradients.

**Fix:** Cap the gradient norm and check that no observation value is huge:

```python
model = PPO("MlpPolicy", env, max_grad_norm=0.5, ...)
```

Then verify no observation has magnitude greater than ~10 (Section 6d).

### 6d. Observation normalization issue

**Symptom:** An obs value routinely exceeds `±10`, saturating activations and producing gradients that look fine until they suddenly explode.

**Diagnosis:** Print the max absolute observation across 1000 steps:

```gdscript
var max_abs = 0.0
for v in get_obs()["obs"]:
    max_abs = max(max_abs, abs(v))
print("max_abs_obs:", max_abs)
```

**Fix:** Divide each obs component by the maximum value it can physically take (see the normalization section of Unit 6). Aim for every observation to lie in `[-1, 1]` or `[0, 1]`.

### 6e. Catastrophic forgetting (DQN only)

**Symptom:** The agent learns a behavior, then suddenly forgets it. Reward drops to baseline.

**Fix:**

- Increase the replay buffer size.
- Decrease `learning_starts` so warm-up doesn't dominate.
- Add **prioritized experience replay** (PER) so important transitions are revisited.

## 7 · TensorBoard looks good but Godot looks wrong

**Symptom:** `ep_rew_mean` is high but in Godot the agent does something visibly weird or unintended.

!!! warning "Reward hacking is the most dangerous failure mode"
    The optimizer is doing exactly what you asked. If the agent looks wrong while the reward looks right, **your reward function is wrong** — not the agent. Fix the reward, then retrain. Never train against a reward function you have not watched the agent exploit for at least 10 episodes.

### 7a. Reward hacking

**Symptom:** The agent has found a high-reward behavior that does not match the intended goal.

**Examples:**

- Spinning in place to accumulate a misnamed "progress" reward.
- Standing still to avoid a movement penalty without ever attempting the task.
- Bumping the goal repeatedly if the terminal reward fires without `done = true`.

**Fix:** Watch 10 full episodes in Godot. **Describe in one sentence what the agent is actually doing.** Then redesign whichever reward component is being gamed. Often the fix is to make the terminal reward larger and the per-step bonuses smaller.

### 7b. Stochastic vs deterministic gap

**Symptom:** Training uses a stochastic policy (so exploration works); evaluation may use stochastic *or* deterministic, and the two look different.

**Fix:** When watching the trained model, always run it deterministically and confirm with the `--viz` flag:

```python
action, _ = model.predict(obs, deterministic=True)
```

### 7c. Different scene between training and eval

**Symptom:** The exported binary has different spawn positions, object placements, or physics parameters than the editor scene used for training.

**Fix:** Re-export the binary from the **same commit** that produced the training run. Verify spawn points, physics tick rate, and any randomization seeds match exactly.

## 8 · ONNX inference issues

**Symptom:** You loaded an ONNX model in Godot, but the agent does nothing or moves randomly.

### 8a. Wrong Sync node settings

**Symptom:** Sync node is still in training mode, so it tries to talk to a Python server that isn't there.

**Fix:** On the Sync node in the running scene:

- `Control Mode` must be `ONNX_INFERENCE`
- `Onnx Model Path` must point to the actual `.onnx` file on disk
- Inspect these in the editor *while the scene is running*, not while editing

### 8b. Observation shape mismatch

**Symptom:** The ONNX model expects a specific input shape but `get_obs()` returns a different size now.

**Fix:** Confirm `get_obs()` returns the exact same number of values as during training. **Re-export the ONNX model from the same codebase that trained it.** Any change to obs since training invalidates the model.

### 8c. ONNX exported from the wrong checkpoint

**Symptom:** Behavior in inference looks like random or early-training behavior.

**Fix:** Export from `best_model.zip` (saved by the SB3 `EvalCallback`), not from the final model — the final model may be a post-collapse checkpoint.

### 8d. Action scaling mismatch

**Symptom:** The ONNX policy outputs raw values; `set_action()` scales them. If `set_action()` changed after training, the scale is wrong now.

**Fix:** Keep `set_action()` **byte-identical** between training and inference. If you must change scaling, retrain.

## 9 · Performance (slow training)

**Symptom:** `steps/sec` is much lower than expected; training takes hours when it should take minutes.

### 9a. Running with visualization enabled

**Symptom:** Using `--viz`, or launching from the Godot editor, slows training 10–50×.

**Fix:** For any training run over 100k steps, use the **exported binary** without `--viz`. Only use `--viz` to inspect a *trained* model.

### 9b. n_parallel too low

**Symptom:** Throughput stays low even on a fast machine.

**Fix:** Run multiple environments in parallel. On most hardware, 8–16 parallel envs is the sweet spot.

```bash
--n_parallel=8
```

And add 8 corresponding in-scene instances so each parallel worker has its own.

### 9c. Physics fps too high

**Symptom:** Godot physics is set to 120 fps when 60 is sufficient. Each step costs twice as much compute.

**Fix:** Project → Project Settings → Physics → Common → **Physics Ticks Per Second = 60**.

### 9d. speedup not set

**Symptom:** Missing `--speedup=20` means Godot runs in real time at 1× — wall-clock training is enormously slow.

**Fix:** Always set `--speedup=20` (or higher for simple envs that don't need precise physics):

```bash
--speedup=20
```

### 9e. GDScript bottleneck

**Symptom:** Even with all of the above, throughput is low and the CPU is pegged on the Godot side.

**Diagnosis:** Add timestamps inside `get_obs()` and the reward function. Look for `for` loops or per-step allocations.

**Fix:** Cache anything that doesn't change every frame (node references, references to materials, etc.). Avoid string concatenation and dictionary allocations inside the hot path.

## 10 · Quick diagnostic commands

!!! tip "Always run the random baseline first"
    Before debugging "the agent isn't learning", confirm what a **random policy** scores. If random scores 0.5 and your agent scores 0.5, the agent is doing nothing. If random scores 0.5 and your agent scores 0.55, training *is* happening — just slowly. This single number saves hours of misdirected debugging.

```bash
# Sanity-check training is producing reward at all
gdrl --env_path=./MyEnv.x86_64 --timesteps=50000 --n_parallel=1 --speedup=20
```

```python
# Random-policy baseline — what does no learning at all look like?
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import numpy as np

env = StableBaselinesGodotEnv(env_path='./MyEnv.x86_64', n_parallel=1, speedup=1)
rewards = []
for _ in range(20):
    obs, total = env.reset(), 0
    done = False
    while not done:
        obs, r, done, _ = env.step(env.action_space.sample())
        total += r
    rewards.append(total)
print(f'Random baseline: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}')
env.close()
```

```bash
# Watch 5 episodes of a trained model, deterministically
gdrl --env_path=./MyEnv.x86_64 \
  --resume_model_path=logs/sb3/myenv/best_model.zip \
  --inference --viz --timesteps=5000
```

## 11 · TensorBoard metrics reference

| Metric | Location | Healthy | Action if unhealthy |
|--------|----------|---------|---------------------|
| `rollout/ep_rew_mean` | Rollout | Rising | See Section 3–5 |
| `rollout/ep_len_mean` | Rollout | Increasing | Too short → see 3e |
| `train/approx_kl` | Train | 0.01–0.02 | >0.05 → lower lr/clip |
| `train/entropy_loss` | Train | Slowly falling | Sudden drop → raise ent_coef |
| `train/value_loss` | Train | Falling | Growing → lower vf_coef |
| `train/policy_gradient_loss` | Train | Near 0 | Exploding → lower lr |
| `train/explained_variance` | Train | Approaching 1.0 | Near 0 → critic not learning |
| `time/fps` | Time | >1000 | Low → see Section 9 |

## 12 · Stretch Goals

**Break a run on purpose.** Pick one failure mode from Sections 3–6 — say, "reward going down" — and engineer a training run that exhibits it. Easiest recipe: multiply your reward by 100 and lower the entropy coefficient. Then walk yourself through the diagnostic flowchart in Section 2 *as if you didn't already know what you did*. The goal is to feel the diagnosis steps in your fingers before the real bug shows up at 11 pm.

**Write your own diagnostic entry.** Find a bug you actually hit this course (any unit, your own training run). Write a new Section in the same `symptom → diagnosis → fix` format and submit it as a PR against this page. The course wants more first-hand bug entries, not fewer. Reference issue #48 if you're not sure where to drop it.

**Automate one alert.** Pick one metric from Section 11 and write a 20-line Python script that tails the TensorBoard `events.out.*` file (or polls `tensorboard --logdir`'s data) and prints a warning when the metric leaves the healthy band — e.g. `explained_variance < 0.1` for 50k steps. The point isn't a production monitor; it's to realise that "watch TensorBoard" can be partially automated.

!!! warning "Pseudocode"
    ```python
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator("logs/sb3/myenv/PPO_1")
    ea.Reload()
    values = [s.value for s in ea.Scalars("train/explained_variance")]
    if values and values[-1] < 0.1:
        print(f"ALERT: explained_variance={values[-1]:.3f} — critic may not be learning")
    ```
