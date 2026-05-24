# Course Gap Analysis (2026-05-24)

Identified after completing the HF Deep RL parity pass. All items below are tracked for integration.

## Status

| Gap | Priority | File | Status |
|-----|----------|------|--------|
| SAC (Soft Actor-Critic) | High | `content/unit-sac.md` | Done ✅ |
| Reward engineering | High | `content/unit-reward-engineering.md` | Done ✅ |
| Debugging RL training | High | `content/unit-debugging.md` | Done ✅ |
| Intrinsic motivation / curiosity | Medium | `content/unit-curiosity.md` | Done ✅ |
| Off-policy vs on-policy deep treatment | Medium | expand `unit-03.md` | Done ✅ |
| Deadly triad callout | Medium | expand `unit-03.md` | Done ✅ |
| Model-based RL overview | Medium | expand `unit-q-learning.md` or `unit-01.md` | Done ✅ |
| Multi-seed evaluation | Medium | expand `unit-05.md` | Done ✅ |
| Hyperparameter search (Optuna) | Low | stretch goal in `unit-04.md` | Done ✅ |

## Detail notes

### SAC
- Off-policy, maximum entropy RL framework
- Uses replay buffer like DQN but outputs a policy distribution
- Dominant for continuous control (robotics, simulation)
- Key concepts: entropy temperature α, reparameterization trick, twin critics (prevents overestimation)
- When to use: sample efficiency matters, continuous action space, stable training needed

### Reward Engineering
- Most important practical skill in RL, currently scattered
- Topics: potential-based shaping (theory), dense vs sparse tradeoffs, reward hacking examples, scale/sign gotchas
- Godot connection: how to design reward functions in `_physics_process`, shaped components, terminal rewards

### Debugging RL
- Diagnostic flowchart: no learning → wrong reward → obs issues → algorithm mismatch → hyperparams
- Common failure modes with symptoms and fixes
- TensorBoard-driven diagnosis

### Intrinsic Motivation / Curiosity
- HF Unit 5 equivalent (Snowball, Pyramid with RND)
- Why: sparse rewards stall standard RL
- RND (Random Network Distillation): train a random target network; measure prediction error as novelty
- sb3-contrib CuriosityWrapper

### Off-policy vs on-policy
- Core distinction missing from DQN unit
- On-policy (PPO): data must come from current policy, discarded after update
- Off-policy (DQN, SAC): replay buffer reuses old data
- Explains sample efficiency difference

### Deadly triad
- Function approximation + bootstrapping + off-policy = potential divergence
- Why DQN needs target network + replay buffer to avoid it

### Model-based RL overview
- Conceptual only: learn dynamics model p(s'|s,a), plan inside it
- Dyna, MuZero, Dreamer as examples
- Much more sample-efficient; harder to train accurate world model

### Multi-seed evaluation
- Single training run ≠ reliable result
- Run N seeds (3–5), report mean ± std across seeds
- Statistical significance for comparing algorithms
