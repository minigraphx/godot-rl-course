# Unit 5 — Unity ML-Agents

**Source:** https://huggingface.co/learn/deep-rl-course/unit5/introduction

## Learning objectives

- Understand game engines as RL environment builders (physics, rendering, 3D).
- Use Unity ML-Agents **without** mastering Unity Editor for basic training.
- Train snowball-shooting and pyramid-knockover agents.
- Use **curiosity** (intrinsic motivation) for hard exploration (pyramid task).
- Push agents to Hub; view in browser.

## Theory topics

| Topic | Student takeaway |
|-------|------------------|
| Engine-based envs | Godot / Unity / Unreal provide simulation |
| ML-Agents toolkit | Bridges Unity scenes to Python trainers |
| Extrinsic vs intrinsic reward | Curiosity when sparse extrinsic signal |
| Exploration in 3D | Navigate, interact, multi-step tasks |

## Hands-on tasks

1. **Snowball** — shoot spawning targets.
2. **Pyramid** — press button, spawn pyramid, knock over, reach gold brick (curiosity-driven exploration).

## Integration notes (Godot course)

| Godot target | What to borrow |
|--------------|----------------|
| **Unit 0–2** | Parallel narrative: “We use Godot + godot-rl-agents instead of Unity ML-Agents” |
| **Architecture doc** | Socket bridge pattern is the same idea |
| Skip | Unity install, ML-Agents package versions |

**Author note:** HF Unit 5 is the closest official analog to this course’s stack — emphasize Godot as first-class, not “lesser Unity.”

**Bonus link:** Huggy the Dog (Unit 1 bonus) also uses ML-Agents.
