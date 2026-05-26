# Foundation Models for Control

Large language models turned out to be surprisingly good at game benchmarks. Now researchers are asking a harder question: can a single **foundation model** — trained on internet-scale data — learn to control robots and embodied agents directly, without per-task reward engineering? RT-2, Octo, OpenVLA, and π0 are four serious answers to that question. This unit is a **literacy and signposts unit**: no hands-on coding, but by the end you will understand what each model does, why they are hard to plug into Godot today, and where to look when the ecosystem matures.

[← World Models / DreamerV3](unit-world-models.md) · [Course home](index.md)

---

!!! info "Three ways to see your AI"
    Read the paper (RT-2's chain-of-thought reasoning output — the model literally prints "move left to grasp") · browse the HuggingFace model card (Octo and OpenVLA have detailed cards with evaluation videos) · watch a π0 demo video (Physical Intelligence's website has slow-motion clips of the robot folding laundry — study what the hands do between keyframes)

---

## 1 · What is a foundation model for control?

The idea is borrowed from NLP: pre-train once on a massive, diverse dataset, then adapt cheaply to new tasks. In language, a single GPT-style model can write code, summarize contracts, and pass the bar exam — all from one set of weights.

**Foundation models for control attempt the same bet for embodied agents:**

```
Foundation model (language / vision):
  text: "put the apple in the bowl"
  image: camera frame from the robot
  →  joint torques / gripper command / Cartesian waypoint

Goal: one model, many tasks, minimal task-specific tuning.
```

Why is this hard? Language models see text — a flat sequence of tokens. A robot arm sees camera images and proprioceptive state, and must output *continuous* actions at 10–50 Hz with sub-centimetre precision. Bridging that gap is the core engineering challenge every model below attacks differently.

---

## 2 · The four models

### 2.1 · RT-2 — Robotic Transformer 2 (Google DeepMind, 2023)

**What it is.** RT-2 fine-tunes a vision-language model (PaLI-X or PaLM-E at 55B parameters) on a mixture of web data and robot demonstrations. Actions are **tokenized as text**: the continuous joint angles are discretized into 256 bins and emitted as ordinary text tokens alongside words.

| Property | Detail |
|----------|--------|
| Base model | PaLI-X 55B (Vision-Language) |
| Training data | 130k robot demonstrations + web-scale text/image data |
| Action space | 7-DOF arm (6 joints + gripper), tokenized as text |
| Key capability | Zero-shot generalization to novel objects and instructions |
| Architectural choice | Represent actions as language tokens — reuse the LLM's next-token machinery |

**Chain-of-thought reasoning.** RT-2 can emit a reasoning trace before outputting the action: `"The instruction says pick up the empty cup. I see two cups. The one on the left has liquid. I should pick the right one. Action: [move_right, close_gripper]"`. This is emergent from the language pre-training, not separately engineered.

**Repo / paper.** No public weights (Google internal), but the paper is at [arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818).

---

### 2.2 · Octo — An Open-Source Generalist Robot Policy (2024)

**What it is.** Octo is a **fully open-source** transformer policy trained on the Open X-Embodiment dataset — 800k+ demonstrations from 22 different robot platforms. The goal is to be the BERT of robot learning: a pre-trained backbone you fine-tune rather than train from scratch.

| Property | Detail |
|----------|--------|
| Base model | 90M parameter transformer (no LLM pre-training) |
| Training data | Open X-Embodiment — 800k+ demonstrations, 22 robot types |
| Action space | Continuous 7-DOF arm + gripper, diffusion head or Gaussian |
| Key capability | Fine-tune to a new robot with ~100 demonstrations |
| Architectural choice | Modular tokenizer: swap in language goal, image goal, or both |

**Why Octo matters.** It is the first large-scale open checkpoint trained on a heterogeneous multi-robot dataset. The architecture separates *what you want* (task token: language or image goal) from *how you move* (action head), so the task token can be swapped without retraining the backbone.

**Repo.** [github.com/octo-models/octo](https://github.com/octo-models/octo) — weights on HuggingFace at `hf.co/rail-berkeley/octo-small` and `hf.co/rail-berkeley/octo-base`.

---

### 2.3 · OpenVLA — Open Vision-Language-Action Model (2024)

**What it is.** OpenVLA fine-tunes **Prismatic-7B** (a 7B LLM with a vision backbone) on the same Open X-Embodiment dataset as Octo. Like RT-2, actions are tokenized as text. Unlike RT-2, the weights are public.

| Property | Detail |
|----------|--------|
| Base model | Prismatic-7B (LLaVA-style VLM) |
| Training data | Open X-Embodiment — 970k demonstrations |
| Action space | 7-DOF absolute joint positions, 256-bin tokenization |
| Key capability | Instruction-following across diverse robot platforms |
| Architectural choice | Extend the LLM vocabulary with action tokens — no separate action head |

**Efficiency work.** Later work (OpenVLA-OFT, 2024) adds parameter-efficient fine-tuning (LoRA + parallel decoding) that gets inference down from ~6 tokens/second to real-time on a single A100. Still not embedded-system territory.

**Repo / weights.** [github.com/openvla/openvla](https://github.com/openvla/openvla) — model at `hf.co/openvla/openvla-7b`.

---

### 2.4 · π0 — Physical Intelligence's Foundation Model (2024)

**What it is.** π0 ("pi-zero") from Physical Intelligence trains a **flow-matching diffusion action head** on top of a PaliGemma vision-language backbone. Rather than discretizing actions into tokens, it learns a continuous action distribution via a diffusion process, which produces smoother trajectories than per-bin tokenization.

| Property | Detail |
|----------|--------|
| Base model | PaliGemma 3B VLM (Google) |
| Training data | Proprietary fleet of robot demonstrations (dexterous manipulation) |
| Action space | Continuous — wrist, finger, base velocities at 50 Hz |
| Key capability | Dexterous tasks: folding laundry, bussing tables, box assembly |
| Architectural choice | Flow-matching diffusion head — continuous action distribution, not tokens |

**Why diffusion for actions?** Tokenizing continuous actions introduces quantization error and forces the model to output one token at a time (slow). A diffusion head learns a full action distribution in one forward pass and can represent multi-modal behaviour (e.g., "move left *or* right are both valid" early in the trajectory).

**Repo / announcement.** No public weights yet. Blog post and demo videos at [physicalintelligence.company/blog/pi0](https://physicalintelligence.company/blog/pi0). Paper: [arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164).

---

## 3 · Why they are hard to plug into Godot today

You might be tempted to drop Octo or OpenVLA into your Godot environment and run it as a policy. Three fundamental mismatches make this non-trivial right now.

### 3.1 · Observation-modality mismatch

Every model above was trained on **real robot camera frames**: slightly blurry, motion-smeared 480×640 RGB images with the subtle colour casts of fluorescent lab lighting. Godot renders **synthetic images**: crisp, aliased, over-saturated, with perfect lighting.

The domain gap between Godot renders and real RGB images is larger than between sim and real physics. The visual backbone of these models has learned features from real images; Godot images will produce activations that fall outside the training distribution. Performance degrades — sometimes catastrophically.

Workarounds people have tried:

- Apply style-transfer to Godot renders (slow, lossy)
- Use depth maps or segmentation masks instead of RGB (modality mismatch but smaller domain gap)
- Fine-tune on Godot-rendered data (requires a GPU cluster and thousands of Godot demonstrations)

### 3.2 · Action-space mismatch

RT-2, Octo, and OpenVLA produce actions for **7-DOF robot arms** with a specific joint ordering, scaling, and physical units (radians or metres at specific frequencies). Godot environments usually have a completely different action space — a `ContinuousAction` vector with game-specific semantics.

Remapping is theoretically possible but not automatic: you need to understand what each action dimension means in the source robot, what it means in your Godot character, and whether the dynamics (inertia, damping, gear ratios) are even comparable.

### 3.3 · No lightweight Godot-friendly checkpoints

The smallest public checkpoint is **Octo-Small** at 27M parameters — fast to run but still designed for CUDA inference with specific robot normalization statistics. There is no `pip install octo && env.step(octo.act(obs))` that just works with a Godot env. You would need to:

1. Write a custom observation adapter (Godot → robot obs format)
2. Override Octo's normalization statistics for your action space
3. Host the model inference server (Python) and call it from the Godot training loop
4. Accept an inference latency of ~50ms per step — which caps your environment to ~20 Hz max

For most Godot RL tasks, a custom PPO or SAC policy trained from scratch converges faster and runs at 200 Hz without infrastructure overhead.

---

## 4 · The Open X-Embodiment dataset

Many of the models above converge on the same training corpus. It is worth knowing what it is.

**Open X-Embodiment** (Collaboration of 33 institutions, Google DeepMind, 2023) aggregated 22 publicly released robot datasets into a single standardized format — 1M+ demonstrations across 22 robot morphologies (arms, mobile manipulators, bi-manual systems, humanoids), 527 skills, and multiple labs.

**Key insight.** Before Open X-E, most robot learning papers trained on 200–5,000 demonstrations from a single lab's robot. Open X-E was the first dataset large enough that cross-embodiment pre-training was measurably better than single-embodiment training from scratch.

**Access.** [robotics-transformer-x.github.io](https://robotics-transformer-x.github.io) — data in RLDS format (TensorFlow Datasets), most subsets require an academic Google account.

---

## 5 · Conceptual map — where these models sit in the RL taxonomy

Foundation models for control are often described as "not RL" — they use behaviour cloning from demonstrations, not reward-driven policy search. That is partially true and partially misleading.

```
Pure BC:    demonstrations → policy  (no reward, no exploration)
Pure RL:    reward function → policy  (no demonstrations, full exploration)

These models:
  Octo/OpenVLA:  demonstrations → pre-trained backbone → fine-tune with BC
  π0:            demonstrations → pre-trained backbone → optionally fine-tune with RL (π0-FAST variant)
  RT-2:          web data + demonstrations → BC → deploy (no RL fine-tuning in original paper)
```

**Why not just RL?** For dexterous manipulation, reward design is extremely hard — how do you write a reward for "fold the shirt neatly"? Imitation learning from human demonstrations sidesteps reward engineering. The tradeoff is that BC-trained models cannot easily discover behaviours not present in the demonstrations.

**Where RL re-enters.** Recent work (π0-FAST, 2024; RT-X with RL fine-tuning) uses RL to fine-tune the foundation model after BC pre-training — similar to RLHF in language. This combines the coverage of human demonstrations with RL's ability to improve beyond the demonstrator.

---

## 6 · Stretch goals

### 6.1 · Load a small VLM as a feature extractor for a Godot policy

One way to get *some* value from foundation model pre-training without running a full 7B model at inference time is to use the visual backbone as a **frozen feature extractor** and train a small RL head on top. The idea:

1. Take the vision encoder from a small open VLM — for example, **SigLIP** (a 400M vision-language encoder from Google, available at `hf.co/google/siglip-base-patch16-224`) or the vision tower from **LLaVA-1.5-7B** (`hf.co/liuhaotian/llava-v1.5-7b`, vision part only).
2. Run the Godot SubViewport pipeline to capture 224×224 RGB frames (see the [Visual Observations unit](unit-visual-observations.md)).
3. Send each frame through the frozen encoder to get a 768-dim or 1152-dim embedding vector.
4. Use that embedding as the observation for a standard PPO or SAC policy. The RL head is small — two hidden layers of 256 is enough.
5. The frozen encoder provides semantic features (object identity, spatial layout) that a CNN trained from scratch would need millions of steps to learn.

**What to expect.** This technique (called "frozen visual pre-training" or "VC-1" in the literature — see [arxiv.org/abs/2303.04137](https://arxiv.org/abs/2303.04137)) can improve sample efficiency on tasks that require semantic understanding of the scene. For pure locomotion it often does not help — the task is geometric, not semantic.

**Infrastructure note.** Running SigLIP-base at inference time costs ~20ms on a CPU. That caps your Godot env to ~50 Hz — acceptable if `n_parallel` is low. On a GPU it drops to ~2ms, so 200 Hz is achievable. Use ONNX export (`optimum-cli export onnx`) to get a runtime-efficient version of the encoder.

### 6.2 · Language-conditioned goals without a 7B model

You do not need RT-2 to get language-conditioned behaviour in Godot. A lighter approach:

1. Embed the text instruction with a frozen **sentence-transformer** (e.g. `all-MiniLM-L6-v2` — 22M params, 384-dim output, CPU-fast). Install with `pip install sentence-transformers`.
2. Concatenate the 384-dim sentence embedding with your existing observation vector.
3. Train PPO normally — the policy learns to condition on the language embedding.
4. At test time, swap the instruction text and the policy generalizes (to the extent that training instructions covered the relevant distribution).

This is the approach used in **CLIP-Fields** and **SayCan (simplified)**. It costs ~5ms per instruction encode on CPU (not per step — you encode once per episode) and adds only 384 dims to your obs space.

### 6.3 · Read the Octo fine-tuning tutorial

The Octo repository has a Colab notebook ([github.com/octo-models/octo/blob/main/examples/02_finetune_new_observation_action.ipynb](https://github.com/octo-models/octo/blob/main/examples/02_finetune_new_observation_action.ipynb)) that shows how to override observation and action spaces for a custom robot. Working through it — even without running it — is the clearest way to understand what an observation-adapter actually requires. Pay attention to the normalization statistics section: that is the step most people get wrong when porting to a new embodiment.

---

## What's next

You have reached the end of the Guides section. Return to the [Course home](index.md) for the full unit list, or revisit any earlier unit to go deeper.

[→ Course home](index.md)

---

[← World Models / DreamerV3](unit-world-models.md) · [Course home](index.md)
