# Hardware Setup & Training Time

[Course home](index.md)

---

Choose your hardware wisely. This guide helps you understand what you need to train each unit, how long it takes, and what to do if your laptop isn't enough.

---

## 1 · Per-unit compute requirements

This table shows CPU adequacy, GPU helpfulness, and realistic training times for each unit.

| Unit | Task | CPU adequate? | GPU helpful? | Estimated time (M1 MacBook / RTX 3060 / RTX 4090) |
|------|------|---------------|--------------|---------------------------------------------------|
| Unit 0–1 | CartPole, Acrobot setup | ✅ Yes | ❌ No | 30 sec / 15 sec / 10 sec |
| Unit 2 | LunarLander (discrete) | ✅ Yes | ❌ No | 3–5 min / 1–2 min / 30 sec |
| Unit 3 | CrossTheRoad (DQN, visual) | ⚠️ Slow | ✅ Strongly | 45–90 min / 8–12 min / 2–3 min |
| Unit 4 | JumperHard (PPO, continuous) | ⚠️ Slow | ✅ Strongly | 60–120 min / 10–15 min / 3–5 min |
| Unit 6 | FlyBy (continuous visual obs) | ⚠️ Slow | ✅ Strongly | 90–150 min / 12–20 min / 4–6 min |
| Visual observations (any) | CNN policies + pixel input | ❌ Not practical | ✅ Required | 2–3 hrs / 15–25 min / 5–8 min |
| Capstone project | 2M steps, custom task | ❌ Not practical | ✅ Required | 3–5 hrs / 30–45 min / 10–15 min |

!!! info "Time assumptions"
    Times assume 1M steps unless the unit specifies otherwise. Actual times depend on:
    - Your CPU clock speed, core count, and RAM bandwidth
    - GPU VRAM size (larger buffer sizes need more VRAM)
    - Number of parallel environments (8–16 typical for training)
    - Hyperparameter choices (learning rate, batch size, entropy regularization)

---

## 2 · When CPU is enough vs when you need a GPU

**Quick rule of thumb:**

| Scenario | CPU OK? | Recommendation |
|----------|---------|-----------------|
| MLP policies + low-dim obs (raycasts, ≤16 dim) + ≤16 parallel envs | ✅ Yes | Any modern CPU (2018+) with 16 GB RAM |
| CNN policies (pixel observations) | ❌ No | GPU required — CPU CNN training is 10–50× slower |
| Replay buffer > 500k steps + pixel observations | ❌ No | GPU required (memory bandwidth + model forward passes) |
| \>32 parallel environments | ⚠️ Risky | Use GPU or a beefy multi-core CPU (16+ cores) |
| SAC or other continuous algo + visual obs | ❌ No | GPU strongly recommended (double update frequency) |

!!! warning "CPU is fine for Units 0–4"
    If you skip visual observations (use raycasts instead), a modern CPU trains Units 0–4 in acceptable time. Unit 3 (CrossTheRoad) *can* work with raycasts; the course includes both versions.

---

## 3 · Recommended setups by budget

| Tier | Hardware | Cost | Notes |
|------|----------|------|-------|
| **Free** | Google Colab (T4 GPU, 12 hrs/session) | $0 | Good for Units 1–6, capstone. Sign in, upload notebook, go. 12-hr session limit requires checkpointing. |
| **Hobbyist** | Used RTX 3060 12 GB (~$250) or AMD RX 6700 XT 12 GB (~$280) | ~$250–300 | Entry-level NVIDIA/AMD. Trains Units 0–6 in 30–60 min. Buy used from eBay/Facebook Marketplace. Requires PCIe slot + power supply. |
| **Student** | RTX 4060 Ti 8 GB (new, ~$450) or RTX 4070 (used, ~$400) | ~$400–500 | 8 GB is tight for pixel obs + large buffer; 12–16 GB ideal. 4070 recommended for future projects. |
| **Serious** | RTX 4070 Ti / 4080 / 4090 desktop | $800–2500 | Trains capstone in <15 min. Future-proof for larger models and longer training. |
| **Research** | Multi-GPU workstation (2× RTX 4090, or A100s via cloud) | $5k+ local or $/hr cloud | Distributed training, bigger experiments, research-scale rollouts. |

!!! tip "Best value for students"
    A used RTX 3060 or 4070 has the best cost-per-training-hour for the course. Used GPUs from eBay/Marketplace are often 50% cheaper than new.

---

## 4 · Apple Silicon — what works and what doesn't

**Apple M1/M2/M3 situation is mixed.** Stable-Baselines3 + PyTorch *do* run on Apple Silicon, but some features are incomplete.

| Feature | Status | Workaround |
|---------|--------|-----------|
| MLP policies (Units 0–4 without visual obs) | ✅ Works well | Default. Fast inference, training ~50% slower than RTX 3060. |
| CPU training explicitly | ✅ Works | `model = PPO("MlpPolicy", env, device="cpu")` (even on GPU-equipped machines) |
| PyTorch MPS (Metal Performance Shaders) | ⚠️ Incomplete | Some SAC + CNN ops fail on float64. Mixed results. |
| CNN training (pixel observations) | ❌ Slow | 5–10× slower than RTX 3060. Works, but not practical for Units 3, 6, Capstone. |
| Large replay buffers (pixel obs) | ❌ Tight | M1 8GB/16GB unified memory bottlenecks on replay buffer I/O. M2/M3 Pro/Max with 32GB+ is better. |

!!! warning "Don't rely on MPS auto-detection"
    PyTorch's automatic MPS fallback is brittle. Explicitly force CPU:
    ```python
    model = PPO("MlpPolicy", env, device="cpu")
    ```
    Training is slower but stable. For visual obs, consider Google Colab instead.

!!! tip "M2/M3 Pro/Max with 32 GB unified memory"
    If you have an M3 Max with 32 GB, you can train visual-obs units, but still ~3–5× slower than RTX 3060. Best for learning; not for research.

---

## 5 · Cloud options (when local isn't enough)

If your laptop can't handle the capstone or you want to parallelize across GPUs, cloud providers offer GPUs by the hour.

| Provider | T4 $/hr | A100 $/hr | Setup ease | Best for | Notes |
|----------|---------|-----------|-----------|----------|-------|
| **Vast.ai** | $0.10–0.20 | $1.50–2.50 | Medium | Cost-sensitive students | Spot instances, huge variety of hardware. Requires docker knowledge. |
| **Lambda Labs** | $0.50 | $2.50 | Easy | Hands-off training | Pre-installed PyTorch, Jupyter. Clean Linux. Mid-priced but reliable. |
| **Google Colab Pro** | Included (T4/A100) | Included (rare) | Easy | Quick experiments | $12/month, 100 compute units/month. Great UX, limited hours. |
| **RunPod** | $0.18–0.35 | $2.00–3.00 | Medium | Europe-based students | Good uptime, competitive pricing, good support. |
| **AWS SageMaker** | $0.35 | $4.50 | Hard | Enterprise only | Most expensive but enterprise SLAs. Not recommended for students. |
| **GCP AI Platform** | $0.35 | $4.00 | Hard | Enterprise only | Similar to AWS; enterprise pricing. |

!!! info "Estimated capstone training cost"
    Training the capstone (2M steps, ~1.5–2 hrs on RTX 3060):
    - Vast.ai: $0.30–0.40
    - Lambda Labs: $1.00–1.25
    - Colab Pro: Included (but session might time out)
    - RunPod: $0.30–0.70
    - AWS: $0.70–0.90

!!! tip "Use spot instances to save 50%"
    Vast.ai and AWS offer spot pricing. Training is fault-tolerant if you checkpoint every 100k steps (SB3 has built-in checkpointing).

---

## 6 · Disk and RAM

**Minimum system specs to complete the course:**

| Resource | Minimum | Recommended | Why |
|----------|---------|-------------|-----|
| **RAM** | 16 GB | 32 GB | Replay buffers for pixel observations eat RAM. Units 0–2 use ~2–4 GB; Units 3+ with large buffer = 12–20 GB. |
| **SSD storage** | 10 GB | 50 GB | Python env (~3 GB) + SB3 + Godot (~2 GB). If you keep all TensorBoard logs and model checkpoints for every unit: another 20–50 GB. |
| **Disk type** | HDD (slow) | SSD (fast) | Replay buffer I/O is frequent. SSD reduces training time by ~20–30% for pixel obs. |

!!! warning "Disk space for capstone"
    If you train 50 capstone variants with checkpoints, you'll easily use 50 GB. Clean up old logs monthly: `rm -rf logs/unit-0*`.

---

## 7 · Quick hardware check

Run this Python snippet to see what GPU (if any) is available on your machine:

```python
import torch
import psutil

print("=" * 50)
print("PyTorch GPU Check")
print("=" * 50)

# CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / 1e9
        print(f"GPU {i}: {props.name}")
        print(f"  VRAM: {total_mem:.1f} GB")

# MPS (Apple Silicon)
if torch.backends.mps.is_available():
    print(f"MPS available: True")
    print("  (Metal Performance Shaders - Apple Silicon)")
else:
    print(f"MPS available: False")

# CPU info
print(f"\nCPU cores: {psutil.cpu_count(logical=False)}")
print(f"CPU (with HT): {psutil.cpu_count(logical=True)}")

# RAM
mem = psutil.virtual_memory()
print(f"RAM: {mem.total / 1e9:.1f} GB total")
print(f"     {mem.available / 1e9:.1f} GB available")

print("=" * 50)
```

Save this as `check_hardware.py`, then run:

```bash
python check_hardware.py
```

!!! info "What to look for"
    - **CUDA available: True** → You have an NVIDIA GPU. Check VRAM—you need ≥8 GB for visual obs.
    - **CUDA available: False** + MPS available: True → You have Apple Silicon. Use CPU for visual tasks.
    - **All False** → You have CPU only. Great for Units 0–4; consider Google Colab for Units 3+.

---

## Next steps

- **Units 0–4?** Your current machine is fine (even CPU-only).
- **Units 3–6 with visual obs?** Upgrade to a GPU or use Google Colab.
- **Capstone project?** Use a GPU (RTX 3060+) or cloud provider to finish in <1 hour.

[← Back to course home](index.md)
