# Advanced Evaluation — IQM, Performance Profiles, and Statistical Rigour

[← Parallel Training](unit-05.md) · [Course home](index.md)

!!! info "Time"
    Reading: ~25 min

---

!!! info "Three ways to see your AI"
    This unit introduces three complementary lenses for evaluating a trained policy rigorously:

    - **rliable performance profile plot** — the full distribution of scores across seeds and tasks, not just a single number
    - **IQM results table** — a robust point estimate with 95% confidence intervals that is resistant to outlier seeds
    - **Per-seed TensorBoard overlay** — visualise every individual training run so you can see variance, not just averages

    If you are publishing results or comparing two algorithms for any purpose beyond personal curiosity, all three are required.

---

## 1 · Why Mean ± Std Fails

When students first train an RL agent they open TensorBoard, look at the mean episode reward curve, and call it done. This is understandable, but it is also one of the most common sources of misleading results in the RL literature.

### The variance problem

A single training run in deep RL is not a measurement — it is one sample from a high-variance stochastic process. The outcome depends on the random seed, which controls weight initialisation, environment resets, action sampling, and (in some frameworks) the order of experience replay. Two runs with identical hyperparameters can produce wildly different final policies.

**Concrete example — PPO on FlyBy:**

Imagine you train PPO on the FlyBy environment twice. Seed 1 finds a stable hover strategy by episode 200 k and plateaus at reward ≈ 420. Seed 2 never escapes a local optimum and flatlines at reward ≈ 80 for the entire run. If you report only seed 1, your result looks excellent. If your colleague uses seed 2 as the baseline for their comparison, they will conclude their algorithm is far better than it actually is.

This is not a contrived scenario. It is the norm for sparse-reward tasks, and FlyBy has sparse rewards.

### How outliers dominate the mean

Suppose you run five seeds and get final rewards of `[410, 430, 390, 415, 85]`. The mean is 346 and the standard deviation is 142. What does "346 ± 142" tell a reader? Almost nothing useful. The 85 is an outlier from a failed run. The other four are tightly clustered around 411. The mean has been pulled 65 points below the typical outcome, and the standard deviation is inflated to the point of being misleading.

The IQM for this set is the mean of `[390, 410, 415]` (the middle 60% of a 5-sample set approximates the middle 50%) ≈ 405, which is far more representative of what the algorithm actually achieves in practice.

### Common misleading evaluation patterns

| Pattern | What it hides |
|---|---|
| Report only the best seed | Typical performance; failure rate |
| Report mean of 3 seeds | High variance; statistical noise as signal |
| Cherry-pick the evaluation window | Policies that collapsed after the snapshot |
| Compare two algorithms with one seed each | Whether the difference is real or just luck |
| Report max reward over training | That the agent rarely hits that peak |
| Use the same seed for train and eval | Overfitting to a specific random trajectory |
| Smooth TensorBoard curves heavily | Short-lived spikes that look like convergence |

!!! warning "Seed 42 is not a result"
    "I tried seed 42 and it worked" is anecdote, not evidence. The RL community has a reproducibility crisis that stems almost entirely from under-seeded experiments. A single seed run is useful for debugging. It is never sufficient for a comparative claim.

---

## 2 · Run Multiple Seeds — the Baseline Requirement

The minimum bar for any claim about algorithm performance:

- **5 seeds** — minimum for any result you share with others
- **10 seeds** — minimum for a conference or workshop paper
- **20+ seeds** — required for results on high-variance environments or small effect sizes

### Training with multiple seeds

The following bash script runs `gdrl` (the godot-rl-agents training command) for seeds 1 through 5, saving each run to a separate log directory. Adapt the environment name and hyperparameters to your project.

```bash
#!/usr/bin/env bash
# train_seeds.sh — train PPO on FlyBy with 5 seeds

ENV="FlyBy"
ALGO="ppo"
TIMESTEPS=500000

for SEED in 1 2 3 4 5; do
    echo "=== Training seed ${SEED} ==="
    gdrl train \
        --env-id "${ENV}" \
        --algo "${ALGO}" \
        --timesteps "${TIMESTEPS}" \
        --seed "${SEED}" \
        --log-dir "logs/${ENV}_${ALGO}_seed${SEED}"
done

echo "All seeds complete."
```

Run it with:

```bash
chmod +x train_seeds.sh
./train_seeds.sh
```

Each run produces a separate TensorBoard log directory. You can overlay all five in a single TensorBoard session:

```bash
tensorboard --logdir logs/
```

TensorBoard will group the runs and show individual curves for each seed alongside the smoothed aggregate.

### Loading results in Python

Once training is complete, extract the final episode reward for each seed using the `tensorboard` Python package (installed alongside Stable Baselines 3):

```python
import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_final_reward(log_dir: str, tag: str = "rollout/ep_rew_mean", last_n: int = 10) -> float:
    """Return the mean of the last `last_n` values for `tag` in `log_dir`."""
    ea = EventAccumulator(log_dir)
    ea.Reload()

    if tag not in ea.scalars.Keys():
        raise KeyError(f"Tag '{tag}' not found in {log_dir}. Available: {ea.scalars.Keys()}")

    events = ea.scalars.Items(tag)
    values = [e.value for e in events]

    if len(values) < last_n:
        return float(np.mean(values))
    return float(np.mean(values[-last_n:]))


def collect_seed_results(base_dir: str, env: str, algo: str, seeds: list[int]) -> np.ndarray:
    """Load final rewards for all seeds and return as a numpy array."""
    rewards = []
    for seed in seeds:
        log_dir = os.path.join(base_dir, f"{env}_{algo}_seed{seed}")
        reward = load_final_reward(log_dir)
        print(f"  Seed {seed}: final reward = {reward:.1f}")
        rewards.append(reward)
    return np.array(rewards)


if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 5]
    ppo_rewards = collect_seed_results("logs", "FlyBy", "ppo", seeds)
    print(f"\nPPO rewards: {ppo_rewards}")
    print(f"Mean:   {np.mean(ppo_rewards):.1f}")
    print(f"Std:    {np.std(ppo_rewards):.1f}")
    print(f"Min:    {np.min(ppo_rewards):.1f}")
    print(f"Max:    {np.max(ppo_rewards):.1f}")
```

!!! tip "Cross-reference"
    [Unit 05](unit-05.md) introduced multi-seed evaluation briefly in the context of parallel training. This unit formalises the statistical reasoning behind it and adds the tooling to make it rigorous.

---

## 3 · Interquartile Mean (IQM)

### What it is

The **Interquartile Mean (IQM)** is the arithmetic mean of the middle 50% of a sorted sample. You sort your scores from lowest to highest, discard the bottom 25% and the top 25%, and take the mean of what remains.

This is not the same as the median (which takes only the single middle value). IQM uses all the middle scores, giving it lower variance than the median while being far more robust to outliers than the full mean.

### Why it is better than the mean

The mean is sensitive to every value in the sample. One catastrophically failed seed (reward = 5 when the rest are around 400) pulls the mean down by ~80 points in a 5-seed experiment. One lucky seed inflates it equally. IQM discards both.

Agarwal et al. (2021) show that IQM has substantially lower sample complexity than the mean — you need fewer runs to achieve the same statistical confidence. For RL, where each run is expensive (hours of training time), this matters.

### Formula

Given `N` scores sorted in ascending order `s[0] ≤ s[1] ≤ … ≤ s[N-1]`:

```
IQM = mean( s[i] for i in [N//4, N//4 + 1, …, 3*N//4 - 1] )
```

The slice `[N//4 : 3*N//4]` in Python notation selects the middle 50%.

### Python implementation

```python
import numpy as np


def iqm(scores: np.ndarray) -> float:
    """
    Interquartile Mean: mean of the middle 50% of sorted scores.

    Parameters
    ----------
    scores : np.ndarray
        1-D array of scalar scores (e.g. final episode rewards across seeds).

    Returns
    -------
    float
        The IQM of the input scores.
    """
    sorted_scores = np.sort(scores)
    n = len(sorted_scores)
    lower = n // 4
    upper = 3 * n // 4
    return float(np.mean(sorted_scores[lower:upper]))


def iqm_with_stderr(scores: np.ndarray, n_bootstrap: int = 50_000) -> tuple[float, float, float]:
    """
    IQM point estimate plus 95% CI via bootstrap resampling.

    Returns
    -------
    tuple[float, float, float]
        (iqm_value, ci_lower, ci_upper)
    """
    point_estimate = iqm(scores)

    bootstrap_iqms = np.array([
        iqm(np.random.choice(scores, size=len(scores), replace=True))
        for _ in range(n_bootstrap)
    ])

    ci_lower = float(np.percentile(bootstrap_iqms, 2.5))
    ci_upper = float(np.percentile(bootstrap_iqms, 97.5))

    return point_estimate, ci_lower, ci_upper


# Example
if __name__ == "__main__":
    scores = np.array([410.0, 430.0, 390.0, 415.0, 85.0])
    val, lo, hi = iqm_with_stderr(scores)
    print(f"IQM = {val:.1f}  (95% CI: [{lo:.1f}, {hi:.1f}])")
    print(f"Mean = {np.mean(scores):.1f}")
```

### When to use IQM vs mean

| Situation | Use |
|---|---|
| Fewer than 5 seeds, quick check | Mean (acknowledge the limitation) |
| 5+ seeds, any comparative claim | IQM |
| Publication | IQM + 95% CI via bootstrap |
| Aggregating across multiple tasks | IQM (Agarwal et al. recommendation) |

!!! note "The middle 50% in small samples"
    For N=5, `[5//4 : 3*5//4]` = `[1:3]`, which is only 2 values (indices 1 and 2 of the sorted array). This is mathematically valid but noisy. For N=10 you get 5 values, which is much more stable. This is one reason the paper recommends 10 seeds for publication.

---

## 4 · Performance Profiles

### What they are

A **performance profile** is the Complementary Cumulative Distribution Function (CCDF) of normalised scores across seeds (and optionally tasks). For a given score threshold τ, the profile shows the fraction of runs that achieved at least τ.

Formally: `ρ(τ) = P(score ≥ τ)` where the probability is over all (algorithm, seed, task) combinations in your experiment.

A curve that is higher on the left and drops to zero further to the right represents an algorithm that achieves higher scores more reliably.

### Why they are better than point estimates

A point estimate (mean, IQM, median) collapses the full distribution to a single number. Two algorithms can have identical IQMs with completely different risk profiles. Algorithm A might always score around 350. Algorithm B might score 500 half the time and 200 the other half, with IQM ≈ 350. From a deployment perspective these are very different algorithms.

The performance profile shows this. At τ=400, algorithm A's profile would be at 0.0 (it never reaches 400), while algorithm B's would be at 0.5 (half its runs do).

### Install rliable

```bash
pip install rliable
```

rliable is the companion library for Agarwal et al. 2021. It implements performance profiles, IQM, probability of improvement, and optimality gap with statistically correct confidence intervals.

### Code: performance profiles for PPO vs SAC on FlyBy

The following uses synthetic data that approximates what you would expect after real training. Replace the arrays with your actual seed results from `collect_seed_results()` above.

```python
import numpy as np
import matplotlib.pyplot as plt
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

# -----------------------------------------------------------------
# 1. Scores: shape (num_runs, num_tasks)
#    For a single task (FlyBy), num_tasks=1.
#    Scores should be normalised to [0, 1] where 0 = random and
#    1 = a reference score (e.g. expert performance or max reward).
# -----------------------------------------------------------------

# Synthetic data: 10 seeds for PPO, 10 seeds for SAC
# Normalised to reference max reward of 500.
rng = np.random.default_rng(0)

ppo_raw = np.concatenate([
    rng.normal(loc=380, scale=30, size=8),
    rng.normal(loc=90, scale=20, size=2),   # two failed seeds
])
sac_raw = rng.normal(loc=420, scale=25, size=10)

MAX_REWARD = 500.0
ppo_scores = np.clip(ppo_raw / MAX_REWARD, 0, 1).reshape(10, 1)
sac_scores = np.clip(sac_raw / MAX_REWARD, 0, 1).reshape(10, 1)

score_dict = {
    "PPO": ppo_scores,
    "SAC": sac_scores,
}

# -----------------------------------------------------------------
# 2. Compute performance profiles
# -----------------------------------------------------------------

thresholds = np.linspace(0.0, 1.0, 201)

score_distributions, score_distributions_cis = rly.create_performance_profile(
    score_dict,
    tau_list=thresholds,
)

# -----------------------------------------------------------------
# 3. Plot
# -----------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

plot_utils.plot_performance_profiles(
    score_distributions,
    thresholds,
    performance_profile_cis=score_distributions_cis,
    colors={"PPO": "#E07B54", "SAC": "#4A90D9"},
    xlabel=r"Normalised Score $(\tau)$",
    ax=ax,
)

ax.set_title("Performance Profile — FlyBy (PPO vs SAC, 10 seeds)")
ax.set_ylabel(r"Fraction of runs with score $\geq \tau$")
plt.tight_layout()
plt.savefig("performance_profile_flyby.png", dpi=150)
plt.show()
print("Saved: performance_profile_flyby.png")
```

### How to read a performance profile

- **Y-axis** — fraction of runs that achieved at least the corresponding score on the X-axis
- **X-axis** — normalised score threshold (0 = random, 1 = reference/expert)
- **A curve that is entirely above another** — that algorithm dominates the other stochastically; it is better at every threshold
- **Curves that cross** — neither algorithm dominates; one is safer (higher left tail), the other achieves higher peak scores
- **Wide confidence bands** — you need more seeds; the uncertainty is too large to draw conclusions
- **Left tail near 1.0** — the algorithm almost never catastrophically fails

!!! tip "Choosing the reference score"
    Normalisation requires a reference score. Use the maximum achievable reward if known, or a human/expert baseline. Be consistent: the same reference must be used for all algorithms in the comparison. Document it clearly in any report.

---

## 5 · Probability of Improvement

### What it is

The **Probability of Improvement** (P(A > B)) is the fraction of all (seed_A, seed_B) pairs in which algorithm A achieves a higher score than algorithm B.

With N_A seeds for algorithm A and N_B seeds for algorithm B, there are N_A × N_B pairs. P(A > B) counts how many pairs algorithm A wins.

This is a non-parametric test that makes no assumptions about the shape of the score distributions. It is directly interpretable: 0.5 means A and B are indistinguishable, 1.0 means A always wins.

### Interpretation guide

| P(A > B) | Interpretation |
|---|---|
| 0.45 – 0.55 | No meaningful difference; coin flip |
| 0.55 – 0.65 | Weak evidence for A; probably needs more seeds |
| 0.65 – 0.75 | Moderate evidence for A |
| 0.75 – 0.90 | Strong evidence for A |
| > 0.90 | Very strong evidence; likely a real difference |

### Code with confidence intervals

```python
import numpy as np
from rliable import library as rly
from rliable import metrics

# Reuse score_dict from the performance profile section above.
# score_dict = {"PPO": ppo_scores, "SAC": sac_scores}

# Probability of improvement: SAC vs PPO
# rly.get_interval_estimates returns (point_estimates, confidence_intervals)

algorithms = ["PPO", "SAC"]
pairs = {"SAC,PPO": (score_dict["SAC"], score_dict["PPO"])}

# rliable expects shape (num_runs, num_tasks) for each algorithm.
# We already have that from the previous section.

aggregate_func = lambda scores: np.array([metrics.probability_of_improvement(
    score_dict["SAC"], score_dict["PPO"]
)])

poi_estimates, poi_cis = rly.get_interval_estimates(
    {"SAC,PPO": score_dict},           # rliable will ignore this dict structure
    aggregate_func,
    reps=50_000,
)

# --- Manual computation (clearer for learning purposes) ---

def probability_of_improvement(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """
    P(A > B): fraction of (a, b) seed pairs where a > b.

    Parameters
    ----------
    scores_a, scores_b : np.ndarray
        1-D arrays of scalar scores for each algorithm.
    """
    wins = 0
    total = 0
    for a in scores_a:
        for b in scores_b:
            if a > b:
                wins += 1
            total += 1
    return wins / total


def poi_with_bootstrap_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 50_000,
) -> tuple[float, float, float]:
    """Return (P(A>B), ci_lower, ci_upper) via bootstrap."""
    point = probability_of_improvement(scores_a, scores_b)

    bootstrap_pois = []
    for _ in range(n_bootstrap):
        resample_a = np.random.choice(scores_a, size=len(scores_a), replace=True)
        resample_b = np.random.choice(scores_b, size=len(scores_b), replace=True)
        bootstrap_pois.append(probability_of_improvement(resample_a, resample_b))

    bootstrap_pois = np.array(bootstrap_pois)
    ci_lower = float(np.percentile(bootstrap_pois, 2.5))
    ci_upper = float(np.percentile(bootstrap_pois, 97.5))
    return point, ci_lower, ci_upper


if __name__ == "__main__":
    # Use the raw arrays from section 4 (not normalised, for interpretability)
    ppo_raw_1d = ppo_scores.flatten() * MAX_REWARD
    sac_raw_1d = sac_scores.flatten() * MAX_REWARD

    poi, lo, hi = poi_with_bootstrap_ci(sac_raw_1d, ppo_raw_1d)
    print(f"P(SAC > PPO) = {poi:.3f}  (95% CI: [{lo:.3f}, {hi:.3f}])")
```

!!! warning "P(A > B) is not transitive"
    Probability of improvement is pairwise. If P(A > B) = 0.7 and P(B > C) = 0.7, it does not follow that P(A > C) = 0.7. Always compute pairwise comparisons directly.

---

## 6 · Practical Evaluation Workflow

Here is the complete workflow from raw training runs to a publishable result table.

### Step-by-step

1. **Train** — run N seeds for each algorithm using the bash script in Section 2
2. **Collect** — load final rewards for each seed using `collect_seed_results()`
3. **Normalise** — divide by the reference score so all values are in [0, 1]
4. **Compute IQM** — use `iqm_with_stderr()` from Section 3
5. **Plot profile** — use rliable as in Section 4
6. **Compute P(A > B)** — use `poi_with_bootstrap_ci()` from Section 5
7. **Report** — include seeds, timesteps, hardware, and all three metrics

### Template evaluation script

The following ~60-line script ties everything together. It loads TensorBoard logs from a standard directory layout, computes IQM with 95% CI, and prints a results table.

```python
#!/usr/bin/env python3
"""
evaluate.py — load gdrl TensorBoard logs, compute IQM and 95% CI.

Directory layout expected:
    logs/{ENV}_{ALGO}_seed{N}/
        events.out.tfevents.*

Usage:
    python evaluate.py --env FlyBy --algos ppo sac --seeds 1 2 3 4 5
"""

import argparse
import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ---- Data loading -------------------------------------------------------

def load_final_reward(log_dir: str, tag: str = "rollout/ep_rew_mean", last_n: int = 10) -> float:
    ea = EventAccumulator(log_dir)
    ea.Reload()
    events = ea.scalars.Items(tag)
    values = [e.value for e in events]
    return float(np.mean(values[-last_n:] if len(values) >= last_n else values))


def collect_results(base_dir: str, env: str, algo: str, seeds: list[int]) -> np.ndarray:
    return np.array([
        load_final_reward(os.path.join(base_dir, f"{env}_{algo}_seed{s}"))
        for s in seeds
    ])


# ---- Statistics ---------------------------------------------------------

def iqm(scores: np.ndarray) -> float:
    s = np.sort(scores)
    n = len(s)
    return float(np.mean(s[n // 4: 3 * n // 4]))


def bootstrap_ci(scores: np.ndarray, stat_fn, n_reps: int = 50_000, alpha: float = 0.05):
    samples = [stat_fn(np.random.choice(scores, size=len(scores), replace=True)) for _ in range(n_reps)]
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


# ---- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="FlyBy")
    parser.add_argument("--algos", nargs="+", default=["ppo", "sac"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--ref-score", type=float, default=500.0,
                        help="Reference score for normalisation")
    args = parser.parse_args()

    print(f"\nEnvironment : {args.env}")
    print(f"Seeds       : {args.seeds}")
    print(f"Ref score   : {args.ref_score}")
    print("-" * 55)
    print(f"{'Algorithm':<12} {'IQM':>8} {'95% CI':>20} {'Mean':>8}")
    print("-" * 55)

    for algo in args.algos:
        raw = collect_results(args.log_dir, args.env, algo, args.seeds)
        norm = raw / args.ref_score
        val = iqm(norm)
        lo, hi = bootstrap_ci(norm, iqm)
        mean = float(np.mean(norm))
        print(f"{algo.upper():<12} {val:>8.3f} [{lo:.3f}, {hi:.3f}]{mean:>8.3f}")

    print("-" * 55)
    print("\nAll values are normalised to reference score.")
    print("IQM = Interquartile Mean | CI = 95% bootstrap confidence interval")


if __name__ == "__main__":
    main()
```

### What to include in a report or README

Every result you share should state:

```
Algorithm : PPO (Stable Baselines 3 v2.3)
Environment: FlyBy (godot-rl-agents v0.5)
Seeds : 10 (seeds 1–10, fixed before training)
Timesteps : 1,000,000 per seed
Hardware : NVIDIA RTX 3080, AMD Ryzen 9 5900X
IQM (normalised) : 0.81 ± 0.06 (95% CI: [0.74, 0.88])
Reference score : 500 (maximum achievable reward)
```

!!! tip "Reproducibility checklist"
    - [ ] Fixed seeds documented
    - [ ] Exact library versions recorded (`pip freeze > requirements.txt`)
    - [ ] Hardware and OS noted
    - [ ] Normalisation reference score defined
    - [ ] Training timesteps per seed stated
    - [ ] Evaluation window (last N episodes) specified

!!! check "Done when"
    You have TensorBoard logs for at least 5 seeds per algorithm (the `train_seeds.sh` layout from Section 2), and `evaluate.py` prints the results table — IQM and 95% bootstrap CI per algorithm — over **your own runs**, not the synthetic data. You can state which claim the table supports: either the CIs do not overlap ("no overlap = significant", as in the Summary), or they do and your honest conclusion is "no detectable difference — more seeds needed". If you also ran the Section 4 profile script on your scores, `performance_profile_flyby.png` exists and tells the same story as the table.

---

## 7 · Applied Example — PPO vs SAC on FlyBy

This section walks through a complete comparison using realistic synthetic data. The numbers are chosen to match what you would plausibly see after training both algorithms for 1 M steps on FlyBy.

### Setup

| | PPO | SAC |
|---|---|---|
| Seeds | 10 | 10 |
| Timesteps | 1,000,000 | 1,000,000 |
| Ref score | 500 | 500 |

### Synthetic results

```python
import numpy as np

rng = np.random.default_rng(42)

# PPO: 8 seeds converge well, 2 fail to escape local optima
ppo_raw = np.concatenate([
    rng.normal(loc=375, scale=28, size=8),
    rng.normal(loc=88, scale=18, size=2),
])

# SAC: more consistent, slightly higher ceiling
sac_raw = rng.normal(loc=418, scale=22, size=10)

MAX_REWARD = 500.0
ppo_norm = np.clip(ppo_raw / MAX_REWARD, 0.0, 1.0)
sac_norm = np.clip(sac_raw / MAX_REWARD, 0.0, 1.0)
```

### Results table

| Algorithm | IQM (norm.) | 95% CI | Mean (norm.) | Failure rate |
|---|---|---|---|---|
| PPO | 0.763 | [0.701, 0.812] | 0.635 | 20% (2/10 seeds) |
| SAC | 0.836 | [0.792, 0.876] | 0.836 | 0% (0/10 seeds) |

### Interpretation

**SAC wins on FlyBy under these conditions.** The evidence is:

1. SAC's IQM (0.836) is above PPO's entire 95% CI (upper bound 0.812). The confidence intervals do not overlap — this is a statistically meaningful difference.
2. P(SAC > PPO) ≈ 0.78 (computed from pairwise seed comparisons). This is in the "strong evidence" range.
3. The performance profile for SAC dominates PPO at every threshold above τ ≈ 0.5.
4. PPO has a 20% failure rate (seeds that collapse to near-zero reward). SAC has 0%. For deployment, this matters enormously.

**What the CI tells you:** The 95% CI for PPO is [0.701, 0.812]. This means: if you ran this experiment many times, 95% of the time the true PPO IQM would fall in this range. The width (0.111) reflects genuine uncertainty — partly because of the two failed seeds. With 20 seeds, the CI would narrow.

!!! note "The mean hides the story"
    PPO's mean (0.635) is much lower than its IQM (0.763) because the two failed seeds pull it down. If you reported only the mean, PPO would look far worse than it typically is. If you reported only the best 8 seeds, it would look far better than a practitioner should expect. IQM gives the honest picture.

---

## 8 · Viz Checkpoint

Statistical tables tell you what happened numerically. Video tells you *why*.

### Record the best-seed and worst-seed policies

After identifying your best and worst seeds from the results table, render a video of each:

```bash
# Record best seed (seed 3 in this example — replace with your actual best)
gdrl eval \
    --env-id FlyBy \
    --model-path logs/FlyBy_ppo_seed3/best_model.zip \
    --n-eval-episodes 5 \
    --record-video \
    --video-path videos/flyby_ppo_seed3_best.mp4

# Record worst seed (seed 7 in this example — replace with your actual worst)
gdrl eval \
    --env-id FlyBy \
    --model-path logs/FlyBy_ppo_seed7/best_model.zip \
    --n-eval-episodes 5 \
    --record-video \
    --video-path videos/flyby_ppo_seed7_worst.mp4
```

Watch both videos side by side. Questions to answer:

- **What strategy did the best seed learn?** Is it a stable hover, a direct approach, something unexpected?
- **What went wrong for the worst seed?** Did it learn a degenerate policy (spinning in place, immediate crash)? Did it never explore past the initial state?
- **Is the failure mode reproducible?** If you restart the worst seed with slightly different initialisation, do you see the same failure? This distinguishes environment bugs from optimisation variance.
- **Would a user notice?** Sometimes the failed seed still produces plausible-looking behaviour that scores low due to a subtle policy error. Video makes this visible.

!!! tip "Qualitative failure taxonomy"
    Build a vocabulary for failure modes in your environment. For FlyBy, common failures include:
    - **Hover lock** — agent learns to hover at spawn point, never approaching the target
    - **Oscillation** — agent overshoots target repeatedly without damping
    - **Cliff walking** — agent reaches the boundary of the flight volume and gets stuck
    
    Documenting these helps you diagnose whether a bad seed failed due to optimisation variance or a policy that found a different (bad) local optimum.

---

## 9 · Stretch Goals

These exercises go beyond the unit. They are suitable for students on the researcher track who want publication-level rigour.

### 1 · Reproduce a paper result with rliable

Find a recent RL paper that reports results on an Atari or MuJoCo benchmark. Download the score data from the paper's supplementary material or from the authors' GitHub repository. Reproduce their performance profile and IQM using rliable. Questions to investigate:

- Does your reproduced figure match the paper's figure?
- What reference score did the authors use for normalisation?
- How many seeds did they use? Is it enough?

The Atari-57 scores from Agarwal et al. 2021 are publicly available and make an excellent starting point.

### 2 · Add IQM logging to a CleanRL training loop

[unit-cleanrl.md](unit-cleanrl.md) covers running training with CleanRL's single-file training scripts. Extend the logging in a CleanRL script to compute and log IQM across a rolling window of recent episode rewards during training:

```python
# Add to CleanRL's main training loop, inside the evaluation block:
from collections import deque

eval_rewards = deque(maxlen=100)

# After each evaluation episode:
eval_rewards.append(episode_reward)

if len(eval_rewards) >= 10:
    scores_arr = np.sort(np.array(eval_rewards))
    n = len(scores_arr)
    rolling_iqm = float(np.mean(scores_arr[n // 4: 3 * n // 4]))
    writer.add_scalar("eval/iqm_reward", rolling_iqm, global_step)
```

This gives you an IQM training curve in TensorBoard instead of just the mean, which is more robust during the unstable early phase of training.

### 3 · Build a results dashboard

Write a Python script using `watchdog` (file system events) and `matplotlib` that:

1. Monitors the `logs/` directory for new TensorBoard event files
2. Automatically re-loads results when a seed completes
3. Recomputes IQM and updates a live performance profile plot
4. Saves the updated plot to `dashboard.png` on each update

```bash
pip install watchdog matplotlib rliable tensorboard
```

This gives you a live view of statistical significance accumulating as your seeds complete overnight, without having to manually re-run the evaluation script.

---

## Summary

| Concept | Why it matters | Tool |
|---|---|---|
| Multi-seed training | Variance in RL is real; one seed proves nothing | bash loop + gdrl |
| IQM | Robust point estimate that resists outlier seeds | numpy |
| 95% bootstrap CI | Quantifies uncertainty; enables "no overlap = significant" | numpy |
| Performance profile | Shows full distribution, not just a point | rliable |
| P(A > B) | Direct pairwise comparison without distribution assumptions | rliable / numpy |

The core message: **report IQM ± CI from at least 5 seeds, plot a performance profile, and show a video of your best and worst runs.** Anything less is incomplete.

For debugging policies that fail to learn, see [→ Debugging](unit-debugging.md).

For adaptive hyperparameter search across a population of agents, see [→ Population-Based Training](unit-pbt.md).

---

[← Parallel Training](unit-05.md) · [Course home](index.md) · [→ Population-Based Training](unit-pbt.md)
