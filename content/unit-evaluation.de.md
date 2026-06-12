# Fortgeschrittene Evaluation — IQM, Performance Profiles und statistische Strenge

[← Paralleles Training](unit-05.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~25 min

---

!!! info "Drei Wege, deine KI zu beobachten"
    Diese Unit führt drei komplementäre Linsen ein, um eine trainierte Policy rigoros zu evaluieren:

    - **rliable-Performance-Profile-Plot** — die volle Verteilung der Scores über Seeds und Tasks, nicht nur eine einzelne Zahl
    - **IQM-Ergebnistabelle** — eine robuste Punktschätzung mit 95 %-Konfidenzintervallen, die gegen Ausreißer-Seeds resistent ist
    - **Per-Seed-TensorBoard-Overlay** — visualisiere jeden einzelnen Trainingslauf, damit du Varianz siehst, nicht nur Durchschnitte

    Wenn du Ergebnisse publizierst oder zwei Algorithmen für irgendetwas jenseits persönlicher Neugier vergleichst, sind alle drei Pflicht.

---

## 1 · Warum Mittelwert ± Std versagt

Wenn Studierende zum ersten Mal einen RL-Agenten trainieren, öffnen sie TensorBoard, schauen auf die mittlere Episodenbelohnungs-Kurve und erklären die Sache für erledigt. Das ist verständlich, aber auch eine der häufigsten Quellen irreführender Ergebnisse in der RL-Literatur.

### Das Varianzproblem

Ein einzelner Trainingslauf in Deep RL ist keine Messung — er ist eine Stichprobe aus einem hochvarianten stochastischen Prozess. Das Ergebnis hängt vom Random-Seed ab, der Gewichtsinitialisierung, Umgebungs-Resets, Aktions-Sampling und (in manchen Frameworks) die Reihenfolge des Experience Replay steuert. Zwei Läufe mit identischen Hyperparametern können wild unterschiedliche finale Policies erzeugen.

**Konkretes Beispiel — PPO auf FlyBy:**

Stell dir vor, du trainierst PPO zweimal auf der FlyBy-Umgebung. Seed 1 findet bis Episode 200 k eine stabile Hover-Strategie und plateaut bei Belohnung ≈ 420. Seed 2 entkommt nie einem lokalen Optimum und stagniert bei Belohnung ≈ 80 für den gesamten Lauf. Berichtest du nur Seed 1, sieht dein Ergebnis exzellent aus. Nutzt dein Kollege Seed 2 als Baseline für seinen Vergleich, wird er folgern, sein Algorithmus sei weit besser, als er tatsächlich ist.

Das ist kein konstruiertes Szenario. Das ist die Norm für Sparse-Reward-Tasks, und FlyBy hat Sparse Rewards.

### Wie Ausreißer den Mittelwert dominieren

Angenommen, du läufst fünf Seeds und erhältst finale Belohnungen von `[410, 430, 390, 415, 85]`. Der Mittelwert ist 346 und die Standardabweichung 142. Was sagt „346 ± 142" einer Leserin? Fast nichts Nützliches. Die 85 ist ein Ausreißer aus einem fehlgeschlagenen Lauf. Die anderen vier liegen eng um 411. Der Mittelwert wurde 65 Punkte unter das typische Ergebnis gezogen, und die Standardabweichung ist so aufgebläht, dass sie irreführend ist.

Der IQM für diese Menge ist der Mittelwert von `[390, 410, 415]` (die mittleren 60 % einer 5-Stichproben-Menge approximieren die mittleren 50 %) ≈ 405, was deutlich repräsentativer ist für das, was der Algorithmus in der Praxis tatsächlich erreicht.

### Gängige irreführende Evaluations-Muster

| Muster | Was es verbirgt |
|---|---|
| Nur den besten Seed berichten | Typische Performance; Ausfallrate |
| Mittelwert von 3 Seeds berichten | Hohe Varianz; statistisches Rauschen als Signal |
| Das Evaluations-Fenster cherry-picken | Policies, die nach dem Snapshot kollabierten |
| Zwei Algorithmen mit je einem Seed vergleichen | Ob der Unterschied real ist oder nur Glück |
| Max-Belohnung über das Training berichten | Dass der Agent diesen Peak selten erreicht |
| Denselben Seed für Training und Eval nutzen | Overfitting auf eine spezifische Zufalls-Trajektorie |
| TensorBoard-Kurven stark glätten | Kurzlebige Spikes, die wie Konvergenz aussehen |

!!! warning "Seed 42 ist kein Ergebnis"
    „Ich habe Seed 42 ausprobiert und es funktionierte" ist Anekdote, kein Beweis. Die RL-Community hat eine Reproduzierbarkeitskrise, die fast ausschließlich aus unter-seeded Experimenten stammt. Ein einzelner Seed-Lauf ist nützlich zum Debuggen. Er ist nie ausreichend für eine vergleichende Aussage.

---

## 2 · Mehrere Seeds laufen lassen — die Baseline-Anforderung

Die Mindestanforderung für jede Aussage über Algorithmen-Performance:

- **5 Seeds** — Minimum für jedes Ergebnis, das du mit anderen teilst
- **10 Seeds** — Minimum für ein Konferenz- oder Workshop-Paper
- **20+ Seeds** — erforderlich für Ergebnisse auf hochvarianten Umgebungen oder bei kleinen Effektgrößen

### Training mit mehreren Seeds

Das folgende Bash-Skript führt `gdrl` (den godot-rl-agents-Trainingsbefehl) für die Seeds 1 bis 5 aus und speichert jeden Lauf in einem eigenen Log-Verzeichnis. Passe Umgebungsname und Hyperparameter an dein Projekt an.

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

Starte es mit:

```bash
chmod +x train_seeds.sh
./train_seeds.sh
```

Jeder Lauf erzeugt ein eigenes TensorBoard-Log-Verzeichnis. Du kannst alle fünf in einer TensorBoard-Session überlagern:

```bash
tensorboard --logdir logs/
```

TensorBoard gruppiert die Läufe und zeigt einzelne Kurven für jeden Seed neben dem geglätteten Aggregat.

### Ergebnisse in Python laden

Sobald das Training abgeschlossen ist, extrahiere die finale Episodenbelohnung für jeden Seed mit dem `tensorboard`-Python-Paket (neben Stable Baselines 3 installiert):

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

!!! tip "Querverweis"
    [Unit 05](unit-05.md) führte Multi-Seed-Evaluation kurz im Kontext des parallelen Trainings ein. Diese Unit formalisiert das statistische Argument dahinter und ergänzt das Tooling, das sie rigoros macht.

---

## 3 · Interquartile Mean (IQM)

### Was er ist

Der **Interquartile Mean (IQM)** ist das arithmetische Mittel der mittleren 50 % einer sortierten Stichprobe. Du sortierst deine Scores von niedrig nach hoch, verwirfst die unteren 25 % und die oberen 25 % und nimmst den Mittelwert des Restes.

Das ist nicht dasselbe wie der Median (der nur den einzelnen Mittelwert nimmt). IQM nutzt alle mittleren Scores, was ihm eine geringere Varianz als der Median verleiht und ihn zugleich weit robuster gegen Ausreißer macht als der volle Mittelwert.

### Warum er besser ist als der Mittelwert

Der Mittelwert ist sensitiv für jeden Wert in der Stichprobe. Ein katastrophal fehlgeschlagener Seed (Belohnung = 5, während der Rest um 400 liegt) zieht den Mittelwert in einem 5-Seed-Experiment um ~80 Punkte nach unten. Ein glücklicher Seed bläst ihn gleich stark auf. IQM verwirft beides.

Agarwal et al. (2021) zeigen, dass IQM eine deutlich geringere Stichprobenkomplexität hat als der Mittelwert — du brauchst weniger Läufe für dasselbe statistische Vertrauen. Für RL, wo jeder Lauf teuer ist (Stunden an Trainingszeit), ist das wichtig.

### Formel

Gegeben `N` Scores in aufsteigender Reihenfolge sortiert `s[0] ≤ s[1] ≤ … ≤ s[N-1]`:

```
IQM = mean( s[i] for i in [N//4, N//4 + 1, …, 3*N//4 - 1] )
```

Der Slice `[N//4 : 3*N//4]` in Python-Notation wählt die mittleren 50 %.

### Python-Implementierung

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

### Wann IQM vs Mittelwert

| Situation | Nutze |
|---|---|
| Weniger als 5 Seeds, schneller Check | Mittelwert (Limitierung anerkennen) |
| 5+ Seeds, jede vergleichende Aussage | IQM |
| Publikation | IQM + 95 % CI via Bootstrap |
| Aggregation über mehrere Tasks | IQM (Agarwal et al. Empfehlung) |

!!! note "Die mittleren 50 % bei kleinen Stichproben"
    Für N=5 ist `[5//4 : 3*5//4]` = `[1:3]`, das sind nur 2 Werte (Indizes 1 und 2 des sortierten Arrays). Das ist mathematisch valide, aber verrauscht. Für N=10 erhältst du 5 Werte, was deutlich stabiler ist. Das ist einer der Gründe, warum das Paper 10 Seeds für Publikationen empfiehlt.

---

## 4 · Performance Profiles

### Was sie sind

Ein **Performance Profile** ist die Complementary Cumulative Distribution Function (CCDF) der normalisierten Scores über Seeds (und optional Tasks). Für einen gegebenen Score-Schwellenwert τ zeigt das Profil den Anteil der Läufe, die mindestens τ erreichten.

Formal: `ρ(τ) = P(score ≥ τ)`, wobei die Wahrscheinlichkeit über alle (Algorithmus, Seed, Task)-Kombinationen in deinem Experiment läuft.

Eine Kurve, die links höher liegt und weiter rechts auf null fällt, repräsentiert einen Algorithmus, der zuverlässiger höhere Scores erreicht.

### Warum sie besser sind als Punktschätzungen

Eine Punktschätzung (Mittelwert, IQM, Median) komprimiert die volle Verteilung auf eine einzelne Zahl. Zwei Algorithmen können identische IQMs mit komplett unterschiedlichen Risikoprofilen haben. Algorithmus A erreicht vielleicht stets um 350. Algorithmus B scort die Hälfte der Zeit 500 und die andere Hälfte 200, mit IQM ≈ 350. Aus Deployment-Sicht sind das sehr unterschiedliche Algorithmen.

Das Performance Profile zeigt das. Bei τ=400 wäre A's Profil bei 0,0 (er erreicht nie 400), während B's bei 0,5 läge (die Hälfte seiner Läufe schaffen es).

### rliable installieren

```bash
pip install rliable
```

rliable ist die Begleitbibliothek zu Agarwal et al. 2021. Sie implementiert Performance Profiles, IQM, Probability of Improvement und Optimality Gap mit statistisch korrekten Konfidenzintervallen.

### Code: Performance Profiles für PPO vs SAC auf FlyBy

Das Folgende nutzt synthetische Daten, die approximieren, was du nach echtem Training erwarten würdest. Ersetze die Arrays durch deine tatsächlichen Seed-Ergebnisse aus `collect_seed_results()` oben.

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

### Wie man ein Performance Profile liest

- **Y-Achse** — Anteil der Läufe, die mindestens den entsprechenden Score auf der X-Achse erreichten
- **X-Achse** — normalisierter Score-Schwellenwert (0 = zufällig, 1 = Referenz/Experte)
- **Eine Kurve, die komplett über einer anderen liegt** — dieser Algorithmus dominiert den anderen stochastisch; er ist bei jedem Schwellenwert besser
- **Sich kreuzende Kurven** — keiner dominiert; einer ist sicherer (höherer linker Tail), der andere erreicht höhere Peak-Scores
- **Breite Konfidenzbänder** — du brauchst mehr Seeds; die Unsicherheit ist zu groß, um Schlussfolgerungen zu ziehen
- **Linker Tail nahe 1,0** — der Algorithmus versagt fast nie katastrophal

!!! tip "Den Referenz-Score wählen"
    Normalisierung erfordert einen Referenz-Score. Nutze die maximal erreichbare Belohnung, falls bekannt, oder eine menschliche/Experten-Baseline. Sei konsistent: derselbe Referenzwert muss für alle Algorithmen im Vergleich genutzt werden. Dokumentiere ihn klar in jedem Bericht.

---

## 5 · Probability of Improvement

### Was sie ist

Die **Probability of Improvement** (P(A > B)) ist der Anteil aller (seed_A, seed_B)-Paare, in denen Algorithmus A einen höheren Score erreicht als Algorithmus B.

Mit N_A Seeds für Algorithmus A und N_B Seeds für Algorithmus B gibt es N_A × N_B Paare. P(A > B) zählt, wie viele Paare Algorithmus A gewinnt.

Das ist ein nicht-parametrischer Test, der keine Annahmen über die Form der Score-Verteilungen macht. Er ist direkt interpretierbar: 0,5 bedeutet, A und B sind ununterscheidbar, 1,0 bedeutet, A gewinnt immer.

### Interpretations-Leitfaden

| P(A > B) | Interpretation |
|---|---|
| 0,45 – 0,55 | Kein bedeutsamer Unterschied; Münzwurf |
| 0,55 – 0,65 | Schwache Evidenz für A; wahrscheinlich mehr Seeds nötig |
| 0,65 – 0,75 | Moderate Evidenz für A |
| 0,75 – 0,90 | Starke Evidenz für A |
| > 0,90 | Sehr starke Evidenz; wahrscheinlich echter Unterschied |

### Code mit Konfidenzintervallen

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

!!! warning "P(A > B) ist nicht transitiv"
    Probability of Improvement ist paarweise. Wenn P(A > B) = 0,7 und P(B > C) = 0,7, folgt daraus nicht, dass P(A > C) = 0,7. Berechne paarweise Vergleiche stets direkt.

---

## 6 · Praktischer Evaluations-Workflow

Hier ist der vollständige Workflow von rohen Trainingsläufen bis zu einer publizierbaren Ergebnistabelle.

### Schritt für Schritt

1. **Trainieren** — N Seeds für jeden Algorithmus mit dem Bash-Skript aus Abschnitt 2 laufen lassen
2. **Sammeln** — finale Belohnungen für jeden Seed mit `collect_seed_results()` laden
3. **Normalisieren** — durch den Referenz-Score teilen, sodass alle Werte in [0, 1] liegen
4. **IQM berechnen** — `iqm_with_stderr()` aus Abschnitt 3 nutzen
5. **Profil plotten** — rliable wie in Abschnitt 4 nutzen
6. **P(A > B) berechnen** — `poi_with_bootstrap_ci()` aus Abschnitt 5 nutzen
7. **Berichten** — Seeds, Timesteps, Hardware und alle drei Metriken angeben

### Template-Evaluations-Skript

Das folgende ~60-Zeilen-Skript fügt alles zusammen. Es lädt TensorBoard-Logs aus einem Standard-Verzeichnis-Layout, berechnet IQM mit 95 % CI und druckt eine Ergebnistabelle.

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

### Was in einen Bericht oder eine README gehört

Jedes Ergebnis, das du teilst, sollte angeben:

```
Algorithm : PPO (Stable Baselines 3 v2.3)
Environment: FlyBy (godot-rl-agents v0.5)
Seeds : 10 (seeds 1–10, fixed before training)
Timesteps : 1,000,000 per seed
Hardware : NVIDIA RTX 3080, AMD Ryzen 9 5900X
IQM (normalised) : 0.81 ± 0.06 (95% CI: [0.74, 0.88])
Reference score : 500 (maximum achievable reward)
```

!!! tip "Reproduzierbarkeits-Checkliste"
    - [ ] Fixe Seeds dokumentiert
    - [ ] Exakte Bibliotheksversionen festgehalten (`pip freeze > requirements.txt`)
    - [ ] Hardware und OS notiert
    - [ ] Normalisierungs-Referenz-Score definiert
    - [ ] Training-Timesteps pro Seed angegeben
    - [ ] Evaluations-Fenster (letzte N Episoden) spezifiziert

!!! check "Fertig, wenn"
    Du hast TensorBoard-Logs für mindestens 5 Seeds pro Algorithmus (im `train_seeds.sh`-Layout aus Abschnitt 2), und `evaluate.py` druckt die Ergebnistabelle — IQM und 95 % Bootstrap-CI pro Algorithmus — über **deine eigenen Läufe**, nicht die synthetischen Daten. Du kannst sagen, welche Aussage die Tabelle stützt: Entweder überlappen die CIs nicht („keine Überlappung = signifikant", wie in der Zusammenfassung), oder sie überlappen und deine ehrliche Schlussfolgerung lautet „kein nachweisbarer Unterschied — mehr Seeds nötig". Hast du zusätzlich das Profil-Skript aus Abschnitt 4 auf deinen Scores laufen lassen, existiert `performance_profile_flyby.png` und erzählt dieselbe Geschichte wie die Tabelle.

---

## 7 · Angewendetes Beispiel — PPO vs SAC auf FlyBy

Dieser Abschnitt geht einen vollständigen Vergleich mit realistischen synthetischen Daten durch. Die Zahlen sind so gewählt, dass sie zu dem passen, was du nach dem Training beider Algorithmen für 1 M Schritte auf FlyBy plausibel sehen würdest.

### Setup

| | PPO | SAC |
|---|---|---|
| Seeds | 10 | 10 |
| Timesteps | 1.000.000 | 1.000.000 |
| Ref Score | 500 | 500 |

### Synthetische Ergebnisse

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

### Ergebnistabelle

| Algorithmus | IQM (norm.) | 95 % CI | Mittelwert (norm.) | Ausfallrate |
|---|---|---|---|---|
| PPO | 0,763 | [0,701, 0,812] | 0,635 | 20 % (2/10 Seeds) |
| SAC | 0,836 | [0,792, 0,876] | 0,836 | 0 % (0/10 Seeds) |

### Interpretation

**SAC gewinnt auf FlyBy unter diesen Bedingungen.** Die Evidenz:

1. SACs IQM (0,836) liegt über dem gesamten 95 %-CI von PPO (Obergrenze 0,812). Die Konfidenzintervalle überlappen nicht — das ist ein statistisch bedeutsamer Unterschied.
2. P(SAC > PPO) ≈ 0,78 (berechnet aus paarweisen Seed-Vergleichen). Das liegt im Bereich „starke Evidenz".
3. Das Performance Profile für SAC dominiert PPO bei jedem Schwellenwert über τ ≈ 0,5.
4. PPO hat eine 20 %-Ausfallrate (Seeds, die auf nahezu null Belohnung kollabieren). SAC hat 0 %. Für Deployment ist das enorm wichtig.

**Was das CI dir sagt:** Das 95 %-CI für PPO ist [0,701, 0,812]. Das heißt: Wenn du dieses Experiment viele Male laufen ließest, würde 95 % der Zeit der wahre PPO-IQM in diesem Bereich liegen. Die Breite (0,111) reflektiert echte Unsicherheit — teilweise wegen der zwei fehlgeschlagenen Seeds. Mit 20 Seeds würde sich das CI verengen.

!!! note "Der Mittelwert verbirgt die Geschichte"
    PPOs Mittelwert (0,635) ist viel niedriger als sein IQM (0,763), weil die zwei fehlgeschlagenen Seeds ihn nach unten ziehen. Berichtest du nur den Mittelwert, sieht PPO weit schlechter aus, als es typischerweise ist. Berichtest du nur die besten 8 Seeds, sieht es viel besser aus, als eine Praktikerin erwarten sollte. IQM gibt das ehrliche Bild.

---

## 8 · Viz-Checkpoint

Statistische Tabellen sagen dir, was numerisch passierte. Video sagt dir *warum*.

### Best-Seed- und Worst-Seed-Policies aufnehmen

Nachdem du deinen besten und schlechtesten Seed aus der Ergebnistabelle identifiziert hast, rendere von jedem ein Video:

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

Schau beide Videos nebeneinander. Fragen, die du beantworten solltest:

- **Welche Strategie hat der beste Seed gelernt?** Ist es ein stabiles Hovering, eine direkte Annäherung, etwas Unerwartetes?
- **Was lief beim schlechtesten Seed schief?** Hat er eine degenerierte Policy gelernt (auf der Stelle drehen, sofortiger Absturz)? Hat er nie über den Anfangszustand hinaus exploriert?
- **Ist der Fehlermodus reproduzierbar?** Wenn du den schlechtesten Seed mit leicht anderer Initialisierung neu startest, siehst du denselben Fehler? Das unterscheidet Umgebungs-Bugs von Optimierungs-Varianz.
- **Würde eine Nutzerin es bemerken?** Manchmal produziert der fehlgeschlagene Seed plausibel aussehendes Verhalten, das wegen eines subtilen Policy-Fehlers niedrig scort. Video macht das sichtbar.

!!! tip "Qualitative Fehler-Taxonomie"
    Baue ein Vokabular für Fehlermodi in deiner Umgebung. Für FlyBy umfassen häufige Fehler:
    - **Hover-Lock** — der Agent lernt, am Spawn-Punkt zu hovern, ohne sich dem Ziel zu nähern
    - **Oszillation** — der Agent schießt wiederholt über das Ziel hinaus, ohne zu dämpfen
    - **Cliff-Walking** — der Agent erreicht die Grenze des Flugvolumens und bleibt stecken
    
    Diese zu dokumentieren hilft dir zu diagnostizieren, ob ein schlechter Seed an Optimierungs-Varianz scheiterte oder an einer Policy, die ein anderes (schlechtes) lokales Optimum fand.

---

## 9 · Stretch Goals

Diese Übungen gehen über die Unit hinaus. Sie eignen sich für Studierende auf dem Forscher-Track, die Publikationsniveau-Strenge wollen.

### 1 · Ein Paper-Ergebnis mit rliable reproduzieren

Finde ein aktuelles RL-Paper, das Ergebnisse auf einem Atari- oder MuJoCo-Benchmark berichtet. Lade die Score-Daten aus dem Supplementary Material des Papers oder dem GitHub-Repo der Autoren herunter. Reproduziere deren Performance Profile und IQM mit rliable. Fragen zur Untersuchung:

- Passt deine reproduzierte Grafik zur Grafik des Papers?
- Welchen Referenz-Score nutzten die Autoren für die Normalisierung?
- Wie viele Seeds nutzten sie? Ist das genug?

Die Atari-57-Scores aus Agarwal et al. 2021 sind öffentlich verfügbar und ein exzellenter Ausgangspunkt.

### 2 · IQM-Logging zu einer CleanRL-Trainingsschleife hinzufügen

[unit-cleanrl.md](unit-cleanrl.md) behandelt das Training mit CleanRLs Single-File-Trainingsskripten. Erweitere das Logging in einem CleanRL-Skript, sodass IQM über ein gleitendes Fenster der letzten Episodenbelohnungen während des Trainings berechnet und geloggt wird:

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

Das gibt dir eine IQM-Trainingskurve in TensorBoard statt nur des Mittelwerts, was während der instabilen frühen Trainingsphase robuster ist.

### 3 · Ein Ergebnis-Dashboard bauen

Schreibe ein Python-Skript mit `watchdog` (Dateisystem-Events) und `matplotlib`, das:

1. Das `logs/`-Verzeichnis auf neue TensorBoard-Event-Dateien überwacht
2. Ergebnisse automatisch neu lädt, wenn ein Seed fertig ist
3. IQM neu berechnet und einen Live-Performance-Profile-Plot aktualisiert
4. Den aktualisierten Plot bei jedem Update als `dashboard.png` speichert

```bash
pip install watchdog matplotlib rliable tensorboard
```

Das gibt dir eine Live-Ansicht der statistischen Signifikanz, die sich akkumuliert, während deine Seeds über Nacht fertig werden, ohne dass du das Evaluations-Skript manuell neu starten musst.

---

## Zusammenfassung

| Konzept | Warum es wichtig ist | Tool |
|---|---|---|
| Multi-Seed-Training | Varianz in RL ist real; ein Seed beweist nichts | bash-Schleife + gdrl |
| IQM | Robuste Punktschätzung, die Ausreißer-Seeds widersteht | numpy |
| 95 % Bootstrap-CI | Quantifiziert Unsicherheit; ermöglicht „keine Überlappung = signifikant" | numpy |
| Performance Profile | Zeigt volle Verteilung, nicht nur einen Punkt | rliable |
| P(A > B) | Direkter paarweiser Vergleich ohne Verteilungs-Annahmen | rliable / numpy |

Die Kernbotschaft: **berichte IQM ± CI aus mindestens 5 Seeds, plotte ein Performance Profile und zeige ein Video deines besten und schlechtesten Laufs.** Alles weniger ist unvollständig.

Zum Debuggen von Policies, die nicht lernen, siehe [→ Debugging](unit-debugging.md).

Für adaptive Hyperparameter-Suche über eine Population von Agenten siehe [→ Population-Based Training](unit-pbt.md).

---

[← Paralleles Training](unit-05.md) · [Kursstartseite](index.md) · [→ Population-Based Training](unit-pbt.md)
