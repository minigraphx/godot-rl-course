# Unit 5 — Paralleltraining (parallel training)

Dieselbe **BallChase**-Umgebung, die du bereits kennst — aber diesmal öffnest du den Quellcode, fügst parallele Umgebungsinstanzen innerhalb eines einzigen Godot-Prozesses hinzu und misst den Durchsatz (throughput)-Gewinn. Die neue Fähigkeit hier ist **Skalierung**, nicht eine neue Umgebung.

[← Unit 4: JumperHard & PPO](unit-04.md) · [Kursübersicht](index.md)

!!! note "Voraussetzungen"
    - **[Unit 4](unit-04.md)** — ein funktionierender PPO-Trainingslauf + sicherer Umgang mit TensorBoard
    - **[Unit 2](unit-02.md)** — BallChase-Quellcode mindestens einmal im Editor geöffnet
    - Sicherer Umgang mit dem Export eines headless Godot-Binarys (Unit 3 §9)
    - Kein neues Python — diese Unit ändert nur die *Trainingsszene* und CLI-Flags

!!! info "Zeit"
    Lesen: ~25 min · Training: ~20 min GPU / ~1 Std CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    Godot (Schritte/Sekunde-Zähler + Viz-Checkpoint) · TensorBoard (Wanduhrzeit vs. ep_rew_mean) · Knotenanzahl in der Trainingsszene

---

## 1 · Warum Parallelisierung hilft

Jede RL-Aktualisierung benötigt einen Batch aus vielfältigen Übergängen. Eine einzelne Umgebung erzeugt korrelierte Übergänge (aufeinanderfolgende Frames aus derselben Episode). Der Betrieb von **N parallelen Umgebungen** liefert gleichzeitig N unabhängige Trajektorien:

- **Mehr Vielfalt** → bessere Gradientenschätzungen → schnellere Konvergenz
- **Höhere GPU/CPU-Auslastung** — der Trainer wartet nicht mehr auf eine einzelne Umgebung
- **Gleiche Wanduhrzeit, mehr Schritte** — N Umgebungen laufen nicht N× langsamer; Godot verwaltet sie in einem Prozess

Der Kompromiss: mehr RAM pro Umgebungsinstanz, und die Trainingsszene wird größer und schwieriger visuell zu debuggen.

```
1 env  × 1M steps = 1M transitions, ~60 min
8 envs × 125k steps each = 1M transitions, ~8 min  (approximately)
```

---

## 2 · BallChase aus dem Quellcode öffnen

1. Klone [godot_rl_agents_examples](https://github.com/edbeeching/godot_rl_agents_examples), falls noch nicht geschehen
2. Öffne `examples/BallChase` in Godot .NET (diesmal nicht das Hub-Binary — du wirst die Trainingsszene bearbeiten)
3. Aktiviere das Godot RL Agents-Plugin

---

## 3 · Parallele Umgebungsinstanzen hinzufügen

Öffne `training_scene.tscn`. Du solltest bereits einen `Sync`-Knoten und eine `BallChase`-Umgebungswurzel haben.

**Instanzen hinzufügen:**

1. Wähle im Szenenbaum den Umgebungs-Wurzelknoten (z. B. `BallChase`)
2. Dupliziere ihn (**Ctrl+D** oder Rechtsklick → Duplizieren) 7 Mal → du hast jetzt 8 Instanzen
3. Verteile sie räumlich, damit sie sich nicht überlappen (jede auswählen, mit dem Transform-Gizmo verschieben)
4. Alle Instanzen teilen denselben `Sync`-Knoten — keine zusätzliche Konfiguration erforderlich

**Den Sync-Knoten überprüfen:**

| Eigenschaft | Empfohlener Wert |
|-------------|-----------------|
| Control Mode | `TRAINING` |
| Speed Up | `20` |
| Action Repeat | `1` |

Exportiere nach dem Speichern der Szene ein neues Binary (Projekt → Exportieren).

---

## 4 · Den Durchsatzgewinn messen

Führe drei Experimente durch — 1, 4 und 8 parallele Instanzen — und vergleiche die Wanduhrzeit, um denselben `ep_rew_mean`-Wert zu erreichen:

```bash
conda activate godot_env
tensorboard --logdir=logs &

# 1 env
gdrl --env_path=./BallChase.x86_64 \
  --experiment_name=ballchase_1env \
  --timesteps=500000 --n_parallel=1 --speedup=20

# 4 envs
gdrl --env_path=./BallChase.x86_64 \
  --experiment_name=ballchase_4env \
  --timesteps=500000 --n_parallel=4 --speedup=20

# 8 envs
gdrl --env_path=./BallChase.x86_64 \
  --experiment_name=ballchase_8env \
  --timesteps=500000 --n_parallel=8 --speedup=20
```

Schalte in TensorBoard die x-Achse auf **Wanduhrzeit** (nicht Schritte), um den tatsächlichen Speedup zu sehen.

!!! check "Fertig, wenn"
    Alle drei Läufe starten sauber, trainieren bis 500k Schritte und erscheinen als separate Experimente in TensorBoard. Mit der x-Achse auf **Wanduhrzeit** erreicht der 8-env-Lauf jedes gegebene `ep_rew_mean`-Niveau deutlich schneller als der 1-env-Lauf. Erwarte einen großen Speedup — das genaue Verhältnis hängt aber von CPU-Kernzahl und Seed ab. Prüfkriterium ist die *Reihenfolge* (8 > 4 > 1 beim Durchsatz), nicht ein bestimmter Faktor.

!!! tip "n_parallel vs. Instanzen in der Szene"
    `--n_parallel` startet **separate Godot-Prozesse**. Instanzen in der Szene laufen innerhalb **eines Prozesses**. Beide erhöhen die Parallelität; ihre Kombination ergibt maximalen Durchsatz. Instanzen in der Szene sind einfacher einzurichten; `--n_parallel` skaliert besser auf Mehrkern-Maschinen.

---

## 5 · Evaluierungsprotokoll

Verwende dieselbe deterministische Evaluierungsschleife aus Unit 4 — führe 20 Episoden mit `deterministic=True` aus und berichte Mittelwert ± Standardabweichung. Ein korrekt trainierter BallChase-Agent sollte bei 500k Schritten mit 8 Umgebungen durchschnittlich > 80 Belohnung erzielen.

```python
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import numpy as np

env = StableBaselinesGodotEnv(env_path="./BallChase.x86_64", n_parallel=1, speedup=1)
model = PPO.load("logs/sb3/ballchase_8env/best_model")

rewards = []
for _ in range(20):
    obs, done, total = env.reset(), False, 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _ = env.step(action)
        total += r
    rewards.append(total)

print(f"Mean ± std: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
env.close()
```

**Viz-Checkpoint** — Wiederhole eine Evaluierungsepisode mit `show_window=True`. Bestätige, dass der Agent den Ball zuverlässig verfolgt.

---

## 6 · Warum ein einziger Seed niemals ausreicht

RL-Training weist eine hohe Varianz zwischen zufälligen Seeds auf. Zwei Läufe mit identischen Hyperparametern und derselben Anzahl von Zeitschritten — aber unterschiedlichen zufälligen Seeds — können `ep_rew_mean`-Werte erzeugen, die sich bei der Konvergenz um 50 % oder mehr unterscheiden.

Ein einzelner Trainingslauf, der „funktioniert", könnte ein glücklicher Seed sein. Ein Lauf, der „scheitert", könnte ein unglücklicher sein. Wenn du Hyperparameter auf einem einzigen Seed abstimmst, optimierst du möglicherweise für Zufälligkeit statt für die Qualität des Algorithmus.

**Standardpraxis:** Führe N = 3–5 Seeds aus, berichte **Mittelwert ± Standardabweichung über Seeds hinweg** (nicht Mittelwert und Standardabweichung innerhalb eines einzelnen Laufs).

```python
import subprocess
import numpy as np
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

seeds = [0, 1, 2, 3, 4]
final_rewards = []

for seed in seeds:
    # Train with this seed
    subprocess.run([
        "gdrl",
        "--env_path=./BallChase.x86_64",
        f"--experiment_name=ballchase_seed{seed}",
        "--timesteps=500000",
        "--n_parallel=8",
        "--speedup=20",
        f"--seed={seed}",
    ])

    # Evaluate the trained model
    env = StableBaselinesGodotEnv(env_path="./BallChase.x86_64", n_parallel=1, speedup=1)
    model = PPO.load(f"logs/sb3/ballchase_seed{seed}/best_model")

    ep_rewards = []
    for _ in range(20):
        obs, done, total = env.reset(), False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _ = env.step(action)
            total += r
        ep_rewards.append(total)

    final_rewards.append(np.mean(ep_rewards))
    env.close()

print(f"Mean across seeds: {np.mean(final_rewards):.1f} ± {np.std(final_rewards):.1f}")
```

!!! tip "Wie viele Seeds?"
    Für ein Kursprojekt: 1 Seed ist ausreichend — du lernst, du veröffentlichst nicht.
    Für einen Vergleich zwischen zwei Methoden: mindestens 3–5 Seeds.
    Für ein Paper oder eine Produktionsentscheidung: 10 Seeds.

**Zwei Algorithmen korrekt vergleichen (PPO vs SAC):**

Verwende *dieselben* Seeds für beide Algorithmen und vergleiche dann Mittelwert ± Standardabweichung über diese Seeds hinweg:

```python
# WRONG: PPO on seeds [0,1,2], SAC on seeds [3,4,5]
# The seed sets are different — any difference might be due to seed luck

# CORRECT: both algorithms trained on seeds [0, 1, 2, 3, 4]
ppo_rewards = [train_and_eval("PPO", seed) for seed in seeds]
sac_rewards  = [train_and_eval("SAC",  seed) for seed in seeds]

print(f"PPO: {np.mean(ppo_rewards):.1f} ± {np.std(ppo_rewards):.1f}")
print(f"SAC: {np.mean(sac_rewards):.1f}  ± {np.std(sac_rewards):.1f}")
```

Gepaarte Seeds kontrollieren die Umgebungszufälligkeit — wenn PPO SAC bei Seed 0, 1, 2, 3 und 4 schlägt, ist das eine viel stärkere Schlussfolgerung als ein Einzelseed-Vergleich.

Verwende in TensorBoard die Schattierungsansicht (IQM oder Mittelwert ± Standardabweichung), um Mehrfach-Seed-Ergebnisse zu visualisieren. Die Breite des schattierten Bereichs sagt mehr über die Stabilität des Algorithmus aus als die Mittellinie allein.

---

## 7 · Stretch Goals (eines auswählen)

- **Skalierungskurve** — zeichne Schritte/Sekunde vs. N Umgebungen (1, 2, 4, 8, 16). Ab wann flacht der Gewinn ab?
- **Batch-Size-Skalierung** — wenn du `n_parallel` verdoppelst, verdopple auch `--batch_size`. Hilft das?
- **Andere Umgebung** — wende dieselbe Parallelisierungstechnik auf deinen Lunar Lander aus Unit 2 an

---

## Was kommt als Nächstes

**Unit 6:** Kontinuierliche Steuerung (continuous control) in 3D — FlyBy / HovercraftRacing, kontinuierliche Aktionsräume (action spaces), Beobachtungsnormalisierung für 3D-Sensoren.

!!! info "Selbstcheck, bevor du weitermachst"
    Kannst du diese Fragen mit eigenen Worten beantworten?

    1. Warum liefern N parallele Umgebungen *bessere* Gradientenschätzungen, nicht nur *mehr* davon?
    2. Was ist der Unterschied zwischen prozessinternen parallelen Umgebungen (der Trainingsszene) und `--n_parallel`-Subprozessen, und wann würdest du was verwenden?
    3. Wenn Schritte/Sekunde bis 8 Umgebungen linear skaliert, aber bei 16 abflacht — wo liegt der Engpass vermutlich?
    4. Warum ist eine Belohnungskurve mit nur einem Seed nie ein ausreichender Beleg dafür, dass eine Änderung „geholfen" hat?
    5. Warum braucht eine Verdopplung von `n_parallel` oft eine entsprechende Erhöhung von `batch_size`?

    Wenn du alle fünf beantworten kannst — bist du bereit.

??? success "Antworten zum Selbstcheck"
    1. Eine einzelne Umgebung erzeugt **korrelierte Übergänge** — aufeinanderfolgende Frames aus derselben Episode. N parallele Umgebungen liefern gleichzeitig N unabhängige Trajektorien, sodass jeder Batch mehr vom Zustandsraum abdeckt; genau diese Vielfalt — nicht bloß die größere Menge — verbessert die Gradientenschätzung.
    2. Instanzen in der Szene sind duplizierte Umgebungswurzeln innerhalb **eines Godot-Prozesses** — schnell eingerichtet, alle teilen denselben `Sync`-Knoten. `--n_parallel` startet **separate Godot-Prozesse** und skaliert besser auf Mehrkern-Maschinen. Nimm Instanzen in der Szene für den bequemen Einstieg, `--n_parallel` (oder beides kombiniert) für maximalen Durchsatz.
    3. Eine geteilte Hardware-Ressource — am wahrscheinlichsten **ausgelastete CPU-Kerne** (16 Umgebungen konkurrieren um weniger physische Kerne); danach kommen RAM pro Instanz und der einzelne Trainer-Prozess in Frage. Ab diesem Punkt erzeugen zusätzliche Umgebungen nur noch Konkurrenz, keinen Durchsatz.
    4. Die **Seed-Varianz** im RL ist enorm — identische Hyperparameter können bei der Konvergenz um 50 % oder mehr in `ep_rew_mean` auseinanderliegen, eine einzelne „verbesserte" Kurve kann also schlicht ein glücklicher Seed sein. Erst Mittelwert ± Standardabweichung über 3–5 gepaarte Seeds trennt einen echten Effekt von Zufall.
    5. Eine Verdopplung von `n_parallel` verdoppelt die pro Rollout gesammelten Daten; bleibt `batch_size` fix, verschiebt sich das **Verhältnis von Updates zu Daten**, und jedes Gradienten-Update sieht nur einen kleinen Ausschnitt des nun vielfältigeren Buffers. Wer `batch_size` mitwachsen lässt, nutzt die zusätzliche Vielfalt in jedem Update tatsächlich aus.

[→ Unit 6: Kontinuierliche 3D-Steuerung](unit-06.md)

!!! tip "Wenn paralleles Training langsam oder instabil wirkt"
    Siehe den [Leitfaden zum Debuggen von RL-Training](unit-debugging.md#9-performance-langsames-training) für Durchsatz-Engpässe, Probleme beim Start von Subprozessen und VecEnv-Fehlkonfigurationen.
