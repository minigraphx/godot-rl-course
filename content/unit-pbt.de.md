# Population-Based Training — Hyperparameter-AutoML für RL

!!! info "Zeit"
    Lesen: ~30 min · Training: ~45 min GPU / ~3 h CPU

[Kursstartseite](index.md)

---

!!! info "Drei Wege, deine KI zu beobachten"
    - **Ray Tune Dashboard** (`ray[tune]` startet eine lokale Web-UI auf `http://localhost:8265`) — beobachte die Belohnungskurve jedes Trials in Echtzeit, sieh, welche Hyperparameter-Konfigurationen am Leben sind vs. früh terminiert wurden.
    - **TensorBoard** — vergleiche den besten PBT-Trial mit deiner Optuna-Baseline bei identischen Step-Budgets; die Lücke ist deine PBT-Dividende.
    - **Hyperparameter-Trajektorien-Plot** — ein Liniendiagramm pro Hyperparameter, das zeigt, wie `learning_rate`, `ent_coef` etc. sich im Training für jeden Agenten in der Population entwickeln. Statische Suchmethoden produzieren flache Linien; PBT produziert dynamische Kurven, die adaptieren.

---

## 1 · Warum Grid Search für RL versagt

Standard-Hyperparameter-Optimierung stellt eine Frage: **bei einem festen Zeit-Budget, welche statische Konfiguration performt am besten?** Grid Search, Random Search und Optuna teilen alle diese Annahme. Du wählst einen Satz Hyperparameter, führst Training bis zum Ende aus, misst den finalen Score und gehst zur nächsten Konfiguration.

Für supervised Learning funktioniert das gut. Eine Learning Rate von `1e-3`, die 94 % Genauigkeit in 100 Epochs erreicht, erreicht ungefähr dieselbe Genauigkeit, egal ob du in Epoch 1, Epoch 50 oder Epoch 100 evaluierst. Die Hyperparameter sind stabile Merkmale einer Konfiguration.

RL bricht diese Annahme auf drei klare Arten.

**Nicht-stationäre Performance.** RL-Training ist nicht monoton. Ein Agent, der mit `learning_rate=1e-3` trainiert wird, kann bei 200k Schritten vor `learning_rate=3e-4` liegen, aber bei 1M Schritten zurückliegen, weil die größere Learning Rate spätere Updates destabilisiert. Alle Konfigurationen am selben finalen Timestep zu evaluieren übersieht das — der „Sieger" bei Schritt 1M war möglicherweise zu keinem früheren Checkpoint der Sieger.

**Interaktion zwischen Hyperparametern und Trainings-Phase.** PPOs `clip_range` existiert, um Updates zu verhindern, die zu groß sind. Früh im Training sind große Updates oft vorteilhaft — die Policy ist zufällig und muss sich schnell bewegen. Spät im Training verursachen große Updates Instabilität. Die ideale `clip_range` ist keine Konstante: sie sollte permissiv beginnen und sich verengen, während die Policy reift. Grid Search wählt einen Wert und lebt damit für den gesamten Lauf.

**Kombinatorische Explosion.** Eine bescheidene PPO-Suche über `learning_rate` (5 Werte) × `ent_coef` (4 Werte) × `gamma` (3 Werte) × `n_steps` (3 Werte) × `clip_range` (3 Werte) produziert 540 Konfigurationen. Bei 1M Schritten pro Lauf auf einer einzelnen Maschine sind das 540M Umgebungsschritte — Tage Wall-Clock-Zeit, bevor du ein einzelnes Ergebnis aus dem vollen Grid siehst. Random Search und Bayesian Optimization reduzieren die Anzahl der Evaluationen, aber sie binden jeden Lauf weiterhin an eine statische Konfiguration für seine gesamte Lebenszeit.

Das Kernproblem: **die optimalen Hyperparameter für einen RL-Agenten ändern sich während des Trainings**, und jede Methode, die eine Konfiguration als fix behandelt, lässt Performance auf dem Tisch liegen.

---

## 2 · Optuna-Recap — statische Suche und ihre Decke

Du hast wahrscheinlich gesehen, wie Optuna für Hyperparameter-Suche genutzt wird: definiere einen Suchraum, lass Optuna Konfigurationen mit TPE (Tree-structured Parzen Estimator) samplen und nutze Early Stopping, um schlechte Läufe zu killen, bevor sie das volle Budget verschwenden.

```python
import optuna
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)

    env = StableBaselinesGodotEnv(env_path="./FlyBy.x86_64", n_parallel=4, speedup=20)
    model = PPO(
        "MlpPolicy", env, verbose=0,
        learning_rate=lr,
        ent_coef=ent_coef,
        clip_range=clip_range,
    )
    model.learn(total_timesteps=500_000)
    mean_reward = evaluate_policy(model, env, n_eval_episodes=10)[0]
    env.close()
    return mean_reward

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(study.best_params)
```

Optuna ist exzellent und das richtige Werkzeug für viele RL-Probleme. Seine Limitierung ist genau das, was wir gerade beschrieben haben: die Hyperparameter `lr`, `ent_coef` und `clip_range` werden einmal gesampelt und für alle 500k Trainingsschritte konstant gehalten. Optuna findet die beste *statische* Konfiguration. Es kann nicht entdecken, dass `lr=1e-3` für die ersten 200k Schritte gefolgt von `lr=1e-4` für die nächsten 300k besser ist als jeder Wert allein.

!!! tip "Optuna ist nicht obsolet"
    PBT ersetzt Optuna nicht — es adressiert einen anderen Use Case. Der Vergleich in Abschnitt 8 wird explizit sagen, wann jedes Werkzeug gewinnt. Für die meisten Godot-Tasks mit billigen Simulatoren ist Optuna mit 30–50 Trials der richtige Startpunkt. PBT verdient seinen Overhead, wenn Trainingsläufe lang sind (> 2M Schritte) und du Hardware hast, um eine Population parallel zu fahren.

---

## 3 · Der PBT-Algorithmus

Population-Based Training (Jaderberg et al., DeepMind, 2017) lässt **N Agenten simultan** laufen, jeder mit seiner eigenen Hyperparameter-Konfiguration und Policy-Gewichten. Periodisch — alle `T` Schritte — wird die Population evaluiert und zwei Operationen werden angewendet.

### Die zwei Operationen: Exploit und Explore

**Exploit** — Gewichte von einem Top-Performer kopieren.

Jeder Agent wird nach seiner jüngsten Belohnung gerankt. Die unteren 20–25 % der Population schauen auf die oberen 20–25 % und kopieren deren neuronale Netzwerk-Gewichte direkt. Der verlierende Agent fährt nicht mit seinen alten Gewichten fort — er erbt die Gewichte des Siegers und macht von dort weiter.

Das ist der Schlüssel-Mechanismus, der PBT fundamental anders macht als unabhängige parallele Läufe: Agenten können Lern-Fortschritt von besser-performenden Peers *erben*.

**Explore** — die geerbten Hyperparameter perturbieren.

Nach dem Kopieren der Gewichte perturbiert der Agent die geerbten Hyperparameter, indem er jeden mit einem zufälligen Faktor aus `{0.8, 1.2}` (ein üblicher Default) multipliziert und dann Werte auf die Suchraum-Grenzen klemmt. Der neue Agent setzt das Training mit denselben Gewichten, aber modifizierten Hyperparametern fort.

Das stellt sicher, dass die Population nie auf eine einzelne Konfiguration kollabiert: jeder Exploit-Schritt wird unmittelbar von Exploration gefolgt.

### Synchrones vs asynchrones PBT

| Variante | Wie es funktioniert | Trade-off |
|----------|---------------------|-----------|
| **Synchron** | Alle Agenten pausieren bei Schritt T; Exploit/Explore passiert; alle setzen fort | Saubere Vergleiche; GPU sitzt während der Synchronisation untätig |
| **Asynchron** | Jeder Agent triggert Exploit/Explore unabhängig, wenn er T Schritte erreicht | Bessere Hardware-Auslastung; Vergleich ist approximativ (nicht alle am selben Schritt) |

Ray Tunes `PopulationBasedTraining`-Scheduler nutzt standardmäßig die asynchrone Variante, weshalb er für Situationen, in denen Agenten unterschiedliche Wall-Clock-Geschwindigkeiten haben, gut geeignet ist.

### Ein konkretes Beispiel mit N=4

```
Initial population (step 0):
  Agent A: lr=1e-3,  ent=0.02,  reward=—
  Agent B: lr=3e-4,  ent=0.01,  reward=—
  Agent C: lr=5e-4,  ent=0.05,  reward=—
  Agent D: lr=2e-4,  ent=0.005, reward=—

After T=200k steps, evaluate:
  Agent A: reward=320  ← top 25%
  Agent B: reward=280
  Agent C: reward=175  ← bottom 25%
  Agent D: reward=260

Exploit: Agent C copies Agent A's weights.
Explore: Agent C perturbs lr: 1e-3 × 1.2 = 1.2e-3, ent: 0.02 × 0.8 = 0.016

New population:
  Agent A: lr=1e-3,   ent=0.02   (unchanged)
  Agent B: lr=3e-4,   ent=0.01   (unchanged)
  Agent C: lr=1.2e-3, ent=0.016  (weights from A, hyperparams perturbed)
  Agent D: lr=2e-4,   ent=0.005  (unchanged)
```

Nach weiteren `T` Schritten wiederholt sich der Zyklus. Gewinnende Hyperparameter-Regionen bekommen mehr Exploration um sie herum; verlierende Regionen werden aufgegeben.

### Populationsgröße N

N = 4 ist die minimale tragfähige Population für sinnvolle Selektion (mindestens ein Sieger und ein Verlierer pro Zyklus). N = 8 ist der praktische Sweet Spot, der Diversität gegen Compute-Kosten balanciert. N = 16+ ist für große Experimente, wo du einen Cluster verfügbar hast.

!!! warning "N=2 ist nicht PBT"
    Mit nur zwei Agenten killt jeder Zyklus den Verlierer und ersetzt ihn durch eine Kopie des Siegers. Das ist äquivalent zu Restart-mit-Perturbation, nicht Populations-Selektion. Sinnvolle Diversität entsteht ab N ≥ 4.

---

## 4 · Ray-Tune-Integration

Ray Tune ist die Standard-Python-Bibliothek für verteilte Hyperparameter-Suche und PBT. Sie wrappt jede Trainingsfunktion, verwaltet Trials und handhabt Early Stopping.

### Installation

```bash
pip install "ray[tune]"
```

Wenn du auf einer Maschine mit mehreren GPUs bist oder das volle Dashboard willst:

```bash
pip install "ray[tune]" tensorboard
```

### SB3 in Ray Tune verdrahten

Ray Tune erwartet eine Trainingsfunktion mit Signatur `train_fn(config: dict)`. Das `config`-Dict hält die Hyperparameter für diesen Trial. Die Funktion sollte `ray.air.session.report(metrics)` periodisch aufrufen, damit Ray Trials vergleichen und Exploit/Explore planen kann.

```python
import ray
from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import numpy as np


def train_pbt(config: dict):
    """One PBT trial. Ray Tune calls this in a separate process per agent."""

    env = StableBaselinesGodotEnv(
        env_path="./FlyBy.x86_64",
        n_parallel=4,
        speedup=20,
        show_window=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=config["learning_rate"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        gamma=config["gamma"],
        n_steps=config["n_steps"],
    )

    # If Ray Tune passed checkpoint weights from an exploit step, load them.
    checkpoint = air.session.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            model.set_parameters(f"{checkpoint_dir}/model")

    # Train for one reporting interval, then report metrics.
    STEPS_PER_REPORT = 50_000
    for _ in range(10):  # 10 × 50k = 500k total steps per trial
        model.learn(total_timesteps=STEPS_PER_REPORT, reset_num_timesteps=False)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, warn=False)

        # Save weights so Ray can copy them during exploit.
        with tune.checkpoint_dir(step=model.num_timesteps) as checkpoint_dir:
            model.save(f"{checkpoint_dir}/model")

        air.session.report(
            {"mean_reward": mean_reward, "timesteps": model.num_timesteps},
        )

    env.close()
```

### Der PBT-Scheduler

```python
pbt_scheduler = PopulationBasedTraining(
    time_attr="timesteps",               # the metric Ray uses to sync the population
    metric="mean_reward",                # what to maximise
    mode="max",
    perturbation_interval=100_000,       # exploit/explore every 100k steps
    hyperparam_mutations={
        # Ray will perturb these by ×0.8 or ×1.2
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "clip_range":    tune.uniform(0.1, 0.4),
        "ent_coef":      tune.loguniform(1e-4, 0.1),
        "gamma":         tune.uniform(0.95, 0.999),
        "n_steps":       [512, 1024, 2048],
    },
)
```

### Die Population starten

```python
ray.init()

initial_config = {
    "learning_rate": tune.loguniform(1e-5, 1e-3),
    "clip_range":    tune.uniform(0.1, 0.4),
    "ent_coef":      tune.loguniform(1e-4, 0.1),
    "gamma":         tune.uniform(0.95, 0.999),
    "n_steps":       tune.choice([512, 1024, 2048]),
}

tuner = tune.Tuner(
    train_pbt,
    tune_config=tune.TuneConfig(
        scheduler=pbt_scheduler,
        num_samples=8,          # population size N=8
    ),
    param_space=initial_config,
    run_config=air.RunConfig(
        name="pbt_flyby",
        local_dir="./ray_results",
        stop={"timesteps": 500_000},
    ),
)

results = tuner.fit()
best = results.get_best_result(metric="mean_reward", mode="max")
print("Best config:", best.config)
print("Best reward:", best.metrics["mean_reward"])
```

### Suchraum für PPO und SAC

| Hyperparameter | PPO-Range | SAC-Range | Notizen |
|----------------|-----------|-----------|---------|
| `learning_rate` | `[1e-5, 1e-3]` log-uniform | `[1e-5, 1e-3]` log-uniform | Wirkungsvollster Parameter in beiden Algorithmen |
| `clip_range` | `[0.1, 0.4]` | N/A | PPO-spezifisch; kontrolliert Policy-Update-Größe |
| `ent_coef` | `[1e-4, 0.1]` log-uniform | nutze `"auto"` | SAC tunt das automatisch; für PPO einbeziehen |
| `gamma` | `[0.95, 0.999]` | `[0.95, 0.999]` | Sehr task-abhängig; nahe 1,0 für lange Horizonte |
| `n_steps` | `{512, 1024, 2048}` | N/A | PPO-Rollout-Länge; diskrete Menge funktioniert besser als kontinuierlich |
| `batch_size` | `{64, 128, 256}` | `{128, 256, 512}` | GPU-abhängig; oft nicht zuerst lohnenswert zu tunen |
| `tau` | N/A | `[0.001, 0.05]` | SAC-Target-Netzwerk-Update-Rate |

!!! tip "Starte mit weniger Parametern"
    In der Praxis verlangsamt das Mutieren von mehr als 3–4 Hyperparametern gleichzeitig die Konvergenz — der Explore-Schritt nimmt den Agenten zu weit von einer funktionierenden Konfiguration weg. Starte mit `learning_rate` + `ent_coef` + `clip_range` (PPO) oder `learning_rate` + `gamma` (SAC) und füge mehr nur hinzu, wenn der initiale Lauf plateaut.

---

## 5 · Godot + PBT — mehrere Instanzen laufen lassen

PBT erfordert, N Agenten simultan laufen zu lassen, jeden mit seiner eigenen Godot-Umgebung. Das bedeutet N separate Godot-Prozesse auf deiner Maschine (oder über einen Cluster).

### Architektur-Übersicht

```
Ray Tune orchestrator (Python)
    │
    ├── Trial 0 ──► PPO process ──► StableBaselinesGodotEnv ──► FlyBy.x86_64 (port 11000)
    ├── Trial 1 ──► PPO process ──► StableBaselinesGodotEnv ──► FlyBy.x86_64 (port 11001)
    ├── Trial 2 ──► PPO process ──► StableBaselinesGodotEnv ──► FlyBy.x86_64 (port 11002)
    ├── Trial 3 ──► PPO process ──► StableBaselinesGodotEnv ──► FlyBy.x86_64 (port 11003)
    ├── Trial 4 ──► PPO process ──► StableBaselinesGodotEnv ──► FlyBy.x86_64 (port 11004)
    ├── Trial 5 ──► PPO process ──► StableBaselinesGodotEnv ──► FlyBy.x86_64 (port 11005)
    ├── Trial 6 ──► PPO process ──► StableBaselinesGodotEnv ──► FlyBy.x86_64 (port 11006)
    └── Trial 7 ──► PPO process ──► StableBaselinesGodotEnv ──► FlyBy.x86_64 (port 11007)
```

Jeder Godot-Prozess lauscht auf einem einzigartigen Port. Der `StableBaselinesGodotEnv`-Wrapper akzeptiert ein `port`-Argument; Ray Tunes Trial-Index liefert eine natürliche eindeutige ID.

```python
def train_pbt(config: dict):
    # Use Ray's trial ID to assign a unique port.
    trial_id = int(tune.get_trial_id().split("_")[-1])
    port = 11000 + trial_id

    env = StableBaselinesGodotEnv(
        env_path="./FlyBy.x86_64",
        port=port,
        n_parallel=2,       # 2 parallel envs per trial — keep total manageable
        speedup=20,
        show_window=False,
    )
    # ... rest of training function
```

### Ressourcen-Allokation

Jeder Godot-Prozess verbraucht CPU. Mit N=8 Trials und 2 parallelen Envs pro Trial fährst du 16 Godot-Prozesse gleichzeitig. Auf einer 16-Core-Maschine saturiert das die CPU. Nutze Rays Resource-Hints, um Übersubskription zu verhindern:

```python
tuner = tune.Tuner(
    tune.with_resources(train_pbt, resources={"cpu": 2}),   # 2 cores per trial
    tune_config=tune.TuneConfig(
        scheduler=pbt_scheduler,
        num_samples=8,
    ),
    # ...
)
```

!!! warning "GPU-Speicher mit PBT"
    N=8 PPO-Modelle simultan im Speicher können VRAM auf kleineren Karten erschöpfen. PPOs MLP-Policy ist klein (< 50 MB jeweils), aber N=8 summiert sich. Wenn du CUDA-Out-of-Memory-Fehler siehst, reduziere `num_samples` auf 4–6 oder fahre Trials auf CPU (`device="cpu"` im PPO-Konstruktor). PBTs Nutzen kommt aus dem Selektions-Mechanismus, nicht aus GPU-Durchsatz.

### Headless Godot auf Linux

Alle Godot-Instanzen sollten headless laufen (kein Display), um nicht um den Bildschirm zu kämpfen. Exportiere dein Projekt mit dem `--headless`-Flag:

```bash
./FlyBy.x86_64 --headless --port 11000 &
```

Der `StableBaselinesGodotEnv`-Wrapper handhabt das automatisch, wenn `show_window=False`.

---

## 6 · Was mit PBT tunen

Nicht alle Hyperparameter sind es gleichermaßen wert, in den PBT-Mutations-Satz aufgenommen zu werden. Die Tabelle unten reflektiert typische Erfahrung über Godot-Umgebungen mit PPO.

| Hyperparameter | Mutations-Priorität | Warum |
|----------------|---------------------|-------|
| `learning_rate` | **Hoch** | Der wirkungsvollste Einzelparameter. Der optimale Wert ändert sich tatsächlich während des Trainings — früh hoch, spät niedriger. |
| `ent_coef` | **Hoch** | Kontrolliert Explorations-Druck. Hoch früh im Training treibt Exploration; niedriger später, während die Policy konvergiert. PBT entdeckt diesen Schedule natürlich. |
| `clip_range` | **Mittel** | Größere Werte erlauben schnelleres Lernen früh; kleinere Werte geben Stabilität spät. Das über das Training zu adaptieren ist ein bekannter PBT-Gewinn. |
| `gamma` | **Mittel** | Task-abhängig. Wenn Episoden stark variable Längen haben, beeinflusst Gamma Credit Assignment signifikant. |
| `n_steps` | **Niedrig** | Eine diskrete Wahl, die die Rollout-Länge setzt. Sie mitten im Training zu ändern kann den Advantage-Estimator stören. Sicherer fix zu lassen, außer du hast einen spezifischen Grund. |
| `batch_size` | **Niedrig** | Hardware-beschränkter als Task-beschränkter. Üblicherweise nicht lohnenswert zu mutieren. |
| `n_epochs` | **Niedrig** | Selten vorteilhaft nach initialem Tuning zu ändern. |

### Praktischer Startpunkt

Für die meisten Godot-Tasks, starte mit diesem minimalen Mutations-Satz:

```python
hyperparam_mutations={
    "learning_rate": tune.loguniform(1e-5, 1e-3),
    "ent_coef":      tune.loguniform(1e-4, 0.1),
    "clip_range":    tune.uniform(0.1, 0.4),
}
```

Füge `gamma` hinzu, wenn der Task lange Episoden involviert (> 1000 Schritte) oder wenn du schlechtes Credit Assignment siehst (der Agent lernt, was am Ende von Episoden zu tun ist, aber nicht am Anfang).

---

## 7 · Ergebnis-Analyse — Hyperparameter-Trajektorien-Plots

Die definierende Visualisierung für PBT ist die **Hyperparameter-Trajektorie**: eine Linie pro Agent, die zeigt, wie jeder Hyperparameter sich über das Training entwickelt. Statische Methoden produzieren flache horizontale Linien; PBT produziert dynamische Kurven.

### Trajektorien aus Ray-Tune-Ergebnissen extrahieren

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the experiment results directory produced by Ray Tune
results_df = results.get_dataframe()

# PBT logs hyperparameter values alongside metrics at each reporting step
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for trial_id in results_df["trial_id"].unique():
    trial_data = results_df[results_df["trial_id"] == trial_id]

    axes[0].plot(trial_data["timesteps"], trial_data["config/learning_rate"], alpha=0.7)
    axes[1].plot(trial_data["timesteps"], trial_data["config/ent_coef"], alpha=0.7)
    axes[2].plot(trial_data["timesteps"], trial_data["config/clip_range"], alpha=0.7)

axes[0].set_ylabel("learning_rate")
axes[0].set_yscale("log")
axes[1].set_ylabel("ent_coef")
axes[1].set_yscale("log")
axes[2].set_ylabel("clip_range")
axes[2].set_xlabel("Environment steps")

fig.suptitle("PBT Hyperparameter Trajectories — FlyBy PPO (N=8)")
plt.tight_layout()
plt.savefig("pbt_trajectories.png", dpi=150)
```

### Worauf im Trajektorien-Plot zu achten ist

Ein gut funktionierender PBT-Lauf zeigt mehrere charakteristische Muster:

- **Learning-Rate-Decay:** Agenten, die am längsten überleben, neigen dazu, eine `learning_rate` zu haben, die über das Training nach unten perturbiert wurde. Das bestätigt, dass PBT den wohlbekannten „start high, end low"-Learning-Rate-Schedule automatisch entdeckt, ohne dass du ihn spezifizierst.
- **Entropy-Coef-Konvergenz:** `ent_coef`-Trajektorien konvergieren oft zu einer gemeinsamen Region, während das Training voranschreitet — die Überlebenden haben das richtige Explorations-Level für den Task entdeckt.
- **Exploit-Events als Diskontinuitäten:** wenn ein Agent Gewichte von einem Top-Performer kopiert, springen seine Hyperparameter-Werte diskontinuierlich. Diese Sprünge sind als vertikale Stufen in den Trajektorien-Linien sichtbar und zeigen, dass der Exploit-Explore-Mechanismus feuert.
- **Geprunte Trials:** Agenten, die früh gekillt wurden (via Successive Halving oder andere Mechanismen), erscheinen als kurze Linien, die vor dem Ende des Trainings terminieren.

Wenn alle Trajektorien flach sind und nie Diskontinuitäten zeigen, triggert der Exploit-Schritt nicht — prüfe, dass `perturbation_interval` klein genug relativ zu deinem totalen Trainings-Budget ist und dass `metric` und `mode` korrekt gesetzt sind.

---

## 8 · Vergleich: PBT vs Optuna

Das ist der praktisch wichtigste Abschnitt in der Unit. Die ehrliche Antwort ist, dass jede Methode in anderen Umständen gewinnt.

### Direkter Vergleich

| Dimension | Optuna | PBT |
|-----------|--------|-----|
| **Hyperparameter-Schedule** | Statisch — ein Wert für den gesamten Lauf | Dynamisch — adaptiert im Training |
| **Parallelität** | Embarrassingly parallel über Trials | Erfordert Kommunikation zwischen Trials (Exploit-Schritt) |
| **Overhead** | Niedrig — Standard-Python, keine Ray-Dependency | Höher — Ray-Cluster-Setup, Inter-Prozess-Kommunikation |
| **Minimal nützliche Läufe** | 10–20 Trials | 4–8 Agenten (N ≥ 4) |
| **Am besten für** | Kurze-bis-mittlere Läufe (< 1M Schritte), schnelle Simulatoren | Lange Läufe (> 2M Schritte), teure Simulatoren |
| **Ergebnis-Interpretierbarkeit** | Klar: beste statische Config + Importance-Scores | Komplex: Schedules, nicht eine einzelne Config |
| **Wiederverwendbarkeit** | Beste Config ist fix, einfach auf neue Läufe anwendbar | Schedules transferieren nicht — PBT muss neu gelaufen werden |
| **Early Stopping** | Pruners (Median, Hyperband) killen schlechte Trials früh | Untere 25 % durch Exploit ersetzt — inhärentes Early Stopping |
| **Implementations-Aufwand** | Niedrig — 20 Zeilen Optuna-Code | Mittel — Ray-Tune-Setup + Reporting-Loop |

### Wann Optuna gewinnt

- **Godot-Umgebungen mit schneller Simulation** (BallChase, CrossTheRoad, JumperHard mit 20×-Speedup). Diese erledigen 1M Schritte in Minuten. Optunas 30–50 Trials sind unter zwei Stunden fertig, und der Overhead von PBTs Exploit/Explore-Mechanismus ist nicht gerechtfertigt.
- **Du willst eine übertragbare Konfiguration.** Optuna gibt dir ein einzelnes Dict mit Hyperparametern, das du in jedes Trainings-Skript einfügen kannst. PBT gibt dir einen Schedule, der für einen spezifischen Task gelernt wurde und für jede neue Umgebung neu gelernt werden muss.
- **Begrenzte Hardware.** 8 parallele Godot-Instanzen zu fahren erfordert eine Maschine mit mindestens 8 freien Cores und genug RAM für alle Prozesse simultan. Ein Laptop mit 4 Cores ist keine gute PBT-Maschine.
- **Diagnose-Phase einer neuen Umgebung.** Während du noch herausfindest, ob eine Belohnungsfunktion überhaupt funktioniert, sind Optunas sequenzielle Trials einfacher zu debuggen. PBTs Parallelität macht es schwerer zu isolieren, welche Konfiguration ein spezifisches Verhalten verursachte.

### Wann PBT gewinnt

- **Lange Trainingsläufe (> 2M Schritte)**. Der dynamische Scheduling-Vorteil kompoundiert über lange Läufe. Bei 500k Schritten ist die Lücke zwischen statischen und dynamischen Hyperparametern klein; bei 5M Schritten kann sie substanziell sein.
- **Teure Simulatoren, wo Re-Runs schmerzen.** Wenn ein Trainingslauf 6 Stunden dauert, sind 30 Optuna-Trials prohibitiv. PBT extrahiert mehr Signal aus einem einzelnen parallelen Lauf.
- **Tasks, in denen die optimale Learning Rate bekanntermaßen abnimmt.** MuJoCo-Continuous-Control-Benchmarks zeigen konsistente PBT-Siege, gerade weil Learning-Rate-Decay gut motiviert ist und PBT ihn automatisch entdeckt.
- **Du hast Ray-Infrastruktur bereits.** Wenn dein Team Ray für andere Zwecke nutzt (verteilte Datenverarbeitung, RLlib), hat das Integrieren von PBT nahezu null zusätzlichen Overhead.

### Praktische Empfehlung

!!! tip "Die Entscheidungsregel"
    **Starte mit Optuna.** Lass 30–50 Trials mit dem Optuna-Snippet aus Abschnitt 2 laufen. Das gibt dir eine solide statische Baseline in wenigen Stunden auf jedem modernen Laptop. Wenn diese Baseline gut genug für deinen Task ist, stoppe — du bist fertig.

    **Steige zu PBT auf, wenn** entweder (a) dein Trainingslauf länger als 2M Schritte ist *und* du Hardware für N ≥ 4 parallele Prozesse hast, oder (b) du bereits eine gute statische Config mit Optuna gefunden hast und die letzten paar Prozent Performance mit adaptivem Scheduling herauskitzeln willst.

    Skippe Optuna nicht und springe direkt zu PBT. Die Optuna-Baseline ist sowohl nützlich (du lernst, welcher Hyperparameter-Bereich funktioniert) als auch notwendig (PBTs initiale Population sollte aus einem vernünftigen Bereich geseedet werden, nicht aus einem blinden Prior).

---

## 9 · Stretch Goals

- **Den Learning-Rate-Decay reproduzieren.** Lass PBT auf FlyBy mit `learning_rate` als einzigem mutiertem Parameter laufen. Plotte die Trajektorien für die top 3 überlebenden Agenten. Bestätige, dass Überlebende das Training mit niedrigeren Learning Rates beenden, als sie gestartet haben. Vergleiche ihre finale Belohnung mit einem Optuna-Lauf mit der besten statischen `learning_rate` aus demselben Suchbereich.

- **PBT aus Optuna-Ergebnissen seeden.** Lass zuerst 20 Optuna-Trials laufen. Nutze die top-5-Configs als initiale PBT-Population (statt zufällig aus dem Prior zu samplen). Beschleunigt Warm-Starting die PBT-Konvergenz? Miss Time-to-Threshold (Schritte bis `mean_reward > target`) für Cold-Start-PBT vs Optuna-geseedetes PBT.

- **PBT auf SAC.** Ersetze PPO durch SAC in der `train_pbt`-Funktion. Mutiere `learning_rate` und `gamma` (SACs `ent_coef` wird auto-getuned und sollte als `"auto"` belassen werden). Der Schlüsselunterschied: SAC mit Replay-Buffer resettet nicht sauber zwischen Exploit-Schritten — der Buffer enthält Transitionen aus dem alten Hyperparameter-Regime. Beobachte, ob das Instabilität verursacht, und versuche, falls ja, den Replay-Buffer nach jedem Exploit-Schritt zu leeren.

- **Lies das originale PBT-Paper.** Jaderberg et al. 2017, "Population Based Training of Neural Networks." Frei verfügbar auf arXiv. Abschnitt 3 (der Algorithmus) und Abschnitt 4.2 (die Atari-Ergebnisse) sind die relevantesten für diesen Kurs. Das Headline-Ergebnis — PBT auf Atari matcht den besten handgetunten Hyperparameter-Schedule — ist, was die Methode motivierte.

- **Vergleiche Wall-Clock-Zeit, nicht nur Schritte.** Lass Optuna (30 Trials, 500k Schritte je, sequenziell) und PBT (N=8, 500k Schritte pro Agent, parallel) laufen und time beides mit `time.time()`. Auf einer Maschine, in der alle 8 PBT-Agenten parallel passen, sollte PBT in ungefähr derselben Wall-Clock-Zeit wie 8 Optuna-Trials fertig werden — nicht 30. Berechne das Wall-Clock-Effizienz-Verhältnis.

---

## Was kommt als Nächstes

Population-Based Training ist die Spitze des Hyperparameter-Optimierungs-Stacks, der in diesem Kurs behandelt wird. Du hast nun drei Tool-Level: manuelles Tuning (Trial and Error), statische Suche (Optuna) und dynamische adaptive Suche (PBT). Das richtige Level hängt von deinem Task, deiner Hardware und davon ab, wie sehr der Hyperparameter-Schedule für dein spezifisches Problem zählt.

Für die meisten Godot-Projekte ist Optuna mit 30 Trials die praktische Decke — schnell zu laufen, einfach zu interpretieren und portabel über Umgebungen. Reserviere PBT für lange Läufe, wo die dynamische Scheduling-Dividende groß genug ist, um den Setup-Aufwand zu rechtfertigen.

[Kursstartseite](index.md)
