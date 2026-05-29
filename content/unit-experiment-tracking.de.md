# Experiment-Tracking — W&B, MLflow und Hyperparameter-Sweeps

[Kursstartseite](index.md) · [← Debugging](unit-debugging.md) · [→ Fortgeschrittene Evaluation](unit-evaluation.md)

!!! info "Zeit"
    Lesen: ~25 min

---

## 1 · Warum TensorBoard im Maßstab nicht reicht

TensorBoard ist der Standard-Logger für Stable-Baselines3 und funktioniert gut — bis du anfängst, mehr als zwei oder drei Läufe zu vergleichen. Sobald du zehn oder mehr Hyperparameter-Sweeps parallel laufen lässt, werden seine Limitierungen zu Blockern:

- **Kein Config-Logging.** TensorBoard erfasst Kurven, nicht die Hyperparameter, die sie produziert haben. Nach einer Woche Experimenten hast du 20 Belohnungskurven und keine verlässliche Möglichkeit zu wissen, welche Learning Rate, `n_steps` oder welcher Entropy-Coefficient zu welcher Kurve gehört.
- **Kein Artifact-Tracking.** Es gibt keinen eingebauten Link zwischen einer Checkpoint-Datei auf der Disk und dem Trainingslauf, der sie erzeugt hat. Der Checkpoint, der dein bestes Ergebnis erzielt, kann leicht von dem Lauf, der ihn erzeugte, abgekoppelt werden.
- **Vergleich ist manuell und fragil.** 10+ Läufe in TensorBoard zu überlagern erfordert sorgfältige Verzeichnis-Benennung und produziert dennoch eine überfüllte, schwer teilbare HTML-Seite.
- **Kollaboration ist mühsam.** Ergebnisse mit einem Teammitglied oder einer Reviewerin zu teilen heißt entweder ein `logs/`-Verzeichnis zu zippen oder Server-Zugang zu gewähren. Keines skaliert.

W&B (Weights & Biases) und MLflow lösen all das. Sie loggen Hyperparameter-Configs neben Metriken, speichern und versionieren Modell-Artifacts, bieten eine gehostete (oder selbst-gehostete) Vergleichs-UI und machen Teilen zu einer permanenten URL statt einer Zip-Datei.

!!! tip "Wann wechseln"
    Läuft nur ein einzelner Trainingsjob, um zu prüfen, dass deine Belohnungsfunktion funktioniert, ist TensorBoard okay. Wechsle zu W&B oder MLflow, sobald du Hyperparameter tunst oder Algorithmus-Varianten vergleichst.

---

## 2 · Weights & Biases in 10 Minuten

### Installation und Authentifizierung

```bash
pip install wandb
wandb login   # paste your API key from wandb.ai/authorize
```

### Custom Callback

Der Code unten zeigt eine minimale W&B-Integration als standard SB3 `BaseCallback`. Sie Zeile für Zeile zu verstehen ist nützlich, bevor du zur eingebauten Integration wechselst.

```python
# pip install wandb
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        if self.n_calls % 1000 == 0:
            wandb.log({
                "rollout/ep_rew_mean": self.locals.get("infos", [{}])[0].get("episode", {}).get("r", 0),
                "train/loss": self.model.logger.name_to_value.get("train/loss", 0),
            }, step=self.num_timesteps)
        return True

wandb.init(
    project="godot-rl-course",
    config={
        "algorithm": "PPO",
        "env": "FlyBy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "total_timesteps": 1_000_000,
    }
)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/")
model.learn(total_timesteps=1_000_000, callback=WandbCallback())
wandb.finish()
```

Der `wandb.init(config=...)`-Aufruf ist die Schlüssel-Ergänzung gegenüber TensorBoard. Jeder hier übergebene Hyperparameter wird neben jeder Metrik erfasst, sodass du Läufe in der W&B-UI nach jedem Config-Schlüssel filtern und gruppieren kannst.

### Eingebaute SB3-Integration (empfohlen)

W&B liefert einen fertigen SB3-Callback, der alles automatisch loggt — Belohnung, Losses, Learning-Rate-Schedules und mehr. Nutze diesen, außer du brauchst benutzerdefinierte Metriknamen.

```python
from wandb.integration.sb3 import WandbCallback

run = wandb.init(
    project="godot-rl-course",
    config={
        "algorithm": "PPO",
        "env": "FlyBy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "total_timesteps": 1_000_000,
    },
    sync_tensorboard=True,  # mirrors all SB3 TensorBoard logs into W&B
)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"logs/{run.id}")
model.learn(total_timesteps=1_000_000, callback=WandbCallback(verbose=2))
wandb.finish()
```

`sync_tensorboard=True` heißt, du musst deine bestehenden Logging-Aufrufe nicht ändern — W&B fängt die TensorBoard-Events ab und re-indexiert sie unter dem Lauf.

!!! tip "Free Tier ist großzügig"
    W&B bietet unbegrenzte Läufe und 100 GB Artifact-Storage in seinem kostenlosen Personal-/Academic-Plan. Für ein Kursprojekt ist das mehr als genug.

---

## 3 · W&B Sweeps — Hyperparameter-Suche

Ein Sweep definiert den Suchraum und die Strategie in einer YAML-Datei und startet dann Agents (Worker-Prozesse), die jeweils eine Hyperparameter-Config ziehen, Training durchführen und Ergebnisse an den Sweep-Controller zurückmelden.

### Sweep-Config (`sweep.yaml`)

```yaml
program: train.py
method: bayes
metric:
  name: rollout/ep_rew_mean
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  n_steps:
    values: [1024, 2048, 4096]
  ent_coef:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.1
```

`method: bayes` nutzt Bayesian Optimization — es modelliert die Beziehung zwischen Hyperparametern und der Ziel-Metrik und schlägt Configs vor, die wahrscheinlich auf bereits Gesehenes verbessern. Nutze `method: random` für einen einfacheren Baseline-Sweep oder wenn deine Läufe sehr kurz sind.

### Starten

```bash
# Step 1 — register the sweep and get an ID
wandb sweep sweep.yaml

# Step 2 — start one or more agents (each runs train.py with a sampled config)
wandb agent <sweep-id>

# To run multiple agents in parallel on separate machines or tmux panes:
wandb agent <sweep-id> &
wandb agent <sweep-id> &
```

Deine `train.py` sollte Hyperparameter aus `wandb.config` lesen, damit Agents die gesampelten Werte aufnehmen:

```python
import wandb

wandb.init()  # sweep agent populates wandb.config automatically
cfg = wandb.config

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=cfg.learning_rate,
    n_steps=cfg.n_steps,
    ent_coef=cfg.ent_coef,
)
model.learn(total_timesteps=500_000, callback=WandbCallback(verbose=2))
```

!!! warning "Sweep-Agents blockieren bis zum Lauf-Ende"
    Jeder Agent läuft eine Config zur Zeit. Für Godot-Umgebungen mit langsamer Physik setze `speedup` hoch (32–64×) und reduziere `total_timesteps` in Sweep-Läufen — du willst genug Signal, um Configs zu ranken, nicht einen vollen Produktions-Trainingslauf.

---

## 4 · Was über `ep_rew_mean` hinaus zu loggen ist

Episodenbelohnung ist notwendig, aber nicht ausreichend zum Diagnostizieren von Training. Die Tabelle unten listet Metriken, die Probleme aufdecken, die Belohnung allein nicht offenlegen kann.

| Metrik | Was sie dir sagt | Wie loggen |
|--------|------------------|------------|
| KL-Divergenz | Policy-Stabilität — große KL heißt, Updates sind zu aggressiv | SB3 loggt `train/approx_kl` automatisch |
| Gradient-Norm | Explodierende Gradienten | Logge `train/explained_variance` als Proxy; niedrig und sinkend heißt, der Critic lernt nicht |
| Episodenlängen-Verteilung | Terminieren Episoden korrekt? Kurze Episoden heißen evtl., dass der Agent unerwartet stirbt oder resettet | Als Histogramm loggen: `wandb.log({"ep_len": wandb.Histogram(ep_lens)})` |
| Beobachtungs-Statistiken | Sind Beobachtungen im erwarteten Bereich? Out-of-Range-Obs verursachen stille Normalisierungs-Fehler | Mittelwert und Std des Obs-Buffers pro Rollout loggen |
| Aktions-Verteilungs-Entropie | Konvergiert die Policy zu schnell? Entropie-Kollaps früh im Training heißt, der Agent hört auf zu explorieren | SB3 loggt `train/entropy_loss` automatisch |

!!! tip "Histogramme in W&B"
    `wandb.Histogram` akzeptiert eine Liste oder ein numpy-Array und rendert ein interaktives Histogramm in der W&B-UI. Nutze es für Episodenlängen, Aktionsmagnituden und Beobachtungs-Channels — alles, wo die Verteilungs-Form zählt, nicht nur der Mittelwert.

---

## 5 · MLflow — selbst-gehostete Alternative

Nutze MLflow statt W&B, wenn:

- Deine Trainings-Maschinen keinen Internetzugang haben (air-gapped Labs, Cloud-VPCs mit Egress-Restriktionen).
- Dein Team Datenschutz-Anforderungen hat, die das Senden von Lauf-Daten an eine Drittanbieter-Cloud verbieten.
- Du MLflow bereits als Teil einer breiteren ML-Plattform deployed hast.

### Schnelles Setup

```bash
pip install mlflow
mlflow server --host 0.0.0.0 --port 5000   # start the tracking server
export MLFLOW_TRACKING_URI=http://localhost:5000
```

Dann öffne `http://localhost:5000` im Browser für die UI.

### SB3-Integration

```python
import mlflow
from stable_baselines3.common.callbacks import BaseCallback

class MLflowCallback(BaseCallback):
    def _on_step(self):
        if self.n_calls % 1000 == 0:
            mlflow.log_metric(
                "ep_rew_mean",
                self.locals.get("infos", [{}])[0].get("episode", {}).get("r", 0),
                step=self.num_timesteps,
            )
        return True

with mlflow.start_run():
    mlflow.log_params({"algorithm": "PPO", "lr": 3e-4, "n_steps": 2048})
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000, callback=MLflowCallback())
    mlflow.log_artifact("flyby_ppo.zip")   # save the checkpoint into the run
```

!!! warning "mlflow.log_metric ist nicht gebatcht"
    `mlflow.log_metric` jeden Schritt aufzurufen, wird dein Training langsam machen. Logge alle 1.000–5.000 Timesteps wie oben gezeigt, oder batche Metriken mit `mlflow.log_metrics(dict, step=n)`.

---

## 6 · W&B vs MLflow Vergleich

| Feature | W&B | MLflow |
|---------|-----|--------|
| Setup | 1 Kommando (`wandb login`) | Selbst-gehosteter Server |
| UI | Exzellent — reichhaltige interaktive Charts, Parallel Coordinates für Sweeps | Gut — funktional, weniger poliert |
| Kosten | Kostenlose Personal-/Academic-Tier; bezahlt für Teams | Kostenlos (selbst-host); verwaltete Tiers verfügbar |
| Privacy | Cloud (US/EU Datenresidenz-Optionen) | On-Prem — Daten verlassen nie dein Netzwerk |
| Sweeps | Eingebaute Bayesian / Random / Grid | Optuna-Integration via `mlflow.tracking` |
| Artifact-Storage | W&B Artifact Registry | MLflow Artifact Store (S3, GCS, local FS) |
| Am besten für | Einzelne Forscher, akademische Projekte | Teams mit bestehender Infra oder Privacy-Anforderungen |

Für diesen Kurs ist W&B der empfohlene Default. Wechsle zu MLflow, wenn du auf eine Privacy- oder Konnektivitäts-Einschränkung triffst.

---

## 7 · Artifact-Tracking

Metriken zu loggen ist nur die halbe Geschichte. Ohne Checkpoints an Läufe zu binden, kannst du ein Ergebnis sechs Monate später nicht reproduzieren — du hast Kurven, aber keine Policy.

### Checkpoints zu W&B speichern

```python
# After training finishes
model.save("flyby_ppo")               # writes flyby_ppo.zip
wandb.save("flyby_ppo.zip")           # uploads zip to W&B artifacts, linked to this run
```

### In Intervallen speichern mit einem Checkpoint-Callback

```python
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_cb = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints/",
    name_prefix="flyby_ppo",
)

model.learn(
    total_timesteps=1_000_000,
    callback=[WandbCallback(verbose=2), checkpoint_cb],
)

# Upload all checkpoints as a versioned artifact
artifact = wandb.Artifact("flyby-checkpoints", type="model")
artifact.add_dir("./checkpoints/")
wandb.log_artifact(artifact)
```

### Ein Artifact später erneut laden

```python
run = wandb.init(project="godot-rl-course")
artifact = run.use_artifact("flyby-checkpoints:v3", type="model")
artifact_dir = artifact.download()
model = PPO.load(f"{artifact_dir}/flyby_ppo_1000000_steps.zip", env=env)
```

Das Suffix `:v3` pinnt eine exakte Version. Artifacts sind immutable — das Hochladen eines neuen Checkpoints erzeugt eine neue Version, statt die alte zu überschreiben.

!!! tip "Warum das wichtig ist"
    Es ist üblich, einen Benchmark zu schlagen, weiterzuziehen und dann diese Policy Monate später für ein Paper oder eine Demo neu evaluieren zu müssen. Ohne Artifact-Tracking bist du auf lokale Disk angewiesen, was fragil ist. W&B-Artifact-URLs sind permanent.

---

## 8 · Godot-spezifische Notizen

W&B und MLflow integrieren auf SB3-Level, nicht auf Umgebungs-Level. Sie funktionieren identisch, egal ob deine Umgebung ein Godot-Binary, ein Gymnasium-Wrapper oder etwas anderes ist — du musst deine Godot-Szene oder dein GDScript überhaupt nicht ändern.

Dennoch gibt es einige Godot-spezifische Parameter, die es wert sind, als Run-Config geloggt zu werden:

```python
wandb.init(
    project="godot-rl-course",
    config={
        "algorithm": "PPO",
        "env": "FlyBy",
        "env_path": "builds/FlyBy.x86_64",
        "n_parallel": 4,          # number of parallel Godot subprocesses
        "speedup": 32,            # physics speedup factor
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "total_timesteps": 1_000_000,
    }
)
```

`n_parallel` und `speedup` zu loggen erlaubt dir, Wall-Clock-Effizienz über Maschinen zu vergleichen: ein Lauf mit `n_parallel=8, speedup=64`, der die gleiche Belohnung in der halben Wall-Time erreicht, ist ein bedeutsames Ergebnis.

**Zusätzliche Metriken, die für gängige Algorithmen explizit zu loggen sind:**

- **DQN:** `rollout/exploration_rate` — verfolgt, wie schnell Epsilon abklingt; Stagnation hier heißt, Exploration verläuft nicht wie erwartet.
- **SAC:** `train/ent_coef` — SAC lernt seinen Entropy-Coefficient automatisch; sein frühes Kollabieren zu beobachten kann Belohnungs-Stagnation vorhersagen, bevor sie in der Belohnungskurve erscheint.
- **PPO (Godot):** logge den Anteil truncated vs terminated Episoden, falls deine Godot-Szene `is_done` vs `is_truncated` nutzt — eine Diskrepanz hier ist eine häufige Quelle unsichtbarer Bugs.

!!! warning "Parallele Godot-Subprozesse und Logging"
    Wenn `n_parallel > 1`, gibt godot-rl-agents gebatchte `infos` zurück. Indexiere korrekt beim Extrahieren der Episodenbelohnung: `infos[0].get("episode", {}).get("r", 0)` fängt nur den ersten Subprozess. Nutze `np.mean([i.get("episode", {}).get("r", 0) for i in infos if "episode" in i])` für einen repräsentativeren Mittelwert über alle parallelen Envs.

## 9 · Stretch Goals

**Einen 3-Achsen-Sweep laufen lassen.** Nutze die W&B-Sweep-Config aus Abschnitt 3 als Startpunkt und füge dann eine dritte Achse hinzu — z. B. `n_steps ∈ {1024, 2048, 4096}`. Das gibt dir eine 3D-Parallel-Coordinates-Ansicht in der W&B-UI. Trainiere mindestens 5 Läufe pro Zelle und schreib auf, welche Achse die anderen dominiert. Es geht darum zu fühlen, wie schnell die Kosten wachsen, sobald ein Sweep mehr als 1D ist.

**Einen Lauf nur aus Artifacts reproduzieren.** Von einem alten W&B-Lauf (deinem eigenen, aus irgendeiner Unit) lade nur die Config + das Modell-Artifact und rekreiere das trainierte Modell auf einer frischen Maschine, ohne lokalen Code zu kopieren. Spiele 10 Episoden ab. Hast du dieselbe Belohnung bekommen? Wenn nicht, was fehlte im Artifact — war es das Env-Binary, ein Seed, ein Code-Commit-Hash? Patche die Lücke in deinem Logging-Template, sodass der nächste Lauf wirklich reproduzierbar ist.

**MLflow neben W&B für einen Lauf einrichten.** Verdrahte sowohl `MlflowOutputFormat` als auch `WandbCallback` in dasselbe SB3-Trainingsskript (siehe Abschnitte 2 und 5). Vergleiche die beiden UIs nebeneinander am selben Lauf. Entscheide — für dich selbst — welche du behalten würdest, wenn du dich entscheiden müsstest, und schreib auf *warum*. Die Antwort variiert je nach Team und Bedrohungsmodell; die Übung ist, sich eine eigene Meinung zu bilden.

!!! warning "Pseudocode"
    ```python
    import mlflow
    from wandb.integration.sb3 import WandbCallback
    import wandb

    wandb.init(project="godot-rl-course", sync_tensorboard=True)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("godot-rl-course")

    with mlflow.start_run():
        mlflow.log_params({"algorithm": "PPO", "env": "FlyBy"})
        model.learn(total_timesteps=200_000, callback=WandbCallback())
        mlflow.log_artifact("ppo_flyby.zip")
    wandb.finish()
    ```
