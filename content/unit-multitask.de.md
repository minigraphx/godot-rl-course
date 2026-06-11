# Multi-Task-RL — eine Policy für mehrere Aufgaben

[← Hierarchisches RL](unit-hierarchical.md) · [Kursstartseite](index.md) · [→ Imitation Learning](unit-09.md)

!!! info "Zeit"
    Lesen: ~35 min · Training: ~30 min GPU / ~2 Std CPU

---

Ein einzelner Roboterarm, der greifen, schieben und aufnehmen-und-ablegen kann. Ein Character-Controller, der normale Fortbewegung, Hindernisvermeidung und Sprints unter Zeitdruck bewältigt — alles aus einem einzigen Satz Netzgewichte. Das ist das Versprechen von Multitask-Reinforcement-Learning (multi-task reinforcement learning).

Statt N getrennte Spezialisten zu trainieren, trainierst du einen Generalisten. Gut gemacht, ist die geteilte Repräsentation dateneffizienter als jeder einzelne Spezialist, und die Policy generalisiert auf Aufgabenkombinationen, die kein Spezialist je gesehen hat.

Schlecht gemacht, kann Multitask-Training jede einzelne Aufgabe schlechter zurücklassen als eine dedizierte Single-Task-Policy. Dieser Fehlermodus hat einen Namen — **negativer Transfer** (negative transfer) — und ihn zu vermeiden ist die zentrale Fähigkeit, die diese Unit vermittelt.

**Querverweise**

- [Goal-Conditioned RL & HER](unit-her.md) — Goal Conditioning ist ein Spezialfall von
  Multitask, bei dem die „Aufgabe" die Zielposition ist. Erst dieses zu verstehen, macht
  die Generalisierung hier sauberer.
- [Hierarchisches RL](unit-hierarchical.md) — ein Manager+Worker-System ist ein anderer
  Weg, eine einzelne Low-Level-Policy über Aufgaben hinweg wiederzuverwenden; vergleiche
  die beiden Architekturen.
- [Sim-to-Real-Transfer](unit-sim-to-real.md) — Multitask-Training ist eines der stärksten
  Werkzeuge für Domain Randomization; Aufgaben mit unterschiedlicher Physik verhalten sich
  wie separate Domänen.
- [Reward Engineering](unit-reward-engineering.md) — jede Aufgabe braucht ihre eigene
  gut geformte Belohnung; schwaches Reward-Design pro Aufgabe verstärkt negativen Transfer.

!!! warning "Versuche zuerst separate Policies"
    Multi-Task-RL ist komplexer, als es aussieht. Bevor du danach greifst, frage:

    - Kann ich es mir leisten, N separate Policies zu trainieren? Wenn N ≤ 5, ist das
      meist die richtige Antwort.
    - Sind die Aufgaben ähnlich genug, um eine Repräsentation zu teilen, oder so
      unterschiedlich, dass eine Policy die andere immer kompromittiert?
    - Brauche ich wirklich Aufgabenwechsel zur Laufzeit, oder kann ich einfach einen
      anderen Checkpoint laden?

    Wenn separate Policies machbar sind und du keinen Laufzeit-Aufgabenwechsel brauchst,
    nimm sie. Multi-Task-RL zahlt sich aus, wenn N groß ist, Inferenzkosten begrenzt sind
    oder du echten Vorwärtstransfer auf noch nicht trainierte Aufgaben brauchst.

!!! info "Drei Wege, deine KI zu beobachten"
    - **Godot-Szene** — ein Aufgabenindikator in der Ecke des Viewports zeigt, welche
      Variante aktiv ist (farbiges Badge: blau = Reach, orange = Reach + Hindernisse,
      rot = zeitlich begrenzter Sprint); beobachte denselben Agenten, wie er seinen
      Bewegungsstil an jedes Badge anpasst.
    - **TensorBoard** — getrennte Kurven `reward/task_0`, `reward/task_1`, `reward/task_2`
      lassen dich den Moment erkennen, in dem negativer Transfer einsetzt (eine Kurve
      fällt, während eine andere steigt).
    - **Erfolgsraten-Tabelle pro Aufgabe** — nach dem Training 100 Evaluationsepisoden
      pro Aufgabe ausführen und ein `3 × 1`-Balkendiagramm anzeigen; eine gut trainierte
      Multitask-Policy sollte auf jedem Balken ≥ 80 % erreichen.

---

## 1 · Jenseits von Goal Conditioning

Goal-Conditioned RL (behandelt in [unit-her.md](unit-her.md)) generalisiert über *Ziele*
innerhalb einer einzigen Aufgabenstruktur. Die Physik, die Belohnungsfunktion und die
erforderlichen motorischen Fähigkeiten sind fest — nur die Zielposition ändert sich.

Multi-Task-RL generalisiert über *Aufgaben selbst*. Aufgaben können sich unterscheiden in:

- **Belohnungsfunktion** — das Ziel erreichen (dichte Distanzbelohnung) vs. Hindernissen
  ausweichen (Strafe bei Kollision) vs. innerhalb eines Zeitlimits fertig werden
  (zeitbasierter Bonus).
- **Physik** — ein Agent, der gleichzeitig in niedriger und hoher Schwerkraft trainiert
  wird, erwirbt eine robustere Fortbewegungsfähigkeit als einer, der nur in einer von
  beiden trainiert wird.
- **Erforderliche Fähigkeiten** — „vorwärts bewegen" erfordert nichts Besonderes; „den
  Würfel aufheben" erfordert Greifen; „die Tür öffnen" erfordert Drehmoment. Eine
  gemeinsame Policy muss alle drei motorischen Subprogramme enthalten.

### Taxonomie

| Typ | Was variiert | Wie GCRL? |
|---|---|---|
| Goal-Conditioned RL | Zielposition / Objekt | Ja — Aufgaben sind bis aufs Ziel identisch |
| Multi-Task-RL (gleiche Struktur) | Belohnungsfunktions-Gewichte | Nah — ein MDP, variierende Ziele |
| Multi-Task-RL (unterschiedliche Physik) | Übergangsdynamik | Nein — verschiedene MDPs |
| Multi-Task-RL (unterschiedliche Fähigkeiten) | Aktionsraum-Anforderungen | Nein — qualitativ unterschiedliches Verhalten |

Je weiter unten in dieser Tabelle, desto schwieriger wird Multitask und desto
wahrscheinlicher tritt negativer Transfer auf.

### Die zentrale strukturelle Änderung

Im Single-Task-RL ist die Policy `π(a | s)`. Im Multi-Task-RL ist die Policy
`π(a | s, z)`, wobei `z` ein **Task-Encoding** ist — zusätzliche Information, mit der
die Policy weiß, welche Aufgabe sie gerade löst. Die richtige Form für `z` zu wählen
ist die erste architektonische Entscheidung, die diese Unit behandelt.

---

## 2 · Aufgabenrepräsentation

Wie du `z` codierst, bestimmt, wie viel die Policy generalisieren kann. Drei
Hauptansätze:

### One-Hot-Aufgaben-ID

Die einfachste Variante. Bei N Aufgaben ist die Aufgaben-ID ein N-dimensionaler
Binärvektor mit genau einer `1`:

```
task 0 (reach goal):          z = [1, 0, 0]
task 1 (avoid + reach):       z = [0, 1, 0]
task 2 (timed reach):         z = [0, 0, 1]
```

**Vorteile**: trivial einfach; kein Training nötig; die Policy kann im Prinzip komplett
unterschiedliches Verhalten pro Aufgabe lernen, indem sie auf dieses Signal konditioniert.

**Nachteile**: Aufgaben werden als kategorisch verschieden behandelt, ohne Begriff von
Ähnlichkeit. Eine auf 3 Aufgaben trainierte Policy kann nicht mit `z = [0.5, 0.5, 0]`
zwischen ihnen interpoliert werden. Generalisiert nicht auf ungesehene Aufgaben-IDs.

### Aufgabenparameter-Vektor

Codiere jede Aufgabe als ihre wesentlichen numerischen Parameter:

```
task 0: z = [reward_weight_distance=1.0, obstacle_penalty=0.0, time_bonus=0.0]
task 1: z = [reward_weight_distance=1.0, obstacle_penalty=0.5, time_bonus=0.0]
task 2: z = [reward_weight_distance=1.0, obstacle_penalty=0.0, time_bonus=2.0]
```

**Vorteile**: Die Policy kann auf ungesehene Parameterkombinationen (z. B.
`obstacle_penalty=0.3`) ohne erneutes Training generalisieren. Glatte Interpolation
zwischen Aufgaben ist möglich.

**Nachteile**: Du musst die Aufgabenparameter identifizieren und zugänglich machen, was
nicht immer möglich ist.

### Natürlichsprachliches Embedding

Codiere Aufgabenbeschreibungen mit einem vortrainierten Sprachmodell als dichte Vektoren:

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("all-MiniLM-L6-v2")

task_descriptions = [
    "reach the goal as fast as possible",
    "reach the goal while avoiding red obstacles",
    "reach the goal before the timer runs out",
]
z_vectors = encoder.encode(task_descriptions)  # shape: (3, 384)
```

**Vorteile**: das ausdrucksstärkste Encoding; ermöglicht Instruction Following und
Generalisierung auf völlig neue Aufgabenbeschreibungen zur Inferenzzeit. Brücke zu
großen Sprachmodellen.

**Nachteile**: hochdimensional (384+); rechenintensiv; das Sprachmodell muss
eingefroren bleiben oder sorgfältig mittrainiert werden.

### Vergleichstabelle

| Encoding | Dimensionen | Generalisiert auf ungesehene Aufgaben | Implementierungsaufwand |
|---|---|---|---|
| One-Hot-ID | N (Anzahl Aufgaben) | Nein | Trivial |
| Aufgabenparameter | K (Anzahl Parameter) | Ja (Interpolation) | Niedrig |
| Sprach-Embedding | 384–768 | Ja (Zero-Shot-Text) | Hoch |

**Kursempfehlung**: Starte für die Hands-on-Übung mit One-Hot. Wechsle zu
Aufgabenparametern, sobald dein Multi-Task-Setup funktioniert und du Generalisierung
testen möchtest.

---

## 3 · Multi-Task-PPO

### Architektur: gemeinsamer Trunk + aufgabenkonditionierter Input

Die Standard-Multi-Task-Policy verwendet ein einzelnes neuronales Netz, bei dem das
Task-Encoding direkt an die Beobachtung konkateniert wird:

```
input = concat(observation, task_encoding)
  ↓
shared MLP trunk (256 → 256)
  ↓
policy head → action distribution
  ↓
value head → V(s, z)
```

Sowohl der Policy-Head als auch der Value-Head sehen das Task-Encoding. Entscheidend ist,
dass der Trunk **gemeinsam** ist — die Policy ist gezwungen, eine Repräsentation zu
lernen, die über alle Aufgaben hinweg nützlich ist. Das ist die Quelle sowohl von
positivem Transfer (gemeinsame Features helfen allen) als auch von negativem Transfer
(widersprüchliche Gradientenrichtungen schaden jemandem).

### Alternative: aufgabenspezifische Heads

Für Aufgaben mit sehr unterschiedlichen Ausgabeanforderungen ersetzt du den gemeinsamen
Head durch N aufgabenspezifische Heads und nutzt die Aufgaben-ID, um auszuwählen, welcher
Head aktiviert wird:

```
shared trunk
  ↓
task_id → select head
  ├── head_0 → action distribution for task 0
  ├── head_1 → action distribution for task 1
  └── head_2 → action distribution for task 2
```

Aufgabenspezifische Heads reduzieren negativen Transfer in der Ausgabeschicht, kosten
aber mehr Parameter und verhindern Interpolation zwischen den Aufgaben-Heads.

### SB3 mit einem Task-ID-Observation-Wrapper

Stable-Baselines3 PPO unterstützt Multitask-Training nicht nativ, aber ein einfacher
`gymnasium.ObservationWrapper`, der die Aufgaben-ID an den Beobachtungsvektor anhängt,
reicht in den meisten Fällen aus:

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

class TaskIDWrapper(gym.ObservationWrapper):
    """Prepends a one-hot task ID to the flat observation vector."""

    def __init__(self, env, task_id: int, n_tasks: int):
        super().__init__(env)
        self.task_id = task_id
        self.n_tasks = n_tasks

        # Use np.prod so this works for any flat obs shape (MLP-only wrapper).
        # For image observations, flatten first or use a different approach.
        original_shape = int(np.prod(env.observation_space.shape))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(original_shape + n_tasks,),
            dtype=np.float32,
        )

    def observation(self, obs):
        one_hot = np.zeros(self.n_tasks, dtype=np.float32)
        one_hot[self.task_id] = 1.0
        return np.concatenate([one_hot, obs])
```

### Trainingsschleife: Round-Robin-Aufgabensampling

Die einfachste Multitask-Trainingsstrategie ist, Aufgaben im Round-Robin-Verfahren zu
rotieren und für jede einen Rollout zu sammeln, bevor ein gemeinsames Update ausgeführt
wird:

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

N_TASKS = 3

def make_task_env(task_id: int):
    """Returns a factory function for DummyVecEnv."""
    def _make():
        # Replace with your actual per-task environment
        env = gym.make("LunarLander-v2")
        return TaskIDWrapper(env, task_id=task_id, n_tasks=N_TASKS)
    return _make

# Create one vectorised environment per task
envs = [DummyVecEnv([make_task_env(i)]) for i in range(N_TASKS)]

# Single policy — observation space must match across all tasks
# (guaranteed by the wrapper)
model = PPO(
    "MlpPolicy",
    envs[0],
    verbose=1,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    tensorboard_log="logs/multitask_ppo/",
)

# Manually implement round-robin: collect rollouts from each task env,
# then call a single gradient update
TOTAL_TIMESTEPS = 3_000_000
STEPS_PER_TASK = 2048

for iteration in range(TOTAL_TIMESTEPS // (STEPS_PER_TASK * N_TASKS)):
    for task_id, env in enumerate(envs):
        # Swap the environment and collect one rollout batch
        model.set_env(env)
        model.learn(
            total_timesteps=STEPS_PER_TASK,
            reset_num_timesteps=False,
            tb_log_name=f"task_{task_id}",
        )
```

!!! tip "Pro-Aufgabe-TensorBoard-Logging"
    Logge `reward/task_0`, `reward/task_1` und `reward/task_2` als separate Skalare,
    statt sie zu aggregieren. Aggregation verbirgt negativen Transfer — zwei Aufgaben
    können sich zu gesund aussehenden 0,5 mitteln, während eine bei 0,0 feststeckt und
    die andere bei 1,0 ist.

---

## 4 · Negativer Transfer

Negativer Transfer tritt auf, wenn das Lernen einer Aufgabe die Leistung auf einer
anderen aktiv verschlechtert. Es ist die zentrale praktische Herausforderung von
Multi-Task-RL und kann schwer zu diagnostizieren sein.

### Warum es passiert: Gradienteninterferenz

Jede Aufgabe erzeugt ihre eigene Gradientenrichtung im Parameterraum. Wenn der Gradient
von Aufgabe A nach „Norden" zeigt und der von Aufgabe B nach „Südosten", zeigt der
kombinierte Gradient nach „Nord-Nordost" — keine Aufgabe erhält das Update, das sie
will. Wenn der Winkel zwischen zwei Aufgabengradienten 90 Grad überschreitet, ist das
Skalarprodukt negativ, was bedeutet, dass die Aufgaben aktiv im Konflikt stehen. Das
Update der einen Aufgabe macht die andere buchstäblich schlechter.

```
Task A gradient: ▲  (wants to increase value of action "jump")
Task B gradient: ▼  (wants to decrease value of action "jump")
Combined:        →  (neither task is served correctly)
```

Dieser Konflikt ist am schlimmsten, wenn:

- Aufgaben qualitativ unterschiedliche Verhaltensweisen erfordern (z. B. aggressiv vs.
  vorsichtig).
- Aufgaben unterschiedliche Belohnungsskalen haben, sodass eine die Gradientennorm
  dominiert.
- Die Policy gezwungen ist, alle Parameter einschließlich der letzten Ausgabeschicht zu
  teilen.

### Wie man ihn erkennt: Pro-Aufgabe-TensorBoard-Kurven

Das klarste Signal sind **divergierende Pro-Aufgabe-Belohnungskurven**:

```
Step 0         Step 500K      Step 1M
──────────────────────────────────────
task_0: 0.2 → 0.8 → 0.9    (improving)
task_1: 0.1 → 0.3 → 0.1    (regressing after initial gain)
task_2: 0.0 → 0.0 → 0.0    (never learned)
```

Aufgabe 1, die regrediert, während Aufgabe 0 sich weiter verbessert, ist die kanonische
Signatur von negativem Transfer. Aufgabe 2, die überhaupt nie lernt, deutet auf ein
Ungleichgewicht in Belohnungsskala oder Schwierigkeit hin.

```python
# Log per-task rewards during evaluation
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs/multitask_eval")

def evaluate_per_task(model, envs, n_episodes=20):
    for task_id, env in enumerate(envs):
        total_reward = 0.0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
        mean_reward = total_reward / n_episodes
        writer.add_scalar(f"eval/reward_task_{task_id}", mean_reward, global_step)
```

### Mitigationsstrategien

| Strategie | Idee | Wann verwenden |
|---|---|---|
| Belohnungsnormalisierung | Belohnung jeder Aufgabe auf Mittelwert 0, Varianz 1 normalisieren | Immer — erstes, was du versuchen solltest |
| Aufgabenspezifische Heads | Separate Ausgabeschichten pro Aufgabe | Wenn Aufgaben unterschiedliche Aktionsverteilungen brauchen |
| Gradient Surgery (PCGrad) | Konfliktäre Gradienten projizieren, um Interferenz zu entfernen | Wenn Aufgaben bekannt im Konflikt sind |
| Separate Value-Funktionen | Actor teilen, Critic pro Aufgabe trennen | Moderater Konflikt; günstig umzusetzen |
| Curriculum-Ordnung | Aufgaben sequentiell einführen, nicht alle auf einmal | Wenn eine Aufgabe deutlich schwerer ist |

!!! warning "Reward-Skalen-Diskrepanz ist die häufigste Ursache"
    Bevor du Gradienteninterferenz beschuldigst, prüfe die Belohnungsskalen. Wenn
    Aufgabe 0 Belohnungen in `[-1, 1]` liefert und Aufgabe 1 in `[-100, 100]`, wird der
    Policy-Gradient vollständig von Aufgabe 1 dominiert. Normalisiere alle
    Aufgabenbelohnungen auf dieselbe Skala, bevor du irgendetwas anderes debuggst.

---

## 5 · Curriculum über Aufgaben

Nicht alle Aufgaben sind gleich schwer. Eine Anfänger-Policy ab Schritt 1 auf die
schwerste Aufgabe loszulassen, führt zu langsamer Konvergenz oder komplettem Scheitern.
Ein **Aufgaben-Curriculum** — einfach anfangen und schwerere Aufgaben einführen, sobald
die Policy reift — beschleunigt das Lernen dramatisch.

### Festes manuelles Curriculum

Definiere Aufgabenphasen explizit:

```python
CURRICULUM = [
    # (start_step, task_ids_to_include)
    (0,           [0]),          # Phase 1: reach goal only
    (500_000,     [0, 1]),       # Phase 2: add obstacle avoidance
    (1_500_000,   [0, 1, 2]),    # Phase 3: add time pressure
]

def get_active_tasks(global_step: int) -> list[int]:
    active = [0]
    for start_step, task_ids in CURRICULUM:
        if global_step >= start_step:
            active = task_ids
    return active
```

Das erfordert manuelles Tuning der Übergangsschwellen, kostet Aufwand, gibt aber
präzise Kontrolle.

### Automatisches Curriculum (Erfolgsschwelle pro Aufgabe)

Führe eine neue Aufgabe erst ein, wenn die Policy eine Mindest-Erfolgsquote auf der
aktuellen Aufgabenmenge erreicht:

```python
SUCCESS_THRESHOLD = 0.75  # require 75% success before adding next task

def maybe_expand_curriculum(current_tasks, per_task_success_rates, all_tasks):
    min_success = min(per_task_success_rates[t] for t in current_tasks)
    if min_success >= SUCCESS_THRESHOLD:
        next_task_id = len(current_tasks)
        if next_task_id < len(all_tasks):
            print(f"Curriculum expanding: adding task {next_task_id}")
            return current_tasks + [next_task_id]
    return current_tasks
```

!!! tip "Automatisches Curriculum kann stocken"
    Wenn die Policy den `SUCCESS_THRESHOLD` auf einer harten Aufgabe nie erreicht,
    stockt das Curriculum dauerhaft. Füge eine maximale Wartezeit hinzu — führe die
    nächste Aufgabe nach N Schritten unabhängig von der Erfolgsquote ein, sonst
    erreichst du möglicherweise die späteren Aufgaben überhaupt nie.

### Sampling-Gewichte über Aufgaben

Statt harter Curriculum-Phasen verwende eine weiche Wahrscheinlichkeitsverteilung über
Aufgaben, die sich mit fortschreitendem Training verschiebt:

```python
import numpy as np

def task_sampling_weights(per_task_success: list[float], temperature: float = 1.0) -> np.ndarray:
    """
    Weight tasks inversely by their current success rate:
    harder tasks (low success) get sampled more often.
    """
    difficulties = [1.0 - s for s in per_task_success]
    # Avoid zero weights
    difficulties = [max(d, 0.05) for d in difficulties]
    weights = np.array(difficulties) ** (1.0 / temperature)
    return weights / weights.sum()

# Example: tasks at success rates [0.9, 0.4, 0.1]
weights = task_sampling_weights([0.9, 0.4, 0.1])
# → task 2 (hardest) gets sampled most often
task_id = np.random.choice(N_TASKS, p=weights)
```

Das ist eine einfache Form von **Prioritized Task Replay**, analog zu Prioritized
Experience Replay in DQN, aber auf Aufgabenebene.

---

## 6 · Godot-Multitask-Beispiel

Dieser Abschnitt implementiert einen vollständigen Drei-Aufgaben-AIController in
Godot 4. Ein einzelner Agent wechselt auf Anfrage zwischen drei Varianten:

| Aufgabe | ID | Belohnungsstruktur | Erforderliche Fähigkeit |
|---|---|---|---|
| Reach goal | 0 | Dichte Distanzbelohnung | Effiziente Navigation |
| Reach goal, avoid obstacles | 1 | Distanzbelohnung − Kollisionsstrafe | Navigation + Hinderniswahrnehmung |
| Reach goal under time pressure | 2 | Distanzbelohnung + Zeitbonus | Schnelle Navigation |

### Szenenaufbau

```
MultiTaskEnv (Node3D)
├── Agent (CharacterBody3D)
│   └── MultiTaskAIController (extends AIController3D)
├── Goal (Area3D + MeshInstance3D)          ← moves each episode
├── ObstacleSpawner (Node3D)                ← spawns 0–5 obstacles per episode
├── TaskIndicator (Label3D)                 ← displays current task ID in viewport
└── TimerBar (ProgressBar — CanvasLayer)    ← visible only in task 2
```

### GDScript: MultiTaskAIController

```gdscript
# MultiTaskAIController.gd
extends AIController3D

const N_TASKS := 3
const OBS_DIM := 8       # base observation size before task ID
const MAX_EPISODE_STEPS := 200
const TIME_PRESSURE_LIMIT := 100  # steps before time bonus decays to zero (task 2)

var current_task_id := 0
var _step_count := 0
var _episode_reward := 0.0

# Obstacles cached each episode
var _obstacle_positions: Array[Vector3] = []

func _ready() -> void:
    # Task ID is set externally by the training script via set_task()
    _randomise_episode()

func set_task(task_id: int) -> void:
    current_task_id = clamp(task_id, 0, N_TASKS - 1)
    get_node("../TaskIndicator").text = ["REACH", "AVOID+REACH", "TIMED"][current_task_id]

func get_obs() -> Array:
    var agent := get_node("../Agent") as CharacterBody3D
    var goal := get_node("../Goal") as Area3D
    var to_goal: Vector3 = goal.global_position - agent.global_position

    # Base observation (8 values)
    var obstacle_delta := _nearest_obstacle_delta()
    var base_obs := [
        to_goal.x / 20.0,          # normalised delta X
        to_goal.z / 20.0,          # normalised delta Z
        agent.velocity.x / 10.0,
        agent.velocity.z / 10.0,
        obstacle_delta.x / 20.0,
        obstacle_delta.z / 20.0,
        float(_step_count) / float(MAX_EPISODE_STEPS),
        to_goal.length() / 20.0,   # distance to goal (scalar)
    ]

    # One-hot task ID prepended (3 values)
    var task_one_hot := [0.0, 0.0, 0.0]
    task_one_hot[current_task_id] = 1.0

    return task_one_hot + base_obs  # total: 11 values

func get_action_space() -> Dictionary:
    return {
        "move": {"size": 2, "action_type": "continuous"}
    }

func set_action(action: Dictionary) -> void:
    var agent := get_node("../Agent") as CharacterBody3D
    var move := action["move"] as Array
    agent.velocity.x = move[0] * 6.0
    agent.velocity.z = move[1] * 6.0
    _step_count += 1

func get_reward() -> float:
    var agent := get_node("../Agent") as CharacterBody3D
    var goal := get_node("../Goal") as Area3D
    var dist := agent.global_position.distance_to(goal.global_position)

    var reward := 0.0

    # Component shared across all tasks: dense distance reward
    reward += -dist * 0.01

    # Task-specific components
    match current_task_id:
        0:
            # Reach goal: bonus for reaching
            if dist < 1.0:
                reward += 1.0
        1:
            # Avoid obstacles: penalty per collision
            if dist < 1.0:
                reward += 1.0
            reward += -0.5 * float(_current_collision_count())
        2:
            # Time pressure: bonus decays linearly with remaining steps
            if dist < 1.0:
                var steps_remaining := MAX_EPISODE_STEPS - _step_count
                var time_bonus := float(steps_remaining) / float(TIME_PRESSURE_LIMIT)
                reward += 1.0 + clamp(time_bonus, 0.0, 2.0)

    return reward

func get_done() -> bool:
    var agent := get_node("../Agent") as CharacterBody3D
    var goal := get_node("../Goal") as Area3D
    var dist := agent.global_position.distance_to(goal.global_position)
    return dist < 1.0 or _step_count >= MAX_EPISODE_STEPS

func reset() -> void:
    _step_count = 0
    _episode_reward = 0.0
    _randomise_episode()

func _randomise_episode() -> void:
    # Randomise goal position
    get_node("../Goal").global_position = Vector3(
        randf_range(-8.0, 8.0), 0.5, randf_range(-8.0, 8.0)
    )
    # Spawn obstacles only for tasks 1 and 2
    _obstacle_positions.clear()
    if current_task_id >= 1:
        for i in range(randi_range(2, 5)):
            _obstacle_positions.append(Vector3(
                randf_range(-7.0, 7.0), 0.5, randf_range(-7.0, 7.0)
            ))
    get_node("../ObstacleSpawner").rebuild(_obstacle_positions)

func _nearest_obstacle_delta() -> Vector3:
    if _obstacle_positions.is_empty():
        return Vector3(20.0, 0.0, 20.0)  # far away — no obstacle
    var agent_pos := get_node("../Agent").global_position
    var nearest := _obstacle_positions[0]
    for pos in _obstacle_positions:
        if pos.distance_to(agent_pos) < nearest.distance_to(agent_pos):
            nearest = pos
    return nearest - agent_pos

func _current_collision_count() -> int:
    # Count obstacles within collision radius
    var agent_pos := get_node("../Agent").global_position
    var count := 0
    for pos in _obstacle_positions:
        if pos.distance_to(agent_pos) < 1.2:
            count += 1
    return count
```

### Python-Trainingsskript mit Aufgabenrotation

```python
# train_multitask.py
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

N_TASKS = 3
STEPS_PER_ROTATION = 4096  # collect this many steps per task before rotating


class TaskRotationCallback(BaseCallback):
    """Rotates the active task ID in the Godot environment every K steps."""

    def __init__(self, n_tasks: int, steps_per_rotation: int):
        super().__init__()
        self.n_tasks = n_tasks
        self.steps_per_rotation = steps_per_rotation
        self._current_task = 0
        self._steps_since_rotation = 0

    def _on_step(self) -> bool:
        self._steps_since_rotation += 1
        if self._steps_since_rotation >= self.steps_per_rotation:
            self._steps_since_rotation = 0
            self._current_task = (self._current_task + 1) % self.n_tasks
            # Signal the Godot environment to switch tasks
            # (environment must expose a set_task method or info channel)
            self.training_env.env_method("set_task", self._current_task)
            self.logger.record("curriculum/active_task", self._current_task)
        return True


env = StableBaselinesGodotEnv(
    env_path="builds/multitask_env.x86_64",
    n_parallel=4,
    speedup=10,
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    tensorboard_log="logs/multitask_ppo/",
    policy_kwargs=dict(net_arch=[256, 256]),
)

callback = TaskRotationCallback(n_tasks=N_TASKS, steps_per_rotation=STEPS_PER_ROTATION)
model.learn(total_timesteps=3_000_000, callback=callback)
model.save("models/multitask_ppo_final")
```

!!! check "Fertig, wenn"
    Wenn `train_multitask.py` durchgelaufen ist, führe die Pro-Aufgabe-Evaluation aus
    Abschnitt 8 aus (100 Episoden pro Aufgabe): Eine gut trainierte Policy erreicht die
    Marke von ≥ 80 % Erfolg auf **jeder** Aufgabe, wie im Balkendiagramm aus „Drei Wege,
    deine KI zu beobachten". Während des Trainings sollten alle drei
    `reward/task_*`-Kurven in TensorBoard gemeinsam steigen — verrauscht und nicht im
    Gleichschritt. Fällt aber eine Kurve, während eine andere weiter steigt, ist das
    negativer Transfer und kein Lauf, der nur mehr Schritte braucht: Arbeite Abschnitt 4
    durch (zuerst die Belohnungsskalen prüfen), bevor du länger trainierst.

---

## 7 · Multi-Task-SAC (optional beim ersten Lesen)

!!! note "Erster Durchgang? Überfliege oder überspringe diesen Abschnitt."
    Der Kernpfad durch diese Unit sind die Abschnitte 1–6 (Task-Encodings,
    Multi-Task-PPO, negativer Transfer, Aufgaben-Curricula, das Godot-Beispiel),
    Abschnitt 8 (Pro-Aufgabe-Evaluation) und Abschnitt 10 (die Entscheidung Multi-Task
    vs. separate Policies). SAC ist eine Off-Policy-Alternative, die Sample-Effizienz
    bringt — komm darauf zurück, sobald dein PPO-Lauf funktioniert.

PPO ist on-policy: es sammelt vor jedem Update frische Rollouts und verwirft sie dann.
In Multitask-Settings ist das verschwenderisch — Erfahrung aus Aufgabe 0 wird verworfen,
bevor Aufgabe 1 anfängt zu sammeln.

**SAC** (Soft Actor-Critic) ist off-policy: es speichert alle Erfahrungen in einem
Replay Buffer und kann Erfahrungen aus allen Aufgaben in jedem Gradientenupdate mischen.
Für Multitask-Training ist das ein signifikanter Vorteil: ein einzelner, über Aufgaben
geteilter Replay Buffer gibt dem Gradienten jeder Aufgabe Information über die
Transitions jeder anderen Aufgabe.

### Aufgabenkonditionierte SAC-Architektur

Der Critic muss sowohl auf die Beobachtung als auch auf das Task-Encoding konditionieren:

```python
from stable_baselines3 import SAC
import gymnasium as gym
import numpy as np

# SAC with task-ID obs uses the same TaskIDWrapper as PPO
# — SAC's MultiInputPolicy or MlpPolicy handles the concatenated obs identically

model = SAC(
    "MlpPolicy",
    env,                         # TaskIDWrapper-wrapped env
    verbose=1,
    learning_rate=3e-4,
    buffer_size=1_000_000,       # shared replay buffer across all tasks
    learning_starts=10_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",             # automatic entropy tuning
    tensorboard_log="logs/multitask_sac/",
    policy_kwargs=dict(net_arch=[256, 256]),
)
```

!!! tip "SAC-Sample-Efficiency-Vorteil"
    Im direkten Vergleich auf dem Drei-Aufgaben-Godot-Beispiel erreicht SAC typischerweise
    dieselbe Pro-Aufgabe-Erfolgsquote wie PPO in etwa der Hälfte der Umgebungsschritte.
    Der Preis: SAC braucht mehr Speicher (den Replay Buffer) und ist schwerer zu tunen
    (Actor-Critic-Instabilitäten, Sensibilität des Entropie-Koeffizienten). Für eine
    Godot-Umgebung, die mit 10× Speedup läuft, ist der Wall-Clock-Unterschied meist
    klein — nimm PPO, wenn du anfängst, und SAC, wenn du die letzten Prozent an
    Sample-Effizienz brauchst.

### Mixed-Task-Replay-Sampling

Standardmäßig sampelt SB3-SAC den Replay Buffer uniform. Für Multitask-Training hilft
es, uniform über Aufgaben zu sampeln (nicht uniform über alle Transitions, was seltene
Aufgaben untersampeln würde):

!!! warning "Pseudocode"
    Die Klasse unten illustriert das Konzept. Die volle SB3-Integration erfordert das
    Subclassing von `ReplayBuffer` und das Überschreiben von `sample` mit der korrekten
    SB3-Buffer-API. Füge das nicht in ein Trainingsskript ein, ohne diese Details
    zu vervollständigen.

```python
class MultiTaskReplayBuffer:
    """Wraps SB3 ReplayBuffer to ensure uniform task sampling."""

    def __init__(self, base_buffer, n_tasks):
        self.buffers = [base_buffer.__class__(...) for _ in range(n_tasks)]
        self.n_tasks = n_tasks

    def add(self, obs, next_obs, action, reward, done, infos):
        task_id = self._extract_task_id(obs)
        self.buffers[task_id].add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size):
        per_task = batch_size // self.n_tasks
        batches = [buf.sample(per_task) for buf in self.buffers]
        return self._concatenate_batches(batches)
```

---

## 8 · Evaluation

Eine Multitask-Policy muss pro Aufgabe evaluiert werden, nicht als einzelnes Aggregat.

### Erfolgsquote pro Aufgabe

```python
def evaluate_multitask(model, task_envs: dict, n_episodes: int = 100) -> dict:
    """
    Args:
        task_envs: {task_id: gym.Env} mapping
    Returns:
        {task_id: success_rate}
    """
    results = {}
    for task_id, env in task_envs.items():
        successes = 0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            if info.get("is_success", False):
                successes += 1
        results[task_id] = successes / n_episodes
    return results
```

### Vorwärtstransfer

**Vorwärtstransfer** (forward transfer) misst, ob das Lernen von Aufgabe A beim späteren
Lernen von Aufgabe B geholfen hat. Die Metrik vergleicht:

- `AUC(single-task B)` — die Fläche unter der Lernkurve, wenn Aufgabe B von Grund auf
  trainiert wird.
- `AUC(multi-task B)` — die Fläche unter der Lernkurve für Aufgabe B, wenn sie gemeinsam
  mit Aufgabe A trainiert wird.

Positiver Vorwärtstransfer: Multi-Task-B konvergiert schneller (höhere AUC bei gleicher
Anzahl Schritten). Negativer Transfer: Multi-Task-B konvergiert langsamer.

```python
def forward_transfer(single_task_curve, multitask_curve) -> float:
    """Returns > 0 for positive transfer, < 0 for negative transfer."""
    auc_single = np.trapz(single_task_curve)
    auc_multi = np.trapz(multitask_curve)
    return (auc_multi - auc_single) / auc_single
```

### Zero-Shot-Generalisierung

Nach dem Training auf Aufgaben 0, 1, 2, evaluiere die Policy auf einer zurückgehaltenen
Aufgabe 3 (z. B. Ziel erreichen UND Hindernisse vermeiden UND unter Zeitdruck) ohne
weiteres Training:

```python
# Construct task 3 encoding from seen task parameter vectors
# (only possible with task-parameter encoding, not one-hot)
task_3_params = np.array([1.0, 0.5, 1.5])  # novel combination of seen parameters
obs_with_task = np.concatenate([task_3_params, raw_observation])
action, _ = model.predict(obs_with_task, deterministic=True)
```

One-Hot-Encoding kann überhaupt nicht auf zurückgehaltene Aufgaben-IDs generalisieren.
Aufgabenparameter-Encoding generalisiert per Interpolation (innerhalb der
Trainingsverteilung der Parameter). Sprach-Embedding generalisiert am weitesten, zu
semantisch ähnlichen, aber nie gesehenen Anweisungen.

### Ehrliches Evaluationsprotokoll

| Metrik | Was sie misst | Wann berichten |
|---|---|---|
| Erfolgsquote pro Aufgabe | Löst die Policy jede trainierte Aufgabe? | Immer |
| Aggregierte Belohnung | Zusammenfassung über Aufgaben | Nützlich, verbirgt aber negativen Transfer; neben Pro-Aufgabe berichten |
| Vorwärtstransfer-Verhältnis | Hat Multitask vs. Single-Task geholfen? | Beim Vergleich gegen N separate Policies |
| Zero-Shot-Erfolgsquote | Generalisiert die Policy auf ungesehene Aufgaben? | Nur wenn Aufgabenparameter- oder Sprach-Encoding verwendet wird |

---

## 9 · Verbindung zu Foundation Models (optional beim ersten Lesen)

!!! note "Erster Durchgang? Überfliege oder überspringe diesen Abschnitt."
    Nichts hiervon wird für die Hands-on-Übung gebraucht. Der Kernpfad sind die
    Abschnitte 1–6, Abschnitt 8 und Abschnitt 10; dieser Abschnitt ist ein
    konzeptioneller Ausblick, der die aufgabenkonditionierte Policy dieser Unit mit
    generalistischen Agenten wie Gato und RT-2 verbindet — lies ihn als Einordnung,
    sobald dein eigener Multitask-Agent trainiert.

Multi-Task-RL ist der konzeptionelle Vorläufer des ehrgeizigsten Ziels im Feld:
**generalistische Agenten**, die Anweisungen folgen und Aufgaben lösen können, die sie
nie explizit trainiert haben.

### Das Skalierungs-Argument

Wenn die Anzahl der Trainingsaufgaben N wächst:

- N = 3 → Multi-Task-RL (diese Unit)
- N = 100 → Meta-RL: die Policy lernt, sich zur Inferenzzeit an *neue* Aufgaben *anzupassen*
- N = 600 → Gato (DeepMind, 2022): ein Transformer, gleichzeitig auf Atari, Robotik,
  Bildbeschreibung, Question Answering und Textdialog trainiert
- N = ∞ → hypothetischer generalistischer Agent, trainiert auf alle als Sprache
  ausdrückbaren Aufgaben

Die zentrale Erkenntnis aus Gato: ein einzelner großer Transformer mit
aufgabenkonditionierten Inputs erreicht wettbewerbsfähige Leistung auf Hunderten
verschiedener Aufgaben. Die Multitask-Policy aus dieser Unit ist architektonisch
identisch mit einem winzigen Gato — dieselbe Struktur, nur ohne Skalierung.

### Instruction-Following-Roboter

Sprach-Embedding (Abschnitt 2) ist die Brücke von Multi-Task-RL zu Robotiksystemen wie:

- **RT-2** (Google DeepMind, 2023) — Vision-Language-Action-Modell; Sprachbefehle
  konditionieren Manipulations-Policies des Roboters.
- **SayCan** (Google, 2022) — LLM erzeugt Aufgabenpläne; jeder Schritt wird von einer
  spezialisierten, mit RL trainierten Skill-Policy ausgeführt.

Beide beruhen auf demselben Prinzip: eine aufgabenkonditionierte Policy, bei der der
Konditionierungsvektor aus natürlicher Sprache abgeleitet wird. Die Lücke zwischen der
Godot-Übung in dieser Unit und diesen Systemen liegt primär in Skalierung und
Pre-Training-Daten, nicht in architektonischer Neuheit.

### Praktische Erkenntnis

Multi-Task-RL mit Sprach-Embeddings in einem Godot-Spiel ist das Training eines
generalistischen Agenten im Kleinen. Die hier eingeübten Gewohnheiten — sorgfältige
Pro-Aufgabe-Evaluation, Überwachung von negativem Transfer, Curriculum-Design — sind
genau die Gewohnheiten, die für die Arbeit mit größeren Systemen auf Forschungsskala
nötig sind.

---

## 10 · Wann Multi-Task-RL vs. separate Policies verwenden

Das ist die Frage, mit der die Unit anfing. Hier eine strukturierte Antwort.

### Verwende separate Policies, wenn

- N ≤ 5 Aufgaben und du dir leisten kannst, N Checkpoints zu trainieren und zu speichern.
- Aufgaben qualitativ sehr unterschiedlich sind (unterschiedliche Beobachtungsräume,
  unterschiedliche Aktionsräume) — der gemeinsame Trunk bringt wenig Nutzen.
- Pro-Aufgabe-Leistung wichtiger ist als Generalisierung. Spezialisten übertreffen
  Generalisten auf ihrer eigenen Aufgabe fast immer.
- Du die einfachste mögliche Debugging-Story willst. Jedes Versagen einer Policy ist
  isoliert.

### Verwende Multi-Task-RL, wenn

- N groß ist (> 10) und das Training von N separaten Policies undurchführbar ist.
- Inferenzzeit oder Speicher begrenzt sind (ein Modell passt dort, wo N Modelle nicht
  passen).
- Du Aufgabenwechsel zur Laufzeit ohne Laden eines neuen Checkpoints brauchst.
- Du Vorwärtstransfer willst: du hast Belege (oder starken Glauben), dass einige Aufgaben
  anderen helfen werden.
- Du auf Zero-Shot-Generalisierung auf zurückgehaltene Aufgaben hinarbeitest — das
  erfordert per Definition fast immer Multitask-Training.

### Der Benchmark-Vergleich, den du immer laufen lassen solltest

Bevor du dich auf Multitask-Training festlegst, führe diesen Vergleich durch:

```python
# 1. Train separate PPO policies for each task
single_task_results = {}
for task_id in range(N_TASKS):
    model = PPO("MlpPolicy", make_task_env(task_id)(), verbose=0)
    model.learn(total_timesteps=1_000_000)
    single_task_results[task_id] = evaluate_task(model, task_id)

# 2. Train one multi-task PPO policy
multitask_model = PPO("MlpPolicy", multitask_env, verbose=0)
multitask_model.learn(total_timesteps=1_000_000 * N_TASKS)  # same total budget
multitask_results = evaluate_multitask(multitask_model, task_envs)

# 3. Compare per-task success rates
for task_id in range(N_TASKS):
    delta = multitask_results[task_id] - single_task_results[task_id]
    print(f"Task {task_id}: single={single_task_results[task_id]:.2f}, "
          f"multi={multitask_results[task_id]:.2f}, delta={delta:+.2f}")
```

Wenn die Multitask-Policy alle Single-Task-Policies erreicht oder schlägt: Multitask
gewinnt. Wenn irgendeine Aufgabe in Multitask substantiell schlechter ist: untersuche
negativen Transfer, bevor du die Multitask-Policy deployst.

---

## 11 · Stretch Goals

### Stretch 1 — Gradient Surgery (PCGrad)

Implementiere PCGrad (Yu et al. 2020, „Gradient Surgery for Multi-Task Learning"). Für
jedes Paar von Aufgaben mit negativem Gradient-Skalarprodukt projiziere den Gradienten
einer Aufgabe vor dem Aufsummieren auf die Normalebene der anderen:

```python
import torch

def pcgrad_update(gradients: list[torch.Tensor]) -> torch.Tensor:
    """
    gradients: list of per-task gradient vectors (one per task, flat)
    Returns: combined gradient with conflicts resolved
    """
    n_tasks = len(gradients)
    combined = torch.zeros_like(gradients[0])
    for i in range(n_tasks):
        g_i = gradients[i].clone()
        for j in range(n_tasks):
            if i == j:
                continue
            g_j = gradients[j]
            dot = torch.dot(g_i, g_j)
            if dot < 0:
                # Project g_i to remove the component in direction of g_j
                g_i -= (dot / (g_j.norm() ** 2 + 1e-8)) * g_j
        combined += g_i
    return combined
```

Schließe das an die SB3-PPO-Trainingsschleife an, indem du `OnPolicyAlgorithm.train()`
subclassed. Vergleiche die Pro-Aufgabe-Leistungskurven mit und ohne PCGrad auf der
Drei-Aufgaben-Godot-Umgebung. Berichte, ob die Regressionskurve für irgendeine Aufgabe
verschwindet.

### Stretch 2 — Generalisierung auf zurückgehaltene Aufgaben

Erweitere die Godot-Umgebung um eine vierte Aufgabe: das Ziel erreichen und dabei
Hindernisse vermeiden UND unter Zeitdruck (Kombination von Aufgaben 1 und 2). Trainiere
nur auf Aufgaben 0–2. Evaluiere zero-shot auf Aufgabe 3 mit:

1. One-Hot-Encoding (kann nicht generalisieren — verwende einen neuen Index und
   beobachte Versagen).
2. Aufgabenparameter-Encoding (sollte teilweise per Interpolation generalisieren).
3. Sprach-Embedding (falls GPU vorhanden — sollte am besten generalisieren).

Halte die Erfolgsquoten für jeden Encoding-Typ auf der zurückgehaltenen Aufgabe fest
und schreibe eine einseitige Analyse.

### Stretch 3 — Multi-Task-SAC vs. separate SAC

Führe ein kontrolliertes Experiment durch, das Wall-Clock- und Sample-Effizienz vergleicht:

| Bedingung | Setup |
|---|---|
| Separate SAC | Ein SAC pro Aufgabe trainieren, je 1 Mio. Schritte |
| Multi-Task-SAC | Ein SAC auf allen 3 Aufgaben trainieren, 3 Mio. Schritte gesamt |
| Multi-Task-SAC (Shared Buffer) | Gleich, aber uniformes Aufgabensampling aus dem Replay Buffer erzwungen |

Miss: finale Erfolgsquote pro Aufgabe, Trainings-Wall-Clock-Zeit, GPU-Gesamtspeicher.
Berichte, welche Bedingung die beste Pro-Aufgabe-Erfolgsquote pro GPU-Stunde erreicht.

### Stretch 4 — Sprachkonditionierte Policy in Godot

Ersetze die One-Hot-Aufgaben-ID durch ein SBERT-Embedding der Aufgabenbeschreibung.
Verkable `sentence-transformers` in das Python-Trainingsskript, codiere drei
Aufgabenbeschreibungs-Strings zu 384-dim Vektoren und übergib sie als Task-Encoding an
die Policy. Reduziere die Embedding-Dimension auf 8 mit einer gelernten linearen
Projektion (gemeinsam mit der Policy trainiert), um den Beobachtungsvektor klein zu
halten. Evaluiere zero-shot auf zwei neuen Aufgabenbeschreibungen, die die Policy nie
gesehen hat.

---

[← Hierarchisches RL](unit-hierarchical.md) · [Kursstartseite](index.md) · [→ Imitation Learning](unit-09.md)
