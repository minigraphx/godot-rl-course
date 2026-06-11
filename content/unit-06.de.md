# Unit 6 — Kontinuierliche 3D-Steuerung

Wechsle von diskreten Tastendrücken zu **kontinuierlichen Kräften und Lenkung**. Studiere **FlyBy** oder **HovercraftRacing** — beide exponieren einen kontinuierlichen Aktionsraum (action space) — und lerne, wie du Beobachtungen und Aktionen normalisierst, damit das neuronale Netz nicht sättigt.

[← Unit 5: Paralleltraining](unit-05.md) · [Kursübersicht](index.md)

!!! note "Voraussetzungen"
    - **[Unit 4](unit-04.md)** — PPO end-to-end, sicherer Umgang mit `clip_range`, GAE-λ, `approx_kl`
    - **[Unit 5](unit-05.md)** — parallele Umgebungen und das Evaluierungsprotokoll
    - GDScript-Sicherheit: Vektoren, Raycasts, `_physics_process`
    - **[Policy-Gradients-Einheit](unit-policy-gradients.md)** (empfohlen) — Gaußsche Policies lassen Abschnitt 1 verständlich werden

!!! info "Zeit"
    Lesen: ~30 min · Training: ~30 min GPU / ~2 Std CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    Godot (Viz-Checkpoint — flüssige vs. ruckartige Bewegung sagt viel aus) · TensorBoard (`ep_rew_mean` und `train/std` der Aktionsverteilung) · `ai_controller.gd`-Normalisierungsbereiche

---

## 1 · Kontinuierliche Aktionsräume

Bisher war jede Aktion **diskret**: Triebwerk zünden / nicht zünden. Kontinuierliche Aktionen sind **reelle Zahlen** — „wende 0,73 Newton Schub bei 12° links an". Das entspricht eher der Funktionsweise echter Motoren und Lenkungen.

| | Diskret | Kontinuierlich |
|--|--|--|
| `get_action_space()` | `"action_type": "discrete", "size": N` | `"action_type": "continuous", "size": N` |
| Policy-Ausgabe | Softmax → ein Index | Gaußsche Verteilung → N Floats |
| Algorithmus | DQN oder PPO | **Nur PPO** (DQN erfordert diskret) |
| Fallstrick | Keiner | Werte müssen begrenzt sein; vor Verwendung normalisieren |

```gdscript
# Continuous action space — 2 outputs: [thrust, steering]
func get_action_space() -> Dictionary:
    return {
        "motion": {"size": 2, "action_type": "continuous"}
    }

func set_action(action) -> void:
    var thrust   = action["motion"][0]   # roughly in [-1, 1]
    var steering = action["motion"][1]
    apply_force(Vector3.FORWARD * thrust * max_thrust)
    rotate_y(steering * max_steering_rate * get_physics_process_delta_time())
```

Die Policy gibt zu Beginn des Trainings Werte in etwa **[−1, 1]** aus — ohne Normalisierung dominieren jedoch rohe Physikwerte (z. B. Geschwindigkeit in m/s = 42,7) die Beobachtungen und destabilisieren das Lernen.

---

## 2 · Normalisierung — das Wichtigste überhaupt

!!! warning "Alles an Systemgrenzen normalisieren"
    Das neuronale Netz funktioniert am besten, wenn alle Eingaben in **[−1, 1]** oder **[0, 1]** liegen. Nicht normalisierte Beobachtungen (Position in Welteinheiten, Geschwindigkeit in Pixel/s) verursachen langsames Lernen oder Divergenz.

**Beobachtungsnormalisierung** — durch das erwartete Maximum dividieren:

```gdscript
func get_obs() -> Dictionary:
    var max_speed    = 20.0   # tune to your scene
    var max_dist     = 100.0
    var max_angle    = PI

    return {"obs": [
        linear_velocity.x / max_speed,
        linear_velocity.y / max_speed,
        linear_velocity.z / max_speed,
        rotation.y        / max_angle,
        angular_velocity.y / 5.0,
        raycast_forward.get_collision_distance() / max_dist,
        raycast_left.get_collision_distance()    / max_dist,
        raycast_right.get_collision_distance()   / max_dist,
    ]}
```

**Aktionsnormalisierung** — die Policy gibt in [−1, 1] aus; skaliere auf deine Physikeinheiten in `set_action()`:

```gdscript
func set_action(action) -> void:
    var raw_thrust   = action["motion"][0]          # [-1, 1]
    var raw_steering = action["motion"][1]          # [-1, 1]
    var thrust   = raw_thrust   * max_thrust        # scale to Newtons
    var steering = raw_steering * max_steering_rate
    ...
```

**Wie du überprüfst, ob die Normalisierung funktioniert:** Gib `get_obs()` für einige Schritte im menschlichen Steuermodus aus. Jeder Wert sollte innerhalb von [−2, 2] bleiben. Wenn ein Wert regelmäßig ±5 überschreitet, verringere seinen Divisor.

---

## 3 · FlyBy oder HovercraftRacing öffnen

Beide Beispiele befinden sich in [godot_rl_agents_examples](https://github.com/edbeeching/godot_rl_agents_examples).

- **FlyBy** — einfacher, Luftnavigation mit Strahlenwerfer (raycasts)
- **HovercraftRacing** — schwieriger, Rennstrecke, gemischtes Terrain

Beginne für deine erste kontinuierliche Unit mit FlyBy.

1. Öffne das Projekt in Godot .NET, aktiviere das Plugin
2. Öffne `ai_controller.gd` — lies `get_obs()`, `get_action_space()`, `set_action()`
3. Gib Beobachtungswerte während der menschlichen Steuerung aus, um die Normalisierung zu überprüfen:

```gdscript
func get_obs() -> Dictionary:
    var obs = { ... }
    if _ai.heuristic == "human":
        print(obs)
    return obs
```

4. Exportiere ein headless Binary

---

## 4 · RayCast3D-Sensoren

Kontinuierliche 3D-Umgebungen verwenden häufig **RayCast3D**-Knoten, um dem Agenten räumliches Bewusstsein ohne visuelles Rendering zu geben.

```gdscript
# In _ready():
@onready var rays = $RayCastGroup.get_children()  # array of RayCast3D nodes

func get_obs() -> Dictionary:
    var ray_obs = []
    for ray in rays:
        if ray.is_colliding():
            ray_obs.append(ray.get_collision_point().distance_to(global_position) / max_ray_dist)
        else:
            ray_obs.append(1.0)   # no collision = max distance
    return {"obs": ray_obs + [linear_velocity.x / max_speed, ...]}
```

Platziere RayCast3D-Knoten als Kinder einer `Node3D`-Gruppe, die nach vorne/links/rechts/oben/unten zeigen. Drehe die Gruppe mit dem Agenten mit. Mehr Strahlen = reicherer Zustand; beginne mit 5–9.

---

## 5 · Training

```bash
conda activate godot_env
tensorboard --logdir=logs &

gdrl --env_path=./FlyBy.x86_64 \
  --experiment_name=flyby_ppo \
  --timesteps=2_000_000 \
  --n_parallel=8 \
  --speedup=20 \
  --n_steps=512 \
  --batch_size=256
```

Kontinuierliche Aufgaben benötigen typischerweise mehr Zeitschritte als diskrete — 1–5M ist üblich. `n_steps=512` gibt PPO längere Rollouts, um den Vorteil über die längeren Episoden hinweg zu schätzen.

**TensorBoard-Signale, die du beobachten solltest:**

| Signal | Gesund | Problem |
|--------|--------|---------|
| `train/std` der Aktionen | Beginnt bei ~1,0, sinkt langsam | Kollabiert sofort auf 0 → füge `--ent_coef=0.005` hinzu |
| `rollout/ep_rew_mean` | Steigt bis 1M Schritte | Noch negativ bei 500k → Beobachtungsnormalisierung überprüfen |
| `train/approx_kl` | < 0,02 | Spitzen → `--learning_rate` oder `--clip_range` reduzieren |

!!! check "Fertig, wenn"
    FlyBy hat keinen veröffentlichten Benchmark — beurteile den Lauf also an seinen eigenen Signalen: `rollout/ep_rew_mean` liegt deutlich über dem Niveau der zufälligen Anfangsphase und steigt bei ~1M Schritten noch weiter, `train/std` sinkt allmählich statt zu kollabieren, und `train/approx_kl` bleibt unter 0,02. Bestätige dann mit dem Viz-Checkpoint (Abschnitt 6): flüssiger, zielgerichteter Flug zu den Checkpoints — kein ruckartiges Oszillieren und kein Drehen auf der Stelle.

---

## 6 · Viz-Checkpoint

Nach dem Training erneut mit `--viz` oder im Editor ausführen:

```bash
gdrl --env_path=./FlyBy.x86_64 \
  --resume_model_path=logs/sb3/flyby_ppo/best_model.zip \
  --inference \
  --viz
```

**Was flüssige vs. ruckartige Bewegung verrät:**

- **Flüssig, zielgerichtet** — die Policy hat eine stabile Aktionsverteilung gelernt; Normalisierung ist gut
- **Ruckartig, oszillierend** — Aktionswerte sind gesättigt (zu groß); reduziere den Skalierungsfaktor in `set_action()`
- **Dreht sich auf der Stelle** — der Belohnungsterm für die Winkelgeschwindigkeit könnte dominieren; Belohnungsformung (reward shaping) erneut prüfen

---

## 7 · Eigene kontinuierliche Umgebung erstellen

Wende das kontinuierliche Aktionsmuster auf eine neue Szene an:

1. Erstelle ein `RigidBody3D`-Fahrzeug oder -Flugzeug
2. Erweitere `AIController3D` (nicht `AIController2D`)
3. Implementiere 3D-Beobachtungen mit RayCast3D-Sensoren
4. Verwende einen 2–4-dimensionalen kontinuierlichen Aktionsraum
5. Forme Belohnungen: Distanz zum Ziel + Geschwindigkeitsstrafe + Überlebensbonus

---

## 8 · VecNormalize — Beobachtungs- und Belohnungsnormalisierung

Während die manuelle Normalisierung in `get_obs()` gut funktioniert, bietet SB3 auch **VecNormalize** — einen vektorisierten Wrapper, der einen laufenden Mittelwert und eine laufende Standardabweichung über alle parallelen Umgebungen hinweg verfolgt und sie spontan normalisiert.

**Was VecNormalize tut:**

- Pflegt einen laufenden Mittelwert und eine Standardabweichung für jede Beobachtungsdimension über alle `n_parallel`-Umgebungen hinweg
- Normalisiert jede Beobachtung auf ungefähr **N(0, 1)**, bevor sie das Policy-Netzwerk erreicht
- Normalisiert optional Belohnungen durch ihre laufende Standardabweichung (reduziert Varianz ohne das Vorzeichen zu ändern)
- Schneidet normalisierte Werte bei `clip_obs` (Standard 10,0) ab, um zu verhindern, dass Ausreißer dominieren

**Warum Kontinuierliche Steuerung es mehr braucht als diskrete:**

In diskreten Umgebungen sind Beobachtungen oft bereits begrenzt (z. B. ein Gitterindex oder ein boolesches Flag). In kontinuierlichen 3D-Umgebungen leben rohe Physikwerte auf wildly unterschiedlichen Skalen: Eine Gelenkgeschwindigkeit könnte 0,03 rad/s betragen, während eine Weltposition 847,2 m sein könnte. Ohne Normalisierung dominiert die betragsmäßig größte Beobachtungsdimension die Gradientenaktualisierung, und die anderen Dimensionen werden effektiv ignoriert.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="./FlyBy.x86_64", n_parallel=8, speedup=20)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/")
model.learn(total_timesteps=2_000_000)

# CRITICAL: save the normalization stats alongside the model
env.save("flyby_vecnormalize.pkl")
model.save("flyby_ppo")
```

!!! warning "VecNormalize-Statistiken immer speichern"
    Bei der Inferenz musst du dieselben Normalisierungsstatistiken neu laden, sonst erhält der ONNX-Export nicht normalisierte Beobachtungen und gibt unsinnige Aktionen aus.

    ```python
    env = StableBaselinesGodotEnv(env_path="./FlyBy.x86_64", n_parallel=1)
    env = VecNormalize.load("flyby_vecnormalize.pkl", env)
    env.training = False   # freeze the running stats during inference
    env.norm_reward = False

    model = PPO.load("flyby_ppo", env=env)
    ```

!!! tip "Wann du VecNormalize überspringen kannst"
    Wenn du bereits jede Beobachtung auf **[−1, 1]** innerhalb von `get_obs()` in GDScript normalisierst (der empfohlene Ansatz für diesen Kurs), ist VecNormalize überflüssig. Manuelle GDScript-Normalisierung ist transparenter und leichter zu debuggen während der Entwicklung. Greife auf VecNormalize zurück, wenn du einen Wert nicht einfach an der Quelle begrenzen kannst — zum Beispiel eine kumulative Distanz oder eine unbegrenzte Physikkraft.

---

## 9 · FlyBy vs. HovercraftRacing — dein Benchmark wählen

Beide Beispiele werden mit `godot_rl_agents_examples` geliefert und exponieren einen kontinuierlichen Aktionsraum, haben aber bedeutsam unterschiedliche Eigenschaften:

| | FlyBy | HovercraftRacing |
|--|-------|-----------------|
| Physik | 6-DOF-Freier Flug | Bodenbegrenzt, hohe Reibung |
| Aktionsraum | 3D-Schub + Gieren | 2D-Schub + Lenken |
| Belohnung | Checkpoint-Nähe | Rennposition + Geschwindigkeit |
| Schwierigkeit | Mittel — 3D-Orientierung ist schwer | Schwer — enge Kurven, Gegner |
| Empfohlen für | Kontinuierliche 3D-Beobachtungen lernen | Wettbewerbsfähige kontinuierliche Steuerung |
| Typische Konvergenz | 1–2 M Schritte | 3–5 M Schritte |

**Was zuerst verwenden:** Beginne mit FlyBy. Es hat einfachere Physik und konvergiert schneller, sodass du schnell über dein Beobachtungs- und Belohnungsdesign iterieren kannst. Wechsle zu HovercraftRacing, sobald du bestätigt hast, dass deine Beobachtungsnormalisierung, Aktionsskalierung und Belohnungsformung alle korrekt funktionieren — die zusätzliche Komplexität einer Rennstrecke und Gegner fügt dem Debuggen nur Rauschen hinzu, wenn die Grundlagen noch nicht solide sind.

**Wesentliche Unterschiede im Beobachtungsdesign:**

- *FlyBy* benötigt die Orientierung relativ zum nächsten Checkpoint (ein 3D-Einheitsvektor), lineare Geschwindigkeit und Strahlenwerfer-Distanzen. Die Checkpoint-Richtung liefert ein klares Lernsignal — die Belohnung steigt, wenn der Agent sich darauf zubewegt.
- *HovercraftRacing* exponiert typischerweise die streckenrelative Position, Geschwindigkeitskomponenten und die Distanz zur nächsten Wand oder zum nächsten Wegpunkt. Da das Hovercraft bodenbegrenzt ist, kannst du die vertikale Geschwindigkeitskomponente und die Auf-/Ab-Strahlen weglassen, was den Beobachtungsvektor kleiner hält.

**Wesentliche Unterschiede in der Belohnungsformung:**

- *FlyBy*: Eine dichte Belohnung proportional zu `max(0, prev_dist_to_checkpoint - curr_dist_to_checkpoint)` funktioniert gut. Füge einen kleinen Zeitlebensbonus hinzu, um frühzeitiges Absturzen zu vermeiden.
- *HovercraftRacing*: Reine Distanzbelohnung kann dem Agenten beibringen, enge Linien zu fahren, Gegner aber zu ignorieren. Erwäge das Hinzufügen einer Strafe für laterale Distanz zur Streckenmitte und eines Bonus für Überholen.

!!! tip "Benchmark-Progression"
    FlyBy → HovercraftRacing spiegelt das allgemeine Muster wider, mit Einfachem zu beginnen und Komplexität hinzuzufügen. Widerstehe der Versuchung, direkt zur schwierigeren Umgebung zu springen — eine Policy, die in HovercraftRacing nicht konvergiert, könnte aus einem Dutzend verschiedener Gründe scheitern; in FlyBy gibt es weit weniger Stellen, an denen man suchen muss.

---

## 10 · RayCast3D-Sensordesign für 3D-Umgebungen (optional beim ersten Lesen)

!!! note "Erster Durchgang? Überfliege oder überspringe diesen Abschnitt."
    Der Kernpfad dieser Unit sind die Abschnitte 1–7: kontinuierliche Aktionsräume (1), Normalisierung (2), FlyBy öffnen (3), Raycast-Grundlagen (4), Training (5), Viz-Checkpoint (6) und eine eigene Umgebung bauen (7). Abschnitt 4 deckt genug Raycast-Mechanik ab, um FlyBy zu trainieren — komm hierher zurück, wenn du Sensoren für deine eigene Umgebung entwirfst.

Das kurze Strahlenwerfer-Beispiel in Abschnitt 4 behandelt die Mechanik. Dieser Abschnitt geht tiefer auf Designentscheidungen ein.

**Wichtige `RayCast3D`-Parameter, die du im Inspector festlegen solltest:**

| Parameter | Zweck | Empfohlener Startwert |
|-----------|-------|-----------------------|
| `target_position` | Richtung und maximale Länge des Strahls | `Vector3(0, 0, -20)` für vorwärts, 20 m max |
| `collision_mask` | Welche Physikschichten erfasst werden | Passe deine Hindernis-/Wandschichten an; schließe die eigene Schicht des Agenten aus |
| `enabled` | Ob der Strahl aktiv ist | Immer `true` während des Trainings; siehe Warnung unten |

**Normalisiertes Distanzmuster für `get_obs()`:**

```gdscript
@onready var rays = [$RayForward, $RayLeft, $RayRight, $RayUp, $RayDown]
const RAY_MAX = 20.0  # meters — must match target_position length

func _get_ray_obs() -> Array:
    var obs = []
    for ray in rays:
        if ray.is_colliding():
            obs.append(ray.get_collision_point().distance_to(global_position) / RAY_MAX)
        else:
            obs.append(1.0)  # no hit = max distance (already normalized)
    return obs
```

Rufe `_get_ray_obs()` innerhalb von `get_obs()` auf und verknüpfe es mit deinen Geschwindigkeits- und Orientierungsbeobachtungen.

!!! tip "Wie viele Strahlen?"
    Fünf Strahlen (vorwärts, links, rechts, oben, unten) geben genug räumliches Bewusstsein für FlyBy und halten den Beobachtungsvektor klein. Mehr Strahlen erhöhen die Beobachtungsgröße linear — ein größerer Beobachtungsvektor verlangsamt das Training und erfordert möglicherweise mehr Netzwerkkapazität. Beginne mit 5. Füge nur mehr hinzu, wenn der Agent Hindernisse trifft, die er nicht sehen konnte.

!!! warning "Kollisionsmaske muss zwischen Training und Inferenz übereinstimmen"
    Wenn du Szenengeometrie oder Physikschichtzuweisungen nach dem Export des ONNX-Modells änderst, können die Strahlen für dieselbe Situation unterschiedliche Werte zurückgeben. Trainiere nach strukturellen Szenenänderungen immer neu. Überprüfe während der ONNX-Inferenz in einer Live-Godot-Szene, dass Kollisionsmasken auf den `RayCast3D`-Knoten exakt mit dem übereinstimmen, was während des Trainings verwendet wurde.

---

## 11 · Stretch Goals

Diese Übungen erweitern die Unit und sind optional, aber sehr empfohlen, bevor du weitermachst.

**VecNormalize vs. manuelle Normalisierung**

Trainiere FlyBy zweimal — einmal mit `VecNormalize` (Abschnitt 8), einmal mit allen manuell auf [−1, 1] normalisierten Beobachtungen innerhalb von `get_obs()` in GDScript. Protokolliere beide Läufe in TensorBoard und vergleiche:

- Welcher erreicht eine Belohnung von +50 schneller?
- Welcher ist einfacher zu debuggen, wenn etwas schiefläuft?
- Was passiert, wenn du vergisst, die VecNormalize-Statistiken bei der Inferenz neu zu laden?

**Strahlenanzahl-Ablation**

Trainiere FlyBy mit 3 Strahlen, 5 Strahlen und 9 Strahlen. Halte alle anderen Hyperparameter fest. Messe:

- Schritte bis zum Erreichen einer stabilen positiven Belohnung
- Endgültiger `ep_rew_mean` bei 2 M Schritten

Ab wann hört das Hinzufügen von mehr Strahlen auf, die Leistung zu verbessern? Schadet es jemals?

**HovercraftRacing-Mehrfachagenten**

Verwende das in Unit 7 eingeführte Mehrfachagenten-Setup. Lass zwei HovercraftRacing-Agenten in derselben Szene gegeneinander rennen. Vergleiche:

- Durchschnittliche Rundenzeit vs. ein einzelner Agent, der gegen einen statischen Hindernisparcours fährt
- Verbessert oder verschlechtert der Wettbewerb (ein Gegner zum Ausweichen) die Rundenzeiten?
- Entwickeln die Agenten kooperative oder gegnerische Fahrstile?

Um das Mehrfachagenten-Rennen einzurichten, dupliziere den Hovercraft-`Node3D` (einschließlich seines `AIController3D`-Kindes) und gib jedem Controller eine eindeutige `player_id`. Das `godot-rl-agents`-Plugin übernimmt die agentenbezogene Aktionsverteilung automatisch.

**Belohnungsformungs-Herausforderung**

Entwerfe eine Belohnungsfunktion für FlyBy, die gleichzeitig alle drei dieser Bedingungen erfüllt:

1. Der Agent muss den Checkpoint erreichen (dichte Distanzbelohnung)
2. Der Agent darf sich nicht schneller als 90°/s drehen (Winkelgeschwindigkeitsstrafe)
3. Der Agent muss in unter 30 Sekunden ankommen (Zeitdruckbonus)

Wie gewichtest du diese drei Terme, ohne dass einer die anderen überwältigt? Protokolliere jede Belohnungskomponente separat in TensorBoard — `ep_rew_mean` allein sagt dir nicht, welcher Term dominiert.

---

## Was kommt als Nächstes

**Visuelle Beobachtungen:** Wechsle von Strahlenwerfer-Sensoren zu rohen Pixeln — SubViewport-Pipeline, NatureCNN und Frame-Stacking in Godot.

!!! info "Selbstcheck, bevor du weitermachst"
    Kannst du diese Fragen mit eigenen Worten beantworten?

    1. Warum kann DQN CrossTheRoad trainieren, aber nicht FlyBy — was kann PPO, das DQN nicht kann?
    2. Zwei Raycast-Sensoren liefern Rohwerte: `dist = 47.3` Meter und `velocity = 0.012`. Warum ist das fatal für das neuronale Netz, und wie behebst du es?
    3. Was verfolgt `VecNormalize` pro Umgebung, und warum musst du diese Statistiken mit dem Modell speichern?
    4. Was sagt dir die `train/std` der Aktionsverteilung darüber, wie sicher die Policy ist?
    5. Wähle eine der beiden Umgebungen (FlyBy vs. HovercraftRacing) — was genau an ihrer Belohnungsform macht sie zu einem anderen Lehrbeispiel als die andere?

    Wenn du alle fünf beantworten kannst — bist du bereit.

??? success "Antworten zum Selbstcheck"
    1. DQN wählt eine Aktion per Argmax über eine endliche Menge von Q-Werten und braucht daher einen **diskreten** Aktionsraum. FlyBys Aktionen sind reelle Zahlen (Schub, Lenkung) — PPO löst das, indem es eine **Gaußsche Verteilung** ausgibt und daraus N Floats zieht; dafür hat DQN keinen Mechanismus.
    2. Die beiden Werte unterscheiden sich um den Faktor ~4000 in der **Skala** — die Gradientenaktualisierung wird vom großen Distanzwert dominiert, die Geschwindigkeitsdimension wird praktisch ignoriert. Behebe es an der Quelle: Teile jede Beobachtung in `get_obs()` durch ihr erwartetes Maximum, sodass alles ungefähr in [−1, 1] landet.
    3. `VecNormalize` führt einen **laufenden Mittelwert und eine laufende Standardabweichung** für jede Beobachtungsdimension (und optional für Belohnungen) über alle parallelen Umgebungen hinweg. Die trainierte Policy hat nur normalisierte Eingaben gesehen — bei der Inferenz musst du genau diese Statistiken neu laden, sonst bekommt sie Rohwerte und gibt unsinnige Aktionen aus.
    4. `train/std` ist die Breite der Gaußschen Aktionsverteilung — also wie stark die Policy noch **exploriert**. Ein Start nahe 1,0 mit langsamem Absinken bedeutet wachsende Sicherheit; ein sofortiger Kollaps auf 0 bedeutet verfrühte Gewissheit (füge `--ent_coef` hinzu).
    5. Beispiel — **FlyBy**: Die dichte Belohnung für Checkpoint-Nähe liefert ein einziges, klares Lernsignal — das macht es zum saubereren Lehrbeispiel für die Grundlagen kontinuierlicher 3D-Steuerung. Die Belohnung von HovercraftRacing (Rennposition + Geschwindigkeit) mischt Gegner und Streckenzwänge hinein, ein fehlschlagender Lauf hat dort also weit mehr mögliche Ursachen.

[→ Visuelle Beobachtungen](unit-visual-observations.md)
