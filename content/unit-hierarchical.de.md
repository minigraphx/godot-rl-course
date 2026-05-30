# Hierarchisches RL — Long-Horizon-Aufgaben zerlegen

[← Self-Play](unit-self-play.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~35 min · Training: ~30 min GPU / ~2 Std CPU

---

Long-Horizon-Aufgaben brechen flaches RL. Wenn eine Belohnung erst nach 1 000 Schritten
ankommt, werden Gradientensignale so abgeschwächt, dass ein Standard-PPO-Agent Millionen
von Frames damit verbringen kann, nahe der Startposition herumzuwirbeln. Hierarchisches
Reinforcement Learning (HRL) reduziert das Problem auf eine handhabbare Größe: ein
**Manager** (High-Level-Policy) wählt *Subziele*, und ein **Worker** (Low-Level-Policy)
erreicht sie. Jede Schicht arbeitet auf ihrer eigenen Zeitskala und erhält ihr eigenes
Belohnungssignal, was bedeutet, dass die Credit-Zuweisung handhabbar bleibt.

Diese Unit baut das Konzept aus ersten Prinzipien auf, implementiert einen zweistufigen
Controller in Godot und zeigt dir, wie du die beiden Hälften zusammen trainierst.

**Cross-Unit-Links**

- [Goal-Conditioned RL & HER](unit-her.md) — der Low-Level-Worker IST eine
  goal-conditioned Policy; lies diese Unit vor Abschnitt 3.
- [Multi-Agent RL](unit-07.md) — Manager und Worker können als zwei kooperierende Agenten
  gerahmt werden.
- [Curiosity-Driven Exploration](unit-curiosity.md) — eine günstigere Alternative, wenn die
  Aufgabe nur 1–2 natürliche Engpässe hat.
- [Reward Engineering](unit-reward-engineering.md) — versuche immer zuerst dichtes Shaping,
  bevor du nach HRL greifst.

!!! warning "Versuche zuerst Reward Shaping"
    Hierarchisches RL führt zwei Policies, zwei Lernraten, zwei Belohnungssignale und
    mindestens einen neuen Hyperparameter (das Manager-Schrittintervall *k*) ein. Bevor du
    dich dieser Komplexität verpflichtest, geh zurück zu
    [Reward Engineering](unit-reward-engineering.md) und frage:

    - Kann ich einen Shaping-Term für das Erreichen jedes Zwischenraums hinzufügen?
    - Kann ich Curiosity hinzufügen, um die Exploration über die erste Tür hinaus zu treiben?

    Wenn dichtes Shaping plus Curiosity immer noch plateauiert, *dann* ist HRL das richtige
    Werkzeug.

!!! info "Drei Wege, deine KI zu beobachten"
    - **Godot-Szene** — Subziel-Marker ändern in Echtzeit die Farbe, wenn die High-Level-Policy
      Ziele wechselt; grün = aktuelles Subziel, grau = inaktiv.
    - **TensorBoard** — zwei separate Belohnungskurven: `reward/high_level` (spärlich,
      schrittweise um *k*) und `reward/low_level` (dicht, jeden Schritt); beobachte, wie sie
      mit unterschiedlichen Geschwindigkeiten konvergieren.
    - **Hierarchisches Entscheidungsdiagramm** — eine scrollende Zeitleisten-Spur, die zeigt,
      welche Option zu jedem Zeitschritt aktiv ist, sodass du erkennst, wann der Manager
      unentschlossen ist oder der Worker sauberes Terminieren verfehlt.

---

## 1 · Das Long-Horizon-Problem

### Warum flaches PPO bei 1 000+ Schritten scheitert

Die Policy-Gradient-Update von PPO ist:

```
∇J(θ) = E[ ∇ log π(aₜ|sₜ) · Aₜ ]
```

Der Advantage `Aₜ` wird aus dem diskontierten Return berechnet. Mit γ = 0,99 und einer
Belohnung, die erst bei Schritt 1 000 ankommt, ist der Diskontfaktor am Anfang der Episode
`0,99^1000 ≈ 2,5 × 10⁻⁵`. Dieser Faktor multipliziert jedes Gradient-Update für Aktionen,
die nahe dem Anfang der Episode gemacht wurden. In der Praxis erhält die Policy fast kein
nützliches Lernsignal für frühe Entscheidungen — sie kann nicht erkennen, welche der
tausend Aktionen, die sie gemacht hat, zum schließlichen Erfolg beigetragen hat.

Das ist das **Credit-Assignment**-Problem. Es ist keine PPO-spezifische Einschränkung; es
betrifft jeden On-Policy-Algorithmus, und es ist auch bei Off-Policy-Methoden schwerwiegend
(Replay-Buffer-Einträge aus dem frühen Teil einer Episode tragen winzige Gradienten bei).

### Konkretes Beispiel: Multi-Raum-Navigation in Godot

Stell dir eine Godot-Szene mit drei Räumen vor, verbunden durch schmale Türen.

```
┌────────┐   door 1   ┌────────┐   door 2   ┌────────┐
│        │◄──────────►│        │◄──────────►│        │
│  Room  │            │  Room  │            │  Room  │
│   A    │            │   B    │            │   C    │
│ (start)│            │        │            │  ★GOAL │
└────────┘            └────────┘            └────────┘
```

Der Agent startet in Raum A. Die einzige Belohnung ist `+1` für das Einsammeln des Sterns
in Raum C. Ein flacher PPO-Agent muss zufällig durch zwei Türen stolpern — eine Konjunktion
unwahrscheinlicher Ereignisse — bevor er jemals eine positive Belohnung sieht. Mit kleinen
Türen und einem großen Raum kann das in einer vernünftigen Anzahl von Trainingsschritten
nie passieren.

**Typischer Vergleich von Lernkurven**

| Algorithmus | Schritte bis zur ersten Belohnung | Finale Erfolgsrate |
|---|---|---|
| Flat PPO (spärlich) | Oft nie in 10 M Schritten | < 5 % |
| Flat PPO + Curiosity | ~3 M Schritte | 40–60 % |
| HRL (Manager + HER-Worker) | ~800 K Schritte | 85–95 % |

(Zahlen sind illustrativ; exakte Werte hängen von Raumgröße und Türbreite ab.)

### Wann du HRL brauchen könntest

Wende diese Checkliste an, bevor du zu hierarchischen Methoden greifst:

1. Episodenlänge überschreitet durchschnittlich **500 Schritte**.
2. Belohnung ist **spärlich** — weniger als ein Belohnungsereignis pro 100 Schritte.
3. Die Aufgabe hat **natürliche Teilaufgaben**, die du einem Menschen in jeweils einem Satz
   beschreiben könntest („geh zur Tür", „öffne die Tür", „überquere zum nächsten Raum").
4. Dichtes Reward Shaping und Curiosity wurden **bereits versucht** und sind plateauiert.

Wenn alle vier zutreffen, ist HRL ein vernünftiger nächster Schritt.

---

## 2 · Das Options-Framework

Das Options-Framework (Sutton, Precup & Singh 1999) ist der grundlegende Formalismus für HRL.
Es ersetzt primitive Aktionen durch **Makro-Aktionen** namens *Optionen*.

### Formale Definition

Eine Option `ω` ist ein Tripel `(I, π, β)`:

| Komponente | Symbol | Bedeutung |
|---|---|---|
| Initiationsmenge | `I ⊆ S` | Zustände, in denen die Option gestartet werden kann |
| Intra-Option-Policy | `π: S × A → [0,1]` | Die Policy, die ausgeführt wird, während die Option läuft |
| Terminierungsbedingung | `β: S → [0,1]` | Wahrscheinlichkeit, in jedem Zustand zu terminieren |

Der **Manager** wählt eine Option `ω` aus der Menge `Ω`. Die Kontrolle geht dann an `π_ω`
über bis zur Terminierung, an welchem Punkt der Manager erneut wählt.

### Semi-MDP: zwei Zeitskalen

Die Standard-MDP tickt bei jedem primitiven Schritt *t*. Das Options-Framework erzeugt
eine **Semi-MDP** (SMDP) auf Manager-Ebene, bei der ein „Schritt" die gesamte Dauer einer
Option überspannt — potenziell viele primitive Schritte.

```
Primitive time:   t₀  t₁  t₂  t₃  t₄  t₅  t₆  t₇  t₈  t₉  …
                  │←── option ω₁ ──────►│←── option ω₂ ──►│ …
Manager time:     τ₀                     τ₁                 τ₂
```

Der Manager sieht nie die dazwischenliegenden primitiven Schritte; er sieht nur Zustände
an Optionsgrenzen. Das verkürzt den effektiven Episoden-Horizont auf Manager-Ebene massiv.

### Navigationsoptionen

Für das Drei-Räume-Beispiel definiere vier Optionen:

| Optionsname | Initiationsmenge | Terminiert wenn |
|---|---|---|
| `go_to_door_1` | Raum A | Agent innerhalb 30 px von Tür 1 |
| `go_to_door_2` | Raum B | Agent innerhalb 30 px von Tür 2 |
| `go_to_door_3` | Raum C | Agent innerhalb 30 px vom Ziel |
| `go_to_goal` | Raum C | Agent sammelt den Stern ein |

Der Manager wählt, welche Option zu aktivieren ist; die interne Policy jeder Option
übernimmt die Bewegung von Moment zu Moment. Der Manager muss jetzt nur ein 4-Schritt-Problem
lösen statt eines 1 000-Schritt-Problems.

### Hand-kodierte vs gelernte Optionen

Optionen können sein:

- **Hand-kodiert** — du definierst `I`, `π` und `β` explizit (am einfachsten zu starten).
- **Gelernt** — der Agent entdeckt nützliche Optionen durch Optionsentdeckungs-Algorithmen
  (z. B. Eigenoptions, Covering Options). Das ist ein aktives Forschungsgebiet und außerhalb
  des Umfangs dieser Unit.

Für die praktische Übung starten wir mit hand-kodierten Optionen und trainieren nur den
Manager.

---

## 3 · Goal-Conditioned Low-Level-Policy

Hand-Kodieren von Optionen funktioniert für kleine, strukturierte Aufgaben. Für größere
Aufgaben wollen wir, dass die Low-Level-Policy *allgemein* ist — fähig, jedes Subziel im
Beobachtungsraum zu erreichen — sodass die High-Level-Policy beliebige Ziele vorschlagen
kann.

### Die praktische HRL-Schleife

```
Every k primitive steps:
    manager observes sₜ
    manager selects subgoal gₜ  (a position or feature vector)

Every primitive step:
    worker observes [sₜ, gₜ]
    worker selects action aₜ
    worker receives intrinsic reward r_worker = -‖pos(sₜ₊₁) - gₜ‖

Every k steps:
    manager receives extrinsic reward r_manager = sum of environment rewards
    manager updates its policy
```

Der Worker ist eine **goal-conditioned Policy** — er nimmt sowohl die aktuelle Beobachtung
als auch das Subziel als Eingabe. Das Training des Workers mit HER macht ihn robust gegen
Subziele, die er im frühen Training selten erreicht.

### Verbindung zu unit-her.md

[Hindsight Experience Replay (HER)](unit-her.md) ist die Standard-Trainingsmethode für die
Low-Level-Policy im praktischen HRL:

- Für jede Trajektorie, die der Worker erzeugt, relabelt HER retroaktiv erfolglose Episoden
  mit dem Zustand, den der Worker *tatsächlich* erreicht hat, als „Ziel".
- Das gibt dem Worker ein dichtes Lernsignal, selbst wenn er das vom Manager beabsichtigte
  Subziel verfehlt.
- Ohne HER kann der Worker Millionen von Schritten brauchen, um grundlegende Navigation zu
  lernen; mit HER konvergiert er typischerweise in Zehntausenden.

**Schlüsselimplikation**: trainiere zuerst die Low-Level-Policy mit HER auf einem
Random-Subgoal-Curriculum, frier sie dann ein (oder trainiere sie weiter feinabgestimmt mit
einer niedrigeren Lernrate), während du die High-Level-Policy trainierst.

### Zeitskalen-Mismatch und Nichtstationarität

Ein subtiles Problem entsteht, weil die Low-Level-Policy sich weiter verbessert, während die
High-Level-Policy trainiert. Aus Sicht der High-Level-Policy ist die „Umgebung" (die den
Worker einschließt) nichtstationär — dasselbe Subziel `g` mag bei Schritt 500 K des Trainings
leichter zu erreichen sein als bei Schritt 100 K. Das ist die zentrale technische
Herausforderung von HRL und wird in Abschnitt 5 durch HIRO adressiert.

### Code-Skizze: die zweistufige Trainingsschleife

```python
# Pseudocode — not Godot-specific
MANAGER_INTERVAL = 20  # k

obs = env.reset()
subgoal = manager.select_subgoal(obs)

for step in range(MAX_STEPS):
    # Worker acts
    worker_obs = concat(obs, subgoal)
    action = worker.predict(worker_obs)
    next_obs, env_reward, done, info = env.step(action)

    # Worker reward: negative distance to subgoal
    worker_reward = -distance(next_obs["position"], subgoal)
    worker.store_transition(worker_obs, action, worker_reward, next_obs)
    worker.train_step()

    # Manager acts every k steps
    if step % MANAGER_INTERVAL == 0:
        manager_reward = sum_of_env_rewards_since_last_manager_step
        manager.store_transition(obs_at_last_decision, subgoal, manager_reward, next_obs)
        subgoal = manager.select_subgoal(next_obs)
        manager.train_step()

    obs = next_obs
    if done:
        obs = env.reset()
        subgoal = manager.select_subgoal(obs)
```

---

## 4 · Godot Multi-Raum-Navigationsbeispiel

### Szenenaufbau

Erstelle eine Godot 4-Szene mit der folgenden Knotenhierarchie:

```
MultiRoomEnv (Node3D)
├── Room_A (StaticBody3D + CollisionShape3D)
├── Room_B (StaticBody3D + CollisionShape3D)
├── Room_C (StaticBody3D + CollisionShape3D)
├── Door_1 (Area3D)           ← triggers room transition detection
├── Door_2 (Area3D)
├── Goal (Area3D + MeshInstance3D)  ← collectible star
├── Agent (CharacterBody3D)
│   ├── HighLevelAIController
│   └── LowLevelAIController
└── SubgoalMarker (MeshInstance3D)  ← visual indicator, changes colour
```

### Beobachtungsräume

**High-Level (Manager) Beobachtung** — kompakte, raumweite Information:

| Feld | Typ | Beschreibung |
|---|---|---|
| `current_room` | int (0–2) | In welchem Raum sich der Agent gerade befindet |
| `door_1_pos` | Vector2 | 2D-Position von Tür 1 im Weltraum |
| `door_2_pos` | Vector2 | 2D-Position von Tür 2 im Weltraum |
| `goal_pos` | Vector2 | 2D-Position des Sterns |
| `steps_since_last_decision` | float (normalisiert) | Zeitdrucksignal |

**Low-Level (Worker) Beobachtung** — präzise, bewegungsweite Information:

| Feld | Typ | Beschreibung |
|---|---|---|
| `agent_pos` | Vector2 | Agentenposition im Weltraum |
| `agent_vel` | Vector2 | Agentengeschwindigkeit |
| `subgoal_pos` | Vector2 | Aktuelle Subzielposition, gesetzt vom Manager |
| `subgoal_delta` | Vector2 | `subgoal_pos - agent_pos` (redundant, hilft aber beim Lernen) |
| `wall_distances` | float[4] | Raycast-Distanzen N/S/O/W |

### Aktionsräume

- **High-Level**: `Discrete(4)` — Indizes mappen auf `[door_1, door_2, goal, explore]`
- **Low-Level**: `Box([-1,-1], [1,1])` — kontinuierliche 2D-Bewegung (x-, z-Geschwindigkeitsziele)

### GDScript: zweistufiges AIController-Muster

```gdscript
# HighLevelAIController.gd
extends AIController3D

const MANAGER_INTERVAL := 20  # primitive steps between manager decisions
var _step_counter := 0
var _current_subgoal_index := 0
var _subgoal_positions: Array[Vector3] = []

func _ready() -> void:
    # Subgoal positions are set by the scene after doors are placed
    _subgoal_positions = [
        get_node("../Door_1").global_position,
        get_node("../Door_2").global_position,
        get_node("../Goal").global_position,
        Vector3.ZERO,  # "explore" — low-level falls back to curiosity
    ]

func get_obs() -> Array:
    var agent := get_node("../Agent") as CharacterBody3D
    return [
        _current_room_index(),
        get_node("../Door_1").global_position.x,
        get_node("../Door_1").global_position.z,
        get_node("../Door_2").global_position.x,
        get_node("../Door_2").global_position.z,
        get_node("../Goal").global_position.x,
        get_node("../Goal").global_position.z,
        float(_step_counter) / float(MANAGER_INTERVAL),
    ]

func get_action_space() -> Dictionary:
    return {
        "subgoal": {"size": 4, "action_type": "discrete"}
    }

func set_action(action: Dictionary) -> void:
    _step_counter += 1
    if _step_counter >= MANAGER_INTERVAL:
        _step_counter = 0
        _current_subgoal_index = action["subgoal"]
        _broadcast_subgoal(_subgoal_positions[_current_subgoal_index])

func _broadcast_subgoal(subgoal: Vector3) -> void:
    # Push subgoal to the low-level controller and update visual marker
    var low_level := get_node("../LowLevelAIController") as LowLevelAIController
    low_level.current_subgoal = subgoal
    get_node("../SubgoalMarker").global_position = subgoal
    _update_subgoal_colour(_current_subgoal_index)

func _update_subgoal_colour(idx: int) -> void:
    var colours := [Color.RED, Color.ORANGE, Color.GREEN, Color.BLUE]
    var marker := get_node("../SubgoalMarker") as MeshInstance3D
    var mat := marker.get_surface_override_material(0) as StandardMaterial3D
    if mat:
        mat.albedo_color = colours[idx]

func _current_room_index() -> int:
    var agent_pos := get_node("../Agent").global_position
    # Simplified: rooms are laid out along the X axis
    if agent_pos.x < -10.0:
        return 0
    elif agent_pos.x < 10.0:
        return 1
    else:
        return 2
```

```gdscript
# LowLevelAIController.gd
extends AIController3D

var current_subgoal := Vector3.ZERO

func get_obs() -> Array:
    var agent := get_node("../Agent") as CharacterBody3D
    var delta := current_subgoal - agent.global_position
    return [
        agent.global_position.x,
        agent.global_position.z,
        agent.velocity.x,
        agent.velocity.z,
        current_subgoal.x,
        current_subgoal.z,
        delta.x,
        delta.z,
        _raycast_distance(Vector3.LEFT),
        _raycast_distance(Vector3.RIGHT),
        _raycast_distance(Vector3.FORWARD),
        _raycast_distance(Vector3.BACK),
    ]

func get_action_space() -> Dictionary:
    return {
        "move": {"size": 2, "action_type": "continuous"}
    }

func set_action(action: Dictionary) -> void:
    var agent := get_node("../Agent") as CharacterBody3D
    var move := action["move"] as Array
    agent.velocity.x = move[0] * 5.0
    agent.velocity.z = move[1] * 5.0

func get_reward() -> float:
    # Intrinsic: negative distance to current subgoal
    var agent := get_node("../Agent") as CharacterBody3D
    var dist := agent.global_position.distance_to(current_subgoal)
    return -dist * 0.01  # scale so reward stays in [-1, 0]

func _raycast_distance(direction: Vector3) -> float:
    var space := get_world_3d().direct_space_state
    var query := PhysicsRayQueryParameters3D.create(
        get_node("../Agent").global_position,
        get_node("../Agent").global_position + direction * 10.0
    )
    var result := space.intersect_ray(query)
    if result.is_empty():
        return 1.0  # normalised: 10 m = 1.0
    return result["position"].distance_to(get_node("../Agent").global_position) / 10.0
```

### Trainingsablauf

Trainiere in zwei Phasen, um das Henne-Ei-Problem zu vermeiden (Manager braucht einen
funktionierenden Worker; Worker braucht Subziele zum Üben):

**Phase 1 — Trainiere den Low-Level-Worker isoliert (≈ 1 M Schritte)**

```bash
# Use a dedicated scene that spawns random subgoal targets
gdrl train --config config/low_level_her.yaml
```

In `low_level_her.yaml`:
```yaml
trainer_type: ppo
n_envs: 8
use_her: true
her_replay_k: 4
reward_signal:
  - type: distance_to_subgoal
    weight: 1.0
```

**Phase 2 — Friere Worker ein, trainiere Manager (≈ 500 K Schritte)**

```bash
gdrl train --config config/high_level_ppo.yaml \
    --load-worker-checkpoint checkpoints/low_level_her_final.ckpt
```

Optional Feinabstimmung beider zusammen mit einer niedrigeren Lernrate für den Worker.

---

## 5 · HIRO — Off-Policy HRL

Der zweiphasige Trainingsansatz oben ist praktisch, aber suboptimal, weil die High-Level-Policy
auf einem eingefrorenen Worker trainiert, der nicht unter der tatsächlichen
Subziel-Verteilung des Managers trainiert wurde. **HIRO** (Nachum et al. 2018, „Data-Efficient
Hierarchical Reinforcement Learning") löst das mit **Off-Policy-Korrektur**.

### Das Nichtstationaritäts-Problem

Wenn der Manager Subziel `g` zum Zeitpunkt `τ` auswählt, ist die Low-Level-Policy `π_worker`,
die `g` ausführen wird, die *aktuelle* Policy zum Zeitpunkt `τ`. Aber bis der Replay-Buffer-Eintrag
des Managers für das Training gesampelt wird (viel später), hat sich `π_worker` geändert. Der
gespeicherte `(state, subgoal, reward, next_state)`-Übergang wurde von einem *alten* Worker
erzeugt, aber der Policy-Gradient des Managers wird unter dem *aktuellen* Worker ausgewertet.
Dieser Mismatch macht naives Off-Policy-Training für den Manager inkorrekt.

### HIROs Korrektur

HIRO relabelt gespeicherte High-Level-Übergänge, indem es fragt: „Gegeben die Sequenz von
Zuständen, die der Worker tatsächlich besucht hat (`s_τ, s_τ₊₁, …, s_τ₊ₖ`), welches Subziel
`g'` hätte der *aktuelle* Worker zu erreichen versucht, um dieselben Aktionen zu produzieren?"

Formal findet es:

```
g* = argmax_{g'} Σₜ log π_worker(aₜ | sₜ, g' + sₜ - sτ)
```

Dieses re-labelte `g*` ersetzt das ursprüngliche `g` im Replay-Buffer und macht den
gespeicherten Übergang approximativ konsistent mit dem aktuellen Worker.

### Architektur

Beide Ebenen verwenden **SAC** (Soft Actor-Critic) für Off-Policy-Training mit kontinuierlichen
Aktionen:

```
High-level (manager):   SAC with subgoal space = obs space
Low-level (worker):     SAC with goal-conditioned obs + intrinsic reward
Off-policy correction:  Applied when sampling manager's replay buffer
```

### Verfügbarkeit in SB3

Stable-Baselines3 enthält HIRO nicht. Optionen:

- **rl-games** — GPU-beschleunigt; einige HRL-Unterstützung über Custom Callbacks.
- **stable-baselines3-contrib** — prüfe auf Community-beigetragenes HIRO; Verfügbarkeit
  variiert je nach Version.
- **Referenzimplementierung** — Nachum et al.s ursprünglicher TensorFlow-Code ist auf GitHub
  (`tensorflow/models/research/efficient-hrl`); ein PyTorch-Port existiert als
  `HIRO-PyTorch` von verschiedenen Community-Autoren.
- **Für diesen Kurs** — wir verwenden den zweiphasigen PPO/HER-Ansatz (Abschnitt 4), der
  einfacher und ausreichend für die Multi-Raum-Aufgabe ist.

---

## 6 · Feudale Netzwerke (Konzeptionell)

Feudal Networks (Vezhnevets et al. 2017, „FeUdal Networks for Hierarchical Reinforcement
Learning") gehen die Manager-Worker-Idee einen Schritt weiter: der Manager arbeitet in einem
**latenten Feature-Raum** statt im rohen Beobachtungsraum.

### Architektur

```
Observation sₜ  →  Perception module  →  feature vector zₜ
                                              │
                              ┌───────────────┤
                              │               │
                         Manager            Worker
                    (slow, latent space)  (fast, raw obs)
                              │               │
                     direction dₜ in z-space  │
                              └───────────────►
                                     Worker reward:
                                     cos(zₜ₊c - zₜ, dₜ)
                                     (did worker move in
                                      the direction manager wanted?)
```

### Schlüsselideen

- Der **Manager** gibt eine *Richtung* im Feature-Raum aus — „bewege den Feature-Vektor in
  diese Richtung über die nächsten `c` Schritte".
- Der **Worker** erhält eine Belohnung proportional zur Kosinusähnlichkeit zwischen der
  Richtung, in die er sich tatsächlich im Feature-Raum bewegt hat, und der Richtung, die der
  Manager angefordert hat.
- Der Manager muss nie über Pixel- oder Positions-Details nachdenken; er denkt über
  High-Level-Strukturen nach, die in der geteilten Feature-Repräsentation erfasst sind.

### Warum nützlich

- Der Feature-Raum ist typischerweise viel niedriger-dimensional und glatter als rohe
  Beobachtungen, was das Lernproblem des Managers einfacher macht.
- Das Ziel des Managers (ein Richtungsvektor) ist robuster gegen Änderungen in der Policy des
  Workers als ein spezifisches Positionsziel, was das Nichtstationaritäts-Problem teilweise
  adressiert.

### Praktischer Status

Feudal Networks sind konzeptionell elegant, aber schwer zu tunen. Das geteilte
Perzeptionsmodul, die Zeitskala `c` und das Kosinus-Belohnungssignal interagieren alle. Für
die meisten Godot-Projekte ist der einfachere Goal-Conditioned + HER-Ansatz (Abschnitt 3–4)
oder HIRO (Abschnitt 5) vorzuziehen.

---

## 7 · Wann hierarchisches RL NICHT zu verwenden ist

HRL ist ein Präzisionswerkzeug, kein Default. Bevor du es wählst, geh diese Checkliste durch:

### Entscheidungsdiagramm

```
Episode < 500 steps?
  YES → Use flat PPO (maybe + curiosity). Stop here.
  NO  ↓
Reward dense enough (> 1 event per 100 steps)?
  YES → Use reward shaping (see unit-reward-engineering.md). Stop here.
  NO  ↓
Fewer than 3 natural subtasks?
  YES → Use curiosity (see unit-curiosity.md) + sparse reward. Stop here.
  NO  ↓
Already tried shaping + curiosity?
  NO  → Try those first. Stop here.
  YES → Consider HRL.
```

### Spezifische Warnungen

**Debugging-Schwierigkeit** — wenn etwas schiefgeht, musst du bestimmen:
- Wählt der Manager schlechte Subziele?
- Scheitert der Worker daran, das Subziel zu erreichen?
- Stehen die beiden Belohnungssignale im Konflikt?
- Ist das Manager-Intervall `k` zu kurz oder zu lang?

Jede Frage erfordert separate Analyse. Plane extra Debug-Zeit ein.

**Hyperparameter-Explosion** — HRL fügt mindestens hinzu:
- Manager-Lernrate (gewöhnlich 10× niedriger als Worker)
- Manager-Schrittintervall `k`
- Subzielraum-Dimensionalität
- Skalierungskoeffizient für intrinsische Belohnung

**Lokale Optima im Manager** — der Manager kann lernen, nur die Subziele auszuwählen, die der
Worker bereits gemeistert hat, und ignoriert schwerere Subziele, die es dem Agenten erlauben
würden, weiter voranzukommen. Überwache die Optionen-/Subziel-Nutzungshäufigkeit während des
Trainings.

**Praktische Faustregel**: wenn die Aufgabe weniger als 3 natürliche Teilaufgaben hat, die du
eindeutig beschreiben kannst, wird flaches PPO + Curiosity HRL wahrscheinlich in der
Wall-Clock-Trainingszeit erreichen oder übertreffen. HRLs Vorteil wächst mit der
Aufgabenkomplexität und der Anzahl der Teilaufgaben.

---

## 8 · Viz-Checkpoint

Gute Visualisierung ist essentiell zum Debuggen eines zweistufigen Systems. Ohne sie fliegst
du blind.

### In-Szenen-Visualisierung (Godot)

Färbe den `SubgoalMarker`-Knoten danach ein, welche Option oder welches Subziel gerade aktiv
ist:

| Aktives Subziel | Markerfarbe |
|---|---|
| go_to_door_1 | Rot |
| go_to_door_2 | Orange |
| go_to_goal | Grün |
| explore | Blau |

Füge ein zweites schwebendes Label über dem Agenten hinzu, das den aktuellen Optionsnamen und
die Anzahl der Schritte seit dem letzten Manager-Wechsel anzeigt. Eine lang laufende Option
deutet darauf hin, dass der Worker feststeckt; schnelles Wechseln deutet darauf hin, dass der
Manager unsicher ist.

Du kannst auch eine Linie vom Agenten zum aktuellen Subziel mit `draw_line` in einem
`_draw()`-Override auf einem `CanvasLayer` zeichnen — das macht sofort offensichtlich, ob der
Worker in die richtige Richtung steuert.

### TensorBoard-Metriken

Logge diese Metriken während des Trainings:

```python
# High-level metrics (logged every manager step)
writer.add_scalar("reward/high_level", manager_reward, global_step)
writer.add_scalar("manager/option_entropy", option_entropy, global_step)
writer.add_scalar("manager/steps_per_option", avg_option_duration, global_step)

# Low-level metrics (logged every primitive step)
writer.add_scalar("reward/low_level", worker_reward, global_step)
writer.add_scalar("worker/distance_to_subgoal", subgoal_dist, global_step)
writer.add_scalar("worker/action_std", action_std, global_step)
```

**Wie gutes HRL auf TensorBoard aussieht**:

- `reward/low_level` konvergiert zuerst (gewöhnlich innerhalb der ersten 200 K Schritte).
- `reward/high_level` beginnt erst sich zu verbessern, nachdem der Worker zuverlässig ist.
- `manager/steps_per_option` steigt über die Zeit — der Manager lernt, sich länger auf
  Subziele festzulegen, je besser der Worker beim Erreichen wird.
- `worker/action_std` sinkt über die Zeit — die Policy des Workers wird entschiedener.

### Die hierarchische Entscheidungs-Zeitleiste

Zeichne ein horizontales Strip-Chart mit Zeit auf der X-Achse und Optionsindex auf der
Y-Achse, farblich nach Option ausgefüllt. Ein gut trainierter Manager zeigt lange
zusammenhängende Blöcke (er verpflichtet sich zu einem Subziel, der Worker erreicht es, dann
zieht der Manager weiter). Ein schlecht trainierter Manager zeigt schnelles Flackern zwischen
Optionen.

```python
# Pseudocode for logging option timeline
timeline = []  # list of (timestep, option_index)
for step, option in enumerate(option_history):
    timeline.append((step, option))

# Visualise with matplotlib or log to W&B as a custom plot
```

---

## 9 · Stretch Goals

Diese Übungen erweitern die Unit für Studierende, die tiefer gehen möchten.

### Stretch 1 — Hand-kodierte Optionen

Implementiere das vollständige Options-Framework in Godot mit vier hand-kodierten Optionen:

- `go_north` — bewege dich zum nördlichsten Punkt des aktuellen Raums
- `go_south` — bewege dich zum südlichsten Punkt des aktuellen Raums
- `go_east` — bewege dich durch die östlichste Tür (falls vorhanden)
- `go_west` — bewege dich durch die westlichste Tür (falls vorhanden)

Jede Option hat ein hart kodiertes `β` (innerhalb von 50 px vom Ziel terminieren) und ein
hart kodiertes `π` (PID-Controller zum Ziel). Trainiere nur die Meta-Policy (Manager) mit
flachem PPO und der SMDP-Formulierung. Vergleiche die Konvergenzgeschwindigkeit mit der
vollständig gelernten Variante.

### Stretch 2 — Flat PPO + Curiosity vs HRL

Führe einen kontrollierten Vergleich auf der Multi-Raum-Aufgabe durch:

1. **Baseline A**: flat PPO + nur spärliche Belohnung.
2. **Baseline B**: flat PPO + ICM-Curiosity (siehe [unit-curiosity.md](unit-curiosity.md)).
3. **HRL**: Manager + HER-Worker (Abschnitt 4 dieser Unit).

Messe:
- Schritte bis zur ersten erfolgreichen Episode
- Erfolgsrate bei 1 M, 2 M und 5 M Schritten
- Wall-Clock-Trainingszeit für 5 M Schritte

Plotte alle drei auf denselben Achsen und teile deine Ergebnisse im Kurs-Discord.

### Stretch 3 — Dreistufige Hierarchie

Füge eine dritte Ebene über dem Manager hinzu: einen **Meta-Manager**, der auswählt, welcher
*Raum* als nächstes erkundet werden soll. Der Meta-Manager arbeitet auf einer noch langsameren
Zeitskala (alle 100 primitiven Schritte). Seine Beobachtung ist nur die Menge der besuchten
Räume und die aktuelle Erfolgsrate für jeden Raum. Seine Aktion ist `Discrete(3)` — wähle den
nächsten zu priorisierenden Raum.

Das erzeugt den vollen dreistufigen Stack:

```
Meta-manager (every 100 steps) → which room?
    Manager (every 20 steps)   → which subgoal?
        Worker (every step)    → which primitive action?
```

Beobachte, wie der Meta-Manager lernt, Trainingsaufwand auf Räume zu lenken, die der Agent
noch nicht gemeistert hat.

---

[← Self-Play](unit-self-play.md) · [Kursstartseite](index.md) · [→ Multi-Task RL](unit-multitask.md)
