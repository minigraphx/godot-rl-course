# Lokomotion-Agenten — Walker, Crawler & Worm in Godot

Hast du **AI Warehouse** oder andere Unity-ML-Agents-Showcases gesehen, sind die Demos, die am meisten Aufmerksamkeit bekommen, die zur Lokomotion (locomotion): ein Biped lernt das Gehen von Grund auf, ein Vierbeiner entdeckt einen Trab, ein Wurm findet heraus, wie er sich schlängelt. Die wirken wie Magie. Sind sie nicht — das sind Reward-Design und Physik, und du kannst dasselbe in Godot bauen.

Diese Unit zeigt dir wie.

[← Roboter-Observations & Sensoren](unit-robotics.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~35 min · Training: ~45 min GPU / ~3 h CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    Godot (geht der Agent vorwärts oder klappt er sofort zusammen?) · TensorBoard (`rollout/ep_rew_mean` sollte über 5–10M Steps langsam steigen — Lokomotion lernt langsam) · Viz-Checkpoint nach jeweils 1M Steps: beobachte, wie sich der Gangstil über das Training ändert

---

## 1 · Was Lokomotion-RL eigentlich ist

Lokomotion-RL trainiert eine Policy, **eine Kette von Gelenken so zu regeln**, dass sich der Körper in eine gewünschte Richtung bewegt. Es gibt keinen fest einprogrammierten Gang — der Agent entdeckt einen von Grund auf, nur über das Reward-Signal.

| Unity-ML-Agents-Beispiel | Godot-Äquivalent | Schlüsselherausforderung |
|------------------------|------------------|---------------|
| Walker (Biped) | `RigidBody3D`-Torso + 2 Beine via `Generic6DOFJoint3D` | Balance halten und vorwärts bewegen |
| Crawler (Vierbeiner, kein Rumpf) | 4 unabhängige Glieder auf gemeinsamer Basis | Koordination ohne zentralen Körper |
| Worm | Kette von `RigidBody3D`-Segmenten, `HingeJoint3D` | Bodenreibung ohne Beine ausnutzen |
| Ant (4 Beine + Rumpf) | Rumpf + 4 Beine, je 2 Gelenke | Hochdimensionale Gelenkregelung |

Alle vier nutzen dasselbe Grundrezept: **kontinuierliche Gelenkregelung + geformte Lokomotion-Belohnung**. Ändere Körpergeometrie und Gelenkanzahl; behalte die Belohnungsstruktur.

---

## 2 · Szenenaufbau — der Biped-Walker

Baue diese Szene Schritt für Schritt. Am Ende hast du einen trainierbaren Biped.

### 2.1 · Körperteile

Erzeuge eine `Node3D`-Szene namens `Walker`. Darin:

| Node | Typ | Größe (m) | Position |
|------|------|----------|----------|
| `Torso` | `RigidBody3D` | 0,4 × 0,6 × 0,2 | (0, 1,2, 0) |
| `UpperLegL` | `RigidBody3D` | 0,12 × 0,35 × 0,12 | (−0,15, 0,85, 0) |
| `LowerLegL` | `RigidBody3D` | 0,10 × 0,35 × 0,10 | (−0,15, 0,48, 0) |
| `FootL` | `RigidBody3D` | 0,20 × 0,06 × 0,10 | (−0,12, 0,27, 0) |
| `UpperLegR` | `RigidBody3D` | 0,12 × 0,35 × 0,12 | (0,15, 0,85, 0) |
| `LowerLegR` | `RigidBody3D` | 0,10 × 0,35 × 0,10 | (0,15, 0,48, 0) |
| `FootR` | `RigidBody3D` | 0,20 × 0,06 × 0,10 | (0,12, 0,27, 0) |

Gib jedem eine `CollisionShape3D` passender Größe. Massen setzen: Torso = 8 kg, Oberschenkel = 2 kg, Unterschenkel = 1,5 kg, Füße = 0,8 kg.

### 2.2 · Gelenke

Zwischen jedem Paar verbundener Teile ein `Generic6DOFJoint3D` einfügen. Alle Translationsachsen sperren. Winkelgrenzen setzen:

| Gelenk | Verbindet | Winkelgrenzen (rad) |
|-------|----------|---------------------|
| `HipL` | Torso ↔ UpperLegL | X: [−1,0, 1,0], Y: [−0,3, 0,3], Z: [−0,5, 0,5] |
| `KneeL` | UpperLegL ↔ LowerLegL | X: [0,0, 1,8] (nur Vorwärtsbeugung) |
| `AnkleL` | LowerLegL ↔ FootL | X: [−0,6, 0,6] |
| `HipR` | Torso ↔ UpperLegR | X: [−1,0, 1,0], Y: [−0,3, 0,3], Z: [−0,5, 0,5] |
| `KneeR` | UpperLegR ↔ LowerLegR | X: [0,0, 1,8] |
| `AnkleR` | LowerLegR ↔ FootR | X: [−0,6, 0,6] |

Aktiviere Motoren an jeder Winkelachse, die der Agent regeln soll. Setze `PARAM_ANGULAR_MOTOR_FORCE_LIMIT` auf 40 N·m (Hüfte), 30 N·m (Knie), 15 N·m (Knöchel).

### 2.3 · Kontaktsensoren

Füge an der Unterseite jedes Fußes einen `Area3D` ein. Verbinde `body_entered` → ein Flag `foot_contact` in einem kleinen Skript an dieser Area3D. Das fließt in Observation und Reward.

### 2.4 · AIController und Sync

`AIController3D`- und `Sync`-Nodes an die Szenenwurzel anhängen. `reset()` und `get_reward()` wie üblich verdrahten.

---

## 3 · Observation-Space

Folge dem egozentrischen Muster aus der [Roboter-Observations-Unit](unit-robotics.md). Für den Biped:

```gdscript
extends AIController3D

@onready var torso        = $Torso
@onready var upper_leg_l  = $UpperLegL
@onready var lower_leg_l  = $LowerLegL
@onready var foot_l_body  = $FootL
@onready var upper_leg_r  = $UpperLegR
@onready var lower_leg_r  = $LowerLegR
@onready var foot_r_body  = $FootR

@onready var joints = {
    "hip_l":   $HipL,
    "knee_l":  $KneeL,
    "ankle_l": $AnkleL,
    "hip_r":   $HipR,
    "knee_r":  $KneeR,
    "ankle_r": $AnkleR,
}

@onready var foot_l = $FootContactL   # Area3D flag node
@onready var foot_r = $FootContactR

const MAX_SPEED     = 5.0    # m/s
const MAX_ANG_VEL   = 4.0    # rad/s
const MAX_JOINT_VEL = 8.0    # rad/s

var target_speed := 2.0      # m/s forward — set per episode or fixed

# Returns the x-axis rotation of `child` relative to `parent` in parent-local space.
# Use this to read joint angles — NOT Generic6DOFJoint3D parameters, which are
# static limits, not the current angle.
func _joint_angle_x(child: RigidBody3D, parent: RigidBody3D) -> float:
    var rel_basis = parent.global_transform.basis.inverse() * child.global_transform.basis
    return rel_basis.get_euler().x

func get_obs() -> Dictionary:
    var obs = []

    # Torso state — egocentric
    var fwd = -torso.global_transform.basis.z   # forward direction
    var vel = torso.linear_velocity

    obs.append(vel.dot(fwd)                         / MAX_SPEED)  # forward speed
    obs.append(vel.dot(Vector3.UP)                  / MAX_SPEED)  # vertical speed
    obs.append(vel.dot(fwd.cross(Vector3.UP))        / MAX_SPEED)  # lateral drift

    obs.append(torso.rotation.x / PI)                              # pitch
    obs.append(torso.rotation.z / PI)                              # roll
    obs.append(sin(torso.rotation.y))                              # yaw sin (avoids ±π wrap)
    obs.append(cos(torso.rotation.y))                              # yaw cos

    obs.append(torso.angular_velocity.x / MAX_ANG_VEL)
    obs.append(torso.angular_velocity.y / MAX_ANG_VEL)
    obs.append(torso.angular_velocity.z / MAX_ANG_VEL)

    obs.append(torso.global_position.y / 1.5)                     # height above ground

    # Per-joint: current angle (via relative body transform) + motor target velocity
    # child/parent pairs for each joint, in the same order as the action array
    var joint_pairs = [
        [upper_leg_l, torso],         # hip_l
        [lower_leg_l, upper_leg_l],   # knee_l
        [foot_l_body, lower_leg_l],   # ankle_l
        [upper_leg_r, torso],         # hip_r
        [lower_leg_r, upper_leg_r],   # knee_r
        [foot_r_body, lower_leg_r],   # ankle_r
    ]
    var joint_names = ["hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r"]
    for i in range(joint_names.size()):
        var angle = _joint_angle_x(joint_pairs[i][0], joint_pairs[i][1])
        var vel_cmd = joints[joint_names[i]].get_param_x(
            Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY)
        obs.append(angle    / PI)
        obs.append(vel_cmd  / MAX_JOINT_VEL)

    # Foot contact (binary)
    obs.append(1.0 if foot_l.foot_contact else 0.0)
    obs.append(1.0 if foot_r.foot_contact else 0.0)

    # Target speed command (enables a single policy trained at multiple speeds)
    obs.append(target_speed / MAX_SPEED)

    return {"obs": obs}
```

**Observation-Anzahl:** 11 (Torso) + 12 (6 Gelenke × 2) + 2 (Füße) + 1 (Zielgeschwindigkeit) = **26 Dimensionen.**

Ein Crawler (4 Beine, keine Y-Rotation des Torsos) hat 8 Gelenke mehr = ~42 Dim. Ein Worm mit 6 Segmenten nutzt ~30 Dim.

!!! warning "`PARAM_ANGULAR_LOWER_LIMIT` ist ein statisches Limit, nicht der aktuelle Winkel"
    Häufiger Fehler: `get_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_LOWER_LIMIT)` liefert immer das feste Gelenklimit, das du im Inspector gesetzt hast — es ändert sich nie während der Simulation. Verwende `_joint_angle_x()` oben, um den tatsächlichen aktuellen Winkel aus der relativen Body-Transform zu lesen.

---

## 4 · Action-Space

Zehn kontinuierliche Ausgaben — eine je geregelter DOF. Die Gelenkmotoren akzeptieren eine **Zielgeschwindigkeit**; das Motor-Force-Limit deckelt die angewandte Kraft.

```gdscript
func get_action_space() -> Dictionary:
    # 10 outputs: hip_l (x,y,z), knee_l (x), ankle_l (x),
    #             hip_r (x,y,z), knee_r (x), ankle_r (x)
    return {"joints": {"size": 10, "action_type": "continuous"}}

func set_action(action) -> void:
    var a = action["joints"]
    var MAX_VEL = 6.0   # rad/s — tune to your joint force limits

    joints["hip_l"].set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, a[0] * MAX_VEL)
    joints["hip_l"].set_param_y(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, a[1] * MAX_VEL)
    joints["hip_l"].set_param_z(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, a[2] * MAX_VEL)
    joints["knee_l"].set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY,  a[3] * MAX_VEL)
    joints["ankle_l"].set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, a[4] * MAX_VEL)
    joints["hip_r"].set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY,   a[5] * MAX_VEL)
    joints["hip_r"].set_param_y(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY,   a[6] * MAX_VEL)
    joints["hip_r"].set_param_z(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY,   a[7] * MAX_VEL)
    joints["knee_r"].set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY,  a[8] * MAX_VEL)
    joints["ankle_r"].set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, a[9] * MAX_VEL)
```

---

## 5 · Reward-Design

Die Lokomotion-Belohnung ist eine **gewichtete Summe von vier Komponenten**. Triffst du sie richtig, entsteht der Gang. Triffst du sie falsch, entdeckt der Agent kreative Wege zu schummeln.

```gdscript
var _alive := true

func _physics_process(_delta):
    if _ai.needs_reset:
        reset()
        return
    _compute_reward()
    _check_termination()

func _compute_reward():
    var fwd     = -torso.global_transform.basis.z
    var vel     = torso.linear_velocity
    var fwd_vel = vel.dot(fwd)           # positive = moving forward

    # 1. Forward velocity — the primary drive
    var r_vel = clampf(fwd_vel, -1.0, target_speed) / target_speed

    # 2. Energy penalty — discourage flailing
    var energy = 0.0
    for j in joints.values():
        var v = j.get_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY)
        energy += v * v
    var r_energy = -energy * 0.001

    # 3. Upright bonus — torso should stay level
    var up_dot = torso.global_transform.basis.y.dot(Vector3.UP)  # 1.0 = perfectly upright
    var r_upright = (up_dot - 0.5) * 0.1

    # 4. Alive bonus — prefer longer episodes over early termination
    var r_alive = 0.002

    _ai.reward += r_vel + r_energy + r_upright + r_alive

func _check_termination():
    if torso.global_position.y < 0.5:
        _alive = false
        _ai.reward -= 1.0
        _ai.done = true
```

### Warum jede Komponente existiert

| Komponente | Ohne | Mit |
|-----------|-----------|---------|
| Vorwärtsgeschwindigkeit | Agent bewegt sich nicht | Agent bewegt sich vorwärts |
| Energiestrafe | Agent vibriert Gelenke bei Maximaltempo (sieht robotisch aus, beschädigt Gelenke) | Glatter, effizienter Gang |
| Aufrecht-Bonus | Agent kriecht auf dem Gesicht oder hüpft seitwärts | Bleibt balanciert |
| Lebend-Bonus | Agent fällt sofort, um schlechte Episoden schneller zu beenden | Bevorzugt längere Episoden, um Belohnungen zu sammeln |

!!! warning "Koeffizientenordnung zählt"
    Halte `r_vel` im Bereich [0, 1]. Alle anderen Terme sollten ≤ 10 % des Vorwärtsgeschwindigkeitssignals sein. Dominiert die Energiestrafe, steht der Agent still. Dominiert „Aufrecht", balanciert der Agent, ohne zu gehen.

---

## 6 · Trainingskonfiguration

Lokomotion braucht mehr Timesteps als jeder andere Beispieltyp in diesem Kurs. Starte damit:

```bash
conda activate godot_env
tensorboard --logdir=logs &

gdrl --env_path=./Walker.x86_64 \
  --experiment_name=walker_ppo \
  --timesteps=10_000_000 \
  --n_parallel=16 \
  --speedup=20 \
  --n_steps=2048 \
  --batch_size=512 \
  --learning_rate=0.0003 \
  --ent_coef=0.01
```

`n_steps=2048` gibt PPO lange Rollouts — Lokomotionsepisoden dauern Hunderte von Steps und der Advantage-Schätzer braucht Zeit, um durch einen vollen Gangzyklus zurück zu propagieren.

### Was über das Training zu erwarten ist

| Timesteps | Typisches Verhalten |
|-----------|------------------|
| 0–500k | Fällt sofort. `ep_len_mean` = 20–50 Steps |
| 500k–2M | Entdeckt, wie er aufrecht bleibt. Noch keine Vorwärtsbewegung |
| 2M–5M | Beginnt vorwärts zu schlurfen. Erkennbarer Protogang |
| 5M–10M | Gang stabilisiert sich. Geschwindigkeit nähert sich Ziel |
| 10M+ | Feinschliff des Gangs, Energieeffizienz verbessert sich |

Lokomotion ist die am langsamsten konvergierende Aufgabe in diesem Kurs. **Beurteile sie nicht bei 1M Steps.**

---

## 7 · Häufige Fehlermodi

### Die Statue
Agent steht perfekt still. `ep_rew_mean` > 0, aber `ep_len_mean` läuft auf das Episoden-Timeout aus.

**Ursache:** Lebend-Bonus + Aufrecht-Bonus > Vorwärtsgeschwindigkeit-Reward. Der Agent verdient mehr durch Stehen als durch das Risiko zu fallen.

**Fix:** `target_speed` in der Geschwindigkeitsbelohnung erhöhen oder `r_alive` halbieren.

### Der Spinner
Agent lernt, auf der Stelle zu rotieren. Vorwärtsgeschwindigkeit ≈ 0, aber Winkelgeschwindigkeit ≈ max.

**Ursache:** Keine Seitwärtsdrift-Strafe. Ein rotierender Körper hat netto null Vorwärtsgeschwindigkeit, aber die Belohnungsfunktion bestraft Drehung nie.

**Fix:** `r_lat = -abs(vel.dot(lateral)) * 0.1` hinzufügen, um Seitwärtsbewegung zu bestrafen. Auch `r_yaw = -abs(torso.angular_velocity.y) * 0.05` hinzufügen.

### Der Hopper
Agent lernt einen Einbeinhüpfer — bewegt sich technisch vorwärts, sieht aber nicht aus wie die ML-Agents-Demos.

**Ursache:** Einbeinhüpfen ist ein gültiges lokales Optimum. Es erfüllt die Vorwärtsgeschwindigkeit-Belohnung mit weniger Gelenkkoordination als ein vollständiger Gang.

**Fix:** Einen **Fußwechsel-Bonus** hinzufügen: Belohne, wenn linke und rechte Kontaktsignale alternieren (nicht beide an, nicht beide aus). `r_contact = abs(float(foot_l.foot_contact) - float(foot_r.foot_contact)) * 0.05`.

### Sofortiger Zusammenbruch
Agent fällt jede Episode, Belohnung steigt nie über den Lebend-Bonus.

**Ursache:** Ausgangspose ist instabil — der Körper spawnt mit genug Drehmoment oder Höhe, dass die Schwerkraft gewinnt, bevor die Policy handelt.

**Fix:** Torso tiefer spawnen (0,8 m statt 1,2 m) oder am Episodenstart einen 0,5-sekündigen Physik-Settled-Freeze einfügen, bevor die Policy Aktionen ausgibt.

---

## 8 · Anpassung an andere Körpertypen

Sobald der Walker funktioniert, gilt dieselbe Struktur für andere ML-Agents-artige Körper:

### Crawler (Vierbeiner, keine Rumpfrotation)

- Rumpfrotationsstrafe aus der Belohnung entfernen (der Körper ist nahe am Boden — Rollen weniger katastrophal)
- 4 Beine × 2 Gelenke = 8-DOF-Action-Space (8 Ausgaben)
- Alle 4 Fußkontakt-Signale in die Observation aufnehmen
- `n_parallel` auf 32+ erhöhen — Vierbeinertraining profitiert mehr von Datenvolumen

### Worm (Segmentkette)

- 6 `RigidBody3D`-Kapseln verbunden durch `HingeJoint3D`, einachsige Rotation pro Gelenk
- Keine Fußkontakte — ersetzen durch „Höhe des Kopfsegments über dem Boden"
- Vorwärtsgeschwindigkeit am Kopfsegment messen
- Worm entdeckt ein sinusförmiges Wellenmuster bei 3–5M Steps — klar im Viz-Checkpoint sichtbar

### Variables Geschwindigkeitskommando

Um eine **einzige Policy** zu trainieren, die mit mehreren Geschwindigkeiten geht (wie die ML-Agents-Walker-Demo), randomisiere `target_speed` pro Episode:

```gdscript
func reset() -> void:
    target_speed = randf_range(0.5, 3.0)   # m/s — wide range forces the policy to condition on it
    _alive = true
    _ai.reset()
```

Nimm `target_speed / MAX_SPEED` in `get_obs()` auf, damit die Policy das aktuelle Kommando lesen kann. Der Agent lernt, zwischen Geschwindigkeiten zu interpolieren, ohne separate Policies.

---

## 9 · Viz-Checkpoint — wie ein trainierter Walker aussehen sollte

Mit `--viz` bei 3M, 6M und 10M Steps laufen lassen. Achte auf:

**3M:** Aufrecht, aber unbeholfen. Schlurft vorwärts. Fällt gelegentlich. Das ist die schwerste Phase zum Zuschauen — sieht aus wie ein Mensch, der gehen lernt. Es funktioniert.

**6M:** Erkennbarer alternierender Gang. Fällt selten auf ebenem Boden. Beinschwung ist sichtbar.

**10M:** Glatter Gang. Bleibt aufrecht. Nähert sich `target_speed`. Energieverbrauch ist gesunken (beobachte `train/std` der Action-Verteilung — sie wird enger, je entschiedener die Policy wird).

**Terrain-Test:** Eine leichte Steigung oder ein niedriges Hindernis hinzufügen. Eine 10M-Policy passt sich ohne Nachtraining an. Fällt sie sofort, ist die Domäne zu weit außerhalb des Trainierten — füge Terrainvariation zur Trainingsumgebung hinzu.

---

## 10 · Stretch Goals

- **Crawl → Walk Curriculum:** Trainiere mit Torso bei 0,3 m Höhe (zwingt zum Kriechen). Nach 2M Steps die Spawn-Höhe auf 1,2 m heben. Überträgt sich die Krieche-Policy oder muss sie umlernen?
- **Terrainvariation:** Zufällige Höhenvariation am Boden via `HeightMapShape3D` hinzufügen. Wie viele zusätzliche Timesteps braucht die Policy, um auf unebenem Grund aufrecht zu bleiben?
- **Zwei-Agenten-Rennen:** Nutze das Multi-Agent-Setup aus [Unit 7](unit-07.md). Zwei Walker konkurrieren um Vorwärtsposition. Beschleunigt oder verlangsamt der Wettkampf die Gangqualität?
- **Export nach ONNX:** Folge [Unit 10](unit-10.md), um die Walker-Policy zu exportieren. Bette sie als reine Inferenz-Demo in die Godot-Szene ein — kein Python zur Laufzeit. Teile den HTML5-Build.

---

## Was kommt als Nächstes

Dein Walker/Crawler nutzt eine feste, handgemachte Belohnung. **Hindsight Experience Replay (HER)** ist die Technik, die spärliche, ziel-bedingte Aufgaben — „erreiche diese Zielposition" — ohne dichtes Shaping lernbar macht. Derselbe artikulierte Körper wird zu einem Greifarm.

[→ Goal-Conditioned RL & HER](unit-her.md)
