# Roboter-Observations, Aktionen & Sensoren

Du kannst einen Agenten trainieren, der Spiele gewinnt. Aber ein **Roboter** ist kein Spiel — er hat Gelenke, die verschleißen, Sensoren, die driften, Motoren, die überhitzen, und ein Stromkabel, das an ihm zieht. Bevor du die Sim-to-Real-Lücke überquerst, musst du deine Observation- und Action-Spaces **so entwerfen, als existiere der Roboter bereits in der echten Welt** — auch wenn du noch in Godot arbeitest.

Diese Unit lehrt die robotikspezifischen Muster, die du in jeder Manipulations-, Lokomotion- und Sim-to-Real-Unit von hier an verwendest.

[← Ship Your Brain](unit-10.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~30 min

---

!!! info "Drei Wege, deine KI zu beobachten"
    Godot (beobachte den Arm, der nach einem Ziel greift — spärliche Belohnung = anfangs spärliche Bewegung) · TensorBoard (Gelenklimit-Verletzungen pro Episode verfolgen — sollten gegen null tendieren) · Gelenkzustände im Human-Mode drucken (prüfen, dass dein Action-Mapping das richtige Gelenk in die richtige Richtung bewegt)

---

## 1 · Was Robotik-RL anders macht

Bisher in diesem Kurs war jede Umgebung ein **Spiel**. Der Agent lebt im Simulator, sammelt Punkte, stirbt, respawnt. Funktioniert die Policy in Godot, lieferst du sie aus.

Robotik-RL ist nicht so.

| Belang | Game AI | Robotik-RL |
|---------|---------|-------------|
| Agent-Körper | Virtuell, unzerstörbar | Physisch, zerbrechlich, teuer |
| Sensoren | Perfekt, instantan | Verrauscht, verzögert, manchmal fehlend |
| Aktionen | Kraft direkt anwenden | Kommandos an Motoren mit Latenz senden |
| Kosten eines Fehlschlags | Episode neu starten | Getriebe austauschen |
| „Done"-Kriterium | Policy schlägt Benchmark | Policy übersteht 8 Stunden Dauerbetrieb |
| Reproduzierbarkeit | `seed=42` | Der Motor verschleißt jeden Tag anders |

**Die Kernherausforderung ist die Sim-to-Real-Lücke** — der Unterschied zwischen deiner Godot-Welt und dem echten Roboter. Wir widmen einer ganzen Unit dem Schließen dieser Lücke. Aber die Lücke **beginnt hier**, in dieser Unit, damit wie du deine Observation- und Action-Spaces entwirfst.

> **Schlüsselprinzip.** Entwirf deine Obs- und Action-Spaces, *als existiere der Roboter bereits* — auch wenn du noch in reiner Simulation bist. Jede Abkürzung in der Sim („einfach den perfekten Gelenkwinkel via `Skeleton3D.get_bone_pose()` lesen") ist eine Schuld, die du später zurückzahlst, wenn der Encoder dir einen verrauschten Wert 8 ms zu spät liefert.

---

## 2 · Propriozeption vs. Exterozeption

Robotiker teilen Sensoren in zwei Kategorien ein. Du musst beide Begriffe kennen, weil jedes Paper sie benutzt.

### Propriozeption (proprioception) — den eigenen Körper spüren

Das „Selbstgefühl" des Roboters: wo seine Gelenke sind, wie schnell sie sich bewegen, wie viel Drehmoment jeder Motor erzeugt.

- **Gelenkwinkel** (von Encodern an jedem Motor)
- **Gelenkgeschwindigkeiten** (Encoder-Deltas)
- **Gelenkmomente / -ströme** (Motorstrom → Drehmoment)
- **IMU**: Orientierung, Winkelgeschwindigkeit, Linearbeschleunigung der Basis / des Torsos
- **Motortemperaturen** (ja, echte Policies brauchen die manchmal)

### Exterozeption (exteroception) — die Welt außen spüren

Alles, was davon abhängt, was *um* den Roboter herum ist:

- **Kameras** (RGB, Tiefe, Stereo)
- **LiDAR / 3D-Tiefenscanner**
- **Distanzsensoren** (Ultraschall, ToF)
- **Kraft-/Drehmomentsensor an Handgelenk oder Füßen**
- **Mikrofone, Kontaktschalter, taktile Haut**

### Warum diese Unterscheidung für RL zählt

Propriozeptive Obs sind auf echter Hardware **immer verfügbar, schnell und genau**. Jeder Servo wird mit einem Encoder ausgeliefert. Die Werte sind typisch innerhalb weniger Hundertstel Grad sauber.

Exterozeptive Obs sind **teuer, verrauscht und unzuverlässig**. Kameras müssen kalibriert werden. LiDAR hat Verdeckungen. Kraftsensoren driften mit der Temperatur. Manche Sensoren liefern mit 30 Hz, während deine Regelschleife mit 200 Hz läuft.

> **Faustregel.** Bevorzuge Propriozeption für den **Actor**. Nutze Exterozeption für den **Critic** (asymmetrischer Actor-Critic — siehe Sim-to-Real-Unit). So hängt der Actor nur von Signalen ab, denen du beim Deployment vertraust, während der Critic die privilegierte Sicht bekommt, die er zum Lernen guter Value-Schätzungen braucht.

---

## 3 · Aufbau eines Roboter-Observation-Space in Godot

Hier das kanonische Muster für einen 6-DOF-Roboterarm auf Basis von Godots `Skeleton3D` plus einer Kette von `Generic6DOFJoint3D`-Nodes:

```gdscript
# ai_controller.gd for a robot arm
extends AIController3D

@onready var skeleton     = $RobotArm/Skeleton3D
@onready var joints       = $RobotArm.get_children().filter(
    func(n): return n is Generic6DOFJoint3D)
@onready var end_effector = $RobotArm/EndEffector

const MAX_JOINT_VEL = 3.14   # rad/s — typical servo limit
const MAX_FORCE     = 50.0   # N — load cell range
const ARM_REACH     = 0.85   # m — fully extended

var _prev_angles : Array  = []

func get_obs() -> Dictionary:
    var obs = []

    # --- Proprioceptive: joint state (angle + velocity per joint) ---
    for bone_idx in range(skeleton.get_bone_count()):
        var pose  = skeleton.get_bone_pose(bone_idx)
        var angle = pose.basis.get_euler()          # rotation as Euler angles
        obs.append(angle.x / PI)                    # normalize to [-1, 1]
        obs.append(angle.y / PI)
        obs.append(angle.z / PI)
        # Note: angular velocity requires tracking previous pose
        # (computed in _physics_process and cached)

    # --- End-effector position relative to base ---
    var ee_local = to_local(end_effector.global_position)
    obs.append(ee_local.x / ARM_REACH)
    obs.append(ee_local.y / ARM_REACH)
    obs.append(ee_local.z / ARM_REACH)

    # --- Exteroceptive: target position (goal-conditioned — see HER unit) ---
    var goal_local = to_local(goal.global_position)
    obs.append(goal_local.x / ARM_REACH)
    obs.append(goal_local.y / ARM_REACH)
    obs.append(goal_local.z / ARM_REACH)

    return {"obs": obs}
```

### Größe des Observation-Vektors

Für einen 6-DOF-Arm mit dem obigen Template:

| Block | Komponenten | Anzahl |
|-------|-----------|-------|
| Gelenkwinkel (6 Gelenke × 3 Euler) | x, y, z pro Gelenk | 18 |
| Endeffektor-Position (lokal) | x, y, z | 3 |
| Zielposition (lokal) | x, y, z | 3 |
| **Gesamt** | | **24** |

Mit Gelenkgeschwindigkeiten verdoppelst du den Gelenkblock — typische 6-DOF-Arme landen bei 36–48 Dimensionen.

### Die Normalisierungsregel

> **Normiere alles auf [-1, 1] oder [0, 1].** Neuronale-Netz-Optimierer sind auf Eingaben mit etwa Einheitsvarianz ausgelegt. Ein roher Winkel von `2,93 rad` und eine rohe Position von `0,04 m` werden im Netz kombiniert — ist eine zwei Größenordnungen größer, verschwinden Gradienten auf der kleinen.

Teile durch das **physikalische Maximum** (Gelenklimit, Armreichweite, Max-Servo-Geschwindigkeit), nicht durch den größten beobachteten Wert.

---

## 4 · Gelenk-Regelungsmodi

Das ist die wichtigste Einzelentscheidung in der Robotik-RL und wird in Game-RL-Kursen fast nie diskutiert. **Dein Action-Space definiert, was deine Policy ausdrücken kann.**

| Modus | Was er regelt | Godot-Umsetzung | Echte Hardware |
|------|-----------------|---------------------|---------------|
| **Positionsregelung** | Ziel-Gelenkwinkel | `rotation` setzen oder PID auf Ziel | Servomotoren (die meisten Hobby-Roboter) |
| **Geschwindigkeitsregelung** | Ziel-Gelenkgeschwindigkeit | `PARAM_ANGULAR_MOTOR_TARGET_VELOCITY` | DC-Motoren mit Encodern |
| **Drehmomentregelung** | Rohe Kraft / Drehmoment | `apply_torque_impulse` am Bone-Body | High-End-Aktuatoren (ANYdrive, Dynamixel Pro) |

### Positionsregelung

Der Agent gibt einen **Zielwinkel** aus. Ein innerer Regler (PID auf echter Hardware, ein steifer Motor in Godot) treibt das Gelenk auf den Zielwert.

```gdscript
# Position control — agent outputs target angles
func set_action(action) -> void:
    for i in range(joints.size()):
        var target_angle = action["joints"][i] * PI            # action ∈ [-1, 1] → [-π, π]
        var error        = target_angle - current_angles[i]
        joints[i].set_param_x(
            Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY,
            error * position_gain)
```

### Geschwindigkeitsregelung

Der Agent gibt eine **Ziel-Winkelgeschwindigkeit** aus. Der Motor versucht, diese Drehzahl zu halten.

```gdscript
# Velocity control — agent outputs angular velocities
func set_action(action) -> void:
    for i in range(joints.size()):
        var target_vel = action["joints"][i] * MAX_JOINT_VEL   # action ∈ [-1, 1] → [-vmax, vmax]
        joints[i].set_param_x(
            Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY,
            target_vel)
```

### Drehmomentregelung

Der Agent gibt **rohes Drehmoment** für das Glied aus.

```gdscript
# Torque control — agent outputs raw forces
func set_action(action) -> void:
    for i in range(joints.size()):
        var torque = action["joints"][i] * MAX_TORQUE          # action ∈ [-1, 1] → [-τmax, τmax]
        bones[i].apply_torque(Vector3(torque, 0, 0))
```

### Welchen Modus wählen?

!!! tip "Wahl des Regelungsmodus"
    - **Positionsregelung** → einfachster, sicherster Weg, gut für langsame, präzise Aufgaben. Nimm das zuerst für Manipulation, Pick-and-Place, Reaching.
    - **Geschwindigkeitsregelung** → natürlich für Lokomotion, differentialgelenkte Mobilroboter und jede Aufgabe, bei der „wie schnell" mehr zählt als „wo genau".
    - **Drehmomentregelung** → maximale Ausdruckskraft, am schwersten zu trainieren, nötig für wirklich dynamische Aufgaben wie Laufen, Springen, Werfen, Fangen.

Nützliche Heuristik: **je niedriger der Regelungsmodus, desto schwerer das Lernproblem und desto dynamischer das erreichbare Verhalten.** Starte bei Positionsregelung und steige nur eine Stufe tiefer, wenn die Aufgabe es fundamental erfordert.

!!! warning "Positionsregelung kann Bugs verstecken"
    Ein steifer Positionsregler *zwingt* das Gelenk auf den kommandierten Winkel, sogar durch Kollisionen. Deine Policy kann lernen zu „teleportieren", indem sie wilde Winkel kommandiert, weil der Motor mit unendlicher Kraft im Simulator gehorcht. Auf echter Hardware löst das den Überstromschutz aus und der Arm schaltet ab. Begrenze die Änderung des kommandierten Winkels pro Schritt (Action Smoothing), damit das Verhalten physikalisch erreichbar bleibt.

---

## 5 · Injektion von Sensorrauschen

Jeder echte Sensor lügt ein wenig. Encoder quantisieren. IMUs driften. Kraftaufnehmer reagieren auf Temperatur.

Trainierst du gegen perfekte Sim-Werte, wird die Policy süchtig nach Präzision, die sie auf dem echten Roboter nicht hat. Der Fix: **Rauschen beim Training injizieren**, damit die Policy lernt, robust unter Unsicherheit zu handeln:

```gdscript
# In get_obs() — add realistic noise to all readings
func _add_noise(value: float, std: float) -> float:
    return value + randfn(0.0, std)

# Proprioceptive noise (small — encoders are accurate)
obs.append(_add_noise(angle.x / PI, 0.005))

# IMU noise (moderate — gyros drift, accelerometers vibrate)
obs.append(_add_noise(angular_velocity.x / MAX_VEL, 0.02))

# Force / torque noise (larger — load cells are noisy and temperature-sensitive)
obs.append(_add_noise(contact_force / MAX_FORCE, 0.05))
```

### Typische Rauschstärken (hier starten, aus Datenblättern tunen)

| Sensor | Vernünftiges σ (normiert) |
|--------|--------------------------|
| Encoder-Winkel | 0,002 – 0,01 |
| Gelenkgeschwindigkeit (Encoder-Diff) | 0,02 – 0,05 |
| IMU-Winkelgeschwindigkeit | 0,01 – 0,03 |
| IMU-Linearbeschleunigung | 0,03 – 0,08 |
| Kraft-/Drehmomentsensor | 0,05 – 0,10 |
| Tiefenkamera-Distanz | 0,02 – 0,05 |

### Rauschen als Einstieg in Domain Randomization

Das ist dein erster Vorgeschmack auf **Domain Randomization** — die Technik hinter fast jedem erfolgreichen Sim-to-Real-Transfer. Wir gehen in der Sim-to-Real-Unit viel tiefer, aber das Muster ist dasselbe: trainiere statt in einer perfekten Welt in einer *Verteilung* leicht kaputter Welten. Variiere `std` pro Episode und die Policy lernt, robust über die ganze Familie zu sein.

```gdscript
func reset() -> void:
    # Vary sensor noise scale each episode
    encoder_std = randf_range(0.002, 0.010)
    imu_std     = randf_range(0.010, 0.040)
    # ...
```

---

## 6 · Sicherheits- und Constraint-Belohnungen

Ein Game-Agent, der über die Klippe fährt, respawnt. Ein echter Roboter, der über die Klippe fährt, ist weg.

Robotik-Belohnungsfunktionen kombinieren fast immer eine **Aufgabenbelohnung** mit einem Stapel **Sicherheits- und Effizienz-Strafen**, die Verhalten zu etwas formen, das ein echter Motor durchhält:

```gdscript
func _physics_process(delta):
    if _ai.needs_reset:
        reset()
        return

    # --- Task reward (sparse — only at success) ---
    if end_effector_near_goal():
        _ai.reward += 1.0
        _ai.done = true

    # --- Joint limit penalty ---
    for i in range(joints.size()):
        var angle = get_joint_angle(i)
        var limit = joint_limits[i]
        if abs(angle) > limit * 0.9:           # warn at 90% of limit
            _ai.reward -= 0.1
        if abs(angle) > limit:                  # hard stop — would damage real hardware
            _ai.reward -= 1.0
            _ai.done = true                     # terminate the episode

    # --- Energy efficiency (minimize power = torque × velocity) ---
    var power = 0.0
    for i in range(joints.size()):
        power += abs(get_joint_torque(i) * get_joint_velocity(i))
    _ai.reward -= power * 0.0001                # tiny coefficient — don't dominate task reward

    # --- Smoothness / jerk penalty ---
    var jerk = (linear_velocity - _prev_velocity).length() / delta
    _ai.reward -= jerk * 0.00001
    _prev_velocity = linear_velocity

    # --- Survival bonus (discourage immediate failure) ---
    _ai.reward += 0.001
```

!!! warning "Gelenklimit-Verletzungen zerstören echte Hardware"
    Ein Positionsbefehl jenseits des mechanischen Anschlags treibt den Motor mit vollem Drehmoment gegen den Endanschlag. Auf einem echten Arm strippt das die Zahnräder in Sekunden. **Immer eine harte Gelenklimit-Strafe einbauen *und* die Episode beenden** — die Policy soll Limitüberschreitungen als katastrophal behandeln, nicht als „leicht negativ". Bestrafst du nur weich, entdeckt die Policy vielleicht, dass es die Kosten wert ist.

### Koeffizienten tunen

Die Zahlen oben (`-0,1`, `-1,0`, `0,0001`, `0,00001`, `0,001`) sind nicht magisch. Sie spiegeln eine relative Ordnung:

1. **Aufgabenbelohnung** ist das größte Signal (`+1,0` bei Erfolg).
2. **Harte Sicherheitsverletzungen** sind vergleichbar groß (`-1,0`) — sie sollen wirklich wehtun.
3. **Sanfte Warnungen** sind 10× kleiner (`-0,1`) — sie formen Verhalten, ohne zu dominieren.
4. **Effizienz und Glätte** sind 1000–10000× kleiner (`1e-4`, `1e-5`) — sie polieren eine funktionierende Policy, treiben aber kein Lernen.
5. **Überlebensbonus** ist klein und positiv — ermutigt, lang genug zu leben, um Belohnung zu finden.

Ist dein Roboter zappelig, erhöhe den Glätte-Koeffizienten. Ist er langsam und übervorsichtig, senke die Energiestrafe. **Tune einen nach dem anderen.**

---

## 7 · Der Standard-Robotik-Simulationsstack

Du wirst Robotik-RL-Paper lesen. Sie erwähnen Simulatoren, die nicht Godot heißen. Hier, was jedes Werkzeug ist und wo Godot hineinpasst:

| Werkzeug | Hauptnutzung | Physik | GPU-parallel | Godot-Äquivalent |
|------|-------------|---------|--------------|------------------|
| **MuJoCo** | Lokomotion, Manipulationsforschung | Exzellenter Kontakt | Nein (CPU) | Nah — anderer Solver |
| **Isaac Gym / Isaac Lab** | Massive parallele Trainings, Sim-to-Real | Gut | **Ja (tausende Envs auf einer GPU)** | `n_parallel`, aber CPU-gebunden |
| **PyBullet** | Freie MuJoCo-Alternative, klassische Baselines | Gut | Nein | Ähnlich |
| **gymnasium-robotics** | HER-Benchmark-Envs (Fetch, Hand) | Via PyBullet / MuJoCo | Nein | HER-Unit portiert diese |
| **Webots** | Bildung, ROS-Integration | Gut | Nein | Ähnlicher Open-Source-Geist |
| **Gazebo** | ROS-natives Robotik | OK (älter) | Nein | Schwerer, ROS-gekoppelt |

### Warum Godot trotzdem?

- **Visuelle Qualität.** Hübsch out of the box — nützlich, wenn deine reale Aufgabe Kameras nutzt.
- **Game-ready.** Menschen, Türen, interaktive Props, volle UI hinzufügen, ohne die Engine zu verlassen.
- **Lizenz.** Frei, Open Source, MIT-artig. Kein Login-Wall, keine Enterprise-Stufe.
- **Cross-Platform-Export.** Dein trainierter Agent kann als HTML5-Demo oder Desktop-Spiel ausgeliefert werden.

### Wo Godot schwächer ist

- **Durchsatz.** Isaac Gym läuft 4 000+ Umgebungen auf einer einzigen GPU. Godots `n_parallel` ist CPU-gebunden und endet bei deiner Core-Zahl.
- **Kontaktphysik.** MuJoCos Soft-Contact-Solver ist Goldstandard für Manipulation. Godot-Physik ist für die meisten Aufgaben fein, aber bei feinem Kontakt merkt man den Unterschied.
- **Robotik-Ökosystem.** Kein URDF-Importer, keine ROS-Bridge, weniger fertige Robotermodelle. Den Arm baust du selbst.

**Wähle Godot, wenn** die Aufgabe vor allem geometrisch / visuell ist und du eine polierte Sim willst, die du Nicht-Forschern in die Hand drücken kannst. **Wechsle zu MuJoCo / Isaac, wenn** Kontaktpräzision oder massive Parallelität zum Engpass werden.

---

## 8 · Einen minimalen Roboterarm in Godot bauen

Bauen wir einen 3-DOF-planaren Arm — den einfachsten nicht-trivialen Manipulator. Das wird deine Umgebung für die HER (Hindsight Experience Replay)-Unit.

### Schritte

1. **Szene erstellen.** Neuer `Node3D` namens `RobotArm`.
2. **Drei Segmente hinzufügen.** Drei `RigidBody3D`-Kinder, jedes mit `BoxShape3D`-Collider (z. B. 0,3 × 0,05 × 0,05 m). Stapele sie entlang der X-Achse.
3. **Hinge-Joints ergänzen.** Zwischen je zwei Segmenten ein `HingeJoint3D`. Verbinde `node_a` und `node_b` mit den zwei Segmenten.
4. **Gelenklimits setzen.** An jedem `HingeJoint3D` das Limit aktivieren und auf ±90° (±1,57 rad) setzen. Markiere das Basis-Segment-`freeze` als true, damit der Arm eine feste Wurzel hat.
5. **Controller anhängen.** Einen `AIController3D`-Node an die Wurzel. Setze `action_space = {"joints": {"size": 3, "action_type": "continuous"}}`.
6. **Endeffektor hinzufügen.** Kleine `Sphere3D` (Radius 0,02 m) als Kind des dritten Segments, positioniert an der Spitze.
7. **Ziel hinzufügen.** Kugel (Radius 0,03 m, ohne Kollision), die bei `reset()` an einer zufälligen, erreichbaren Position respawnt.
8. **`get_obs()` implementieren.** Gelenkwinkel (3) + Endeffektor-Position (3) + Zielposition (3) = **9-dim** Obs.
9. **`set_action()` implementieren.** Mappe `action["joints"][i] ∈ [-1, 1]` auf die Zielgeschwindigkeit für Hinge `i` via `set_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY, ...)`.

### Reset-Logik

```gdscript
func reset() -> void:
    # Zero all joint angles
    for seg in segments:
        seg.angular_velocity = Vector3.ZERO
        seg.linear_velocity  = Vector3.ZERO
    # Spawn goal somewhere reachable
    var theta = randf_range(-PI/2, PI/2)
    var r     = randf_range(0.3, 0.85)
    goal.position = Vector3(cos(theta) * r, sin(theta) * r, 0.0)
    _ai.reset()
```

Das ist alles. Die HER-Unit nimmt diese Szene auf und ergänzt Goal-Relabeling.

---

## 9 · Observation-Muster für Lokomotion

Lokomotionsroboter (Beinroboter, Räderroboter, Humanoide) haben ein anderes kanonisches Observation-Layout. Es ist aus der MuJoCo-Ant- / HalfCheetah-Konvention adaptiert, die du in jedem Lokomotion-Paper sehen wirst:

```gdscript
# Standard locomotion obs (MuJoCo Ant-style)
func get_obs() -> Dictionary:
    var obs = []

    # --- Torso state ---
    obs.append(linear_velocity.x  / max_speed)    # forward velocity (reward signal)
    obs.append(linear_velocity.y  / max_speed)    # vertical
    obs.append(linear_velocity.z  / max_speed)    # lateral
    obs.append(rotation.x         / PI)           # pitch
    obs.append(rotation.z         / PI)           # roll
    obs.append(angular_velocity.x / max_ang_vel)  # pitch rate
    obs.append(angular_velocity.z / max_ang_vel)  # roll rate

    # --- Per-leg state (× n_legs) ---
    for leg in legs:
        obs.append(leg.hip_angle    / hip_limit)
        obs.append(leg.knee_angle   / knee_limit)
        obs.append(leg.hip_velocity / max_joint_vel)
        obs.append(leg.knee_velocity / max_joint_vel)
        obs.append(1.0 if leg.foot_in_contact else 0.0)

    # --- Height above ground (useful for fall detection) ---
    obs.append(global_position.y / max_height)

    return {"obs": obs}
```

### Was bewusst *nicht* drin steht

- **Absolute X-/Z-Weltposition** — die Policy soll über den ganzen Boden generalisieren, nicht „Geh zur Koordinate (3,7, 0, -2,1)" auswendig lernen. Ersetze durch ein gewünschtes Geschwindigkeitskommando.
- **Absoluter Yaw** — gleicher Grund. Ersetze durch Zielheading relativ zum aktuellen Heading.
- **Die tatsächliche Zielposition im Weltframe** — übergib eine *Richtung* und *Distanz*, lokal.

Das ist die **egozentrische** Sicht: die Policy sieht die Welt durch den eigenen Frame des Roboters, nie durch ein globales Koordinatensystem. Egozentrische Observations sind der größte einzelne Gewinn für Generalisierung in der Lokomotion.

### Yaw als sin/cos

Für jede Winkel-Observation, die wickelt (Yaw, Hüftwinkel an einem unbegrenzten Gelenk), übergib sie als **zwei** Komponenten — `sin(θ)` und `cos(θ)` — statt der rohen Radiant. Das entfernt die Unstetigkeit bei ±π, die das Netz verwirrt.

```gdscript
obs.append(sin(yaw))
obs.append(cos(yaw))
```

---

## 10 · Viz-Checkpoint

Trainiere den 3-DOF-Arm aus § 8 mit PPO für 100 Episoden (`--n_parallel 4 --speedup 8` reicht) und schaue mit `--viz` zu. Frag dich:

- **Erreicht der Arm das Ziel?** Bei spärlicher Belohnung sehen die ersten 50 Episoden zufällig aus. Um Episode 70–100 solltest du intentionale Bewegung sehen. Falls nicht — Reward Shaping oder Normalisierung ist falsch, debugge zuerst die Obs-Werte.
- **Glättet die Energiestrafe wirklich die Bewegung?** Mache zwei Trainings — eines mit, eines ohne. Vergleiche visuell. Der bestrafte Lauf soll spürbar weniger zappelig wirken.
- **Liegen Gelenkgeschwindigkeiten innerhalb Hardware-Limits?** Füge einen Debug-Print in `_physics_process` ein:

  ```gdscript
  if Engine.get_physics_frames() % 30 == 0:
      print("joint vels: ", joints.map(func(j): return get_joint_velocity(j)))
  ```

  Überschreitet ein Wert regelmäßig `MAX_JOINT_VEL`, verlangt die Policy auf echter Hardware unmögliche Bewegung. Action-Skalierung enger ziehen oder Geschwindigkeitsstrafe hinzufügen.

- **TensorBoard-Check.** Plotte `episode_length` und jeden eigenen `joint_limit_violations`-Zähler. Die Verletzungszahl sollte fallen — bleibt sie flach, ist deine harte Strafe nicht stark genug.

!!! check "Fertig, wenn"
    Für den 3-DOF-Arm gibt es keinen veröffentlichten Benchmark, also miss den Erfolg an den eigenen Signalen der Unit: Am Ende des 100-Episoden-Laufs oben zeigt der Arm klar intentionale Bewegung Richtung Ziel statt zufälligen Ruderns (erwarte das etwa ab Episode 70–100), dein Debug-Print zeigt Gelenkgeschwindigkeiten innerhalb von `MAX_JOINT_VEL`, und der `joint_limit_violations`-Zähler in TensorBoard fällt. Sieht die Bewegung am Ende des Laufs immer noch zufällig aus, häng nicht einfach Episoden an — arbeite die Checkliste oben noch einmal durch: zuerst die Obs-Werte prüfen, dann die Action-Skalierung, dann die Stärke der harten Gelenklimit-Strafe aus Abschnitt 6.

---

## 11 · Stretch Goals

- **Ein 4. DOF — eine Handgelenkrotation hinzufügen.** Neu trainieren. Wie ändert sich die Sample-Effizienz? (Spoiler: jeder zusätzliche DOF verdoppelt die Trainingszeit grob. Fluch der Dimensionalität.)
- **Geschwindigkeitsregelung gegen Positionsregelung tauschen.** Welcher Modus erreicht das Ziel in weniger Umgebungsschritten? Welcher produziert glattere Bewegung?
- **Simulierte Kraftmessdose.** Füge eine Kraftmessung am Endeffektor hinzu, wenn er die Zielkugel berührt. Nimm sie in die Obs auf. Lernt die Policy, Kontaktfeedback zum „Fühlen" des Ziels zu nutzen, oder ignoriert sie den zusätzlichen Kanal?
- **Vorschau asymmetrischer Actor-Critic.** Gib dem Critic die Weltposition des Ziels direkt, aber zwinge den Actor, nur seinen eigenen Gelenkzustand plus eine verrauschte Distanzschätzung zu sehen. (Volle Behandlung in der Sim-to-Real-Unit.)
- **Den Arm randomisieren.** Pro Episode Glied-Längen ±10 %, Massen ±20 %, Gelenkreibung ±50 % variieren. Übersteht die trainierte Policy das? Das ist dein erstes Domain-Randomization-Experiment.

---

## Was kommt als Nächstes

Du kannst jetzt:

- Propriozeptive von exterozeptiven Sensoren unterscheiden und je Rolle die richtigen wählen
- Einen Roboter-Observation-Space bauen, der dich beim Portieren auf echte Hardware nicht anlügt
- Einen Regelungsmodus (Position / Geschwindigkeit / Drehmoment) passend zur Dynamik der Aufgabe wählen
- Rauschen injizieren, um die Policy gegen die Realität der Sensoren zu härten
- Sicherheits-, Effizienz- und Glätte-Belohnungen stapeln, ohne das Aufgabensignal zu ertränken
- Robotik-Paper lesen und wissen, ob deren MuJoCo- / Isaac-Setup auf das passt, was du in Godot hast

Die nächste Unit zeigt, wie du Walker-, Crawler- und Worm-Agenten in Godot von Grund auf baust — die Lokomotion-Demos, die du aus AI Warehouse kennst, neu mit der Belohnungsstruktur gebaut, die du jetzt kennst.

[→ Lokomotion-Agenten](unit-locomotion.md)
