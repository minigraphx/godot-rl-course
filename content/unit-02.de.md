# Unit 2 — LunarLander nach Godot RL Agents portieren

Zwei Erfolge in einer Einheit: **Phase A** — Ausführen und Anpassen von **SimpleReachGoal** (Raycasts, diskrete Aktionen (discrete actions)). **Phase B** — Lunar Lander von Grund auf neu bauen, mit denselben `AIController`-Mustern. Training mit Stable-Baselines3 über **godot-rl**.

[← Belohnungsdesign](unit-reward-engineering.md) · [Kursübersicht](index.md)

!!! note "Voraussetzungen"
    - **[Unit 0](unit-00.md) abgeschlossen** — Conda env, Godot .NET, erfolgreicher BallChase-Lauf
    - **[RL Essentials](unit-01.md) gelesen** — RL-Schleife, Belohnung, Policy, PPO auf hoher Ebene
    - **[Belohnungsdesign](unit-reward-engineering.md) gelesen** — potential-basiertes Shaping; du schreibst die geformte Belohnung in §5
    - Sicherer Umgang mit GDScript (Variablen, Funktionen, Signale — keine erweiterten Features)
    - Keine PyTorch-, keine SB3-Interna-, keine Game-Engine-Erfahrung erforderlich

!!! info "Zeit"
    Lesen: ~35 min · Training: ~30 min GPU / ~2 Std CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    Godot (`--viz` oder Editor) · TensorBoard · Belohnungs-/Beobachtungszeilen in `ai_controller.gd` / `lander.gd`

---

## Phase A — SimpleReachGoal als Aufwärmübung (~30–45 Min) { #phase-a }

**Erst ausführen, dann bauen**

Hol dir einen weiteren schnellen Erfolg, bevor du mit dem Lunar-Lander-Aufbau beginnst. Klone [SimpleReachGoal](https://github.com/edbeeching/godot_rl_agents_examples/tree/main/examples/SimpleReachGoal) und öffne es in Godot .NET (aktiviere das Godot-RL-Agents-Plugin).

```bash
conda activate godot_env
gdrl --experiment_name=unit2-warmup --timesteps=100000 --viz
```

Godot — öffne die SimpleReachGoal-Trainingsszene, drücke **F6** (Szene abspielen).

**Erkunden → Anpassen → Neu trainieren**

1. Öffne das Projekt in Godot; verfolge `AIController`, Raycasts und die Trainingsszene.
2. Ändere einen Sensorabstand oder einen Belohnungsterm (gleiche Idee wie in RL Essentials).
3. Trainiere kurz neu; vergleiche das Godot-Verhalten und TensorBoard mit deiner Vorhersage.

Dann geht es weiter zu Phase B — du kopierst diese Muster, du entdeckst sie nicht von Null an.

---

## Gymnasium → Godot Konzeptvergleich

| Gymnasium / SB3 | Godot RL Agents Entsprechung |
|-----------------|------------------------------|
| `gym.make("LunarLander-v2")` | Godot-Szene mit `RigidBody2D`-Lander |
| `env.observation_space` (8 Floats) | `get_obs()` gibt `{"obs": [...]}` zurück |
| `env.action_space` (4 diskrete) | `get_action_space()` gibt diskrete Größe 4 zurück |
| Belohnungslogik pro Schritt in Python | Belohnungsformung (reward shaping) in `lander.gd` + `_ai_controller.reward` |
| `make_vec_env(n_envs=16)` | N Kopien des Env-Wurzelknotens in `training_scene.tscn` |
| `PPO("MlpPolicy", env)` | `PPO("MultiInputPolicy", StableBaselinesGodotEnv(...))` |
| `model.learn(1_000_000)` | `gdrl --timesteps=1_000_000` |
| Gespeichertes `.zip`-Modell | ONNX-Modellpfad im Sync-Knoten-Inspektor |

!!! info "Phase B — Bauen"
    Wenn du [Phase A](#phase-a) abgeschlossen hast, hast du SimpleReachGoal bereits untersucht. Diese Tabelle zeigt, wie die Gymnasium-Konzepte zu Godot-Knoten für den Lunar Lander werden, den du als Nächstes erstellst.

!!! info "Setup bereits erledigt?"
    Wenn du Unit 0 abgeschlossen hast, überspringe Abschnitte 1–3 und beginne bei [Abschnitt 4 — Lander-Szene bauen](#4-lander-szene-bauen-landertscn).

---

## 1 · Benötigte Werkzeuge installieren

Wenn du Unit 0 abgeschlossen hast, sind deine Werkzeuge bereits installiert — springe zu [Abschnitt 4](#4-lander-szene-bauen-landertscn).

Andernfalls folge [Setup](setup.md) für Godot .NET, Miniconda, `godot_env` und das Plugin, und kehre dann hierher zurück.

---

## 2 · Das Godot-Projekt anlegen

- Öffne den Godot **.NET-Editor**, klicke auf **New Project**
- Gib dem Projekt einen Namen, z. B. `LunarLanderGodot`
- Wähle einen Ordner, wähle **Renderer: Compatibility** (am schnellsten für das Training)
- Klicke auf **Create & Edit**

---

## 3 · Das godot-rl-Godot-Plugin installieren

Siehe [Setup → Godot-Plugin](setup.md#godot-plugin-godot-rl-agents) für Installations- und Aktivierungsschritte. Überprüfe, ob unter „Add Node" `Sync` und `AIController2D` angezeigt werden, bevor du weitermachst.

---

## 4 · Lander-Szene bauen (`lander.tscn`)

**Zu erstellende Knotenhierarchie**

```
Lander            ← RigidBody2D  (root, script: lander.gd)
  ├─ CollisionShape2D  (capsule or polygon shape)
  ├─ Sprite2D          (your lander artwork, or placeholder)
  ├─ LeftLeg           (Area2D — detects ground contact)
  │   └─ CollisionShape2D
  ├─ RightLeg          (Area2D — detects ground contact)
  │   └─ CollisionShape2D
  └─ AIController2D    (script: ai_controller.gd)
```

!!! tip "Verwende ein 2D-Projekt"
    Das entspricht am genauesten der ursprünglichen LunarLander-v2-Physik. Eine 3D-Version ist ebenfalls möglich — erweitere dafür stattdessen `AIController3D`.

**Boden- und Landeplatz-Knoten hinzufügen**

- Erstelle einen `StaticBody2D` für den Boden mit einer `CollisionShape2D`
- Platziere zwei kleine `StaticBody2D`-Pads als Landezielpunkte
- Füge eine `Camera2D` hinzu, die dem Lander folgt (optional, hilfreich beim Beobachten des Trainings)
- Exportiere eine `@export var landing_pad_position: Vector2` aus dem Lander-Szenen-Wurzelknoten, damit sie in der Belohnungsfunktion (reward function) ausgelesen werden kann

**Bein-Kontaktsignale verbinden**

Wähle jeden `Area2D`-Beinknoten aus, gehe zu **Node → Signals**, verbinde `body_entered` und `body_exited` mit `lander.gd`, um die booleschen Variablen `left_leg_contact` / `right_leg_contact` zu setzen.

---

## 5 · `lander.gd` schreiben (Physik + Belohnung)

!!! warning "Training stagniert?"
    Prüfe der Reihe nach: (1) Vorzeichen und Skalierung der Belohnung (reward) — ist „gut" wirklich positiv? (2) Sparse rewards — bekommt der Agent (agent) irgendein Signal vor dem Ziel? (3) Beobachtungsfehler (observation bugs) — werden Sensoren nach einem Reset aktualisiert? (4) TensorBoard flach, aber Godot sieht gut aus — eventuell ist längeres Training oder ein Viz-Checkpoint nötig.

```gdscript
extends RigidBody2D

# ── Exported config ──────────────────────────────────
@export var landing_pad_position : Vector2 = Vector2(0, 300)
@export var main_thrust           : float  = 800.0
@export var side_thrust           : float  = 300.0

# ── State ────────────────────────────────────────────
var left_leg_contact  : bool = false
var right_leg_contact : bool = false
var _thrust           : Vector2 = Vector2.ZERO
var _firing_main      : bool = false
var _firing_side      : bool = false

@onready var _ai : AIController2D = $AIController2D

# ── Setup ─────────────────────────────────────────────
func _ready() -> void:
    _ai.init(self)
    reset()

# ── Reset ─────────────────────────────────────────────
func reset() -> void:
    position = Vector2(
        landing_pad_position.x + randf_range(-150, 150),
        50.0
    )
    rotation           = 0.0
    linear_velocity    = Vector2(randf_range(-30,30), randf_range(-10,10))
    angular_velocity   = 0.0
    left_leg_contact   = false
    right_leg_contact  = false
    _thrust            = Vector2.ZERO

# ── Thrust API called by AIController ────────────────
func set_thrust(direction: Vector2, is_main: bool) -> void:
    _thrust      = direction
    _firing_main = is_main
    _firing_side = (direction != Vector2.ZERO) and not is_main

# ── Physics loop ─────────────────────────────────────
func _physics_process(_delta: float) -> void:
    if _ai.needs_reset:
        _ai.reset()
        reset()
        return

    if _ai.heuristic == "human":
        _read_human_input()

    if _thrust != Vector2.ZERO:
        apply_central_force(_thrust)

    # ── Per-step reward shaping (mirrors LunarLander-v2) ──
    var dist = global_position.distance_to(landing_pad_position)
    _ai.reward -= dist * 0.003
    _ai.reward -= abs(linear_velocity.x) * 0.001
    _ai.reward -= abs(linear_velocity.y) * 0.001
    _ai.reward -= abs(rotation)          * 0.002
    if left_leg_contact:  _ai.reward += 0.01
    if right_leg_contact: _ai.reward += 0.01
    if _firing_main: _ai.reward -= 0.30
    if _firing_side: _ai.reward -= 0.03

    _thrust = Vector2.ZERO

# ── Human input ───────────────────────────────────────
func _read_human_input() -> void:
    if Input.is_action_pressed("ui_up"):
        set_thrust(Vector2.UP * main_thrust, true)
    elif Input.is_action_pressed("ui_left"):
        set_thrust(Vector2.LEFT * side_thrust, false)
    elif Input.is_action_pressed("ui_right"):
        set_thrust(Vector2.RIGHT * side_thrust, false)

# ── Terminal outcomes ─────────────────────────────────
func game_over(terminal_reward: float) -> void:
    _ai.reward     += terminal_reward
    _ai.done        = true
    _ai.needs_reset = true

# ── Leg contact callbacks (connect in editor) ─────────
func _on_left_leg_body_entered(_body):  left_leg_contact  = true
func _on_left_leg_body_exited(_body):   left_leg_contact  = false
func _on_right_leg_body_entered(_body): right_leg_contact = true
func _on_right_leg_body_exited(_body):  right_leg_contact = false
```

!!! info "Absturz- und Landeerkennung"
    Rufe `game_over(-100.0)` auf, wenn der Körper zu schnell den Boden berührt (prüfe `linear_velocity.y` in `body_entered` an einem Boden-`Area2D`), und `game_over(+100.0)`, wenn beide Beine landen und die Geschwindigkeit gering ist.

---

## 6 · `ai_controller.gd` schreiben (RL-Schnittstelle)

```gdscript
extends AIController2D

# ── Observation space (8 floats, matches LunarLander-v2) ─
func get_obs() -> Dictionary:
    var lander = get_parent() as RigidBody2D
    var pad    = lander.landing_pad_position

    return {"obs": [
        (lander.global_position.x - pad.x) / 300.0,
        (lander.global_position.y - pad.y) / 300.0,
        lander.linear_velocity.x            / 200.0,
        lander.linear_velocity.y            / 200.0,
        lander.rotation                     / PI,
        lander.angular_velocity             / 5.0,
        float(lander.left_leg_contact),
        float(lander.right_leg_contact),
    ]}

# ── Action space: 4 discrete ──────────────────────────
func get_action_space() -> Dictionary:
    return {
        "engine": {"size": 4, "action_type": "discrete"}
    }

# ── Apply action ──────────────────────────────────────
func set_action(action) -> void:
    var lander = get_parent()
    match int(action["engine"]):
        0: lander.set_thrust(Vector2.ZERO,                       false)
        1: lander.set_thrust(Vector2.LEFT  * lander.side_thrust, false)
        2: lander.set_thrust(Vector2.UP    * lander.main_thrust, true)
        3: lander.set_thrust(Vector2.RIGHT * lander.side_thrust, false)

# ── Reward passthrough ────────────────────────────────
func get_reward() -> float:
    return reward
```

!!! info "Episode-Timeout und Reset erfolgen automatisch"
    Der Basisknoten `AIController2D` zählt Schritte und setzt `needs_reset`, sobald sein exportierter `reset_after`-Wert überschritten wird — kein eigener `_physics_process` ist hier nötig. `lander.gd` überwacht `needs_reset` und lässt den Lander neu erscheinen.

!!! tip "Menschliche Steuerung liegt in lander.gd"
    godot-rls `AIController` hat keinen `get_user_input()`-Hook. Tastatureingaben werden in `lander.gd` gelesen, das prüft, ob `_ai.heuristic == "human"` gilt, wenn Sync im HUMAN-Modus ist.

---

## 7 · Trainingsszene bauen (`training_scene.tscn`)

**Szenenstruktur**

```
TrainingScene    (Node2D, root)
  ├─ Sync        (add via Add Node → search "Sync")
  ├─ Env_0       (instance of your lander.tscn)
  ├─ Env_1       (another instance)
  ├─ Env_2
  └─ …Env_N      (duplicate as many as you want, e.g. 8–16)
```

!!! tip "Mehr Umgebungen = schnelleres Training"
    Jede Instanz läuft parallel im selben Godot-Prozess und alle liefern Daten an den Trainer. 8–16 Instanzen sind ein guter Ausgangspunkt für das CPU-Training.

**Den Sync-Knoten konfigurieren**

Wähle den **Sync**-Knoten aus und setze folgende Eigenschaften im Inspektor:

| Eigenschaft | Wert für das Training |
|-------------|----------------------|
| Control Mode | `TRAINING` |
| Speed Up | `20` (oder höher auf schneller Hardware) |
| Action Repeat | `1` |
| ONNX Model Path | *leer lassen* |

**Die Env-Instanzen verteilen**

Verschiebe jede `Env_N`-Instanz an eine andere Position, damit sie sich visuell nicht überlappen. Sie müssen nicht sichtbar sein, aber es hilft beim Debuggen.

---

## 8 · Zuerst mit menschlicher Steuerung testen

!!! warning "Diesen Schritt nicht überspringen"
    Manuelles Spielen ist der schnellste Weg, Fehler in der Beobachtungsnormalisierung und Belohnungsformung zu finden, bevor stundenlange Trainingszeit verschwendet wird.

1. Setze im Sync-Knoten-Inspektor den **Control Mode** auf `HUMAN`.
2. Drücke **F6** (Aktuelle Szene ausführen). Steuere den Lander mit den Pfeiltasten. Überprüfe im Godot-Output-Panel, dass:
    - Belohnungen (rewards) sich ansammeln (positiv in der Nähe des Pads, negativ weiter entfernt)
    - Die Flags `done` / `needs_reset` korrekt bei Absturz oder Landung ausgelöst werden
    - Die Episode (episode) zurückgesetzt wird und der Lander neu erscheint
3. Setze den **Control Mode** des Sync-Knotens vor dem Training wieder auf `TRAINING`.

---

## 9 · Training starten (im Editor)

!!! info "Training im Editor ist der empfohlene Weg"
    Python und Godot kommunizieren über einen lokalen Socket, sodass du direkt aus dem Editor heraus trainieren kannst — kein exportiertes Binary erforderlich. Dies ist der einfachste und portabelste Ansatz; auf macOS vermeidet er außerdem architekturabhängige Probleme mit exportierten Binaries. Für großangelegtes paralleles Training mit einem exportierten Binary, siehe den optionalen [Abschnitt 12](#12-spiel-binary-exportieren-optional).

**Schritt 1 — Conda-Umgebung aktivieren**

```bash
conda activate godot_env
```

**Schritt 2 — gdrl-Training-Listener starten**

`gdrl` wird mit `pip` installiert — es gibt kein Skript zum Herunterladen:

```bash
gdrl --experiment_name=ppo-lunarlander-godot \
     --timesteps=1_000_000 \
     --save_model_path=lander_ppo \
     --onnx_export_path=lander_ppo.onnx
```

Die Konsole hält bei `waiting for remote GODOT connection on port 11008` an — das ist erwartet.

**Schritt 3 — In Godot auf Play drücken, um zu verbinden**

Wechsle zum Godot-Editor, öffne `training_scene.tscn` und drücke **F6** (Szene abspielen). Godot verbindet sich mit dem wartenden `gdrl`-Prozess und SB3 beginnt, eine Metriktabelle auszugeben.

**Was zu erwarten ist — `ep_rew_mean`-Meilensteine**

| ep_rew_mean | Bedeutung |
|-------------|-----------|
| < 0 | Agent (agent) stürzt noch ab / driftet — anfangs normal |
| 50–150 | Grundlegende Stabilität gelernt, arbeitet an der Landung |
| ≥ 200 | Landet zuverlässig — ein gelöster Lunar Lander |

!!! note "Einfrieren während des Trainings ist normal"
    Bei der Standard-SB3-PPO-Konfiguration friert Godot kurz ein, während Modellgewichte aktualisiert werden (die Python-Seite blockiert den Socket). Das ist kein Absturz.

**Training nach Unterbrechung fortsetzen**

`--save_model_path=lander_ppo` schreibt `lander_ppo.zip`. Zum Fortsetzen:

```bash
gdrl --resume_model_path=lander_ppo.zip \
     --timesteps=500_000 \
     --onnx_export_path=lander_ppo.onnx
```

**Checkpoints automatisch speichern**

```bash
# Add this flag to save a checkpoint every 50 000 steps:
--save_checkpoint_frequency=50000
```

Checkpoints werden unter `logs/sb3/<experiment_name>/` gespeichert und können mit `--resume_model_path` geladen werden.

!!! check "Fertig, wenn"
    Der mittlere Episodenertrag erreicht **`ep_rew_mean ≥ 200`** — der Standardwert für „gelöst" bei LunarLander. Ein gesunder Lauf klettert über 200 und hält sich dort; beobachte das im Editor-Log oder in TensorBoard. Wenn du nach 1M Schritten noch unter ~100 liegst, steckt fast immer ein Fehler in der Belohnungsformung oder den Beobachtungen — arbeite die Checkliste *Training stagniert?* ab, bevor du länger trainierst.

---

## 10 · Training mit TensorBoard überwachen

```bash
# In a second terminal:
conda activate godot_env
tensorboard --logdir=logs/sb3
```

Dann öffne `http://localhost:6006`.

**Wichtige Metriken**

| Metrik | Bedeutung | Ziel |
|--------|-----------|------|
| `rollout/ep_rew_mean` | Mittlere Episode-Belohnung | ≥ 200 |
| `rollout/ep_len_mean` | Mittlere Episodenlänge | Stabilisiert sich |
| `train/policy_gradient_loss` | PPO-Richtlinienverlust | Sinkende Tendenz |
| `train/entropy_loss` | Explorations-Entropie | Langsam sinkend |
| `train/approx_kl` | Richtlinienänderung pro Update | Bleibt klein (< 0.02) |

**Tuning-Tipps bei stagniertem Training**

- Belohnung (reward) steigt nie → Beobachtungsnormalisierung prüfen; `get_obs()` ausgeben und sicherstellen, dass kein Wert immer 0 oder gesättigt ist
- Belohnung schwankt → Lernrate verringern: `--learning_rate=0.0001` hinzufügen
- Training ist langsam → **Speed Up** des Sync-Knotens erhöhen oder mehr `Env`-Instanzen hinzufügen. Für echtes `--n_parallel`-Skalieren ein Binary exportieren (Abschnitt 12)
- Policy-Gradient-Verlust explodiert → `--clip_range` auf `0.1` reduzieren

---

## 11 · ONNX exportieren und Inferenz in Godot ausführen

**Die ONNX-Datei wird automatisch exportiert**

Da du `--onnx_export_path=lander_ppo.onnx` übergeben hast, wird die Datei geschrieben, wenn das Training endet (oder wenn du es mit **Ctrl+C** abbrichst).

**Das Modell ins Projekt importieren**

Ziehe `lander_ppo.onnx` direkt in Godots **FileSystem**-Dock und platziere es unter `res://lander_ppo.onnx`.

**Eine Inferenzszene erstellen (`onnx_inference_scene.tscn`)**

Dupliziere `training_scene.tscn`. In der Kopie:

- Behalte nur **eine** Env-Instanz (lösche die übrigen)
- Wähle den **Sync**-Knoten und setze:

| Eigenschaft | Wert |
|-------------|------|
| Control Mode | `ONNX Inference` |
| ONNX Model Path | `res://lander_ppo.onnx` |
| Speed Up | `1` (oder niedriger — verlangsamte Darstellung sieht schön aus) |

**Die Inferenzszene ausführen**

Öffne `onnx_inference_scene.tscn` und drücke **F6** — *kein Python erforderlich*. Beobachte, wie der trainierte Agent autonom landet.

!!! info "Kein Python zur Laufzeit"
    Das ONNX-Modell läuft vollständig innerhalb von Godot über die kompilierte C# OnnxRuntime-Schicht des Plugins. Das Spiel benötigt weder Python noch eine aktive Conda-Umgebung — du kannst es als eigenständiges Spiel ausliefern.

**Überprüfen, ob der Agent funktioniert** — füge einen temporären Debug-Print zu `lander.gd` hinzu:

```gdscript
func game_over(terminal_reward: float) -> void:
    if terminal_reward > 0:
        print("✅ Landed! reward=", terminal_reward)
    else:
        print("💥 Crashed or timed out")
    ...
```

Ein gut trainierter Agent sollte in der Mehrzahl der Episoden (episodes) erfolgreich landen.

---

## 12 · Spiel-Binary exportieren (optional)

!!! warning "Optional — nur für großangelegtes paralleles Training"
    Training im Editor (Abschnitt 9) ist der empfohlene Weg für Units 0–2. Exportiere ein Binary nur, wenn du mehrere Godot-Prozesse gleichzeitig mit `--n_parallel` für höheren Durchsatz ausführen möchtest — typischerweise auf einem Linux-Trainingsrechner oder in der Cloud. Auf macOS sind exportierte Binaries architekturabhängig und umständlich, daher ist Training im Editor dort vorzuziehen.

1. Setze `training_scene.tscn` als **Main Scene** unter **Project → Project Settings → Application → Run**
2. **Project → Export…** → **Add…** → Plattform auswählen (Linux/Windows/macOS)
3. Export-Templates bei Aufforderung herunterladen (**Manage Export Templates → Download**)
4. Auf **Export Project…** klicken, einen Ordner wählen, z. B. `build/LunarLander`
5. Das Binary ausführbar machen (Linux/macOS):

```bash
chmod +x build/LunarLander/LunarLander.x86_64
```

**Gegen das Binary mit `--env_path` trainieren**

```bash
gdrl --env_path=build/LunarLander/LunarLander.x86_64 \
     --n_parallel=4 \
     --speedup=20 \
     --experiment_name=ppo-lunarlander-godot \
     --timesteps=1_000_000 \
     --onnx_export_path=lander_ppo.onnx
```

---

## Referenz: Beobachtungsraum (observation space)

| Index | Wert | Normalisierung | Bereich |
|-------|------|----------------|---------|
| 0 | Horizontaler Abstand zum Pad | `/ 300.0` | −1 … 1 |
| 1 | Vertikaler Abstand zum Pad | `/ 300.0` | −1 … 1 |
| 2 | Horizontale Geschwindigkeit | `/ 200.0` | −1 … 1 |
| 3 | Vertikale Geschwindigkeit | `/ 200.0` | −1 … 1 |
| 4 | Rotationswinkel | `/ PI` | −1 … 1 |
| 5 | Winkelgeschwindigkeit | `/ 5.0` | −1 … 1 |
| 6 | Bodenkontakt linkes Bein | boolean → float | 0 oder 1 |
| 7 | Bodenkontakt rechtes Bein | boolean → float | 0 oder 1 |

!!! tip "Normalisierung ist wichtig"
    Alle Werte sollten ungefähr in [−1, 1] liegen. Wenn ein Wert viel größer werden kann, sättigt das neuronale Netz und das Training stagniert. Passe die Divisoren an deinen Szenenmaßstab an.

## Referenz: Aktionsraum (action space)

| Aktionsindex | Aktiviertes Triebwerk | Aufgebrachte Kraft |
|-------------|----------------------|--------------------|
| 0 | Keines (nichts tun) | `Vector2.ZERO` |
| 1 | Linkes Lagekontrolltriebwerk | `Vector2.LEFT * side_thrust` |
| 2 | Haupttriebwerk (aufwärts) | `Vector2.UP * main_thrust` |
| 3 | Rechtes Lagekontrolltriebwerk | `Vector2.RIGHT * side_thrust` |

## Referenz: Belohnungsfunktion (reward function)

| Ereignis | Belohnungsänderung | Typ |
|----------|--------------------|-----|
| Abstand zum Pad | `− dist × 0.003` pro Schritt | Geformt |
| Horizontale Geschwindigkeit | `− \|vx\| × 0.001` pro Schritt | Geformt |
| Vertikale Geschwindigkeit | `− \|vy\| × 0.001` pro Schritt | Geformt |
| Neigungswinkel | `− \|θ\| × 0.002` pro Schritt | Geformt |
| Kontakt linkes Bein | `+ 0.01` pro Schritt | Geformt |
| Kontakt rechtes Bein | `+ 0.01` pro Schritt | Geformt |
| Haupttriebwerk aktiv | `− 0.30` pro Schritt | Geformt |
| Seitentriebwerk aktiv | `− 0.03` pro Schritt | Geformt |
| Sichere Landung | `+ 100.0` | Terminal |
| Absturz | `− 100.0` | Terminal |
| Zeitüberschreitung | `0.0` | Terminal |

## Referenz: `gdrl`-Kommandozeilenargumente

| Argument | Standard | Beschreibung |
|----------|----------|--------------|
| `--env_path` | Keine | Pfad zu einem exportierten Binary. Weglassen für Training im Editor. |
| `--timesteps` | 1 000 000 | Gesamte Umgebungsschritte. |
| `--n_parallel` | 1 | Zu startende Binary-Instanzen. Erfordert `--env_path`. |
| `--speedup` | 1 | Godot-Physik-Geschwindigkeitsmultiplikator (Binary-Modus). |
| `--experiment_name` | experiment | In TensorBoard angezeigter Name. |
| `--experiment_dir` | logs/sb3 | TensorBoard-Log-Verzeichnis. |
| `--n_steps` | 64 | Schritte pro Umgebung pro PPO-Rollout. |
| `--batch_size` | 64 | PPO-Minibatch-Größe. Muss `n_steps × n_envs` teilen. |
| `--learning_rate` | 0.0003 | Adam-Lernrate. |
| `--ent_coef` | 0.0001 | Entropie-Bonus (fördert Exploration). |
| `--clip_range` | 0.2 | PPO-Clipping-Bereich. |
| `--onnx_export_path` | Keine | ONNX-Modell nach dem Training exportieren. |
| `--save_model_path` | Keine | SB3-`.zip`-Checkpoint speichern. |
| `--save_checkpoint_frequency` | Keine | Checkpoint alle N Schritte speichern. |
| `--resume_model_path` | Keine | Vorhandenes `.zip` laden, um Training fortzusetzen. |
| `--viz` | false | Godot-Fenster beim Training gegen ein Binary anzeigen. |
| `--linear_lr_schedule` | false | Lernrate linear bis 0 über das Training verringern. |
| `--inference` | false | Inferenz ausführen (kein Training) aus einem geladenen Modell. |

---

## Checkliste

!!! success "Du bist bereit für Unit 3, wenn..."
    1. Godot 4 **.NET-Edition** installiert, sowie das .NET SDK
    2. `godot_env` mit Python 3.10 erstellt; `godot-rl[sb3]` und `tensorboard` installiert
    3. Neues Godot-Projekt erstellt; godot-rl-Plugin installiert und aktiviert
    4. `lander.tscn` gebaut: `RigidBody2D` + zwei Bein-`Area2D` + `AIController2D`
    5. `lander.gd` geschrieben: Physik, Schub, Belohnungsformung, `game_over()`, `reset()`
    6. `ai_controller.gd` geschrieben: `get_obs()`, `get_action_space()`, `set_action()`, `get_reward()`
    7. `training_scene.tscn` gebaut: Sync-Knoten + 8–16 Env-Instanzen
    8. Manuell im HUMAN-Modus getestet — Belohnung und Reset funktionieren korrekt
    9. Im Editor trainiert: `ep_rew_mean ≥ 200` erreicht
    10. TensorBoard geöffnet; Trainingskurven sehen gesund aus
    11. `lander_ppo.onnx` importiert und im Sync-Knoten geladen; Inferenzszene läuft korrekt
    12. *(Optional)* Spiel-Binary für `--n_parallel`-Training exportiert (Abschnitt 12)

## 13 · Stretch Goals

**Einen zweiten Sensor hinzufügen.** Füge einen zweiten `RayCast2D` hinzu, der 45° versetzt vom ersten zeigt. Exportiere das Binary neu, trainiere 300k Schritte nach und vergleiche die TensorBoard-Kurven. Navigiert der Agent Ecken flüssiger? Prüfe `get_obs()` — ändert das Hinzufügen eines Sensors `obs_size` im Sync-Knoten? (Das sollte es — baue den Beobachtungsraum (observation space) bei Bedarf neu auf.)

**Mehrere Ziele pro Episode.** Ändere deine Umgebung (environment) so, dass der Agent **drei Ziele nacheinander** erreichen muss, bevor `done = true` gilt. Erzeuge jedes folgende Ziel zufällig, wenn das vorherige erreicht wurde. Du musst `get_reward()`, `get_done()` ändern und einen `_goals_reached`-Zähler in deinem Skript verfolgen. Tipp: Gib eine kleine geformte Belohnung für jeden Schritt näher an das aktuelle Ziel, plus einen größeren Bonus beim Erreichen.

**Zielposition beim Reset zufällig setzen.** In `reset()` erzeuge das Ziel an einer zufälligen `Vector2`-Position innerhalb der Arena-Grenzen statt an einer festen Position. Führe 500k Schritte durch und vergleiche mit deiner Baseline mit festem Ziel. Der Agent muss nun über Zielpositionen hinweg verallgemeinern — braucht er mehr Zeitschritte, um zu konvergieren?

```gdscript
func reset() -> void:
    var arena_half = 8.0
    goal.global_position = Vector2(
        randf_range(-arena_half, arena_half),
        randf_range(-arena_half, arena_half)
    )
    _prev_dist_to_goal = agent.global_position.distance_to(goal.global_position)
    _ai.reset()
```

!!! info "Selbstcheck, bevor du weitermachst"
    Kannst du diese Fragen mit eigenen Worten beantworten?

    1. Was stellt der `AIController` der Python-Seite bereit, und was besitzt `lander.gd`?
    2. Warum gibt `get_obs()` *normalisierte* Floats zurück statt roher Positionen und Geschwindigkeiten?
    3. Was ist der Unterschied zwischen einer *terminalen* Belohnung (Absturz / Landung) und einer *schrittweise geformten* Belohnung, und warum brauchst du für LunarLander beide?
    4. Worüber entscheidet der *Control Mode* des Sync-Knotens, und was tut jeder Modus?
    5. Was gibt dir der ONNX-Export nach dem Training, was der `.zip`-SB3-Checkpoint nicht bietet?

    Wenn du alle fünf beantworten kannst — bist du bereit.

??? success "Antworten zum Selbstcheck"
    1. Der **`AIController`** stellt die RL-Schnittstelle zu Python bereit — `get_obs()`, `get_action_space()`, `set_action()`, `get_reward()`. **`lander.gd`** besitzt das Spiel selbst — Physik, Schub, Kollisionen, Belohnungsformung, `game_over()`, `reset()`. Der Controller ist die Brücke; das Spielskript ist die Welt.
    2. Neuronale Netze trainieren am besten mit Eingaben ähnlicher, begrenzter Skala (≈[-1, 1]). Rohe Positionen und Geschwindigkeiten unterscheiden sich um Größenordnungen, was die Gradienten schlecht konditioniert und das Lernen verlangsamt oder destabilisiert. Normalisieren hält die Updates wohlverhalten.
    3. Eine **terminale** Belohnung (Landung +, Absturz −) definiert das eigentliche Ziel, ist aber sparse; eine **schrittweise geformte** Belohnung (näher zur Plattform, Winkel-/Treibstoffstrafen) gibt dichte Führung, sodass der Agent ein Signal bekommt, *bevor* er je landet. Du brauchst die terminale Belohnung, um Erfolg zu definieren, und die geformte, um ihn lernbar zu machen.
    4. Der **Control Mode** wählt, wer die Agenten steuert: *Training* schickt Beobachtungen an Python und wendet die zurückgegebenen Aktionen an; *ONNX Inference* führt die exportierte Policy lokal ohne Python aus; *Human* lässt dich manuell steuern, um Belohnungen und Resets zu testen.
    5. **ONNX** ist ein portables, framework-unabhängiges Modell, das Godot zur Inferenzzeit ohne Python oder SB3 ausführen kann. Der `.zip`-Checkpoint lädt nur in SB3 unter Python zurück.

**Was kommt als Nächstes:** Als Nächstes kommt **Q-Learning** — die tabellarische Intuition, die DQN verständlich macht — dann **CrossTheRoad** mit **DQN** in Unit 3.

[→ Q-Learning](unit-q-learning.md)
