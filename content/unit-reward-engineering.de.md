# Reward Engineering — Signale entwerfen, die wirklich funktionieren

[← RL Foundations Deep Dive](unit-rl-foundations-deep.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~35 min

!!! info "Drei Wege, deine KI zu beobachten"
    - **Godot-Viz** — der einzige Ort, an dem Reward Hacking sichtbar ist. Hoher TensorBoard-Score plus seltsam wirkender Agent in Godot heißt: deine Belohnung ist kaputt, nicht dein Algorithmus.
    - **TensorBoard** — die Form von `ep_rew_mean` über die Zeit erzählt die ganze Geschichte: flach = kein Signal, steigend = Lernen, oszillierend = Gewichtungs-Ungleichgewicht.
    - **Reward-Sanity-Check-Skript** — fahre vor dem Training 100 zufällige Episoden. Sieht eine zufällige Policy nie eine positive Belohnung, wird dein Agent das auch nicht.

---

## 1 · Warum Reward Engineering zählt

Die Belohnung ist das **einzige** Signal, das der Agent von der Welt bekommt. Die Umgebung sagt ihm nicht, was zu tun ist — sie reicht nach jeder Aktion nur eine Zahl zurück. Diese Zahl ist die *gesamte* Definition von „Erfolg" für den Agenten.

Das bedeutet:

- Eine schlechte Belohnung = **kein Lernen**, **falsches Verhalten** oder **Reward Hacking**.
- Der schwierigste Teil von angewandtem RL ist nicht die Wahl zwischen PPO, SAC oder DQN — sondern die Belohnung richtig hinzubekommen.
- Du wirst mehr Zeit mit dem Tuning der Belohnung verbringen als mit Hyperparametern. Deutlich mehr.

### Die drei Fehlermodi

Jede kaputte Belohnung fällt in einen dieser Eimer:

1. **Kein Signal** — der Agent lernt nie, weil jede Episode dieselbe flache Belohnung liefert. Häufig bei reinen spärlichen Belohnungen schwerer Aufgaben.
2. **Falsches Signal** — der Agent lernt *etwas*, aber nicht, was du wolltest. Die geschriebene Belohnung passt nicht zum gedachten Verhalten.
3. **Belohnung austricksen** — der Agent findet eine unbeabsichtigte Abkürzung. Die Belohnung wird technisch maximiert, aber so, dass es den Zweck verfehlt.

### Ein reales Beispiel

OpenAI trainierte einen Bootsrennen-Agenten in einem Spiel, in dem Treibstoff-Pickups kleine Boni gaben und das Ziel einen großen. Der Agent entdeckte, dass er **in Kreisen nahe einem respawnenden Treibstoff-Cluster fahren** und so mehr Punkte sammeln konnte als beim eigentlichen Rennen. Er beendete kein einziges Rennen, aber sein TensorBoard-Score war prima.

Das ist die kanonische Reward-Hacking-Geschichte. Die Lehre: **dein Agent weiß nicht, was du *gemeint* hast. Er kennt nur die Zahl, die du gibst.**

---

## 2 · Dichte vs. spärliche Belohnungen

Jedes Belohnungsdesign liegt auf einem Spektrum zwischen zwei Extremen.

### Spärliche Belohnung

Der Agent erhält ein Signal **nur am Ende** der Episode (oder zu seltenen Schlüsselmomenten).

- **Sauber**: trifft das echte Ziel exakt. Keine Mehrdeutigkeit, was du willst.
- **Problem**: die meisten Episoden produzieren null Signal. Ohne Gradient, dem zu folgen wäre, strampelt der Agent.
- **Beispiel**: CrossTheRoad — nur `+1` beim Ziel, `-1` beim Tod. Alles dazwischen null.

```gdscript
# Pure sparse reward
if reached_goal:
    _ai.reward += 1.0
    _ai.done = true
elif died:
    _ai.reward -= 1.0
    _ai.done = true
# All other steps: reward = 0
```

### Dichte Belohnung

Der Agent erhält ein Signal in **jedem Schritt**.

- **Liefert ständig Lernsignal** — es gibt immer eine Richtung zur Verbesserung.
- **Risiko**: der Agent optimiert das *dichte* Signal statt das echte Ziel. Die dichte Belohnung wird zum Ziel.
- **Beispiele**: Abstand zum Ziel, Geschwindigkeit zum Ziel, Höhengewinn, Ausrichtung zum Zielkurs.

```gdscript
# Pure dense reward
_ai.reward += velocity_toward_goal * 0.01
_ai.reward -= distance_to_goal * 0.001
```

### Die Kernerkenntnis

> **Dichte Belohnungen müssen mit dem spärlichen Ziel konsistent sein.**

Widerspricht deine dichte Belohnung der spärlichen, wird der Agent die **dichte** optimieren. Warum? Weil die dichte ständig feuert und die spärliche nur einmal. In Summe gewinnt fast immer die dichte.

**Der Diagnose-Test:** Kann ein Mensch auf der dichten Belohnung hoch scoren, *während er das spärliche Ziel verfehlt*?

- **Ja** → deine dichte Belohnung ist fehlausgerichtet. Neu entwerfen.
- **Nein** → die dichte Belohnung ist zumindest konsistent zum Ziel.

Beispiel einer fehlausgerichteten dichten Belohnung: „Geschwindigkeit" belohnen ohne „Geschwindigkeit *zum Ziel hin*". Der Agent rast vergnügt in die falsche Richtung.

---

## 3 · Potential-basiertes Reward-Shaping (Die Theorie)

Es gibt tatsächlich eine formale Theorie, die dir sagt, welche dichten Belohnungen sicher sind. Sie stammt von Ng, Harada und Russell (1999).

### Das Theorem

Jede Belohnung der Form

$$F(s, s') = \gamma \cdot \Phi(s') - \Phi(s)$$

**erhält die optimale Policy** des zugrundeliegenden MDP.

- $\Phi(s)$ heißt **Potentialfunktion** — eine beliebige Funktion des Zustands.
- Addiere $F$ zur Umweltbelohnung: $r_{shaped} = r + \gamma \cdot \Phi(s') - \Phi(s)$
- Die optimale Policy unter $r_{shaped}$ ist **dieselbe** wie unter $r$.

### Warum das funktioniert (Intuition)

Summierst du den Shaping-Term $F$ über eine Episode, teleskopieren die aufeinanderfolgenden $\Phi$-Terme:

$$\sum_t F(s_t, s_{t+1}) = \gamma \Phi(s_T) - \Phi(s_0)$$

Es hängt nur von Start- und Endzustand ab. Es belohnt keinen besonderen *Pfad* — es addiert nur einen konstanten Offset über die Episode. Die Anreizstruktur bleibt erhalten.

### Gängige Potentiale

| Potential Φ(s)         | F = γΦ(s') − Φ(s)         | Effekt                          |
|------------------------|---------------------------|---------------------------------|
| `-distance_to_goal`    | Fortschritt zum Ziel      | Bewegung zum Ziel               |
| `height`               | Höhengewinn               | Klettern                        |
| `speed_toward_goal`    | Beschleunigung zum Ziel   | Schnelleres Vorankommen         |
| `-time_elapsed`        | Konstanter negativer Offset | Tempo / Effizienz             |

!!! tip "Potential-basiertes Shaping ist der sichere Default"
    Wenn du unsicher bist, ob eine dichte Belohnung das Verhalten verfälscht, schreibe sie als `γ·Φ(s') - Φ(s)` statt als rohe Belohnung. Du bekommst das Lernsignal, ohne die optimale Policy zu verändern.

!!! warning "Beliebige dichte Belohnungen haben diese Eigenschaft NICHT"
    Eine Belohnung wie `+0.01 dafür, in Zone A zu sein` ist **nicht** potential-basiert und **kann** die optimale Policy verändern. Der Agent campt evtl. in Zone A, selbst wenn das dem echten Ziel schadet.

---

## 4 · Belohnungs-Komponenten in GDScript

Das Standardmuster, um Belohnungen in einem Godot-Agent-Skript zusammenzusetzen:

```gdscript
extends RigidBody3D

@onready var _ai = $AIController3D
@onready var goal = get_node("../Goal")

var _prev_dist_to_goal: float = 0.0
var time_alive: float = 0.0
var max_time: float = 30.0
var max_dist: float = 50.0

func _physics_process(delta):
    if _ai.needs_reset:
        reset()
        return

    time_alive += delta

    # --- Dense shaped reward (runs every physics step) ---
    var dist_to_goal = global_position.distance_to(goal.global_position)
    var prev_dist    = _prev_dist_to_goal
    _prev_dist_to_goal = dist_to_goal

    # Progress reward: potential-based (γ·Φ(s') - Φ(s))
    # Φ(s) = -dist_to_goal, so progress = prev_dist - dist_to_goal
    var progress = (prev_dist - dist_to_goal) / max_dist
    _ai.reward += progress * 0.5

    # Survival bonus: small positive per step to discourage suicide
    _ai.reward += 0.001

    # Velocity penalty: discourages erratic jitter
    _ai.reward -= linear_velocity.length() * 0.0001

    # --- Sparse terminal reward (fires once on episode end) ---
    if dist_to_goal < 1.0:
        _ai.reward += 1.0     # goal reached
        _ai.done = true
        _ai.needs_reset = true
    elif time_alive > max_time:
        _ai.reward -= 0.5     # timeout penalty
        _ai.done = true
        _ai.needs_reset = true

func reset():
    # CRITICAL: initialize ALL reward state here, not in _ready()
    _prev_dist_to_goal = global_position.distance_to(goal.global_position)
    time_alive = 0.0
    _ai.needs_reset = false
```

### Die kaputteste Zeile in Studi-Code

> **Initialisiere `_prev_dist_to_goal` immer in `reset()`, nicht in `_ready()`.**

Initialisierst du es in `_ready()`, hält `_prev_dist_to_goal` nach dem *ersten* Reset noch den Wert der *vorherigen* Episode. Im ersten Schritt der neuen Episode ist `progress` ein riesiger Spike (positiv oder negativ, je nach Position). Der Agent lernt aus diesem Schritt Müll.

Das ist der häufigste stille Bug in Studi-Belohnungsfunktionen. Er stürzt nichts ab. Er macht das Training nur verrauscht und langsam.

---

## 5 · Skalierung und Normalisierung der Belohnung

Neuronale Netze lernen am besten, wenn Eingaben und Ziele in einem vorhersehbaren Bereich liegen. Ist deine Belohnung manchmal `+1000` und meist `0.0001`, muss die Wertfunktion sechs Größenordnungen überspannen — das tut sie nicht.

### Skalierungs-Richtwerte

| Komponente                  | Magnitude          |
|-----------------------------|--------------------|
| Terminale Belohnungen (gewinnen/verlieren) | ±1,0 bis ±10,0 |
| Pro-Schritt-Shaped-Belohnungen | 0,001 bis 0,01 |
| Überlebensboni              | ~0,001             |
| Pro-Schritt-Strafen         | 0,0001 bis 0,001   |

Faustregel: **Pro-Schritt-Shaped-Belohnungen sollten deutlich kleiner sein als terminale Belohnungen.**

Wenn pro Schritt >> terminal: der Agent ignoriert das Ziel und farmt das Pro-Schritt-Signal. Der Episoden-Return wird von „wie viele Schritte habe ich überlebt und kleine Belohnungen gesammelt" dominiert, nicht von „bin ich erfolgreich".

Wenn pro Schritt << terminal: du hast praktisch eine spärliche Belohnung. Ob das ein Problem ist, hängt davon ab, ob der Agent durch reine Erkundung die terminale Belohnung erreicht.

### Der Normalisierungs-Check

Nach Belohnungsdesign, aber **vor** dem PPO-Training: 100 Episoden mit einer **Zufalls-Policy** laufen lassen und Statistiken drucken:

```python
# Quick reward sanity check script
import numpy as np
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="./MyEnv.x86_64", n_parallel=1, speedup=1)

rewards_per_episode = []
step_rewards = []
terminal_hits = 0

for ep in range(100):
    obs = env.reset()
    total = 0.0
    done = False
    while not done:
        action = env.action_space.sample()   # random policy
        obs, r, done, info = env.step(action)
        total += r
        step_rewards.append(r)
    rewards_per_episode.append(total)
    if total > 0:
        terminal_hits += 1

print(f"Random policy stats over 100 episodes:")
print(f"  mean episode reward: {np.mean(rewards_per_episode):.3f}")
print(f"  min episode reward:  {min(rewards_per_episode):.3f}")
print(f"  max episode reward:  {max(rewards_per_episode):.3f}")
print(f"  per-step reward range: [{min(step_rewards):.4f}, {max(step_rewards):.4f}]")
print(f"  episodes with positive terminal: {terminal_hits}/100")
env.close()
```

### Was die Ausgabe verrät

- **`terminal_hits == 0`** bei 100 Zufallsepisoden → dein spärliches Signal ist unerreichbar. Shaped Reward oder Curriculum hinzufügen.
- **Pro-Schritt-Bereich riesig** (z. B. `[-100, 100]`) → neu skalieren. Das Netz passt das nicht.
- **`ep_rew_mean ≈ 0`** in den ersten 100k–500k Trainingsschritten → du hast ein **Signal**-Problem, kein **Algorithmus**-Problem. Aufhören, PPO zu drehen, und die Belohnung fixen.

---

## 6 · Häufige Fehlermodi und Lösungen

Diese Bugs wirst du treffen. Alle. Früher oder später.

### Fehler 1: Vorzeichen falsch

- **Symptom**: `ep_rew_mean` geht über das Training *runter*. Der Agent wird *schlechter*, je mehr er trainiert.
- **Ursache**: irgendwo wurde ein Vorzeichen verkehrt. Der Agent maximiert korrekt die (negierte) Sache.
- **Fix**: Vorzeichen der schuldigen Komponente kippen.
- **Debug**: Belohnungswerte in den ersten Schritten einer Episode drucken:

```gdscript
print("step=%d reward=%.4f progress=%.4f" % [step, _ai.reward, progress])
```

### Fehler 2: Agent farmt die Shaped-Belohnung

- **Symptom**: hoher TensorBoard-Score, aber der Agent erreicht in Godot nie das Ziel.
- **Beispiel**: Agent pendelt vor dem Ziel hin und her, um die Fortschrittsbelohnung zu sammeln, ohne die Schwelle zu überqueren.
- **Fix**: Gewicht der Shaped-Komponente senken; Zeitstrafe pro Schritt hinzufügen, damit Trödeln Belohnung kostet; sicherstellen, dass die terminale Belohnung echt größer ist als das Farmbare.

### Fehler 3: Belohnung zu spärlich, Agent lernt nie

- **Symptom**: `ep_rew_mean` über 500k+ Schritte flach; der Random-Sanity-Check zeigte 0 positive terminale Episoden.
- **Ursache**: die Exploration trifft das Ziel nie zufällig — kein positives Signal zum Lernen.
- **Fix**: potential-basiertes Shaping (Abstand zum Ziel ist am sichersten); Aufgabe kürzen; Curriculum Learning; oder Neugier ergänzen (siehe Curiosity-Unit).

### Fehler 4: Reset mitten im Fortschritt

- **Symptom**: Agent macht ein paar Schritte gut, dann „gibt auf" und stirbt schnell.
- **Ursache**: `reset()` stellt `_prev_dist_to_goal` nicht wieder her. Der erste Schritt der neuen Episode erzeugt einen riesigen negativen Spike in `progress`. Der Agent lernt: Leben ist schlecht und schnelles Sterben vermeidet den Spike.
- **Fix**: **alle** Belohnungs-State-Variablen in `reset()` zurücksetzen. Checkliste jeder beteiligten Variable führen.

### Fehler 5: Reward Cliffs

- **Symptom**: der Agent meidet ansonsten belohnte Regionen. Verhalten wirkt seltsam konservativ.
- **Ursache**: hohe negative Belohnung an einer Grenze (z. B. `-100` beim Plattformfall). Die Wertfunktion bekommt einen riesigen Negativ-Spike nahe der Grenze, und die Policy weigert sich, sich zu nähern, selbst wo es nützlich wäre.
- **Fix**: terminale Strafen 5–10× größer skalieren als typische Pro-Schritt-Belohnungen, **nicht** 100×. Der Agent soll Misserfolg meiden, nicht in Angst erstarren.

---

## 7 · Reward Hacking und wie man es verhindert

!!! warning "Reward Hacking"
    **Definition**: der Agent findet einen unbeabsichtigten Weg zu hoher Belohnung, der das echte Ziel verfehlt. Das ist kein Bug im Agenten — es ist ein Bug in deiner Belohnung.

### Berühmte Beispiele

- **Bootsrennen-Agent** (OpenAI): kreist um einen Treibstoff-Cluster statt zu rennen. Mehr Punkte als die Renner.
- **Simulierter Walker** (DeepMind): „läuft", indem er nach vorn fällt und einen Physik-Glitch nutzt, der gratis x-Fortschritt gibt.
- **Greifroboter** (Forschungslabor): hält den Greifer vor die Kamera, damit der „Objekt nicht gegriffen"-Detektor falsch antwortet. Belohnung = „Objekt gegriffen" technisch erreicht.
- **Schummelnder Block-Stapler**: kippt den Block auf die Seite, weil „Oberkante hoch" erfüllt ist, wenn er horizontal und höher als breit liegt.

In jedem Fall tut der Agent exakt das, was die Belohnung verlangt. Die Belohnung hat nur nicht das Richtige verlangt.

### Präventionsstrategien

1. **Echtes Ziel spezifizieren**: stets eine terminale spärliche Belohnung für das *tatsächliche* Ziel. Wie gut das dichte Shaping auch ist — die spärliche Belohnung ankert „Erfolg".
2. **Shaped-Komponenten begrenzen**: jede zusätzliche Shaped-Belohnung ist ein potenzielles Schlupfloch. Weniger Signale = weniger Schlupflöcher. Mit einem Shaping-Term starten, erst bei Bedarf erweitern.
3. **In Godot zuschauen**: Viz-Checkpoint öffnen. Reward Hacking ist für Menschen *sofort* sichtbar — der Agent wirkt komisch, tut Dasselbe immer wieder, ignoriert offensichtliche Pfade.
4. **Belohnungsbedingungen randomisieren**: hängen Shaped-Belohnungen an festen Level-Eigenschaften (z. B. fixe Plattform-Position), merkt der Agent das und nutzt es aus. Layouts, Spawn-Positionen, Zielorte randomisieren.
5. **Adversarial gegen deine Belohnung testen**: frage „was ist das dümmste Verhalten, das hier hoch scort?" Das ist oft genau, was der Agent findet.

---

## 8 · Multi-Objective-Belohnungen

Echte Aufgaben enthalten meist Trade-offs. Ein Renn-Agent soll schnell sein *und* nicht crashen. Ein Lieferroboter soll schnell sein *und* nicht in Personen rennen.

```gdscript
# Example: racing agent — fast but not crashy
_ai.reward += speed_toward_finish * 0.3     # go fast
_ai.reward -= collision_force * 0.5         # don't crash
if finished:
    _ai.reward += 5.0                       # actually finish the race
    _ai.done = true
```

### Gewichtungs-Tuning

Behandle Belohnungs-Gewichte als **Hyperparameter**. Sie sind so wichtig wie die Lernrate.

Vorgehen:

1. Default-Gewichte wählen (bestes Bauchgefühl).
2. Kurzes Experiment laufen lassen (500k Schritte).
3. **Ein** Gewicht ändern und erneut laufen.
4. Läufe in TensorBoard vergleichen.

| Gewichtskonfig                | Beobachtetes Verhalten        |
|-------------------------------|-------------------------------|
| `speed=0.3, crash=-0.5`       | Schnell, einige Crashes       |
| `speed=0.1, crash=-1.0`       | Langsam, aber sicher          |
| `speed=0.3, crash=0.0`        | Schnell, ständige Crashes     |
| `speed=0.0, crash=-0.5`       | Stillstehen, um Crashes zu vermeiden |

Zwei Dinge:

- **`speed=0.0`** erzeugt einen Agenten, der nichts tut. Ein positiver Anreiz wegzunehmen ist genauso zerstörerisch wie ihn falsch zu kalibrieren.
- **`crash=0.0`** erzeugt einen Agenten, der Sicherheit komplett ignoriert. Negative Anreize zählen.

Die besten Gewichte findet man meist per Trial-and-Error. Es gibt keine geschlossene Lösung. Budget dafür einplanen.

---

## 9 · Curriculum-Learning-Vorschau

Manchmal sind selbst gut entworfene Belohnungen für einen rein zufällig erkundenden Agenten zu schwer zu finden.

Beispiel: in Lunar Lander mit kleiner, ferner Landefläche landet eine Zufalls-Policy nie. Ohne erfolgreiche Landungen kein positives Signal. Training stockt.

Lösung: **einfach anfangen, schwerer werden**.

- **Manuelles Curriculum**: Agent früh näher am Ziel spawnen. Wenn `ep_rew_mean` steigt, Abstand erhöhen.
- **Automatisches Curriculum**: `ep_rew_mean` gegen eine Schwelle prüfen. Schlägt der Agent Schwelle A, weiter zu B. Schlägt er B, weiter zu C.
- **Domain Randomization**: Startbedingungen randomisieren, Bereich wächst über die Zeit.

Skizze in GDScript:

```gdscript
# In your environment manager
var curriculum_level: int = 0
var success_buffer: Array = []

func on_episode_end(success: bool):
    success_buffer.append(success)
    if success_buffer.size() > 100:
        success_buffer.pop_front()
    var rate = success_buffer.count(true) / float(success_buffer.size())
    if rate > 0.8 and curriculum_level < 5:
        curriculum_level += 1
        success_buffer.clear()
        print("Advancing curriculum to level ", curriculum_level)
```

Ein eigenes großes Thema — siehe Stretch Goals und eine spätere Unit.

---

## 10 · Sicherheits-Constraints im Belohnungsdesign

!!! info "Vorerst überspringen — in der Robotik-Phase zurückkommen"
    Dieser Abschnitt betrifft den Einsatz auf echter Hardware. Komm zurück, wenn du [Sim-to-Real Transfer](unit-sim-to-real.md) oder [Safe RL](unit-safe-rl.md) in Phase 6 erreichst — nichts in den Phasen 1–5 hängt davon ab.

Für die meisten Kursprojekte heißt „schlechte Episode": der Agent kippt oder verfehlt das Ziel. Auf echter Hardware kann eine schlechte Episode ein gebrochenes Servo, einen verbrannten Motor oder ein zerstörtes Getriebe bedeuten. Das Belohnungsdesign muss die Kostenstruktur des realen Systems abbilden.

### Harte Constraints vs. weiche Strafen

Es gibt zwei Wege, eine Nebenbedingung in der Belohnung zu kodieren:

```gdscript
# Soft penalty — discourages the behavior but does not terminate
if abs(joint_angle) > safe_range * 0.8:
    _ai.reward -= 0.1   # warns the policy to back off

# Hard constraint — terminates immediately and applies a large penalty
if abs(joint_angle) > hard_limit:
    _ai.reward -= 5.0   # strong negative signal
    _ai.done = true     # end the episode — this would break hardware
    _ai.needs_reset = true
```

Soft Penalties formen die Policy lange vor der harten Grenze in sicheren Betrieb. Harte Constraints beenden Episoden, die in physikalisch gefährliches Terrain laufen. Die zweistufige Struktur — Warnung bei 70 %, Termination bei 90 % — ist eine praktische Faustregel aus Sim-to-Real-Arbeiten.

### Häufige Hardware-Sicherheitsbedingungen

| Constraint | Godot-Umsetzung | Warum es zählt |
|---|---|---|
| Gelenkwinkel-Limits | Knochenrotation gegen Limit prüfen | Servo-Strip / mechanische Anschläge |
| Geschwindigkeits-Limits | `linear_velocity.length()` gegen Max prüfen | Motor-Überhitzung bei Dauerlast |
| Beschleunigungs-Limits | Velocity-Differenz pro Physikschritt prüfen | Getriebe-Schockbelastung bei abrupten Starts/Stopps |
| Bodenkontaktkraft | Normalkraft des Kontakts prüfen | Beinschaden bei harten Landungen |
| Arbeitsraum-Limits | `global_position`-Grenzen prüfen | Arm/Endeffektor trifft feste Oberfläche |

### Sicherheit vs. Lerntempo

Strenge Sicherheits-Constraints senken das Lerntempo, weil mehr Episoden früh enden — weniger Transitionen erreichen späte Aufgabenzustände. Zu lockere Constraints lassen die Policy gefährliche Verhaltensweisen finden, die auf echter Hardware Schaden anrichten würden.

Praktischer Startpunkt: Hart-Termination bei 90 % des physikalischen Limits; Soft Penalty ab 70 %. Trifft die Policy die Grenze zu oft, beide Schwellen senken oder den Soft-Penalty-Koeffizienten erhöhen.

!!! warning "Sicherheits-Constraints sind nicht gratis"
    Jede zusätzliche Terminations-Bedingung erhöht effektiv die Curriculum-Schwierigkeit. Stockt das Training nach einer Sicherheitsbedingung, prüfe, ob die Überlebenszeit in TensorBoard (`rollout/ep_len_mean`) stark gefallen ist. Sind Episoden sehr kurz, ist die Sicherheitsgrenze möglicherweise enger, als die Policy früh zuverlässig vermeiden kann — überlege ein curriculum-basiertes Verschärfen.

---

## 11 · Energieeffizienz im Belohnungsdesign

!!! info "Vorerst überspringen — in der Robotik-Phase zurückkommen"
    Energiestrafen zählen für echte Hardware und natürlich wirkende Gangarten. Komm zurück, wenn du [Fortbewegungsagenten](unit-locomotion.md) oder [Sim-to-Real Transfer](unit-sim-to-real.md) in Phase 6 erreichst.

Energieeffizienz zählt in zwei Kontexten:

- **Echte Roboter**: Akkulaufzeit, Motorhitze, mechanische Langlebigkeit — alle direkt davon abhängig, wie viel Leistung die Policy abruft.
- **Virtuelle Roboter**: Energiestrafen erzeugen natürlich wirkende, menschenähnlichere Bewegungen, indem sie die „spastischen" hochfrequenten Gelenk-Oszillationen unterdrücken, die unbeschränkte Policies gerne entdecken.

### Leistungsbasierte Strafe

Der physikalisch korrekte Weg: die tatsächlich verbrauchte Leistung an jedem Gelenk bestrafen.

```gdscript
# Power = force × velocity (translational) or torque × angular_velocity (rotational)
var translational_power = applied_force.dot(linear_velocity)
var rotational_power    = applied_torque.dot(angular_velocity)
var total_power = abs(translational_power) + abs(rotational_power)
_ai.reward -= total_power * power_penalty_coeff   # typically 0.0001 to 0.001
```

Der Koeffizient `power_penalty_coeff` ist hier der sensibelste Hyperparameter. Zu groß und die Policy friert (keine Bewegung = keine Leistung = hohe Belohnung). Zu klein und sie hat keinen Effekt. Mit `0.0001` starten und erhöhen, bis die Gangqualität steigt, ohne dass die Lokomotion stoppt.

### Aktions-Magnitude-Strafe (einfache Näherung)

Wenn Gelenkmoment-Daten nicht griffbereit sind, die Magnitude des Aktionsvektors direkt bestrafen:

```gdscript
# Penalize large actions regardless of outcome — reduces jerkiness
var action_magnitude = 0.0
for a in last_action.values():
    if a is Array:
        for v in a: action_magnitude += v * v
    else:
        action_magnitude += a * a
_ai.reward -= action_magnitude * 0.0005
```

Das ist dasselbe wie MuJoCos `ctrl_cost` (Summe der quadrierten Aktionskomponenten mal Kostenkoeffizient), in jedem Standard-Lokomotion-Benchmark. Die Intuition: große Aktionen verlangen große Kräfte, große Ströme, große Leistung. Aktions-Magnitude ist ein billiger, effektiver Proxy.

### Glättungs-Strafe

Die Energiestrafe senkt die mittlere Leistung. Die Glättungs-Strafe senkt die Peak-to-Peak-Variation — sie unterdrückt Policies, die rasch zwischen hohem und niedrigem Moment wechseln:

```gdscript
# Penalize rapid action changes (jerk) — produces smoother policies
if _prev_action != null:
    var action_delta = 0.0
    for key in last_action:
        var curr = last_action[key] if last_action[key] is float else last_action[key][0]
        var prev = _prev_action[key] if _prev_action[key] is float else _prev_action[key][0]
        action_delta += (curr - prev) * (curr - prev)
    _ai.reward -= action_delta * 0.001
_prev_action = last_action.duplicate(true)
```

Beachte `duplicate(true)` — eine tiefe Kopie ist nötig. Speicherst du nur die Referenz auf `last_action`, spiegelt `_prev_action` im nächsten Schritt die aktuelle Aktion, und das Delta ist immer null.

### Wann was

| Situation | Empfehlung |
|---|---|
| Simulierte Lokomotion, natürlich wirkender Gang erwünscht | Aktions-Magnitude-Strafe (einfach, effektiv) |
| Sim-to-Real-Transfer | Leistungsbasierte Strafe (physikalisch begründet) |
| Policy produziert Jitter/Vibration | Glättungs-Strafe auf Aktionsdeltas |
| Alle drei Probleme | Alle drei kombinieren mit separaten, abstimmbaren Koeffizienten |

---

## 12 · Reward-Engineering-Checkliste

Druck das aus. Klebe es an deinen Monitor.

### Vor dem Training

- [ ] Kann ich in **einem Satz** sagen, was der Agent maximieren soll?
- [ ] Gibt es eine **terminale Belohnung** für das echte Ziel (nicht nur Shaped Signals)?
- [ ] Ist die terminale Belohnung **erreichbar** durch eine Zufalls-Policy, selten reicht? (Sanity-Skript laufen lassen)
- [ ] Sind Pro-Schritt-Belohnungen **weniger als 1/10** der terminalen Magnitude?
- [ ] Habe ich **alle** Belohnungs-State-Variablen (`_prev_dist`, Zähler, Timer) in `reset()` initialisiert?
- [ ] Habe ich **100 Zufallsepisoden** gelaufen und `ep_rew_mean`, min, max gedruckt?
- [ ] Sind Shaped Rewards **potential-basiert**, wo möglich?
- [ ] Habe ich gefragt „wie scort man hier am dümmsten hoch?" — und das Schlupfloch ausgeschlossen?

### Nach 500 k Schritten

- [ ] Steigt `ep_rew_mean`? (Wenn nicht: Belohnungs- oder Beobachtungs-Problem, kein Algorithmus-Problem)
- [ ] Passt der Godot-Viz-Checkpoint zum TensorBoard-Score? (Wenn nicht: Reward Hacking)
- [ ] Tut der Agent **etwas Sinnvolles**, auch wenn suboptimal?
- [ ] Sind die Komponenten-Magnituden **ausbalanciert**? (Separat in TensorBoard loggen)

### Hardware-Sicherheit (für reale Roboter)

- [ ] Gelenklimitverletzungen beenden Episoden mit harter Strafe
- [ ] Soft Penalty ab 70 % des Limits; Hart-Strafe + `done = true` ab 90 %
- [ ] Energie-Effizienz-Strafe aktiv gegen Überhitzungsverhalten
- [ ] Aktions-Magnitude- oder Glättungs-Strafe gegen Verschleiß

Schlägt ein Punkt fehl: **Training stoppen. Belohnung fixen. Neustarten.** Mehr Schritte auf eine kaputte Belohnung zu werfen, hilft nie.

---

## 13 · Stretch Goals

Für mehr Übung nach der Unit:

1. **Curriculum nach Abstand** — Agent zufällig im `[0, max_dist]` spawnen. Nach Training Erfolgsquote gegen Spawn-Abstand plotten. Wo bricht der Agent ein?
2. **Belohnungs-Ablation** — drei Agenten auf derselben Aufgabe: nur terminale Belohnung, nur Shaped, beide. Lernkurven in TensorBoard vergleichen. Welcher konvergiert am schnellsten? Welcher zur besten Endpolicy?
3. **Kaputte Belohnung debuggen** — bewusst einen Vorzeichenfehler einbauen. 100 k Schritte trainieren. Den Bug allein aus der TensorBoard-Kurve identifizieren, ohne in den Code zu schauen. Das ist das häufigste reale Debug-Szenario.
4. **Reward Hacking provozieren** — kleine Umgebung entwerfen, deren Shaped Reward sich austricksen lässt; wie lange braucht der Agent für den Exploit? Dann die Belohnung fixen, bis der Hack verschwindet.

---

## Was kommt als Nächstes

Du hast jetzt die wichtigste Fähigkeit im angewandten RL: Belohnungen entwerfen, die das gewünschte Verhalten erzeugen.

In der nächsten Unit setzt du das direkt in die Praxis um: du baust **Lunar Lander** in Godot von Grund auf und schreibst die Per-Schritt-Belohnungsfunktion des Landers in GDScript. Das Shaping, das du dort in §5 schreibst, ist potential-basiert — genau die Theorie, die du gerade gelernt hast.

Wenn du nur eines aus dieser Unit mitnimmst: **die Belohnung ist keine Beschreibung des Ziels. Die Belohnung *ist* das Ziel.** Was immer du aufschreibst — das wird der Agent maximieren. Stelle sicher, dass es das ist, was du willst.

[← RL Foundations Deep Dive](unit-rl-foundations-deep.md) · [Kursstartseite](index.md) · [→ Unit 2: Lunar Lander in Godot bauen](unit-02.md)
