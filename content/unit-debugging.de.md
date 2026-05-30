# Debugging im RL-Training — Symptome, Diagnose und Fixes

[Kursstartseite](index.md) · [Referenz](reference.md)

!!! info "Zeit"
    Lesen: ~25 min

!!! info "Wie du diesen Leitfaden nutzt"
    Das ist eine diagnostische Referenz, kein Tutorial. Starte beim **Top-Level-Flowchart** unten, folge dem Zweig, der zu deinem Symptom passt, und springe zum referenzierten Abschnitt. Jeder Abschnitt ist gleich aufgebaut: **Symptom → Diagnose → konkreter Fix**. Das Prinzip „Drei Wege, deine KI zu beobachten" gilt auch hier — gegenprüfe **Godot-Verhalten**, **TensorBoard-Kurven** und **Code-Inspektion**, bevor du etwas änderst. Widersprechen zwei der drei, ist genau diese Diskrepanz der Hinweis.

## 1 · Wie du diesen Leitfaden liest

- Starte beim Top-Level-Flowchart in Abschnitt 2.
- Folge dem Symptom, das zu deiner Situation passt.
- Jeder nachgelagerte Abschnitt liefert **Symptom**, **Diagnose** und einen **konkreten Fix** — in dieser Reihenfolge.
- Bestätige die Diagnose stets mit allen drei Linsen, bevor du einen Fix anwendest:
    - **Godot-Verhalten** — was tut der Agent tatsächlich auf dem Bildschirm?
    - **TensorBoard** — was sagen die Kurven?
    - **Code-Inspektion** — tun `get_obs()`, `set_action()` und der Belohnungs-Code das, was du denkst?
- Wenn du vorschnell „einfach Dinge ausprobierst", behebst du das falsche Problem und verlierst Stunden. Erst diagnostizieren.

## 2 · Top-Level-Diagnose-Flowchart

```
Ändert sich ep_rew_mean überhaupt?
        |
   Nein (flach) ──────────────────────────→ Abschnitt 3: Kein Lernsignal
        |
   Ja (ändert sich)
        |
   Geht RUNTER? ──────────────────────────→ Abschnitt 4: Belohnungsvorzeichen / -skala
        |
   Steigt, plateaut aber früh? ────────────→ Abschnitt 5: Policy zu früh konvergiert
        |
   Oszilliert wild? ───────────────────────→ Abschnitt 6: Trainingsinstabilität
        |
   TensorBoard sieht gut aus, Godot falsch? → Abschnitt 7: Reward Hacking / Eval-Mismatch
        |
   Funktioniert im Training, nicht im Export? → Abschnitt 8: Inferenz- / ONNX-Probleme
        |
   Sehr langsam (steps/sec niedrig)? ──────→ Abschnitt 9: Performance
```

Öffne TensorBoard nebeneinander mit Godot, bevor du startest. Die meisten „mysteriösen" Fehler lösen sich, sobald du beide gleichzeitig sehen kannst.

## 3 · Kein Lernsignal (ep_rew_mean flach)

**Symptom:** `rollout/ep_rew_mean` bleibt 500 k+ Schritte nahe `0`. Das Trainingslog zeigt akkumulierende Schritte, aber kein Belohnungssignal. Auch die Episodenlänge ist möglicherweise verdächtig konstant.

Arbeite die Checkliste **der Reihe nach** ab. Überspringe nichts — frühere Prüfungen sind günstiger und schließen die häufigsten Ursachen aus.

### 3a. Belohnung ist immer null

**Symptom:** Belohnung feuert nie, auch nicht bei Erfolg.

**Diagnose:** Drucke die Belohnung in den ersten 100 Physikschritten:

```gdscript
func _physics_process(delta):
    print(_ai.reward)
```

Ist sie je ungleich null? Wenn nein, feuert deine Belohnungsbedingung nie. Häufigste Ursache: das `done`-Flag wird nie gesetzt, also endet die Episode nie und die terminale Belohnung wird nie ausgeführt.

**Fix:** Verifiziere, dass beide Zeilen in `game_over()` existieren:

```gdscript
func game_over():
    _ai.reward += 1.0
    _ai.done = true
    _ai.needs_reset = true
```

### 3b. Belohnungsvorzeichen falsch, aber ungleich null

Siehst du Belohnung ungleich null, aber mit falschem Vorzeichen, siehe **Abschnitt 4**.

### 3c. Beobachtungen sind alle null oder konstant

**Symptom:** Die Policy kann nicht lernen, weil jede Beobachtung für das Netz identisch aussieht.

**Diagnose:** Drucke `get_obs()` für 10 Schritte, während du den Agenten manuell bewegst:

```gdscript
if _ai.heuristic == "human":
    print(get_obs())
```

Sind alle Werte `0.0`? `RayCast3D`-Knoten sind evtl. nicht aktiviert, im falschen Collision-Layer oder erst nach `_ready()` in den Scene-Tree gehängt. Ändern sich die Werte, wenn sich der Agent bewegt? Wenn nicht, sind die Sensoren kaputt.

**Fix:** Strahlen explizit aktivieren (`enabled = true`), `collision_mask` prüfen und `force_raycast_update()` erzwingen, bevor du Treffer in `get_obs()` liest.

### 3d. Aktionsraum-Mismatch

**Symptom:** `get_action_space()` deklariert eine Form, `set_action()` liest eine andere. Die Policy gibt Müll relativ zu dem aus, was die Umgebung erwartet.

**Diagnose:** Lies beide Funktionen nebeneinander.

```gdscript
func get_action_space():
    return {"move": {"size": 4, "action_type": "continuous"}}

func set_action(action):
    velocity.x = action["move"][0]
    velocity.z = action["move"][1]
    # bug: only reads 2 of 4 declared dimensions
```

**Fix:** `size` muss exakt der Anzahl Elemente entsprechen, die du in `set_action()` liest. Bei diskreten Aktionen ist `size` die Anzahl der Wahlmöglichkeiten. Bei kontinuierlichen ist `size` die Anzahl Floats.

### 3e. Episode zu kurz

**Symptom:** Episoden enden in < 5 Schritten. Der Agent stirbt oder resettet, bevor er eine sinnvolle Belohnung trifft.

**Diagnose:** Drucke die Episodenlänge beim Reset:

```gdscript
func reset():
    print("ep_len:", _ai.episode_steps)
    _ai.episode_steps = 0
```

**Fix:** Füge einen kleinen Überlebensbonus hinzu, damit selbst kurze, ereignislose Episoden ein Gradientensignal erzeugen:

```gdscript
_ai.reward += 0.001  # per-step survival bonus
```

### 3f. Aufgabe für zufällige Erkundung tatsächlich zu schwer

**Symptom:** Eine Zufalls-Policy erreicht das Ziel in 10 k Episoden nie. Es gibt keinen Gradienten, dem PPO folgen könnte, weil jede Episode gleich aussieht.

**Fix:** Wähle eines oder mehrere:

- Füge eine **Shaped-Belohnung** basierend auf Abstands-Fortschritt hinzu (`prev_dist - current_dist`).
- **Spawne den Agenten zunächst näher am Ziel** und entferne ihn schrittweise (manuelles Curriculum).
- Nutze **intrinsische Neugier** (siehe Curiosity-Unit), sodass neue Zustände selbst belohnend sind.

## 4 · Belohnung sinkt (Agent wird schlechter)

**Symptom:** `ep_rew_mean` startet ungleich null, tendiert über die Zeit aber nach unten. Schließlich wirkt der Agent schlechter als zufällig.

### 4a. Belohnungsvorzeichen gedreht

**Symptom:** Agent lernt, das *Gegenteil* von dem zu tun, was du willst.

**Diagnose:** Lies jede `_ai.reward += ...`-Zeile und frage: „Ist dieser Wert positiv, wenn der Agent das Richtige tut?"

Klassischer Fehler:

```gdscript
_ai.reward += -dist_to_goal   # WRONG: becomes more negative as agent approaches!
```

Die Belohnung ist überall negativ, also lernt der Agent, die Episode möglichst schnell zu beenden, um keine weitere Strafe zu sammeln — indem er in eine Wand läuft.

**Fix:** Nutze **Fortschritt**, nicht den rohen Abstand:

```gdscript
_ai.reward += (prev_dist_to_goal - current_dist_to_goal)
prev_dist_to_goal = current_dist_to_goal
```

### 4b. Belohnungsskala explodiert

**Symptom:** Eine Belohnungskomponente überwiegt alle anderen. Der Agent optimiert nur den dominanten Term.

**Diagnose:** Drucke die Magnituden jeder Belohnungskomponente einzeln für 1000 Schritte:

```gdscript
print({"goal": r_goal, "progress": r_progress, "speed": r_speed})
```

Erreicht `r_speed` `100.0` pro Schritt, während die terminale Belohnung `+1.0` ist, ist das terminale Signal für den Optimierer unsichtbar.

**Fix:** Teile jede Pro-Schritt-Belohnungskomponente durch ihr erwartetes Maximum, sodass **alle Pro-Schritt-Belohnungen unter ~0,1 bleiben**. Die terminale Belohnung sollte dominieren.

### 4c. Falscher Algorithmus für den Aktionsraum

**Symptom:** Loss sieht bizarr aus; Agent verbessert sich nie; Training startet nie.

**Diagnose:** DQN unterstützt nur diskrete Aktionen. PPO und SAC unterstützen kontinuierliche (PPO auch diskrete).

**Fix:**

- Kontinuierlicher Aktionsraum → **PPO** oder **SAC**
- Diskreter Aktionsraum → **DQN** oder **PPO**

### 4d. Lernrate zu hoch

**Symptom:** Belohnung fällt nach kurzem Anstieg; `train/approx_kl` in TensorBoard konstant > `0.05`.

**Fix:** Lernrate halbieren:

```bash
--learning_rate=1e-4   # was 3e-4
```

## 5 · Policy zu früh konvergiert (Plateau)

**Symptom:** `ep_rew_mean` steigt rasch auf einen Wert und verbessert sich dann nicht mehr. Godot zeigt einen Agenten, der *etwas* Sinnvolles tut, aber nicht die volle Aufgabe löst.

### 5a. Entropie kollabiert

**Symptom:** `train/entropy_loss` in TensorBoard fällt plötzlich nahe null. Erkundung ist tot, die Policy hat sich auf ihr erstes brauchbares Verhalten festgesetzt.

**Fix:** Entropie-Koeffizient erhöhen, damit die Policy länger stochastisch bleibt:

```bash
--ent_coef=0.01    # try 0.01 to 0.05
```

### 5b. Lokales Optimum in der Belohnungslandschaft

**Symptom:** Der Agent hat ein Shortcut-Verhalten gefunden, das gut scort, aber nicht die volle Aufgabe ist.

**Beispiele:**

- Der Agent bleibt in der sicheren Spawn-Zone und sammelt Überlebensboni, statt zum Ziel zu gehen.
- Der Agent dreht sich auf der Stelle, um eine geschwindigkeitsbasierte Belohnung zu maximieren, ohne sich fortzubewegen.

**Fix:** Senke das Gewicht der ausgenutzten Hilfs-Belohnung. Stelle sicher, dass die **terminale Zielbelohnung über eine typische Episode die kumulierten Hilfs-Belohnungen dominiert**. Beobachte die Godot-Visualisierung, um das Shortcut-Verhalten präzise zu identifizieren.

### 5c. n_steps zu kurz

**Symptom:** Kurze Rollouts heißen, der Agent sieht immer nur die ersten Schritte einer Episode und kann Long-Horizon-Credit-Assignment nicht lernen.

**Fix:** `--n_steps` erhöhen:

```bash
--n_steps=2048   # was 64 or 128
```

### 5d. Zu wenig Training

**Symptom:** `ep_rew_mean` stieg noch, als du gestoppt hast.

**Fix:** Timesteps verdoppeln. Inspiziere stets die **Steigung** der Belohnungskurve, bevor du das Training für beendet erklärst. Eine über 200 k+ Schritte flache Kurve ist ein Plateau; eine weiterhin nach oben gebogene Kurve nicht.

## 6 · Trainingsinstabilität (oszillierende Kurven)

**Symptom:** `ep_rew_mean` oszilliert wild oder steigt und kollabiert dann auf null oder negativ.

!!! warning "NaN-Losses sind Notfälle"
    Siehst du auch nur einmal `nan` im Trainingslog, **stoppe das Training sofort**. NaN heißt, deine Modellgewichte sind korrumpiert. Ein Resume von einem NaN-Checkpoint rettet nichts. Weiter zu 6c.

### 6a. approx_kl zu groß

**Symptom:** `train/approx_kl` konstant > `0.05`. Die Policy ändert sich zwischen Updates zu schnell und überschießt.

**Fix:** Eines reduzieren:

```bash
--clip_range=0.1        # was 0.2
--learning_rate=1e-4    # was 3e-4
```

### 6b. Wertfunktion divergiert

**Symptom:** `train/value_loss` wächst über die Zeit, statt zu fallen.

**Fix:** Wertfunktions-Koeffizient senken und Belohnungsskala (Abschnitt 4b) prüfen:

```bash
--vf_coef=0.25   # was 0.5
```

### 6c. Gradient explodiert

**Symptom:** Trainingslog zeigt `NaN`-Loss oder `inf`-Gradienten.

**Fix:** Gradient-Norm kappen und prüfen, dass kein Beobachtungswert riesig ist:

```python
model = PPO("MlpPolicy", env, max_grad_norm=0.5, ...)
```

Dann verifizieren, dass keine Beobachtung Magnitude größer als ~10 hat (Abschnitt 6d).

### 6d. Beobachtungs-Normalisierungs-Problem

**Symptom:** Ein Obs-Wert übersteigt regelmäßig `±10`, sättigt Aktivierungen und erzeugt Gradienten, die normal aussehen, bis sie plötzlich explodieren.

**Diagnose:** Drucke das absolute Maximum der Beobachtung über 1000 Schritte:

```gdscript
var max_abs = 0.0
for v in get_obs()["obs"]:
    max_abs = max(max_abs, abs(v))
print("max_abs_obs:", max_abs)
```

**Fix:** Teile jede Obs-Komponente durch den maximal möglichen physikalischen Wert (siehe Normalisierungs-Abschnitt von Unit 6). Ziel: jede Beobachtung in `[-1, 1]` oder `[0, 1]`.

### 6e. Katastrophales Vergessen (nur DQN)

**Symptom:** Der Agent lernt ein Verhalten, vergisst es plötzlich. Belohnung fällt auf Baseline.

**Fix:**

- Replay-Buffer-Größe erhöhen.
- `learning_starts` senken, damit das Warm-up nicht dominiert.
- **Prioritized Experience Replay** (PER) hinzufügen, damit wichtige Transitionen erneut besucht werden.

## 7 · TensorBoard sieht gut aus, Godot sieht falsch aus

**Symptom:** `ep_rew_mean` ist hoch, aber in Godot tut der Agent etwas sichtbar Seltsames oder Unbeabsichtigtes.

!!! warning "Reward Hacking ist der gefährlichste Fehlermodus"
    Der Optimierer tut genau, was du verlangst. Sieht der Agent falsch aus, während die Belohnung richtig aussieht, ist **deine Belohnungsfunktion falsch** — nicht der Agent. Belohnung fixen, dann neu trainieren. Trainiere nie gegen eine Belohnungsfunktion, deren Ausnutzung du nicht mindestens 10 Episoden beobachtet hast.

### 7a. Reward Hacking

**Symptom:** Der Agent hat ein Verhalten mit hoher Belohnung gefunden, das nicht zum gedachten Ziel passt.

**Beispiele:**

- Auf der Stelle drehen, um eine fehlbenannte „Fortschrittsbelohnung" zu sammeln.
- Stillstehen, um eine Bewegungsstrafe zu vermeiden, ohne die Aufgabe je zu versuchen.
- Das Ziel wiederholt anstoßen, wenn die terminale Belohnung ohne `done = true` feuert.

**Fix:** Schau 10 volle Episoden in Godot. **Beschreibe in einem Satz, was der Agent tatsächlich tut.** Dann gestalte die ausgenutzte Belohnungskomponente neu. Oft ist der Fix, die terminale Belohnung größer und die Pro-Schritt-Boni kleiner zu machen.

### 7b. Stochastisch-deterministisch-Gap

**Symptom:** Training nutzt eine stochastische Policy (damit Erkundung wirkt); Evaluation nutzt evtl. stochastisch *oder* deterministisch, und beide sehen unterschiedlich aus.

**Fix:** Beim Beobachten des trainierten Modells stets deterministisch laufen und mit dem `--viz`-Flag bestätigen:

```python
action, _ = model.predict(obs, deterministic=True)
```

### 7c. Unterschiedliche Szene zwischen Training und Eval

**Symptom:** Das exportierte Binary hat andere Spawn-Positionen, Objekt-Platzierungen oder Physikparameter als die Editor-Szene, die für das Training genutzt wurde.

**Fix:** Re-exportiere das Binary aus dem **gleichen Commit**, der das Training erzeugt hat. Verifiziere, dass Spawn-Punkte, Physik-Tickrate und Randomisierungs-Seeds exakt übereinstimmen.

## 8 · ONNX-Inferenz-Probleme

**Symptom:** Du hast ein ONNX-Modell in Godot geladen, aber der Agent tut nichts oder bewegt sich zufällig.

### 8a. Falsche Sync-Knoten-Einstellungen

**Symptom:** Der Sync-Knoten ist noch im Trainingsmodus und versucht, mit einem Python-Server zu reden, der nicht da ist.

**Fix:** Am Sync-Knoten in der laufenden Szene:

- `Control Mode` muss `ONNX_INFERENCE` sein
- `Onnx Model Path` muss auf die tatsächliche `.onnx`-Datei auf der Disk zeigen
- Diese im Editor inspizieren, *während die Szene läuft*, nicht beim Editieren

### 8b. Beobachtungs-Shape-Mismatch

**Symptom:** Das ONNX-Modell erwartet eine bestimmte Eingabeform, aber `get_obs()` gibt nun eine andere Größe zurück.

**Fix:** Bestätige, dass `get_obs()` exakt dieselbe Anzahl Werte zurückgibt wie beim Training. **Re-exportiere das ONNX-Modell aus derselben Codebasis, die es trainiert hat.** Jede Änderung der Obs seit dem Training invalidiert das Modell.

### 8c. ONNX vom falschen Checkpoint exportiert

**Symptom:** Verhalten in der Inferenz sieht zufällig oder nach Früh-Training aus.

**Fix:** Exportiere aus `best_model.zip` (vom SB3-`EvalCallback` gespeichert), nicht aus dem Endmodell — das Endmodell kann ein Post-Kollaps-Checkpoint sein.

### 8d. Aktions-Skalierungs-Mismatch

**Symptom:** Die ONNX-Policy gibt rohe Werte aus; `set_action()` skaliert sie. Wurde `set_action()` nach dem Training geändert, stimmt die Skala nicht mehr.

**Fix:** Halte `set_action()` **byte-identisch** zwischen Training und Inferenz. Musst du die Skalierung ändern, trainiere neu.

## 9 · Performance (langsames Training)

**Symptom:** `steps/sec` viel niedriger als erwartet; Training dauert Stunden, wo es Minuten dauern sollte.

### 9a. Mit aktivierter Visualisierung gestartet

**Symptom:** `--viz` zu nutzen oder aus dem Godot-Editor zu starten, verlangsamt das Training 10–50×.

**Fix:** Für jeden Trainingslauf über 100 k Schritten das **exportierte Binary** ohne `--viz` nutzen. `--viz` nur, um ein *trainiertes* Modell zu inspizieren.

### 9b. n_parallel zu niedrig

**Symptom:** Durchsatz bleibt selbst auf schneller Maschine niedrig.

**Fix:** Mehrere Umgebungen parallel laufen lassen. Auf den meisten Maschinen sind 8–16 parallele Envs der Sweet Spot.

```bash
--n_parallel=8
```

Und 8 entsprechende In-Scene-Instanzen hinzufügen, damit jeder parallele Worker seine eigene hat.

### 9c. Physik-FPS zu hoch

**Symptom:** Godot-Physik steht auf 120 fps, obwohl 60 reichen. Jeder Schritt kostet doppelt so viel Rechenzeit.

**Fix:** Projekt → Projekteinstellungen → Physik → Common → **Physics Ticks Per Second = 60**.

### 9d. speedup nicht gesetzt

**Symptom:** Fehlt `--speedup=20`, läuft Godot in Echtzeit mit 1× — Wall-Clock-Training wird enorm langsam.

**Fix:** Stets `--speedup=20` (oder höher für einfache Envs, die keine präzise Physik brauchen):

```bash
--speedup=20
```

### 9e. GDScript-Bottleneck

**Symptom:** Selbst mit allem obigen ist der Durchsatz niedrig und die CPU auf der Godot-Seite voll ausgelastet.

**Diagnose:** Zeitstempel in `get_obs()` und die Belohnungsfunktion einbauen. Nach `for`-Schleifen oder Pro-Schritt-Allokationen suchen.

**Fix:** Alles cachen, was sich nicht jeden Frame ändert (Node-Referenzen, Material-Referenzen etc.). String-Konkatenation und Dictionary-Allokationen im Hot Path vermeiden.

## 10 · Schnelle Diagnose-Kommandos

!!! tip "Zuerst immer die Zufalls-Baseline laufen lassen"
    Bevor du „der Agent lernt nicht" debuggst, bestätige, was eine **Zufalls-Policy** scort. Scort Zufall 0,5 und dein Agent 0,5, tut der Agent nichts. Scort Zufall 0,5 und dein Agent 0,55, *findet* Training statt — nur langsam. Diese eine Zahl spart Stunden fehlgeleiteten Debuggens.

```bash
# Sanity-check training is producing reward at all
gdrl --env_path=./MyEnv.x86_64 --timesteps=50000 --n_parallel=1 --speedup=20
```

```python
# Random-policy baseline — what does no learning at all look like?
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import numpy as np

env = StableBaselinesGodotEnv(env_path='./MyEnv.x86_64', n_parallel=1, speedup=1)
rewards = []
for _ in range(20):
    obs, total = env.reset(), 0
    done = False
    while not done:
        obs, r, done, _ = env.step(env.action_space.sample())
        total += r
    rewards.append(total)
print(f'Random baseline: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}')
env.close()
```

```bash
# Watch 5 episodes of a trained model, deterministically
gdrl --env_path=./MyEnv.x86_64 \
  --resume_model_path=logs/sb3/myenv/best_model.zip \
  --inference --viz --timesteps=5000
```

## 11 · TensorBoard-Metriken-Referenz

| Metrik | Ort | Gesund | Aktion bei ungesund |
|--------|-----|--------|---------------------|
| `rollout/ep_rew_mean` | Rollout | Steigend | Siehe Abschnitt 3–5 |
| `rollout/ep_len_mean` | Rollout | Wächst | Zu kurz → siehe 3e |
| `train/approx_kl` | Train | 0,01–0,02 | >0,05 → lr/clip senken |
| `train/entropy_loss` | Train | Langsam fallend | Plötzlicher Abfall → ent_coef erhöhen |
| `train/value_loss` | Train | Fallend | Wächst → vf_coef senken |
| `train/policy_gradient_loss` | Train | Nahe 0 | Explodiert → lr senken |
| `train/explained_variance` | Train | Nähert sich 1,0 | Nahe 0 → Critic lernt nicht |
| `time/fps` | Time | >1000 | Niedrig → siehe Abschnitt 9 |

## 12 · Stretch Goals

**Einen Lauf absichtlich kaputtmachen.** Wähle einen Fehlermodus aus Abschnitt 3–6 — etwa „Belohnung sinkt" — und konstruiere einen Trainingslauf, der ihn zeigt. Einfachstes Rezept: Belohnung mit 100 multiplizieren und Entropie-Koeffizient senken. Dann durchlaufe das Diagnose-Flowchart in Abschnitt 2 *so, als wüsstest du nicht, was du getan hast*. Ziel ist, die Diagnose-Schritte in den Fingern zu spüren, bevor der echte Bug um 23 Uhr auftaucht.

**Eigenen Diagnose-Eintrag schreiben.** Such einen Bug, den du in diesem Kurs tatsächlich getroffen hast (egal welche Unit, dein eigener Trainingslauf). Schreib einen neuen Abschnitt im selben Format `Symptom → Diagnose → Fix` und reiche ihn als PR gegen diese Seite ein. Der Kurs will mehr Erste-Hand-Bug-Einträge, nicht weniger. Referenziere Issue #48, falls unklar ist, wohin damit.

**Eine Warnung automatisieren.** Wähle eine Metrik aus Abschnitt 11 und schreibe ein 20-Zeilen-Python-Skript, das die TensorBoard-`events.out.*`-Datei tailt (oder `tensorboard --logdir`-Daten pollt) und eine Warnung druckt, wenn die Metrik das gesunde Band verlässt — z. B. `explained_variance < 0.1` über 50 k Schritte. Es geht nicht um einen Produktionsmonitor; es geht darum zu begreifen, dass „TensorBoard beobachten" teilweise automatisierbar ist.

!!! warning "Pseudocode"
    ```python
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator("logs/sb3/myenv/PPO_1")
    ea.Reload()
    values = [s.value for s in ea.Scalars("train/explained_variance")]
    if values and values[-1] < 0.1:
        print(f"ALERT: explained_variance={values[-1]:.3f} — critic may not be learning")
    ```
