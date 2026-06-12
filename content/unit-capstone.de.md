# Capstone — Bau dein eigenes RL-Projekt

[← Ship Your Brain](unit-10.md) · [Kursstartseite](index.md)

!!! note "Voraussetzungen"
    - **Alle Phasen 1–3** — Units 0–4 und mindestens eine Theorie-Unit; du musst flüssig zwischen PPO / DQN / SAC wählen können
    - **[Unit 10](unit-10.md)** — du kannst ONNX exportieren und ein Binary shippen
    - **[Debugging-Guide](unit-debugging.md)** in einem anderen Tab offen — du wirst ihn brauchen
    - Eine konkrete Projektidee, wie roh auch immer — Abschnitt 8 hat eine Liste, falls du eine brauchst

!!! info "Zeit"
    Lesen: ~20 min · Training: variiert

---

!!! info "Drei Wege, deine KI zu beobachten"
    Deine eigene Godot-Umgebung — ein Agent, den du entworfen hast, in einer Szene, die du gebaut hast · TensorBoard, das eine Belohnungskurve verfolgt, die du geschrieben hast · die finale itch.io-Seite oder das Desktop-Build, das du in die Welt geshippt hast.

---

Du hast jede Unit abgeschlossen. Du weißt, wie Q-Learning Wert-Schätzungen bootstrapped, wie PPO seine eigenen Gradient-Updates clippt, warum SAC Entropie maximiert, wie Neugier Erkundung antreibt, wie Imitation Learning aus Demonstrationen bootstrapped und wie man einen Roboterarm physisch simuliert. Jetzt fallen die Stützräder weg.

Diese Unit ist ein **Gerüst**, kein Tutorial. Es gibt keine Starter-Szene, keine vorgeschriebene Reward-Funktion, keine Referenz-Implementierung, gegen die du prüfen kannst. Du wählst ein Problem, entwirfst das MDP, wählst den Algorithmus, fährst die Trainings-Pipeline, evaluierst das Ergebnis und shippst es. Die folgenden Abschnitte führen dich durch jede Entscheidung in der Reihenfolge, in der sie auf dich zukommt.

---

## 1 · Eine Umgebung wählen

Der häufigste Capstone-Fehler ist, einen zu großen Scope zu wählen. Ein zweiräumiges Labyrinth, das in einer Stunde gelernt wird, lehrt dich mehr über RL als ein halbfertiges Open-World-Spiel. Starte kleiner, als du denkst, shippe es und expandiere von dort.

### Entscheidungsbaum

```
Is the task physically continuous (arms, drones, cars)?
├── Yes → Robotics / physics sim path
│         Recommended: unit-robotics.md, unit-sac.md
│         Action space: continuous (Box)
│         Consider: unit-sim-to-real.md if you want real hardware
└── No  → Game / grid path
          ├── Turn-based or discrete movement?
          │   └── Yes → Discrete actions (Discrete or MultiDiscrete)
          │             Recommended: unit-q-learning.md, unit-03.md
          └── Real-time, smooth motion?
              └── Yes → Continuous or hybrid actions
                        Recommended: unit-ppo-deep.md, unit-sac.md
```

```
How many agents are there?
├── One → Single-agent (start here)
└── Multiple → Are they cooperative, competitive, or mixed?
    ├── Competitive (zero-sum) → self-play, unit-self-play.md
    ├── Cooperative → MAPPO / shared reward, unit-self-play.md
    └── Mixed → contact the course forum before proceeding
```

```
How dense is your reward signal?
├── Agent gets feedback every step → Dense reward → standard PPO / SAC
└── Agent only finds out at the end → Sparse reward
    ├── Episode < 200 steps → try shaped reward first (unit-reward-engineering.md)
    └── Episode > 200 steps → add curiosity (unit-curiosity.md) or HER (unit-her.md)
```

### Empfohlene Startpunkte nach Ziel

| Dein Ziel | Umgebungs-Typ | Aktionsraum | Algorithmus | Schwierigkeit |
|---|---|---|---|---|
| Schnell die volle Pipeline lernen | 2D-Grid-Spiel | Diskret | PPO | Einsteiger |
| Glatte Fortbewegung | 2D- oder 3D-Physik | Continuous Box | SAC | Mittel |
| Adversarial-KI | Beliebiges 2-Spieler-Spiel | Diskret oder kontinuierlich | PPO + Self-Play | Mittel |
| Langhorizon-Planung | Puzzle / Strategie | Diskret | PPO + Neugier | Mittel |
| Echter Roboter | Physik-Sim | Continuous Box | SAC | Fortgeschritten |
| Emergentes Multi-Agent-Verhalten | Beliebig | Diskret oder kontinuierlich | MAPPO / Self-Play | Fortgeschritten |

!!! warning "Scope-Creep ist Capstone-Killer Nr. 1"
    Wähle **eine** Umgebung, **einen** Agententyp und **ein** Erfolgskriterium, bevor du eine Zeile GDScript schreibst. Wenn deine Projektbeschreibung das Wort „und" mehr als zweimal enthält, halbiere sie. Komplexität kannst du nach deiner ersten funktionierenden Policy immer noch ergänzen.

---

## 2 · Beobachtungs-Design

Der Beobachtungsvektor ist alles, was der Agent zu jedem Timestep über die Welt weiß. Mach ihn falsch und der Agent ignoriert entweder den Großteil seiner Eingaben (zu viel Info) oder kann die Aufgabe nie lösen, weil er nicht weiß, wo er ist (zu wenig Info).

### Die Sensor-Checkliste

Bevor du eine Komponente zum Beobachtungsvektor hinzufügst, stelle für jeden Kandidatenwert diese Fragen:

1. **Ändert er sich sinnvoll mit dem Agentenzustand?** Wenn der Wert über alle Episoden nahezu konstant ist, entferne ihn.
2. **Ist er in der Realität beobachtbar?** Gib dem Agenten keine Information, die eine echte Entität nicht wahrnehmen könnte (z. B. das interne Ziel des Gegners).
3. **Ist er redundant zu einer anderen Komponente?** Stark korrelierte Eingaben verschwenden Kapazität und verlangsamen das Lernen.
4. **Ist er normalisiert?** Neuronale Netze erwarten Eingaben in einem konsistenten Bereich. Alles muss in `[-1, 1]` oder `[0, 1]` liegen.

### Normalisierung in GDScript

```gdscript
# BAD — raw world coordinates (can be thousands of units)
obs.append(global_position.x)
obs.append(global_position.y)

# GOOD — normalised relative position (always -1 to 1 if arena is 200 units wide)
const ARENA_HALF_WIDTH := 100.0
const ARENA_HALF_HEIGHT := 100.0
obs.append(clamp(global_position.x / ARENA_HALF_WIDTH, -1.0, 1.0))
obs.append(clamp(global_position.y / ARENA_HALF_HEIGHT, -1.0, 1.0))
```

```gdscript
# BAD — unnormalised velocity (unbounded)
obs.append(linear_velocity.x)

# GOOD — velocity normalised by maximum expected speed
const MAX_SPEED := 500.0
obs.append(clamp(linear_velocity.x / MAX_SPEED, -1.0, 1.0))
obs.append(clamp(linear_velocity.y / MAX_SPEED, -1.0, 1.0))
```

### Typische Beobachtungs-Fehler

| Fehler | Symptom | Fix |
|---|---|---|
| Pixelpositionen in einer großen Welt einbeziehen | Policy lernt nichts, Value-Loss explodiert | Auf arena-relative Koordinaten normalisieren |
| Zielposition exponieren, aber nicht den relativen Vektor zum Ziel | Policy konvergiert langsam | `(goal_pos - agent_pos).normalized()` als Obs hinzufügen |
| Verbleibende Episodenzeit einbeziehen | Policy ignoriert sie früh, übergewichtet sie spät | Entfernen, außer Zeitdruck ist eine Kernmechanik |
| Beobachtungen am Episodenende nicht zurücksetzen | Veraltete Werte aus voriger Episode bluten durch | Allen mutablen Obs-State in `_reset()` zurücksetzen |
| Partielle Beobachtbarkeit ohne Gedächtnis | Policy oszilliert, konvergiert nie | RecurrentPPO hinzufügen (siehe [unit-08](unit-08.md)) |

!!! tip "Minimal starten"
    Beginne mit dem kleinsten Beobachtungsvektor, der die Aufgabe theoretisch lösen könnte. Füge Komponenten nur hinzu, wenn das Training stagniert und du identifizieren kannst, welche Information fehlt. Eine 6-Element-Obs, die funktioniert, schlägt eine 60-Element-Obs, die das Netz verwirrt.

---

## 3 · Reward-Design

Die Reward-Funktion ist das Wichtigste, was du schreiben wirst. Verbringe mehr Zeit damit als mit der Algorithmus-Wahl.

### Der strukturierte Prozess

Befolge diese Schritte der Reihe nach. Überspringe nichts.

**Schritt 1 — Erfolg in einfachen Worten definieren.**
Schreibe einen Satz: „Der Agent ist erfolgreich, wenn ___." Das wird deine terminale Reward-Bedingung. Sei konkret. „Schießt ein Tor" ist konkret. „Spielt gut" nicht.

**Schritt 2 — Zuerst den terminalen Reward schreiben.**

```gdscript
func _get_reward() -> float:
    # Terminal reward only — shaped components come later
    if _goal_reached():
        return 1.0
    if _episode_timeout():
        return -0.1  # mild penalty for timing out, not failing
    return 0.0
```

Trainiere nur mit diesem Reward. Findet der Agent das Ziel innerhalb einer sinnvollen Schrittzahl (versuche 1–5 Millionen), bist du fertig. Springe zu Schritt 5.

**Schritt 3 — Den Random-Policy-Sanity-Check laufen lassen.**

Lasse vor dem Training 100 Episoden mit zufälliger Policy laufen und logge das Return:

```python
# Python / SB3 sanity check before training
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

env = your_env  # your GodotEnv or wrapped env
check_env(env)

# Random rollout
obs, _ = env.reset()
total_reward = 0.0
for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        obs, _ = env.reset()
print(f"Random policy return: {total_reward:.3f}")
```

Bekommt eine zufällige Policy in 100 Episoden nie einen positiven Reward, wird auch eine trainierte Policy das wahrscheinlich nicht. Füge vor dem Training eine kleine geformte Komponente hinzu.

**Schritt 4 — Geformte Komponenten nur ergänzen, wenn das Training stagniert.**

```gdscript
func _get_reward() -> float:
    var reward := 0.0

    # Terminal reward (primary signal — keep this weight at 1.0)
    if _goal_reached():
        reward += 1.0
        return reward  # early return, don't add shaping on terminal step
    if _episode_timeout():
        reward -= 0.1
        return reward

    # Shaped components (secondary — keep weights small, << 1.0)
    var dist_to_goal: float = global_position.distance_to(_goal_position)
    var dist_delta: float = _prev_dist_to_goal - dist_to_goal  # positive = getting closer
    reward += 0.01 * dist_delta  # potential-based shaping: safe and consistent

    _prev_dist_to_goal = dist_to_goal
    return reward
```

**Schritt 5 — Auf Reward-Hacking prüfen.**
Nach jedem Trainingslauf den Agenten in Godot beobachten. Ein hoher TensorBoard-Score bei bizarr aussehendem Verhalten heißt: dein Reward wurde gehackt. Siehe [unit-reward-engineering](unit-reward-engineering.md) für die volle Theorie und weitere Anti-Hacking-Techniken.

!!! note "Reward-Skalierung ist wichtig"
    Terminale Rewards sollten dominieren. Summieren sich deine geformten Komponenten über eine Episode zu mehr als dein terminaler Reward, optimiert der Agent stattdessen das geformte Signal. Faustregel: `shaped_total_per_episode ≤ 0,5 × terminal_reward`.

Die vollständige Reward-Engineering-Referenz findest du in [unit-reward-engineering.md](unit-reward-engineering.md).

---

## 4 · Einen Algorithmus wählen

Nutze dieses Flussdiagramm, um den Algorithmus zu wählen. Überdenke es nicht — die Algorithmuswahl zählt viel weniger als das Reward-Design.

```
Are your actions continuous (Box action space)?
├── Yes → SAC (preferred for robotics / smooth control)
│          or PPO (simpler, works well for game-like tasks)
│          → see unit-sac.md
└── No  → PPO with Discrete or MultiDiscrete action space
           → see unit-ppo-deep.md

Is your reward sparse (agent rarely sees a positive reward)?
├── Yes → Add curiosity (RND or ICM) on top of your chosen algorithm
│          → see unit-curiosity.md
│          or add HER if your task is goal-conditioned
│          → see unit-her.md
└── No  → Standard PPO or SAC without modification

Does the agent need memory (partial observability, sequence-dependent decisions)?
├── Yes → RecurrentPPO (LSTM-augmented PPO)
│          → see unit-08.md
└── No  → Standard PPO or SAC

Is there more than one agent?
├── Competitive (zero-sum) → PPO + self-play
│                             → see unit-self-play.md
├── Cooperative → MAPPO with shared reward
│                  → see unit-self-play.md
└── Single-agent → any of the above
```

### Schnellreferenz

| Szenario | Algorithmus | Config-Key |
|---|---|---|
| Diskretes Spiel, dichter Reward | PPO | `algorithm: ppo` |
| Kontinuierliche Steuerung, dichter Reward | SAC | `algorithm: sac` |
| Beliebige Aufgabe, spärlicher Reward | PPO + Neugier | `use_curiosity: true` |
| Goal-conditioned-Robotik | SAC + HER | `use_her: true` |
| Partielle Obs / Gedächtnis nötig | RecurrentPPO | `algorithm: rppo` |
| Kompetitiver Multi-Agent | PPO + Self-Play | `self_play: true` |

!!! tip "Im Zweifel mit PPO starten"
    PPO ist der robusteste Algorithmus in diesem Kurs. Er bewältigt diskrete und kontinuierliche Aktionen, ist tolerant gegenüber Hyperparameter-Wahl und trainiert stabil bei nahezu jedem Reward-Design. Wechsle nur dann zu SAC, wenn du speziell Sample-Effizienz auf einer Continuous-Control-Aufgabe brauchst.

---

## 5 · Trainings-Pipeline

### Headless-Export

Trainiere immer mit einem Headless-Godot-Build — der Renderer kostet ~30 % deiner Wall-Clock-Zeit beim Training. Aus dem Godot-Editor exportieren:

`Projekt → Exportieren → Linux/macOS (Headless) → Projekt exportieren`

Nenne das Binary `game.x86_64` (Linux) oder `game` (macOS). Platziere es im Projekt-Root.

### Launch-Command-Template

Kopiere und passe diesen Befehl an. Stelle `n_parallel` auf die Anzahl physischer CPU-Kerne ein (nicht Hyper-Threads):

```bash
# Minimum viable training run (3 seeds, 4 parallel envs each)
for SEED in 1 2 3; do
  gdrl train \
    --config-name=ppo \
    env.path=./game.x86_64 \
    env.n_parallel=4 \
    train.timesteps=3_000_000 \
    train.seed=$SEED \
    train.checkpoint_freq=100_000 \
    hydra.run.dir=runs/seed_${SEED} \
    &
done
wait
echo "All seeds finished"
```

Für SAC (Single-Process, keine vektorisierten Envs):

```bash
for SEED in 1 2 3; do
  gdrl train \
    --config-name=sac \
    env.path=./game.x86_64 \
    env.n_parallel=1 \
    train.timesteps=1_000_000 \
    train.seed=$SEED \
    hydra.run.dir=runs/sac_seed_${SEED} \
    &
done
wait
```

### TensorBoard-Setup

```bash
# In a separate terminal — keep this running throughout training
tensorboard --logdir=runs/ --port=6006
# Open http://localhost:6006 in your browser
```

Schlüssel-Metriken zum Beobachten:

| Metrik | Was sie dir sagt |
|---|---|
| `rollout/ep_rew_mean` | Lernt der Agent überhaupt? Sollte ansteigen |
| `rollout/ep_len_mean` | Episodenlänge — stabil, sobald der Agent die Aufgabe löst |
| `train/value_loss` | Critic-Genauigkeit — sollte sinken, dann plateauen |
| `train/entropy_loss` | Erkundung — fällt sie auf null, steckt der Agent fest |
| `train/approx_kl` | PPO-Update-Größe — durchgehend > 0,05 → lr senken |

### Checkpoint-Frequenz

Setze `checkpoint_freq` für Aufgaben unter 3 M Schritten auf alle 100 000 Schritte und für längere Läufe auf alle 250 000 Schritte. Checkpoints lassen dich eine gute Policy zurückgewinnen, auch wenn das Training später divergiert.

!!! warning "Immer mindestens 3 Seeds laufen lassen"
    RL-Training hat hohe Varianz. Ein einzelner Seed, der gut performt, kann ein Ausreißer sein. Drei Seeds geben dir aussagekräftige Mittelwerte und Standardabweichungen. Weniger und du kannst nicht entscheiden, ob dein Design funktioniert oder du nur Glück hattest.

---

## 6 · Evaluation

Das Training endet. Jetzt beweise, dass dein Agent tatsächlich etwas gelernt hat.

### Baselines zuerst

Vergleiche immer gegen eine zufällige Policy. Die Random-Baseline ist dein Boden. Schlägt dein trainierter Agent sie nur knapp, stimmt etwas nicht.

```python
# Compute random baseline before any training
import numpy as np
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="./game.x86_64", show_window=False)
random_returns = []
for episode in range(50):
    obs, _ = env.reset()
    ep_ret = 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_ret += reward
        done = terminated or truncated
    random_returns.append(ep_ret)
env.close()

print(f"Random baseline — mean: {np.mean(random_returns):.3f}  std: {np.std(random_returns):.3f}")
```

### Multi-Seed-Zusammenfassung

Nach Abschluss des Trainings über alle Seeds, berechne:

```python
import numpy as np

# Load per-seed final episode returns (e.g. from TensorBoard CSV export)
seed_returns = [
    [final_ep_rets_from_seed_1],
    [final_ep_rets_from_seed_2],
    [final_ep_rets_from_seed_3],
]
all_returns = np.concatenate(seed_returns)
print(f"Trained agent — mean: {np.mean(all_returns):.3f}  std: {np.std(all_returns):.3f}")

# Interquartile mean (IQM) — more robust than plain mean for RL
q25, q75 = np.percentile(all_returns, [25, 75])
iqm = np.mean(all_returns[(all_returns >= q25) & (all_returns <= q75)])
print(f"IQM: {iqm:.3f}")
```

Das **Interquartile Mean** (IQM) schneidet die oberen und unteren 25 % der Episoden vor der Mittelung ab. Es ist robuster als der einfache Mittelwert, wenn Episoden-Returns Ausreißer haben, was in RL üblich ist. Verwende es in deinem Report.

### Visualisierungs-Checkpoint-Video

```bash
# Run a trained checkpoint in render mode and record with OBS or ffmpeg
gdrl enjoy \
  --config-name=ppo \
  enjoy.checkpoint=runs/seed_1/best_model.zip \
  env.path=./game.x86_64 \
  env.show_window=true \
  enjoy.n_episodes=5

# Alternatively, pipe Godot window to ffmpeg (Linux)
ffmpeg -video_size 1280x720 -framerate 30 -f x11grab -i :0.0 \
  -c:v libx264 -preset fast capstone_demo.mp4
```

### Wie „gut" aussieht

„Gut" ist immer relativ zu deiner Aufgabe. Definiere es, bevor du mit dem Training beginnst:

- Ist Erfolg binär (Ziel erreicht / nicht), sollte ein „guter" Agent das Ziel in > 80 % der Eval-Episoden erreichen.
- Ist Erfolg ein Score, sollte ein „guter" Agent die Random-Baseline deutlich übertreffen (mindestens 2× Mittelwert-Return).
- Ist Erfolg qualitativ (glatte Fortbewegung, kohärente Strategie), nimm ein Video auf und schau es kritisch an.

---

## 7 · Shipping

Wenn du mit deinem Agenten zufrieden bist, exportiere das ONNX-Modell und führe es ohne Python in Godot aus.

### ONNX-Export

```python
# Export from SB3 checkpoint to ONNX
from godot_rl.wrappers.onnx.stable_baselines_export import export_ppo_model_as_onnx

export_ppo_model_as_onnx(
    model_path="runs/seed_1/best_model.zip",
    onnx_model_path="exported/agent.onnx",
)
print("ONNX export complete → exported/agent.onnx")
```

Für SAC:

```python
from godot_rl.wrappers.onnx.stable_baselines_export import export_sac_model_as_onnx

export_sac_model_as_onnx(
    model_path="runs/sac_seed_1/best_model.zip",
    onnx_model_path="exported/agent.onnx",
)
```

### Godot-Inferenz-Modus

Setze in deinem `AIController`-Node `inference_mode = true` im Editor-Inspector und zeige `onnx_model_path` auf deine exportierte Datei:

```gdscript
# AIController.gd — inference mode snippet
extends AIController3D  # or AIController2D

func _ready() -> void:
    # These are set in the Inspector — shown here for reference
    # inference_mode = true
    # onnx_model_path = "res://exported/agent.onnx"
    pass

func get_obs() -> Array:
    # Must match exactly what you returned during training
    var obs: Array = []
    obs.append(clamp(global_position.x / ARENA_HALF_WIDTH, -1.0, 1.0))
    obs.append(clamp(global_position.y / ARENA_HALF_HEIGHT, -1.0, 1.0))
    obs.append(clamp(linear_velocity.x / MAX_SPEED, -1.0, 1.0))
    obs.append(clamp(linear_velocity.y / MAX_SPEED, -1.0, 1.0))
    return obs
```

!!! warning "Reihenfolge der Beobachtungen muss identisch sein"
    Die ONNX-Runtime füttert deinen Obs-Vektor in der Reihenfolge ins Netz, in der du ihn aus `get_obs()` zurückgibst. Weicht die Reihenfolge vom Training ab, produziert die Policy Müll. Halte `get_obs()` unter Versionskontrolle und ordne sie zwischen Training und Export nie um.

Für HTML5-Export und volle Shipping-Details siehe [unit-10.md](unit-10.md).

---

## 8 · Projektideen

Wähle eine, die dich interessiert. Die Schwierigkeits-Ratings nehmen an, dass du alle Kurs-Units abgeschlossen hast.

| Projekt | Schwierigkeit | Umgebungs-Typ | Aktionsraum | Relevanteste Units |
|---|---|---|---|---|
| **Fußball-KI** — 1-gegen-1-Agent schießt Tore gegen einen Gegner | Einsteiger | 2D-Physik | Kontinuierlich | unit-04, unit-self-play |
| **Plattformer-NPC** — Agent lernt, durch ein handgebautes Level zu navigieren | Einsteiger | 2D-Physik | Diskret | unit-01, unit-ppo-deep |
| **Tower Defense** — Agent platziert Türme, um Wellen zu überleben | Mittel | 2D-Grid | Diskret (MultiDiscrete) | unit-03, unit-reward-engineering |
| **Koch-Spiel** — Agent kombiniert Zutaten vor Ablauf des Timers | Mittel | 2D-Grid | Diskret | unit-08, unit-curiosity |
| **Verkehrs-Simulation** — Agenten an einer Kreuzung minimieren Wartezeit | Mittel | 2D-Physik | Diskret | unit-self-play, unit-reward-engineering |
| **Drohnen-Rennen** — Quadrotor-Agent absolviert einen Checkpoint-Parcours | Mittel | 3D-Physik | Continuous Box | unit-sac, unit-robotics |
| **Lagerhaus-Roboter** — Arm platziert Kisten der Reihe nach auf Paletten | Mittel | 3D-Physik | Continuous Box | unit-robotics, unit-her |
| **Verstecken-Spielen** — Versteck-Agenten vs. Sucher mit emergenter Strategie | Fortgeschritten | 3D-Physik | Kontinuierlich | unit-self-play, unit-curiosity |
| **Schach-Variante Self-Play** — Agent lernt modifiziertes Schach | Fortgeschritten | 2D-Grid | Diskret (groß) | unit-self-play, unit-03 |
| **Musik-Rhythmus-Agent** — Agent trifft Beats mit präzisem Timing | Fortgeschritten | 2D | Kontinuierlich | unit-sac, unit-reward-engineering |

!!! tip "Empfehlung für Einsteiger"
    Starte mit **Plattformer-NPC** oder **Fußball-KI**. Beide haben einfache, prüfbare Erfolgsbedingungen, kurze Episoden und dichte genug Rewards, sodass eine zufällige Policy gelegentlich Erfolg hat — genau die Bedingungen, unter denen RL zuverlässig funktioniert.

!!! note "Multi-Agent-Projekte"
    Verstecken-Spielen, Verkehrs-Simulation und Schach-Varianten-Self-Play brauchen alle Multi-Agent-Infrastruktur. Schließe [unit-self-play.md](unit-self-play.md) ab, bevor du eines davon versuchst. Unterschätze nicht die zusätzliche Komplexität von Multi-Agent-Debugging.

---

## 9 · Häufige Fehlermodi

Das sind Probleme, die speziell bei eigenen Projekten auftauchen — sie kommen in den geführten Beispielen nicht vor, weil der Starter-Code sie bereits behandelt. Sie stehen nicht im Standard-Debugging-Guide ([unit-debugging.md](unit-debugging.md)), weil sie sich auf Design-Entscheidungen beziehen, nicht auf Trainings-Pathologien.

### 1 · Falscher Aktionsraum-Typ

**Symptom:** Agent macht in jedem Schritt scheinbar dasselbe, oder Aktionen haben keinen sichtbaren Effekt.

**Diagnose:** Du hast einen `Box` (kontinuierlich) Aktionsraum definiert, aber dein Spiel erwartet einen Integer-Index, oder umgekehrt. Prüfe die Aktionsraum-Deklaration in deinem `AIController` und den `set_action()`-Handler.

```gdscript
# WRONG — Box action when you want discrete
func get_action_space() -> Dictionary:
    return {"move": {"size": 4, "action_type": "continuous"}}

# CORRECT — Discrete action for 4 directions
func get_action_space() -> Dictionary:
    return {"move": {"size": 4, "action_type": "discrete"}}
```

### 2 · Beobachtungen werden am Episodenende nicht zurückgesetzt

**Symptom:** Agent performt in der ersten Episode gut, degradiert dann oder verhält sich in folgenden Episoden erratisch.

**Diagnose:** Mutabler State, der in `get_obs()` verwendet wird (z. B. `_prev_dist_to_goal`, gestapelte Frame-Buffer), wird beim Episodenende nicht zurückgesetzt.

```gdscript
func _reset() -> void:
    # ALWAYS reset all mutable observation state here
    _prev_dist_to_goal = global_position.distance_to(_goal_position)
    _prev_velocity = Vector2.ZERO
    _step_count = 0
    # ... any other stateful obs components
```

### 3 · Reward-Skalen-Mismatch zwischen geformten und terminalen Komponenten

**Symptom:** TensorBoard zeigt hohen `ep_rew_mean`, aber der Agent erreicht das Ziel nie — er erntet geformten Reward unendlich.

**Diagnose:** Dein geformter Reward pro Episode übersteigt deinen terminalen Reward. Der Agent vermeidet rational das Beenden der Episode.

```gdscript
# BAD — shaped reward dwarfs terminal reward over a 500-step episode
# Step reward: +0.01 × 500 steps = +5.0 total shaped
# Terminal reward: +1.0
# Agent prefers staying alive and collecting +5.0 over ending the episode

# GOOD — keep shaped total << terminal
# Step reward: +0.001 × 500 steps = +0.5 total shaped
# Terminal reward: +1.0
# Agent prefers ending the episode (efficiency) over harvesting shaped reward
```

### 4 · Vergessen, kontinuierliche Beobachtungen zu normalisieren

**Symptom:** Value-Loss oszilliert wild, Policy-Loss divergiert, oder der Agent lernt nach Millionen Schritten nichts trotz vernünftigem Reward.

**Diagnose:** Eine oder mehrere Obs-Komponenten haben eine sehr andere Skala als der Rest. Die Gewichte des neuronalen Netzes können sich nicht stabilisieren, wenn Eingaben im selben Vektor von `0,001` bis `10 000` reichen.

**Fix:** Audite jede von `get_obs()` zurückgegebene Komponente. Kann ein Wert irgendwo `1,0` übersteigen oder unter `-1,0` fallen, normalisiere ihn wie in Abschnitt 2 gezeigt.

### 5 · Nur einen Seed laufen lassen und Sieg erklären

**Symptom:** Dein Agent „funktioniert großartig", aber ein Kommilitone kann deine Ergebnisse nicht reproduzieren, oder dein Agent performt an einem Eval-Tag gut und eine Woche später auf demselben Checkpoint schlecht.

**Diagnose:** Du hast einen Seed gelaufen. RL hat hohe Varianz. Ein einzelner glücklicher Seed kann scheinbar eine Aufgabe lösen, die dein Design eigentlich nicht zuverlässig löst.

**Fix:** Lasse immer mindestens drei Seeds laufen (Abschnitt 5). Berichte Mittelwert ± Std oder IQM. Variiert die Performance enorm über die Seeds, ist dein Reward-Design oder Beobachtungs-Design zu sensibel — untersuche das, bevor du shippst.

### 6 · Ein 3D-Problem als 2D-Problem behandeln (oder umgekehrt)

**Symptom:** Agent löst die Aufgabe in 2D-Testszenarien, scheitert aber in der vollen 3D-Umgebung. Oder: der `AIController` des Agenten erbt von `AIController2D` in einer 3D-Szene (oder umgekehrt), was Physik-Queries falsche Werte zurückgeben lässt.

**Diagnose:** Prüfe, von welcher Basisklasse dein `AIController` erbt. Prüfe, dass Raycasts, Kollisions-Shapes und Positions-Queries das richtige Koordinatensystem nutzen.

```gdscript
# For a 2D scene
extends AIController2D

# For a 3D scene
extends AIController3D

# Do NOT mix them — the parent class determines which physics server is queried
```

---

## 10 · Stretch Goals

Du hast einen funktionierenden Agenten geshippt. Hier sind drei Wege, weiterzugehen.

### Auf ein Leaderboard oder itch.io einreichen

Exportiere dein Spiel als HTML5-Build ([unit-10.md](unit-10.md)) und veröffentliche es auf [itch.io](https://itch.io). Schreibe eine kurze Beschreibung, was der Agent gelernt hat und wie du ihn trainiert hast. Teile den Link im Kurs-Discord.

Passt dein Projekt zu einem existierenden Benchmark (z. B. eine Fußball-Variante), erwäge eine Einreichung beim Godot-RL-Agents-Community-Leaderboard.

### Einen 1-seitigen Technical Report schreiben

Struktur:
1. **Aufgabe** (2 Sätze) — was ist die Umgebung und Erfolgsbedingung?
2. **MDP-Design** (3–4 Sätze) — Beobachtungsraum, Aktionsraum, Reward-Funktion.
3. **Algorithmus** (1 Satz) — welcher Algorithmus und warum?
4. **Ergebnisse** (3–4 Sätze) — Multi-Seed-Mittelwert ± Std, IQM, Vergleich zur Random-Baseline.
5. **Was nicht funktioniert hat** (2–3 Sätze) — mindestens ein gescheiterter Ansatz. Das ist der wertvollste Abschnitt.
6. **Zukünftige Arbeit** (1–2 Sätze) — was würdest du als Nächstes versuchen?

Über Misserfolge zu schreiben ist wertvoller als Erfolge zu melden. Misserfolge enthalten die echten Lehren.

### Ein 2-minütiges Video-Demo aufnehmen

Zeige:
1. Die zufällige Policy (erste 30 Sekunden) — damit Zuschauer einschätzen können, wie schwer die Aufgabe ist.
2. Einen Mid-Training-Checkpoint (30 Sekunden) — zeige den Agenten beim Lernen.
3. Die finale trainierte Policy (60 Sekunden) — bestes Verhalten, das du beobachtet hast.

Kommentiere jeden Abschnitt. „Der Agent hat gelernt, Wänden auszuweichen, weiß aber noch nicht, wie er Tore schießt" ist informativer als Schweigen.

---

!!! tip "Du bist jetzt ein RL-Practitioner"
    Jedes Projekt, das du von hier an baust, wird schneller sein als das letzte. Das Schwere — das MDP entwerfen, Reward-Hacking debuggen, TensorBoard interpretieren, Multi-Seed-Evaluationen fahren — ist jetzt vertraut. Algorithmen und Frameworks werden sich ändern; dieser Prozess nicht.

---

!!! info "Selbstcheck, bevor du das Projekt startest"
    Kannst du diese in eigenen Worten zu *deiner* Capstone-Idee beantworten?

    1. Formuliere deine Aufgabe in einem Satz, dann schreibe ihr MDP: Beobachtungsraum, Aktionsraum, Reward, terminale Bedingung.
    2. Begründe deine Algorithmuswahl (PPO / DQN / SAC / RecurrentPPO), indem du die *spezifische* Eigenschaft deiner Aufgabe nennst, die ihn passend macht.
    3. Was ist dein **Minimum Viable Result** — die kleinste Version davon, die du in 4–8 Stunden Training shippen kannst?
    4. Was ist dein Evaluations-Protokoll — wie viele Seeds, wie viele Eval-Episoden, deterministische oder stochastische Policy?
    5. Liste die drei wahrscheinlichsten Reward-Hacking-Fehlermodi auf, die dein Design ermöglichen könnte, und jeweils eine Gegenmaßnahme.

    Wenn du alle fünf beantworten kannst — leg los. Wenn nicht, liegt die Antwort in den Abschnitten 1–4 oben.

??? success "Antworten zum Selbstcheck"
    1. Durchgerechnetes Beispiel (Platformer NPC) — „Der Agent ist erfolgreich, wenn er den Level-Ausgang erreicht." Sein **MDP**: Beobachtungen = normalisierter Relativvektor zum Ausgang plus normalisierte Geschwindigkeit (Abschnitt 2); Aktionen = `Discrete(4)`-Bewegung; Reward = `+1.0` terminal beim Erreichen des Ausgangs, `-0.1` bei Timeout, Shaping nur, falls das Training stagniert (Abschnitt 3); terminale Bedingung = Ausgang erreicht oder Schrittlimit. Egal welche Aufgabe: Deine Antwort muss alle vier Teile explizit benennen — ist einer unscharf, geh zurück zu den Abschnitten 2–3, bevor du eine Zeile GDScript schreibst.
    2. Eine echte Begründung nennt eine konkrete **Aufgaben-Eigenschaft** aus dem Flowchart in Abschnitt 4: z. B. „diskrete Aktionen, dichter Reward → PPO", „kontinuierliche Box-Aktionen, bei denen Sample-Effizienz zählt → SAC", „partielle Beobachtbarkeit → RecurrentPPO". „PPO, weil ich es kenne" ist ebenfalls akzeptabel — die Algorithmuswahl zählt weit weniger als das Reward-Design, und PPO ist der robuste Standard des Kurses.
    3. Das **Minimum Viable Result** ist die kleinste Version mit einer Umgebung, einem Agenten und einem Erfolgskriterium, deren trainierte Policy die Random-Baseline klar schlägt — ein einzelnes kleines Level, nicht das ganze Spiel. Die Regel aus Abschnitt 1 gilt: Scope Creep ist der Capstone-Killer Nr. 1, und Komplexität kannst du nach der ersten funktionierenden Policy jederzeit ergänzen.
    4. Das Standard-Protokoll des Kurses: mindestens **3 Seeds** (Abschnitt 5), berichtet als Mittelwert ± Std plus **IQM** (Abschnitt 6), mit den Returns pro Seed aus den letzten Trainingsepisoden (TensorBoard-Export aus Abschnitt 6) und der **50-Episoden-Random-Policy-Baseline** als Maßstab. Deterministisch oder stochastisch evaluieren ist deine Entscheidung — nenne sie nur im Report und halte sie über alle Seeds konsistent.
    5. Die drei klassischen Muster aus den Abschnitten 3 und 9: (a) **Shaped-Reward-Harvesting** — der Agent erntet Shaping, statt fertig zu werden → halte die Shaping-Summe pro Episode ≤ 0,5 × terminaler Reward; (b) Hin- und Herpendeln zum Ziel, um Distanz-Delta-Reward erneut einzusammeln → nutze potenzialbasiertes Shaping und setze `_prev_dist_to_goal` in `_reset()` zurück; (c) Vermeiden des Episodenendes, weil Überleben sich mehr lohnt → lass den terminalen Reward dominieren und bestrafe Timeouts. Eine Gegenmaßnahme deckt alle drei ab: Schau dir den Agenten nach jedem Run in Godot an — eine hohe Kurve bei bizarrem Verhalten heißt, der Reward wurde ausgetrickst.

---

[← Ship Your Brain](unit-10.md) · [Kursstartseite](index.md)
