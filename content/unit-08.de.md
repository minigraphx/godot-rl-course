# Unit 8 — Memory & POMDPs

Trainiere **FPS / RobotFPS** — Umgebungen, in denen der Agent nicht alles auf einmal sehen kann. Lerne, warum Gedächtnis wichtig ist, wie LSTM-Policy-Netze funktionieren und wie du **RecurrentPPO** aus `sb3-contrib` benutzt.

[← Unit 7: Multi-Agent](unit-07.md) · [Kursstartseite](index.md)

!!! note "Voraussetzungen"
    - **[Unit 4](unit-04.md)** — PPO end-to-end, sicheres Lesen von `train/*`-Kurven
    - **[Unit 6](unit-06.md)** — kontinuierliches Aktionsdesign (FPS verwendet kontinuierliches Zielen + diskretes Feuer)
    - **[Visuelle Beobachtungen](unit-visual-observations.md)** (optional) — nur wenn du aus Pixeln trainierst
    - High-Level-Vertrautheit mit RNN- / LSTM-Ideen (wir erklären sie in §2 erneut)

!!! info "Zeit"
    Lesen: ~35 min · Training: ~30 min GPU / ~2 Std CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    Godot (beobachte, wie der Agent zögert, wenn ein Ziel aus dem Blickfeld gerät — ein Zeichen, dass Gedächtnis funktioniert) · TensorBoard (`ep_rew_mean` mit vs ohne RecurrentPPO) · Beobachtungsdesign: was der Agent sehen kann und was nicht

---

## 1 · Teilweise beobachtbare Umgebungen (POMDPs)

Alle vorherigen Units verwendeten **vollständig beobachtbare** Umgebungen — die Beobachtung des Agenten enthielt alles, was er zur Aktionswahl brauchte. Reale Umgebungen sind selten so sauber:

- Ein FPS-Charakter kann nicht durch Wände sehen
- Ein Roboter in einem Labyrinth weiß nicht, wo er gestartet ist
- Ein Lander mit einem verrauschten Sensor kann sich seiner exakten Höhe nicht sicher sein

Das sind **Partially Observable MDPs (POMDPs)**. Die Markov-Eigenschaft (Beobachtung = vollständiger Zustand) gilt nicht mehr. Der Agent muss sich an vergangene Beobachtungen **erinnern**, um den verborgenen Zustand zu inferieren.

**Standard-PPO bricht bei POMDPs** — es behandelt jeden Schritt unabhängig. Bei nur der aktuellen (partiellen) Beobachtung ist die optimale Aktion mehrdeutig.

**RecurrentPPO** fügt dem Policy-Netz eine **LSTM**-Schicht (Long Short-Term Memory) hinzu. Das LSTM trägt einen verborgenen Zustand über Zeitschritte hinweg und gibt dem Agenten effektiv Gedächtnis.

---

## 2 · Wie LSTM-Gedächtnis in RecurrentPPO funktioniert

```
Observation_t  →  [Shared MLP]  →  [LSTM cell]  →  [Policy head]  →  Action_t
                                        ↕
                                   hidden state h_t  →  h_{t+1}
```

Der verborgene Zustand des LSTM ist:
- **Zurückgesetzt an Episodengrenzen** — Gedächtnis leckt nicht zwischen Episoden
- **Geteilt über parallele Envs** — jede Env-Instanz hat ihren eigenen verborgenen Zustand
- **Feste Größe** — gesteuert durch `lstm_hidden_size` (Standard: 256)

Der Agent lernt, relevante Informationen in den verborgenen Zustand zu schreiben (z. B. „Ziel wurde zuletzt links gesehen") und sie bei Bedarf zurückzulesen.

---

## 3 · sb3-contrib installieren

RecurrentPPO lebt in `sb3-contrib`, nicht im Basis-`stable-baselines3`:

```bash
conda activate godot_env
pip install sb3-contrib
```

---

## 4 · FPS oder RobotFPS öffnen

1. Aus [godot_rl_agents_examples](https://github.com/edbeeching/godot_rl_agents_examples), öffne `examples/FPS` oder `examples/RobotFPS`
2. Aktiviere das Godot RL Agents Plugin
3. Lies `ai_controller.gd`:

**Was der Agent sehen kann** (`get_obs()`):
```gdscript
func get_obs() -> Dictionary:
    return {"obs": [
        # RayCast readings — local perception only
        ray_forward.get_collision_distance() / max_dist,
        ray_left.get_collision_distance()    / max_dist,
        ray_right.get_collision_distance()   / max_dist,
        # No global position — agent can't see where it is on the map
        linear_velocity.x / max_speed,
        linear_velocity.z / max_speed,
        # Target visible? (0 or 1)
        float(target_in_sight),
    ]}
```

Beachte, was **fehlt**: globale Position, Kartenlayout, Zielposition, wenn außer Sicht. Der Agent muss diese aus dem Gedächtnis inferieren.

---

## 5 · Partielle Beobachtungen entwerfen (eigene bauen)

Wenn du einer existierenden Umgebung eine Gedächtnisanforderung hinzufügen willst:

**Globale Information entfernen:**
```gdscript
# Before (fully observable):
(global_position.x - target.global_position.x) / 100.0

# After (partial — agent must remember where it last saw the target):
float(target_in_sight) * (global_position.x - target.global_position.x) / 100.0
# When target is not in sight, this returns 0.0 — the agent loses the signal
```

**Rauschen hinzufügen:**
```gdscript
# Noisy altitude reading
(global_position.y + randf_range(-0.5, 0.5)) / max_height
```

---

## 6 · Mit RecurrentPPO trainieren

```python
from sb3_contrib import RecurrentPPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(
    env_path="./RobotFPS.x86_64",
    n_parallel=8,
    speedup=20,
)

model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/",
    n_steps=512,
    batch_size=256,
    lstm_hidden_size=256,
    n_lstm_layers=1,
)
model.learn(total_timesteps=3_000_000)
model.save("robotfps_recurrent")
env.close()
```

```bash
conda activate godot_env
tensorboard --logdir=logs &
python train_robotfps.py
```

---

## 7 · PPO vs RecurrentPPO Vergleich

Führe beide auf derselben Umgebung aus und vergleiche in TensorBoard:

```python
# Standard PPO baseline
from stable_baselines3 import PPO
model_ppo = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/")
model_ppo.learn(total_timesteps=3_000_000)
```

**Erwartetes Ergebnis:** PPO plateauiert oder scheitert bei Aufgaben, die Gedächtnis erfordern. RecurrentPPO verbessert sich weiter. Die Lücke ist gewöhnlich nach 1–2M Schritten sichtbar.

| Metrik | PPO | RecurrentPPO |
|--------|-----|-------------|
| `ep_rew_mean`-Peak | Niedriger | Höher |
| Konvergenzgeschwindigkeit | Schneller früh | Langsamer früh, höhere Decke |
| RAM-Nutzung | Niedrig | Höher (LSTM-Zustände pro Env) |

---

## 8 · Viz-Checkpoint

Beobachte den Agenten mit `--viz` nach dem Training:

- **Sucht** der Agent, wenn das Ziel aus dem Blickfeld gerät, oder friert er ein?
- **Erinnert** er sich an die Richtung, in der er das Ziel zuletzt gesehen hat?
- Behandelt er **Sackgassen** (umdrehen) oder bleibt er stecken?

Ein funktionierender LSTM-Agent wird kurz zögern, wenn er die Sicht verliert, dann sich in die letzte bekannte Richtung bewegen — klares, menschlich lesbares Gedächtnisverhalten.

---

## 9 · Stretch Goals

- **Frames stapeln** statt LSTM: verwende einen `VecFrameStack`-Wrapper, um dem Agenten die letzten N Beobachtungen als Eingabe für Standard-PPO zu geben
- **Längeres Gedächtnis** — erhöhe `lstm_hidden_size` auf 512; miss, ob es bei einer labyrinthartigen Aufgabe hilft
- **Eine Gedächtnisaufgabe bauen** — entwirf eine Umgebung, in der der Agent sich erinnern muss, welche von zwei Türen er in der letzten Episode geöffnet hat

---

## Was kommt als Nächstes

**Self-Play:** Trainiere Agenten, indem sie gegen Kopien von sich selbst antreten — AirHockey, eingefrorene Checkpoints, League-basiertes Training und ELO-Tracking.

!!! info "Selbstcheck, bevor du weitermachst"
    Kannst du diese in eigenen Worten beantworten?

    1. Welche Eigenschaft bricht ein POMDP, und was bedeutet das konkret für das Netz?
    2. Was trägt der **verborgene Zustand** des LSTM über Zeitschritte hinweg, was eine Feed-Forward-Policy nicht kann?
    3. Wann wäre Frame Stacking ausreichend, und wann brauchst du wirklich RecurrentPPO?
    4. Warum verwendet RecurrentPPO truncated BPTT, anstatt die volle Episode auszurollen?
    5. Nenne eine Godot-Umgebung in diesem Kurs, in der du *kein* Gedächtnis verwenden würdest, und eine, in der du es würdest.

    Wenn du alle fünf beantworten kannst — bist du bereit.

[→ Self-Play](unit-self-play.md)
