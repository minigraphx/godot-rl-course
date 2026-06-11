# Unit 4 — JumperHard & PPO-Benchmarking

Trainiere das **JumperHard**-Beispiel — einen 3D-Springer-Roboter, der als Standard-PPO-Benchmark im godot-rl-agents-Repo dient. Schwerpunkte: PPO-Hyperparameter (hyperparameters) lesen, anpassen und erkennen, wann das Training die Aufgabe wirklich gelöst hat.

[← Unit 3: CrossTheRoad & DQN](unit-03.md) · [Kursübersicht](index.md)

!!! note "Voraussetzungen"
    - **[Unit 2](unit-02.md)** — ein vollständig laufender SB3-PPO-Durchlauf (Training + ONNX-Export)
    - **[Unit 3](unit-03.md)** — DQN auf CrossTheRoad trainiert; Unterscheidung on-policy vs. off-policy
    - Sicherheit im Lesen von TensorBoard-Skalaren (`ep_rew_mean`, `approx_kl`, `entropy_loss`)
    - **[Actor-Critic-Einheit](unit-actor-critic.md)** — **mach das zuerst, wenn du durchgehend liest:** PPO ist ein Actor-Critic-Verfahren, und Abschnitt 0 setzt die Actor-/Critic-Aufteilung voraus.

!!! info "Zeit"
    Lesen: ~40 min · Training: ~45 min GPU / ~3 Std CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    Godot (Visualisierungs-Checkpoint nach dem Training) · TensorBoard (Vergleich mit deiner Unit-2-Lander-Baseline) · `AIController`-Hyperparameteränderungen

---

## 0 · Warum PPO? Die Theorie in 5 Minuten

Bevor du die Konfigurationsdateien von JumperHard anfasst, lohnt es sich zu verstehen, *warum* PPO der Standard-Algorithmus in godot-rl-agents ist — und was ihn von DQN unterscheidet.

**PPO ist ein Actor-Critic-Verfahren**

Während DQN nur Q-Werte lernt (ein Kritiker), lernt PPO *zwei* Dinge gleichzeitig:

| Komponente | Was sie lernt | Ausgabe des neuronalen Netzes |
|-----------|---------------|----------------------|
| **Actor** (Policy π) | Welche Aktion in Zustand s ausgeführt werden soll | Aktionsverteilung (Mittelwert + Standardabweichung für kontinuierlich, Logits für diskret) |
| **Critic** (Wert V) | Wie gut Zustand s ist, unabhängig von der Aktion | Ein einzelner Skalarwert V(s) |

Der Actor nutzt die Schätzungen des Critics, um die Varianz seiner Gradientenaktualisierungen zu reduzieren. Der Critic verbessert sich, indem er seine Vorhersagen mit den tatsächlichen Erträgen vergleicht. Sie trainieren gemeinsam und bauen aufeinander auf.

Die vollständige Actor-Critic-Herleitung findest du in der [Actor-Critic-Einheit](unit-actor-critic.md). Hier konzentrieren wir uns auf die Anwendung.

**Die PPO-Aktualisierungsschleife**

```
1. Run the current policy for n_steps transitions  → collect a rollout
2. Compute advantages using GAE (Generalized Advantage Estimation)
3. Make n_epochs gradient updates on that rollout
4. Throw away the rollout and repeat from step 1
```

Der wesentliche Unterschied zum DQN-Replay-Buffer: PPO verwirft die Rollout-Daten nach n_epochs Durchläufen. DQN speichert Übergänge für immer; PPO ist On-Policy und verwendet Daten nur aus der aktuellen Policy-Version.

**Das geclippte Ziel — das „P" in PPO**

Das Kernprinzip von Proximal Policy Optimization ist ein geclipptes Ersatzziel, das verhindert, dass sich die Policy bei einer einzelnen Aktualisierung zu stark verändert:

```
L_CLIP = E[ min(
    r_t(θ) · A_t,
    clip(r_t(θ), 1-ε, 1+ε) · A_t
)]

where r_t(θ) = π_θ(a|s) / π_θ_old(a|s)   (probability ratio)
      A_t    = advantage estimate
      ε      = clip_range (default 0.2)
```

Wenn die neue Policy sich um mehr als `clip_range` relativ zur alten Policy ändern möchte, wird der Gradient auf null gesetzt — dies erzwingt effektiv eine **Trust Region** ohne die aufwändige Optimierung zweiter Ordnung von TRPO.

!!! tip "Warum Clipping für JumperHard wichtig ist"
    JumperHard hat eine unebene Belohnungslandschaft — kleine Änderungen im Sprungzeitpunkt führen zu großen Änderungen in der Belohnung. Ohne Clipping könnte ein einzelner guter Rollout die Policy so weit verschieben, dass sie alles andere vergisst. Das geclippte Ziel hält Aktualisierungen konservativ und stabil.

Die vollständige PPO-Herleitung mit Beweisen findest du im [PPO Deep Dive](unit-ppo-deep.md).

---

## 1 · Was JumperHard lehrt

JumperHard ist aus zwei Gründen schwerer als der Lander:

1. **3D-Physik** — der Roboter muss beim Springen die Balance halten, was den Zustandsraum vergrößert und die Belohnungslandschaft rauer macht
2. **Spärlich an den Grenzen** — ein Bonus wird nur bei schweren Sprüngen ausgelöst, sodass der Agent dichte geformte Belohnungen *und* gelegentliche spärliche Boni zum Lernen benötigt

Das macht es zu einem guten Benchmark: Wenn deine Hyperparameter JumperHard lösen können, werden sie gut verallgemeinern.

---

## 2 · PPO-Hyperparameter — mit Theorie

Das sind die Stellschrauben, an denen du drehst. Die Standardwerte funktionieren für die meisten Aufgaben; du musst immer nur einen auf einmal ändern. Der theoretische Grund für jeden Parameter ist angegeben, damit du verstehst, *warum* eine Änderung helfen sollte.

| Parameter | Standard | Theoretische Funktion | Wann ändern |
|-----------|---------|-----------------|----------------|
| `--learning_rate` | 0.0003 | Schrittgröße im Gradientenraum — steuert, wie weit der Optimierer pro Aktualisierung geht | Verringern, wenn die Belohnung oszilliert; erhöhen, wenn das Lernen sehr langsam ist |
| `--n_steps` | 64 | Rollout-Länge — steuert den Bias-Varianz-Kompromiss bei der verallgemeinerten Vorteilsschätzung (Generalized Advantage Estimation, GAE) (länger = niedrigerer Bias, höhere Varianz; siehe [PPO Deep Dive](unit-ppo-deep.md)) | Erhöhen (256–2048) für längere Episoden; benötigt mehr Speicher |
| `--batch_size` | 64 | Mini-Stapelgröße (batch size) innerhalb jeder Epoche — muss `n_steps × n_envs` teilen | An `n_steps` anpassen; größer = stabilere Gradienten |
| `--clip_range` | 0.2 | Das ε im geclippten Ziel — steuert die Größe der Trust Region; wie weit sich die Policy pro Aktualisierung bewegen darf (Clipping-Bereich (clip range)) | Verringern (0.1), wenn der Policy-Gradientenverlust explodiert |
| `--ent_coef` | 0.0001 | Entropiekoeffizient (entropy coefficient) — fügt dem Verlust einen Term hinzu, der die Stochastizität der Policy belohnt und Exploration fördert | Erhöhen (0.01), wenn der Agent zu früh auf eine suboptimale Policy konvergiert |
| `--gae_lambda` | 0.95 | λ in der verallgemeinerten Vorteilsschätzung — interpoliert zwischen reinem TD (λ=0, niedrige Varianz, hoher Bias) und reinem Monte Carlo (λ=1, hohe Varianz, niedriger Bias) | Muss selten geändert werden; 0,9–0,99 ist ein sicherer Bereich |
| `--n_epochs` | 10 | Anzahl der Gradientendurchläufe über jeden Rollout — mit Clipping ist die Wiederverwendung sicher; zu viele Epochen schieben die Policy aus der Trust Region heraus | Verringern, wenn `approx_kl` groß wird |
| `--vf_coef` | 0.5 | Verlustkoeffizient der Wertfunktion — skaliert, wie stark der Critic im Vergleich zum Actor trainiert wird | Erhöhen (0,75–1,0), wenn `value_loss` stagniert, während sich die Policy verbessert |

!!! warning "Training gestoppt?"
    Überprüfe der Reihe nach: (1) Vorzeichen und Skalierung der Belohnung, (2) spärliche Belohnungen — gibt es eine geformte Komponente? (3) `approx_kl` > 0,02 → `--clip_range` oder `--learning_rate` reduzieren, (4) `ep_rew_mean` nach 500k Schritten flach → `--n_steps` und `--ent_coef` erhöhen.

**Die Lernrate (learning rate) ist der empfindlichste Parameter.** Im Zweifel halbieren und neu trainieren. Große Lernraten destabilisieren das geclippte Ziel; kleine trainieren einfach langsam. „Zu langsam" ist behebbar; „divergiert" nicht.

**n_steps und batch_size müssen gemeinsam angepasst werden.** `batch_size` muss `n_steps × n_parallel` gleichmäßig teilen. Wenn du `n_steps=512` mit `n_parallel=8` setzt, beträgt die Gesamt-Rollout-Größe 4096 — setze `batch_size` auf 256 oder 512.

!!! info "GAE und der Bias-Varianz-Kompromiss"
    GAE mit λ=0,95 ist eine gewichtete Summe über n-Schritt-Erträge. Höheres λ schaut weiter in die Zukunft (niedrigerer Bias, verrauschtere Schätzungen). Niedrigeres λ verlässt sich mehr auf das Bootstrap der Wertfunktion (glatter, aber durch Wertfunktionsfehler verzerrt). Für JumperHards lange Sprungsequenzen ist λ=0,95 in der Regel korrekt. Sieh den [PPO Deep Dive](unit-ppo-deep.md) für die vollständige GAE-Herleitung.

---

## 3 · JumperHard öffnen

1. Öffne das Projekt aus [examples/JumperHard](https://github.com/edbeeching/godot_rl_agents_examples/tree/main/examples/JumperHard) in Godot .NET
2. Aktiviere das Godot RL Agents-Plugin
3. Lese `ai_controller.gd`:
    - **`get_obs()`** — Körperposition, Geschwindigkeit, Abstände zu nahen Plattformen (RayCast3D-Sensoren)
    - **`get_action_space()`** — kontinuierliche Kräfte auf Gelenke, oder diskret Sprung/Bewegung
    - **Belohnung** — Vorwärtsfortschritt + Sprungbonus + Überlebenszeit
4. Exportiere eine Headless-Binärdatei (Projekt → Exportieren → Linux/Windows/macOS)

---

## 4 · Baseline-Lauf

Starte zuerst mit den Standardwerten. Das gibt dir eine Referenzkurve, die du übertreffen kannst:

```bash
conda activate godot_env
tensorboard --logdir=logs &

gdrl --env_path=./JumperHard.x86_64 \
  --experiment_name=jumper_baseline \
  --timesteps=1_000_000 \
  --n_parallel=8 \
  --speedup=20
```

!!! check "Fertig, wenn"
    JumperHard hat keinen veröffentlichten Benchmark, beurteile den Erfolg also auf zwei Arten: (1) `rollout/ep_rew_mean` ist stetig aus seinem anfänglichen verrauschten Band herausgestiegen und hat sich nach 1M Schritten über 150–200 stabilisiert, und (2) der **Visualisierungs-Checkpoint** (das Evaluierungsskript aus Abschnitt 7 mit `show_window=True`) zeigt, dass der Roboter in den meisten Episoden die Plattformen überwindet. Beurteile nur anhand des 200k–1M-Schritte-Fensters — die ersten 50–100k Schritte werden von den zufälligen Anfangsgewichten dominiert (Abschnitt 6).

---

## 5 · Hyperparameter-Sweep (eine Änderung auf einmal)

Führe drei Experimente durch und variiere dabei jeweils einen Parameter. Verwende unterschiedliche `--experiment_name`-Werte, damit TensorBoard sie überlagert anzeigt.

```bash
# Experiment A — larger rollout buffer
gdrl --env_path=./JumperHard.x86_64 \
  --experiment_name=jumper_nsteps512 \
  --n_steps=512 --batch_size=256 \
  --timesteps=1_000_000 --n_parallel=8 --speedup=20

# Experiment B — more exploration
gdrl --env_path=./JumperHard.x86_64 \
  --experiment_name=jumper_entropy \
  --ent_coef=0.01 \
  --timesteps=1_000_000 --n_parallel=8 --speedup=20

# Experiment C — tighter trust region
gdrl --env_path=./JumperHard.x86_64 \
  --experiment_name=jumper_clip01 \
  --clip_range=0.1 \
  --timesteps=1_000_000 --n_parallel=8 --speedup=20
```

Vergleiche in TensorBoard `rollout/ep_rew_mean` und `train/approx_kl` über die vier Läufe hinweg.

---

## 6 · TensorBoard-Diagnoseleitfaden

TensorBoard richtig zu lesen ist die Fähigkeit, die systematisches Tuning von bloßem Raten unterscheidet. Verwende diese Tabelle als Referenz, während deine Experimente laufen.

| Metrik | Gesunder Bereich | Wenn außerhalb des Bereichs — Maßnahme |
|--------|--------------|--------------------------|
| `train/approx_kl` | 0,01–0,02 | > 0,05: `--learning_rate` oder `--clip_range` senken; Policy verändert sich pro Aktualisierung zu stark |
| `train/entropy_loss` | Langsam, stetig fallend | Plötzlicher Einbruch auf ~0: `--ent_coef` erhöhen; Policy ist vorzeitig deterministisch geworden |
| `train/value_loss` | Stetig fallend | Stagniert, während Belohnung noch niedrig ist: `--n_steps` oder `--vf_coef` erhöhen; Critic bekommt nicht genug Signal |
| `train/policy_gradient_loss` | Kleine Schwankungen nahe 0 | Plötzlicher großer positiver Ausreißer: `--learning_rate` senken, Gradientenclipping hinzufügen (`max_grad_norm=0.5`) |
| `rollout/ep_rew_mean` | Steigend | Bei 1M Schritten flach: `--ent_coef` erhöhen (mehr Exploration), Reward-Shaping in `AIController` prüfen |
| `rollout/ep_len_mean` | Stabil oder steigend | Plötzlicher Einbruch: Agent stirbt früh — Kollisionsbelohnung oder Episoden-Reset-Logik prüfen |

**`approx_kl` zu lesen ist hier die wertvollste Einzelfähigkeit.** Die KL-Divergenz misst, wie stark sich die Policy von alt zu neu in jeder Aktualisierung verändert hat. Gesundes PPO hält sie zwischen 0,01 und 0,02. Über 0,05 bedeutet, dass Aktualisierungen zu groß sind und du die Policy riskierst zu destabilisieren. SB3 protokolliert sie automatisch.

!!! info "Warum Entropie bei JumperHard wichtig ist"
    JumperHard verlangt, dass der Agent Zeitstrategien für Sprünge erkundet — ein Roboter, der sich zu früh auf einen Sprungrhythmus festlegt, wird nie bessere entdecken. Der Entropieverlust verfolgt die Stochastizität der Policy: ein gesunder Entropieverlust sinkt *langsam*. Wenn er in den ersten 200k Schritten zusammenbricht, hat sich der Agent auf eine suboptimale Strategie festgelegt, bevor er genug vom Zustandsraum gesehen hat.

!!! warning "TensorBoard nicht zu früh lesen"
    Die ersten 50–100k Schritte eines PPO-Laufs werden stark von den anfänglichen zufälligen Gewichten beeinflusst. Belohnungskurven in dieser Phase sind verrauscht und nicht aussagekräftig. Beurteile Läufe anhand des 200k–1M-Schritte-Fensters.

---

## 7 · Evaluierungsprotokoll

Sobald du ein trainiertes Modell hast, evaluiere es sorgfältig — lies nicht einfach die Trainingskurve ab.

```python
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import numpy as np

env = StableBaselinesGodotEnv(
    env_path="./JumperHard.x86_64",
    n_parallel=1,
    speedup=1,
    show_window=True,       # viz checkpoint
)
model = PPO.load("logs/sb3/jumper_baseline/best_model")

episode_rewards = []
for _ in range(20):                 # 20 fixed episodes
    obs = env.reset()
    done, total = False, 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)   # no exploration
        obs, reward, done, _ = env.step(action)
        total += reward
    episode_rewards.append(total)

print(f"Mean: {np.mean(episode_rewards):.1f}  Std: {np.std(episode_rewards):.1f}")
env.close()
```

!!! info "Warum deterministic=True?"
    Während des Trainings sampelt die Policy Aktionen stochastisch. Bei der Evaluierung möchtest du die **beste** Aktion bei jedem Schritt — `deterministic=True` wählt die Aktion mit der höchsten Wahrscheinlichkeit, anstatt zu sampeln. Verwende es immer für Benchmarking.

**Visualisierungs-Checkpoint** — führe das Evaluierungsskript mit `show_window=True` aus und beobachte 3–5 Episoden. Bestätige, dass das Verhalten mit dem gemessenen `ep_rew_mean` übereinstimmt.

---

## 8 · Checkpoints speichern & laden

```bash
# Save a checkpoint every 100k steps
gdrl --env_path=./JumperHard.x86_64 \
  --experiment_name=jumper_final \
  --timesteps=2_000_000 \
  --save_model_path=jumper_ppo \
  --save_checkpoint_frequency=100000 \
  --onnx_export_path=jumper_ppo.onnx \
  --n_parallel=8 --speedup=20

# Resume if interrupted
gdrl --env_path=./JumperHard.x86_64 \
  --resume_model_path=jumper_ppo.zip \
  --experiment_name=jumper_final_resume \
  --timesteps=1_000_000 \
  --onnx_export_path=jumper_ppo.onnx
```

---

## 9 · Belohnungsmodellierung für Fortbewegung (beim ersten Lesen optional)

!!! note "Erster Durchgang? Überfliege oder überspringe diesen Abschnitt."

Navigationsaufgaben haben ein einziges Ziel: von A nach B gelangen. Fortbewegung ist anders — der Roboter muss einen **Gang** (ein koordiniertes Muster von Gliedmaßenbewegungen) als emergenten Nebeneffekt der Maximierung des Vorwärtsfortschritts entdecken. Die Belohnungsfunktion misst nicht nur Erfolg; sie formt, welcher Gang entsteht.

### Fortbewegungsspezifische Belohnungskomponenten

**Vorwärtsgeschwindigkeitsbelohnung (das primäre Signal)**

```gdscript
# Reward forward velocity — encourages the robot to move fast
var forward_vel = linear_velocity.dot(global_transform.basis.z)  # z = forward axis
_ai.reward += forward_vel / max_speed * 0.1
```

Das ist das Kernsignal, aber es hat eine gefährliche Fehlermode: Der Roboter lernt, **vorwärts zu fallen**. Ein einziger großer Vorwärtsstoß vor dem Kollaps maximiert die Vorwärtsgeschwindigkeit für einen Schritt — dann endet die Episode. Lösung: kombiniere es mit einem Überlebensbonus.

**Überlebensbonus**

```gdscript
_ai.reward += 0.005   # per physics step — gives robot reason to stay alive
```

Ohne diesen hat der Agent keinen Anreiz, auf den Beinen zu bleiben. Der Überlebensbonus ist es, der „vorwärts fallen" in „weiter vorwärts bewegen" umwandelt.

**Belohnung für aufrechte Haltung**

```gdscript
# Reward staying upright — dot product of up-vector with world up
var uprightness = global_transform.basis.y.dot(Vector3.UP)   # 1.0 = upright, -1.0 = upside down
_ai.reward += max(0.0, uprightness) * 0.01
```

Das hält Fortbewegungspolicys davon ab, niedrig zu bleiben oder stark zu lehnen. Kombiniert mit dem Überleben treibt es den Agenten in Richtung einer aufrechten Haltung, bevor er sich um effizientes Gehen kümmern muss.

**Energieeffizienz (Leistungsaufnahme minimieren)**

```gdscript
# Power = torque × angular_velocity (in Watts)
# Penalizing it encourages smooth, efficient gaits
var total_power = 0.0
for joint in joints:
    total_power += abs(joint_torques[joint] * joint_angular_velocities[joint])
_ai.reward -= total_power * 0.0001
```

Das ist der wichtigste Formungsterm für natürlich aussehende Gänge. Ohne eine Energiestrafe entdecken Fortbewegungspolicys häufig „Galoppier"-Bewegungen, die mechanisch unvernünftig sind und auf echter Hardware nicht funktionieren würden. Die Energiestrafe erzeugt natürlicherweise gangähnliche Bewegungen, weil Gehen metabolisch günstig ist — jedes Bein trägt kurz das Gewicht, überträgt Schwung und schwingt mit minimalem Aufwand nach vorne.

**Glättungsbelohnung (Ruck minimieren)**

```gdscript
# Penalize rapid changes in velocity (mechanical wear, instability)
var jerk = (linear_velocity - _prev_linear_velocity).length() / delta
_ai.reward -= jerk * 0.00001
_prev_linear_velocity = linear_velocity
```

Ruck ist die Ableitung der Beschleunigung. Hoher Ruck bedeutet, dass der Roboter ruckt statt fließt. Das Bestrafen davon tendiert dazu, glattere Trajektorien zu erzeugen, und reduziert das visuelle „Vibrieren", das in frühen Fortbewegungspolicys auftritt.

**Kontaktbelohnung (für Mehrbeiner-Roboter)**

```gdscript
# For robots with legs: reward foot-ground contact to encourage proper gait
for foot in feet:
    if foot.is_colliding():
        _ai.reward += 0.001
```

Für Laufroboter mit diskreten Füßen lenkt das Belohnen von Bodenkontakt die Policy dazu, jeden Fuß periodisch aufzusetzen und anzuheben — die Grundlage eines Gangs — anstatt am Boden entlangzugleiten oder auf einem Bein zu hüpfen.

**Sturz-Terminierung**

```gdscript
# End episode if robot falls — saves training time
if global_position.y < fall_threshold:
    _ai.reward -= 1.0   # penalty for falling
    _ai.done = true
    _ai.needs_reset = true
```

Das Beenden der Episode bei einem Sturz hat zwei Vorteile: Es spart Rechenleistung (keine weiteren Schritte aus einem fehlgeschlagenen Zustand) und sendet ein starkes Signal, dass Fallen kostspielig ist. Die Strafgröße sollte etwa das 10–20-fache des Überlebensbonus pro Schritt betragen.

### Gang-Entstehung

Der entstehende Gang hängt direkt davon ab, welche Belohnungskomponenten aktiv sind. Komponenten einzeln hinzuzufügen und den Godot-Visualisierungs-Checkpoint bei jedem Schritt zu beobachten ist der beste Weg, Intuition aufzubauen:

| Belohnungskomponenten | Typischerweise entstehender Gang |
|---|---|
| Nur Vorwärtsgeschwindigkeit | Fällt vorwärts (ein Schritt, dann tot) |
| Geschwindigkeit + Überleben | Schlurfend, am Boden entlangziehend |
| Geschwindigkeit + Überleben + aufrecht | Aufrecht hüpfend oder wippend |
| + Energieeffizienz | Gangähnliche Bewegung mit weniger Verschwendung |
| + Glättung | Glatte Bewegung mit weniger Ruckeln und Vibrieren |
| Alle oben genannten | Natürlich aussehender Fortbewegungsgang |

Jede Zeile ist die vorherige Zeile plus ein Signal. Das ist kein Zufall — jede Komponente schließt eine spezifische Lücke, die der Agent in der Zeile darüber ausgenutzt hat.

!!! tip "JumperHards Sprungbonus ist eine geformte spärliche Komponente"
    JumperHard verwendet einen Sprungbonus — eine spärliche Komponente, die nur bei „schweren" Sprüngen ausgelöst wird. Das ist ein spärliches Signal, das auf dichten Fortbewegungsbelohnungen aufgeschichtet ist. Die dichten Belohnungen halten das Lernen zwischen Sprüngen stabil; der spärliche Bonus lenkt die Policy auf das eigentliche Aufgabenziel. Füge `print(_ai.reward)` für 10 Schritte hinzu, um zu sehen, welche Komponente in einer bestimmten Trainingsphase dominiert.

### Vergleich mit MuJoCo-Fortbewegungs-Benchmarks

Wenn du jemals eine Fortbewegungsarbeit liest — HalfCheetah, Ant, Hopper, Walker2D — sieht die Belohnungsfunktion fast identisch zu dem oben Beschriebenen aus:

- **`forward_reward`** → Vorwärtsgeschwindigkeit (gleich wie das primäre Signal hier)
- **`healthy_reward`** oder **`survive_reward`** → Überlebensbonus (gleiches Konzept)
- **`ctrl_cost`** → `sum(action²) × weight` — das ist die MuJoCo-Annäherung an Energieeffizienz. Anstatt tatsächliches Drehmoment × Winkelgeschwindigkeit zu messen, bestraft MuJoCo die Größe des Aktionsvektors als Proxy für den Steuerungsaufwand.
- **`contact_cost`** (Ant) → bestraft hohe Kontaktkräfte, ähnlich wie Sicherheitsbeschränkungen

Der wesentliche Unterschied ist, dass MuJoCo die Aktionsgröße als Proxy für Leistung verwendet, während Godots Physik-Engine dir erlaubt, tatsächliche Gelenkdrehmomente zu berechnen. Beide Ansätze erzeugen in der Praxis ähnliches Gangverhalten.

JumperHards Belohnung zu verstehen reicht aus, um den Belohnungsabschnitt jedes Fortbewegungsartikels zu lesen. Das Vokabular ist dasselbe; nur die Koeffizientenwerte unterscheiden sich.

---

## Stretch Goals

### Automatisierte Hyperparametersuche mit Optuna

Anstatt manuell jeweils ein Experiment nach dem anderen durchzuführen, verwende Optuna, um den Hyperparameterraum automatisch zu durchsuchen:

```python
import optuna
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import numpy as np

def objective(trial):
    lr         = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps    = trial.suggest_categorical("n_steps", [64, 128, 256, 512])
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef   = trial.suggest_float("ent_coef", 1e-4, 0.05, log=True)
    
    env = StableBaselinesGodotEnv(env_path="./JumperHard.x86_64", n_parallel=4, speedup=20)
    model = PPO(
        "MlpPolicy", env,
        learning_rate=lr,
        n_steps=n_steps,
        clip_range=clip_range,
        ent_coef=ent_coef,
        batch_size=64,
        verbose=0,
    )
    model.learn(total_timesteps=200_000)
    
    # Quick eval
    rewards = []
    for _ in range(10):
        obs, done, total = env.reset(), False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _ = env.step(action)
            total += r
        rewards.append(total)
    env.close()
    return np.mean(rewards)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print("Best params:", study.best_params)
```

Installation: `pip install optuna`

!!! tip
    Optuna ist das praktischste Hyperparameter-Suchwerkzeug für SB3. RL-Zoo3 (verwendet in HF-Kurs Unit 3) verwendet Optuna intern.

---

## Was kommt als nächstes

**Unit 5:** Gleiche BallChase-Umgebung — neue Fähigkeit: Skalierung paralleler Rollouts mit `n_parallel` und ein ordentliches Evaluierungsprotokoll.

!!! info "Selbstcheck, bevor du weitermachst"
    Kannst du diese Fragen mit eigenen Worten beantworten?

    1. Was genau beschneidet (clippt) die **Clip-Range** in PPO, und was verhindert sie?
    2. Wie tauscht GAE-λ Bias gegen Varianz ab, und worauf reduziert sich λ = 1?
    3. Wenn `approx_kl` dauerhaft über 0,05 liegt — welchen Hyperparameter würdest du zuerst anpassen, und in welche Richtung?
    4. Warum schneidet Optunas Pruner schlechte Trials früh ab, und was würde passieren, wenn du nicht prunst?
    5. Was ist der Unterschied zwischen „Training konvergiert" und „Policy ist gut genug zum Ausliefern"?

    Wenn du alle fünf beantworten kannst — bist du bereit.

??? success "Antworten zum Selbstcheck"
    1. Sie beschneidet das **Wahrscheinlichkeitsverhältnis** r_t(θ) = π_θ(a|s) / π_θ_old(a|s) auf das Band [1−ε, 1+ε]. Aktualisierungen, die die Policy weiter bewegen würden, bekommen einen Gradienten von null — eine günstige **Trust Region**, die verhindert, dass ein einzelner glücklicher Rollout alles bisher Gelernte überschreibt.
    2. Höheres λ gewichtet längere n-Schritt-Erträge stärker: **niedrigerer Bias, höhere Varianz**. Niedrigeres λ stützt sich auf das Bootstrap der Wertfunktion: glattere Schätzungen, aber durch Critic-Fehler verzerrt. Bei **λ = 1** reduziert sich GAE auf reine **Monte-Carlo**-Erträge.
    3. Zuerst **`--learning_rate`** senken — die Lernrate ist der empfindlichste Parameter — oder alternativ `--clip_range` senken. Beides verkleinert, wie weit sich die Policy pro Aktualisierung bewegt, und genau das ist es, was ein hohes `approx_kl` dir als Problem anzeigt.
    4. Jeder Trial kostet einen vollen Trainingslauf (200k Schritte im Skript dieser Einheit). Ein **Pruner** bricht Trials ab, deren frühe Belohnungskurve bereits hinter den anderen zurückliegt, und verlagert Rechenzeit auf vielversprechende Bereiche. Ohne Pruning verbraucht jede schlechte Kombination ihr gesamtes Budget, und du erkundest für dieselbe Rechenzeit deutlich weniger Konfigurationen.
    5. „Konvergiert" heißt nur, dass die Trainingskurve abgeflacht ist — möglicherweise bei einer **suboptimalen Policy**. „Gut genug zum Ausliefern" verlangt das **Evaluierungsprotokoll** aus Abschnitt 7: `deterministic=True` über feste Episoden, ein Mean ± Std, dem du vertraust, und einen Visualisierungs-Checkpoint, der bestätigt, dass das Verhalten zu den Zahlen passt.

[→ Unit 5: Paralleles Training](unit-05.md)
