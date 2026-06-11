# Unit 3 — CrossTheRoad & DQN

Studiere das offizielle **CrossTheRoad**-Beispiel — diskrete 2D-Navigation mit spärlichen Belohnungen — und trainiere es dann mit **DQN** statt PPO. Ab hier ist der Standardarbeitsablauf: exportiertes Binary, ohne Fenster (headless).

[← Q-Learning](unit-q-learning.md) · [Kursübersicht](index.md)

!!! note "Voraussetzungen"
    - **[RL Essentials](unit-01.md)** — MDP-Schleife, Policy, Return, Diskontierungsfaktor γ
    - **[RL Foundations Deep Dive](unit-rl-foundations-deep.md)** — Q-Werte (Q-values), wertbasierte vs. richtlinienbasierte Familien, on-policy vs. off-policy
    - **[Unit 2](unit-02.md)** — die `AIController`-Schnittstelle, `get_obs()` / `set_action()`, einen PPO-Agenten end-to-end trainieren
    - **[Q-Learning-Einheit](unit-q-learning.md)** — tabellarisches Q-Learning. **Mach das zuerst, wenn du durchgehend liest:** DQN ist „Q-Learning mit einem neuronalen Netz", und die Tabellenversion lässt jeden Trick unten verständlich werden.
    - Sicherer Umgang mit dem Export eines Godot-Projekts in ein Binary ohne Fenster (headless)

!!! info "Zeit"
    Lesen: ~45 min · Training: ~45 min GPU / ~3 Std CPU

---

!!! warning "Ab hier primär die Kommandozeile"
    In den Units 0–2 wurde der Editor zum Bauen und Debuggen genutzt. Ab Unit 3 trainierst du mit einem exportierten Binary und lässt `--viz` für mehr Geschwindigkeit weg — danach führst du einen kurzen **Viz-Checkpoint** durch, wenn das Training abgeschlossen ist (siehe Abschnitt 9).

!!! info "Drei Wege, deine KI zu beobachten"
    Godot (Viz-Checkpoint) · TensorBoard (DQN vs. Unit 2 PPO) · `AIController`-Belohnungsanpassungen

!!! warning "Training stockt?"
    Prüfe in dieser Reihenfolge: (1) Vorzeichen und Skala der Belohnung — ist „gut" wirklich positiv? (2) Spärliche Belohnungen — erhält der Agent überhaupt ein Signal vor dem Ziel? (3) Beobachtungsfehler — werden Sensoren nach Resets aktualisiert? (4) TensorBoard flach, Godot sieht aber gut aus — du brauchst möglicherweise längeres Training oder einen Viz-Checkpoint.

---

## 1 · DQN auf einer Seite

**Wertbasierte (value-based)** Methoden lernen, wie gut jede Aktion in jedem Zustand ist — diese werden als **Q-Werte (Q-values)** bezeichnet. **DQN (Deep Q-Network)** verwendet ein neuronales Netz, um diese Werte zu approximieren, mit zwei stabilisierenden Tricks:

- **Erfahrungswiedergabe (Experience Replay)** — Übergänge werden in einem Puffer gespeichert und zufällig abgerufen, was Korrelationen zwischen aufeinanderfolgenden Beobachtungen aufbricht
- **Zielnetzwerk (Target Network)** — ein periodisch kopiertes, eingefrorenes Netz wird zur Berechnung der Zielwerte verwendet und verhindert Oszillationen

Verwende DQN, wenn der Aktionsraum **diskret** und die Belohnungen spärlich sind — wie beim Überqueren von Fahrspuren.

| | PPO (Unit 2) | DQN (diese Unit) |
|--|--|--|
| Familie | Richtlinienbasiert (Policy-based) | Wertbasiert (Value-based) |
| Lernt | Direkte Richtlinie | Q-Werte → Richtlinie |
| Erkundung | Stochastische Richtlinie + Entropie | ε-Greedy (anfangs zufällig, später gierig) |
| Am besten für | Dichte Belohnungen, kontinuierlich oder diskret | Spärliche Belohnungen, diskrete Aktionen |

!!! tip "Erkundung (Exploration): ε-Greedy"
    DQN erkundet mit **Epsilon-Greedy-Exploration**: anfangs zufällige Aktionen, später gierige Q-Aktionen. Wenn die Kurve flach bleibt, zerfällt ε möglicherweise zu schnell — überprüfe den Erkundungszeitplan deines Trainings-Skripts. Vergleiche nach dieser Unit PPOs Durchlauf aus Unit 2 in TensorBoard.

---

## 2 · Von Q-Tabellen zu neuronalen Netzen

Falls du die Q-Learning-Unit noch nicht gelesen hast, ist jetzt der richtige Zeitpunkt — **mach das zuerst, wenn du durchgehend liest:** DQN ist „Q-Learning mit einem neuronalen Netz", und die Tabellen-Version lässt jeden Trick hier unten „klick" machen: [Q-Learning-Unit](unit-q-learning.md).

**Das Problem mit Q-Tabellen**

Klassisches Q-Learning baut eine Nachschlagetabelle auf: jede Zeile ist ein Zustand, jede Spalte eine Aktion, jede Zelle enthält einen Q-Wert. Das funktioniert perfekt, wenn der Zustandsraum klein ist. Sobald du dich zu echten Umgebungen bewegst, bricht es zusammen:

- **CrossTheRoad (diese Unit):** Position auf einem Gitter — handhabbar, vielleicht einige tausend Zustände
- **Atari Pong:** Rohe Pixelframes — 210 × 160 Pixel × 3 Kanäle → ungefähr 10^18.000 mögliche Zustände
- **Godot RayCast-Beobachtungen:** Gleitkomma-Vektoren — buchstäblich unendlich viele Zustände

Eine Tabelle mit 10^18.000 Zeilen kann nicht existieren. Wir brauchen einen Funktionsapproximator, der über ähnliche Zustände *verallgemeinert*.

**Das Deep Q-Network**

Ersetze die Tabelle durch ein neuronales Netz:

```
Input layer:   observation  (pixels / raycasts / any vector)
Hidden layers: learned feature extraction
Output layer:  one Q-value per action
               Q(s, a_1), Q(s, a_2), ..., Q(s, a_n)
```

Das Netz bildet einen Zustand *s* auf einen Vektor von Q-Werten ab — einen pro Aktion. Der Agent wählt die Aktion mit dem höchsten Q-Wert (wenn er gierig agiert). Die Bellman-Gleichung (Bellman update rule) ist gegenüber dem tabellarischen Q-Learning unverändert; wir wenden sie lediglich auf die Ausgaben des Netzes statt auf eine Tabellenzelle an.

**Der Atari-Durchbruch (2013/2015)**

DeepMinds Paper von 2013 „Playing Atari with Deep Reinforcement Learning" und das darauffolgende Nature-Paper von 2015 zeigten, dass eine *einzige* DQN-Architektur — dieselben Netzgewichte, derselbe Algorithmus — lernen konnte, 49 Atari-Spiele direkt aus rohen Pixeleingaben zu spielen, wobei bei vielen das menschliche Niveau erreicht oder übertroffen wurde. Die entscheidenden Zutaten waren genau die beiden in Abschnitt 1 genannten Tricks: Erfahrungswiedergabe und ein Zielnetzwerk. Ohne diese Tricks war das Training extrem instabil.

!!! info "Warum schafft das Skalieren von Q-Learning auf tiefe Netze neue Probleme?"
    Neuronales Netz-Training setzt i.i.d.-Daten (unabhängig und identisch verteilt) voraus. RL-Übergänge sind beides nicht — aufeinanderfolgende Frames sind nahezu identisch, und die Zielwerte, auf die wir trainieren, verschieben sich ständig, während das Netz lernt. Erfahrungswiedergabe und das Zielnetzwerk sind technische Lösungen für beide Probleme. Wir behandeln beides ausführlich weiter unten.

---

## 3 · Erfahrungswiedergabe (Experience Replay)

**Warum aufeinanderfolgende Übergänge ein Problem sind**

Stell dir vor, der Agent macht die Schritte *s_t → s_{t+1} → s_{t+2}* über eine Straße. Diese drei Beobachtungen sind nahezu identisch — leicht unterschiedliche Positionen auf derselben Straße. Wenn du sie der Reihe nach trainierst, enthält jedes Mini-Batch nur eine Art von Erfahrung. Das Netz überanpasst sich an „in der Nähe von Position X" und vergisst alles, was es zuvor über die Positionen A, B und C gelernt hat.

Dies ist **katastrophales Vergessen (catastrophic forgetting)** — die neuronale Netzversion eines Schülers, der ein Thema so intensiv lernt, dass er die anderen vergisst.

**Der Wiederholungspuffer (Replay Buffer)**

Die Lösung ist konzeptionell einfach: Speichere jeden Übergang, den der Agent jemals erlebt, und trainiere dann auf *zufälligen* Mini-Batches aus der gesamten Geschichte.

Ein einzelner Übergang ist ein Tupel:

```
(s, a, r, s', done)
 │   │   │   │    └── did the episode end?
 │   │   │   └─────── next state
 │   │   └─────────── reward received
 │   └─────────────── action taken
 └─────────────────── current state
```

In Python-Pseudocode:

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)  # circular: old entries drop off

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
```

**Wichtige Designentscheidungen:**

- **Kapazität:** 10k–1M Übergänge. Größere Puffer behalten ältere, vielfältigere Erfahrungen im Pool. CrossTheRoad kann 50k verwenden; Atari-skalierte Aufgaben verwenden 1M.
- **Zufälliges Sampling:** Jeder Trainingsschritt zieht ein zufälliges Mini-Batch. Übergänge aus 100 Episoden zuvor mischen sich mit Übergängen aus 5 Episoden zuvor — keine zeitliche Korrelation.
- **Zirkulär (deque):** Wenn der Puffer voll ist, fällt der älteste Eintrag heraus. Das verhindert, dass der Puffer mit veralteten, vortrainierten Erfahrungen gefüllt wird.

!!! tip "Puffergröße vs. Arbeitsspeicher"
    Das Speichern roher Pixelbeobachtungen bei 1M Kapazität kostet Gigabytes RAM. Für godot-rl-agents-Aufgaben sind RayCast-Beobachtungen kleine Gleitkommazahlen — 100k Kapazität ist in der Regel ausreichend und hält den Speicherverbrauch handhabbar.

### Priorisierte Erfahrungswiedergabe (Prioritized Experience Replay, PER)

Gleichmäßiges zufälliges Sampling behandelt jeden Übergang gleich — aber die meisten Übergänge in einem großen Puffer sind „langweilig" (Belohnung = 0, Q-Ziel nahe der aktuellen Schätzung). **PER** sampelt Übergänge proportional zu ihrem Temporal-Differenz-Fehler (TD error): Übergänge, bei denen das Netz am meisten falsch lag, werden häufiger erneut betrachtet.

**Prioritätsformel:**

```
p_i = |δ_i|^α + ε_per
```

- `δ_i` — Temporal-Differenz-Fehler (TD error) für Übergang i (großer Fehler → hohe Priorität)
- `α` — steuert das Ausmaß der Priorisierung (0 = gleichmäßig, 1 = voll proportional)
- `ε_per` — kleine Konstante, damit jeder Übergang eine von null verschiedene Chance hat, gesampelt zu werden

Das Sampling nach Priorität führt zu einer Verzerrung (nicht-gleichmäßige Datenverteilung), die durch Importance-Sampling-Gewichte `w_i = (1 / N·P(i))^β` im Verlust korrigiert wird.

**SB3 via sb3-contrib:**

```python
from stable_baselines3 import DQN
from sb3_contrib.common.buffers import PrioritizedReplayBuffer

model = DQN(
    "MlpPolicy", env,
    replay_buffer_class=PrioritizedReplayBuffer,
    replay_buffer_kwargs={"alpha": 0.6},
    learning_starts=1000,
)
```

**Wann PER hilft:** Aufgaben mit spärlichen Belohnungen, bei denen wenige Übergänge fast das gesamte nützliche Signal enthalten (erstes Ziel erreicht, erste Kollision). **Wann es nicht hilft:** Umgebungen mit dichten Belohnungen, bei denen fast jeder Übergang informativ ist — gleichmäßiges Sampling ist dort bereits gut genug. Beobachte die Varianz von `train/td_loss`: hohe Varianz früh, die schnell sinkt, ist ein Zeichen dafür, dass PER wirkt.

---

## 4 · Zielnetzwerk (Target Network)

**Das Problem des sich bewegenden Ziels**

Die DQN-Verlustfunktion ist:

```
L = (r + γ · max_a' Q(s', a') − Q(s, a))²
         └── bootstrap target ──┘
```

Sowohl `Q(s, a)` (die Vorhersage) als auch `Q(s', a')` (das Ziel) kommen aus demselben Netz. Jedes Mal, wenn wir das Netz aktualisieren, um diesen Verlust zu reduzieren, *bewegen sich beide Seiten*. Wir jagen ein Ziel, das bei jedem Schritt wegläuft — als würde man versuchen, einen Ball zu treffen, der sich bei jedem Schlag bewegt.

In der Praxis führt dies dazu, dass das Training oszilliert oder vollständig divergiert.

**Die Lösung: Das Ziel einfrieren**

Halte *zwei* Kopien des Netzes vor:

| Netz | Rolle | Wie oft aktualisiert |
|------|------|----------------------|
| **Q-Netz** (online) | Macht Vorhersagen, wird bei jedem Schritt trainiert | Bei jedem Gradientenschritt |
| **Zielnetzwerk Q̂** | Liefert Bootstrap-Zielwerte | Alle C Schritte (harte Kopie) oder kontinuierliches sanftes Update |

Der Verlust wird zu:

```
L = (r + γ · max_a' Q̂(s', a') − Q(s, a))²
                   ↑ frozen target network
```

Jetzt ist die Zielseite für C Schritte stabil, was dem Online-Netz etwas Festes zum Konvergieren gibt.

**Hartes Update vs. Sanftes Update**

```python
# Hard update — copy weights every C steps (e.g., C = 1000)
if step % 1000 == 0:
    target_net.load_state_dict(q_net.state_dict())

# Soft update — blend weights every step (more stable, slower lag)
tau = 0.005
for p, pt in zip(q_net.parameters(), target_net.parameters()):
    pt.data = tau * p.data + (1 - tau) * pt.data
```

SB3s DQN verwendet standardmäßig ein sanftes Update (`tau=1.0` bedeutet hartes Update; niedrigere Werte ergeben ein sanftes). Der Parameter `target_update_interval` steuert, wie oft Updates stattfinden.

!!! info "Zwei Netze, gleiche Architektur"
    Zielnetzwerk und Online-Netz teilen exakt dieselbe Architektur — nur die Gewichte unterscheiden sich. Das Zielnetzwerk empfängt keine direkten Gradientenaktualisierungen; es erhält lediglich periodische Kopien der Gewichte des Online-Netzes.

!!! warning "Die tödliche Triade — warum DQN divergieren kann"
    Die Kombination von drei Dingen führt dazu, dass Q-Learning divergiert:

    1. **Funktionsapproximation** (neuronales Netz statt Tabelle)
    2. **Bootstrapping** (Verwendung von Q̂ zur Schätzung von Q — die Bellman-Gleichung)
    3. **Off-Policy-Lernen** (Wiederholungspuffer enthält Daten aus alten Richtlinien)

    Zwei davon sind in Ordnung. Alle drei zusammen erzeugen Instabilität.

    DQN überlebt durch sorgfältiges Engineering:

    - Zielnetzwerk: macht Bootstrap-Ziele stabiler (adressiert die Interaktion von #2 + #3)
    - Wiederholungspuffer mit nur aktuellen Daten: begrenzt, wie weit die Daten von der aktuellen Richtlinie abweichen
    - Gradientenclipping: verhindert, dass die Funktionsapproximation überschießt

    Deshalb kann man nicht einfach ein beliebiges neuronales Netz in die Bellman-Gleichung einsetzen — das Engineering ist entscheidend.

---

## 5 · Epsilon-Greedy-Exploration in DQN

**Das Dilemma zwischen Erkundung (Exploration) und Ausnutzung (Exploitation)**

Ein rein gieriger Agent wählt immer die Aktion mit dem höchsten aktuellen Q-Wert. Früh im Training sind diese Q-Werte reines Rauschen — gierig zu handeln bedeutet zufällig schlechte Entscheidungen, die sich nie verbessern. Rein zufällig zu handeln ist sicher fürs Lernen, konvergiert aber nie.

ε-Greedy findet den Mittelweg:

```
With probability ε:     take a random action  (explore)
With probability 1-ε:   take argmax Q(s, a)  (exploit)
```

**Der Zerfallszeitplan**

Beginne mit intensiver Erkundung, zerfalle dann zur Ausnutzung, während der Agent Erfahrungen sammelt:

```
ε_start = 1.0      # 100% random at step 0
ε_end   = 0.05     # 5% random after decay period
decay_steps = 100_000

ε(t) = ε_end + (ε_start - ε_end) × max(0, 1 - t / decay_steps)
```

Bei Schritt 0 handelt der Agent zufällig und entdeckt verschiedene Überquerungen. Ab Schritt 100k nutzt er hauptsächlich seine gelernten Q-Werte aus, mit 5 % zufälliger Erkundung, um nicht in lokalen Optima stecken zu bleiben.

**Drei Zerfallszeitpläne**

| Zeitplan | Formel | Wann verwenden |
|----------|--------|----------------|
| **Linear** | `ε(t) = ε_end + (ε_start - ε_end) · max(0, 1 - t/decay_steps)` | Die meisten Aufgaben — vorhersehbar und leicht zu tunen |
| **Exponentiell** | `ε(t) = ε_end + (ε_start - ε_end) · exp(-t / decay_rate)` | Schnelle frühe Erkundung; bleibt am Ende über `ε_end` |
| **Curriculum** | Stufenfunktion an Meilensteinen | Wenn sich die Aufgabenschwierigkeit in diskreten Phasen ändert |

Exponentieller Zerfall erkundet früh aggressiv und verlangsamt sich langsam — nützlich, wenn du den Agenten schnell den Zustandsraum abdecken lassen willst, aber eine lange „Verfeinerungsphase" möchtest. Curriculum-Zeitpläne (z. B. ε = 0,5 für 100k Schritte halten, während die Aufgabe noch zufällig ist, dann auf 0,05 abfallen) sind in mehrstufigen Umgebungen üblich.

**Praktische Regeln:**

- `exploration_fraction` — Anteil der gesamten Zeitschritte, über den ε zerfällt (SB3-DQN-Parameter). `exploration_fraction=0.1` bedeutet, dass ε über die ersten 10 % des Trainings zerfällt.
- `exploration_final_eps` — ε am Ende des Zerfallszeitraums (= `ε_end` in der Formel). Standard `0.05`.
- `exploration_initial_eps` — Start-ε. Standard `1.0`.

```python
model = DQN("MlpPolicy", env,
    exploration_fraction=0.2,       # decay over first 20% of timesteps
    exploration_final_eps=0.05,     # 5% random at convergence
    verbose=1)
```

TensorBoard verfolgt `rollout/exploration_rate` — beobachte den Zerfall und vergleiche ihn mit `ep_rew_mean`. Die Belohnung sollte beginnen zu steigen, wenn ε ungefähr 0,2–0,3 unterschreitet (Agent beginnt, gelernte Q-Werte auszunutzen). Wenn `ep_rew_mean` bei ε = 0,05 noch immer flach ist, liegt das Problem nicht an der Erkundung — überprüfe das Belohnungsdesign.

**Training vs. Auswertung**

- **Beim Training:** Verwende den ε-Zeitplan — viel Erkundung früh
- **Bei der Auswertung:** Setze ε = 0 (vollständig gierig) oder verwende `deterministic=True` — du willst das *beste* Verhalten des Agenten, keine zufällige Erkundung

!!! tip "Den ε-Zerfall in CrossTheRoad beobachten"
    Der DQN-Agent, der CrossTheRoad erkundet, macht anfangs zufällige Züge — in den ersten paar tausend Schritten wirst du sehen, wie er ständig von der Straße fällt. Wenn ε gegen 0,05 zerfällt, beginnt der Agent, seine gelernten Q-Werte auszunutzen, und du wirst sehen, wie er zielgerichtete Überquerungsversuche unternimmt. Die flache → scharfe Sprungkurve in TensorBoard entspricht direkt dem ε-Zerfall, der auf eine ausreichende Menge an Wiederholungspuffer-Erfahrung trifft.

### Design des Zerfallszeitplans

**Linearer Zerfall** — der SB3-Standard, gesteuert durch zwei Parameter:

```python
model = DQN(
    "MlpPolicy", env,
    exploration_fraction=0.1,   # fraction of total_timesteps over which ε decays
    exploration_final_eps=0.05, # ε_min at the end of the decay period
)
# ε(t) = max(ε_min, ε_start − (ε_start − ε_min) × (step / decay_steps))
```

**Exponentieller Zerfall** — aggressive frühe Erkundung, langsam später:

```python
epsilon = epsilon_min + (epsilon_start - epsilon_min) * math.exp(-step / decay_rate)
```

| Zeitplan | Form | Wann verwenden |
|----------|------|----------------|
| Linear | Konstante Rate | Die meisten Aufgaben — vorhersehbar und leicht zu tunen |
| Exponentiell | Früh schnell, spät langsam | Wenn du in frühen Trainingsphasen intensive Erkundung möchtest |
| Curriculum | Stufenfunktion | Wenn sich die Aufgabenschwierigkeit ändert (mehrere Räume, gestufte Umgebungen) |

**Praktische Regeln:** `exploration_fraction=0.1` (Zerfall über 10 % der Zeitschritte) ist für spärliche Aufgaben oft zu schnell — versuche `0.3`. `exploration_final_eps=0.05` verhindert, dass die Richtlinie brüchig wird. Überwache `rollout/exploration_rate` in TensorBoard, um zu bestätigen, dass ε mit der erwarteten Rate zerfällt.

---

## 6 · CrossTheRoad öffnen

**Klonen und importieren**

1. Öffne das Projekt aus [examples/CrossTheRoad](https://github.com/edbeeching/godot_rl_agents_examples/tree/main/examples/CrossTheRoad) in Godot .NET
2. Aktiviere das Godot RL Agents-Plugin (Projekt → Projekteinstellungen → Plugins)
3. Finde die Trainingsszene und das `AIController`-Skript

**Exportieren für headloses Training**

Projekt → Exportieren → deine Plattformvorlage hinzufügen → Binary exportieren. Dann dagegen trainieren:

```bash
conda activate godot_env
gdrl --env_path=./CrossTheRoad.x86_64 \
  --experiment_name=CrossTheRoad_DQN \
  --timesteps=500000 \
  --speedup=8 \
  --n_parallel=4
```

!!! info "DQN via SB3"
    Der Standard-`gdrl`-Befehl verwendet PPO. Um DQN zu verwenden, schreibe ein kurzes Trainings-Skript:

    ```python
    from stable_baselines3 import DQN
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

    env = StableBaselinesGodotEnv(env_path="./CrossTheRoad.x86_64", n_parallel=4, speedup=8)
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="logs/")
    model.learn(total_timesteps=500_000)
    model.save("crosstheroad_dqn")
    env.close()
    ```

---

## 7 · Den Code lesen

Verfolge diese in der Reihenfolge — derselbe Rhythmus wie SimpleReachGoal in Unit 2:

- **`get_obs()`** — was der Agent sieht (Position, nahegelegene Gefahren, Entfernung zum Ziel)
- **`get_action_space()`** — diskrete Bewegungen (warten / oben / unten / links / rechts)
- **Belohnungslogik** — spärliches Signal beim Ziel + Strafe bei Kollisionen; beachte den Kontrast zu den dicht geformten Lander-Belohnungen aus Unit 2
- **Sync-Knoten** — Anzahl der parallelen Umgebungswurzeln in der Trainingsszene

**Wichtige Frage vor dem Training:** Erhält der Agent *irgendein* Belohnungssignal während einer typischen Episode, oder erst ganz am Ende? Spärliche Belohnungen erfordern mehr Erkundungszeit — berücksichtige das bei deinem Zeitschrittbudget.

---

## 8 · Headless trainieren

```bash
conda activate godot_env
tensorboard --logdir=logs &
python train_crosstheroad_dqn.py
```

Beobachte `ep_rew_mean` — bei spärlichen Belohnungen kann es über tausende Episoden flach bleiben, dann scharf springen, wenn der Agent sichere Überquerungen entdeckt. Das ist normal für DQN bei spärlichen Aufgaben; PPO auf derselben Aufgabe würde einen sanfteren Anstieg zeigen.

**Was nach dieser Unit in TensorBoard zu vergleichen ist:**

| Metrik | PPO (Unit 2 Lander) | DQN (CrossTheRoad) |
|--------|--------------------|--------------------|
| Form der `ep_rew_mean`-Kurve | Sanfter, gradueller Anstieg | Flach → scharfer Sprung |
| `train/entropy_loss` | Vorhanden | Nicht anwendbar |
| `train/loss` | Richtlinien- + Wertverluste | Nur TD-Verlust |

!!! check "Fertig, wenn"
    CrossTheRoad hat keinen veröffentlichten Benchmark, beurteile den Erfolg also auf zwei Arten: (1) der **Viz-Checkpoint** (Abschnitt 9) zeigt, dass der Agent in der Mehrheit der Episoden die andere Seite erreicht, und (2) `ep_rew_mean` ist klar aus seiner anfänglichen flachen Phase herausgetreten und hat sich stabilisiert — die charakteristische DQN-Kurve „flach → scharfer Sprung". Eine nach deinem vollen Schrittbudget noch immer flache Kurve deutet auf den ε-Zeitplan oder einen Vorzeichenfehler in der Belohnung hin, nicht auf zu wenig Trainingszeit.

---

## 9 · Anpassen & Viz-Checkpoint

**Viz-Checkpoint (~5 Min.)**

Führe die trainierte Richtlinie erneut mit `--viz` oder Play Scene in Godot aus. Mache einen Screenshot des Verhaltens, das der TensorBoard-Kurve entspricht (oder ihr widerspricht) — das verhindert, dass headloses Training unsichtbar bleibt.

**Stretch Goals (wähle eines):**

- Erhöhe die Kollisionsstrafe um das 2-Fache — beschleunigt sich das Lernen oder stockt es?
- Füge eine kleine Belohnung für Fortschritte nach vorne hinzu — vergleiche mit dem reinen spärlichen Setup
- Trainiere dieselbe Umgebung mit PPO — welcher Algorithmus erreicht zuerst zuverlässige Überquerungen?

---

## 10 · DQN-Einschränkungen und -Varianten (optional beim ersten Lesen)

!!! note "Erster Durchgang? Überfliege oder überspringe diesen Abschnitt."
    Die Abschnitte 1–9 sind die DQN-Kernlektion und alles, was du brauchst, um CrossTheRoad zu trainieren. Die Varianten unten (Double, Dueling, Noisy, Rainbow) vertiefen dein Verständnis, sind aber nicht erforderlich, um die Unit abzuschließen — komm zurück, wenn du sie brauchst.

DQN ist elegant, hat aber echte Einschränkungen, auf die du in späteren Units stoßen wirst.

**Nur diskrete Aktionen**

Q-Werte sind über eine endliche Menge von Aktionen definiert: Q(s, a_1), Q(s, a_2), ..., Q(s, a_n). Das Maximum zu nehmen ist O(n) — machbar, wenn n 5 (CrossTheRoad) oder 18 (Atari) ist. Bei *kontinuierlichen* Aktionen (z. B. Gelenkdrehmomente in JumperHard oder Gas in FlyBy) ist n unendlich. Man kann nicht über unendlich viele Aktionen aufzählen und maximieren. Deshalb sind wertbasierte Methoden größtenteils auf diskrete Steuerung beschränkt.

**Überschätzungsverzerrung (Overestimation Bias)**

Der `max`-Operator ist optimistisch: er neigt dazu, Q-Werte zu überschätzen, weil er immer die höchste verrauschte Schätzung wählt. Im Laufe der Zeit bootstrappen sich überschätzte Werte gegenseitig, und Q-Werte werden aufgebläht. Das kann das Lernen verlangsamen oder das späte Training destabilisieren.

**Double DQN — die Lösung**

Trenne Aktions*auswahl* von Aktions*bewertung*:

```
# Standard DQN (biased):
target = r + γ · max_a' Q̂(s', a')                     # target net picks AND evaluates

# Double DQN (unbiased):
a_star = argmax_a' Q(s', a')                            # online net selects best action
target = r + γ · Q̂(s', a_star)                         # target net evaluates that action
```

Das Online-Netz zur Auswahl der Aktion und das Zielnetzwerk zur Bewertung zu verwenden, entfernt die systematische Aufwärtsverzerrung.

**Dueling DQN — die Erweiterung**

Teile die letzten Schichten des Netzes in zwei Ströme auf:

- **Wertstrom (Value Stream)** V(s) — wie gut ist dieser Zustand unabhängig von der Aktion?
- **Vorteilsstrom (Advantage Stream)** A(s, a) — wie viel besser ist Aktion a als der Durchschnitt?
- Kombinieren: Q(s, a) = V(s) + (A(s, a) − mean_a A(s, a))

Das hilft dem Agenten zu lernen, dass manche Zustände einfach schlecht sind, unabhängig davon, was er tut — nützlich für CrossTheRoad, wo das Hineinfahren in den Verkehr katastrophal ist, egal welchen Zug du als nächstes machst.

!!! info "SB3 handhabt diese automatisch"
    Stable-Baselines3s `DQN`-Klasse unterstützt Double DQN via `policy_kwargs={"optimize_memory_usage": False}` und Dueling-Netze via `policy_kwargs={"dueling": True}`. Du musst sie nicht von Grund auf implementieren.

!!! tip "Brücke zur kontinuierlichen Steuerung"
    Für kontinuierliche Aktionen — Unit 6 FlyBy, JumperHard mit Gelenkdrehmomenten — brauchen wir richtlinienbasierte Methoden, die Aktions*verteilungen* statt Q-Wert-Tabellen ausgeben. Genau das macht PPO, und deshalb konzentriert sich die nächste Unit darauf. Siehe [Unit 4: JumperHard & PPO](unit-04.md).

### Noisy Networks und Rainbow DQN (fortgeschritten)

**Noisy Networks** ersetzen die letzten linearen Schichten des Q-Netzes durch `NoisyLinear`-Schichten, die *gelerntes* Rauschen in die Gewichte injizieren. Das Netz kontrolliert seine eigene Erkundung durch Anpassung der Rauschstärke — kein ε-Zeitplan nötig.

**Rainbow DQN** (Hessel et al. 2017) kombiniert sechs DQN-Verbesserungen in einem einzigen Agenten:

| Komponente | Was sie hinzufügt |
|------------|------------------|
| Double DQN | Unverzerrte Ziel-Q-Werte |
| Dueling DQN | Separate Wert- und Vorteilsströme |
| Priorisierte Erfahrungswiedergabe (Prioritized Replay, PER) | Wichtige Übergänge häufiger sampeln |
| Multi-Step-Returns | n-Schritt-TD statt 1-Schritt |
| Distributionelles RL (C51) | Vollständige Rückgabeverteilung modellieren, nicht nur den Mittelwert |
| Noisy Networks | Gelernte Erkundung — kein ε-Zeitplan |

Das wegweisende Ergebnis: Die Kombination aller sechs übertrifft jede einzelne Verbesserung bei Atari deutlich. Die Verbesserungen sind komplementär, nicht redundant.

**Praktischer Hinweis für Godot-Aufgaben:** Vollständiges Rainbow ist nicht in SB3. Verwende einzelne Komponenten: Double DQN ist standardmäßig aktiviert; Dueling via `policy_kwargs={"dueling": True}`; PER via sb3-contrib (siehe Abschnitt 3.1 oben). Für die meisten Godot-Umgebungen übertrifft PPO jeden DQN-Ableger — DQN glänzt bei diskreten, spärlich belohnten Aufgaben. Wenn du bei einer diskreten Aufgabe mit spärlichen Belohnungen bist und DQN noch immer schlecht abschneidet, versuche PER + Dueling, bevor du zu Rainbow greifst.

---

## 11 · Off-Policy vs. On-Policy: Warum das wichtig ist

Diese Unterscheidung ist eine der praktisch wichtigsten im RL — sie erklärt, warum DQN und PPO sehr unterschiedliche Infrastruktur benötigen.

**On-Policy (PPO):** Trainingsdaten müssen von der *aktuellen* Richtlinie stammen. Nach jeder Gradientenaktualisierung werden alle gesammelten Übergänge verworfen — sie sind jetzt „veraltet" (von der alten Richtlinie) und können nicht wiederverwendet werden.

| | On-Policy (PPO) | Off-Policy (DQN, SAC) |
|--|--|--|
| Datenquelle | Nur aktuelle Richtlinie | Beliebige Richtlinie (einschließlich alter, zufälliger Richtlinien) |
| Nach einem Update | Alle Übergänge verwerfen | Alle Übergänge im Wiederholungspuffer behalten |
| Stichprobeneffizienz | Gering — jeder Übergang wird einmal verwendet | Hoch — jeder Übergang wird viele Male verwendet |
| Stabilität | Theoretisch fundiert, stabil | Erfordert Korrekturmechanismen |

**Off-Policy (DQN):** Daten von *beliebigen* Richtlinien können für das Training verwendet werden. Der Wiederholungspuffer speichert Millionen von Übergängen, die von vielen verschiedenen Richtlinienversionen gesammelt wurden — einschließlich der zufälligen Richtlinie aus dem frühen Training. Der Agent trainiert auf zufälligen Mini-Batches aus dieser gesamten Geschichte.

**Warum das Zielnetzwerk von DQN dadurch notwendig wird:**

Da der Wiederholungspuffer Übergänge enthält, die von alten Richtlinien gesammelt wurden, wurden die Q-Werte, von denen du bootstrappst, von einer anderen (oft schlechteren) Richtlinie als der aktuell trainierten generiert. Ohne das Zielnetzwerk, das die Bootstrap-Ziele stabilisiert, würde diese Off-Policy-Natur dazu führen, dass die Verlustfunktion einem sich verschiebenden, inkonsistenten Ziel nachjagt — was zur Divergenz führt.

**Praktische Auswirkungen:**

- DQN mit einem großen Wiederholungspuffer kann stichprobeneffizienter als PPO bei Aktionen mit diskretem Aktionsraum sein — jeder Übergang wird hunderte Male wiederverwendet
- PPO skaliert besser mit parallelen Umgebungen (siehe [Unit 5: Paralleles Training](unit-05.md)) — das Ausführen von N Umgebungen parallel liefert N-mal mehr On-Policy-Daten pro Sekunde
- Für kontinuierliche Aktionen verwendet SAC dasselbe Off-Policy-Prinzip wie DQN, erweitert es aber auf kontinuierliche Steuerung — siehe [Unit SAC](unit-sac.md)

---

## 12 · Stretch Goals

**Den ε-Zeitplan durchprobieren.** Führe CrossTheRoad dreimal aus und ändere nur den Explorationszeitplan: (a) schneller Zerfall — ε erreicht 0,05 bei 25 % des Trainings, (b) der Standard, (c) langsamer Zerfall — ε liegt bei 75 % des Trainings noch bei 0,3. Sage voraus, welche Kurve am schnellsten steigt, welche am höchsten plateauiert und welche sich nie erholt. Prüfe dann TensorBoard. Die Lektion: DQNs Wanduhrzeit bis zur Lösung hängt ebenso vom ε-Zeitplan ab wie vom Netz.

**DQN vs. PPO im direkten Vergleich.** Trainiere CrossTheRoad mit PPO bei gleichem Gesamtschritt-Budget (z. B. 500k). Zeichne beide `ep_rew_mean`-Kurven in dasselbe TensorBoard. Welche erreicht das Ziel zuerst? Welche endet höher? Schreibe eine Ein-Satz-Hypothese für das *Warum* auf, bevor du es ausführst. Die diskreten Aktionen + spärlichen Belohnungen von CrossTheRoad begünstigen DQN — bestätige oder widerlege das auf deiner eigenen Maschine.

**Double DQN von Hand implementieren.** Ohne SB3: schreibe eine kleine Trainingsschleife, die CartPole-v1 mit zwei Netzen lernt: ein Online-Q-Netz für die Aktionsauswahl und ein Zielnetz für die Bewertung. Das Double-DQN-Ziel ist `r + γ · Q_target(s', argmax_a Q_online(s', a))` — nicht `r + γ · max_a Q_target(s', a)`. Kopiere die Online-Gewichte alle 500 Schritte in das Zielnetz.

!!! warning "Pseudocode"
    ```python
    import gymnasium as gym
    import torch
    import torch.nn as nn

    env = gym.make("CartPole-v1")
    q_online = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))
    q_target = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))
    q_target.load_state_dict(q_online.state_dict())

    # In the update step:
    next_action = q_online(next_obs).argmax(dim=-1)              # online picks
    next_q = q_target(next_obs).gather(-1, next_action.unsqueeze(-1))  # target evaluates
    td_target = reward + gamma * next_q.squeeze(-1) * (1 - done)
    loss = ((q_online(obs).gather(-1, action.unsqueeze(-1)).squeeze(-1) - td_target.detach()) ** 2).mean()
    ```

    Beobachte, wie `ep_rew_mean` Richtung 500 steigt. Vergleiche mit einer Einzelnetz-Baseline — der Abstand ist bei CartPole klein, bei Atari groß.

---

## Was kommt als Nächstes

**Unit 4:** JumperHard — der kanonische PPO-Benchmark, headless-Export, Hyperparameter-Tuning.

!!! info "Selbstcheck, bevor du weitermachst"
    Kannst du diese Fragen mit eigenen Worten beantworten?

    1. Warum ersetzt ein neuronales Netz die Q-Tabelle, sobald du Gitterwelten verlässt?
    2. Was behebt der **Experience-Replay-Buffer**, worum sich On-Policy-Methoden nicht kümmern müssen?
    3. Was geht schief, wenn das Target-Netz jeden Schritt statt alle N Schritte aktualisiert wird?
    4. Warum ist ε-greedy die natürliche Explorationsstrategie für DQN, aber nicht für PPO?
    5. Wähle eine Godot-Umgebung aus Phase 2 — würdest du zu DQN oder PPO greifen, und warum?

    Wenn du alle fünf beantworten kannst — bist du bereit.

??? success "Antworten zum Selbstcheck"
    1. Echte Umgebungen haben zu viele (oder kontinuierliche) Zustände, um je einen Q-Wert pro Zustand zu speichern. Ein **Netz approximiert Q(s, a)** und verallgemeinert über ähnliche, nie gesehene Zustände — eine Tabelle kann das nicht.
    2. Der **Replay-Buffer** bricht die zeitliche Korrelation zwischen aufeinanderfolgenden Übergängen (und lässt jeden Übergang wiederverwenden). Zufällige Minibatches verhalten sich eher wie i.i.d.-Daten, was das Training stabilisiert. On-Policy-Methoden verwerfen Daten nach jedem Update und haben dieses Korrelations-/Wiederverwendungsproblem daher nie.
    3. Wird das Target-Netz **jeden Schritt** aktualisiert, wandert das Bootstrap-Ziel mit dem Online-Netz — das Netz jagt einem ständig wandernden Ziel hinterher, was Oszillation oder Divergenz verursacht. Es für N Schritte einzufrieren gibt ein stabiles Ziel, auf das hin regrediert wird.
    4. DQN lernt deterministische **Q-Werte** ohne eingebaute Zufälligkeit und braucht daher einen expliziten Explore/Exploit-Regler — **ε-greedy**. PPO hat bereits eine **stochastische Policy + Entropie-Bonus**, die Exploration ist also intrinsisch und ε unnötig.
    5. Beispiel — **CrossTheRoad: DQN.** Diskrete Aktionen plus spärliche Belohnungen sind DQNs Stärke, und Off-Policy-Replay ist dort stichprobeneffizient. (Eine Umgebung mit dichten Belohnungen oder kontinuierlichen Aktionen würde eher auf PPO deuten.)

[→ Unit 4: JumperHard & PPO](unit-04.md)
