# Q-Learning — Von Tabellen zu tiefen Netzen

Bevor wir Policies durch neuronale Netze ersetzen (Unit 3 / DQN), müssen wir den Algorithmus im Herzen des wertbasierten RL verstehen. Diese Unit füllt die Lücke zwischen den RL-Grundlagen und Deep Q-Networks: sie erklärt die **Bellman-Gleichung**, die **Q-Tabelle**, die **Q-Learning-Update-Regel** und zeigt eine ~50-zeilige Python-Implementierung, die FrozenLake von Grund auf löst.

[← Unit 2: Lunar Lander](unit-02.md) · [Kursstartseite](index.md)

!!! note "Voraussetzungen"
    - **[RL Essentials](unit-01.md)** — MDP-Schleife, Policy, Return, Diskontierungsfaktor γ
    - **[Unit 2](unit-02.md)** — ein laufender PPO-Trainingsdurchlauf (nützlich für den Kontrast in §1)
    - Sicheres Ausführen eines Python-Skripts (`pip install gymnasium`, dann laufen lassen)
    - Keine Vorkenntnis in dynamischer Programmierung nötig — wir leiten Bellman in §2 her

!!! info "Zeit"
    Lesen: ~35 min · Training: ~20 min GPU / ~1,5 Std CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    Python-Terminal (Q-Tabelle ausgeben) · matplotlib (Trainingskurve) · Policy-Visualisierung (Pfeil-Gitter)

---

## Warum es diese Unit gibt

Bisher hast du Policies mit PPO trainiert — ein neuronales Netz, das **Aktionswahrscheinlichkeiten** direkt ausgibt. Das ist eine *policy-basierte* Methode. Eine ganz andere Familie von RL-Algorithmen geht anders vor: statt zu lernen, *was zu tun ist*, lernt sie, *wie gut jede Wahl ist*, und handelt dann gierig auf dieses Wissen. Das ist die *wertbasierte* Familie, und **Q-Learning** ist ihr berühmtester Vertreter.

Deep Q-Networks (DQN) — die den ersten Agenten trainierten, der Menschen in Atari-Spielen schlug — sind nichts weiter als Q-Learning mit einem neuronalen Netz statt einer Tabelle. Wenn du die Tabellen-Version verstehst, wird DQN ein kleiner Schritt statt eines Sprungs.

---

## 1 · Warum wertbasierte Methoden?

Es gibt zwei große Strategien im Reinforcement Learning:

| Familie | Was sie lernt | Beispielalgorithmen |
|---------|---------------|---------------------|
| **Policy-basiert** | π(a\|s) — direkte Abbildung von Zuständen auf Aktionswahrscheinlichkeiten | REINFORCE, PPO, A2C |
| **Wertbasiert** | Q(s,a) — wie gut jede Aktion in jedem Zustand ist | Q-Learning, SARSA, DQN |

Die Kernidee wertbasierter Methoden ist wunderbar einfach:

> **Wenn du den Wert jedes (Zustand, Aktion)-Paares kennst, ist die Auswahl der besten Aktion trivial — nimm die Aktion mit dem höchsten Wert.**

Du lernst nie explizit eine Policy. Die Policy ist *implizit* in der Wertfunktion: „In Zustand s nimm argmax über a von Q(s,a)".

**Wann wertbasierte Methoden glänzen:**

- **Diskrete Aktionsräume.** `argmax` über 4 Aktionen ist billig. Über einen kontinuierlichen Aktionsraum (z. B. Lenkwinkel ∈ [−1, 1]) nicht.
- **Wohldefinierte Zustandsräume.** Gitterwelten, Brettspiele, diskrete Navigation.
- **Stichproben-Effizienz.** Wertbasierte Methoden können alte Erfahrung wiederverwenden (Off-Policy) — siehe Abschnitt 4.

**Wann sie schwächeln:**

- Kontinuierliche Aktionen (stattdessen DDPG / SAC / PPO).
- Riesige oder kontinuierliche Zustandsräume (stattdessen DQN — Abschnitt 8).

---

## 2 · Die Bellman-Gleichung

Die Bellman-Gleichung ist *die* Grundgleichung des Reinforcement Learning. Fast alles im wertbasierten RL ist eine Variation davon.

**Intuition zuerst, Mathe danach.**

Stell dir vor, du stehst in einem Zustand `s`. Du wählst eine Aktion `a`, erhältst eine Belohnung `r` und landest in einem neuen Zustand `s'`. Bellmans Frage: *Wie wertvoll war diese Aktion?*

Die Antwort hat zwei Teile:

1. Die **direkte Belohnung**, die du gerade erhalten hast: `r`.
2. Der **Zukunftswert** dort, wo du gelandet bist: was immer du von `s'` ab einsammeln kannst, wenn du dort optimal spielst.

Das war's. Der Wert einer Aktion ist *was du jetzt bekommst* plus *was du später bekommen kannst*.

### Bellmans Optimalitätsgleichung für Q

$$
Q^*(s, a) = \mathbb{E}\big[\, r + \gamma \cdot \max_{a'} Q^*(s', a') \,\big]
$$

Term für Term:

| Symbol | Bedeutung |
|--------|-----------|
| `Q*(s,a)` | Der *optimale* Q-Wert von Aktion `a` in Zustand `s`. Das Beste, was möglich ist. |
| `E[...]` | Erwartungswert (Mittel über die Zufälligkeit der Umwelt-Übergänge). |
| `r` | Direkte Belohnung nach Aktion `a` in `s`. |
| `γ` (Gamma) | Der **Diskontierungsfaktor**, zwischen 0 und 1. Wie sehr uns zukünftige Belohnungen wichtig sind. |
| `s'` | Der nächste Zustand nach `a`. |
| `max_a' Q*(s', a')` | Der Wert der **besten** Aktion im nächsten Zustand — die *gierige* Zukunft. |

**Im Klartext:** *„Ein Zustand-Aktions-Paar ist so viel wert wie die direkte Belohnung plus der diskontierte Wert des Besten, was ich als Nächstes tun kann."*

### Warum der Diskontierungsfaktor?

`γ` (typisch 0,9–0,99) hat zwei Jobs:

- **Mathematisch** hält er die Summe zukünftiger Belohnungen auch bei unendlichem Horizont endlich.
- **Konzeptionell** drückt er die Präferenz für frühere Belohnungen aus. Eine Belohnung von 1 in 100 Schritten ist `γ^100 ≈ 0,37` wert (bei γ=0,99), im Vergleich zu 1 jetzt.

### Warum das `max`?

Weil Q* das *optimale* Spiel beschreibt. Erreichst du `s'`, wirst du dort das Bestmögliche tun — nicht irgendetwas Zufälliges. Der Zukunftswert ist also der Wert der *besten* nächsten Aktion, nicht der Durchschnitt.

---

## 3 · Die Q-Tabelle

In kleinen, diskreten Umgebungen können wir Q-Werte buchstäblich in einer Tabelle ablegen:

|              | Aktion 0 | Aktion 1 | Aktion 2 | Aktion 3 |
|--------------|----------|----------|----------|----------|
| **Zustand 0** | Q(0,0)  | Q(0,1)   | Q(0,2)   | Q(0,3)   |
| **Zustand 1** | Q(1,0)  | Q(1,1)   | Q(1,2)   | Q(1,3)   |
| **Zustand 2** | Q(2,0)  | Q(2,1)   | Q(2,2)   | Q(2,3)   |
| ...          | ...      | ...      | ...      | ...      |

Jede Zelle `Q(s,a)` ist der **erwartete Return**, wenn man in `s` Aktion `a` wählt und danach optimal spielt.

**Trainingsablauf:**

1. Alle Zellen mit 0 initialisieren — der Agent weiß nichts.
2. Den Agenten mit der Umgebung interagieren lassen.
3. Nach jedem Schritt die relevante Zelle mit der **Q-Learning-Update-Regel** (Abschnitt 4) updaten.
4. Tausende Episoden wiederholen.

### Beispiel: ein 4×4-FrozenLake

FrozenLake ist ein 4×4-Gitter:

```
S F F F
F H F H
F F F H
H F F G
```

- `S` = Start (Zustand 0)
- `F` = gefrorenes Eis (sicher)
- `H` = Loch (Episodenende, Belohnung 0)
- `G` = Ziel (Belohnung +1, Episodenende)

Zustände sind 0–15 nummeriert (zeilenweise). Die 4 Aktionen sind:

| Aktion | Bedeutung |
|--------|-----------|
| 0      | ← links   |
| 1      | ↓ unten   |
| 2      | → rechts  |
| 3      | ↑ oben    |

Nach dem Training könnte die Q-Tabelle für Zustand 0 (Start) so aussehen:

| Zustand 0   | ← (0) | ↓ (1) | → (2) | ↑ (3) |
|-------------|-------|-------|-------|-------|
| Q-Wert      | 0,59  | 0,66  | 0,62  | 0,59  |

Der Agent hat gelernt, dass **runter** der beste Zug ist (höchster Q-Wert). Er wählt Aktion 1.

Das für alle 16 Zustände — und du hast eine vollständige Policy.

---

## 4 · Die Q-Learning-Update-Regel

Das ist der Algorithmus in einer Zeile:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \cdot \big[\, r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \,\big]
$$

Term für Term:

| Symbol | Bedeutung |
|--------|-----------|
| `α` (Alpha) | **Lernrate**. Wie aggressiv geupdatet wird. 0 = nie lernen; 1 = komplett überschreiben. Typisch 0,01–0,8. |
| `r + γ · max_a' Q(s',a')` | Das **TD-Target** — die neue, bessere Schätzung dessen, was Q(s,a) *sein sollte*. |
| `Q(s,a)` (das abgezogen wird) | Die aktuelle Schätzung. |
| `r + γ · max Q(s') − Q(s,a)` | Der **TD-Fehler** — die „Überraschung". Wie falsch war unsere alte Schätzung? |

**Im Klartext:** *„Verschiebe den aktuellen Q-Wert um einen kleinen Schritt in Richtung unserer besten neuen Schätzung. Die Schrittweite hängt davon ab, wie überrascht wir waren."*

### Warum heißt es „off-policy"?

Schau genau hin. Das Target nutzt `max_a' Q(s', a')` — den Wert der *gierigen* nächsten Aktion. Aber der Agent muss die gierige Aktion in `s'` nicht *genommen* haben; er könnte erkunden (Abschnitt 5).

Heißt: wir updaten Richtung *optimaler* Policy, auch wenn wir mit einer anderen (explorativen) Policy *handeln*. Behavior-Policy ≠ Target-Policy. Das ist die Definition von **off-policy**-Lernen.

Großer praktischer Vorteil: wir können alte Erfahrung wiederverwenden. In DQN ermöglicht das **Replay-Buffer** — Transitionen speichern, vielmals wiedergeben.

### Konvergenz

Q-Learning hat eine schöne theoretische Garantie: **bei genug Exploration** (jedes Zustand-Aktions-Paar unendlich oft besucht) **und einer korrekt abklingenden Lernrate** konvergiert Q gegen Q*. In der Praxis bricht man früh ab, wenn die Performance stagniert.

---

## 5 · Exploration vs. Exploitation

Hier das ewige RL-Dilemma:

- **Exploit**: nimm die Aktion mit dem aktuell höchsten Q-Wert — sichere Belohnung jetzt.
- **Explore**: nimm eine zufällige Aktion — vielleicht entdeckst du etwas Besseres.

Ein rein gieriger Agent, der an einer lokal-guten Aktion hängt, wird *nie* entdecken, dass ein anderer Pfad zu größerer Belohnung führt. Er denkt, er wisse Bescheid, sein Wissen basiert aber auf einer winzigen Stichprobe.

### ε-greedy

Die einfachste und effektivste Strategie:

```
With probability ε:      take a random action  (explore)
With probability 1 − ε:  take argmax Q(s, ·)   (exploit)
```

### ε-Abklingplan

Wir wollen früh viel Exploration (wenn Q meist Rauschen ist) und spät viel Exploitation (wenn Q vertrauenswürdig ist). Also lassen wir ε mit der Zeit abklingen:

| Phase | ε-Wert | Verhalten |
|-------|--------|-----------|
| Start | 1,0    | 100 % zufällig — reine Exploration |
| Mitte | 0,3    | 30 % zufällig, 70 % gierig |
| Ende  | 0,05   | Meist gierig, etwas Exploration |

Zwei gängige Pläne:

- **Linearer Abfall**: `ε ← ε − ε_step` pro Episode, beschnitten bei `ε_min`.
- **Exponentieller Abfall**: `ε ← ε · decay_rate` pro Episode.

!!! tip "Warum nie ganz auf ε = 0?"
    Ein kleines ε (z. B. 0,05) verhindert, dass der Agent in einer stochastischen Umgebung dauerhaft festhängt. Eine günstige Versicherung.

---

## 6 · Monte Carlo vs. Temporal Difference

Es gibt zwei grundverschiedene Wege, Werte zu schätzen:

### Monte Carlo (MC)

Warte bis zum Episodenende. Berechne den **tatsächlichen** diskontierten Gesamtreturn:

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
$$

Update Q(s,a) Richtung `G_t`. Du hast die **echte** Zukunft genutzt, keine Schätzung.

### Temporal Difference (TD)

Warte nicht. Nach **einem Schritt** update mit der nächsten Schätzung:

$$
\text{target} = r + \gamma \cdot \max_{a'} Q(s', a')
$$

Das nennt sich **Bootstrapping** — eine Schätzung mit einer anderen Schätzung updaten. Q-Learning ist eine TD-Methode (speziell TD(0), 1-Schritt).

### Vergleich

| Eigenschaft | Monte Carlo | TD (Q-Learning) |
|-------------|-------------|------------------|
| Bias        | Unverzerrt (echte Returns) | Verzerrt (Schätzungen) |
| Varianz     | Hoch (ganze Trajektorie)   | Niedriger (ein Schritt) |
| Episodenanforderung | Vollständige Episoden | Schritt-für-Schritt |
| Aufgaben ohne Ende | Geht nicht | Geht problemlos |
| Lerntempo | Langsam (1 Update/Episode) | Schnell (1 Update/Schritt) |
| Stichproben-Effizienz | Niedriger | Höher (mit Replay sehr hoch) |

In der Praxis dominieren TD-Methoden im Deep RL — niedrigere Varianz und schrittweises Lernen. Q-Learning ist das kanonische Beispiel.

---

## 7 · Hands-on: Q-Learning auf FrozenLake

Zeit, den Algorithmus zu schreiben. Wir lösen FrozenLake in etwa 50 Zeilen NumPy.

### Setup

```bash
pip install gymnasium numpy matplotlib
```

### Der vollständige Agent

```python
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
Q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.8       # learning rate
gamma = 0.95      # discount
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.005
n_episodes = 10_000

for ep in range(n_episodes):
    state, _ = env.reset()
    done = False
    while not done:
        # Epsilon-greedy action
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-Learning update
        td_target = reward + gamma * np.max(Q[next_state]) * (not done)
        td_error  = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

        state = next_state

    epsilon = max(epsilon_min, epsilon - epsilon_decay)

print("Trained Q-table:")
print(Q.reshape(4, 4, 4))  # 4x4 grid, 4 actions
```

### Was passiert Zeile für Zeile?

- `Q = np.zeros(...)` — 16 Zustände × 4 Aktionen = 16×4-Nullen-Matrix.
- Die äußere Schleife sind **Episoden**. Wir spielen 10 000 Spiele.
- Die innere Schleife sind **Schritte innerhalb einer Episode**.
- Der ε-greedy-Block wählt mit Wahrscheinlichkeit ε eine zufällige Aktion, sonst die bestbekannte.
- `td_target` ist `r + γ · max Q(s')`, multipliziert mit `(not done)`, damit Endzustände keinen Zukunftswert beitragen (nach dem Ende kommt nichts mehr).
- Die Update-Zeile ist exakt die Q-Learning-Formel aus Abschnitt 4.
- Nach jeder Episode klingt ε linear gegen 0,05 ab.

### Erwartete Ausgabe

Die meisten Zellen werden 0 sein (nie besuchte Zustände oder ohne Pfad zum Ziel). Zellen entlang des optimalen Pfads sind ungleich null und steigen zum Ziel an:

```
[[[0.59 0.66 0.62 0.59]
  [0.59 0.   0.34 0.55]
  [0.55 0.49 0.39 0.46]
  [0.34 0.   0.16 0.41]]
 ...
```

Werte nahe dem Ziel nähern sich `γ^k`, wobei `k` die Restschritte sind. Die Zelle neben dem Ziel wird einen Q-Wert nahe 1 haben.

### Evaluation

Nach dem Training die *gierige* Policy (ohne Exploration) über 100 Episoden bewerten:

```python
successes = 0
for _ in range(100):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state])
        state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
    successes += int(reward == 1)

print(f"Success rate: {successes}%")
```

Auf **nicht-rutschigem** FrozenLake solltest du **100 %** Erfolg erreichen. Die gierige Policy aus der gelernten Q-Tabelle ist optimal.

---

## 8 · Warum Q-Tabellen nicht skalieren

Der Tabellen-Ansatz funktioniert wunderbar bei FrozenLake. Er funktioniert auch bei etwas größeren Problemen wie Taxi-v3 (500 Zustände, 6 Aktionen = 3 000 Zellen). Aber er fällt sehr schnell auseinander:

### Zustandsraum-Explosion

Atari Pong hat einen Bildschirm von **210 × 160 RGB-Pixeln**. Die Zahl möglicher Bildschirme:

```
256^(210 · 160 · 3)  ≈  10^242 932
```

Mehr Zustände als Atome im beobachtbaren Universum. Tabelle? Vergiss es.

### Kontinuierliche Zustände

CartPole hat 4 kontinuierliche Zustandsvariablen (Position, Geschwindigkeit, Winkel, Winkelgeschwindigkeit). Jede ist eine reelle Zahl. Du kannst keine Tabelle mit einer reellen Zahl indizieren — es gibt unendlich viele.

Naiver Workaround: **diskretisieren** (in Bins werfen). Aber:

- Grobe Bins → schlechte Policy.
- Feine Bins → Zustandsraum-Explosion.
- Du verlierst jegliche Generalisierung zwischen ähnlichen Zuständen.

### Fluch der Dimensionalität

Die Zellzahl einer Q-Tabelle wächst **exponentiell** mit der Zahl der Zustandsdimensionen. 10 binäre Features = 1024 Zustände. 20 = 1 Mio. 30 = 1 Mrd.

### Die Lösung: Funktionsapproximation

Statt Q(s,a) in einer Tabelle abzulegen, sage es mit einem neuronalen Netz **vorher**:

```
Q(s, a) ≈ f_θ(s)[a]
```

Das Netz nimmt einen Zustand und gibt pro Aktion einen Q-Wert aus. Ähnliche Zustände bekommen ähnliche Ausgaben (Generalisierung), und das Netz hat eine *feste* Parameterzahl, unabhängig von der Zustandsraumgröße.

Das ist der Schritt von **Q-Learning** zu **Deep Q-Networks**. Thema von [Unit 3 — Deep Q-Networks](unit-03.md). Alles, was du gerade gelernt hast, gilt weiter — nur der Speichermechanismus ändert sich.

---

## 9 · Verbindung zu Godot

Warum diese ganze NumPy-Arbeit in einem Godot-Kurs?

Weil die Godot-Agenten der späteren Units **exakt dieselben Ideen** nutzen.

### Mapping

| Q-Learning-Konzept | godot-rl-agents-Pendant |
|--------------------|--------------------------|
| Zustand `s` (Integer 0–15) | Beobachtung aus `get_obs()` im `AIController` (Raycasts, Position, Geschwindigkeit) |
| Aktion `a` (Integer 0–3) | Diskrete Aktion aus `get_action_space()` |
| Q-Tabellenzelle `Q[s, a]` | Eine skalare Netz-Ausgabe für Aktion `a` |
| ε-greedy-Exploration | Eingebaut in DQN-Training in `stable_baselines3.DQN` |
| Q-Learning-Update | Geschieht in `DQN.learn()` — du rufst nur `.learn(total_timesteps=...)` |
| TD-Fehler / Loss | Der Loss, den der DQN-Optimizer minimiert |

### Beispiel: CrossTheRoad (Unit 3)

CrossTheRoad nutzt DQN. Konzeptionell:

- **Zustand** ist die lokale Sicht (Gitterzellen / Raycast-Abstände + Ego-Position).
- **Aktionen** sind diskrete Bewegungen (links / rechts / oben / unten / warten).
- **Q-Funktion** ist ein neuronales Netz: `Q(observation) → 5 Q-Werte, einer pro Aktion`.
- Training ist Q-Learning mit Replay-Buffer + Target-Netz. Die Tricks in Unit 3.

Wenn du später `model = DQN("MlpPolicy", env)` siehst, übersetze mental: *„Baue einen Funktionsapproximator für Q(s,a) und update ihn mit der Regel aus Abschnitt 4."*

---

## 10 · Viz-Checkpoint

Eine Q-Tabelle ist analytisch nützlich, aber nicht sehr *visuell*. Der Trick, der es klick machen lässt: die **gierige Policy** als Pfeil-Gitter drucken.

```python
actions = ['←', '↓', '→', '↑']
policy = np.argmax(Q, axis=1).reshape(4, 4)
for row in policy:
    print(' '.join(actions[a] for a in row))
```

### Vor dem Training (zufällige Q-Tabelle)

```
← ← ← ←
← ← ← ←
← ← ← ←
← ← ← ←
```

Alles Null → `argmax` gibt überall 0 → alle Pfeile zeigen nach links. Der Agent hat keine Präferenzen.

### Nach dem Training

```
↓ → ↓ ←
↓ ← ↓ ←
→ ↓ ↓ ←
← → → ←
```

Du kannst den Pfad mit den Augen lesen: vom Start (oben links), den Pfeilen nach unten und rechts folgen, an Löchern vorbei, zum Ziel (unten rechts). Zellen in Löchern oder unerreichbaren Zuständen zeigen Schrott — egal, weil der Agent sie unter der gierigen Policy nie besucht.

!!! tip "Sanity-Check"
    Sieht deine trainierte Policy zufällig aus, hast du wahrscheinlich zu wenige Episoden trainiert oder ε klang nie ab (Agent hat nie ausgenutzt). Drucke ε am Ende des Trainings — es sollte nahe `epsilon_min` sein.

!!! check "Fertig, wenn"
    Evaluiere die **gierige** Policy (keine Exploration) über 100 Episoden auf nicht-rutschigem FrozenLake (`is_slippery=False`): Sie sollte in **~100 %** davon das Ziel erreichen. Die Umgebung ist deterministisch, eine optimale Q-Tabelle löst sie also jedes Mal. Deutlich darunter heißt zu wenige Trainingsepisoden oder ε klang nie ab — drucke ε am Ende des Trainings; es sollte nahe `epsilon_min` liegen.

---

## 11 · Stretch Goals

Wenn du FrozenLake schnell fertig hattest, hier drei sinnvolle nächste Schritte. Alle nutzen den Code oben mit kleinen Änderungen.

### A · Rutschiges FrozenLake

Ein Flag umlegen:

```python
env = gym.make("FrozenLake-v1", is_slippery=True)
```

Jetzt ist die Umgebung **stochastisch** — drückst du „rechts", gibt's 1/3 Chance „rechts" und je 1/3 „links" oder „unten". Q-Learning funktioniert weiter (die Bellman-Gleichung umfasst einen Erwartungswert über Übergänge), aber:

- Konvergenz deutlich langsamer (mehr Episoden).
- Optimale Erfolgsquote unter 100 % — selbst ein optimaler Agent rutscht.
- Eventuell `alpha` senken (z. B. 0,1), um Rausch-Updates zu mindern.

Frage zu beantworten: *Welche Erfolgsquote erreicht dein Agent? Wie ändert sie sich bei `n_episodes = 50 000`?*

### B · Taxi-v3

Eine viel größere diskrete Umgebung:

```python
env = gym.make("Taxi-v3")
```

- 500 Zustände (Taxi-Position, Fahrgast-Ort, Ziel).
- 6 Aktionen (4 Bewegungen + Pickup + Dropoff).
- Q-Tabellengröße: 500 × 6 = 3 000 Zellen.

Derselbe Code passt — `Q = np.zeros((env.observation_space.n, env.action_space.n))` skaliert mit. Probier ~25 000 Episoden.

### C · Trainingskurve plotten

Erfolgsquote alle 100 Episoden verfolgen:

```python
import matplotlib.pyplot as plt

window = []
rates  = []
for ep in range(n_episodes):
    # ... training code ...
    window.append(int(reward == 1))
    if len(window) == 100:
        rates.append(sum(window) / 100)
        window = []

plt.plot(rates)
plt.xlabel("Training batch (x100 episodes)")
plt.ylabel("Success rate")
plt.title("FrozenLake Q-Learning")
plt.show()
```

Du solltest eine verrauschte, aber steigende Kurve sehen: anfangs nahe null, steigt, während ε abklingt und die Q-Tabelle füllt, plateau nahe 1,0 (nicht-rutschig) oder darunter (rutschig).

---

## 12 · Modellbasiertes RL — eine ganz andere Familie

Alle bisher behandelten Methoden sind **modellfrei**: der Agent interagiert mit der Umgebung und lernt direkt aus Erfahrung — kein internes Weltmodell, nur eine direkte Abbildung von Erfahrung auf Policy- oder Wertschätzungen.

**Modellbasiertes RL** geht anders vor: lerne erst ein Dynamikmodell — p(s'|s,a) — und plane darin.

Der Kernloop:

1. **Weltmodell**: gegeben s und a, sage nächsten Zustand s' und Belohnung r vorher.
2. **Planung**: tausende Trajektorien im Modell simulieren, beste Aktion wählen.
3. **Modell aktualisieren** mit echten Daten, wiederholen.

### Klassisches Beispiel: Dyna-Q (Sutton 1990)

Dyna-Q kombiniert Q-Learning mit einem gelernten Modell für „imaginäre" Übergänge. Nach jedem realen Schritt simuliert der Agent mehrere Modellschritte und nutzt sie als zusätzliche Trainingsdaten — mehr Erfahrung, ohne die echte Umwelt zu nutzen.

### Moderne Beispiele

| Algorithmus | Ansatz |
|-------------|--------|
| **MuZero** (DeepMind) | Lernt ein Modell und nutzt MCTS-Planung — beherrschte Schach, Go und Atari, ohne die Regeln zu kennen |
| **Dreamer** (Google Brain) | Lernt ein kompaktes latentes Weltmodell; trainiert Actor-Critic vollständig in der „Imagination" |
| **MBPO** | Model-Based Policy Optimization (Janner 2019) — nutzt ein gelerntes Dynamikmodell für kurze imaginierte Rollouts, um echte Erfahrung zu ergänzen; trainiert einen SAC-Agenten auf gemischten Daten |

### Modellfrei vs. modellbasiert

| Aspekt | Modellfrei (PPO, DQN, SAC) | Modellbasiert (MuZero, Dreamer) |
|--------|----------------------------|---------------------------------|
| Lernt | Policy und/oder Wertfunktion | Dynamikmodell + Policy |
| Stichproben-Effizienz | Niedriger | Viel höher (10–100×) |
| Trainings-Stabilität | Höher | Niedriger (Modellfehler) |
| Am besten für | Schnelle Simulatoren, Spielumgebungen | Teure Sims, echte Roboter |
| In diesem Kurs | Hauptansatz | Nur konzeptionelle Referenz |

### Warum es zählt — und warum wir es hier überspringen

**Stichproben-Effizienz**: modellbasierte Methoden können 10–100× effizienter sein. Wenn jede Interaktion teuer ist (echter Roboter, langsame Physik), zählt das.

**Warum nicht hier**: modellbasiertes RL ist schwerer stabil zu trainieren. Modellfehler kumulieren bei Planung — leicht abweichendes Modell, mehrstufige Rollouts driften noch weiter. Bei spielartigen Godot-Envs ist Simulation schnell und billig, die Komplexität lohnt sich nicht. PPO mit parallelen Envs liefert quasi unbegrenzte Daten — modellfrei ist hier praktisch.

Modellbasiertes RL ist zu kennen: gehst du von Games zu Robotik oder Domänen, in denen Simulation langsam/unmöglich ist, wird das essenziell.

---

## 13 · Q-Learning in Godot — CrossTheRoad neu betrachtet

CrossTheRoad (Unit 3) nutzt DQN — eine neuronale Q-Funktion. Konzeptionell IST es Q-Learning, nur mit Netz statt Tabelle. Jede Idee dieser Unit bildet exakt ab, was `stable_baselines3.DQN` intern tut.

### Mapping der Konzepte

| Q-Learning (diese Unit) | CrossTheRoad / DQN |
|-------------------------|--------------------|
| **Zustände** — Integer 0–15 | Raycast-Werte + Position (kontinuierlich, nicht tabellarisch — gleiche Idee) |
| **Aktionen** — 0, 1, 2, 3 | links, rechts, oben, unten, warten — 5 diskrete Aktionen |
| **Q-Werte** — eine Zelle pro (s, a) | DQN gibt pro Aktion einen Q-Wert aus: Q(obs, links), Q(obs, rechts), … |
| **ε-greedy** — deine `epsilon_decay`-Schleife | SB3s `exploration_fraction` + `exploration_final_eps` — derselbe Plan |
| **Bellman-Update** — Formel aus Abschnitt 4 | passiert in SB3 jeden Trainingsschritt, mit dem Replay-Buffer, den du jetzt verstehst |

### DQNs Q-Werte in SB3 inspizieren

Du kannst direkt ins trainierte Netz greifen und Q-Werte für jede Beobachtung lesen — dieselben Zahlen wie in deiner Q-Tabelle, nur als Netz-Ausgaben:

```python
from stable_baselines3 import DQN
import torch, numpy as np
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="./CrossTheRoad.x86_64", n_parallel=1, speedup=1)
model = DQN.load("logs/sb3/crosstheroad_dqn/best_model", env=env)

# Get an observation and inspect Q-values
obs, _ = env.reset()
obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    q_values = model.q_net(obs_tensor)
print("Q-values per action:", q_values.numpy())
# Output: [[-0.23, 0.87, -0.45, 0.61, -0.12]]
# Greedy action = argmax = action 1 (move right)
env.close()
```

Die fünf Zahlen sind exakt `Q(obs, links)`, `Q(obs, rechts)`, `Q(obs, oben)`, `Q(obs, unten)`, `Q(obs, warten)`. Der Agent nimmt `argmax` — dieselbe Regel wie in Abschnitt 7.

### Die zentrale Erkenntnis

Die FrozenLake-Q-Tabelle, die du gebaut hast, ist eine winzige Version dessen, was DQNs Netz für CrossTheRoad berechnet — generalisiert auf kontinuierliche Beobachtungen. Eine Tabelle kommt mit hunderten möglichen Raycast-Werten nicht klar — das Netz lernt eine komprimierte Darstellung, die zwischen ähnlichen Eingaben generalisiert. Die Update-Regel ist identisch.

In Unit 3 wirst du sehen, dass DQNs **Experience Replay** und **Target-Netz** technische Lösungen sind, die das Q-Learning-Update auf Netz-Skala stabil machen. Sie ändern den Algorithmus nicht — sie verhindern Divergenz, wenn die Q-Funktion ein nichtlinearer Approximator statt einer einfachen Tabelle ist.

---

## Was kommt als Nächstes

Du verstehst nun die Kernmaschinerie des wertbasierten RL:

- Die **Bellman-Gleichung** definiert, wie optimale Werte aussehen.
- Das **Q-Learning-Update** schiebt geschätzte Werte Schritt für Schritt Richtung Bellman-Target.
- **ε-greedy** balanciert Exploration und Exploitation.
- Tabellen funktionieren für winzige Welten; **neuronale Netze** übernehmen für große.

In der nächsten Unit ersetzt du die Tabelle durch ein neuronales Netz und triffst die Engineering-Tricks (Replay-Buffer, Target-Netz, Huber-Loss), die Deep Q-Networks in der Praxis stabil machen — und wendest sie dann auf eine Godot-Szene an.

!!! info "Selbstcheck, bevor du weitermachst"
    Kannst du diese Fragen in eigenen Worten beantworten?

    1. Was repräsentiert Q(s, a) — in einem Satz?
    2. Schreibe die Q-Learning-Update-Regel aus dem Kopf. Was ist α, was der TD-Fehler?
    3. Warum ist Q-Learning **off-policy** — und was bedeutet das konkret für die genutzten Daten?
    4. Worauf reduziert γ = 0 das Verhalten, worauf γ → 1?
    5. Warum passt FrozenLake in eine Tabelle, CrossTheRoad aber nicht?

    Wenn du alle fünf beantworten kannst — du bist bereit für DQN.

??? success "Antworten zum Selbstcheck"
    1. **Q(s, a)** ist der erwartete (abgezinste) Ertrag, wenn man in Zustand s die Aktion a wählt und danach optimal weiterhandelt — „wie gut ist diese Aktion, hier".
    2. **Q(s, a) ← Q(s, a) + α · [ r + γ · maxₐ′ Q(s′, a′) − Q(s, a) ]**. α ist die **Lernrate** (Schrittweite); die Klammer ist der **TD-Fehler** δ — die Lücke zwischen dem gebootstrappten Ziel r + γ·maxₐ′Q(s′,a′) und der aktuellen Schätzung.
    3. Das Update bootstrappt von **maxₐ′ Q(s′, a′)** (der gierigen Zielpolicy), unabhängig davon, welche Aktion die Exploration tatsächlich gewählt hat. Es lernt also die *optimale* Policy aus Daten, die eine *andere* Verhaltenspolicy erzeugt hat (ε-greedy oder sogar zufällig) — konkret kann es aus alten oder explorativen Übergängen lernen.
    4. **γ = 0** reduziert das Verhalten auf reine Gier nach der unmittelbaren Belohnung (kurzsichtig); **γ → 1** lässt den Agenten den langfristigen kumulierten Ertrag schätzen und viele Schritte vorausplanen.
    5. FrozenLake hat einen winzigen, diskreten, aufzählbaren Zustandsraum (16 Zellen), sodass jedes Q(s, a) in eine Tabelle passt. CrossTheRoad hat weit mehr (effektiv kontinuierliche) Zustände, als du aufzählen kannst, und braucht daher ein neuronales Netz, um über ungesehene Zustände zu **verallgemeinern**.

[→ Deep Q-Learning](unit-03.md)
