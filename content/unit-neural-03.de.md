# Neuronale Grundlagen 3 — Aus Belohnung lernen

[← RL Essentials](unit-01.md) · [Kursstartseite](index.md)

!!! info "Drei Wege, die Berechnung zu sehen"
    Trajektorie in der Punktroboter-Welt · diskontierte Returns aus einer Episode · Fünf-Seed-Evaluationsmetriken

In Grundlagen 1 und 2 lernte das Netzwerk aus beschrifteten Antworten. In dieser Unit lernt das Netzwerk aus Belohnung: Es sampelt Aktionen, erhält Konsequenzen und aktualisiert die Policy in Richtung Trajektorien mit höherem Return.

> **Frage für beide Pfade:** Wie verbessert sich eine Policy, wenn niemand die korrekte Aktion vorgibt?

---

## 1 · Von beschrifteten Beispielen zur Belohnung

Überwachtes Lernen kann eine Vorhersage mit einem bekannten Ziel vergleichen. Reinforcement Learning kann das meist nicht. Der Punktroboter sieht nur:

- seine aktuelle Beobachtung;
- die Aktion, die er gesampelt hat;
- die Belohnung nach dieser Aktion;
- ob die Episode endete.

Das reicht zum Lernen, aber das Signal ist noisiger. Eine schlechte Aktion kann trotzdem in einer erfolgreichen Episode vorkommen, und eine gute Aktion in einer gescheiterten.

---

## 2 · Policy, Trajektorie und Return

Eine **Policy** bildet Beobachtungen auf Aktionswahrscheinlichkeiten ab. Eine **Trajektorie** ist eine Episode aus Beobachtungen, Aktionen und Belohnungen. Der **Return** ist die diskontierte Summe zukünftiger Belohnungen:

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots
$$

Der getestete Helper in `examples/neural_foundations/research/reinforce.py`
berechnet diese Returns direkt. Für Belohnungen `[1.0, 2.0, 3.0]` und
`gamma = 0.5` sind die Returns `[2.75, 3.5, 3.0]`.

---

## 3 · Warum Aktionen beim Training gesampelt werden müssen

Während der Inference ist die wahrscheinlichste Aktion oft in Ordnung. Während des Trainings muss die Policy Aktionen sampeln, damit sie bessere Trajektorien entdecken kann. REINFORCE
behält die Log-Wahrscheinlichkeit jeder gesampelten Aktion, wartet auf den Episoden-Return und
schiebt die Policy dann zu Aktionen, die in Episoden mit hohem Return vorkamen.

!!! warning "Pseudocode"
    ```text
    collect one episode with sampled actions
    compute discounted returns
    increase log probability for actions with high return
    decrease it for actions with low return
    ```

---

## 4 · Wähle deinen Pfad

Der Research-Pfad nutzt eine kleine Python-Punktroboter-Umgebung und eine handgeschriebene REINFORCE-Schleife. Der Game-Pfad ist verfügbar und nutzt den Arcade-Racer, PPO-Training und native Godot-Inference.

Schließe zuerst den Research-Pfad ab, wenn du jeden Tensor im Policy-Gradient-Update sehen willst, bevor du zum größeren Spielebeispiel wechselst.

---

## 5 · Research-Pfad — 2D-Punktroboter

Der Punktroboter lebt in einem quadratischen Raum. Seine Beobachtung enthält drei normalisierte Wand-Ray-Distanzen und die normalisierte Peilung zum Ziel. Seine Aktionen sind:

- vorwärts-links;
- vorwärts;
- vorwärts-rechts.

Führe die deterministischen Tests aus:

```bash
conda activate godot_env
python -m unittest examples.neural_foundations.research.tests.test_point_robot -v
```

Bevor du trainierst, sage voraus, was die drei Ray-Werte tun sollten, wenn der Roboter zur Wand dreht. Untersuche dann `PointRobotEnv.observation()` und verifiziere, dass die Zahlen zur Geometrie passen.

---

## 6 · REINFORCE bauen

Die Implementierung in `examples/neural_foundations/research/reinforce.py` hat
drei kleine Teile:

- `discounted_returns()` wandelt Episoden-Belohnungen in Trainingsziele um;
- `train_episode()` sampelt eine Trajektorie und aktualisiert die Policy;
- `evaluate()` führt feste Seeds mit gierigen Aktionen aus und meldet Metriken.

Führe den Return-Test aus:

```bash
python -m unittest examples.neural_foundations.research.tests.test_returns -v
```

Verfolge dann eine Episode von Hand: notiere drei Belohnungen, berechne ihre diskontierten Returns und vergleiche sie mit dem Helper.

---

## 7 · Fünf Seeds evaluieren

Erzeuge die Referenz-Zusammenfassung:

```bash
python -m examples.neural_foundations.research.reinforce \
  --save examples/neural_foundations/research/results/reinforce_five_seed.json
```

Die gespeicherte JSON-Datei meldet mittleren Return, Return-Standardabweichung, Erfolgsrate und mittlere Episodenlänge über Seeds `0` bis `4`. Behandle die Standardabweichung als Warnlabel: Ein glücklicher Seed beweist keine stabile Policy.

---

## 8 · Sensoren und Reward Shaping ablatieren

Kaputt mache jeweils einen Teil und führe die Fünf-Seed-Evaluation erneut aus:

1. entferne die Ziel-Peilung aus der Beobachtung;
2. reduziere die Kollisionsstrafe;
3. entferne die Fortschrittsbelohnung und behalte nur Erfolg oder Kollision;
4. verkürze `max_steps`.

Notiere für jede Ablation sowohl das sichtbare Symptom als auch die Metrik, die es erfasst hat. Du suchst die Verbindung zwischen Reward-Design, Sensoren und dem Verhalten, das die Policy entdeckt.

---

## 9 · Game-Pfad — den Arcade-Racer bauen

Der Game-Pfad nutzt `examples/neural_foundations/game/unit_03_racer/`. Die Szene
ist absichtlich primitiv: eine rechteckige Strecke, sichtbare Rays, Checkpoint-Markierungen
und ein Dreieck-Auto.

!!! note "Aktueller nativer Scope"
    Diese Kurskopie importiert den macOS-arm64-nativen Runner aus dem lokalen
    `godot-native-rl`-Checkout. Windows- und Linux-Binaries bleiben außerhalb des Scopes, bis
    die Multi-Plattform-Version verfügbar ist.

---

## 10 · Beobachtungen und Aktionen definieren

Die Racer-Beobachtung enthält drei normalisierte Ray-Distanzen, Heading-Fehler zum
nächsten Checkpoint, normalisierte Geschwindigkeit und Checkpoint-Fortschritt. Der Aktionsraum
ist ein kontinuierlicher Zwei-Werte-Head:

```text
drive = [steering, throttle]
```

Führe die deterministischen Mathe-Checks aus:

```bash
godot --headless \
  --path examples/neural_foundations/game \
  --script res://test/test_racer_math.gd
```

---

## 11 · Belohnung und Episodengrenzen entwerfen

Die Belohnung kombiniert:

- eine kleine Schrittstrafe;
- geordnete-Checkpoint-Belohnung;
- Kollisionsstrafe;
- Immobilitäts-Timeout.

Die wichtige Gewohnheit ist, Reward-Mathematik vor dem Training zu testen. Ein falsches Vorzeichen in einem Belohnungsterm kann stundenlang wie ein Algorithmusproblem aussehen.

---

## 12 · Mit PPO trainieren

Die Trainingsszene nutzt `NcnnSync` im Trainingsmodus und dasselbe Socket-Protokoll
wie `godot-rl`. Starte den Referenzbefehl:

```bash
conda activate godot_env
./scripts/train-foundations-racer.sh
```

Der Befehl speichert einen Stable-Baselines3-Checkpoint und exportiert ONNX unter
`examples/neural_foundations/game/unit_03_racer/models/`.

---

## 13 · Den ONNX-Graphen inspizieren

Der exportierte Graph hat Input `obs` und Output `out0`. Bevor du ihn konvertierst, prüfe,
dass die Shapes zu den sechs Beobachtungswerten und zwei Drive-Outputs passen.

---

## 14 · PyTorch, ONNX und ncnn verifizieren

Führe den Paritäts-Verifier aus:

```bash
conda activate godot_env
python scripts/verify_racer_policy.py
```

Der Verifier nutzt zwanzig feste Beobachtungen. PyTorch und ONNX müssen innerhalb von
`1e-5` übereinstimmen; ONNX und ncnn innerhalb von `1e-2`.

---

## 15 · Native Inference in Godot ausführen

Die Evaluationsszene nutzt `NcnnSync` im nativen Inference-Modus und zeigt den Agenten
auf die konvertierten ncnn-Dateien:

```bash
godot --path examples/neural_foundations/game \
  res://unit_03_racer/racer_eval.tscn
```

Beobachte dieselben Rays und Checkpoint-Markierungen wie beim Training. Native
Inference sollte sich wie die Python-Policy bei den festen Starts verhalten.

---

## 16 · Belohnungs- und Sensorfehler diagnostizieren

Wenn der Racer scheitert, inspiziere in dieser Reihenfolge:

1. Ray-Normalisierung;
2. Heading-Fehler-Vorzeichen;
3. Lenk- und Gas-Klemmung;
4. Checkpoint-Reihenfolge;
5. Kollisions- und Timeout-Bedingungen.

Die deterministischen Tests decken diese Teile ab, damit du Umgebungsbugs von Trainingsvarianz trennen kannst.

---

## 17 · Die beiden Pfade vergleichen

Der Research-Pfad machte das Policy-Gradient-Update sichtbar. Der Game-Pfad behält dieselbe Idee, fügt aber Godot-Timing, native Inference und Modell-Export hinzu. In beiden Fällen ist die Belohnung der Lehrer.

---

## 18 · Stretch Goals

- Füge eine zweite Strecke hinzu und vergleiche die Erfolgsrate.
- Plotte Checkpoint-Erreichungszeit über das Training.
- Ändere die Ray-Winkel und führe Paritäts-Checks erneut aus.
- Füge einen Belohnungsterm für sanftes Lenken hinzu und teste, ob sich das Verhalten ändert.

---

## Was kommt als Nächstes

Du kannst Belohnungslernen jetzt mit dem Algorithmus-Vokabular im Deep Dive verbinden:
Returns, Bootstrapping, Exploration, On-Policy-Training und Actor-Critic-Methoden.

[← RL Essentials](unit-01.md) · [Kursstartseite](index.md) · [→ RL Foundations Deep Dive](unit-rl-foundations-deep.md)
