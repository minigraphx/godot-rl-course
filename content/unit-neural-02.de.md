# Neuronale Grundlagen 2 — Ein kleines Netzwerk lernt

[← Neuronale Grundlagen 1](unit-neural-01.md) · [Kursstartseite](index.md)

!!! info "Drei Wege, die Berechnung zu sehen"
    Entscheidungsregionen · Trainings-/Validierungskurven · Gradienten, die du mit PyTorch vergleichen kannst

Ein Neuron zeichnet eine gerade Grenze. Ein kleines Netzwerk kann diese Grenze biegen, indem es mehrere Neuronen in einer versteckten Schicht kombiniert. In dieser Unit berechnest du einen Forward Pass, verifizierst einen Gradienten und trainierst ein kleines Netzwerk, bis der Loss sinkt.

> **Frage für beide Pfade:** Was ändert sich, wenn ein Netzwerk eine versteckte Schicht hat und aus Beispielen lernt?

Die Lernschleife ist weiterhin **Predict → Play → Build → Break → Explain**. Schließe einen Hauptpfad ab und verbringe dann zehn Minuten mit dem anderen Pfad.

---

## 1 · Warum ein Neuron nicht reicht

Ein einzelnes Neuron kann Daten nur mit einer Linie trennen. Das reicht für manche Probleme, aber nicht für Muster wie XOR:

| Input 1 | Input 2 | Ziel |
|---:|---:|---:|
| 0 | 0 | niedrig |
| 0 | 1 | hoch |
| 1 | 0 | hoch |
| 1 | 1 | niedrig |

Keine einzelne gerade Grenze kann die beiden diagonalen Ecken als dieselbe Klasse markieren. Eine versteckte Schicht löst das, indem sie mehrere Zwischenmerkmale erzeugt und sie dann zur endgültigen Ausgabe kombiniert.

---

## 2 · Zwei Schichten, ein Forward Pass

Der Research-Pfad nutzt ein `2 → 4 → 1`-Netzwerk:

```text
two inputs → four hidden neurons → one output
```

Der Game-Pfad nutzt dieselbe Idee mit einem `4 → 4 → 2`-Netzwerk:

```text
gem direction + hazard offset → four hidden neurons → movement x/y
```

Die getestete Python-Implementierung liegt in
`examples/neural_foundations/research/tiny_mlp.py`. Die getestete Godot-
Implementierung liegt in `examples/neural_foundations/game/shared/tiny_mlp.gd`.

Führe die Verifikation aus:

```bash
conda activate godot_env
python -m unittest examples.neural_foundations.research.tests.test_tiny_mlp -v

godot --headless \
  --path examples/neural_foundations/game \
  --script res://test/test_tiny_mlp.gd
```

Beide Tests nutzen feste Gewichte, damit du jeden Zwischenwert vergleichen kannst.

---

## 3 · Loss misst die Abweichung

Training braucht eine Zahl, die sagt, wie falsch die Ausgabe ist. Diese Unit nutzt den mittleren quadratischen Fehler:

$$
\text{loss} = \frac{1}{n}\sum_i(\hat{y}_i - y_i)^2
$$

Ein kleiner Loss bedeutet, dass Vorhersagen zu den Zielbeispielen passen. Ein großer Loss bedeutet, dass das Netzwerk noch das falsche Verhalten produziert.

Führe die Research-Trainingsdemo aus:

```bash
python examples/neural_foundations/research/train_tiny_mlp.py \
  --save artifacts/neural-foundations/tiny-mlp.png
```

Das Skript gibt Trainings-/Validierungs-Loss, Genauigkeit und Seed-Vergleiche aus und speichert dann Entscheidungsregionen, Loss-Kurven, Gradientenzeichen und Seed-Nachweise.

---

## 4 · Ein Gradient von Hand

Ein Gradient sagt einem Gewicht, in welche Richtung es sich bewegen soll, um den Loss zu reduzieren. Du musst dir noch keine lange Herleitung merken. Die wichtige Kette ist:

```text
weight → hidden value → output → loss
```

Die Test-Suite vergleicht den manuellen NumPy-Gradienten mit PyTorch autograd
innerhalb von `1e-5`. Das gibt dir eine vertrauenswürdige Referenz und hält die
lernerorientierte Implementierung trotzdem klein und inspizierbar.

!!! success "Checkpoint"
    Wenn der manuelle Gradient und der PyTorch-Gradient übereinstimmen, ist deine Buchführung für dieses Beispiel korrekt.

---

## 5 · Backpropagation als Buchführung

Backpropagation ist keine Magie. Es ist sorgfältige Buchführung:

1. Forward Pass ausführen und die Zwischenwerte speichern;
2. den Loss messen;
3. rückwärts von der Ausgabe zur versteckten Schicht gehen;
4. berechnen, wie stark jedes Gewicht zum Fehler beigetragen hat;
5. ein kleines Update mit der Lernrate anwenden.

Die Lernrate zählt. In der festen Research-Demo lernt `0.05` zuverlässig.
Ein viel größerer Wert kann das Netzwerk in ein schlechtes Plateau drücken, in dem der Loss nicht mehr sinkt.

---

## 6 · Wähle deinen Pfad

| | Research-Pfad | Game-Pfad |
|---|---|---|
| Inputs | Zwei normalisierte Merkmale | Edelsteinrichtung und Gefahren-Offset |
| Output | Ein Klassen-Score | Zwei Bewegungswerte |
| Nachweis | Trainings-/Validierungs-Loss, Seed-Vergleich, Entscheidungsregionen | Input-Linien, Output-Balken, Ziel- versus Vorhersage-Pfeile |
| Tool | NumPy, PyTorch-Check, Matplotlib | Standard Godot 4 + GDScript |

Wähle einen Hauptpfad:

- **Research:** trainiere den nichtlinearen Klassifikator und vergleiche Gradienten mit PyTorch.
- **Spieleentwicklung:** trainiere den Arena-Sammler aus sichtbaren Lehrerbeispielen.

---

## 7 · Research-Pfad — nichtlineare Klassifikation

Starte mit dem Skript:

```bash
python examples/neural_foundations/research/train_tiny_mlp.py
```

Bevor du es ausführst, sage voraus, was die Loss-Kurve tun sollte. Führe es dann aus und beantworte:

1. Ist der finale Loss unter den Anfangs-Loss gesunken?
2. Hat die Genauigkeit mindestens 75 % erreicht?
3. Erzählen Trainings- und Validierungskurve dieselbe Geschichte?
4. Welcher Seed endete mit dem niedrigsten Validierungs-Loss?

??? success "Lösungsschlüssel"
    Mit dem Standard-Seed und der Standard-Lernrate sinkt der Loss stark und die festen Beispiele erreichen 100 % Genauigkeit. Die Validierungskurve kann höher bleiben oder sich anders bewegen, weil sie ausgeklammerte Beispiele enthält. Mit den Standard-Vergleichs-Seeds endet Seed `1` mit dem niedrigsten Validierungs-Loss. Wenn du die Lernrate zu weit erhöhst, kann das Modell aufhören, sich zu verbessern, obwohl der Code korrekt ist.

---

## 8 · Game-Pfad — Arena-Sammler

Der Game-Pfad nutzt dieselbe Hidden-Layer-Idee für einen kleinen Sammler:

- Inputs beschreiben, wo der Edelstein ist und ob die Gefahr bedrohlich ist;
- ein skriptierter Lehrer liefert Ziel-Bewegungsbeispiele;
- das Netzwerk sagt die Bewegung voraus;
- grüne und rote Linien zeigen Edelstein- und Gefahren-Inputs;
- Output-Balken zeigen Ziel- und vorhergesagte Bewegungskomponenten;
- Pfeile vergleichen Ziel- und vorhergesagte Bewegung;
- Zähler zeigen Schritte, Sammlungen, Gefahren-Treffer und Replay-Status.

Führe die aktuelle Szene aus:

```bash
godot --path examples/neural_foundations/game \
  res://unit_02_collector/unit_02_collector.tscn
```

Baue die Szene zuerst aus Primitiven. Halte den Lehrer sichtbar und einfach:

!!! warning "Pseudocode"
    ```text
    if hazard is close: move away from hazard
    else: move toward gem
    ```

Der Lehrer ist nicht die finale KI. Er existiert, um Beispiele zu erzeugen, damit du überwachtes Lernen beobachten kannst, bevor Belohnungslernen beginnt.

---

## 9 · Training absichtlich kaputtmachen

Probiere jeweils eine Änderung:

1. setze die Lernrate zu hoch und beobachte, wie der Loss stagniert oder springt;
2. reduziere die versteckte Schicht auf ein Neuron und untersuche Underfitting;
3. entferne Beispiele nahe der Gefahr und suche nach dem blinden Fleck;
4. füge Sensorrauschen hinzu und vergleiche Trainings-Loss mit Gameplay-Verhalten.

Notiere für jeden Bruch das sichtbare Symptom und die Metrik, die es bestätigt.

---

## 10 · Training versus Inference

Während des Trainings ändert das Netzwerk seine Gewichte nach dem Sehen von Beispielen. Während der Inference bleiben die Gewichte fest und das Netzwerk führt nur den Forward Pass aus.

Diese Unterscheidung zählt für den Rest des Kurses:

- **Training:** langsamer, nutzt Loss oder Belohnung, aktualisiert Gewichte;
- **Inference:** schnell, nutzt feste Gewichte, wählt Aktionen.

Der finale Racer nutzt dieselbe Aufteilung: Python trainiert die Policy, dann führt Godot die trainierte Policy für Inference aus.

---

## 11 · Stretch Goals

- Plotte drei Lernraten in derselben Loss-Grafik.
- Vergleiche fünf Seeds und erkläre den instabilsten Lauf.
- Speichere die trainierten Tiny-Network-Gewichte und lade sie wieder.
- Füge dem Game-Pfad eine zweite Gefahrenposition hinzu und teste, ob sich das Verhalten noch verbessert.

---

## Was kommt als Nächstes

Du hast jetzt die Bausteine innerhalb eines Policy-Netzwerks: Inputs, versteckte Aktivierungen, Loss, Gradienten und Gewichts-Updates. In [RL Essentials](unit-01.md) verbindest du dieses Netzwerk mit der Reinforcement-Learning-Schleife: Beobachtungen, Aktionen, Belohnungen, Episoden und Exploration.

[← Neuronale Grundlagen 1](unit-neural-01.md) · [Kursstartseite](index.md) · [→ RL Essentials](unit-01.md)
