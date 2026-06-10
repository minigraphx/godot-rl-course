# Neuronale Grundlagen 1 — Ein Neuron, eine Entscheidung

[← Einheit 0](unit-00.md) · [Kursstartseite](index.md)

!!! info "Drei Wege, die Berechnung zu sehen"
    Laufende Visualisierung · aktuelle Zahlen · Code, den du geschrieben hast

Eine trainierte Policy kann rätselhaft wirken, aber jede Entscheidung beginnt mit
gewöhnlicher Arithmetik. In dieser Einheit baust du ein Neuron, beobachtest jede
Größe beim Wechsel und nutzt dieselbe Berechnung für eine Forschungsklassifikation
und einen Sprung-Trigger.

> **Frage für beide Pfade:** Wie können zwei Messwerte zu einer sichtbaren
> Entscheidung werden?

Die Lernschleife lautet **Predict → Play → Build → Break → Explain**. Schließe
einen Hauptpfad ab und verbringe dann zehn Minuten mit dem anderen Pfad.

---

## 1 · Vorhersagen, bevor du startest

Beginne mit diesem **Neuron mit festen Zahlen**. Die Namen erklären zuerst, was
die Zahlen bedeuten; die mathematischen Kürzel kommen danach:

| Benannte Eingabe | Wert | Gewicht | Beitrag |
|---|---:|---:|---:|
| Geschwindigkeit | 0.50 | +0.80 | +0.40 |
| Nähe zur Kante | 0.25 | +1.20 | +0.30 |
| Bias | — | — | -0.50 |

Bevor du Python oder Godot nutzt, notiere:

1. die gewichtete Summe;
2. ob `sigmoid` einen Wert über `0.5` liefert;
3. welche Eingabe am meisten zur Entscheidung beiträgt.

Addiere zuerst die benannten Beiträge und den Bias:

$$
\text{Summe} =
(\text{Geschwindigkeit}\times\text{Geschwindigkeitsgewicht}) +
(\text{Nähe}\times\text{Nähegewicht}) +
\text{Bias}
$$

$$
\text{Summe} = (0.5)(0.8) + (0.25)(1.2) - 0.5 = 0.2
$$

Die Aktivierung liefert
\(\operatorname{sigmoid}(0.2) \approx 0.550\). Weil `0.550 > 0.5`, **feuert**
das Neuron und die Spielaktion lautet `SPRINGEN`.

In der Mathematik wird Eingabe oft zu \(x\), Gewicht zu \(w\), Bias zu \(b\)
und die Summe zu \(z\) verkürzt. Dieselbe Rechnung kann später also als
\(z=w_1x_1+w_2x_2+b\) erscheinen. Das sind nur Abkürzungen, keine anderen
Werte. Gewicht wird in diesem Kurs immer mit einem kleinen \(w\) abgekürzt.

??? success "Antwortschlüssel"
    Die Summe ist `0.2`, daher beträgt die Sigmoid-Ausgabe ungefähr `0.550` und
    das Neuron feuert. Geschwindigkeit trägt `+0.40` bei, Nähe `+0.30`, und der
    Bias zieht `0.50` ab.

**Sichtbare Prüfung:** Die automatisierten Beispiele verwenden dieselben Zahlen:

!!! note "Im Repo-Stammverzeichnis ausführen"
    Diese Befehle setzen voraus, dass dein Terminal im [Stammverzeichnis des Kurs-Repos](setup.md#course-repo) liegt und `godot` auf deinem PATH ist — siehe [Godot auf der Kommandozeile](setup.md#godot-cli).

```bash
conda activate godot_env
python -m examples.neural_foundations.research.tests.test_neuron

godot --headless \
  --path examples/neural_foundations/game \
  --script res://test/test_tiny_neuron.gd
```

Beide Befehle geben den Abschnitt-1-Walkthrough aus (`Summe = 0.2`,
`sigmoid(Summe) ≈ 0.550`) und enden mit `OK`. Der Godot-Lauf zeigt zusätzlich
die Live-Labels der Jumper-Demo (Geschwindigkeit, Nähe, Summe, Ausgabe und
`WARTEN`/`SPRINGEN`).

Beide Tests rufen den Forward Pass auf, den du als Nächstes untersuchst.

---

## 2 · Gewichtete Eingaben und Bias

Ein Neuron gibt jeder normalisierten Eingabe ein **Gewicht**:

- ein positives Gewicht lässt größere Eingabewerte die Ausgabe nach oben drücken;
- ein negatives Gewicht lässt sie nach unten drücken;
- ein größerer Betrag gibt dieser Eingabe mehr Einfluss;
- der Bias verschiebt die Entscheidung, bevor irgendein Eingabebeitrag wirkt.

Baue den Forward Pass, bevor du eine Visualisierung änderst. Öffne in deinem
Hauptpfad die passende Datei und tippe die Schleife selbst:

- Research: `examples/neural_foundations/research/neuron.py`
- Game development: `examples/neural_foundations/game/shared/tiny_neuron.gd`

Die gemeinsame Implementierung ist bewusst klein gehalten:

```python
def neuron_output(inputs, weights, bias, activation):
    weighted_sum = sum(
        value * weight for value, weight in zip(inputs, weights)
    )
    return activate(weighted_sum + bias, activation)
```

Führe die Tests aus, nachdem du die Schleife fertig hast. Die Research- und
Godot-Versionen sollten für das handberechnete Beispiel oben denselben Wert
zurückgeben.

### Warum normalisieren?

Angenommen, die Temperatur wird nach Normalisierung als `0.7` erfasst, während der
Druck versehentlich als `80` stehen bleibt. Schon ein kleines Druckgewicht kann
die Berechnung dominieren:

| Merkmal | Eingabe | Gewicht | Beitrag |
|---|---:|---:|---:|
| Temperatur | 0.70 | +1.20 | +0.84 |
| Rohdruck | 80.00 | -0.05 | -4.00 |

Die Ausgabe würde vor allem die Einheiten beschreiben, in denen Druck gemessen
wird — nicht die Beziehung, die du modellieren wolltest. Beide visuellen Beispiele
halten Eingaben zwischen `0` und `1`, damit ihre Beiträge vergleichbar sind.

**Sichtbare Prüfung:** Im Research-Plot spannt jede Achse `0–1`. In der Godot-Szene
zeigen die Gesundheitsleiste und die Distanz-Overlay die normalisierten Werte, bevor
sie das Neuron erreichen.

---

## 3 · Aktivierungsfunktionen

Die Summe \(z\) kann jede Zahl sein. Eine **Aktivierungsfunktion**
wandelt sie in die für die Entscheidung benötigte Form.

| Activation | Ausgabe | Nützliche sichtbare Interpretation |
|---|---|---|
| Step | `0` oder `1` | Harte Klassenumstellung |
| Sigmoid | zwischen `0` und `1` | Konfidenzähnlicher Score |
| Tanh | zwischen `-1` und `1` | Richtung oder vorzeichenbehaftete Tendenz |

Nahe der Grenze erzählen die Funktionen unterschiedliche Geschichten:

| \(z\) | Step | Sigmoid | Tanh |
|---:|---:|---:|---:|
| -0.10 | 0 | 0.475 | -0.100 |
| 0.00 | 1 | 0.500 | 0.000 |
| +0.10 | 1 | 0.525 | +0.100 |

Die Entscheidungsgrenze liegt dort, wo \(z = 0\):

$$
w_1x_1 + w_2x_2 + b = 0
$$

Ein Gewichtswechsel dreht diese Linie. Ein Bias-Wechsel verschiebt sie, ohne sie
zu drehen.

**Sichtbare Prüfung:** Wähle `step`, `sigmoid` und `tanh` im Research-Plot. Die
schwarze Grenze bleibt bei \(z=0\), während sich die angezeigte Ausgabe für die
Stern-Sonde ändert. In Godot löst `sigmoid(z) > 0.5` das Ereignis `SPRINGEN` aus.

---

## 4 · Wähle deinen Pfad

Die Gleichung ist gemeinsam; die Evidenz unterscheidet sich.

| | Research-Pfad | Game-Pfad |
|---|---|---|
| Eingaben | Normalisierte Temperatur und Druck | Normalisierte Geschwindigkeit und Nähe zur Kante |
| Ausgabe | Sichere oder unsichere Klasse | Warten oder Sprung-Ereignis auslösen |
| Hauptvisualisierung | Farbige Punkte und Entscheidungsgrenze | Eingaberegler, sichtbarer Bogen, Klippe und Lava |
| Evidenz | Genauigkeit und falsch klassifizierte Punkte | Erwartetes versus tatsächliches Verhalten |
| Werkzeug | Python + Matplotlib | Standard Godot 4 + GDScript |

Wähle einen Hauptpfad:

- **Research:** Schließe Abschnitt 5 ab und sieh dir den Godot-Vergleich einmal an.
- **Game development:** Schließe Abschnitt 6 ab und sieh dir den Plot-Vergleich einmal an.

In dieser Einheit brauchst du keine native Extension, C#, kein Trainingsframework
und keine vorherige Machine-Learning-Bibliothek.

---

## 5 · Research-Pfad — sichtbare Entscheidungsgrenze

**Forschungsfrage:** Kann ein Neuron sichere und unsichere experimentelle
Bedingungen trennen?

Starte den interaktiven Plot aus dem Repository-Root:

```bash
conda activate godot_env
python examples/neural_foundations/research/plot_neuron.py
```

Der Plot liefert synchronisierte Evidenz:

- Hintergrund und Punktfarben zeigen Vorhersagen;
- die schwarze Linie zeigt \(z=0\);
- rote Ringe zeigen falsche Vorhersagen;
- der Stern markiert die aktuelle numerische Sonde;
- das Seitenpanel zeigt Beiträge, Bias, gewichtete Summe, Aktivierung und
  Genauigkeit;
- Schieberegler steuern `w₁`, `w₂` und Bias.

Bei der Ausgangssonde \([0.65, 0.35]\):

$$
z = (0.65)(1.2) + (0.35)(-0.9) - 0.1 = 0.365
$$

Mit einer Step-Aktivierung ist die Vorhersage Klasse `1` (unsicher).

### Experiment 1 — ein Gewicht umkehren

**Hypothese zuerst:** Sage vorher, welche farbige Region sich ändert, wenn `w₁`
von `+1.2` auf `-1.2` wechselt. Bewege dann nur diesen Schieberegler und notiere
die Genauigkeit vorher und nachher.

| Parameter | Vorher | Nachher |
|---|---:|---:|
| `w₁` | +1.2 | -1.2 |
| `w₂` | -0.9 | -0.9 |
| Bias | -0.1 | -0.1 |

??? success "Antwortschlüssel"
    Höhere Temperatur drückte die Punktzahl ursprünglich Richtung unsicher. Nach
    der Vorzeichenumkehr drückt sie Richtung sicher. Die Grenze ändert die
    Orientierung, viele Hochtemperatur-Punkte wechseln die Klasse, und die
    Genauigkeit für diesen Datensatz sinkt.

### Experiment 2 — Normalisierung entfernen

**Hypothese zuerst:** Sage vorher, was passiert, wenn Druckwerte 100-mal größer
werden, während die Gewichte unverändert bleiben. Ändere in `plot_neuron.py`
vorübergehend die Vorhersageeingabe:

```python
scaled_features = FEATURES.copy()
scaled_features[:, 1] *= 100.0
```

Übergib `scaled_features` an `predict`, führe einmal aus, und stelle dann die
normalisierten Merkmale wieder her.

??? success "Antwortschlüssel"
    Der Druckbeitrag wird etwa 100-mal größer und überwältigt Temperatur und Bias.
    Die meisten Entscheidungen folgen allein dem Druck. Das ist kein Beleg dafür,
    dass Druck wissenschaftlich wichtiger ist; es ist ein Skalierungsfehler.

### Experiment 3 — Aktivierungen an der Grenze vergleichen

Setze die Sonde nahe \(z=0\), wechsle dann zwischen `step`, `sigmoid` und `tanh`,
ohne einen Parameter zu ändern. Notiere die angezeigte Ausgabe.

??? success "Antwortschlüssel"
    Step springt direkt zwischen Klassen. Sigmoid ändert sich glatt um `0.5`; tanh
    ändert sich glatt um `0`. Die Entscheidungsschwelle kann gleich bleiben, auch
    wenn sich die numerischen Ausgaben unterscheiden.

### Research-Evidenz

Speichere diese kleine Tabelle in deinen Notizen:

| Lauf | Hypothese | Geänderter Parameter | Genauigkeit | Grenz-Evidenz |
|---|---|---|---:|---|
| Baseline | — | — | | |
| Vorzeichen des Gewichts | | nur `w₁` | | |
| Skalierungsfehler | | nur Druck | | |
| Aktivierung | | nur Aktivierung | | |

Änderungen jeweils nur einer Variable machen deine Erklärung überprüfbar.

---

## 6 · Game-Pfad — Sprung-Timing an der Klippe

**Game-AI-Frage:** Kann ein Neuron Geschwindigkeit und Distanz kombinieren, um
einen Sprung im richtigen Moment auszulösen?

Öffne das eigenständige Standard-Godot-Projekt:

```bash
godot --editor --path examples/neural_foundations/game
```

Öffne `unit_01_jumper/unit_01_jumper.tscn` und drücke **F6**.

Die Szene startet im **Labor-Modus**. Während du untersuchst, bewegt sich nichts:

- `Speed input` regelt die gedachte Laufgeschwindigkeit;
- `Remaining distance` regelt die Restdistanz zur Klippe;
- `Speed weight`, `Closeness weight` und `Bias` sind deine Parameter;
- jeder Beitrag, die Summe und die Sigmoid-Ausgabe bleiben sichtbar;
- `WAIT` bedeutet: Ausgabe höchstens `0.5`;
- `JUMP` bedeutet: Ausgabe größer als `0.5`.

Die Distanz wird in **Nähe** umgerechnet:

$$
\text{Nähe}=1-\text{Restdistanz}
$$

Damit sind beide positiven Gewichte intuitiv: Mehr Geschwindigkeit drängt zu
einem früheren Sprung, mehr Nähe drängt zu einem Sprung jetzt.

### Experiment 1 — das Distanzsignal sinnvoll machen

Setze die Geschwindigkeit auf `0.30`. Bewege die Restdistanz von `0.80` in
Richtung `0.10`. Ändere nur das `Closeness weight`, bis das Neuron weit
entfernt wartet und nahe an der Kante feuert.

??? success "Das solltest du entdecken"
    Ein positives Nähegewicht lässt den Beitrag wachsen, wenn die Klippe näher
    kommt. Ein negatives Gewicht erzeugt das gefährliche Gegenteil.

### Experiment 2 — Geschwindigkeit verändert das Timing

Halte die Restdistanz bei `0.45`. Vergleiche Geschwindigkeit `0.30` und `0.90`.
Ändere nur das `Speed weight`, bis der schnelle Läufer feuert, während der
langsame noch wartet.

??? success "Das solltest du entdecken"
    Ein positiver Geschwindigkeitsbeitrag bringt den schnellen Fall früher über
    die Schwelle. Eine feste Regel wie `distance < 0.2` kann das nicht.

### Experiment 3 — alle Entscheidungen mit Bias verschieben

Verschiebe mit dem Bias den allgemeinen Auslösepunkt. Zu viel positiver Bias
lässt alle Situationen springen. Zu viel negativer Bias lässt alle warten.
Stelle ihn ein, bis **3 / 3 cases pass** erscheint:

| Fall | Geschwindigkeit | Restdistanz | Erwartet |
|---|---:|---:|---|
| Langsam und weit | 0.30 | 0.80 | WAIT |
| Schnell und mittel | 0.90 | 0.45 | JUMP |
| Langsam und nah | 0.30 | 0.10 | JUMP |

### Game-Development-Evidenz

Notiere die Parameter, die alle drei Fälle bestehen:

| Geschwindigkeitsgewicht | Nähegewicht | Bias | Bestandene Fälle |
|---:|---:|---:|---:|
| | | | / 3 |

Drücke danach mehrmals **Test run**. Der Läufer erhält zufällig eine langsame,
mittlere oder schnelle Geschwindigkeit. Beobachte, ob das Neuron zu früh, zu
spät oder im brauchbaren Zeitfenster feuert. Du erledigst manuell, was ein
Lernalgorithmus später automatisiert: Fehler beobachten, Parameter ändern,
erneut testen.

---

## 7 · Absichtlich kaputtmachen

Wähle einen Fehler aus deinem Hauptpfad und mache ihn offensichtlich:

1. schreibe eine Ein-Satz-Vorhersage;
2. ändere nur einen Parameter;
3. erfasse das sichtbare Ergebnis;
4. identifiziere den dominierenden Beitrag;
5. stelle die Baseline wieder her und bestätige die Erholung.

Nutze diese Diagnosereihenfolge:

| Sichtbares Symptom | Erste Zahl zum Prüfen | Wahrscheinliche Ursache |
|---|---|---|
| Fast jeder Fall hat eine Klasse | Bias-Beitrag | Bias-Betrag zu groß |
| Ein Merkmal steuert alles | Gewichtete Beiträge | Fehlende Normalisierung oder übergroßes Gewicht |
| Entscheidung ist verkehrt herum | Vorzeichen des Beitrags | Umgekehrtes Gewicht |
| Sprung feuert immer | Bias-Beitrag | Bias zu positiv |
| Sprung feuert nie | Summe bleibt unter null | Bias zu negativ oder Gewichte zu klein |

??? question "Abschluss-Check"
    Kannst du eine Ausgabe von Hand berechnen, eine Gewichts- oder Bias-Änderung
    vorhersagen, die Forward-Schleife implementieren, eine nicht normalisierte
    Eingabe identifizieren und den sichtbaren Fehler erklären, ohne nur zu sagen,
    „die KI ist schlecht"?

??? success "Antwortschlüssel"
    Eine vollständige Erklärung nennt Eingabe, Gewicht, Beitrag, gewichtete Summe,
    Aktivierungsausgabe und sichtbare Folge. Beispiel: „Rohdruck machte den zweiten
    Beitrag `-40`, der den Temperaturbeitrag `+0.8` dominierte, sodass fast jeder
    Punkt Klasse `0` wurde."

---

## 8 · Die beiden Pfade vergleichen

Die Research-Grenze und das Jumper-Verhalten sind zwei Ansichten derselben Forward-
Berechnung.

| Gemeinsame Rolle | Research-Visualisierung | Game-Visualisierung |
|---|---|---|
| Eingabe \(x_1\) | Temperaturposition | Geschwindigkeitsregler |
| Eingabe \(x_2\) | Druckposition | Nähe zur Kante |
| Gewichtete Summe \(z\) | Berechnung im Seitenpanel | Berechnung im Overlay |
| Schwelle | Punktfarbe | `WAIT`/`JUMP`-Ereignis |
| Parametereffekt | Grenze dreht oder verschiebt sich | Auslösezeit verändert sich |
| Fehler-Evidenz | Roter Fehlklassifikationsring | Zu früh, zu spät oder gelandet |

Für eine Forscherin fasst die Grenze viele Beobachtungen auf einmal zusammen. Für
eine Spieleentwicklerin zeigt Bewegung einen Zustand, der sich über die Zeit
ändert. Keine Ansicht ändert das Neuron:

```text
normalized inputs → weighted contributions → bias → activation → decision
```

Erkläre die Äquivalenz laut: Eine rotierende Klassifikationsgrenze ändert, auf
welcher Seite Punkte liegen; geänderte Jumper-Gewichte ändern, welche
Geschwindigkeits-Distanz-Kombinationen den Sprung auslösen.

---

## 9 · Stretch Goals

**Research — Evidenz exportieren.** Führe aus:

```bash
MPLBACKEND=Agg python \
  examples/neural_foundations/research/plot_neuron.py \
  --save neuron-boundary.png
```

Füge deine Hypothese und Parametertabelle neben dem gespeicherten Bild hinzu.

**Game development — Coyote Time ergänzen.** Erlaube das Sprung-Ereignis noch
einige Frames nach dem Überqueren der Kante. Vergleiche, wie sich späte Fehler
verändern, ohne die Neuron-Rechnung zu ändern.

**Beide Pfade — dritte normalisierte Eingabe hinzufügen.** Wähle ein sinnvolles
Merkmal, sage sein Vorzeichen voraus, aktualisiere zuerst den Forward-Pass-Test und
dann die Visualisierung. Halte den aktuellen Beitrag sichtbar.

**Beide Pfade — ungültige Formen testen.** Füge einen Test hinzu, der zeigt, dass
Eingaben und Gewichte gleiche Länge haben müssen. Erkläre, warum das stille
Weglassen eines Merkmals die sichtbare Evidenz irreführend machen würde.

---

## Was kommt als Nächstes

Ein Neuron kann nur eine gerade Grenze durch seine Eingaben ziehen. In **Neuronale
Grundlagen 2** verbindest du ein paar Neuronen, erzeugst eine nichtlineare
Entscheidungsregion, misst Fehler und aktualisierst Gewichte aus Beispielen.

[← Einheit 0](unit-00.md) · [Kursstartseite](index.md) · [→ Neuronale Grundlagen 2](unit-neural-02.md)
