# Neuronale Grundlagen 1 — Ein Neuron, eine Entscheidung

[← Einheit 0](unit-00.md) · [Kursstartseite](index.md)

!!! info "Drei Wege, die Berechnung zu sehen"
    Laufende Visualisierung · aktuelle Zahlen · Code, den du geschrieben hast

Eine trainierte Policy kann rätselhaft wirken, aber jede Entscheidung beginnt mit
gewöhnlicher Arithmetik. In dieser Einheit baust du ein Neuron, beobachtest jede
Größe beim Wechsel und nutzt dieselbe Berechnung für eine Forschungsklassifikation
und einen Spielgegner.

> **Frage für beide Pfade:** Wie können zwei Messwerte zu einer sichtbaren
> Entscheidung werden?

Die Lernschleife lautet **Predict → Play → Build → Break → Explain**. Schließe
einen Hauptpfad ab und verbringe dann zehn Minuten mit dem anderen Pfad.

---

## 1 · Vorhersagen, bevor du startest

Beginne mit diesem eingefrorenen Neuron:

| Eingabe | Wert | Gewicht | Beitrag |
|---|---:|---:|---:|
| \(x_1\) | 0.50 | +0.80 | +0.40 |
| \(x_2\) | -0.25 | -0.40 | +0.10 |
| Bias | — | — | +0.10 |

Bevor du Python oder Godot nutzt, notiere:

1. die gewichtete Summe;
2. ob `tanh` einen negativen oder positiven Wert liefert;
3. welche Eingabe am meisten zur Entscheidung beiträgt.

Die vollständige Berechnung lautet:

$$
z = w_1x_1 + w_2x_2 + b
$$

$$
z = (0.8)(0.5) + (-0.4)(-0.25) + 0.1 = 0.6
$$

Die Aktivierung liefert dann \(\tanh(0.6) \approx 0.537\).

??? success "Antwortschlüssel"
    Die gewichtete Summe ist `0.6`, also ist die aktivierte Ausgabe positiv. Der
    erste Beitrag ist am größten: `+0.40`, verglichen mit `+0.10` von der zweiten
    Eingabe und `+0.10` vom Bias.

**Sichtbare Prüfung:** Die automatisierten Beispiele verwenden dieselben Zahlen:

```bash
conda activate godot_env
python -m examples.neural_foundations.research.tests.test_neuron

godot --headless \
  --path examples/neural_foundations/game \
  --script res://test/test_tiny_neuron.gd
```

Beide Befehle geben den Abschnitt-1-Walkthrough aus (`z = 0.6`, `tanh(z) ≈ 0.537`)
und enden mit `OK`. Der Godot-Lauf zeigt zusätzlich die Live-Labels der
Gegner-Demo (Health, Distance, gewichtete Summe, Chase/Retreat).

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

Die gewichtete Summe \(z\) kann jede Zahl sein. Eine **Aktivierungsfunktion**
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
Stern-Sonde ändert. In Godot wählt das Vorzeichen von `tanh(z)` `CHASE` oder
`RETREAT`.

---

## 4 · Wähle deinen Pfad

Die Gleichung ist gemeinsam; die Evidenz unterscheidet sich.

| | Research-Pfad | Game-Pfad |
|---|---|---|
| Eingaben | Normalisierte Temperatur und Druck | Normalisierte Gesundheit und Spielerdistanz |
| Ausgabe | Sichere oder unsichere Klasse | Chase- oder Retreat-Tendenz |
| Hauptvisualisierung | Farbige Punkte und Entscheidungsgrenze | Gegnerfarbe, Bewegung, Linie und Gesundheitsleiste |
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

## 6 · Game-Pfad — chase oder retreat

**Game-AI-Frage:** Kann ein Neuron wählen, ob ein Gegner CHASE oder RETREAT
soll?

Öffne das eigenständige Standard-Godot-Projekt:

```bash
godot --editor --path examples/neural_foundations/game
```

Öffne `unit_01_enemy/unit_01_enemy.tscn` und drücke **F6**. Bewege den Spieler
mit den Pfeiltasten und ändere die Gegnergesundheit mit dem Schieberegler.

Die Szene legt den vollständigen Forward Pass offen:

- die gestrichelte Linie und das Distanz-Label zeigen die erste räumliche Beziehung;
- die Gesundheitsleiste zeigt die zweite Eingabe;
- fünf Berechnungslabels zeigen Beiträge, Bias, \(z\) und `tanh(z)`;
- orange Bewegung bedeutet `CHASE`;
- blaue Bewegung bedeutet `RETREAT`;
- **Pause** friert den Gegner ein, während Werte sichtbar bleiben;
- **Reset** stellt eine wiederholbare Startposition her.

Beim Reset lautet die ungefähre Berechnung:

$$
z = (0.75)(1.4) + (0.68)(-1.2) - 0.1 \approx 0.134
$$

Das positive Ergebnis lässt den Gegner CHASE wählen. Die im Overlay angezeigte
gewichtete Summe verwendet dieselben Eingabewerte, Gewichte und denselben Bias wie
`TinyNeuron.forward()` vor der Aktivierung.

Wähle die Szenenwurzel, um die exportierten `health_weight`, `distance_weight` und
`bias` im Inspector zu ändern.

### Experiment 1 — das Gesundheitsgewicht umkehren

**Vorhersage zuerst:** Was soll ein gesunder Gegner tun, wenn der Spieler auf
derselben Distanz gehalten wird und `health_weight` von `+1.4` auf `-1.4` wechselt?

Teste niedrige, mittlere und hohe Gesundheit. Notiere erwartetes und tatsächliches
Verhalten.

??? success "Antwortschlüssel"
    Mehr Gesundheit senkt jetzt die gewichtete Summe. Der gesündeste Gegner zieht
    eher RETREAT vor, während niedrige Gesundheit weniger von der Summe abzieht.
    Das Overlay legt das falsche Vorzeichen sofort offen: Der Gesundheitsbeitrag
    ist negativ.

### Experiment 2 — positiven Bias hinzufügen

Stelle `health_weight` auf `+1.4` zurück. Erhöhe den Bias von `-0.1` auf `+0.8`,
und teste dann dieselben drei Spielerpositionen.

??? success "Antwortschlüssel"
    Positiver Bias verschiebt jede Situation Richtung CHASE. Der Gegner kann bei
    niedriger Gesundheit oder großer Entfernung aggressiv bleiben, weil `+0.8`
    überwunden werden muss, bevor die Ausgabe negativ wird. Die Linie verursacht
    dieses Verhalten nicht; das Bias-Label enthüllt es.

### Experiment 3 — die Schwelle überschreiten

Stelle die Standardwerte wieder her. Bewege den Spieler langsam um die Distanz, an
der sich das Verhalten ändert. Ändere die Gesundheit nicht.

??? success "Antwortschlüssel"
    Nahe \(z=0\) kippen kleine Distanzänderungen das gewählte Verhalten. Der Gegner
    kann oszillieren, weil eine einzelne harte Schwelle kein Gedächtnis und keine
    Hysterese hat. Die sich glatt ändernde `tanh(z)`-Zahl zeigt, dass die
    zugrunde liegende Punktzahl stabil ist, auch wenn die gewählte Aktion wechselt.

### Game-Development-Evidenz

Pausiere die Szene und fülle aus:

| Situation | Erwartet | Tatsächlich | Gesundheitsterm | Distanzterm | Diagnose |
|---|---|---|---:|---:|---|
| Hohe Gesundheit, nah | | | | | |
| Niedrige Gesundheit, nah | | | | | |
| Hohe Gesundheit, weit | | | | | |

Eine Regel wie `if health > 0.5 and distance < 200` kann dieses Verhalten direkt
autorisieren. Das Neuron wird später nützlich, weil seine differenzierbaren Gewichte
aus Beispielen oder Belohnung angepasst werden können, statt jede Regel von Hand
abzustimmen.

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
| Schnelles Umschalten nahe der Grenze | Gewichtete Summe nahe null | Kein Abstand, Gedächtnis oder Hysterese |

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

Die Research-Grenze und das Gegnerverhalten sind zwei Ansichten derselben Forward-
Berechnung.

| Gemeinsame Rolle | Research-Visualisierung | Game-Visualisierung |
|---|---|---|
| Eingabe \(x_1\) | Temperaturposition | Gesundheitsleiste |
| Eingabe \(x_2\) | Druckposition | Spielerdistanz-Linie |
| Gewichtete Summe \(z\) | Berechnung im Seitenpanel | Berechnung im Overlay |
| Schwelle | Punktfarbe | CHASE/RETREAT-Umschaltung |
| Parametereffekt | Grenze dreht oder verschiebt sich | Verhalten ändert sich im Raum |
| Fehler-Evidenz | Roter Fehlklassifikationsring | Erwartet/Tatsächlich-Abweichung |

Für eine Forscherin fasst die Grenze viele Beobachtungen auf einmal zusammen. Für
eine Spieleentwicklerin zeigt Bewegung einen Zustand, der sich über die Zeit
ändert. Keine Ansicht ändert das Neuron:

```text
normalized inputs → weighted contributions → bias → activation → decision
```

Erkläre die Äquivalenz laut: Eine rotierende Klassifikationsgrenze ändert, auf
welcher Seite Punkte liegen; geänderte Gegnergewichte ändern, welche Spielzustände
auf der CHASE- oder RETREAT-Seite liegen.

---

## 9 · Stretch Goals

**Research — Evidenz exportieren.** Führe aus:

```bash
MPLBACKEND=Agg python \
  examples/neural_foundations/research/plot_neuron.py \
  --save neuron-boundary.png
```

Füge deine Hypothese und Parametertabelle neben dem gespeicherten Bild hinzu.

**Game development — Entscheidungsabstand hinzufügen.** Bleibe bei CHASE, bis die
Ausgabe unter `-0.1` liegt, und bei RETREAT, bis sie über `+0.1` liegt. Vergleiche
Oszillation mit der ursprünglichen Nullschwelle.

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
