# Neuronale Grundlagen 1 — Ein Neuron, eine Entscheidung

[← Mathe-Grundlagen 2](unit-math-02.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~30 Min · Handrechnung: ~15 Min · Experimente in einem Pfad: ~45 Min

!!! success "Was du nach dieser Einheit kannst"
    - Die Ausgabe eines Neurons von Hand aus Eingaben, Gewichten und Bias berechnen
    - Sagen, was die Aktivierungsfunktion beiträgt und warum die Schwelle bei `0.5` liegt
    - Die Kurzschreibweise \(z = w_1x_1 + w_2x_2 + b\) lesen und jeden Buchstaben benennen
    - Vorhersagen, wie eine Änderung an einem Gewicht oder am Bias die Entscheidung verschiebt
    - Eine nicht normalisierte Eingabe allein an ihrem Beitrag erkennen

!!! note "Voraussetzungen"
    - **Einheit 0 abgeschlossen** — Conda, Godot und ein erfolgreicher BallChase-Lauf
    - Rechnen mit Dezimalzahlen. Keine Analysis, kein Vorwissen im Maschinellen Lernen
    - Grundlegende Sicherheit im Terminal

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

## 1 · Was ein Neuron berechnet

Ein Neuron tut drei Dinge in dieser Reihenfolge — und sonst nichts.

**Schritt 1 — jede Eingabe gewichten.** Jede Eingabe wird mit ihrer eigenen Zahl
multipliziert, dem **Gewicht**. Das Ergebnis ist der **Beitrag** dieser Eingabe
zur Entscheidung. Ein großes Gewicht macht die Eingabe sehr wichtig; ein Gewicht
nahe null macht sie fast bedeutungslos; ein negatives Gewicht lässt die Eingabe
*gegen* die Entscheidung argumentieren.

**Schritt 2 — die Beiträge addieren, plus Bias.** Alle Beiträge werden summiert,
und eine zusätzliche Zahl kommt hinzu, die zu keiner Eingabe gehört: der
**Bias**. Der Bias ist die Grundtendenz des Neurons, bevor es überhaupt etwas
gesehen hat. Die Zwischensumme heißt **gewichtete Summe**.

**Schritt 3 — die Summe in eine Entscheidung übersetzen.** Dieser Schritt braucht
eine Erklärung, denn die gewichtete Summe ist eine unhandliche Zahl.

### Warum Schritt 3 nötig ist

Die gewichtete Summe kann überall landen: `-37.2`, `0.02`, `+415.0`. Die Frage
lautet aber nicht „welche Zahl ist das?" — sie lautet „soll die Figur springen?"
Gesucht ist eine Sicherheit zwischen *auf keinen Fall* und *auf jeden Fall*.

Genau diese Umrechnung leistet eine **Aktivierungsfunktion**. Diese Einheit
benutzt die Funktion **Sigmoid**: Sie quetscht jede Zahl, wie groß oder klein
auch immer, in den Bereich zwischen `0` und `1`.

<div class="diagram-scroll">

<svg class="course-diagram" viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="Die Sigmoid-Kurve: Sie nähert sich 0 bei stark negativen Summen, verläuft bei Summe null durch genau 0.5 und nähert sich 1 bei stark positiven Summen">
  <defs>
    <marker id="arS" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 1 L10 5 L0 9 z" fill="#8892b0"/>
    </marker>
  </defs>
  <line x1="70" y1="50" x2="610" y2="50" stroke="#2e3350" stroke-width="1.1" stroke-dasharray="4 4"/>
  <line x1="70" y1="150" x2="610" y2="150" stroke="#ef8354" stroke-width="1.2" stroke-dasharray="5 5"/>
  <path d="M70 250 L616 250" fill="none" stroke="#8892b0" stroke-width="1.4" marker-end="url(#arS)"/>
  <path d="M340 268 L340 26" fill="none" stroke="#8892b0" stroke-width="1.4" marker-end="url(#arS)"/>
  <text x="606" y="270" text-anchor="end" fill="#8892b0" font-size="13">gewichtete Summe</text>
  <text x="330" y="34" text-anchor="end" fill="#8892b0" font-size="13">Ausgabe</text>
  <text x="62" y="55" text-anchor="end" fill="#8892b0" font-size="13">1.0</text>
  <text x="62" y="155" text-anchor="end" fill="#8892b0" font-size="13">0.5</text>
  <text x="62" y="255" text-anchor="end" fill="#8892b0" font-size="13">0.0</text>
  <text x="78" y="142" fill="#ef8354" font-size="13" font-weight="700">0.5 — die Entscheidungsschwelle</text>
  <path d="M70.0 249.5 L81.2 249.4 L92.5 249.2 L103.8 249.0 L115.0 248.7 L126.2 248.3 L137.5 247.8 L148.8 247.2 L160.0 246.4 L171.2 245.4 L182.5 244.1 L193.8 242.5 L205.0 240.5 L216.2 238.0 L227.5 234.8 L238.8 230.9 L250.0 226.2 L261.2 220.4 L272.5 213.5 L283.8 205.5 L295.0 196.2 L306.2 185.8 L317.5 174.5 L328.8 162.4 L340.0 150.0 L351.2 137.6 L362.5 125.5 L373.8 114.2 L385.0 103.8 L396.2 94.5 L407.5 86.5 L418.8 79.6 L430.0 73.8 L441.2 69.1 L452.5 65.2 L463.8 62.0 L475.0 59.5 L486.2 57.5 L497.5 55.9 L508.8 54.6 L520.0 53.6 L531.2 52.8 L542.5 52.2 L553.8 51.7 L565.0 51.3 L576.2 51.0 L587.5 50.8 L598.8 50.6 L610.0 50.5" fill="none" stroke="#4ecca3" stroke-width="2.6" stroke-linejoin="round"/>
  <circle cx="340" cy="150" r="6" fill="#4ecca3"/>
  <text x="352" y="186" fill="#8892b0" font-size="13">Summe 0 → genau 0.5</text>
  <text x="160" y="228" text-anchor="middle" fill="#8892b0" font-size="13">stark negativ → nahe 0</text>
  <text x="520" y="82" text-anchor="middle" fill="#8892b0" font-size="13">stark positiv → nahe 1</text>
</svg>

</div>

Drei Orientierungspunkte lohnen sich zu merken, denn jedes spätere Ablesen dieser
Kurve hängt an ihnen:

| Gewichtete Summe | Sigmoid-Ausgabe | Wie man sie liest |
|---:|---:|---|
| -3.00 | 0.047 | mit ziemlicher Sicherheit nein |
| -0.50 | 0.378 | tendenziell nein |
| 0.00 | **0.500** | vollkommen unentschieden |
| +0.20 | 0.550 | tendenziell ja |
| +3.00 | 0.953 | mit ziemlicher Sicherheit ja |

Daher kommt die Schwelle `0.5`. Sie ist keine willkürliche Grenze: Sigmoid
liefert genau `0.5`, wenn die gewichtete Summe genau `0` ist. Damit sind
**„Ausgabe über 0.5" und „gewichtete Summe über 0" dieselbe Aussage**. Das Neuron
feuert, wenn Beiträge und Bias zusammen etwas Positives ergeben.

!!! note "Die Formel, nur zum Nachschlagen"
    $$
    \operatorname{sigmoid}(z) = \frac{1}{1 + e^{-z}}
    $$

    Du musst sie in diesem Kurs nie von Hand ausrechnen. Lies den Wert an der
    Kurve ab oder lass ihn vom Code berechnen. Wichtig ist die Form: Sie verlässt
    den Bereich `0` bis `1` nie und kreuzt `0.5` bei null.

### Die Kurzschreibweise, der du überall begegnest

Ausgeschriebene Namen werden lang, also kürzt die Mathematik sie ab. Die Kürzel
werden einmal hier eingeführt und im ganzen Kurs weiterverwendet:

| Bedeutung in Worten | Kürzel | Ausgesprochen | Warum dieser Buchstabe |
|---|---|---|---|
| erste Eingabe, zweite Eingabe | \(x_1\), \(x_2\) | „iks eins", „iks zwei" | \(x\) ist der traditionelle Buchstabe für eine unbekannte Größe |
| das Gewicht der jeweiligen Eingabe | \(w_1\), \(w_2\) | „we eins", „we zwei" | \(w\) für englisch **w**eight (Gewicht) — lateinisches \(w\), nicht griechisches \(\omega\) |
| Bias | \(b\) | „be" | \(b\) für **B**ias |
| gewichtete Summe (Ergebnis von Schritt 1 und 2) | \(z\) | „zett" | üblicher Buchstabe für die Summe vor der Aktivierung |
| „addiere das alles zusammen", in Diagrammen als Kasten gezeichnet | \(\Sigma\) | „Sigma" | griechisches großes S, für **S**umme |
| Ausgabe nach der Aktivierungsfunktion | — | „Ausgabe" | behält in diesem Kurs ihren einfachen Namen |

Die Schritte 1 und 2 zusammen schreiben sich also:

$$
z = w_1x_1 + w_2x_2 + b
$$

Vier Lesehinweise, über die viele stolpern:

- \(w_1\) und `w₁` sind **dasselbe**. Fließtext und Slider-Beschriftungen in
  diesem Kurs verwenden `w₁`, Formeln verwenden \(w_1\). Gewicht ist immer ein
  kleines \(w\).
- Das lateinische \(w\) und das griechische \(\omega\) („Omega") sehen sich
  zum Verwechseln ähnlich, besonders handschriftlich und in kursiver
  Mathe-Schrift. In diesem Kurs ist \(w\) immer ein Gewicht. \(\omega\)
  taucht erst sehr viel später auf, in
  [Hierarchisches RL](unit-hierarchical.md), wo es eine *Option* bezeichnet —
  ein völlig anderer Begriff.
- Die kleine tiefgestellte Zahl ist eine Nummerierung, keine Potenz. \(x_1\)
  heißt „die erste Eingabe", nicht „x hoch eins".
- Ein griechischer Buchstabe wird mit seinem Namen gelesen, nie nach seiner
  Form. \(\Sigma\) spricht man „Sigma". Spätere Einheiten bringen
  \(\gamma\) („Gamma"), \(\alpha\) („Alpha") und \(\varepsilon\)
  („Epsilon"); das [Glossar](glossary.md) nennt zu jedem den Namen.

---

## 2 · Vorhersagen, bevor du startest

Wende die drei Schritte jetzt auf ein **Neuron mit festen Zahlen** an. Das ist
eine Vorhersage-Übung: Rechne auf Papier, *bevor* du die Lösung aufklappst oder
etwas startest.

| Benannte Eingabe | Wert | Gewicht |
|---|---:|---:|
| Geschwindigkeit | 0.50 | +0.80 |
| Nähe zur Kante | 0.25 | +1.20 |
| Bias | — | -0.50 |

Notiere in dieser Reihenfolge:

1. den Beitrag jeder Eingabe — Wert × Gewicht;
2. die gewichtete Summe, also Beiträge plus Bias;
3. ob die Sigmoid-Ausgabe über `0.5` liegt — an der Kurve aus Abschnitt 1
   ablesen, kein Taschenrechner nötig;
4. welche Eingabe am stärksten Richtung Sprung drückt.

??? success "Musterlösung — erst aufklappen, wenn du deine Antwort notiert hast"
    **1. Beiträge**

    | Benannte Eingabe | Wert | Gewicht | Beitrag |
    |---|---:|---:|---:|
    | Geschwindigkeit | 0.50 | +0.80 | +0.40 |
    | Nähe zur Kante | 0.25 | +1.20 | +0.30 |
    | Bias | — | — | -0.50 |

    **2. Gewichtete Summe**

    $$
    z =
    (\text{Geschwindigkeit}\times\text{Geschwindigkeitsgewicht}) +
    (\text{Nähe}\times\text{Nähegewicht}) +
    \text{Bias}
    $$

    $$
    z = (0.5)(0.8) + (0.25)(1.2) - 0.5 = 0.2
    $$

    **3. Entscheidung.** Die Summe ist positiv, also muss die Ausgabe über `0.5`
    liegen, noch bevor du irgendetwas ausrechnest:
    \(\operatorname{sigmoid}(0.2) \approx 0.550\). Das Neuron **feuert**, die
    Spielaktion lautet `JUMP`. Die Beschriftungen in der Godot-Szene sind
    englisch, deshalb stehen sie hier unübersetzt: du sollst auf dem Bildschirm
    genau das wiederfinden, was im Text steht.

    **4. Stärkster Schub.** Geschwindigkeit trägt `+0.40` bei, Nähe `+0.30`, und
    der Bias zieht `0.50` ab. Geschwindigkeit drückt am stärksten Richtung
    Sprung — beachte aber, dass der Bias allein größer ist als jeder einzelne
    Beitrag. Genau deshalb liegt die Entscheidung so dicht an der Schwelle.

    Die ganze Rechnung als ein Bild:

    <div class="diagram-scroll">

    <svg class="course-diagram" viewBox="0 0 800 300" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="Ein Neuron: Geschwindigkeit 0.50 mal Gewicht 0.80 ergibt plus 0.40, Nähe 0.25 mal Gewicht 1.20 ergibt plus 0.30, summiert mit Bias minus 0.50 ergibt 0.20; Sigmoid liefert 0.550, das über 0.5 liegt, also feuert das Neuron JUMP">
      <defs>
        <marker id="arN" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 1 L10 5 L0 9 z" fill="#8892b0"/>
        </marker>
      </defs>
      <rect x="20" y="40" width="180" height="60" rx="10" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
      <text x="110" y="65" text-anchor="middle" fill="#e2e8f0" font-size="14" font-weight="700">Geschwindigkeit</text>
      <text x="110" y="86" text-anchor="middle" fill="#8892b0" font-size="13">0.50</text>
      <rect x="20" y="180" width="180" height="60" rx="10" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
      <text x="110" y="205" text-anchor="middle" fill="#e2e8f0" font-size="14" font-weight="700">Nähe zur Kante</text>
      <text x="110" y="226" text-anchor="middle" fill="#8892b0" font-size="13">0.25</text>
      <path d="M200 70 C280 70, 290 115, 350 122" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#arN)"/>
      <text x="272" y="58" text-anchor="middle" fill="#6c8ef7" font-size="12" font-weight="700">× 0.80 → +0.40</text>
      <path d="M200 210 C280 210, 290 165, 350 158" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#arN)"/>
      <text x="272" y="232" text-anchor="middle" fill="#6c8ef7" font-size="12" font-weight="700">× 1.20 → +0.30</text>
      <rect x="350" y="100" width="140" height="80" rx="10" fill="#1a1d27" stroke="#8892b0" stroke-width="1.5"/>
      <text x="420" y="132" text-anchor="middle" fill="#e2e8f0" font-size="15" font-weight="700">Summe + Bias</text>
      <text x="420" y="158" text-anchor="middle" fill="#8892b0" font-size="13">= 0.20</text>
      <path d="M420 242 L420 184" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#arN)"/>
      <text x="420" y="264" text-anchor="middle" fill="#8892b0" font-size="13">Bias −0.50</text>
      <path d="M490 140 L540 140" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#arN)"/>
      <rect x="540" y="100" width="140" height="80" rx="10" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
      <text x="610" y="132" text-anchor="middle" fill="#e2e8f0" font-size="15" font-weight="700">sigmoid</text>
      <text x="610" y="158" text-anchor="middle" fill="#8892b0" font-size="13">≈ 0.550</text>
      <path d="M680 140 L716 140" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#arN)"/>
      <text x="755" y="136" text-anchor="middle" fill="#4ecca3" font-size="16" font-weight="700">JUMP</text>
      <text x="755" y="160" text-anchor="middle" fill="#8892b0" font-size="11">0.550 &gt; 0.5</text>
    </svg>

    </div>

**Sichtbare Prüfung:** Die automatisierten Beispiele nutzen dieselben Zahlen.

!!! note "Aus dem Kurs-Repo-Root ausführen"
    Diese Befehle setzen voraus, dass dein Terminal im [Kurs-Repo-Root](setup.md#course-repo) steht und `godot` in deinem PATH liegt — siehe [Godot auf der Kommandozeile](setup.md#godot-cli).

```bash
conda activate godot_env
python -m examples.neural_foundations.research.tests.test_neuron

godot --headless \
  --path examples/neural_foundations/game \
  --script res://test/test_tiny_neuron.gd
```

Beide Befehle geben dieselbe Rechnung aus (`sum (z) = +0.200`,
`sigmoid(sum) = 0.550`) und enden mit `OK`. Der Godot-Lauf zeigt zusätzlich die
Live-Beschriftungen der Jumper-Demo (Geschwindigkeit, Nähe, Summe, Ausgabe und
`WAIT`/`JUMP`).

Beide Tests rufen den Forward Pass auf, den du dir als Nächstes ansiehst.

---

## 3 · Gewichtete Eingaben und Bias

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

## 4 · Aktivierungsfunktionen

Sigmoid ist nicht der einzige Weg, die Summe \(z\) in eine Entscheidung zu
übersetzen. Für den Sprung-Trigger wird sie benutzt, weil dort eine Sicherheit
zwischen `0` und `1` gebraucht wird — andere Entscheidungen brauchen andere
Ausgabeformen.

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
Stern-Sonde ändert. In Godot löst `sigmoid(z) > 0.5` das Ereignis `JUMP` aus.

---

## 5 · Wähle deinen Pfad

Die Gleichung ist gemeinsam; die Evidenz unterscheidet sich.

| | Research-Pfad | Game-Pfad |
|---|---|---|
| Eingaben | Normalisierte Temperatur und Druck | Normalisierte Geschwindigkeit und Nähe zur Kante |
| Ausgabe | Sichere oder unsichere Klasse | Warten oder Sprung-Ereignis auslösen |
| Hauptvisualisierung | Farbige Punkte und Entscheidungsgrenze | Eingaberegler, sichtbarer Bogen, Klippe und Lava |
| Evidenz | Genauigkeit und falsch klassifizierte Punkte | Erwartetes versus tatsächliches Verhalten |
| Werkzeug | Python + Matplotlib | Standard Godot 4 + GDScript |

Wähle einen Hauptpfad:

- **Research:** Schließe Abschnitt 6 ab und sieh dir den Godot-Vergleich einmal an.
- **Game development:** Schließe Abschnitt 7 ab und sieh dir den Plot-Vergleich einmal an.

In dieser Einheit brauchst du keine native Extension, C#, kein Trainingsframework
und keine vorherige Machine-Learning-Bibliothek.

---

## 6 · Research-Pfad — sichtbare Entscheidungsgrenze

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

## 7 · Game-Pfad — Sprung-Timing an der Klippe

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

## 8 · Absichtlich kaputtmachen

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

## 9 · Die beiden Pfade vergleichen

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

## 10 · Stretch Goals

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

!!! info "Selbsttest, bevor du weitergehst"
    1. Welche drei Schritte führt ein Neuron aus, in welcher Reihenfolge?
    2. Was leistet der Bias, was ein Gewicht nicht leisten kann?
    3. Warum liegt die Entscheidungsschwelle bei `0.5` und nicht bei einer
       anderen Zahl?
    4. Wofür steht in \(z = w_1x_1 + w_2x_2 + b\) jeder einzelne Buchstabe?
    5. Ein Beitrag kommt mit `-4.00` heraus, während alle anderen unter `1`
       liegen. Was ist die wahrscheinlichste Ursache?
    6. Was macht eine Gewichtsänderung mit der Entscheidungsgrenze? Was macht
       eine Bias-Änderung?

??? success "Antworten zum Selbsttest"
    1. Jede Eingabe gewichten, die Beiträge plus Bias addieren, die gewichtete
       Summe durch eine Aktivierungsfunktion schicken.
    2. Der Bias verschiebt alle Entscheidungen zugleich, unabhängig von den
       Eingaben. Er ist die Grundtendenz des Neurons; ein Gewicht wirkt nur,
       wenn seine Eingabe ungleich null ist.
    3. Weil Sigmoid genau `0.5` liefert, wenn die gewichtete Summe `0` ist.
       „Ausgabe über `0.5`" ist nur eine andere Formulierung für „gewichtete
       Summe über `0`".
    4. \(x_1, x_2\) sind die Eingaben, \(w_1, w_2\) ihre Gewichte, \(b\)
       der Bias und \(z\) die gewichtete Summe vor der Aktivierungsfunktion.
    5. Eine fehlende Normalisierung — diese Eingabe steht noch in ihrer
       Roh-Einheit, also überdeckt ihr Beitrag alle anderen.
    6. Ein Gewicht dreht die Grenze; der Bias verschiebt sie, ohne sie zu drehen.

[← Mathe-Grundlagen 2](unit-math-02.md) · [Kursstartseite](index.md) · [→ Neuronale Grundlagen 2](unit-neural-02.md)
