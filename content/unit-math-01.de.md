# Mathe-Grundlagen 1 — Vektoren und Matrizen

[← Einheit 0](unit-00.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~25 Min · Beispiele durcharbeiten: ~20 Min

!!! tip "Überspringbar — komm zurück, wenn du es brauchst"
    In Phase 1 und 2 geht nichts kaputt, wenn du diese Einheit überspringst. Sie
    existiert, damit du eine Stelle zum Nachschlagen hast, sobald eine spätere
    Einheit \(w^\top x\) schreibt oder von „dem Beobachtungsvektor" spricht.
    Wenn dir das Rechnen in Neuronale Grundlagen 1 leichtfiel, geh direkt dorthin
    und komm wieder, sobald etwas nicht mehr aufgeht.

!!! success "Was du nach dieser Einheit kannst"
    - Eine Beobachtung als Vektor lesen und sagen, was ihre Dimension bedeutet
    - Ein Skalarprodukt von Hand rechnen und es als die dir bekannte gewichtete Summe erkennen
    - Erklären, was das Normalisieren eines Vektors bewirkt und warum der Kurs Eingaben in `0`–`1` hält
    - Eine Matrix als Stapel von Gewichtszeilen lesen — eine Zeile pro Neuron
    - Die Ergebnisform eines Matrix-Vektor-Produkts vorhersagen und einen Shape-Fehler diagnostizieren

!!! note "Voraussetzungen"
    - **Einheit 0 abgeschlossen** — Conda und ein funktionierendes `godot_env`
    - Rechnen mit Dezimalzahlen. Keine Analysis
    - Grundlegende Sicherheit im Terminal

!!! info "Drei Wege, die Zahlen zu sehen"
    Python + NumPy (`print(v.shape)`) · das Beobachtungs-Array, das Godot sendet ·
    die Einheiten, in denen jede Idee wiederkommt

Reinforcement Learning ist in der Sprache von Vektoren und Matrizen geschrieben,
aber die Ideen darunter sind klein. Ein Vektor ist eine Liste von Zahlen. Eine
Matrix ist eine Liste von Listen. Alles in dieser Einheit ist Zählen,
Multiplizieren und Addieren — so angeordnet, dass ein Computer Tausende davon
gleichzeitig erledigen kann.

---

## 1 · Ein Vektor ist eine Beobachtung

Wenn ein Godot-Agent meldet, was er sieht, schickt er nicht eine Zahl. Er
schickt mehrere, in fester Reihenfolge:

```text
[ Distanz zum Ziel, Geschwindigkeit, Winkel zum Ziel, Gesundheit ]
[ 0.42,             0.80,            -0.15,           1.00       ]
```

Diese geordnete Liste ist ein **Vektor**. Zwei Eigenschaften zählen:

- seine **Dimension** — wie viele Zahlen er enthält. Der Vektor oben ist
  4-dimensional;
- seine **Reihenfolge** — Position 2 ist immer die Geschwindigkeit. Vertausche
  zwei Einträge, und dem Agenten wird etwas Falsches erzählt.

```python
import numpy as np

observation = np.array([0.42, 0.80, -0.15, 1.00])

print(observation)          # [ 0.42  0.8  -0.15  1.  ]
print(observation.shape)    # (4,)
print(observation[1])       # 0.8  — Geschwindigkeit
```

`shape` liest sich als `(4,)` — eine einzelne Achse mit vier Zahlen. Das ist
dasselbe Array, das `get_obs()` auf der Godot-Seite erzeugt, und dasselbe, das
das Policy-Netz als Eingabe erhält.

!!! warning "Die Reihenfolge ist ein Vertrag"
    Gibt `get_obs()` die Distanz zuerst zurück, während das Netz die
    Geschwindigkeit zuerst erwartet, gibt es **keine** Fehlermeldung. Der Agent
    verhält sich einfach unsinnig. Ein stilles Vertauschen gehört zu den am
    schwersten zu findenden Fehlern in diesem Kurs — deshalb lässt dich
    [Einheit 2](unit-02.md) die Beobachtungsliste aufschreiben, bevor du den
    Code schreibst.

---

## 2 · Addieren und Skalieren

Zwei Vektoren gleicher Dimension addieren sich **elementweise**:

$$
\begin{bmatrix} 2 \\ 1 \end{bmatrix} +
\begin{bmatrix} 3 \\ 4 \end{bmatrix} =
\begin{bmatrix} 5 \\ 5 \end{bmatrix}
$$

Die Multiplikation mit einer einzelnen Zahl — einem **Skalar** — streckt jeden
Eintrag um denselben Faktor:

$$
0.5 \times \begin{bmatrix} 4 \\ 6 \end{bmatrix} =
\begin{bmatrix} 2 \\ 3 \end{bmatrix}
$$

```python
import numpy as np

position = np.array([2.0, 1.0])
velocity = np.array([3.0, 4.0])

print(position + velocity)    # [5. 5.]
print(0.5 * velocity)         # [1.5 2. ]
```

Im Spiel ist das gewöhnliche Bewegung: Position plus Geschwindigkeit ergibt die
nächste Position, und die Geschwindigkeit mit `delta` zu skalieren ist genau das,
was `position += velocity * delta` in GDScript tut. Die Vektorschreibweise ist
keine neue Idee — sie ist dieselbe Codezeile, nur für viele Zahlen auf einmal.

---

## 3 · Das Skalarprodukt ist die gewichtete Summe

Das ist die Verbindung, für die sich die ganze Einheit lohnt.

In [Neuronale Grundlagen 1](unit-neural-01.md) hat ein Neuron jede Eingabe mit
ihrem Gewicht multipliziert und die Ergebnisse addiert:

$$
z = w_1x_1 + w_2x_2 + b
$$

Dieses Muster — zusammengehörige Paare multiplizieren, dann alles addieren — hat
einen Namen. Es ist das **Skalarprodukt** von Eingabevektor und Gewichtsvektor,
geschrieben \(w \cdot x\).

$$
\begin{bmatrix} 0.5 \\ 0.25 \end{bmatrix} \cdot
\begin{bmatrix} 0.8 \\ 1.2 \end{bmatrix}
= (0.5)(0.8) + (0.25)(1.2) = 0.4 + 0.3 = 0.7
$$

Das sind die Zahlen aus Neuronale Grundlagen 1. Addiere den Bias `-0.5` und du
erhältst `0.2` — dieselbe gewichtete Summe, auf demselben Weg. Es ist nichts
Neues passiert; die Rechenoperation hat lediglich einen Namen bekommen.

<div class="diagram-scroll">

<svg class="course-diagram" viewBox="0 0 700 220" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="Zwei Vektoren nebeneinander: Eingaben 0.50 und 0.25, Gewichte 0.80 und 1.20. Zusammengehörige Einträge werden multipliziert und ergeben 0.40 und 0.30, deren Summe die einzelne Zahl 0.70 ist.">
  <text x="90" y="34" text-anchor="middle" fill="#8892b0" font-size="13">Eingaben</text>
  <text x="240" y="34" text-anchor="middle" fill="#8892b0" font-size="13">Gewichte</text>
  <rect x="45" y="50" width="90" height="44" rx="8" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="90" y="78" text-anchor="middle" fill="#e2e8f0" font-size="16">0.50</text>
  <rect x="45" y="106" width="90" height="44" rx="8" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="90" y="134" text-anchor="middle" fill="#e2e8f0" font-size="16">0.25</text>
  <rect x="195" y="50" width="90" height="44" rx="8" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="240" y="78" text-anchor="middle" fill="#e2e8f0" font-size="16">0.80</text>
  <rect x="195" y="106" width="90" height="44" rx="8" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="240" y="134" text-anchor="middle" fill="#e2e8f0" font-size="16">1.20</text>
  <text x="165" y="79" text-anchor="middle" fill="#8892b0" font-size="17">×</text>
  <text x="165" y="135" text-anchor="middle" fill="#8892b0" font-size="17">×</text>
  <text x="330" y="79" text-anchor="middle" fill="#8892b0" font-size="17">=</text>
  <text x="330" y="135" text-anchor="middle" fill="#8892b0" font-size="17">=</text>
  <text x="395" y="80" text-anchor="middle" fill="#e2e8f0" font-size="16">0.40</text>
  <text x="395" y="136" text-anchor="middle" fill="#e2e8f0" font-size="16">0.30</text>
  <line x1="355" y1="156" x2="435" y2="156" stroke="#8892b0" stroke-width="1.4"/>
  <text x="332" y="162" text-anchor="middle" fill="#8892b0" font-size="17">+</text>
  <text x="395" y="184" text-anchor="middle" fill="#4ecca3" font-size="18" font-weight="700">0.70</text>
  <text x="560" y="100" text-anchor="middle" fill="#8892b0" font-size="14">zwei Vektoren hinein,</text>
  <text x="560" y="122" text-anchor="middle" fill="#8892b0" font-size="14">eine einzelne Zahl heraus</text>
</svg>

</div>

```python
import numpy as np

inputs = np.array([0.5, 0.25])
weights = np.array([0.8, 1.2])
bias = -0.5

print(inputs @ weights)                    # 0.7 — das Skalarprodukt
print(round(inputs @ weights + bias, 3))   # 0.2 — Neuronale Grundlagen 1, Abschnitt 2
```

Der Operator `@` ist Pythons Skalarprodukt. `np.dot(inputs, weights)` tut
dasselbe.

!!! note "Warum der Code rundet"
    Ohne `round` gibt die zweite Zeile `0.19999999999999996` aus. Dezimalbrüche
    haben keine exakte binäre Darstellung, also sammeln sich winzige Fehler an.
    Das ist normal und hier harmlos, aber es ist der Grund, warum man
    Fließkommazahlen mit einer Toleranz vergleicht statt mit `==`.

!!! info "Warum es zählt, dass das einen Namen hat"
    Jede Schicht jedes Netzes in diesem Kurs besteht aus Skalarprodukten. Wenn
    eine spätere Einheit \(\pi_\theta(a \mid s)\) schreibt und eine
    Matrizenmultiplikation zeigt, tut sie das, was du gerade von Hand getan hast
    — nur vielfach und parallel.

**Beide Vektoren müssen dieselbe Dimension haben.** Zwei Eingaben brauchen genau
zwei Gewichte. Das ist keine Konvention; es gibt schlicht keine vierte Zahl, die
man mit einem dritten Gewicht paaren könnte.

---

## 4 · Länge und Normalisierung

Die **Länge** eines Vektors — seine **Norm**, geschrieben \(\lVert v \rVert\) —
kommt von Pythagoras:

$$
\left\lVert \begin{bmatrix} 3 \\ 4 \end{bmatrix} \right\rVert
= \sqrt{3^2 + 4^2} = \sqrt{25} = 5
$$

**Normalisieren** heißt, einen Vektor durch seine eigene Länge zu teilen. Das
Ergebnis zeigt in dieselbe Richtung, hat aber die Länge `1`:

```python
import numpy as np

v = np.array([3.0, 4.0])
length = np.linalg.norm(v)

print(length)       # 5.0
print(v / length)   # [0.6 0.8]  — gleiche Richtung, Länge 1
```

Das ist dieselbe Idee wie im Abschnitt „Warum normalisieren?" von
[Neuronale Grundlagen 1](unit-neural-01.md): Rohdruck bei `80` hat eine
Temperatur von `0.7` erdrückt, weil der Beitrag einer Eingabe von ihrer
Größenordnung abhängt. Alle Beobachtungen in `0`–`1` zu halten bedeutet, dass
keine einzelne Eingabe allein über ihre Einheit dominieren kann.

!!! tip "Richtung gegen Betrag"
    Normalisieren trennt *wohin ein Vektor zeigt* von *wie groß er ist*. In
    einem Spiel ist das oft genau die Trennung, die man will: die Richtung zum
    Spieler ist das nützliche Signal, während die rohe Distanz in Pixeln von
    deiner Bildschirmauflösung abhängt.

---

## 5 · Eine Matrix ist eine Schicht von Neuronen

Ein Neuron hält einen Gewichtsvektor. Eine **Schicht** hält mehrere Neuronen mit
je eigenem Gewichtsvektor. Stapelt man diese Vektoren als Zeilen, hat man eine
**Matrix**:

$$
W =
\begin{bmatrix}
0.8 & 1.2 \\
-0.5 & 0.9 \\
0.3 & 0.3
\end{bmatrix}
$$

Drei Zeilen, zwei Spalten: **drei Neuronen, die je zwei Eingaben lesen**. Zeile 1
sind die Gewichte des ersten Neurons, Zeile 2 die des zweiten, Zeile 3 die des
dritten.

Diese Matrix mit einem Eingabevektor zu multiplizieren berechnet alle drei
Skalarprodukte auf einmal:

```python
import numpy as np

W = np.array([
    [0.8, 1.2],     # Gewichte Neuron 1
    [-0.5, 0.9],    # Gewichte Neuron 2
    [0.3, 0.3],     # Gewichte Neuron 3
])
inputs = np.array([0.5, 0.25])

print(W @ inputs)     # [ 0.7  -0.025  0.225 ]
print(W.shape)        # (3, 2)  — 3 Neuronen, je 2 Eingaben
```

Der erste Eintrag, `0.7`, ist das Skalarprodukt aus Abschnitt 3. Die anderen
beiden sind dieselbe Rechnung mit den Gewichten der anderen Neuronen — Neuron 2
wird negativ, weil sein erstes Gewicht negativ ist und es eine positive Eingabe
bekommt.

**Ein Matrix-Vektor-Produkt ist nichts anderes als mehrere gestapelte
Skalarprodukte.** Genau das bedeutet „das Netz macht einen Forward Pass": eine
Matrizenmultiplikation pro Schicht, danach eine Aktivierungsfunktion, und das
wiederholt.

---

## 6 · Die Formen müssen zusammenpassen

Der mit Abstand häufigste Fehler beim Arbeiten mit diesen Arrays ist eine
nicht passende **Form** (englisch *shape*). Die Regel:

> Um `W @ x` zu berechnen, muss die Anzahl der **Spalten** von `W` gleich der
> **Dimension** von `x` sein. Das Ergebnis hat so viele Einträge, wie `W`
> **Zeilen** hat.

$$
\underbrace{(3 \times 2)}_{W} \cdot \underbrace{(2)}_{x} \rightarrow
\underbrace{(3)}_{\text{Ergebnis}}
$$

Die inneren Zahlen müssen übereinstimmen; die äußeren überleben.

| Formen | Geht das? | Ergebnis |
|---|---|---|
| `(3, 2) @ (2,)` | ja | `(3,)` |
| `(3, 2) @ (3,)` | nein | innen stehen `2` und `3` |
| `(4, 8) @ (8,)` | ja | `(4,)` |
| `(2,) @ (2,)` | ja | eine einzelne Zahl — das Skalarprodukt |

Lies die Fehlermeldung, statt zu raten:

```text
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0,
with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 2)
```

Der nützliche Teil steht am Ende: `size 3 is different from 2`. Da kam etwas mit
3 Einträgen an, wo 2 erwartet wurden — fast immer ein Beobachtungsvektor, dessen
Länge nicht mehr zu dem Netz passt, für das die Policy gebaut wurde.

| Symptom | Was du zuerst prüfst |
|---|---|
| `size N is different from M` | Hat `get_obs()` einen Eintrag gewonnen oder verloren? |
| Ergebnis hat die falsche Anzahl Einträge | Multiplizierst du mit Zeilen, wo du Spalten meintest? |
| Läuft in Python, verhält sich in Godot falsch | Ist die Reihenfolge der Beobachtungen auf beiden Seiten gleich? |

---

## 7 · Stretch Goals

- **Kosinus-Ähnlichkeit.** Normalisiere zwei Beobachtungsvektoren und bilde ihr
  Skalarprodukt. Das Ergebnis läuft von `-1` bis `1` und misst, wie ähnlich sich
  zwei Situationen sind — unabhängig von ihren Beträgen. Berechne es für zwei
  selbst gewählte Zustände und prüfe, ob die Zahl deiner Intuition entspricht.
- **Ein echtes Netz durchzählen.** Neuronale Grundlagen 2 baut ein kleines Netz.
  Schreibe die Form jeder Gewichtsmatrix auf und prüfe, dass die Formen von der
  Eingabe bis zur Ausgabe ineinandergreifen.
- **Transponieren.** `W.T` vertauscht Zeilen und Spalten. Sage die Form von `W.T`
  für die `(3, 2)`-Matrix oben voraus und prüfe sie dann. Formuliere in einem
  Satz, was die Zeilen von `W.T` jetzt bedeuten.

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 4.0, 6.0])

cosine = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(cosine)   # 1.0 — b ist nur ein hochskaliertes a, die Richtung ist identisch
```

---

## Was kommt als Nächstes

Du kannst jetzt die Formen lesen, die dir der restliche Kurs entgegenwirft. In
**Mathe-Grundlagen 2** kommt die andere Hälfte: Wahrscheinlichkeit und
Erwartungswert — also das, worüber eine Belohnungskurve eigentlich mittelt und
was eine Policy tatsächlich zurückgibt.

!!! info "Selbsttest, bevor du weitergehst"
    1. Wofür steht die Dimension eines Beobachtungsvektors in Godot?
    2. Berechne \([2, 3] \cdot [4, 1]\) von Hand.
    3. Welcher Abschnitt von Neuronale Grundlagen 1 war heimlich ein Skalarprodukt?
    4. Was bewahrt das Normalisieren eines Vektors, und was wirft es weg?
    5. Eine Matrix hat die Form `(5, 3)`. Wie viele Neuronen sind das, die je wie viele Eingaben lesen?
    6. `(5, 3) @ (5,)` — funktioniert das? Warum oder warum nicht?

??? success "Antworten zum Selbsttest"
    1. Für die Anzahl der Werte, die `get_obs()` zurückgibt — ein Eintrag pro
       Sache, die dem Agenten über die Welt mitgeteilt wird.
    2. \((2)(4) + (3)(1) = 8 + 3 = 11\).
    3. Die gewichtete Summe, \(z = w_1x_1 + w_2x_2 + b\) — das Skalarprodukt von
       Eingabe- und Gewichtsvektor, plus Bias.
    4. Es bewahrt die Richtung und wirft den Betrag weg; das Ergebnis hat immer
       die Länge `1`.
    5. Fünf Neuronen, die je drei Eingaben lesen.
    6. Nein. Die inneren Dimensionen müssen passen: die Matrix hat `3` Spalten,
       der Vektor aber `5` Einträge. `(5, 3) @ (3,)` würde gehen und `(5,)`
       ergeben.

[← Einheit 0](unit-00.md) · [Kursstartseite](index.md) · [→ Mathe-Grundlagen 2](unit-math-02.md)
