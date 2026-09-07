# Mathe-Grundlagen 2 — Wahrscheinlichkeit und Erwartungswert

[← Mathe-Grundlagen 1](unit-math-01.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~25 Min · Beispiele durcharbeiten: ~20 Min

!!! tip "Überspringbar — komm zurück, wenn du es brauchst"
    Wie in der vorigen Einheit geht nichts kaputt, wenn du sie überspringst. Aber
    sobald dich eine Lernkurve verwirrt oder eine Einheit \(\mathbb{E}[\cdot]\)
    schreibt, ist das die Seite, die erklärt, was du da vor dir hast.

!!! success "Was du nach dieser Einheit kannst"
    - Sagen, warum eine Policy Wahrscheinlichkeiten statt einer Aktion zurückgibt
    - Einen Erwartungswert von Hand berechnen und erklären, was er vorhersagt
    - Erklären, was `rollout/ep_rew_mean` in TensorBoard tatsächlich misst
    - Sagen, warum eine einzelne Episode fast nichts aussagt — mit dem Wort *Varianz*
    - \(P(s' \mid s, a)\) und \(\pi(a \mid s)\) vorlesen und sagen, was beides bedeutet

!!! note "Voraussetzungen"
    - **Mathe-Grundlagen 1** hilft, ist aber nicht nötig
    - Rechnen mit Dezimalzahlen und Prozenten
    - Grundlegende Sicherheit im Terminal

!!! info "Drei Wege, die Zahlen zu sehen"
    Python-Simulation (`np.mean` über viele Läufe) · die TensorBoard-Belohnungskurve
    · die Einheiten, in denen jedes Symbol wiederkommt

Reinforcement Learning ist auf Ungewissheit gebaut. Die Umgebung kann auf
dieselbe Aktion unterschiedlich reagieren, die Policy handelt beim Erkunden
absichtlich zufällig, und Belohnungen treffen über die Zeit verstreut ein.
Wahrscheinlichkeit ist keine Komplikation, die man RL angeschraubt hat — sie ist
das Material, aus dem RL besteht.

---

## 1 · Woher der Zufall kommt

Drei getrennte Quellen, und es hilft, sie auseinanderzuhalten:

| Quelle | Was schwankt | Wo du ihr begegnest |
|---|---|---|
| Die Umgebung | Dieselbe Aktion führt in verschiedene Folgezustände | Physik-Jitter, Spawn-Positionen |
| Die Policy | Der Agent wählt bewusst verschiedene Aktionen | Exploration, [RL Essentials](unit-01.md) §5 |
| Die Belohnung | Derselbe Zustand zahlt unterschiedlich aus | Zufällige Ziele, Glücksereignisse |

Ein deterministisches Spiel kann trotzdem ein stochastisches Lernproblem
erzeugen, weil die *Policy* stochastisch ist. Früh im Training zieht der Agent
Aktionen absichtlich zufällig — genau das ist Exploration.

---

## 2 · Eine Verteilung ist eine Policy

Eine **Wahrscheinlichkeitsverteilung** ordnet jedem möglichen Ausgang eine Zahl
zu. Zwei Regeln, und nur zwei:

- jede Wahrscheinlichkeit liegt zwischen `0` und `1`;
- zusammen ergeben sie exakt `1`.

Genau diese Form hat die Ausgabe einer Policy. Zu einem Zustand gibt das Netz
nicht „springen" aus. Es gibt aus, wie wahrscheinlich jede Aktion ist:

<div class="diagram-scroll">

<svg class="course-diagram" viewBox="0 0 620 250" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="Balkendiagramm einer Policy über drei Aktionen: links mit Wahrscheinlichkeit 0.20, springen mit 0.65, rechts mit 0.15. Die drei Balken ergeben zusammen 1.0.">
  <line x1="90" y1="200" x2="560" y2="200" stroke="#8892b0" stroke-width="1.4"/>
  <line x1="90" y1="200" x2="90" y2="40" stroke="#8892b0" stroke-width="1.4"/>
  <text x="82" y="46" text-anchor="end" fill="#8892b0" font-size="12">1.0</text>
  <text x="82" y="126" text-anchor="end" fill="#8892b0" font-size="12">0.5</text>
  <text x="82" y="205" text-anchor="end" fill="#8892b0" font-size="12">0.0</text>
  <rect x="140" y="168" width="90" height="32" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="185" y="222" text-anchor="middle" fill="#8892b0" font-size="13">links</text>
  <text x="185" y="158" text-anchor="middle" fill="#6c8ef7" font-size="14" font-weight="700">0.20</text>
  <rect x="270" y="96" width="90" height="104" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="315" y="222" text-anchor="middle" fill="#8892b0" font-size="13">springen</text>
  <text x="315" y="86" text-anchor="middle" fill="#4ecca3" font-size="14" font-weight="700">0.65</text>
  <rect x="400" y="176" width="90" height="24" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="445" y="222" text-anchor="middle" fill="#8892b0" font-size="13">rechts</text>
  <text x="445" y="166" text-anchor="middle" fill="#6c8ef7" font-size="14" font-weight="700">0.15</text>
  <text x="325" y="248" text-anchor="middle" fill="#8892b0" font-size="13">0.20 + 0.65 + 0.15 = 1.00</text>
</svg>

</div>

Meistens springt dieser Agent. Manchmal nicht — und genau die übrigen `0.35`
erlauben ihm zu entdecken, dass Springen gelegentlich falsch ist.

```python
import numpy as np

actions = ["left", "jump", "right"]
policy = np.array([0.20, 0.65, 0.15])

print(policy.sum())                              # 1.0
rng = np.random.default_rng(seed=0)
print(rng.choice(actions, p=policy))             # eine gezogene Aktion
print(rng.choice(actions, size=10, p=policy))    # zehn davon — meistens "jump"
```

Führe die letzte Zeile ein paar Mal aus. Das Verhalten des Agenten ist nicht
„springen"; es ist diese Verteilung. **Die Policy ist die Verteilung, nicht die
Aktion, die zufällig herauskommt.**

!!! info "Die Notation lesen"
    \(\pi(a \mid s)\) spricht man „pi von a gegeben s": die Wahrscheinlichkeit,
    dass die Policy \(\pi\) die Aktion \(a\) wählt, wenn der Zustand \(s\) ist.
    Der senkrechte Strich bedeutet *gegeben*, nicht Division.

---

## 3 · Der Erwartungswert ist das, was RL maximiert

Der **Erwartungswert** einer zufälligen Größe ist der Durchschnitt, den du
bekämst, wenn du sie unendlich oft laufen ließest. Man berechnet ihn, indem man
jeden Ausgang mit seiner Wahrscheinlichkeit gewichtet und alles addiert:

$$
\mathbb{E}[X] = \sum_i p_i \, x_i
$$

Ein fairer sechsseitiger Würfel:

$$
\mathbb{E}[X] = \tfrac{1}{6}(1) + \tfrac{1}{6}(2) + \dots + \tfrac{1}{6}(6) = 3.5
$$

Beachte: `3.5` ist keine Seite des Würfels. **Ein Erwartungswert muss kein
möglicher Ausgang sein.** Er ist eine Vorhersage über den langfristigen
Durchschnitt, nicht über den nächsten Wurf.

```python
import numpy as np

faces = np.array([1, 2, 3, 4, 5, 6])
probabilities = np.full(6, 1 / 6)

print(probabilities @ faces)     # 3.5 — exakt, nach der Formel

rng = np.random.default_rng(seed=0)
print(rng.integers(1, 7, size=10).mean())        # 2.8     — weit daneben
print(rng.integers(1, 7, size=100_000).mean())   # 3.49805 — nah dran
```

Zwei Dinge lohnen die Aufmerksamkeit. Erstens ist `probabilities @ faces` ein
**Skalarprodukt** — dieselbe Operation wie in
[Mathe-Grundlagen 1](unit-math-01.md) §3. Ein Erwartungswert ist eine gewichtete
Summe, bei der die Gewichte Wahrscheinlichkeiten sind. Zweitens landen zehn
Stichproben nirgends in der Nähe der wahren Antwort, während hunderttausend genau
treffen. Diese Lücke ist das Thema des nächsten Abschnitts.

!!! info "Das ist die Reward-Hypothese, in Symbolen"
    [RL Essentials](unit-01.md) sagt, dass sich Ziele als Maximierung der
    erwarteten kumulativen Belohnung ausdrücken lassen. Jetzt lässt sich der
    Satz wörtlich lesen: der Agent maximiert \(\mathbb{E}[G]\), den erwarteten
    **Return**. Und da der Return zukünftige Belohnungen mit \(\gamma\)
    diskontiert, ist er selbst eine gewichtete Summe — ein Skalarprodukt
    zwischen einer Belohnungsfolge und einem Vektor von \(\gamma\)-Potenzen.

---

## 4 · Warum eine Episode nichts aussagt

Zwei Verteilungen können denselben Erwartungswert haben und sich völlig
unterschiedlich verhalten. Die **Varianz** misst, wie weit die Ausgänge typisch
um den Mittelwert streuen.

| Agent | Episoden-Returns | Mittelwert | Streuung |
|---|---|---:|---|
| A | 10, 10, 10, 10 | 10 | keine |
| B | 0, 20, 0, 20 | 10 | groß |

Beide aus einer einzelnen Episode zu beurteilen ist wertlos: Agent B sieht in der
Hälfte der Fälle wie eine Katastrophe aus und in der anderen wie ein Triumph —
und ist dabei im Mittel exakt so gut wie A.

```python
import numpy as np

rng = np.random.default_rng(seed=1)
returns = rng.choice([0.0, 20.0], size=4)

print(returns)          # [ 0. 20. 20. 20.] — ein kurzer Lauf
print(returns.mean())   # 15.0 — der wahre erwartete Return ist aber 10.0
print(rng.choice([0.0, 20.0], size=5000).mean())   # 9.916 — deutlich näher
```

Vier Stichproben melden `15.0` für eine Größe, deren wahrer Wert `10.0` ist — 50 %
Überschätzung, aus Daten, die vollkommen plausibel aussehen.

Das ist die Arithmetik hinter der Warnung in [RL Essentials](unit-01.md) §5, eine
Policy nie nach einer einzelnen frühen Episode zu beurteilen. Das ist keine
Vorsicht um ihrer selbst willen. Eine einzelne Stichprobe aus einer Verteilung
mit hoher Varianz trägt tatsächlich fast keine Information.

!!! tip "Was TensorBoard dir wirklich zeigt"
    `rollout/ep_rew_mean` ist ein **Stichprobenmittel**: der Durchschnittsreturn
    über die letzten Episoden. Er ist eine *Schätzung* des erwarteten Returns,
    nicht der erwartete Return selbst. Deshalb ist die Kurve zackig, auch wenn
    sich die Policy stetig verbessert — und deshalb lautet die nützliche Frage
    immer „wie ist der Trend über viele Punkte?" statt „ist sie seit dem letzten
    Punkt gestiegen?"

Mehr Stichproben, bessere Schätzung. Das ist das Gesetz der großen Zahlen, und es
ist der Grund, warum RL-Trainingsläufe in Hunderttausenden von Schritten gemessen
werden.

---

## 5 · Bedingte Wahrscheinlichkeit ist die Umgebung

**Bedingte Wahrscheinlichkeit** fragt: gegeben, dass etwas bereits zutrifft — wie
wahrscheinlich ist dann etwas anderes? Geschrieben \(P(B \mid A)\), gesprochen
„P von B gegeben A".

Die beiden zentralen Symbole von RL sind beides bedingte Wahrscheinlichkeiten:

| Symbol | Gesprochen | Bedeutung |
|---|---|---|
| \(P(s' \mid s, a)\) | „P von s-Strich gegeben s und a" | Die **Umgebung**: im Zustand \(s'\) landen, nachdem in \(s\) die Aktion \(a\) gewählt wurde |
| \(\pi(a \mid s)\) | „pi von a gegeben s" | Die **Policy**: Aktion \(a\) wählen, wenn der Zustand \(s\) ist |

Zusammen beschreiben sie die ganze Schleife aus [RL Essentials](unit-01.md): die
Policy entscheidet gegeben den Zustand, die Umgebung antwortet gegeben Zustand
und Aktion. In diesem Kurs *ist* \(P(s' \mid s, a)\) deine Godot-Szene — die
Physik-Engine ist die Übergangsfunktion, und du schreibst sie nie explizit auf.

\(s'\) spricht man „s Strich" und heißt einfach „das nächste s".

```python
import numpy as np

rng = np.random.default_rng(seed=2)

# Die Umgebung: "Schub" geht meistens nach oben, driftet aber manchmal.
outcomes = ["higher", "same", "lower"]
p_given_thrust = [0.70, 0.20, 0.10]

sample = rng.choice(outcomes, size=10_000, p=p_given_thrust)
print((sample == "higher").mean())    # 0.6982 — rekonstruiert P(higher | thrust)
```

Simulieren und zählen ist der Weg, eine bedingte Wahrscheinlichkeit zu *messen*,
die man nicht direkt ablesen kann. Und es ist im Kern genau das, was ein
RL-Algorithmus tut: er sieht \(P(s' \mid s, a)\) nie, er sammelt nur Stichproben
daraus.

---

## 6 · Stretch Goals

- **Die Varianz deiner eigenen Umgebung schätzen.** Lass eine trainierte
  BallChase-Policy 20 Episoden laufen und notiere jeden Return. Berechne den
  Mittelwert und von Hand, wie weit eine typische Episode davon abweicht.
  Entscheide dann, wie viele Episoden du brauchst, bevor du einem Vergleich
  zweier Belohnungsdesigns traust.
- **Die Verteilung kaputt machen.** Nimm die Policy `[0.20, 0.65, 0.15]` und
  ändere einen Eintrag so, dass die drei nicht mehr `1` ergeben. Sage voraus, was
  `rng.choice` tut, bevor du es ausführst.
- **Diskontierung als Skalarprodukt.** Bilde für Belohnungen `[1, 1, 1, 1]` und
  \(\gamma = 0.9\) den Vektor der \(\gamma\)-Potenzen und das Skalarprodukt.
  Vergleiche das Ergebnis mit \(\gamma = 0.5\) und sage in einem Satz, was sich
  an den Prioritäten des Agenten geändert hat.

```python
import numpy as np

rewards = np.array([1.0, 1.0, 1.0, 1.0])

for gamma in (0.9, 0.5):
    discounts = gamma ** np.arange(len(rewards))
    print(gamma, discounts.round(3), discounts @ rewards)
```

---

## Was kommt als Nächstes

Du hast jetzt beide Hälften der Notation, die der restliche Kurs benutzt:
Vektoren und Skalarprodukte dafür, *wie* ein Netz rechnet, Wahrscheinlichkeit und
Erwartungswert dafür, *was* es zu maximieren versucht. **Neuronale Grundlagen 1**
fängt an zu bauen — ein Neuron, jede Zahl sichtbar.

!!! info "Selbsttest, bevor du weitergehst"
    1. Nenne die drei Quellen des Zufalls in einem RL-Problem.
    2. Welche zwei Regeln muss jede Wahrscheinlichkeitsverteilung erfüllen?
    3. Berechne den Erwartungswert einer Belohnung, die mit Wahrscheinlichkeit
       `0.3` den Wert `10` auszahlt und sonst `0`.
    4. Warum kann ein Erwartungswert eine Zahl sein, die nie tatsächlich auftritt?
    5. Was genau schätzt `rollout/ep_rew_mean`?
    6. Lies \(\pi(a \mid s)\) vor und sag, was es bedeutet.

??? success "Antworten zum Selbsttest"
    1. Die Umgebung, die Policy und die Belohnung.
    2. Jede Wahrscheinlichkeit liegt zwischen `0` und `1`, und zusammen ergeben
       sie exakt `1`.
    3. \((0.3)(10) + (0.7)(0) = 3\).
    4. Weil er ein wahrscheinlichkeitsgewichteter Durchschnitt der Ausgänge ist,
       nicht einer von ihnen — wie `3.5` beim sechsseitigen Würfel.
    5. Den erwarteten Return, geschätzt als Stichprobenmittel über die letzten
       Episoden. Es ist eine Schätzung, deshalb ist die Kurve zackig.
    6. „Pi von a gegeben s" — die Wahrscheinlichkeit, dass die Policy die Aktion
       \(a\) wählt, wenn der Zustand \(s\) ist.

[← Mathe-Grundlagen 1](unit-math-01.md) · [Kursstartseite](index.md) · [→ Neuronale Grundlagen 1](unit-neural-01.md)
