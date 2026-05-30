# Unit 1 — Grundlagen des Verstärkungslernens (Reinforcement Learning)

Lerne das mentale Modell des Verstärkungslernens (reinforcement learning) und übertrage es auf Godot — aber warte nicht bis zum Ende, um *etwas zu tun*. Überfliege die Schleife, passe eine BallChase-Belohnungsfunktion (reward function) an, und lies die tiefergehenden Abschnitte, während das Training läuft.

---

!!! success "Was du nach dieser Unit kannst"
    - Die RL-Schleife erklären — *Zustand (state), Aktion (action), Belohnung (reward), nächster Zustand*
    - Die Kernbegriffe definieren: Agent (agent), Umgebung (environment), Beobachtung (observation), Aktionsraum, Belohnungsfunktion (reward function), Rückgabe (return), Richtlinie (policy), Episode (episode)
    - Den Unterschied zwischen richtlinienbasierten und wertbasierten Methoden erklären und sagen, was „Deep" hinzufügt
    - Monte Carlo von Temporal Difference unterscheiden und wissen, welche Methode PPO verwendet
    - Den Explorations-Exploit-Kompromiss erklären und die Mechanismen nennen, die PPO und DQN nutzen
    - Q-Learning, DQN, REINFORCE, PPO und MuZero in einer einzigen Taxonomie-Tabelle einordnen
    - Erklären, wie Godot und Python zusammenarbeiten, um einen Agenten zu trainieren
    - Eine vollständige Trainingseinheit durchführen und deren Fortschritt ablesen

!!! note "Voraussetzungen"
    - **Unit 0 abgeschlossen** — Conda, Godot .NET und ein erfolgreicher BallChase-Lauf
    - Grundlegende Python-Kenntnisse (Skripte ausführen, Pakete installieren) — du wirst *keine* Trainingsschleife schreiben
    - Sicherer Umgang mit einem Terminal
    - **Keine** Vorkenntnisse in Godot, Spieleentwicklung oder maschinellem Lernen erforderlich

    **Zeit:** Schnellpfad ~15 Min. Überfliegen + ~20 Min. Belohnungs-Anpassung + Training im Hintergrund. Vollständiges Lesen der Abschnitte 1–6: add ~40 Minuten. Noch nichts von Grund auf zu bauen.

!!! info "Drei Wege, deine KI zu beobachten"
    Godot-Editor (BallChase lernt live — Ball folgt dem Ziel) · TensorBoard (`rollout/ep_rew_mean` steigt von negativ in Richtung positiv) · Code (die Belohnungsanpassung, die du in Abschnitt 2 vorgenommen hast, erscheint sofort in der Kurve)

---

## ⚡ Schnellpfad (empfohlen für die erste Stunde) { #fast-path }

1. Lies nur [Abschnitt 1](#1-was-ist-verstarkungslernen) und [Abschnitt 2](#2-die-rl-prozessschleife) (~15 Min.) — Agent, Umgebung, Belohnung, Schleife.
2. Springe zu [Abschnitt 7](#7-quick-win-ballchase-recap) — ändere eine Belohnung in BallChase, starte das Training mit `--viz`.
3. Lies während des Trainings die Abschnitte 3–6 und das [Glossar](#glossar).

*Bevorzugst du lineares Lesen? Folge den Abschnitten 1–7 der Reihe nach — Abschnitt 7 enthält nach wie vor die Belohnungsanpassung.*

---

## 1 · Was ist Verstärkungslernen?

**Verstärkungslernen (Reinforcement Learning, RL)** ist eine Methode, Software beizubringen, gute Entscheidungen zu treffen: Ein **Agent** lernt, *wie er sich in einer* **Umgebung (environment)** verhalten soll, indem er *Aktionen ausführt* und *die Ergebnisse beobachtet*. Es gibt kein Lösungsbuch — der Agent lernt ausschließlich durch Versuch und Irrtum, geleitet von einer einzigen Zahl, der **Belohnung (reward)**.

Stell dir vor, wie man einen Hund trainiert. Man kann die Regeln nicht erklären; man lässt ihn Dinge ausprobieren und gibt ihm ein Leckerli, wenn er es gut macht. Nach vielen Versuchen lernt er das Verhalten, das die meisten Leckerlis einbringt. RL ist diese Idee, präzise genug für einen Computer.

!!! info "Wie sich RL vom normalen maschinellen Lernen unterscheidet"
    Überwachtes Lernen (supervised learning) benötigt beschriftete Beispiele („Dieses Bild ist eine Katze"). RL hat keine Beschriftungen — nur das Belohnungssignal. Der Agent muss gutes Verhalten *selbst entdecken*, und seine eigenen Aktionen bestimmen, welche Daten er als Nächstes sieht.

In diesem Kurs ist der Agent ein **Mondlandefahrzeug (lunar lander)**. Seine Umgebung ist eine 2D-Welt mit Schwerkraft und einem Landeplatz. Am Ende von Unit 2 wird er sich beigebracht haben, seine Triebwerke zu zünden und sanft zu landen — ohne dass jemand programmiert hat, wie das geht.

---

## 2 · Die RL-Prozessschleife

**Die Schleife: Zustand → Aktion → Belohnung → nächster Zustand**

RL läuft immer als dieselbe Schleife, die tausende Male wiederholt wird. Bei jedem Zeitschritt:

- Der Agent empfängt einen **Zustand Sₜ** von der Umgebung (z. B. Position und Geschwindigkeit des Landers).
- Basierend auf diesem Zustand führt der Agent eine **Aktion Aₜ** aus (z. B. Haupttriebwerk zünden).
- Die Umgebung wechselt in einen **neuen Zustand Sₜ₊₁**.
- Die Umgebung gibt eine **Belohnung Rₜ₊₁** zurück — eine Zahl, die angibt, wie gut das war.

<div class="diagram-scroll">

<svg class="course-diagram" viewBox="0 0 660 200" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="RL loop: agent and environment exchange actions, states, and rewards">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 1 L10 5 L0 9 z" fill="#8892b0"/>
    </marker>
  </defs>
  <rect x="60" y="70" width="200" height="64" rx="12" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="160" y="96" text-anchor="middle" fill="#e2e8f0" font-size="18" font-weight="700">AGENT</text>
  <text x="160" y="116" text-anchor="middle" fill="#8892b0" font-size="14">the lander's brain</text>
  <rect x="400" y="70" width="200" height="64" rx="12" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="500" y="96" text-anchor="middle" fill="#e2e8f0" font-size="18" font-weight="700">ENVIRONMENT</text>
  <text x="500" y="116" text-anchor="middle" fill="#8892b0" font-size="14">the Godot game world</text>
  <path d="M260 84 C320 84, 340 84, 400 84" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#ar)"/>
  <text x="330" y="74" text-anchor="middle" fill="#6c8ef7" font-size="14" font-weight="700">action Aₜ</text>
  <path d="M400 120 C340 120, 320 120, 260 120" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#ar)"/>
  <text x="330" y="148" text-anchor="middle" fill="#4ecca3" font-size="14" font-weight="700">state Sₜ₊₁ + reward Rₜ₊₁</text>
</svg>

</div>

Diese Schleife erzeugt einen Strom von *Zustand, Aktion, Belohnung, nächster Zustand*. Durch Wiederholung verschiebt sich der Agent schrittweise hin zu Aktionen, die mehr Belohnung einbringen.

**Das Ziel: die erwartete Rückgabe maximieren**

Das Ziel des Agenten ist nicht, eine große Belohnung zu kassieren — es ist, die **kumulative Belohnung** über die Zeit zu maximieren, auch **erwartete Rückgabe (expected return)** genannt.

!!! info "Die Belohnungshypothese"
    RL basiert auf einer kühnen Idee: *jedes* Ziel kann als Maximierung der erwarteten kumulativen Belohnung ausgedrückt werden. „Sanft auf dem Landeplatz landen" wird zu „die meiste Belohnung sammeln", sobald du die Belohnungsfunktion (reward function) entwirfst — was genau das ist, was du in Unit 2 tun wirst.

!!! warning "Training kommt nicht voran?"
    Prüfe der Reihe nach: (1) Vorzeichen und Skalierung der Belohnung — ist „gut" tatsächlich positiv? (2) Sparse Rewards — bekommt der Agent überhaupt ein Signal vor dem Ziel? (3) Beobachtungsfehler — werden die Sensoren nach dem Zurücksetzen aktualisiert? (4) TensorBoard flach, aber Godot sieht gut aus — du brauchst möglicherweise längeres Training oder einen Viz-Checkpoint.

**Markov-Eigenschaft & MDPs**

In wissenschaftlichen Arbeiten wird diese Schleife als **Markov-Entscheidungsprozess (Markov Decision Process, MDP)** bezeichnet. Das Wichtigste für heute: Die **Markov-Eigenschaft** bedeutet, dass der Agent *nur den aktuellen Zustand* benötigt, um seine Aktion zu wählen — nicht die gesamte Historie. Der Zustand muss also alles enthalten, was *gerade jetzt* wichtig ist.

---

## 3 · Monte Carlo vs. Temporal Difference

Sobald der Agent Erfahrungen gesammelt hat, muss er *daraus lernen* — das heißt, seine Schätzung darüber aktualisieren, wie gut ein Zustand ist. Es gibt zwei grundlegend verschiedene Strategien dafür.

### Monte Carlo (MC)

Die Idee: **warte, bis die Episode beendet ist**, und verwende dann die tatsächliche Rückgabe für die Aktualisierung.

Am Ende einer Episode kennst du die tatsächliche Gesamtbelohnung, die ab Schritt *t* gesammelt wurde:

```
Gₜ = rₜ + γ rₜ₊₁ + γ² rₜ₊₂ + … + γᵀ rᵀ
```

Du verschiebst dann deine Wertschätzung V(sₜ) ein wenig in Richtung dieser tatsächlichen Rückgabe Gₜ.

**Vorteile:**
- **Unverzerrend (unbiased)** — du hast die echte Rückgabe verwendet, keine Näherung.
- Einfach zu verstehen und zu implementieren.

**Nachteile:**
- Muss auf die **vollständige Episode** warten, bevor irgendeine Aktualisierung erfolgt — langsam.
- **Hohe Varianz** — eine einzelne glückliche oder unglückliche Episode kann deine Schätzungen stark verzerren.
- Funktioniert nicht bei **fortlaufenden Aufgaben** (kein Episodenende).

### Temporal Difference (TD)

Die Idee: **aktualisiere nach jedem einzelnen Schritt** mit einer *bootstrapped* Schätzung — das heißt, verwende deine *aktuelle* Wertschätzung für den nächsten Zustand, anstatt auf die echte Rückgabe zu warten.

Das **TD-Ziel** ist:

```
rₜ + γ V(sₜ₊₁)
```

Der **TD-Fehler** δₜ misst, wie falsch deine aktuelle Schätzung war:

```
δₜ = rₜ + γ V(sₜ₊₁) − V(sₜ)
```

Du verschiebst dann V(sₜ) um einen kleinen Schritt in Richtung von δₜ.

**Vorteile:**
- Funktioniert bei **fortlaufenden Aufgaben** — kein Episodenende erforderlich.
- **Online-Lernen** — Aktualisierungen fließen ein, während der Agent handelt.
- **Geringere Varianz** als MC, weil du nicht auf eine vollständige, verrauschte Trajektorie wartest.

**Nachteile:**
- **Verzerrt (biased)** — du bootstrappst auf einer unvollkommenen V-Schätzung. Zu Beginn des Trainings ist diese Schätzung falsch, also ist auch das Ziel falsch.

### Die Intuition

!!! info "MC vs. TD auf Deutsch erklärt"
    **Monte Carlo** ist wie das Warten, bis das Schachspiel vorbei ist, und dann das Aktualisieren des mentalen Modells für jede gespielte Stellung.

    **Temporal Difference** ist wie das Aktualisieren der Einschätzung der eigenen Stellung *nach jedem einzelnen Zug*, basierend darauf, wie gut die Stellung *gerade jetzt* aussieht.

    Keine Methode ist immer besser — die richtige Wahl hängt von der Aufgabenstruktur, der Episodenlänge und der tolerierbaren Varianz ab.

### Welche Methode verwendet dieser Kurs?

| Algorithmus | Familie | Verwendet in |
|-------------|---------|--------------|
| PPO (RecurrentPPO) | Multi-step TD | Units 2, 5 |
| Q-Learning | Single-step TD | Unit 3 (Theorie) |
| DQN | Single-step TD | Unit 3 |
| REINFORCE | Monte Carlo | Hintergrundlektüre |

PPO sammelt einen Rollout fester Länge (ein Fenster von Schritten, keine vollständige Episode) und führt dann eine Aktualisierung durch — das macht es zu einer Multi-step-TD-Methode. Du siehst `n_steps` in der PPO-Konfiguration; das ist die Rollout-Länge.

!!! tip "TensorBoard lesen"
    Die `ep_rew_mean`-Kurve ist die gemittelte episodische Rückgabe — eine Monte-Carlo-Größe. Aber das *Lernsignal* innerhalb von PPO ist TD-basiert. Wenn `ep_rew_mean` anfangs langsam steigt und dann beschleunigt, beobachtest du, wie TD-Bootstrapping sich schrittweise verbessert, während die Wertschätzung besser wird. Wir untersuchen das eingehend in der [Q-Learning-Unit](unit-03.md).

---

## 4 · Die Bausteine

**Beobachtungen und Zustände**

Beide sind die Informationen, die der Agent von der Umgebung erhält:

- **Zustand (state)** — eine *vollständige* Beschreibung der Welt, nichts verborgen (z. B. ein Schachbrett: du siehst alles).
- **Beobachtung (observation)** — eine *partielle* Beschreibung (z. B. Super Mario: du siehst nur den Teil des Levels in der Nähe des Spielers).

Der Kurs verwendet „Zustand" locker für beides. Unser Lander meldet eine *Beobachtung*: 8 Zahlen (Position, Geschwindigkeit, Winkel, Bodenkontakt). Die Menge aller möglichen Beobachtungen ist der **Beobachtungs-/Zustandsraum**.

**Aktionsraum — diskret vs. kontinuierlich**

| Typ | Bedeutung | Beispiel |
|-----|-----------|---------|
| **Diskret** | Eine endliche Liste von Aktionen | Lunar Lander: *nichts tun / linkes Triebwerk / Haupttriebwerk / rechtes Triebwerk* — 4 Aktionen |
| **Kontinuierlich** | Unendlich viele Aktionen | Ein selbstfahrendes Auto: lenke 20,0°, 20,1°, 20,15°, … |

Unser Lander verwendet einen **diskreten** Aktionsraum der Größe 4. Ob ein Raum diskret oder kontinuierlich ist, beeinflusst, welchen RL-Algorithmus du später wählst.

**Belohnungen, Rückgabe und Abzinsung**

Die Belohnung ist das *einzige* Feedback, das der Agent erhält. Die **Rückgabe (return)** ist die Summe aller zukünftigen Belohnungen.

Zukünftige Belohnungen sind jedoch ungewiss, daher **diskontieren** wir sie: Multiplikation mit einem **Abzinsungsfaktor (discount factor) γ (gamma)**, zwischen 0 und 1 — üblicherweise **0,95–0,99**.

- γ nahe 1 → der Agent kümmert sich um den *langen Zeithorizont*.
- γ kleiner → der Agent kümmert sich um *unmittelbare* Belohnungen.

!!! tip "Die klassische Intuition: Maus, Käse, Katze"
    Eine Maus will Käse. Käse in der Nähe der Katze ist mehr wert — aber er ist riskant und weit weg, also wird seine Belohnung *diskontiert*. Käse in der Nähe ist eine sicherere Wette. Abzinsung erfasst: „Eine Belohnung, die ich vielleicht nie erreiche, ist weniger wert."

**Episodische vs. fortlaufende Aufgaben**

- **Episodisch** — es gibt einen klaren Anfang und ein Ende (eine **Episode**). Unser Lander: Eine Episode endet, wenn er landet, abstürzt oder die Zeit abläuft, dann wird sie zurückgesetzt.
- **Fortlaufend** — kein Ende; die Aufgabe läuft ewig (z. B. ein Aktienhandels-Agent).

Alles in diesem Kurs ist **episodisch**.

---

## 5 · Wie ein Agent lernt

**Die Richtlinie π — das Gehirn des Agenten**

Die **Richtlinie (policy) π** ist die Funktion, die dem Agenten sagt, welche Aktion er in einem gegebenen Zustand ergreifen soll. Das Training hat eine einzige Aufgabe: die **optimale Richtlinie π\*** finden — diejenige, die die meiste erwartete Rückgabe erzielt.

Eine Richtlinie kann **deterministisch** sein (ein Zustand liefert immer dieselbe Aktion) oder **stochastisch** (sie gibt eine Wahrscheinlichkeitsverteilung über Aktionen aus). Der Lander, den du trainierst, verwendet eine stochastische Richtlinie.

---

### Exploration vs. Exploitation (Erkundung vs. Ausbeutung)

Bevor der Agent eine gute Richtlinie ausnutzen kann, muss er zuerst *entdecken*, was gut ist — und das erfordert Exploration. Diese Spannung ist eine der grundlegendsten Herausforderungen im RL.

**Das Dilemma in einem Satz:** bekannte gute Aktionen ausnutzen und jetzt Belohnung sammeln, oder unbekannte Aktionen erkunden, die noch besser sein könnten.

#### ε-greedy-Exploration

Die klassische Lösung, die in **DQN** (Unit 3) verwendet wird:

- Mit Wahrscheinlichkeit **ε** wird eine *zufällige* Aktion ausgeführt (Exploration).
- Mit Wahrscheinlichkeit **1 − ε** wird die *bestbekannte* Aktion ausgeführt (Exploitation).

| ε-Wert | Verhalten |
|--------|-----------|
| 1,0 | Reine zufällige Exploration — der Agent ignoriert alles, was er gelernt hat |
| 0,5 | Halb zufällig, halb gierig |
| 0,05 | Überwiegend Exploitation, mit kleiner Chance, etwas Neues auszuprobieren |
| 0,0 | Reine Exploitation — gierig, erkundet nie |

In der Praxis wird ε während des Trainings **abgebaut (annealed)** (zerfällt): beginnt bei 1,0, endet bei 0,05. Das erlaubt dem Agenten, den Zustandsraum anfangs frei zu erkunden und sich dann auf die gelernte Richtlinie festzulegen.

#### Entropiebasierte Exploration (PPO)

PPO verwendet *kein* ε-greedy. Stattdessen arbeitet es mit **stochastischen Richtlinien** — das Netzwerk gibt eine Wahrscheinlichkeitsverteilung über Aktionen aus, keine einzelne Aktion.

Die **Entropie** dieser Verteilung misst, wie breit sie gestreut ist:

- **Hohe Entropie** = die Verteilung ist flach = der Agent erkundet viele Aktionen gleichmäßig.
- **Niedrige Entropie** = die Verteilung ist spitz = der Agent ist zuversichtlich und setzt auf eine Aktion.

PPO fügt einen **Entropie-Bonus** zur Verlustfunktion hinzu. Das bestraft die Richtlinie dafür, zu früh auf eine schmale Verteilung zu kollabieren, und hält die Exploration während des gesamten Trainings aufrecht.

```
PPO loss = policy gradient − c_entropy × entropy
```

Der Koeffizient `c_entropy` (oft `ent_coef` in stable-baselines3) steuert, wie stark auf Exploration gedrängt wird. Du wirst diesen in späteren Units anpassen.

#### Neugierbasierte Exploration (Bonuskonzept)

Eine dritte Methodenfamilie gibt dem Agenten eine **intrinsische Belohnung** dafür, neuartige Zustände zu besuchen — Zustände, die er noch nicht gesehen hat oder nicht gut vorhersagen kann. Das ist besonders nützlich, wenn die extrinsische Belohnung sehr dünn (sparse) ist (z. B. ein Puzzlespiel, bei dem die Belohnung erst ganz am Ende kommt). Wir verwenden neugierbasierte Exploration nicht in diesem Kurs, aber es lohnt sich, die Idee zu kennen.

!!! info "Welche Methode wird wo verwendet"
    - **DQN** (Unit 3) — ε-greedy mit linearem ε-Abbau
    - **PPO** (Units 2, 5) — Entropie-Bonus über `ent_coef`
    - **Curiosity / RND** — fortgeschritten, in diesem Kurs nicht behandelt

---

**Zwei Wege zur optimalen Richtlinie**

| Ansatz | Was er lernt | Wie er handelt |
|--------|-------------|----------------|
| **Richtlinienbasiert (policy-based)** | Die Richtlinie direkt — eine Zustand-→-Aktion-Abbildung | Richtlinie nach einer Aktion fragen |
| **Wertbasiert (value-based)** | Eine Wertfunktion (value function) — wie gut jeder Zustand ist | In Richtung des Zustands mit dem höchsten Wert bewegen |

Beide zielen auf dieselbe optimale Richtlinie π\*. Unit 2 verwendet **PPO**, eine richtlinienbasierte Methode. Spätere Units erkunden wertbasierte Methoden wie DQN.

---

### Wertbasiert vs. Richtlinienbasiert vs. Actor-Critic: die vollständige Taxonomie

In der Praxis fallen RL-Algorithmen in vier Familien. Die Karte zu kennen hilft dir, wissenschaftliche Arbeiten zu lesen, Algorithmen auszuwählen und TensorBoard-Metriken zu verstehen.

| Familie | Was sie lernt | Repräsentative Algorithmen | Wann verwenden |
|---------|--------------|---------------------------|----------------|
| **Wertbasiert (value-based)** | Q(s,a) oder V(s) — wie gut jeder Zustand/jede Aktion ist | Q-Learning, DQN, Double DQN | Diskrete Aktionen; stichprobeneffizient; gut für einfache Umgebungen |
| **Richtlinienbasiert (policy-based)** | π_θ(a\|s) direkt — die Richtlinie als neuronales Netz | REINFORCE, TRPO | Kontinuierliche oder stochastische Aktionsräume; keine Wertfunktion benötigt |
| **Actor-Critic** | Sowohl π als auch V gleichzeitig — der Aktor handelt, der Kritiker bewertet | A2C, PPO, SAC | Das Beste beider Familien; geringere Varianz als rein richtlinienbasiert; heute der Standard |
| **Modellbasiert (model-based)** | Ein Dynamikmodell p(s'\|s,a) — vorhersagen, was als Nächstes passiert | Dyna, World Models, MuZero | Hochgradig stichprobeneffizient; nützlich, wenn Simulation teuer ist |

**Dieser Kurs** konzentriert sich auf **Actor-Critic** (PPO), weil es der Branchenstandard für Spiel-KI ist. Unit 3 lehrt auch **DQN** (wertbasiert), damit du den Unterschied spüren kannst. Modellbasierte Methoden werden kurz im fortgeschrittenen Abschnitt behandelt.

!!! info "Warum Actor-Critic?"
    Eine rein richtlinienbasierte Methode (REINFORCE) hat hohe Varianz — jede Aktualisierung basiert auf einer einzelnen verrauschten Trajektorie. Eine rein wertbasierte Methode (DQN) funktioniert nicht einfach mit kontinuierlichen Aktionen. Actor-Critic erzielt niedrige Varianz (durch die Wertschätzung des Kritikers) *und* handhabt jeden Aktionsraum (durch die Richtlinie des Aktors). PPO ist der beliebteste Actor-Critic-Algorithmus in Spielen, weil er auch stabil und stichprobeneffizient ist.

---

**Was „Deep" beim Deep RL bedeutet**

Traditionelles RL pflegte eine **Nachschlagetabelle**: eine Zeile pro Zustand, eine Spalte pro Aktion, in der Q-Werte (Q-values) oder V-Werte gespeichert werden. Das funktioniert gut für kleine Probleme — ein Schachendspiel mit einigen tausend Stellungen oder eine einfache Gitterwelt.

Es versagt vollständig, sobald der Zustandsraum groß wird. Ein Godot-Spiel mit 8 kontinuierlichen Sensorwerten hat *unendlich* viele mögliche Zustände. Man kann keine Zeile für jeden haben.

**Deep RL** ersetzt die Tabelle durch ein **neuronales Netz**:

- **Eingabe:** der rohe Beobachtungsvektor (8 Zahlen für den Lander, Pixel-Arrays für visionsbasierte Agenten, Raycast-Abstände für einen Roboter).
- **Ausgabe:** Aktionswahrscheinlichkeiten (für ein Aktor-/Richtlinien-Netzwerk) *oder* eine Wertschätzung (für ein Kritiker-/Wertfunktions-Netzwerk).

Das Netzwerk *verallgemeinert* — es lernt, für ähnliche Beobachtungen ähnliche Ausgaben zu liefern, was eine Tabelle nicht kann.

**Warum „deep"?** Weil das Netzwerk mehrere verborgene Schichten hat (es ist im neuronalen Netz-Sinne „tief"). Ein flaches Ein-Schicht-Netz kann die komplexen nichtlinearen Funktionen nicht darstellen, die die meisten Spiel-Richtlinien erfordern.

!!! info "Der Kompromiss, den du mit Deep RL eingehst"
    | Tabellarisches RL | Deep RL |
    |-------------------|---------|
    | Exakt, konvergiert nachweisbar | Näherungsweise, kann divergieren |
    | Nur winzige Zustandsräume | Jeder Zustandsraum (Pixel, Sensoren) |
    | Keine Hyperparameter | Lernrate, Architektur, ent_coef, … |
    | Sofortige Aktualisierungen | Benötigt tausende von Gradientenschritten |

    Für jede Spielumgebung jenseits von Spielzeugproblemen ist Deep RL die einzige praktische Option.

!!! tip "Was das für das Debugging bedeutet"
    Wenn das Training stagniert, liegt es oft nicht am RL-Algorithmus — das neuronale Netz schafft es nicht, eine nützliche Darstellung zu lernen. Prüfe: (1) Beobachtungsskalierung (Eingaben sollten ungefähr im Bereich −1 bis 1 liegen), (2) Belohnungsskalierung (sehr große Belohnungen verursachen Gradientenexplosion), (3) Netzwerkgröße (zu klein = Underfitting, zu groß = langsam und instabil).

**Ein Wort zu PPO**

!!! info "PPO in einem Satz"
    **Proximal Policy Optimization (PPO)** ist ein beliebter Actor-Critic-Algorithmus, der die Richtlinie in kleinen, sicheren Schritten verbessert, damit das Training stabil bleibt. Du wirst ihn nicht implementieren — die `stable-baselines3`-Bibliothek stellt ihn bereit, und `godot-rl` verbindet alles.

---

## 6 · Wie Godot RL Agents zusammenarbeitet

Godot RL Agents verwendet zwei Programme, die nebeneinander laufen und über einen schnellen lokalen Netzwerk-Socket kommunizieren. Die **Godot Engine** ist die Umgebung — sie rendert die Welt, führt die Physik aus und meldet Beobachtungen und Belohnungen. **Python** ist das Gehirn — es führt PPO aus und entscheidet die Aktionen.

<div class="diagram-scroll">

<svg class="course-diagram" viewBox="0 0 720 360" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="Godot environment and Python training process connected by socket">
  <defs>
    <marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 1 L10 5 L0 9 z" fill="#8892b0"/>
    </marker>
    <marker id="ar3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 1 L10 5 L0 9 z" fill="#f7c86c"/>
    </marker>
  </defs>
  <!-- Godot -->
  <rect x="20" y="16" width="290" height="238" rx="14" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="165" y="44" text-anchor="middle" fill="#e2e8f0" font-size="16" font-weight="700">Godot Engine</text>
  <text x="165" y="62" text-anchor="middle" fill="#8892b0" font-size="13">Environment</text>
  <rect x="40" y="76" width="250" height="66" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="165" y="102" text-anchor="middle" fill="#e2e8f0" font-size="14">Lander scene + AIController2D</text>
  <text x="165" y="124" text-anchor="middle" fill="#8892b0" font-size="12">observations · actions · reward</text>
  <rect x="40" y="154" width="250" height="66" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="165" y="180" text-anchor="middle" fill="#e2e8f0" font-size="14">Sync node</text>
  <text x="165" y="202" text-anchor="middle" fill="#8892b0" font-size="12" font-family="monospace">godot_rl_agents</text>
  <!-- Python -->
  <rect x="410" y="16" width="290" height="238" rx="14" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="555" y="44" text-anchor="middle" fill="#e2e8f0" font-size="16" font-weight="700">Python</text>
  <text x="555" y="62" text-anchor="middle" fill="#8892b0" font-size="13">Agent's brain</text>
  <rect x="430" y="76" width="250" height="66" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="555" y="102" text-anchor="middle" fill="#e2e8f0" font-size="14">godot-rl wrapper</text>
  <text x="555" y="124" text-anchor="middle" fill="#8892b0" font-size="12" font-family="monospace">gdrl</text>
  <rect x="430" y="154" width="250" height="66" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="555" y="180" text-anchor="middle" fill="#e2e8f0" font-size="14">Stable-Baselines3</text>
  <text x="555" y="202" text-anchor="middle" fill="#8892b0" font-size="12">PPO · neural-net policy</text>
  <!-- Socket traffic (wider gutter) -->
  <path d="M310 112 L410 112" fill="none" stroke="#8892b0" stroke-width="1.8" marker-end="url(#ar2)"/>
  <text x="360" y="98" text-anchor="middle" fill="#4ecca3" font-size="13" font-weight="700">obs + reward</text>
  <path d="M410 192 L310 192" fill="none" stroke="#8892b0" stroke-width="1.8" marker-end="url(#ar2)"/>
  <text x="360" y="208" text-anchor="middle" fill="#6c8ef7" font-size="13" font-weight="700">actions</text>
  <text x="360" y="144" text-anchor="middle" fill="#8892b0" font-size="12">socket</text>
  <text x="360" y="160" text-anchor="middle" fill="#8892b0" font-size="12">port 11008</text>
  <!-- ONNX export (below panels) -->
  <path d="M555 262 C555 288, 165 288, 165 262" fill="none" stroke="#f7c86c" stroke-width="1.6" stroke-dasharray="5,4" marker-end="url(#ar3)"/>
  <text x="360" y="318" text-anchor="middle" fill="#f7c86c" font-size="13" font-weight="700">After training: ONNX model runs in Godot</text>
  <text x="360" y="338" text-anchor="middle" fill="#f7c86c" font-size="12">no Python process required</text>
</svg>

</div>

**Jedes Konzept hat seinen Platz in Godot**

| RL-Konzept | Wo es in Godot RL lebt |
|------------|------------------------|
| Umgebung (environment) | Deine Godot-Szene (der Lander, der Boden, der Landeplatz) |
| Beobachtung (observation) | `get_obs()` im AIController-Skript |
| Aktionsraum | `get_action_space()` im AIController-Skript |
| Belohnung (reward) | `reward` wird in deiner Spiellogik aktualisiert |
| Episodenende | Die `done`- / `needs_reset`-Flags |
| Agent / Richtlinie | Das PPO-neuronale Netz, in Python während des Trainings |
| Kommunikationsschleife | Der **Sync**-Knoten ↔ Python-Socket |

Genau das wirst du in Unit 2 Schritt für Schritt erstellen. Nach dem Training wird die Richtlinie in eine **ONNX**-Datei exportiert und läuft direkt in Godot — kein Python erforderlich, um das fertige Spiel zu spielen.

---

## 7 · Quick win: BallChase-Recap

!!! tip "Dein erster Eigentümermoment"
    In Unit 0 hast du BallChase ausgeführt. Hier änderst du das Belohnungssignal und siehst, wie sich das Verhalten verändert — das verknüpft Vokabular mit Code.

**Eine Belohnung anpassen (mach das vor einem langen Wiederlesen)**

1. Klone oder öffne [BallChase](https://github.com/edbeeching/godot_rl_agents_examples/tree/main/examples/BallChase) in Godot .NET.
2. Finde das Skript, das `reward` am Agenten setzt (oft am Spieler oder `AIController`).
3. Ändere einen Term — z. B. verdopple die Belohnung dafür, dem Ball näher zu kommen, oder füge eine kleine Strafe pro Schritt hinzu.
4. Notiere, was du erwartest: schnelleres Verfolgen, mehr Umherwandern usw.

**BallChase erneut ausführen** *(überspringen, wenn Unit 0 bereits erfolgreich war)*

```bash
conda activate godot_env
python examples/stable_baselines3_example.py \
  --env_path=examples/godot_rl_BallChase/bin/BallChase.x86_64 \
  --experiment_name=unit1-recap --timesteps=100000 --viz
```

**Beim Lernen zusehen (drei Ansichten)**

- **Godot** — mit `--viz` bestätigen, dass die Bewegung deiner Belohnungsänderung entspricht.
- **TensorBoard** — `ep_rew_mean` kann nach einer harten Belohnung niedriger starten; er sollte steigen, während sich die Richtlinie anpasst.
- **Code** — du kannst auf die Zeile zeigen, die du bearbeitet hast, wenn du die MDP-Schleife erklärst.

!!! success "Checkpoint — du bist bereit für Unit 2"
    Du hast eine Belohnung geändert, neu trainiert und gesehen, dass mindestens zwei der drei Ansichten aktualisiert wurden. In **Unit 2** wirst du dich zuerst mit **SimpleReachGoal** aufwärmen (ausführen + anpassen) und dann Lunar Lander von Grund auf bauen.

---

## Glossar { #glossar }

| Begriff | Bedeutung |
|---------|-----------|
| **Agent** | Der Lernende / Entscheider — hier der Lander. |
| **Umgebung (environment)** | Die Welt, in der der Agent handelt — hier die Godot-Szene. |
| **Zustand / Beobachtung (state / observation)** | Die Informationen, die der Agent erhält. Ein Zustand ist vollständig; eine Beobachtung ist partiell. |
| **Aktion (action)** | Eine Wahl, die der Agent trifft und die die Umgebung verändert. |
| **Aktionsraum** | Die Menge aller möglichen Aktionen — diskret (endlich) oder kontinuierlich (unendlich). |
| **Belohnung (reward)** | Die einzelne Zahl, die bewertet, wie gut eine Aktion war — das einzige Feedback des Agenten. |
| **Rückgabe (return)** | Die (diskontierte) Summe aller zukünftigen Belohnungen — was der Agent maximiert. |
| **Abzinsungsfaktor (discount rate) γ** | Faktor 0–1, der zukünftige Belohnungen weniger als unmittelbare zählen lässt. |
| **Richtlinie (policy) π** | Das Gehirn des Agenten — die Funktion, die einen Zustand auf eine Aktion abbildet. |
| **Episode** | Ein Durchlauf vom Start bis zu einem Endzustand (landen, abstürzen oder Zeitlimit). |
| **MDP** | Markov-Entscheidungsprozess — der formale Name für die RL-Schleife. |
| **Monte Carlo (MC)** | Lernen aus vollständigen Episoden; unverzerrend, aber hohe Varianz und langsam. |
| **Temporal Difference (TD)** | Lernen nach jedem Schritt mit bootstrapped Schätzungen; geringere Varianz, funktioniert online. |
| **TD-Fehler δ** | Die Differenz zwischen dem TD-Ziel und der aktuellen Wertschätzung: rₜ + γ V(sₜ₊₁) − V(sₜ). |
| **Bootstrapping** | Die aktuelle Schätzung des zukünftigen Wertes als Ziel verwenden, anstatt auf die echte Rückgabe zu warten. |
| **Exploration (Erkundung)** | Aktionen ausprobieren, die man nicht gut kennt, um bessere Strategien zu entdecken. |
| **Exploitation (Ausbeutung)** | Das nutzen, was man bereits weiß, um Belohnungen zu sammeln. |
| **ε-greedy** | Explorationsstrategie: zufällige Aktion mit Wahrscheinlichkeit ε, sonst die bestbekannte Aktion. |
| **Entropie-Bonus (entropy bonus)** | PPOs Explorationsmechanismus: bestraft übermäßig zuversichtliche Richtlinien, um die Exploration aufrechtzuerhalten. |
| **Richtlinienbasiert (policy-based)** | Methoden, die die Richtlinie π direkt lernen (z. B. REINFORCE, PPO). |
| **Wertbasiert (value-based)** | Methoden, die Q(s,a) oder V(s) lernen und Verhalten daraus ableiten (z. B. DQN). |
| **Actor-Critic** | Methoden, die sowohl π als auch V gleichzeitig lernen (z. B. A2C, PPO). |
| **PPO** | Proximal Policy Optimization — der stabile Actor-Critic-Algorithmus, der in Unit 2 verwendet wird. |
| **Deep RL** | RL, bei dem die Richtlinie oder Wertfunktion ein neuronales Netz ist. |
| **Rollout** | Ein Erfahrungs-Batch, der vor jeder PPO-Aktualisierung gesammelt wird. |
| **ONNX** | Ein portables Modellformat — lässt die trainierte Richtlinie in Godot ohne Python laufen. |

---

## 8 · Stretch Goals (Zusatzaufgaben)

**Die RL-Schleife von Hand implementieren.** Schreibe ohne SB3 oder gdrl eine einfache Python-Schleife, die durch eine gymnasium-Umgebung schreitet und Rückgaben akkumuliert. FrozenLake-v1 (diskret, klein) oder CartPole-v1 (kontinuierliche Beobachtung, diskrete Aktion) sind ideal. Du musst nichts lernen — verwende eine zufällige Richtlinie. Das Ziel ist es, `env.reset()`, `env.step(action)`, `obs`, `reward`, `done` in echtem Code zu sehen, bevor ein Framework sie verbirgt.

```python
import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")
obs, _ = env.reset()
total_reward = 0.0

for _ in range(500):
    action = env.action_space.sample()   # random policy
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"Episode return: {total_reward}")
env.close()
```

**Taxonomie-Selbsttest.** Decke die Algorithmustabelle in Abschnitt 6 ab und beantworte diese fünf Fragen aus dem Gedächtnis:
1. Welcher Algorithmus hat den Replay Buffer eingeführt?
2. Ist PPO on-policy oder off-policy — und warum ist das für den Datenbedarf wichtig?
3. Was fügt „Deep" dem Q-Learning hinzu, das tabellarisches Q-Learning fehlt?
4. Nenne einen modellbasierten Algorithmus und erkläre, welches „Modell" er lernt.
5. Welchen der Kursalgorithmen würdest du zuerst für eine Roboterarm-Aufgabe verwenden — und warum?

Vergleiche deine Antworten mit Abschnitt 6. Alles, was du falsch hattest, lohnt sich erneut zu lesen.

**Belohnungssensitivitäts-Experiment.** Multipliziere in deiner BallChase-Trainingsszene den Belohnungskoeffizienten mit 10 (mache ihn größer). Sage vorher, was mit der TensorBoard-Kurve passieren wird. Führe dann ein kurzes Training (200k Schritte) durch und vergleiche. Wiederhole mit dem Koeffizienten geteilt durch 10 (winziges Belohnungssignal). Erkläre, was du in Bezug auf das Signal-Rausch-Verhältnis der Gradientenaktualisierungen beobachtest.

## Was kommt als Nächstes

Du hast das Vokabular, eine Belohnungsanpassung vorgenommen und einen trainierten Lauf als Referenz. In **Unit 2** beginnst du mit **SimpleReachGoal** (ausführen + anpassen) und baust dann Lunar Lander von Grund auf.

!!! info "Selbstcheck, bevor du weitermachst"
    Kannst du diese Fragen mit eigenen Worten beantworten?

    1. Was sind die vier Teile der RL-Schleife?
    2. Was ist der Unterschied zwischen einer Beobachtung und einem Zustand?
    3. Was steuert der Abzinsungsfaktor γ?
    4. Was ist eine Richtlinie (policy)?
    5. Was ist der wesentliche Unterschied zwischen Monte Carlo- und TD-Lernen?
    6. Wie fördert PPO die Exploration ohne ε-greedy?
    7. Nenne einen Algorithmus aus jeder der vier RL-Familien (wertbasiert, richtlinienbasiert, Actor-Critic, modellbasiert).

    Wenn du alle sieben beantworten kannst — bist du bereit.

[→ Belohnungsdesign](unit-reward-engineering.md)
