# RL-Grundlagen Deep Dive

[← Neuronale Grundlagen 3](unit-neural-03.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~45 min · Am besten gelesen, nachdem du einen Grundlagen-3-Punktroboter- oder Racer-Lauf zum Inspizieren hast

!!! success "Was du nach dieser Unit kannst"
    - Monte Carlo von Temporal Difference unterscheiden
    - Bootstrapping und TD-Fehler aus einer Trajektorie erklären
    - ε-greedy, Entropie und neugiergetriebene Exploration vergleichen
    - gängige Algorithmen in value-based, policy-based, Actor-Critic- und model-based-Familien einordnen
    - erklären, warum On-Policy- und Off-Policy-Methoden Daten unterschiedlich wiederverwenden

!!! note "Voraussetzungen"
    - **RL Essentials abgeschlossen** — Beobachtungen, Aktionen, Belohnungen, Episoden, Return und Policy
    - **Neuronale Grundlagen 3 abgeschlossen** — du hast eine Punktroboter-Trajektorie oder Racer-Lernkurve als Referenz

!!! info "Drei Wege, deine KI zu beobachten"
    Trajektorien-Spuren · Lernkurven · Algorithmus-Familien hinter dem Trainingslauf

Diese Seite gibt Namen für die tieferen Mechanismen, die du gerade beobachtet hast. Halte ein Grundlagen-3-Artefakt offen: entweder eine Punktroboter-Trajektorie mit Schrittbelohnungen oder eine Racer-Lernkurve mit Episoden-Returns.

---

## 1 · Deinen Grundlagen-3-Lauf lesen

Vor der Theorie sammle Nachweise aus deinem Lauf:

- eine Trajektorie oder ein Replay;
- die Belohnung bei mehreren Schritten;
- den Episoden-Return;
- die Lernkurve über viele Episoden;
- einen Fehlerfall.

Stelle zwei Fragen:

1. Hat sich diese Episode verbessert, weil die Policy bessere lokale Entscheidungen traf?
2. Hat sich das Training verbessert, weil der Algorithmus aus ganzen Trajektorien, schrittweisen Schätzungen oder beidem lernte?

Der Rest dieser Seite gibt dir das Vokabular, um diese Fragen zu beantworten.

---

## 2 · Monte Carlo versus Temporal Difference

**Monte Carlo** wartet, bis eine Episode endet, und lernt dann aus dem tatsächlichen Return. Beim Grundlagen-3-Punktroboter bedeutet das: Ein Zustand nahe dem Start bekommt sein Trainingsziel erst, nachdem der Roboter das Ziel erreicht, crasht oder ein Timeout bekommt.

Für Belohnungen ab Schritt `t`:

```text
G_t = r_t + γr_t+1 + γ²r_t+2 + ...
```

Monte Carlo nutzt dieses vollständige `G_t` als Ziel.

**Temporal Difference** aktualisiert nach einem Schritt, indem es die unmittelbare Belohnung mit der aktuellen Schätzung des nächsten Zustands kombiniert:

```text
TD target = r_t + γV(s_t+1)
```

Auf einer Racer-Lernkurve helfen TD-ähnliche Updates, die Wertschätzung zu verbessern, bevor jedes mögliche Runden-Ergebnis gesehen wurde.

| Methode | Lernt aus | Stärke | Schwäche |
|--------|-------------|----------|----------|
| Monte Carlo | abgeschlossenen Episoden | unverzerrtes Ziel | wartet auf Episodenende, hohe Varianz |
| Temporal Difference | jedem Übergang | Online-Updates, geringere Varianz | verzerrt durch aktuelle Schätzungen |

---

## 3 · Bootstrapping und TD-Fehler

**Bootstrapping** bedeutet, die aktuelle Schätzung als Teil des neuen Ziels zu nutzen. Statt auf den wahren vollen Return zu warten, fragt TD: „Belohnung jetzt, plus was ich gerade denke, dass der nächste Zustand wert ist."

Der **TD-Fehler** misst die Überraschung:

```text
δ_t = r_t + γV(s_t+1) - V(s_t)
```

Mache das mit der Punktroboter-Trajektorie konkret:

- wenn ein Vorwärtsschritt näher ans Ziel führt, kann `r_t` positiv sein;
- wenn die nächste Beobachtung in freien Raum zeigt, kann `V(s_t+1)` höher sein;
- wenn der alte Wert `V(s_t)` zu pessimistisch war, wird `δ_t` positiv.

Ein positiver TD-Fehler sagt: „Dieser Zustand war besser als erwartet." Ein negativer TD-Fehler sagt: „Dieser Zustand war schlechter als erwartet."

!!! tip "Eine Kurve lesen"
    Wenn eine Racer-Lernkurve langsam steigt und dann beschleunigt, kann ein Grund
    verbessertes Bootstrapping sein: bessere Wertschätzungen machen spätere TD-Ziele
    weniger noisy.

---

## 4 · Explorationsmechanismen

Exploration ist in Grundlagen 3 sichtbar, wann immer der Punktroboter eine schlechte
Drehung sampelt oder der Racer eine Lenkaktion probiert, die nicht zum aktuellen besten
Verhalten passt. Verschiedene Algorithmen fördern diese Exploration auf unterschiedliche Weise.

### ε-greedy

**ε-greedy** ist üblich in value-based Methoden wie DQN:

- mit Wahrscheinlichkeit `ε` eine zufällige Aktion wählen;
- mit Wahrscheinlichkeit `1 - ε` die bestbekannte Aktion wählen.

Frühes Training nutzt großes `ε`, damit der Agent frei erkundet. Späteres Training
reduziert `ε`, damit der Agent öfter ausnutzt. Auf einer Punktroboter-Trajektorie
würde das anfangs wie viele zufällige Richtungen aussehen und später weniger zufällige Drehungen.

### Entropie

PPO nutzt eine stochastische Policy statt ε-greedy. Die Policy gibt eine
Verteilung über Aktionen aus, und **Entropie** misst, wie breit diese Verteilung ist.

- hohe Entropie: viele Aktionen sind noch plausibel;
- niedrige Entropie: die Policy ist sicher.

Ein Entropie-Bonus verhindert, dass die Racer-Policy zu früh zu sicher wird, was sie davor bewahren kann, in eine schlechte Lenkgewohnheit zu verfallen.

### Curiosity

Neugiergetriebene Methoden fügen interne Belohnung für neuartige Zustände hinzu. Das kann helfen,
wenn die echte Belohnung spärlich ist. In einem Punktroboter-Labyrinth könnte Neugier belohnen,
neue Gänge zu besuchen, noch bevor das Ziel erreicht ist.

Neugier ist nützlich zu kennen, aber dieser Kurs konzentriert sich auf Reward-Design und PPO,
bevor intrinsische Belohnungen hinzukommen.

---

## 5 · Value-, Policy- und Actor-Critic-Methoden

RL-Algorithmen unterscheiden sich danach, was sie lernen. Nutze deinen Grundlagen-3-Lauf als
Anker: Die Policy wählt die Roboter- oder Racer-Aktion, während eine Wertschätzung helfen kann,
einzuschätzen, ob die aktuelle Beobachtung vielversprechend ist.

| Familie | Was sie lernt | Beispiele | Wie du sie erkennst |
|--------|----------------|----------|---------------------|
| Value-based | `Q(s, a)` oder `V(s)` | Q-Learning, DQN | Aktionen aus gelernten Werten wählen |
| Policy-based | `π(a|s)` direkt | REINFORCE, TRPO | die Aktionsverteilung selbst lernen |
| Actor-Critic | Policy und Wert zusammen | A2C, PPO, SAC | Actor handelt, Critic bewertet |
| Model-based | Übergangs- oder Weltmodell | Dyna, World Models, MuZero | vorhersagen, was als Nächstes passiert |

**Actor-Critic** ist die Familie, die du in diesem Kurs am häufigsten siehst. Der
Actor ist die Policy, die den Racer antreibt. Der Critic schätzt, wie gut die
aktuelle Beobachtung ist, und hilft so, das Rauschen der Policy-Updates zu reduzieren.

!!! info "Warum Actor-Critic?"
    Reine Policy-Gradient-Methoden können hohe Varianz haben, weil jede Trajektorie
    noisy ist. Ein Critic gibt der Policy eine bessere Baseline, was besonders
    hilfreich ist, wenn ein Racer viele mittelmäßige Starts hat, bevor er eine Runde schafft.

---

## 6 · On-Policy versus Off-Policy

Eine **On-Policy**-Methode lernt aus Daten, die von der aktuellen Policy gesammelt wurden. PPO ist
On-Policy: Der Racer-Rollout ist nützlich, weil er von der Policy stammt, die jetzt aktualisiert wird.

Eine **Off-Policy**-Methode kann aus Daten lernen, die von älteren oder anderen
Policies gesammelt wurden. DQN ist Off-Policy: Es kann alte Übergänge in einem Replay-Buffer speichern und
später daraus lernen.

| Frage | On-Policy | Off-Policy |
|----------|-----------|------------|
| Kann es alte Erfahrung stark wiederverwenden? | begrenzt | ja |
| Liegen die Daten nahe am aktuellen Verhalten? | ja | nicht immer |
| Häufiges Beispiel | PPO | DQN |

Beim Punktroboter bedeutet On-Policy-Lernen, dass ein Batch wandernder Trajektorien
nach einem Update verworfen werden kann. Off-Policy-Lernen könnte diese alten
Übergänge behalten und erneut besuchen.

Der Kompromiss ist praktisch: On-Policy-Methoden sind oft stabil und einfach
nachvollziehbar, während Off-Policy-Methoden sample-effizienter sein können.

---

## 7 · Model-free versus model-based

Eine **model-free**-Methode lernt, was zu tun ist, ohne einen separaten Simulator
der Welt zu lernen. PPO und DQN sind model-free. Sie versuchen nicht, die nächste
Racer-Beobachtung vor dem Handeln vorherzusagen; sie lernen Aktionen oder Werte aus Erfahrung.

Eine **model-based**-Methode lernt oder nutzt ein Modell der Umgebungsdynamik:

```text
current observation + action → predicted next observation and reward
```

In einer Punktroboter-Aufgabe könnte ein model-based Agent lernen, dass Rechtsdrehen nahe einer
Wand eine Kollision vorhersagt. Er kann dann mit dieser Vorhersage planen, bevor er die
Aktion in der echten Umgebung ausführt.

Model-based Lernen kann sample-effizient sein, fügt aber einen neuen Fehlermodus hinzu:
Das gelernte Modell kann falsch sein. Wenn das Modell vorhersagt, der Racer schafft eine Kurve, aber die echte Physik widerspricht, kann Planung den Fehler verstärken.

---

## 8 · Algorithmus-Karte für den Rest des Kurses

| Algorithmus | Familie | Policy-/Datenstil | Wo er anknüpft |
|-----------|--------|-------------------|-------------------|
| REINFORCE | policy-based | On-Policy Monte Carlo | Grundlagen 3 Research-Pfad |
| PPO | Actor-Critic | On-Policy, clipped updates | Godot-RL-Trainingspfad |
| Q-Learning | value-based | Off-Policy TD | Q-Learning-Theorie-Unit |
| DQN | deep value-based | Off-Policy TD mit Replay | Deep-Q-Learning-Unit |
| SAC | Actor-Critic | Off-Policy, entropy-regularized | fortgeschrittener Continuous-Control-Vergleich |
| MuZero | model-based | gelerntes Modell plus Planung | fortgeschrittener Hintergrund |

Kehre zu deiner Grundlagen-3-Kurve zurück, wenn du diese Tabelle liest. Die Kurve ist kein
„PPO-Magie"; sie ist das sichtbare Ergebnis von Policy-Updates, Wertschätzungen,
Exploration und Reward-Design über viele Episoden.

---

## 9 · Taxonomie-Selbsttest

Beantworte aus dem Gedächtnis und klappe dann den Abschnitt unten auf:

1. Welche Methode wartet auf das Episodenende: Monte Carlo oder Temporal Difference?
2. Was vergleicht der TD-Fehler?
3. Warum nutzt PPO Entropie statt ε-greedy?
4. Was macht PPO zu Actor-Critic?
5. Warum können Off-Policy-Methoden mehr alte Daten wiederverwenden?
6. Was lernt eine model-based Methode, was PPO nicht lernt?

??? success "Antworten"
    1. **Monte Carlo** wartet auf das Episodenende und nutzt den tatsächlichen Return.
    2. TD-Fehler vergleicht die aktuelle Wertschätzung mit `Belohnung + diskontierter Wert des nächsten Zustands`.
    3. PPO sampelt bereits aus einer stochastischen Policy, daher hält Entropie diese
       Verteilung breit genug zum Erkunden.
    4. PPO lernt sowohl eine Actor-Policy als auch eine Critic-Wertschätzung.
    5. Off-Policy-Methoden können aus Daten lernen, die von älteren oder anderen
       Policies erzeugt wurden, oft über Replay-Buffer.
    6. Eine model-based Methode lernt oder nutzt vorhergesagte Dynamik: nächste Beobachtungen
       und Belohnungen.

---

## 10 · Stretch Goals

- Wähle eine Grundlagen-3-Trajektorie und label jeden Schritt mit Belohnung, Return und
  ob die Aktion explorativ wirkte.
- Plotte zwei Racer-Lernkurven mit unterschiedlichen Entropie-Einstellungen und vergleiche
  frühe Exploration.
- Implementiere diskontierte Returns für eine kurze Belohnungsliste und vergleiche die Werte für
  `γ = 0.5`, `0.9` und `0.99`.
- Baue eine kleine Tabelle, die REINFORCE, PPO, DQN, SAC und MuZero nach
  Familie, On-/Off-Policy-Status und model-free/model-based-Status klassifiziert.

---

## Was kommt als Nächstes

Du hast jetzt die Algorithmus-Karte hinter den sichtbaren Läufen. Kehre als Nächstes zur Haupt-
Kurssequenz zurück und nutze diese Karte, wenn Reward-Design, Q-Learning, DQN, PPO-
Konfiguration und native Inference sich überschneiden.

[← Neuronale Grundlagen 3](unit-neural-03.md) · [Kursstartseite](index.md) · [→ Reward Engineering](unit-reward-engineering.md)
