# Unit 7 — Multi-Agent

Trainiere mehrere Agenten gleichzeitig in derselben Umgebung — einige kooperieren, einige konkurrieren. Studiere **Racer** (gemischte diskrete + kontinuierliche Aktionen) und **MultiAgentSimple** (geteilte vs unabhängige Policies).

[← Visuelle Beobachtungen](unit-visual-observations.md) · [Kursstartseite](index.md)

!!! note "Voraussetzungen"
    - **[Unit 4](unit-04.md)** — PPO end-to-end und sicher mit Reward Shaping
    - **[Unit 6](unit-06.md)** — kontinuierliche Aktionsräume (Racer mischt diskret + kontinuierlich)
    - **[Visuelle Beobachtungen](unit-visual-observations.md)** — nur nötig, wenn du Racer aus Pixeln trainierst
    - Vertrautheit damit, mehrere `AIController`-Knoten in einer einzigen Godot-Szene anzuhängen

!!! info "Zeit"
    Lesen: ~30 min · Training: ~30 min GPU / ~2 Std CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    Godot (beobachte Agenteninteraktionen — Kooperation / Konkurrenz ist visuell offensichtlich) · TensorBoard (Belohnungskurven pro Agent) · `AIController`-Policy-Sharing

---

## 1 · Multi-Agent-Grundlagen

In Single-Agent-RL kontrolliert eine Policy einen Agenten. Multi-Agent erweitert das auf zwei Arten:

**Kooperativ** — Agenten teilen sich ein Belohnungssignal und arbeiten zusammen (z. B. koordiniertes Schieben eines Blocks). Training verwendet oft eine **geteilte Policy**: alle Agenten führen dasselbe Netz aus, was den Stichprobenbedarf reduziert.

**Kompetitiv** — Agenten haben gegensätzliche Belohnungen (z. B. Rennen). Training verwendet **unabhängige Policies**: jeder Agent lernt seine eigene Strategie, möglicherweise via **Selbstspiel (self-play)**.

**Gemischt** — die meisten realen Umgebungen mischen beides (Teamkollegen in gegnerischen Teams).

| Setup | Anzahl Policies | Belohnung | Beispiel |
|-------|-------------|--------|---------|
| Kooperativ | 1 geteilt | Summe oder Mittel | MultiAgentSimple |
| Kompetitiv | N unabhängig | Pro Agent | Racer |
| Self-Play | 1 (spielt gegen sich selbst) | Sieg/Niederlage | AirHockey |

---

## 1b · MARL: Das Nichtstationaritäts-Problem

!!! warning "Nichtstationarität bricht Single-Agent-Theorie"
    Standard-RL-Theorie nimmt an, dass die Umgebung **stationär** ist — dieselbe Policy erzeugt dieselbe Verteilung von Übergängen über die Zeit. In Multi-Agent-RL wird diese Annahme per Design verletzt.

In Single-Agent-RL gilt die Markov-Annahme: die Umgebungsdynamik $P(s'|s,a)$ ändert sich nicht. Der Agent kann seine Policy sicher gegen ein festes Ziel aktualisieren.

**In MARL sind andere Agenten TEIL der Umgebung — und sie lernen ebenfalls.**

Aus Sicht von Agent A ist Agent B eine nichtstationäre Komponente der Umgebung. Wenn B seine Policy aktualisiert, ändert sich die Übergangsverteilung, die A erfährt, auch wenn die zugrundeliegenden Spielregeln unverändert bleiben. Das bedeutet:

- Die optimale Policy für A ändert sich jedes Mal, wenn B seine Policy aktualisiert
- Von A geschätzte Q-Werte werden veraltet, sobald B einen Gradientenschritt macht
- Grundlegende Konvergenzgarantien für Policy Gradients gelten im Allgemeinen nicht mehr

**Konsequenz für Q-Learning:** Der Standard-Q-Learning-Konvergenzbeweis erfordert eine stationäre MDP. Mit mehreren lernenden Agenten ist die MDP aus Sicht jedes einzelnen Agenten nichtstationär. Q-Werte können oszillieren statt zu konvergieren.

**Konsequenz für Policy Gradients:** Das Policy-Gradient-Theorem nimmt an, dass die Value Function unter einer festen Umgebung berechnet wird. Mit weiteren lernenden Agenten verschiebt sich das Value-Function-Target kontinuierlich.

**In der Praxis:** Nichtstationarität verhindert nicht immer das Lernen. Es funktioniert oft trotzdem gut, besonders mit:

- **Geteilten Policies** — wenn alle Agenten dasselbe Netz ausführen, sind "andere Agenten" und "ich selbst" dieselbe Entität; Updates sind konsistent
- **Self-Play** — gegen sich selbst zu spielen führt eine kontrollierte Form von Nichtstationarität ein, die handhabbar ist
- **Großen Populationen** — bei vielen Agenten ändert sich das aggregierte Verhalten langsam; jeder einzelne Agent sieht annähernde Stationarität
- **Kooperativen Aufgaben mit seltener Interaktion** — Agenten, die sich selten beeinflussen, erfahren in der Praxis wenig Nichtstationarität

Das Verständnis dieses Problems motiviert die unten beschriebenen anspruchsvolleren Trainingsparadigmen.

---

## 2 · Wie godot-rl mehrere Agenten behandelt

Jeder Agent hat seinen eigenen `AIController`-Knoten. Der `Sync`-Knoten entdeckt alle `AIController`-Knoten in der Szene und routet Beobachtungen/Aktionen einzeln zu jedem.

```
TrainingScene (Node2D)
  ├─ Sync
  ├─ Agent_0
  │   ├─ ... physics nodes ...
  │   └─ AIController    ← unique per agent
  ├─ Agent_1
  │   ├─ ... physics nodes ...
  │   └─ AIController
  └─ Agent_2
      ...
```

Alle `AIController`-Knoten müssen dasselbe `get_obs()`- und `get_action_space()`-Interface implementieren. Der Sync-Knoten sammelt Beobachtungen von allen, sendet sie als gebatchte Beobachtung an Python und routet Aktionen zurück.

**Geteilte Policy** — der Standard: Python behandelt alle Agenten als eine einzige vektorisierte Umgebung. Alle Agenten sehen dieselben Netzgewichte.

**Unabhängige Policies** — instanziiere separate `StableBaselinesGodotEnv`-Wrapper, einen pro Agentengruppe, jeden mit eigenem Modell.

---

## 2b · Independent Learners (IL)

Der einfachste Ansatz für MARL: jeder Agent führt seinen eigenen RL-Algorithmus aus und behandelt alle anderen Agenten als Teil der (nichtstationären) Umgebung. Keine Koordination oder Kommunikation zwischen Agenten ist während des Trainings erforderlich.

**Pro:**

- Einfach zu implementieren — Standard-Single-Agent-Algorithmen funktionieren out of the box
- Skaliert auf viele Agenten ohne architektonische Änderungen
- Kein Inter-Agent-Kommunikationskanal erforderlich, weder beim Training noch bei der Inferenz

**Contra:**

- Theoretisch unsauber wegen Nichtstationarität (siehe Abschnitt 1b)
- Kann bei Aufgaben scheitern, die enge Koordination erfordern — Agenten können das Lernen der anderen nicht berücksichtigen
- Konvergenz ist nicht garantiert; Training kann instabil sein

**Wann IL in der Praxis funktioniert:**

- **Kompetitive / Rennszenarien** — Agenten sind Gegner; entstehende Konkurrenz treibt das Lernen auch ohne Koordination an
- **Aufgaben mit großer Population** — bei vielen Agenten dominiert kein einzelner die Nichtstationarität
- **Forschung zu emergentem Verhalten** — IL-Agenten entwickeln oft überraschend komplexe soziale Verhaltensweisen trotz der theoretischen Einschränkungen

In godot-rl ist das Ausführen separater `StableBaselinesGodotEnv`-Instanzen mit unabhängigen `PPO`-Modellen die einfachste Form von IL.

---

## 2c · Centralized Training / Decentralized Execution (CTDE)

!!! tip "CTDE: der praktische Goldstandard für kooperatives MARL"
    Training mit Zugriff auf globale Informationen. Ausführung nur mit lokalen Beobachtungen. Du bekommst das Beste aus beiden Welten.

**Die Kernidee:**

Während des **Trainings** teilen Agenten Beobachtungen und Aktionen mit einem zentralisierten Critic. Der Critic kann den vollen globalen Zustand sehen — Positionen, Geschwindigkeiten und Aktionen aller Agenten — was ihm viel bessere Value-Schätzungen ermöglicht, als irgendein einzelner Agent allein berechnen könnte.

Während der **Ausführung** (Inferenz) handelt jeder Agent nur auf seiner eigenen lokalen Beobachtung. Kein Kommunikationskanal zwischen Agenten ist zur Laufzeit nötig.

**Warum das mächtig ist:**

- Der zentralisierte Critic löst das Nichtstationaritäts-Problem: er konditioniert auf alle Policies der Agenten gleichzeitig, sodass das Value-Target stabil ist
- Der dezentralisierte Actor macht Deployment praktikabel: jeder Agent führt seine eigene Netzkopie unabhängig aus, genau wie ein Single-Agent-System

**Kanonische Algorithmen:**

| Algorithmus | Basis | Anmerkungen |
|-----------|------|-------|
| MADDPG | DDPG | Zentralisierte Q-Function pro Agent; kontinuierliche Aktionen |
| MAPPO | PPO | Geteilter oder pro-Agent-Critic mit globalen Beobachtungen |

**In Godot-Begriffen:**

Während Python-Training batcht der `Sync`-Knoten alle Agentenbeobachtungen zusammen und übergibt sie an die Python-Seite. Eine CTDE-Implementierung würde diese gemeinsame Beobachtung an den Critic füttern, während jedem Actor nur agentenspezifische Ausschnitte gegeben werden.

Bei der Inferenz exportierst du ein ONNX-Modell pro Actor (oder ein geteiltes Modell). Jeder `AIController` führt seine eigene Kopie lokal aus — kein Python-Prozess, keine Inter-Agent-Kommunikation. Das passt perfekt zu CTDEs dezentralisierter Ausführungsphase.

> Für Agenten, die zusätzlich Gedächtnis über Zeitschritte hinweg benötigen, siehe [Unit 8 — Memory & POMDPs](unit-08.md), das RecurrentPPO und LSTM-Policies behandelt — eine häufige Kombination mit CTDE in teilweise beobachtbaren Multi-Agent-Umgebungen.

---

## 3 · Gemischte Aktionen — Racer

**Racer** verwendet einen **gemischten** Aktionsraum: diskrete Gangwahl + kontinuierliche Lenkung und Gas.

```gdscript
func get_action_space() -> Dictionary:
    return {
        "steering":  {"size": 1, "action_type": "continuous"},   # [-1, 1]
        "throttle":  {"size": 1, "action_type": "continuous"},   # [-1, 1]
        "gear":      {"size": 3, "action_type": "discrete"},     # 0=brake 1=neutral 2=drive
    }

func set_action(action) -> void:
    var steer    = action["steering"][0]
    var throttle = action["throttle"][0]
    var gear     = int(action["gear"])
    vehicle.steering = steer * max_steering
    vehicle.engine_force = (gear == 2) ? throttle * max_force : 0.0
    vehicle.brake = (gear == 0) ? abs(throttle) * max_brake : 0.0
```

SB3s `PPO` mit `MultiInputPolicy` behandelt gemischte Räume automatisch — keine Änderungen auf der Python-Seite nötig.

---

## 4 · Öffne die Beispiele

**Racer:**

1. Klone [godot_rl_agents_examples](https://github.com/edbeeching/godot_rl_agents_examples) → `examples/Racer`
2. Öffne in Godot .NET, aktiviere Plugin
3. Lies `ai_controller.gd`: beachte das gemischte `get_action_space()`, die Belohnung pro Runde und wie Agenten um die Strecke platziert werden
4. Zähle AIController-Knoten in der Trainingsszene — das ist deine implizite Batchgröße

**MultiAgentSimple:**

1. `examples/MultiAgentSimple` — zwei Agenten, ein Ball, kooperatives Schieben
2. Beachte: beide `AIController`-Knoten haben identische `get_obs()` und `get_action_space()` → geteilte Policy funktioniert
3. Belohnung: positiv, wenn der Ball das Ziel erreicht, geteilt zwischen beiden Agenten

---

## 5 · Trainiere

**Racer (kompetitiv, unabhängige Agenten):**

```bash
conda activate godot_env

# Export binary first, then:
gdrl --env_path=./Racer.x86_64 \
  --experiment_name=racer_ppo \
  --timesteps=3_000_000 \
  --n_parallel=4 \
  --speedup=20 \
  --n_steps=512 \
  --batch_size=256
```

**MultiAgentSimple (kooperativ, geteilte Policy):**

```bash
gdrl --env_path=./MultiAgentSimple.x86_64 \
  --experiment_name=multiagent_coop \
  --timesteps=1_000_000 \
  --n_parallel=8 \
  --speedup=20
```

!!! check "Fertig, wenn"
    Weder Racer noch MultiAgentSimple hat einen veröffentlichten Benchmark, und Multi-Agent-Belohnungskurven sind verrauschter als die Single-Agent-Kurven, die du kennst — mach eine saubere Kurvenform also nicht zur harten Bedingung. Beurteile den Erfolg am **Viz-Checkpoint** (Abschnitt 8): in MultiAgentSimple bewegen sich *beide* Agenten zum Ball bzw. Ziel, statt dass einer untätig herumsteht, und in Racer fahren die Autos Runden, ohne in eine triviale Strategie wie Stillstehen zu verfallen. In TensorBoard sollte `ep_rew_mean` trotz des Rauschens über den Lauf hinweg klar aufwärts tendieren. Eine kooperative Kurve, die früh ein Plateau erreicht, deutet meist auf Free-Riding hin (Abschnitt 6) — nicht darauf, dass mehr Timesteps nötig wären.

---

## 6 · Reward Shaping in Multi-Agent-Szenarien

Belohnungsdesign ist in Multi-Agent-Szenarien schwieriger als in Single-Agent-Szenarien. Die falsche Belohnungsstruktur kann stillschweigend Agenten erzeugen, die scheinbar trainieren, aber degenerierte Strategien lernen.

### Kooperative Szenarien

**Geteilte Belohnung (einfach zu implementieren, schwer zu optimieren):**
Alle Agenten erhalten dasselbe Team-Belohnungssignal. Einfach einzurichten, erzeugt aber das **Free-Rider-Problem**: ein Agent lernt ein nützliches Verhalten, und die anderen entdecken, dass Nichtstun immer noch positive Belohnung einbringt. Mit der Zeit könnte der aktive Agent auch aufhören beizutragen.

**Geformte individuelle Belohnungen (schwerer zu implementieren, bessere Koordination):**
Gib jedem Agenten eine Belohnung proportional zu seinem persönlichen Beitrag zum Teamergebnis. Eine übliche Heuristik:

```
individual_bonus = team_reward × (agent_contribution / total_contribution)
```

Das erfordert die Messung des Beitrags pro Agent, was umgebungsspezifisch ist (zurückgelegte Distanz, Kontakte mit dem Ball, zugefügter Schaden usw.).

**Fix für Free-Riding:** Füge eine individuelle Aktionsstrafe hinzu — kleine negative Belohnung für Agenten, die untätig bleiben, während die Teambelohnung positiv ist. Das bricht das Gleichgewicht "nichts tun, Belohnung teilen".

### Kompetitive Szenarien

**Nullsummenspiel:** Die Belohnung eines Agenten ist das exakte Negativ der anderen (`r_A = -r_B`). Theoretisch sauber, kann aber übermäßig vorsichtiges Spiel erzeugen, wenn beide Agenten lernen, Verluste zu vermeiden, statt zu versuchen zu gewinnen.

**Unabhängige Belohnungen:** Jeder Agent bekommt sein eigenes Belohnungssignal basierend auf seinen eigenen Leistungsmetriken, ohne den Gegner explizit zu bestrafen. Weniger theoretisch fundiert, aber oft leichter zu tunen.

### Gemischte kooperativ-kompetitive Szenarien

Füge eine gewichtete Kombination aus Team- und individuellen Belohnungstermen hinzu:

```
r_agent = α × team_reward + (1 - α) × individual_reward
```

Tune `α` während des Trainings. Starte mit `α = 1.0` (vollständig geteilt), um eine Baseline für kooperatives Verhalten zu etablieren, dann reduziere `α`, um individuelle Spezialisierung zu fördern.

---

## 7 · Multi-Agent-TensorBoard-Kurven lesen

Mit einer geteilten Policy über N Agenten meldet `ep_rew_mean` den Mittelwert über alle Agenten und alle Episoden. Worauf achten:

- **Kooperativ:** Belohnung sollte gemeinsam steigen — wenn sie früh plateau erreicht, könnte ein Agent "free-riden" (nicht beitragen). Füge eine individuelle Aktionsstrafe hinzu, um das zu durchbrechen.
- **Kompetitiv:** Belohnung eines Agenten steigt, während die eines anderen fällt, ist zu erwarten. Prüfe, dass keiner zu einer trivialen Strategie konvergiert (z. B. stillstehen).

---

## 7b · Self-Play

Self-Play ist eine Trainingstechnik, bei der der Gegner eines Agenten eine Kopie der eigenen Policy des Agenten ist. Sie vermeidet die Notwendigkeit handgefertigter Gegner und skaliert natürlich, wenn der Agent sich verbessert.

**Einfaches Self-Play:** Spiele immer gegen den neuesten Policy-Checkpoint. Der Agent trainiert gegen einen Gegner, der genauso gut ist wie er selbst. Das kann **strategische Oszillation** verursachen: der Agent lernt, sein aktuelles Selbst zu schlagen, aber der Gegner (jetzt aktualisiert) hat dieselbe Gegenstrategie gelernt, und der Zyklus wiederholt sich ohne klaren Fortschritt.

**League-basiertes Self-Play (AlphaStar-Stil):** Pflege einen Pool vergangener Checkpoints. Sample Gegner aus dem Pool gemäß einem Prioritätsplan (neuere Checkpoints häufiger, historische gelegentlich). Das verhindert Oszillation, indem sichergestellt wird, dass der Agent robust gegen eine Vielzahl von Gegnerstrategien bleibt, nicht nur gegen die aktuelle.

**Praktisches Self-Play in Godot:**

1. Dupliziere die Trainingsszene, um zwei Agenten-Slots zu haben
2. Lade den aktuellen Checkpoint als "eingefrorene Gegner"-Policy
3. Trainiere die "Lerner"-Policy gegen den eingefrorenen Gegner
4. Alle N Episoden (oder wenn die Siegrate eine Schwelle überschreitet), kopiere die Gewichte des Lerners in den Gegner-Slot
5. Wiederhole

```bash
# Pseudocode — actual implementation depends on wrapper
gdrl --env_path=./AirHockey.x86_64 \
  --experiment_name=selfplay_v1 \
  --timesteps=5_000_000 \
  --opponent_policy=checkpoints/selfplay_v1_latest.zip \
  --self_play_swap_freq=50000
```

**Verbindung zur Nichtstationarität:** Self-Play führt eine kontrollierte, geplante Form von Nichtstationarität ein. Da sich die Gegner-Policy nur an expliziten Swap-Schritten ändert (nicht bei jedem Gradient-Update), ist die Trainings-MDP zwischen Swaps annähernd stationär. Deshalb funktioniert einfaches Self-Play in der Praxis oft, trotz der theoretischen Bedenken in Abschnitt 1b.

Das verbindet sich direkt mit dem **Stretch Goal** in Abschnitt 8: League-basiertes Self-Play auf Racer implementieren.

---

## 8 · Viz-Checkpoint

Schaue 3–5 Episoden im Godot-Editor an:

- **Kooperativ:** Bewegen sich beide Agenten zum Ziel oder steht einer untätig?
- **Kompetitiv/Rennen:** Weichen Agenten einander aus oder kollidieren sie wiederholt?
- **Gemischte Aktionen:** Ist die Lenkung glatt oder oszilliert sie? Prüfe Normalisierung, falls ruckartig.

---

## 9 · Stretch Goals

- **Self-Play auf Racer** — lade den neuesten Checkpoint als Gegner; trainiere iterativ dagegen (siehe Abschnitt 7b für das vollständige Vorgehen)
- **Füge einen dritten Agenten hinzu** — dupliziere einen `AIController`-Knoten und trainiere neu; beobachte, wie sich die Qualität der geteilten Policy ändert
- **Individuelle Belohnung** — modifiziere MultiAgentSimple so, dass jeder Agent seine eigene Belohnung basierend auf seinem persönlichen Beitrag erhält (siehe Abschnitt 6 für Reward-Shaping-Strategien)
- **League Self-Play** — pflege einen Pool von 5 vergangenen Checkpoints; sample Gegner nach Aktualitätsgewichtung

---

## Was kommt als Nächstes

**Unit 8:** Memory & POMDPs — FPS / RobotFPS, RecurrentPPO, LSTM-Policy-Netze für teilweise beobachtbare Umgebungen. RecurrentPPO ist auch die Standardwahl, wenn Gedächtnis mit CTDE in kooperativen Multi-Agent-Aufgaben kombiniert wird.

!!! info "Selbstcheck, bevor du weitermachst"
    Kannst du diese in eigenen Worten beantworten?

    1. Was ist das **Nichtstationaritäts-Problem** in Multi-Agent-RL, und warum existiert es nicht in Single-Agent-RL?
    2. Wann würdest du zu einer geteilten Policy greifen, wann zu unabhängigen Policies — was ändert sich, wenn die Agenten *unterschiedliche* Beobachtungsformen haben?
    3. Was hält CTDE zentralisiert, und was führt es dezentralisiert aus? Warum hilft diese Trennung bei kooperativen Aufgaben?
    4. Warum machst du im Self-Play einen Snapshot des Gegners, statt beide Kopien live zu trainieren?
    5. Wenn zwei kooperierende Agenten dieselbe Belohnung sehen, aber einer die ganze Arbeit macht, wie würdest du das allein anhand von TensorBoard erkennen?

    Wenn du alle fünf beantworten kannst — bist du bereit.

??? success "Antworten zum Selbstcheck"
    1. In MARL sind die anderen Agenten Teil der Umgebung — und sie lernen mit. Sobald Agent B seine Policy aktualisiert, verschiebt sich die Übergangsverteilung, die Agent A erlebt: die Umgebung ist **nichtstationär**. In Single-Agent-RL ist die Dynamik $P(s'|s,a)$ fest, die Stationaritätsannahme gilt, und die üblichen Konvergenzgarantien greifen.
    2. Zur **geteilten Policy** greifst du bei kooperativen, austauschbaren Agenten — identische `get_obs()` und `get_action_space()` — das senkt den Stichprobenbedarf und umgeht die Nichtstationarität, weil "andere Agenten" und "ich selbst" dasselbe Netz sind. **Unabhängige Policies** brauchst du bei kompetitiven oder unterschiedlich gebauten Agenten. Unterschiedliche Beobachtungsformen schließen das Teilen ganz aus: dann musst du separate `StableBaselinesGodotEnv`-Wrapper instanziieren, jeden mit eigenem Modell.
    3. **CTDE** zentralisiert den Critic beim Training (er sieht den globalen Zustand — Positionen, Geschwindigkeiten und Aktionen aller Agenten) und führt die Actors dezentral aus (jeder handelt nur auf seiner lokalen Beobachtung). Die Trennung hilft, weil der zentrale Critic auf alle Policies gleichzeitig konditioniert und so das Value-Target gegen die Nichtstationarität stabilisiert — während das Deployment so einfach bleibt wie bei einem Single-Agent-System, ohne Kommunikation zur Laufzeit.
    4. Mit einem **eingefrorenen** Gegner ändert sich dessen Policy nur an expliziten Swap-Schritten; zwischen den Swaps ist die Trainings-MDP annähernd stationär. Trainierst du beide Kopien live, verschiebt sich das Ziel mit jedem Gradient-Update — strategische Oszillation statt klarem Fortschritt.
    5. Am Verlauf von `ep_rew_mean`: erreicht die kooperative Kurve früh ein Plateau, unterhalb dessen, was koordiniertes Spiel schaffen sollte, ist das die TensorBoard-Signatur von **Free-Riding** — ein Agent arbeitet, der andere kassiert die geteilte Belohnung fürs Nichtstun. Abhilfe schafft eine individuelle Aktionsstrafe für untätige Agenten.

[→ Unit 8: Memory & POMDPs](unit-08.md)
