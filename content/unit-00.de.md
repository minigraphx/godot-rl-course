# Unit 0 — Setup & Erster Start

[Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~20 min · Training: ~15 min GPU / ~1 Std CPU

Installiere den Godot-Editor (Standard-Version) und die Python-Toolchain. Starte deine erste Trainingssitzung mit dem mitgelieferten **Foundations-Racer** und bestätige, dass der Godot ↔ Python Socket funktioniert.

---

!!! success "Erster Erfolg (eine Sitzung)"
    1. **Godot** — der Racer bewegt sich auf seiner Strecke (Szene abspielen, während Python trainiert)
    2. **Python** — Terminal zeigt Rollout-Tabellen; `ep_rew_mean` erscheint
    3. **TensorBoard** (optional) — `tensorboard --logdir=logs/foundations_racer` zeigt eine Kurve
    4. **Neuronale Grundlagen** — du wirst sichtbare Neuronen bauen, bevor die RL-Schleife beginnt

!!! info "Drei Wege, deine KI zu beobachten (jede Einheit)"
    Godot-Verhalten · TensorBoard-Kurven · was du in `AIController` änderst

!!! tip "Wenig Zeit?"
    Folge dem [Erster-Abend-Skript](#erster-abend-skript) (~2½–3 Std.), um Unit 0 abzuschließen und Neuronale Grundlagen 1 in einer Sitzung zu beginnen.

---

## 1 · Geteilte Architektur (split architecture)

Zwei Laufzeitumgebungen kommunizieren über einen lokalen Socket — Godot sendet Beobachtungen (observations) und empfängt Aktionen (actions); Python führt die Trainingsschleife aus.

<div class="diagram-scroll">

<svg class="course-diagram" viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="Geteilte Architektur: der Godot-Spielprozess (Szene plus NcnnSync-Node) tauscht Beobachtungen, Belohnungen und Aktionen über einen lokalen Socket mit dem Python-Trainingsprozess aus; nach dem Training führt der NcnnSync-Node das konvertierte Gehirn stattdessen nativ aus">
  <defs>
    <marker id="arS" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 1 L10 5 L0 9 z" fill="#8892b0"/>
    </marker>
  </defs>
  <rect x="20" y="30" width="250" height="200" rx="14" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="145" y="58" text-anchor="middle" fill="#e2e8f0" font-size="16" font-weight="700">Godot</text>
  <text x="145" y="78" text-anchor="middle" fill="#8892b0" font-size="13">Spielprozess</text>
  <rect x="40" y="95" width="210" height="50" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="145" y="125" text-anchor="middle" fill="#e2e8f0" font-size="14">Szene · Physik · Belohnungen</text>
  <rect x="40" y="160" width="210" height="50" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="145" y="190" text-anchor="middle" fill="#e2e8f0" font-size="14">NcnnSync-Node (GDScript-Addon)</text>
  <rect x="450" y="30" width="250" height="200" rx="14" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="575" y="58" text-anchor="middle" fill="#e2e8f0" font-size="16" font-weight="700">Python</text>
  <text x="575" y="78" text-anchor="middle" fill="#8892b0" font-size="13">Trainingsprozess</text>
  <rect x="470" y="95" width="210" height="50" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="575" y="125" text-anchor="middle" fill="#e2e8f0" font-size="14">godot-rl-Wrapper</text>
  <rect x="470" y="160" width="210" height="50" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="575" y="190" text-anchor="middle" fill="#e2e8f0" font-size="14">SB3 — PPO- / DQN-Training</text>
  <path d="M270 120 L450 120" fill="none" stroke="#8892b0" stroke-width="1.8" marker-end="url(#arS)"/>
  <text x="360" y="106" text-anchor="middle" fill="#4ecca3" font-size="13" font-weight="700">Beobachtung + Belohnung</text>
  <path d="M450 190 L270 190" fill="none" stroke="#8892b0" stroke-width="1.8" marker-end="url(#arS)"/>
  <text x="360" y="208" text-anchor="middle" fill="#6c8ef7" font-size="13" font-weight="700">Aktionen</text>
  <text x="360" y="252" text-anchor="middle" fill="#8892b0" font-size="13">lokaler Socket · Port 11008</text>
  <text x="360" y="284" text-anchor="middle" fill="#8892b0" font-size="13" font-style="italic">nach dem Training: NcnnSync führt das konvertierte Gehirn nativ aus — kein Python zur Laufzeit (siehe §5)</text>
</svg>

</div>

| Komponente | Rolle | Laufzeit |
|-----------|------|---------|
| **Godot** | Physik, Beobachtungen, Belohnungen | Godot 4 Standard, GDScript |
| **Addon (GDScript)** | NcnnSync-Node, native ncnn-Bridge | godot-native-rl (im Kurs-Repo gebündelt) |
| **Python** | PPO-Training (SB3) | Conda, Python 3.10 |

!!! tip "Die Standard-Godot-Version genügt"
    Alles auf der Godot-Seite ist GDScript — kein C#, kein .NET SDK, kein Build-Schritt. Lade die **Standard**-Version (4.5+) von [godotengine.org](https://godotengine.org) herunter; die Details stehen im [Setup](setup.md).

---

## 2 · Conda-Umgebung

!!! info "Zum ersten Mal hier?"
    Vollständige Installationsanweisungen — Miniconda, `godot_env` und das Godot-Addon — findest du unter [Setup](setup.md). Schließe diese Seite zuerst ab und komme dann hierher zurück.

Kurze Erinnerung für jedes neue Terminal:

```bash
conda activate godot_env
```

---

## 3 · Godot-Projekt & Addon

Unit 0 und die Einheiten der Neuronalen Grundlagen trainieren im eigenen Godot-Projekt des Kurs-Repos — das **godot-native-rl**-Addon ist darin bereits gebündelt, es gibt also nichts zu installieren oder zu aktivieren:

1. Godot → Import → navigiere zu `godot-rl-course/examples/neural_foundations/game/project.godot`
2. Lass den ersten Import durchlaufen (kein Build-Schritt — das Addon ist reines GDScript)
3. Überprüfen: Node hinzufügen → suche nach `NcnnSync`. Erscheint er, ist das Addon geladen.

!!! info "Examples-Repo für spätere Einheiten"
    Die Einheiten ab [RL Essentials](unit-01.md) nutzen derzeit noch Umgebungen aus dem separaten Repo **godot_rl_agents_examples** (der Legacy-Stack — die Migration wird in den Issues des Kurs-Repos verfolgt). Klone es neben das Kurs-Repo, wenn du diese Einheiten erreichst:

    ```bash
    cd ..   # aus godot-rl-course heraus
    git clone https://github.com/edbeeching/godot_rl_agents_examples.git
    ```

---

## 4 · Erster Trainingslauf

**Im Editor**

Terminal 1 — Python-Trainer starten (aus dem Stammverzeichnis des Kurs-Repos):

```bash
conda activate godot_env
python scripts/train_foundations_racer.py --timesteps 2048
```

Warte auf *"waiting for remote GODOT connection"* in der Log-Ausgabe.

Terminal 2 optional: `tensorboard --logdir=logs/foundations_racer`

Godot — öffne `unit_03_racer/racer_train.tscn`, drücke **F6** (Szene abspielen). Der Racer sollte sich verbinden und Steps ausführen; SB3 gibt Rollout-Tabellen in Terminal 1 aus. Die standardmäßigen 2048 Timesteps sind ein **Smoke-Test** (wenige Minuten) — er beweist, dass Socket, Trainer und Szene zusammenspielen. Für sichtbares Lernen starte erneut mit `--timesteps 50000`.

Nach dem Lauf speichert der Trainer einen Checkpoint und einen ONNX-Export unter `examples/neural_foundations/game/unit_03_racer/models/`.

!!! success "Erfolgskriterien"
    Racer in Godot sichtbar; Rollout-Tabellen mit `ep_rew_mean` erscheinen; keine Socket-Fehler; TensorBoard-Kurve optional, aber empfohlen.

---

## 5 · Trainingsmodi für diesen Kurs

| Phase | Einheiten | Standard |
|-------|-------|---------|
| Erkunden | 0–2 | Editor-Szene abspielen — Physik- und Belohnungsfehler sichtbar machen |
| Skalieren | 3–8 | Exportiertes Binary + `--headless` — schnellere Rollouts |
| Veröffentlichen | 9–10 | Native ncnn-Inferenz im NcnnSync — kein Python zur Laufzeit |

**Native Inferenz-Vorschau (optional, vorerst nur macOS Apple Silicon)**

Nach dem Training kann das exportierte ONNX zu ncnn konvertiert und nativ ausgeführt werden: Die Evaluationsszene `unit_03_racer/racer_eval.tscn` nutzt `NcnnSync` im Inferenzmodus — der Racer läuft ohne Python. [Neuronale Grundlagen 3](unit-neural-03.md) führt durch die komplette Export → Verifizieren → Deployen Pipeline. Unter Windows/Linux überspringe diese Vorschau — die nativen Inferenz-Binärdateien gibt es derzeit nur für macOS Apple Silicon.

---

## Erster-Abend-Skript (~2½–3 Stunden) { #erster-abend-skript }

Eine Sitzung: Tooling funktioniert, Akteur lernt, du änderst eine Belohnung. Die Zeiten sind Richtwerte — Installationsschritte variieren je nach Rechner.

| Block | Zeit | Aufgabe | Fertig wenn |
|-------|------|---------|-----------|
| **1 · Installieren** | 45–75 Min | [Abschnitt 2](#2-conda-umgebung) — Miniconda installieren → `godot_env` erstellen → `pip install`<br>[Abschnitt 3](#3-godot-projekt-addon) — das Kurs-Spielprojekt in Godot öffnen | `import godot_rl` gibt ok aus; Spielprojekt öffnet sich und `NcnnSync` erscheint unter „Node hinzufügen" |
| **2 · Erster Trainingslauf** | 30–45 Min | [Abschnitt 4](#4-erster-trainingslauf) — Python-Trainer + Godot F6<br>Zweites Terminal: `tensorboard --logdir=logs/foundations_racer` | Racer bewegt sich; Rollout-Tabellen erscheinen; keine Socket-Fehler |
| **3 · Grundlagen 1 beginnen** | 45–60 Min | Öffne [Neuronale Grundlagen 1](unit-neural-01.md)<br>Handrechnung vorhersagen (~15 Min) → Research-Plot oder Godot-Enemy-Szene starten<br>Während des Erkundens: Abschnitte 2–3 lesen | Du kannst jeden Beitrag, die gewichtete Summe und den Aktivierungswert benennen |

**Minimales Befehls-Spickzettel (Block 2)**

```bash
conda activate godot_env
tensorboard --logdir=logs/foundations_racer &

python scripts/train_foundations_racer.py --timesteps 2048

# Godot: unit_03_racer/racer_train.tscn → F6 (Szene abspielen)
```

**Checkliste zum Abend**

- [ ] Godot — Akteur in Aktion gesehen
- [ ] Python — Training ohne Verbindungsfehler durchgelaufen
- [ ] TensorBoard — mindestens einmal geöffnet (localhost:6006)
- [ ] Code — einen Neuron-Forward-Pass-Test oder ein visuelles Beispiel ausgeführt

Morgen: Grundlagen 1–2 abschließen, dann [RL Essentials](unit-01.md) (BallChase-Belohnung anpassen) und Unit 2 Phase A (SimpleReachGoal).

!!! warning "Nicht weitergekommen?"
    Die häufigsten Hindernisse am ersten Abend: Godot älter als **4.5**, Godot F6 gedrückt bevor der Python-Trainer wartete, oder Firewall blockiert den localhost-Socket. Lies Abschnitt 1 erneut, wenn die geteilte Architektur unklar ist.

---

## Stretch Goals

**Das exportierte ONNX inspizieren.** Öffne `examples/neural_foundations/game/unit_03_racer/models/foundations_racer.onnx` in [Netron](https://netron.app/) (ziehe die Datei in den Browser-Tab — keine Installation). Identifiziere die Form des Eingabetensors (`obs`), die Form des Ausgabetensors (`out0`) und die Aktivierung zwischen den Schichten. Dieser Graph ist die portable Form des Gehirns — die Export → Verifizieren → Deployen Pipeline in [Neuronale Grundlagen 3](unit-neural-03.md) baut darauf auf.

**Die deterministischen Mathe-Tests ausführen.** Das Spielprojekt liefert Headless-Tests für die Beobachtungs- und Belohnungsmathematik des Racers mit:

```bash
godot --headless --path examples/neural_foundations/game --script res://test/test_racer_math.gd
```

Es geht nicht um die Tests selbst — sondern darum zu bestätigen, dass dein `godot`-Kommandozeilen-Setup aus dem [Setup](setup.md#godot-cli) funktioniert, worauf jede spätere Einheit aufbaut.

## Wie geht es weiter?

Das Tooling funktioniert. In **Neuronale Grundlagen 1** baust du ein sichtbares Neuron und verbindest Netze später mit der RL-Schleife.

[→ Neuronale Grundlagen 1](unit-neural-01.md)
