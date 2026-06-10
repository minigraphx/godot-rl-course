# Unit 0 — Setup & Erster Start

[Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~20 min · Training: ~15 min GPU / ~1 Std CPU

Installiere den Godot-.NET-Editor, die Python-Toolchain und das godot-rl-agents-Plugin. Starte deine erste Trainingssitzung mit dem **BallChase**-Beispiel und bestätige, dass der Godot ↔ Python Socket funktioniert.

---

!!! success "Erster Erfolg (eine Sitzung)"
    1. **Godot** — Akteur (agent) bewegt sich (`gdrl --viz` + Szene abspielen)
    2. **Python** — Terminal zeigt Steps; `ep_rew_mean` steigt
    3. **TensorBoard** (optional) — `tensorboard --logdir=logs` zeigt eine Kurve
    4. **Neuronale Grundlagen** — du wirst sichtbare Neuronen bauen, bevor die RL-Schleife beginnt

!!! info "Drei Wege, deine KI zu beobachten (jede Einheit)"
    Godot-Verhalten · TensorBoard-Kurven · was du in `AIController` änderst

!!! tip "Wenig Zeit?"
    Folge dem [Erster-Abend-Skript](#erster-abend-skript) (~2½–3 Std.), um Unit 0 abzuschließen und Neuronale Grundlagen 1 in einer Sitzung zu beginnen.

---

## 1 · Geteilte Architektur (split architecture)

Zwei Laufzeitumgebungen kommunizieren über einen lokalen Socket — Godot sendet Beobachtungen (observations) und empfängt Aktionen (actions); Python führt die Trainingsschleife aus.

| Komponente | Rolle | Laufzeit |
|-----------|------|---------|
| **Godot** | Physik, Beobachtungen, Belohnungen | Godot 4 .NET + GDScript |
| **Plugin (C#)** | Sync-Node, ONNX-Bridge | .NET / MSBuild |
| **Python** | PPO / DQN Training (SB3) | Conda, Python 3.10 |

!!! warning "Godot-.NET-Edition erforderlich"
    Die Standard-Godot-Version kann die C#- / NuGet-Abhängigkeiten des Plugins nicht laden. Lade die **.NET**-Version von [godotengine.org](https://godotengine.org) herunter und installiere das [.NET SDK](https://dotnet.microsoft.com/download).

---

## 2 · Conda-Umgebung

!!! info "Zum ersten Mal hier?"
    Vollständige Installationsanweisungen — Miniconda, `godot_env` und das Godot-Plugin — findest du unter [Setup](setup.md). Schließe diese Seite zuerst ab und komme dann hierher zurück.

Kurze Erinnerung für jedes neue Terminal:

```bash
conda activate godot_env
```

---

## 3 · Godot-Projekt & Plugin

Die Trainingsumgebungen stammen aus dem separaten Repo **godot_rl_agents_examples** — nicht aus dem `examples/`-Ordner des Kurs-Repos. Klone es neben das Kurs-Repo:

```bash
cd ..   # aus godot-rl-course heraus
git clone https://github.com/edbeeching/godot_rl_agents_examples.git
```

Kein git? Nutze **Code → Download ZIP** auf [github.com/edbeeching/godot_rl_agents_examples](https://github.com/edbeeching/godot_rl_agents_examples) und entpacke das Archiv neben dem Kurs-Repo.

Danach:

1. Godot → Import → navigiere zu `godot_rl_agents_examples/examples/BallChase/project.godot`
2. Projekt → Projekteinstellungen → Plugins → aktiviere **Godot RL Agents**
3. Warte, bis MSBuild fertig ist

---

## 4 · Erster Trainingslauf

**Im Editor**

Terminal 1 — Python-Listener starten:

```bash
gdrl --experiment_name=BallChase_Mac --viz \
  --save_model_path=ballchase_brain \
  --onnx_export_path=ballchase_brain.onnx
```

Warte auf *"Waiting for connection from Godot…"*

Terminal 2 optional: `tensorboard --logdir=logs`

Godot — öffne die Trainingsszene, drücke **F6** (Szene abspielen). Der Akteur sollte sich verbinden und lernen.

!!! success "Erfolgskriterien"
    Akteur in Godot sichtbar; Episoden-Belohnung steigt; keine Socket-Fehler; TensorBoard-Kurve optional, aber empfohlen.

---

## 5 · Trainingsmodi für diesen Kurs

| Phase | Einheiten | Standard |
|-------|-------|---------|
| Erkunden | 0–2 | Editor oder `--viz` — Physik- und Belohnungsfehler sichtbar machen |
| Skalieren | 3–8 | Exportiertes Binary + `--headless` — schnellere Rollouts |
| Veröffentlichen | 9–10 | ONNX im Sync-Node — kein Python zur Laufzeit |

**ONNX-Vorschau (optional)**

Kopiere nach dem Training `ballchase_brain.onnx` in das Godot-Projekt. Am Sync-Node: Control Mode → **ONNX Inference**, ONNX Model Path setzen. Szene abspielen — der Akteur läuft ohne Python.

---

## Erster-Abend-Skript (~2½–3 Stunden) { #erster-abend-skript }

Eine Sitzung: Tooling funktioniert, Akteur lernt, du änderst eine Belohnung. Die Zeiten sind Richtwerte — Installationsschritte variieren je nach Rechner.

| Block | Zeit | Aufgabe | Fertig wenn |
|-------|------|---------|-----------|
| **1 · Installieren** | 45–75 Min | [Abschnitt 2](#2-conda-umgebung) — Miniconda installieren → `godot_env` erstellen → `pip install`<br>[Abschnitt 3](#3-godot-projekt-plugin) — Examples-Repo klonen, BallChase in Godot öffnen | `import godot_rl` gibt ok aus; BallChase-Projekt öffnet sich mit aktiviertem Plugin |
| **2 · Erster Trainingslauf** | 30–45 Min | [Abschnitt 4](#4-erster-trainingslauf) — `gdrl --viz` + Godot F6<br>Zweites Terminal: `tensorboard --logdir=logs` | Akteur bewegt sich; `ep_rew_mean` steigt; keine Socket-Fehler |
| **3 · Grundlagen 1 beginnen** | 45–60 Min | Öffne [Neuronale Grundlagen 1](unit-neural-01.md)<br>Handrechnung vorhersagen (~15 Min) → Research-Plot oder Godot-Enemy-Szene starten<br>Während des Erkundens: Abschnitte 2–3 lesen | Du kannst jeden Beitrag, die gewichtete Summe und den Aktivierungswert benennen |

**Minimales Befehls-Spickzettel (Block 2)**

```bash
conda activate godot_env
tensorboard --logdir=logs &

gdrl --experiment_name=evening_ballchase --viz \
  --save_model_path=ballchase_brain \
  --onnx_export_path=ballchase_brain.onnx

# Godot: BallChase-Trainingsszene → F6 (Szene abspielen)
```

**Checkliste zum Abend**

- [ ] Godot — Akteur in Aktion gesehen
- [ ] Python — Training ohne Verbindungsfehler durchgelaufen
- [ ] TensorBoard — mindestens einmal geöffnet (localhost:6006)
- [ ] Code — einen Neuron-Forward-Pass-Test oder ein visuelles Beispiel ausgeführt

Morgen: Grundlagen 1–2 abschließen, dann [RL Essentials](unit-01.md) (BallChase-Belohnung anpassen) und Unit 2 Phase A (SimpleReachGoal).

!!! warning "Nicht weitergekommen?"
    Die häufigsten Hindernisse am ersten Abend: falsche Godot-Version (du benötigst **.NET**), Plugin nicht aktiviert, Python vor Godot F6 gestartet, oder Firewall blockiert den localhost-Socket. Lies Abschnitt 1 erneut, wenn die geteilte Architektur unklar ist.

---

## Stretch Goals

**Das exportierte ONNX inspizieren.** Öffne `ballchase_brain.onnx` in [Netron](https://netron.app/) (ziehe die Datei in den Browser-Tab — keine Installation). Identifiziere die Form des Eingabetensors, die Form des Ausgabetensors und die Aktivierung zwischen den Schichten. Das ist die Datei, die Godot zur Inferenzzeit in Phase 3 lädt — zu wissen, was darin steckt, zahlt sich in Unit 9 aus.

**Ein zweites Beispiel aus dem Repo ausprobieren.** Öffne `examples/JumperHard` in Godot, führe eine kurze `gdrl --viz`-Sitzung aus und vergleiche die Belohnungskurven in TensorBoard. Das Ziel ist nicht, es gut zu trainieren — es geht darum, zu bestätigen, dass deine Installation mehr als eine Umgebung verarbeitet.

## Wie geht es weiter?

Das Tooling funktioniert. In **Neuronale Grundlagen 1** baust du ein sichtbares Neuron und verbindest Netze später mit der RL-Schleife.

[→ Neuronale Grundlagen 1](unit-neural-01.md)
