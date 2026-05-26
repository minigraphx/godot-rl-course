# Unit 0 — Setup & Erster Start

Installiere den Godot-.NET-Editor, die Python-Toolchain und das godot-rl-agents-Plugin. Starte deine erste Trainingssitzung mit dem **BallChase**-Beispiel und bestätige, dass der Godot ↔ Python Socket funktioniert.

---

!!! success "Erster Erfolg (eine Sitzung)"
    1. **Godot** — Akteur (agent) bewegt sich (Hub-Binary oder `--viz`)
    2. **Python** — Terminal zeigt Steps; `ep_rew_mean` steigt
    3. **TensorBoard** (optional) — `tensorboard --logdir=logs` zeigt eine Kurve
    4. **Unit 1** — du wirst eine Belohnungsfunktion (reward function) anpassen und eine Verhaltensänderung beobachten

!!! info "Drei Wege, deine KI zu beobachten (jede Einheit)"
    Godot-Verhalten · TensorBoard-Kurven · was du in `AIController` änderst

!!! tip "Wenig Zeit?"
    Folge dem [Erster-Abend-Skript](#erster-abend-skript) (~2½–3 Std.), um Unit 0 abzuschließen und Unit 1 in einer Sitzung zu beginnen.

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

**Option A — Hub-Binary (schnellster Smoke-Test)**

```bash
python -c "from godot_rl.env_from_hub import env_from_hub; env_from_hub('edbeeching/godot_rl_BallChase')"
chmod +x examples/godot_rl_BallChase/bin/BallChase.x86_64
```

Unter macOS verwende den heruntergeladenen Binary-Pfad in den Trainingsbefehlen unten.

**Option B — Beispiel-Quellcode öffnen (empfohlen zum Lernen)**

1. Klone [godot_rl_agents_examples](https://github.com/edbeeching/godot_rl_agents_examples)
2. Godot → Import → `examples/BallChase` → öffne `project.godot`
3. Projekt → Projekteinstellungen → Plugins → aktiviere **Godot RL Agents**
4. Warte, bis MSBuild fertig ist

---

## 4 · Erster Trainingslauf

**A — Headless Hub-Binary (console-first)**

```bash
python examples/stable_baselines3_example.py \
  --env_path=examples/godot_rl_BallChase/bin/BallChase.x86_64 \
  --experiment_name=BallChase_smoke \
  --timesteps=50000 \
  --speedup=8
```

Lasse `--viz` weg für Headless-Training. Beobachte im Terminal den ansteigenden Episoden-Rückgabewert (episode reward).

**B — Im Editor (macOS / Debugging)**

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
    Akteur in Godot sichtbar (oder im Hub-Fenster); Episoden-Belohnung steigt; keine Socket-Fehler; TensorBoard-Kurve optional, aber empfohlen.

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

Eine Sitzung: Tooling funktioniert, Akteur lernt, du änderst eine Belohnung. Die Zeiten sind Richtwerte — Installationsschritte variieren je nach Rechner. Nutze **Option A (Hub-Binary)** in Abschnitt 3, sofern du Godot .NET nicht bereits geöffnet hast.

| Block | Zeit | Aufgabe | Fertig wenn |
|-------|------|---------|-----------|
| **1 · Installieren** | 45–75 Min | [Abschnitt 2](#2-conda-umgebung) — Miniconda installieren → `godot_env` erstellen → `pip install`<br>[Abschnitt 3](#3-godot-projekt-plugin) — Hub lädt BallChase herunter | `import godot_rl` gibt ok aus; Binary-Pfad vorhanden |
| **2 · Erster Trainingslauf** | 30–45 Min | [Abschnitt 4B](#4-erster-trainingslauf) — `gdrl --viz` + Godot F6<br>Zweites Terminal: `tensorboard --logdir=logs` | Akteur bewegt sich; `ep_rew_mean` steigt; keine Socket-Fehler |
| **3 · Unit 1 beginnen** | 45–60 Min | Öffne [Unit 1](unit-01.md)<br>MDP-Schleife überfliegen (~15 Min) → BallChase-Quellcode öffnen → eine Belohnung anpassen → mit `--viz` neu trainieren<br>Während des Trainings: Unit 1 Abschnitte 3–5 lesen | Du kannst benennen, welche Zeile du geändert hast; Verhalten oder Kurve hat sich verändert |

**Minimales Befehls-Spickzettel (Block 2)**

```bash
conda activate godot_env
tensorboard --logdir=logs &

gdrl --experiment_name=evening_ballchase --viz \
  --save_model_path=ballchase_brain \
  --onnx_export_path=ballchase_brain.onnx

# Godot: BallChase-Trainingsszene → F6 (Szene abspielen)
```

Headless-Alternative (kein Godot-Fenster): Abschnitt 4A mit `--timesteps=50000 --speedup=8`.

**Checkliste zum Abend**

- [ ] Godot — Akteur in Aktion gesehen
- [ ] Python — Training ohne Verbindungsfehler durchgelaufen
- [ ] TensorBoard — mindestens einmal geöffnet (localhost:6006)
- [ ] Code — eine Belohnungszeile in BallChase bearbeitet

Morgen: Unit 1 Lektüre + Glossar abschließen, dann Unit 2 Phase A (SimpleReachGoal).

!!! warning "Nicht weitergekommen?"
    Die häufigsten Hindernisse am ersten Abend: falsche Godot-Version (du benötigst **.NET**), Plugin nicht aktiviert, Python vor Godot F6 gestartet, oder Firewall blockiert den localhost-Socket. Lies Abschnitt 1 erneut, wenn die geteilte Architektur unklar ist.

---

## Wie geht es weiter?

Das Tooling funktioniert. In **Unit 1** wirst du die MDP-Schleife überfliegen, eine BallChase-Belohnung anpassen und die Theorie vertiefen, während das Training läuft.

[→ Unit 1: RL-Grundlagen](unit-01.md)
