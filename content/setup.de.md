# Setup

Einmalige Installation für alle Einheiten dieses Kurses. Führe dies vor Unit 0 durch.

---

## Godot 4 — .NET-Edition

Lade die **.NET / Mono**-Version von Godot 4 von [godotengine.org](https://godotengine.org) herunter (nicht die Standard-Version). Getestet mit Godot 4.3+.

!!! warning "Verwende die .NET-Edition — nicht die Standard-Version"
    Das godot-rl-Plugin kompiliert native C#-Tasks (NuGet-Referenzen), die Godot mit der ONNX-Laufzeit verbinden. Die Standard-Version kann diese nicht laden. Du musst außerdem das [.NET SDK](https://dotnet.microsoft.com/download) installieren.

---

## Python — Conda-Umgebung (environment)

**Warum Conda?** Es erlaubt dir, Python 3.10 in einem isolierten Ordner zu fixieren, sodass die ML-Pakete nicht mit anderen Projekten kollidieren. godot-rl und Stable-Baselines3 funktionieren am zuverlässigsten mit Python 3.10; neuere Versionen führen häufig zu Problemen mit Paket-Wheels.

**Miniconda installieren** (überspringen, falls `conda` bereits vorhanden):

Lade von [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html) für dein Betriebssystem herunter. Nach der Installation öffne ein neues Terminal und prüfe:

```bash
conda --version
```

!!! tip "macOS / Linux — erster Start"
    Der Installer fordert dich möglicherweise auf, `conda init` auszuführen — folge der Anweisung und öffne dann ein neues Terminal.

**Umgebung erstellen** (einmalig):

```bash
conda create --name godot_env python=3.10 -y
```

**In jedem neuen Terminal — vor jedem Trainingsbefehl aktivieren:**

```bash
conda activate godot_env
pip install "godot-rl[sb3]" tensorboard
```

Überprüfen: `python -c "import godot_rl; print('ok')"`

!!! info "Was installiert wird"
    - `godot-rl[sb3]` — Python ↔ Godot Socket-Bridge, Stable-Baselines3-Wrapper und die `gdrl`-CLI
    - `tensorboard` — Visualisierung der Trainingskurven
    
    Behalte die doppelten Anführungszeichen um `"godot-rl[sb3]"`, damit die Shell die eckigen Klammern nicht expandiert.

---

## Godot-Plugin — godot-rl-agents

Das Godot-seitige Plugin ist vom Python-Paket getrennt.

**Option A — AssetLib (am einfachsten)**

- Öffne den Godot-Editor → Tab **AssetLib** (oben in der Mitte)
- Suche nach **rl**, wähle **Godot RL Agents**
- Klicke auf **Download**, deaktiviere `LICENSE` und `README.md`, dann **Install**

**Option B — Manuell (immer aktuell)**

- Klone [github.com/edbeeching/godot_rl_agents_plugin](https://github.com/edbeeching/godot_rl_agents_plugin)
- Kopiere `addons/godot_rl_agents` in den `addons/`-Ordner deines Projekts

!!! warning "Zwei verschiedene Repositories"
    `godot_rl_agents` ist das *Python*-Paket (`pip install`). Das Godot-Plugin befindet sich im separaten Repository `godot_rl_agents_plugin`.

**Plugin aktivieren**

Projekt → Projekteinstellungen → Plugins → **Godot RL Agents** → Aktiviert. Warte, bis MSBuild fertig ist.

!!! warning "C#-Fehler beim ersten Import"
    Falls Godot beim ersten Öffnen einen Build-Fehler meldet, schließe das Projekt und öffne es erneut — die C#-Assemblies werden beim zweiten Öffnen korrekt erstellt.

!!! tip "Überprüfen"
    Node hinzufügen → suche nach `Sync` und `AIController2D`. Wenn diese erscheinen, funktioniert das Plugin.
