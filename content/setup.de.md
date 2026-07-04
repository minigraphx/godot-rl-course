# Setup

Einmalige Installation für alle Einheiten dieses Kurses. Führe dies vor Unit 0 durch.

---

## Kurs-Repository — zuerst klonen { #course-repo }

Alles in diesem Kurs — `requirements-course.txt`, der `examples/`-Code der Neuronalen Grundlagen, Hilfsskripte — liegt im Kurs-Repository. Klone es und arbeite aus seinem Stammverzeichnis:

```bash
git clone https://github.com/minigraphx/godot-rl-course.git
cd godot-rl-course
```

Kein git? Nutze stattdessen **Code → Download ZIP** auf [github.com/minigraphx/godot-rl-course](https://github.com/minigraphx/godot-rl-course), entpacke das Archiv und wechsle mit `cd` in den entpackten Ordner.

!!! info "Alle Kursbefehle im Repo-Stammverzeichnis ausführen"
    Sofern eine Einheit nichts anderes sagt, setzt jeder `pip`-, `python -m examples.…`- und `godot --headless`-Befehl in diesem Kurs voraus, dass dein Terminal sich im Stammverzeichnis des Kurs-Repos befindet — dem Ordner mit `requirements-course.txt`.

---

## Godot 4 — Standard-Version

Lade die **Standard**-Version von Godot 4 von [godotengine.org](https://godotengine.org) herunter. Der native Runner des Kurses benötigt **Godot 4.5 oder neuer**.

!!! info "Kein .NET SDK, kein C# — die Standard-Version genügt"
    Frühere Fassungen dieses Kurses nutzten ein C#-Plugin, das die .NET/Mono-Edition von Godot plus das .NET SDK erforderte. Der Kurs verwendet jetzt **godot-native-rl** — ein reines GDScript-Addon, das mit dem Kurs-Repo mitgeliefert wird — sodass die Standard-Version ausreicht. Falls du die .NET-Edition bereits installiert hast, funktioniert sie ebenfalls; es gibt nur keinen Grund mehr, sie zu installieren.

---

## Godot auf der Kommandozeile { #godot-cli }

Mehrere Einheiten starten Godot aus dem Terminal (`godot --headless …`). Der Editor-Download legt **keinen** `godot`-Befehl auf deinen PATH — richte das einmalig ein:

**macOS** — die Binärdatei liegt im App-Bundle. Füge deinem Shell-Profil (`~/.zshrc`) einen Alias hinzu:

```bash
alias godot="/Applications/Godot.app/Contents/MacOS/Godot"
```

**Windows** — füge den Ordner mit `Godot_*.exe` zu deinem PATH hinzu (Einstellungen → System → Info → Erweiterte Systemeinstellungen → Umgebungsvariablen) oder rufe die Datei über ihren vollständigen Pfad auf. Forward Slashes funktionieren in jeder Shell:

```bash
C:/Tools/Godot/Godot_v4.5-stable_win64.exe --version
```

**Linux** — mache die heruntergeladene Binärdatei ausführbar und verlinke sie auf deinen PATH:

```bash
chmod +x Godot_v4.5-stable_linux.x86_64
sudo ln -s "$PWD/Godot_v4.5-stable_linux.x86_64" /usr/local/bin/godot
```

!!! tip "Überprüfen"
    Öffne ein neues Terminal und führe `godot --version` aus — es sollte eine 4.x-Versionsnummer ausgeben.

---

## Python — Conda-Umgebung (environment)

**Warum Conda?** Es erlaubt dir, Python 3.10 in einem isolierten Ordner zu fixieren, sodass die ML-Pakete nicht mit anderen Projekten kollidieren. godot-rl und Stable-Baselines3 funktionieren am zuverlässigsten mit Python 3.10; neuere Versionen führen häufig zu Problemen mit Paket-Wheels.

**Miniconda installieren** (überspringen, falls `conda` bereits vorhanden):

Lade von [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html) für dein Betriebssystem herunter. Nach der Installation öffne ein neues Terminal und prüfe:

```bash
conda --version
```

**Umgebung erstellen** (einmalig):

```bash
conda create --name godot_env python=3.10 -y
```

**In jedem neuen Terminal — vor jedem Trainingsbefehl aktivieren:**

```bash
conda activate godot_env
pip install -r requirements-course.txt
```

`requirements-course.txt` befindet sich im Stammverzeichnis des Kurs-Repos, das du [oben geklont hast](#course-repo) — führe den Befehl von dort aus. Es fixiert alle Pakete auf bekannte, funktionierende Versionen — siehe die [Kompatibilitätstabelle](#kompatibilitatstabelle) unten.

!!! info "Was installiert wird"
    - `godot-rl` — Python ↔ Godot Socket-Bridge, Stable-Baselines3-Wrapper und die `gdrl`-CLI
    - `stable-baselines3` — PPO, SAC und andere Algorithmen
    - `torch` — PyTorch-Backend für das Training
    - `tensorboard` — Visualisierung der Trainingskurven
    - `matplotlib`, `onnx`, `onnxruntime`, `ncnn`, `opencv-python` — Plots, Modell-Export und Paritätsprüfungen in Neuronalen Grundlagen

Überprüfen: `python -c "import godot_rl; print('ok')"`

!!! note "Native Inferenz-Binärdateien — vorerst nur macOS Apple Silicon"
    Das *Training* mit dem gebündelten **godot-native-rl**-Addon funktioniert auf allen Plattformen der Kompatibilitätstabelle — die Trainings-Bridge ist reines GDScript. Die *native ncnn-Inferenz* (ein trainiertes Gehirn ohne Python in Godot ausführen, genutzt in [Neuronale Grundlagen 3](unit-neural-03.md) und der Ship-Phase) liefert derzeit nur Binärdateien für **macOS Apple Silicon** mit; Windows- und Linux-Runner stehen auf der Roadmap.

!!! tip "macOS / Linux — erster Start"
    Der Installer fordert dich möglicherweise auf, `conda init` auszuführen — folge der Anweisung und öffne dann ein neues Terminal.

!!! tip "Windows — erster Start"
    Siehe [Windows — erster Start](#windows-first-run) unten für Hinweise zur PowerShell / cmd / Git Bash Aktivierung und Windows-spezifische Besonderheiten.

---

## Kompatibilitätstabelle

Die folgende Tabelle zeigt die Paketversionen aus `requirements-course.txt` und die Godot-Version, mit der sie getestet wurden.

| Kurs-Tag | Godot | godot-native-rl | godot-rl (Python-Bridge) | stable-baselines3 | PyTorch | Python |
|---|---|---|---|---|---|---|
| 2026-05 | 4.5.x (Standard) | Commit `4013370` (gebündelt) | 0.5.0 | 2.3.2 | 2.6.0 | 3.10 |

!!! warning "Pakete während des Kurses nicht aktualisieren"
    godot-rl, SB3 und PyTorch haben inkompatible API-Änderungen zwischen Releases. Bleibe für die Dauer des Kurses bei den fixierten Versionen in `requirements-course.txt`. Nach dem Kurs kannst du gerne neuere Versionen ausprobieren — erstelle dafür einfach eine neue Conda-Umgebung.

---

## Windows — erster Start { #windows-first-run }

Die obigen Schritte funktionieren auf Windows mit kleinen Unterschieden. Lies diesen Abschnitt, bevor du das erste Mal `conda activate` ausführst.

### Shell-Wahl

| Shell | Hinweise |
|-------|----------|
| **Anaconda Prompt** | Am einfachsten — `conda activate godot_env` funktioniert sofort. |
| **PowerShell** | Führe einmalig `conda init powershell` aus (als Administrator), starte PowerShell neu, dann `conda activate godot_env`. |
| **cmd** | Führe einmalig `conda init cmd.exe` aus, starte cmd neu, dann `conda activate godot_env`. |
| **Git Bash** | Führe einmalig `conda init bash` (aus dem Anaconda Prompt) aus, starte Git Bash neu, dann `conda activate godot_env`. |

### `--env_path` mit Windows-Pfaden

godot-rl akzeptiert Schrägstriche (Forward Slashes) unter Windows — bevorzuge diese gegenüber Backslashes, um Shell-Escape-Probleme zu vermeiden:

```bash
# Empfohlen — Forward Slashes funktionieren überall, auch in PowerShell und cmd
gdrl --env_path=C:/Users/DeinName/Projekte/mein_spiel/mein_spiel.exe

# Auch gültig — Backslashes, müssen aber escaped oder in Anführungszeichen gesetzt werden
gdrl --env_path="C:\Users\DeinName\Projekte\mein_spiel\mein_spiel.exe"
```

### Windows Defender / Antivirusproblem mit Sockets

Windows Defender (und viele Antivirusprogramme von Drittanbietern) blockieren manchmal **still und heimlich** den lokalen TCP-Port, den godot-rl für die Kommunikation zwischen Python und Godot verwendet (Standard: Port 11008). Symptome: Das Training scheint zu starten, aber Godot stellt keine Verbindung her; Python wartet auf die erste Beobachtung.

Lösung:

1. Öffne **Windows-Sicherheit → Firewall & Netzwerkschutz → App durch Firewall zulassen**.
2. Füge eine Ausnahme für `python.exe` (das Python deiner Conda-Umgebung) und für die Godot-Anwendung hinzu.
3. Alternativ: Verwende einen anderen Port: `gdrl --port=12000` (und setze denselben Port am Godot-seitigen Sync-Node).

Wenn du ein Antivirusprogramm eines Drittanbieters verwendest, füge den Conda-Umgebungsordner (z. B. `C:\Users\DeinName\miniconda3\envs\godot_env\`) und deinen Godot-Projektordner zur Ausschlussliste hinzu.

### `chmod +x` ist unter Windows nicht erforderlich

Die macOS/Linux-Befehle `chmod +x godot_binary` gelten nicht für Windows. Godot `.exe`-Dateien sind vom Betriebssystem bereits ausführbar.

### WSL2 vs. natives Windows

| Ansatz | Vorteile | Nachteile |
|--------|----------|-----------|
| **Natives Windows** | Einfachste Einrichtung, keine Übersetzungsschicht, Direct3D GPU | Antivirus-/Firewall-Reibung; Pfade verwenden Backslashes |
| **WSL2 (Ubuntu)** | Vollständige Linux-Toolchain, einfachere GPU-Einrichtung via CUDA | GPU-Durchleitung (CUDA in WSL2) erfordert Windows 11 + WSL2-Kernel ≥ 5.15; Godot-GUI kann in WSL2 ohne X-Server oder WSLg nicht rendern |

**Empfehlung für diesen Kurs:** Verwende **natives Windows**, es sei denn, du hast bereits ein funktionierendes WSL2 + GPU-Setup. Godot muss ohnehin auf der Windows-Host-Seite (oder WSLg) laufen; die Kombination von Godot unter Windows und Python in WSL2 erfordert zusätzliche Port-Weiterleitungsschritte, die in diesem Kurs nicht behandelt werden.

---

## Godot-Addon — godot-native-rl

Die Godot-seitige Bridge ist **godot-native-rl**, ein reines GDScript-Addon. Es wird **im Kurs-Repo** unter `examples/neural_foundations/game/addons/godot_native_rl/` mitgeliefert — nichts herunterzuladen, nichts zu bauen.

- **Training:** Der `NcnnSync`-Node spricht dasselbe lokale Socket-Protokoll wie die `gdrl`-Python-Seite. Reines GDScript — funktioniert auf jeder Plattform.
- **Inferenz:** Eine gebündelte ncnn-GDExtension führt trainierte Gehirne nativ in Godot aus. Binärdateien gibt es derzeit nur für macOS Apple Silicon (siehe Hinweis oben); das Training benötigt sie nicht.

Die Node-Klassen des Addons (`NcnnSync`, `NcnnAIController2D`, `NcnnAIController3D`, Sensor-Nodes) registrieren sich automatisch, wenn das Projekt geöffnet wird. Das Aktivieren des Plugins unter Projekt → Projekteinstellungen → Plugins ist **optional** — es ergänzt lediglich eine klare Fehlermeldung, falls die native Inferenz-Binärdatei für deine Plattform fehlt.

!!! warning "Zwei verschiedene Dinge"
    `godot-rl` ist das *Python*-Paket (installiert über `requirements-course.txt`), das den Trainingsserver betreibt. **godot-native-rl** ist das *Godot*-Addon, das mit dem Kurs-Repo mitgeliefert wird. Beide kommunizieren über einen lokalen Socket.

!!! tip "Überprüfen"
    Öffne `examples/neural_foundations/game/project.godot` in Godot, dann Node hinzufügen → suche nach `NcnnSync` und `NcnnAIController3D`. Wenn diese erscheinen, funktioniert das Addon.

**Das Addon im eigenen Projekt verwenden:** Kopiere `addons/godot_native_rl/` in den `addons/`-Ordner deines Projekts. Für native Inferenz kopiere zusätzlich `ncnn_runner.gdextension` und den `bin/`-Ordner.

---

## Legacy-Plugin — godot-rl-agents (Einheiten vor der Migration) { #godot-plugin-godot-rl-agents }

Die Einheiten ab [RL Essentials](unit-01.md) nutzen noch Beispielumgebungen auf Basis des älteren C#-Plugins. **Überspringe diesen Abschnitt, bis du diese Einheiten erreichst** — Unit 0 und die Neuronalen Grundlagen brauchen nichts davon.

Der Legacy-Stack erfordert zusätzlich die .NET/Mono-Edition von Godot und das [.NET SDK](https://dotnet.microsoft.com/download). Danach:

- Klone oder lade [github.com/edbeeching/godot_rl_agents_plugin](https://github.com/edbeeching/godot_rl_agents_plugin) herunter
- Kopiere den Ordner `addons/godot_rl_agents` in den `addons/`-Ordner des Beispielprojekts (die offiziellen Beispielprojekte liefern ihn bereits mit)
- Projekt → Projekteinstellungen → Plugins → **Godot RL Agents** → Aktiviert; warte, bis MSBuild fertig ist
- Falls Godot beim ersten Öffnen einen C#-Build-Fehler meldet, schließe das Projekt und öffne es erneut — die Assemblies werden beim zweiten Öffnen korrekt erstellt
- Überprüfen: Node hinzufügen → suche nach `Sync` und `AIController2D`

Diese Einheiten werden nach und nach auf den nativen Stack migriert; dieser Abschnitt verschwindet, sobald die Migration abgeschlossen ist.
