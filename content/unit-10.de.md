# Unit 10 — Ship Your Brain

Deine Policy ist trainiert. Jetzt geht es darum, sie **aus Python heraus und in Godot hinein** zu bekommen — mit voller Geschwindigkeit, ohne Python-Prozess, als eigenständiges Spiel. Lerne die ONNX-Export-Pipeline, das Laden und Fortsetzen von Checkpoints und optional das Veröffentlichen einer spielbaren HTML5-Demo.

[← Offline RL](unit-offline-rl.md) · [Kursstartseite](index.md)

!!! note "Voraussetzungen"
    - **Irgendein trainierter Agent aus Units 2–9** — du brauchst einen SB3-`.zip`-Checkpoint zum Exportieren
    - **[Unit 2](unit-02.md)** §11 hat den ersten ONNX-Export bereits gezeigt — diese Unit macht ihn produktionsreif
    - Vertrautheit mit Godot-Export-Presets (Desktop und/oder HTML5)
    - Keine PyTorch-Interna nötig; kein ONNX-Hintergrundwissen erforderlich

!!! info "Zeit"
    Lesen: ~20 min · Training: ~10 min GPU / ~30 min CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    Godot (Inferenzmodus — kein laufendes Python, Agent spielt live) · ONNX-Inspector (Netron — visualisiert den exportierten Graphen) · HTML5-Export (teile eine URL, lass jeden gegen deinen Agenten spielen)

---

## 1 · Die Shipping-Pipeline

```
Python training  →  .zip checkpoint  →  ONNX export  →  Godot inference
                        ↕
                   resume training
```

**ONNX (Open Neural Network Exchange)** ist ein Standard-Modellformat. Godot RL Agents bringt eine GDScript-ONNX-Runtime mit — zur Inferenzzeit ist kein Python nötig.

---

## 2 · Checkpoints speichern und fortsetzen

Stelle vor dem Export sicher, dass du ein finales gespeichertes Modell hast. Du kannst Training auch von jedem Checkpoint fortsetzen, falls ein Lauf unterbrochen wurde.

```bash
# Save with checkpoints every 100k steps and export ONNX at the end
gdrl --env_path=./BallChase.x86_64 \
  --experiment_name=ballchase_final \
  --timesteps=1_000_000 \
  --save_model_path=ballchase_final \
  --save_checkpoint_frequency=100000 \
  --onnx_export_path=ballchase_final.onnx \
  --n_parallel=8 \
  --speedup=20
```

```bash
# Resume from a checkpoint if training was interrupted
gdrl --env_path=./BallChase.x86_64 \
  --resume_model_path=ballchase_final.zip \
  --experiment_name=ballchase_resumed \
  --timesteps=500_000 \
  --onnx_export_path=ballchase_final.onnx
```

Das `--resume_model_path`-Flag lädt Gewichte, Optimizer-State und Schrittzähler — das Training läuft genau dort weiter, wo es aufgehört hat.

---

## 3 · ONNX aus einem gespeicherten Modell exportieren

Wenn du bereits ein `.zip`-Modell hast und ONNX separat exportieren willst:

```python
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="./BallChase.x86_64", n_parallel=1, speedup=1)
model = PPO.load("logs/sb3/ballchase_final/best_model", env=env)

# Export to ONNX
model.policy.to("cpu")
import torch

obs = torch.zeros(1, *env.observation_space.shape)
torch.onnx.export(
    model.policy,
    obs,
    "ballchase_final.onnx",
    input_names=["obs"],
    output_names=["action"],
    opset_version=15,
)
print("Exported: ballchase_final.onnx")
env.close()
```

Alternativ den eingebauten gdrl-Export nutzen:

```bash
gdrl --env_path=./BallChase.x86_64 \
  --resume_model_path=ballchase_final.zip \
  --onnx_export_path=ballchase_final.onnx \
  --timesteps=0
```

`--timesteps=0` überspringt das Training und exportiert nur.

---

## 4 · ONNX-Graph inspizieren (optional)

Installiere [Netron](https://netron.app/) — einen browserbasierten ONNX-Viewer — und zieh deine `.onnx`-Datei hinein. Du siehst:

- Input-Node: Beobachtungs-Shape
- MLP-Layer: Gewichtsmatrizen und Aktivierungsfunktionen
- Output-Node: Parameter der Aktionsverteilung

Das hilft beim Debuggen von Shape-Mismatches beim Wechsel zwischen SB3-Versionen.

---

## 5 · ONNX in Godot laden

1. Kopiere `ballchase_final.onnx` in deinen Godot-Projektordner (z. B. `res://models/`)
2. Wähle den `Sync`-Node in deiner Szene
3. Setze `Control Mode` auf `ONNX_INFERENCE`
4. Setze `Onnx Model Path` auf `res://models/ballchase_final.onnx`
5. Starte die Szene — der Agent spielt ohne jeden Python-Prozess

```
Sync node properties:
  Control Mode:      ONNX_INFERENCE
  Onnx Model Path:   res://models/ballchase_final.onnx
  Speed Up:          1          ← real-time, not accelerated
```

Der Agent läuft mit Spielgeschwindigkeit. Du kannst menschlich gesteuerte Charaktere, Hindernisse oder UI rund um den KI-Agenten ergänzen — er ist jetzt einfach ein weiterer Godot-Node.

!!! check "Fertig, wenn"
    Die Szene läuft im `ONNX_INFERENCE`-Modus und der Agent spielt kompetent — **ohne einen einzigen laufenden Python-Prozess**: kein `gdrl` in deiner Prozessliste, keine aktivierte conda-Umgebung, nichts zum Abbrechen mit Strg-C. Am Viz-Checkpoint in Abschnitt 9 zeigt der Agent dieselbe Kompetenz wie am Ende des Trainings — erwarte aber keine Frame-für-Frame-identischen Läufe: Das Training hat Aktionen stochastisch gesampelt, die exportierte Policy handelt deterministisch, etwas glatteres, weniger zittriges Verhalten ist also normal. Ein Agent, der zufällig zittert oder einfriert, deutet auf einen falschen Modellpfad oder einen Shape-Mismatch der Beobachtungen hin (Abschnitt 4), nicht auf eine kaputte Runtime.

---

## 6 · Export für Desktop

Exportiere ein Standard-Desktop-Binary mit eingebackener KI:

1. Projekt → Exportieren → Preset hinzufügen (Linux / Windows / macOS)
2. Ressourcen-Tab: Stelle sicher, dass `*.onnx`-Dateien im Export-Filter enthalten sind
3. Exportieren → Projekt exportieren (nicht PCK)

Das Ergebnis ist eine eigenständige ausführbare Datei — kein Python, keine conda-Umgebung nötig.

---

## 7 · HTML5- / WASM-Export (optional)

Godot kann nach WebAssembly exportieren, sodass das Spiel im Browser läuft. Die ONNX-Runtime funktioniert in WASM.

**Voraussetzungen:**

- Godot 4.x mit installiertem HTML5-Export-Template
- Ein Webserver (GitHub Pages, itch.io, Netlify — alle funktionieren)

**Schritte:**

1. Projekt → Exportieren → Preset hinzufügen → Web
2. `Export Type: Release` aktivieren
3. Ressourcen: `*.onnx` einschließen
4. Exportieren → Projekt exportieren → Ausgabepfad `index.html` wählen
5. Lade den Ausgabeordner auf deinen Webserver hoch

!!! warning "SharedArrayBuffer"
    HTML5-Godot-Exports brauchen `SharedArrayBuffer`, was spezifische HTTP-Header erfordert (`Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Embedder-Policy: require-corp`). GitHub Pages unterstützt das; einfaches File-Hosting nicht. Prüfe die Doku deines Hosters.

**Auf itch.io veröffentlichen:**

1. Lege ein Projekt auf [itch.io](https://itch.io) an
2. Kind: HTML
3. Lade den exportierten Ordner als Zip hoch
4. Hake „This file will be played in the browser" an
5. Teile die URL

Jeder mit einem Browser kann nun deinem trainierten Agenten zusehen — und mit ihm interagieren.

---

## 8 · Agenten zur Laufzeit tauschen (fortgeschritten)

Du kannst zur Laufzeit unterschiedliche ONNX-Modelle per GDScript laden:

```gdscript
# In a scene script — swap agent brain on button press
@onready var sync_node = $Sync

func _on_swap_pressed():
    sync_node.onnx_model_path = "res://models/alternative_brain.onnx"
    sync_node.reload_model()
```

So baust du „Replay-Demos", die zwischen einer Zufalls-Policy, einem BC-Klon und einem fine-getunten PPO-Agenten umschalten — alles in einer Szene, ohne Python-Neustarts.

---

## 9 · Viz-Checkpoint

Lass dein exportiertes Binary oder den HTML5-Build 5 Minuten laufen:

- Ist das Verhalten des Agenten identisch zu dem im Training? (Sollte es sein — ONNX ist deterministisch)
- Bildrate stabil? ONNX-Inferenz fügt für MLP-Policies < 1 ms pro Schritt hinzu
- Für HTML5: teste auf Mobilgerät und Low-End-Laptop — die WASM-Runtime ist schlanker als das Desktop-Godot

---

## 10 · Stretch Goals

- **A/B-Test im Browser** — exportiere zwei Modelle (PPO vs. BC-Fine-Tune aus Unit 9); baue eine Godot-UI, in der Nutzer zwischen ihnen umschalten und abstimmen, welches natürlicher aussieht
- **ONNX-Modell quantisieren** — nutze `onnxruntime`-Tools, um das Modell auf INT8 zu reduzieren; miss Größen- und Geschwindigkeitsunterschied
- **Continuous Deployment** — richte einen GitHub-Actions-Workflow ein: neue Trainingsergebnisse pushen → ONNX automatisch exportieren → HTML5 auf GitHub Pages deployen

---

## Was kommt als Nächstes

Du hast den Kurs abgeschlossen. Hier ist, was du gebaut hast:

| Unit | Skill |
|------|-------|
| 0 | Eine Godot-RL-Umgebung ausführen |
| 1 | Den Agent–Umgebungs-Loop verstehen |
| 2 | Eine eigene Godot-RL-Umgebung von Grund auf bauen |
| 3 | DQN für spärliche diskrete Aufgaben |
| 4 | PPO-Hyperparameter-Tuning |
| 5 | Paralleles Training in Skala |
| 6 | Kontinuierliche Aktionsräume + Normalisierung |
| 7 | Multi-Agent: kooperativ und kompetitiv |
| 8 | Gedächtnis und POMDPs mit RecurrentPPO |
| 9 | Imitation Learning: BC und GAIL |
| 10 | ONNX-Export + Godot-Inferenz + HTML5 |

**Was als Nächstes kommt:** Bring alles im Capstone zusammen — wähle deine eigene Umgebung, gestalte dein eigenes Reward, trainiere es und shippe es.

!!! info "Selbstcheck, bevor du weitermachst"
    Kannst du diese in eigenen Worten beantworten?

    1. Warum führt der Shipping-Pfad über ONNX, statt das SB3-`.zip` direkt in Godot zu laden?
    2. Welche drei Dinge müssen zwischen Training und Inferenz übereinstimmen, damit die Policy nicht stillschweigend Müll-Aktionen produziert?
    3. Was ist der Unterschied zwischen einem *Resume-Training*-Checkpoint und einem *Export-für-Inferenz*-Modell — welchen State enthält jedes?
    4. Wann würdest du HTML5/WASM-Export einem Desktop-Binary vorziehen, und was kostet das den Agenten?
    5. Wie würdest du nach dem Export plausibilisieren, dass die Godot-seitige Inferenz zu den Python-seitigen Trainings-Rollouts passt?

    Wenn du alle fünf beantworten kannst — bist du bereit für das Capstone.

??? success "Antworten zum Selbstcheck"
    1. Das SB3-`.zip` ist ein PyTorch-Artefakt — um es zu laden, braucht es einen laufenden Python-Prozess. **ONNX** ist ein Standard-Austauschformat, und Godot RL Agents bringt eine GDScript-ONNX-Runtime mit, sodass das ausgelieferte Spiel die Policy ganz ohne Python ausführt (Abschnitt 1).
    2. Die **Beobachtungs-Shape**, der Aktionsraum und die Bedeutung/Reihenfolge der Werte, die deine Godot-Umgebung dem Modell liefert. Driftet eines davon zwischen Training und ausgelieferter Szene auseinander, läuft der Graph trotzdem — er macht nur aus Müll-Eingaben Müll-Ausgaben. Netron (Abschnitt 4) fängt den Shape-Teil ab.
    3. Ein *Resume-Training*-Checkpoint (`--resume_model_path`) enthält **Gewichte, Optimizer-State und Schrittzähler**, sodass das Training genau dort weiterläuft, wo es aufgehört hat (Abschnitt 2). Das exportierte ONNX enthält nur den Forward-Pass der Policy — genug zum Handeln, aber nicht zum Weiterlernen.
    4. Wähle **HTML5/WASM**, wenn jeder ohne Installation per URL spielen können soll. Der Preis sind Hosting-Auflagen — der Host muss die `SharedArrayBuffer`-Header (COOP/COEP) setzen, was itch.io als Opt-in anbietet — und weniger Performance-Spielraum, weshalb der Viz-Checkpoint rät, auf Mobilgerät und Low-End-Laptop zu testen.
    5. Führe den **Viz-Checkpoint** aus Abschnitt 9 aus: Lass den exportierten Build einige Minuten laufen. ONNX-Inferenz ist deterministisch — dieselbe Beobachtung erzeugt immer dieselbe Aktion —, das Verhalten ist also über Läufe des Builds hinweg reproduzierbar; erwarte es etwas glatter als die Trainings-Rollouts, die Aktionen stochastisch gesampelt haben. Grobes Fehlverhalten (Zittern, Einfrieren, Umherirren) deutet auf ein Export-Problem hin, nicht auf Zufall.

[→ Capstone-Projekt](unit-capstone.md) · [← Zurück zur Kursstartseite](index.md)
