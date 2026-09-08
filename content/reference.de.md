# Referenz

Schnelles Nachschlagen für die Flags des SB3-Trainingsskripts, die `AIController`-API und den ONNX-Export.

!!! info "Diese Seite dokumentiert den Legacy-Stack `godot-rl-agents`"
    Die Units ab [RL Essentials](unit-01.md) nutzen ihn weiterhin. Unit 0 und die
    Neuronalen Grundlagen verwenden stattdessen **godot-native-rl**, dessen Node-Namen
    abweichen (`NcnnSync`, `NcnnAIController2D` / `3D`) — siehe
    [Setup → Godot-Addon](setup.md#godot-addon-godot-native-rl).

!!! note "Vollständige Referenz"
    Siehe `godot_rl_course_reference.html` für die vollständige Plugin-API-Referenz.

## Das Trainingsskript — nicht `gdrl`

Die Flags unten gehören zu **`stable_baselines3_example.py`**, das im *Repository* von godot-rl
liegt und **nicht** Teil der `pip`-Installation ist. Der installierte Befehl `gdrl` kennt keines
davon: Er trainiert feste 200 000 Schritte und speichert nichts. Upstream hat ihn abgekündigt —
beim Aufruf erscheint *„This use of gdrl is deprecated and will be removed in version 1.0,
please refer to the examples in the github repo."*

Hole das Skript einmalig in den Ordner, aus dem du trainierst:

```bash
curl -O https://raw.githubusercontent.com/edbeeching/godot_rl_agents/main/examples/stable_baselines3_example.py
```

Danach lautet jedes Trainingskommando in diesem Kurs:

```bash
python stable_baselines3_example.py --experiment_name=mein-lauf --viz
```

## Häufige Trainings-Flags

| Flag | Standard | Beschreibung |
|------|----------|--------------|
| `--env_path` | — | Exportiertes Godot-Binary. Weglassen, um gegen den Editor zu trainieren (F6) |
| `--experiment_name` | `experiment` | Name, der in TensorBoard erscheint |
| `--experiment_dir` | `logs/sb3` | Zielordner für die TensorBoard-Logs |
| `--viz` | aus | Godot-Fenster während des Trainings anzeigen |
| `--timesteps` | 1 000 000 | Gesamtanzahl der Umgebungsschritte |
| `--speedup` | 1 | Zeitskalenfaktor der Physik |
| `--n_parallel` | 1 | Anzahl paralleler Godot-Instanzen (benötigt `--env_path`) |
| `--save_model_path` | — | Pfad zum Speichern des trainierten `.zip`-Modells |
| `--onnx_export_path` | — | Pfad zum Export von `.onnx` für Godot-Inferenz |
| `--resume_model_path` | — | Von einem gespeicherten Modell aus weitertrainieren |
| `--inference` | aus | Geladenes Modell ausführen statt trainieren |
| `--n_steps` | 64 | Schritte pro Umgebung und Update |
| `--batch_size` | 64 | Minibatch-Größe |
| `--learning_rate` | 0.0003 | Lernrate |
| `--ent_coef` | 0.0001 | Entropie-Koeffizient |
| `--clip_range` | 0.2 | PPO-Clipping-Bereich |

## AIController-Lebenszyklus

```
_ready()          → bei Sync-Node registrieren
get_obs() → Array → Beobachtungsvektor (observation) zurückgeben
set_action()      → Aktion empfangen und anwenden
get_reward() → float → skalare Belohnung (reward) zurückgeben
get_done() → bool → Episodenende-Flag zurückgeben
```

## ONNX-Inferenz in Godot

1. Trainieren und exportieren: `--onnx_export_path=brain.onnx`
2. `brain.onnx` in den Godot-Projektordner kopieren
3. Sync-Node → **Control Mode** → `Onnx Inference`
4. Sync-Node → **Onnx Model Path** → Pfad zur `.onnx`
5. Szene abspielen — kein Python erforderlich
