# Referenz

Schnelles Nachschlagen für `gdrl`-CLI-Flags, die `AIController`-API und den ONNX-Export.

!!! note "Vollständige Referenz"
    Siehe `godot_rl_course_reference.html` für die vollständige Plugin-API-Referenz.

## Häufige `gdrl`-Flags

| Flag | Standard | Beschreibung |
|------|----------|--------------|
| `--experiment_name` | — | Name für Logs und gespeichertes Modell |
| `--viz` | aus | Godot-Fenster während des Trainings anzeigen |
| `--timesteps` | 1 000 000 | Gesamtanzahl der Umgebungsschritte (environment steps) |
| `--speedup` | 1 | Zeitskalenfaktor (nur headless) |
| `--n_parallel` | 1 | Anzahl paralleler Godot-Instanzen |
| `--save_model_path` | — | Pfad zum Speichern des trainierten `.zip`-Modells |
| `--onnx_export_path` | — | Pfad zum Export von `.onnx` für Godot-Inferenz |

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
