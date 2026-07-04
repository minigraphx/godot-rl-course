# Fehlersuche & FAQ

[Kursstartseite](index.md)

Diese Seite sammelt häufige Fehler, Warnungen und Fragen rund ums Setup und Training von RL-Agenten mit godot-native-rl, stable-baselines3 und Python — plus die Legacy-Beispiele von godot-rl-agents, die noch nicht migrierte Einheiten nutzen. Jeder Eintrag erklärt Ursache und Fix. **Hier zuerst nachsehen, bevor du ein GitHub-Issue öffnest.**

---

## Setup & Installation

### `NcnnSync` / `NcnnAIController3D` nicht unter „Node hinzufügen" zu finden

**Ursache:** Du hast ein Projekt geöffnet, das das godot-native-rl-Addon nicht enthält, oder der erste Projekt-Import ist noch nicht abgeschlossen.

**Fix:**
```bash
# Öffne das gebündelte Spielprojekt des Kurs-Repos — das Addon ist darin enthalten:
# Godot → Import → godot-rl-course/examples/neural_foundations/game/project.godot
# Lass den ersten Import durchlaufen, dann erneut: Node hinzufügen → „NcnnSync" suchen
```

Das Addon ist reines GDScript — es gibt keinen Build-Schritt. Wenn du das Addon in deinem eigenen Projekt nutzt, prüfe, ob `addons/godot_native_rl/` in den `addons/`-Ordner deines Projekts kopiert wurde.

**Siehe auch:** [Unit 0](unit-00.md) § 3

---

### „NcnnRunner"-Klasse fehlt / native Inferenz lädt nicht

**Ursache:** Die native ncnn-Inferenz benötigt eine Plattform-Binärdatei (GDExtension). Der Kurs liefert derzeit nur Binärdateien für **macOS Apple Silicon** mit — unter Windows und Linux registriert sich die `NcnnRunner`-Klasse nicht.

**Fix:** Das Training ist davon nicht betroffen — die Trainings-Bridge ist reines GDScript und funktioniert auf jeder Plattform. Überspringe die (in den Einheiten als macOS-only markierten) nativen Inferenz-Schritte, bis Multi-Plattform-Runner verfügbar sind. Das Aktivieren des Plugins unter Projekt → Projekteinstellungen → Plugins zeigt eine klare Fehlermeldung, wenn die Binärdatei für deine Plattform fehlt.

---

### Godot-Versionskompatibilitätsfehler

**Ursache:** Die gebündelte ncnn-GDExtension deklariert `compatibility_minimum = 4.5`; ältere Godot-Versionen laden das Projekt nicht sauber.

**Fix:**
```bash
# Lade Godot 4.5 oder neuer (Standard-Version) von godotengine.org herunter
# Auf der Kommandozeile prüfen:
godot --version
```

Die Legacy-Beispiele von godot-rl-agents, die ab RL Essentials genutzt werden, haben eigene Versionsanforderungen — prüfe das [godot-rl-agents](https://github.com/edbeeching/godot-rl-agents)-Repo, wenn ein Beispielprojekt Probleme macht.

---

### `gdrl: command not found`

**Ursache:** Das Python-Paket `godot-rl` ist nicht installiert oder deine conda-Umgebung ist nicht aktiv.

**Fix:**
```bash
# Activate your conda environment
conda activate godot_env

# Install using the pinned requirements file (recommended)
pip install -r requirements-course.txt

# Verify
gdrl --version
```

!!! note "Paketname ist `godot-rl`, nicht `godot-rl-agents`"
    Das PyPI-Paket heißt `godot-rl` (Installation via `pip install godot-rl` oder `requirements-course.txt`). Der alte Name `godot-rl-agents` ist nicht mehr der kanonische Paketname. Der Python-Import bleibt `import godot_rl`.

---

### `ModuleNotFoundError: No module named 'stable_baselines3'`

**Ursache:** stable-baselines3 ist im aktiven Python-Env nicht installiert.

**Fix:**
```bash
conda activate godot_env
pip install stable-baselines3
```

Bei Berechtigungsfehler: `pip install --user stable-baselines3` oder prüfen, dass dir `/usr/local/lib/python3.x/` gehört.

---

### `ModuleNotFoundError: No module named 'godot_rl'`

**Ursache:** Das Paket `godot-rl` ist in deiner aktiven conda-Umgebung nicht installiert.

**Fix:**
```bash
conda activate godot_env
pip install -r requirements-course.txt
```

Verifizieren:
```bash
python -c "import godot_rl; print(godot_rl.__version__)"
```

---

### Python-Versions-Inkompatibilität (godot-rl braucht Python 3.8+)

**Ursache:** Dein Python-Env ist älter als 3.8, oder du nutzt Python 2.

**Fix:**
```bash
# Check your Python version
python --version

# If < 3.8, create a new conda environment with Python 3.10
conda create -n godot_env python=3.10
conda activate godot_env
pip install -r requirements-course.txt
```

---

### `onnxruntime` nicht installiert (ONNX-Inferenz schlägt fehl)

**Ursache:** `onnxruntime` wird für ONNX-Export/-Ausführung benötigt, aber nicht automatisch mit stable-baselines3 installiert.

**Fix:**
```bash
conda activate godot_env
pip install onnxruntime
# or for GPU support
pip install onnxruntime-gpu
```

---

## Training startet nicht

### WebSocket-Verbindung verweigert / `ConnectionRefusedError: [Errno 111]`

**Ursache:** Das Python-Skript versucht sich auf Port 11008 (Default) mit Godot zu verbinden, aber Godot läuft nicht, lauscht nicht oder ist auf einem anderen Port.

**Fix:**
```bash
# 1. Start Godot with the training flag or use the visualizer
gdrl --load_path=examples/BallChase --viz

# 2. Then, in another terminal, run your training script
conda activate godot_env
python train.py --env_path=./godot_binary

# 3. If using a custom port, ensure both sides match
# In Python: env = GodotEnv(..., port=12000)
# In Godot: AIController.port = 12000
```

Sicherstellen, dass kein anderer Prozess Port 11008 blockiert:
```bash
lsof -i :11008   # macOS / Linux
# Windows (PowerShell):
# netstat -ano | findstr :11008
```

!!! warning "Windows: Antivirus blockiert eventuell still Port 11008"
    Unter Windows blocken Windows Defender und Dritt-Antivirus-Software den godot-rl-Socket-Port manchmal lautlos. Verbindet sich Godot trotz freien Ports nie, Firewall-Ausnahmen für `python.exe` und die Godot-Executable hinzufügen. Siehe [Windows-Erststart](setup.md#windows-first-run) für eine Schritt-für-Schritt-Anleitung.

---

### „Action space mismatch" / `ValueError: Action space size mismatch`

**Ursache:** Die Anzahl der Aktionen deines AIControllers (`n_actions`) stimmt nicht mit dem `n_actions`-Argument deines SB3-Algorithmus überein.

**Fix:**
```python
# In AIController.cs (or your Godot script)
# Count the exact actions you return
public override int GetActionSpaceSize() {
    return 2;  // e.g., 2 for move_x, move_y
}

# In your Python training script, match exactly
model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,
    ...
)
```

Beide drucken, um sicherzugehen:
```python
print("Python action space:", env.action_space.shape)
print("Godot n_actions:", env._get_obs()['n_actions'])  # or query your AIController
```

---

### `KeyError: 'obs'` beim Trainingsstart

**Ursache:** Dein Beobachtungs-Dictionary aus Godot enthält den `'obs'`-Key nicht oder nutzt den falschen Namen.

**Fix:**
```python
# In your AIController, ensure you return a dictionary with 'obs' key
public override Dictionary<string, object> GetObservation() {
    return new Dictionary<string, object> {
        { "obs", new float[] { position.X, position.Y, ... } }
    };
}

# In Python, access it as
obs = env.reset()
print(obs.keys())  # should include 'obs'
```

Gibst du ein flaches Array statt eines Dicts zurück, wrappen:
```python
return {"obs": observation_array}
```

---

### Training friert nach dem ersten Rollout ein (stiller Godot-Crash)

**Ursache:** Der Godot-Prozess ist abgestürzt oder hängt, aber Python hat den Fehler nicht empfangen. Häufig durch eine Exception in `_process()` oder `_physics_process()`.

**Fix:**
```bash
# 1. Run Godot in the foreground to see stderr/logs
gdrl --load_path=examples/BallChase --viz 2>&1 | tee godot.log

# 2. Check the Godot log for exceptions
# Common culprits: accessing null nodes, division by zero, infinite loops in reward calculation

# 3. Add debug prints in AIController
GD.Print($"Step {_step_count}: obs={obs}, action={action}, reward={reward}");
```

---

### `RuntimeError: Expected all tensors to be on the same device` (PyTorch-Fehler)

**Ursache:** Deine Beobachtung oder Belohnung ist ein PyTorch-Tensor auf der CPU, aber PyTorch erwartet alle Tensoren auf demselben Gerät (CPU oder GPU).

**Fix:**
```python
# Ensure all observations and rewards are NumPy arrays or the same tensor device
def _get_obs(self):
    obs = np.array([...], dtype=np.float32)  # Use NumPy, not torch
    return obs

# Or, if using PyTorch tensors, ensure they are all on the same device
obs = obs.to(device)  # Move to correct device
```

---

## Training läuft, lernt aber nicht

### NaN-Loss nach ein paar tausend Schritten

**Ursache:** Lernrate zu hoch, Belohnung unbeschränkt oder Pixel-Beobachtungen nicht normalisiert. Führt zu explodierenden Gradienten.

**Fix:**
```python
# Reduce the learning rate
model = PPO("MlpPolicy", env, learning_rate=1e-4)  # Instead of 1e-3 or higher

# Ensure rewards are bounded (e.g., -1 to 1)
reward = np.clip(reward, -1.0, 1.0)

# Normalize pixel observations to [0, 1] or [-1, 1]
observation = observation.astype(np.float32) / 255.0
```

---

### `ep_rew_mean` bleibt ewig flach bei 0 oder negativ

**Ursache:** Deine Belohnungsfunktion gibt jeden Schritt 0 (oder einen sehr kleinen Wert) zurück — der Agent hat kein Lernsignal. Prüfe das Vorzeichen — negative Belohnungen entmutigen.

**Fix:**
```csharp
// In AIController: reward should be non-zero when the agent makes progress
float reward = 0.0f;
if (Vector2.Distance(transform.GlobalPosition, target.GlobalPosition) < 1.0f) {
    reward = 1.0f;  // Goal reached
} else {
    reward = -0.01f;  // Small step cost to encourage progress
}

return reward;
```

Belohnungen für TensorBoard loggen, um zu debuggen:
```python
# In your training script, log rewards per episode
episode_rewards.append(total_reward)
logger.record("custom/episode_reward", total_reward)
```

---

### `ep_rew_mean` oszilliert wild und bessert sich nie

**Ursache:** Lernrate zu hoch oder `n_steps` zu klein für PPO, um vor jedem Update genug Erfahrung zu sammeln.

**Fix:**
```python
# Increase n_steps (experience per update)
model = PPO(
    "MlpPolicy",
    env,
    n_steps=4096,     # Instead of 2048
    learning_rate=3e-4,  # Reduce if still unstable
    ent_coef=0.01,    # Increase entropy coefficient to encourage exploration
)
```

Länger trainieren und Trends in TensorBoard über 100k+ Schritte ansehen, nicht nur die ersten 10k.

---

### `RuntimeError: CUDA out of memory` (GPU-Fehler)

**Ursache:** Batch-Größe zu groß für den GPU-Speicher, oder zu viele parallele Umgebungen ohne genug VRAM.

**Fix:**
```python
# Reduce the buffer size and batch size
model = PPO(
    "MlpPolicy",
    env,
    n_steps=1024,    # Smaller rollout buffer
    batch_size=64,   # Smaller training batch
)

# Or, use CPU only
env = DummyVecEnv([lambda: GodotEnv(..., use_discrete_actions=False)])  # CPU training
```

---

### Agent dreht sich auf der Stelle / tut nichts (Aktion ignoriert)

**Ursache:** Aktionsraum in Godot falsch gemappt oder Aktionen werden nicht auf den Physikkörper angewandt.

**Fix:**
```csharp
// In AIController: ensure actions are applied each frame
public override void SetAction(float[] action) {
    // action[0] and action[1] are continuous or discrete actions
    velocity.X = action[0] * max_speed;
    velocity.Y = action[1] * max_speed;
    _body.Velocity = velocity;
    _body.MoveAndSlide();
}

// Check that you call MoveAndSlide() in _physics_process
public override void _PhysicsProcess(float delta) {
    SetAction(_last_action);
    _body.Velocity = velocity;
    _body.MoveAndSlide();
}
```

---

### Entropie kollabiert früh fast auf null (Policy wird deterministisch)

**Ursache:** `ent_coef` zu klein oder die Policy ist auf deterministisches Verhalten konvergiert, bevor sie die Aufgabe gelernt hat.

**Fix:**
```python
# Increase entropy coefficient to encourage exploration
model = PPO(
    "MlpPolicy",
    env,
    ent_coef=0.1,   # Increase from default 0.0
    learning_rate=3e-4,
)

# Or, use a schedule to decay entropy over time
from stable_baselines3.common.callbacks import EvalCallback
model = PPO(
    "MlpPolicy",
    env,
    ent_coef=0.05,
    policy_kwargs={"net_arch": [128, 128]},  # Larger network = more exploration needed
)
```

---

## Verhaltens- & Belohnungs-Debugging

### TensorBoard-Belohnung steigt, aber Agent wirkt in Godot falsch

**Ursache:** Die Belohnung passt nicht zum gewünschten Verhalten. Der Agent maximiert die Zahl, ohne die Aufgabe zu lösen (Reward Hacking).

**Fix:**
```csharp
// Example: agent may "cheat" by moving quickly rather than reaching the goal
// BAD: only reward for velocity
float reward = velocity.Length() * 0.1f;

// BETTER: reward progress toward goal AND reaching goal
float distance_to_goal = Vector2.Distance(transform.GlobalPosition, goal.GlobalPosition);
float reward = -distance_to_goal * 0.1f;  // Progress penalty
if (distance_to_goal < 1.0f) {
    reward += 1.0f;  // Goal bonus
}
```

Den Agenten während des Trainings in Godot beobachten und mit der TensorBoard-Kurve vergleichen. Sie sollten geistig zusammenpassen.

---

### Agent ignoriert Hindernisse / Ziele, die sichtbar sind

**Ursache:** Die Beobachtung enthält oder aktualisiert keine Infos zu Hindernissen/Zielen, oder Beobachtungen sind gecached und werden nicht pro Schritt aktualisiert.

**Fix:**
```csharp
// In AIController: ensure observations are updated every frame
public override Dictionary<string, object> GetObservation() {
    // Observations MUST be recomputed each call, not cached
    var obs = new float[] {
        transform.GlobalPosition.X,
        transform.GlobalPosition.Y,
        target.GlobalPosition.X,
        target.GlobalPosition.Y,
        // ... include obstacle positions
    };
    return new Dictionary<string, object> { { "obs", obs } };
}

// Call in _PhysicsProcess, not _Process (to stay in sync with physics)
public override void _PhysicsProcess(float delta) {
    var obs = GetObservation();
    // Reset position next frame if goal reached
}
```

---

### Agent nutzt die Belohnung aus, statt die Aufgabe zu lösen

**Ursache:** Reward Hacking — der Agent fand einen Weg, die Zahl zu maximieren, ohne das Ziel zu erreichen.

**Fix:**
```csharp
// Example: agent may stay in place and get stuck if you only reward reaching goal
// Add a small step cost to push the agent forward
float reward = -0.01f;  // Step penalty

// Penalize if agent gets stuck
if (stuck_timer > max_stuck_time) {
    reward -= 0.5f;
}

// Reward goal
if (distance < goal_radius) {
    reward += 1.0f;
}

return reward;
```

Manuell prüfen, dass die höchstbelohnten Aktionen deiner Intuition entsprechen.

---

### `rollout/ep_len_mean` ist immer `max_episode_steps` (Ziel nie erreicht)

**Ursache:** Die done-Bedingung wird nie ausgelöst — entweder erreicht der Agent nie Erfolg, oder du gibst kein `done=True` zurück.

**Fix:**
```csharp
// In AIController: check and return done
public override bool IsDone() {
    // Done when goal is reached
    bool goal_reached = Vector2.Distance(transform.GlobalPosition, goal.GlobalPosition) < 1.0f;
    
    // Or when out of bounds
    bool out_of_bounds = transform.GlobalPosition.Length() > max_distance;
    
    // Or when max steps exceeded (framework handles this, but you can override)
    bool timeout = step_count >= max_steps;
    
    return goal_reached || out_of_bounds || timeout;
}
```

Per Print debuggen:
```python
# In Python
done_reasons = {"goal": 0, "timeout": 0, "oob": 0}
for episode in range(100):
    obs, info = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            if info.get("goal_reached"):
                done_reasons["goal"] += 1
            else:
                done_reasons["timeout"] += 1
            break
print(done_reasons)
```

---

## ONNX-Export & Deployment

### ONNX-Export gelingt, aber Godot-Inferenz produziert Müll-Aktionen

**Ursache:** Der ONNX-Export enthielt das Tanh-Squashing am Ausgang nicht, oder der Actor wurde ohne den Log-Std-Layer (SAC) exportiert.

**Fix:**
```python
# For PPO, export the full policy (includes tanh output layer)
model.policy.actor.to("cpu")
model.policy.actor.eval()

# Use the stable_baselines3 export utility
from stable_baselines3.common.policies import BasePolicy
input_dict = {"obs": np.zeros((1, obs_space.shape[0]), dtype=np.float32)}
model.policy.actor.to("cpu")

# Export with opset 11 or 15 to ensure compatibility
import onnx
onnx_model, _ = convert_pytorch(model.policy.actor, input_dict, opset_version=11)
onnx.save(onnx_model, "policy.onnx")
```

Vor dem Godot-Deployment in Python testen:
```python
import onnxruntime as ort
sess = ort.InferenceSession("policy.onnx")
action = sess.run(None, {"obs": obs})
print(action[0].shape, action[0].min(), action[0].max())  # Should be [-1, 1] for tanh
```

---

### ONNX-Modell in Godot liefert falsche Tensor-Form

**Ursache:** `input_names`/`output_names` passen nicht zum exportierten Modell, oder die Reihenfolge ist vertauscht.

**Fix:**
```csharp
// In Godot C#, when loading ONNX:
var model = new OnnxModel();
model.LoadModel("res://policy.onnx");

// Get the input/output node names from the exported model
// Use a tool like Netron to inspect the ONNX file:
// - Input node: "obs" (float32, shape [1, n_obs])
// - Output node: "output_0" (float32, shape [1, n_actions])

var input_dict = new Dictionary<string, Tensor> {
    { "obs", Tensor.FromArray<float>(obs_array) }
};
var outputs = model.Run(input_dict);
var actions = outputs["output_0"];  // or "output_name" from your export
```

ONNX-Datei mit Netron ([netron.app](https://netron.app)) prüfen, um Knotennamen und Formen zu verifizieren.

---

### ONNX-Inferenz in Godot ist langsam

**Ursache:** Modell zu groß (z. B. CNN-Policy für visuelle Obs) oder GPU-inkompatibler Runtime.

**Fix:**
```python
# Use MlpPolicy for real-time inference, not CnnPolicy
model = PPO("MlpPolicy", env, ...)  # Good for real-time
# Avoid: PPO("CnnPolicy", env, ...)  # Slow in game engine

# For visual observations, either:
# 1. Train with CnnPolicy in Python, then distill to MlpPolicy
# 2. Use very small image sizes (e.g., 32x32 instead of 84x84)
```

Typische Inferenzzeit: MlpPolicy in Godot 1–5 ms. CnnPolicy 50–200 ms.

---

### `onnxruntime.InvalidGraph: Load model from ... failed`

**Ursache:** ONNX-Opset-Version inkompatibel mit deinem onnxruntime, oder das Modell wurde nicht korrekt exportiert.

**Fix:**
```python
# Export with opset version 11 or 15 (both widely supported)
from stable_baselines3.common.policies import BasePolicy
# When exporting, specify opset_version=11

# Or, update your onnxruntime
pip install --upgrade onnxruntime
```

Auch prüfen, dass die ONNX-Datei gültig ist:
```bash
python -c "import onnx; model = onnx.load('policy.onnx'); onnx.checker.check_model(model); print('Valid')"
```

---

## Immer noch festhängen?

Wenn dein Problem hier nicht steht, prüfe:

1. **[GitHub Issues](https://github.com/edbeeching/godot-rl-agents/issues)** — nach deiner Fehlermeldung im godot-rl-agents-Repo suchen.
2. **[Godot-Docs](https://docs.godotengine.org/)** — für Godot-spezifische Fehler (Physik, Signale etc.).
3. **[SB3-Docs](https://stable-baselines3.readthedocs.io/)** — für SB3-Algorithmen und Hyperparameter.
4. **Kurs-Units** — jede Unit hat detaillierte Beispiele und eigene Troubleshooting-Abschnitte.
5. **Logging & Debugging** — Print-Statements in Godot und Python; TensorBoard zur Visualisierung.

**Vor dem Öffnen eines Issues bitte angeben:**
- Deine Python-Version (`python --version`)
- Deine Godot-Version
- Die exakte Fehlermeldung (volles Traceback)
- Dein Trainings-Skript (gesäubert)
- Ein minimal reproduzierbares Beispiel (wenn möglich)
