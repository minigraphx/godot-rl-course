# Visuelle Beobachtungen — Agenten das Sehen beibringen

[← Kontinuierliches 3D](unit-06.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~30 min · Training: ~45 min GPU / ~3 Std CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    **Godot** (füge ein `TextureRect` hinzu, das die SubViewport-Textur anzeigt — du kannst zusehen, was der Agent wörtlich sieht) · **TensorBoard** (erwarte Kurven, die 5–10× langsamer sind als deine Raycast-Baseline — das ist normal) · **Pixel-Statistiken ausgeben** (`min`, `max`, `mean` pro Kanal nach Normalisierung — alle Werte sollten in [0, 1] liegen und der Mittelwert sollte deutlich über 0 sein)

---

## 1 · Warum visuelle Beobachtungen?

Jede Unit bisher hat **Raycasts** benutzt, um die Welt zu beobachten: eine Handvoll Strahlen schießen vom Agenten aus, messen die Distanz zur nächsten Fläche und liefern ein kompaktes Float-Array zurück. Raycasts sind schnell, leicht zu durchdenken und funktionieren gut für strukturierte Szenen. Warum solltest du sie also jemals durch Kamerapixel ersetzen?

### Was Raycasts dir geben

Acht Raycasts erzeugen acht Floats. Das sind 8 Werte, jeder mit klarer physikalischer Bedeutung („Wand 2,3 m voraus"). Ein neuronales Netz lernt daraus innerhalb von Zehntausenden von Schritten.

### Was visuelle Beobachtungen dir geben

Ein 64×64 RGB-Bild erzeugt 12 288 Werte. Diese Werte sind rohe Pixelfarben ohne inhärente Semantik — das Netz muss von Grund auf lernen, welche Muster wichtig sind.

| | Raycast | Visuell (64×64 RGB) |
|---|---|---|
| Beobachtungsgröße | ~8 Werte | 12 288 Werte |
| Feature Engineering | Du entscheidest, was gecastet wird | Keines — CNN lernt es |
| Interpretierbarkeit | Leicht auszugeben und zu inspizieren | Sehr schwer zu debuggen |
| Typische Trainingsschritte | 500k–1M | 3M–10M |
| Funktioniert in überfüllten/abwechslungsreichen Szenen | Nur wenn du die richtigen Strahlen entworfen hast | Ja |

### Vorteile visueller Beobachtungen

**Kein Feature Engineering.** Du entscheidest nicht, was gemessen wird. Das Convolutional Neural Network (CNN) entdeckt, welche visuellen Merkmale Belohnung vorhersagen. Das ist sowohl die Stärke als auch der Preis: du gewinnst Allgemeinheit, du bezahlst mit Rechenleistung.

**Funktioniert, wo Raycasts nicht können.** Stell dir eine Szene mit zufällig verstreuten Objekten vieler verschiedener Formen vor. Eine Raycast-Konfiguration zu entwerfen, die jedes relevante Objekt erfasst, ist mühsam. Eine Kamera sieht alles in ihrem Sichtfeld automatisch.

**Transfer-Potenzial.** Ein auf einer visuell ähnlichen Umgebung trainiertes CNN passt sich oft schneller an eine neue an. Die visuellen Low-Level-Merkmale (Kanten, Farben, Texturen) sind auf eine Weise wiederverwendbar, wie es handgefertigte Raycast-Konfigurationen nicht sind.

### Nachteile visueller Beobachtungen

**Viel größerer Beobachtungsraum.** 12 288 Floats vs 8. Jede Netzschicht, die diese Beobachtung verarbeitet, ist proportional größer, langsamer und hungriger nach GPU-Speicher.

**Erfordert einen CNN-Feature-Extractor.** Deine MLP-Policy kann ein 64×64-Bild nicht direkt sinnvoll verarbeiten. Du musst ein Convolutional Network voranstellen. Stable-Baselines3 stellt eines bereit (`NatureCNN`), aber es fügt Tausende von Parametern hinzu und verlangsamt jedes Update.

**Schwerer zu debuggen.** Mit Raycasts kannst du `print(get_obs())` machen und die Zahlen sofort verstehen. Mit Pixeln bedeutet Debugging, dem SubViewport live in Godot zuzusehen oder Feature Maps zu visualisieren — beides erfordert mehr Tooling.

**Braucht weit mehr Trainingsschritte.** Erwarte 3–10× mehr Umgebungsschritte als bei einer äquivalenten Raycast-Aufgabe. Eine Aufgabe, die mit Raycasts in 500k Schritten konvergiert, kann mit visuellen Beobachtungen 3–5 Millionen benötigen.

### Faustregel

> Verwende Raycasts, wenn du kannst. Verwende visuelle Beobachtungen, wenn Raycasts nicht erfassen können, was wichtig ist — oder wenn Generalisierung über abwechslungsreiche visuelle Szenen ein Projektziel ist.

---

## 2 · Godot SubViewport → Beobachtungs-Pipeline

Die Kernidee: ein `SubViewport`-Knoten rendert einen separaten Kamera-Feed innerhalb von Godot. Dein `AIController3D` liest diese gerenderte Textur, flacht die Pixelwerte zu einem Float-Array ab und gibt sie als Agentenbeobachtung zurück.

### Szenenaufbau

1. Füge einen `SubViewport`-Knoten als Kind des Agent-Root-Knotens hinzu.
2. Setze die `SubViewport`-Größe auf **64×64** (oder 84×84 für reicheres Detail — aber das Training ist langsamer).
3. Setze `SubViewport.render_target_update_mode` auf `ALWAYS`, damit er jeden Frame rendert.
4. Füge eine `Camera3D` **innerhalb** des SubViewport hinzu. Platziere sie an der „Augen"-Position des Agenten — normalerweise leicht über und vor dem Zentrum der Kollisionsform.
5. Der SubViewport rendert unabhängig von der Hauptkamera. Der Agent „sieht" durch diese innere Kamera.

!!! warning "Prüfe, ob der Viewport rendert, bevor du trainierst"
    Ein SubViewport, dessen `render_target_update_mode` auf dem Standardwert (`ONCE`) belassen wird, rendert nur den ersten Frame und friert dann ein. Dein Agent trainiert auf einem statischen Bild und lernt nichts Sinnvolles. Setze den Modus auf `ALWAYS`, bevor du eine einzige Zeile Trainingscode schreibst.

### GDScript: Pixel als Beobachtungen erfassen

```gdscript
# ai_controller.gd — capture camera frames as observations
extends AIController3D

@onready var viewport = $SubViewport
@onready var camera   = $SubViewport/Camera3D

const IMG_WIDTH  = 64
const IMG_HEIGHT = 64

func get_obs() -> Dictionary:
    # Get the rendered frame from SubViewport
    var img: Image = viewport.get_texture().get_image()
    img.resize(IMG_WIDTH, IMG_HEIGHT, Image.INTERPOLATE_BILINEAR)
    img.convert(Image.FORMAT_RGB8)

    # Flatten to float array, normalize to [0, 1]
    var obs = []
    for y in range(IMG_HEIGHT):
        for x in range(IMG_WIDTH):
            var pixel = img.get_pixel(x, y)
            obs.append(pixel.r)
            obs.append(pixel.g)
            obs.append(pixel.b)

    return {"obs": obs}

func get_obs_size() -> int:
    return IMG_WIDTH * IMG_HEIGHT * 3  # 64×64×3 = 12,288
```

`Image.get_pixel()` gibt eine `Color` zurück, deren `.r`-, `.g`-, `.b`-Kanäle bereits in `[0.0, 1.0]` liegen, wenn das Format `FORMAT_RGB8` ist. Keine weitere Normalisierung erforderlich.

Die Schleife besucht Pixel in row-major-Reihenfolge: zuerst alle Pixel von Zeile 0, dann Zeile 1 und so weiter. Die Kanalreihenfolge ist R, G, B pro Pixel. Halte diese Reihenfolge konsistent mit der Art, wie du das Array auf der Python-Seite umformst (siehe Abschnitt 4).

---

## 3 · Graustufen und Frame Stacking

Zwei Standardtechniken reduzieren die Beobachtungsgröße und geben dem Agenten zeitliches Bewusstsein, ohne ein rekurrentes Netz hinzuzufügen.

### Graustufen

Die Konvertierung in Luminanz reduziert von drei Kanälen auf einen — eine 3-fache Reduktion der Beobachtungsgröße:

```gdscript
# Grayscale: convert to luminance before flattening
func get_obs() -> Dictionary:
    var img: Image = viewport.get_texture().get_image()
    img.resize(IMG_WIDTH, IMG_HEIGHT, Image.INTERPOLATE_BILINEAR)
    img.convert(Image.FORMAT_L8)  # convert to luminance (grayscale)

    var obs = []
    for y in range(IMG_HEIGHT):
        for x in range(IMG_WIDTH):
            var pixel = img.get_pixel(x, y)
            obs.append(pixel.r)  # luminance stored in .r when FORMAT_L8

    return {"obs": obs}

func get_obs_size() -> int:
    return IMG_WIDTH * IMG_HEIGHT  # 64×64 = 4,096 values
```

!!! tip "Starte mit Graustufen"
    Wenn Farbe nicht wirklich wichtig für die Aufgabe ist (z. B. einen roten Feind von einem grünen Verbündeten zu unterscheiden), trainiere zuerst in Graustufen. Es trainiert 3× schneller und ist leichter zu debuggen. Füge Farbe nur hinzu, wenn Graustufen deutlich scheitern.

### Frame Stacking

Ein einzelner Frame sagt dem Agenten, wo Objekte sind. Er sagt dem Agenten nicht, wie schnell sie sich bewegen oder in welche Richtung. **Frame Stacking** konkateniert die letzten N Frames zu einer Beobachtung und gibt der Policy implizite Geschwindigkeitsinformationen ohne rekurrentes Netz.

```gdscript
# Frame stack: keep the last 4 grayscale frames
const STACK_SIZE = 4
var frame_buffer: Array = []

func get_obs() -> Dictionary:
    var frame = _capture_grayscale_frame()
    frame_buffer.push_back(frame)
    if frame_buffer.size() > STACK_SIZE:
        frame_buffer.pop_front()

    # Pad with zero frames at the start of an episode
    while frame_buffer.size() < STACK_SIZE:
        frame_buffer.push_front(Array.filled(IMG_WIDTH * IMG_HEIGHT, 0.0))

    # Concatenate all frames into one flat array
    var obs = []
    for f in frame_buffer:
        obs.append_array(f)
    return {"obs": obs}  # 4 × 64 × 64 = 16,384 values

func get_obs_size() -> int:
    return STACK_SIZE * IMG_WIDTH * IMG_HEIGHT

func _capture_grayscale_frame() -> Array:
    var img: Image = viewport.get_texture().get_image()
    img.resize(IMG_WIDTH, IMG_HEIGHT, Image.INTERPOLATE_BILINEAR)
    img.convert(Image.FORMAT_L8)
    var frame = []
    for y in range(IMG_HEIGHT):
        for x in range(IMG_WIDTH):
            frame.append(img.get_pixel(x, y).r)
    return frame
```

Stelle sicher, dass du `frame_buffer.clear()` (oder den Buffer zurücksetzen) in `on_episode_end()` aufrufst — sonst kontaminieren Frames der vorigen Episode die nächste.

**Warum Frame Stacking funktioniert — die Atari-DQN-Intuition.** Im ursprünglichen Atari-DQN-Paper (Mnih et al., 2015) ermöglichten vier gestapelte Graustufen-Frames von Pong dem Netz, sowohl die Position des Balls *als auch* seine Geschwindigkeit zu inferieren. Ein einzelner Frame reicht nicht aus, um zu wissen, ob sich der Ball nach links oder rechts bewegt. Vier aufeinanderfolgende Frames machen die Richtung selbst für ein einfaches CNN offensichtlich.

---

## 4 · CNN-Feature-Extractor in SB3

Stable-Baselines3 erwartet Bildbeobachtungen geformt als **(channels, height, width)** — die PyTorch-Konvention, auch „channels first" genannt. Dein Godot-Code gibt ein flaches 1D-Array zurück. Du musst es umformen, bevor die Policy es sieht.

Der sauberste Ansatz ist ein `gym.ObservationWrapper`, der auf der Python-Seite umformt:

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv


class VisualObsWrapper(gym.ObservationWrapper):
    """Reshape flat obs [H*W*C] → (C, H, W) tensor for CNN."""

    def __init__(self, env, height=64, width=64, channels=1):
        super().__init__(env)
        self.h, self.w, self.c = height, width, channels
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(channels, height, width),
            dtype=np.float32,
        )

    def observation(self, obs):
        # obs is shape (H*W*C,); reshape and move channels first
        return obs.reshape(self.c, self.h, self.w)


# Build the environment
base_env = StableBaselinesGodotEnv(
    env_path="./VirtualCamera.x86_64",
    n_parallel=4,
    speedup=10,
)
env = VisualObsWrapper(base_env, height=64, width=64, channels=1)  # grayscale

# "CnnPolicy" uses NatureCNN under the hood
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/",
    n_steps=512,        # smaller rollout — large obs eats memory faster
    batch_size=64,      # smaller batch for the same reason
    learning_rate=1e-4, # lower lr — CNN has far more parameters than MLP
)

model.learn(total_timesteps=3_000_000)  # visual tasks need 3–5× more steps
model.save("virtualcamera_cnn")
env.close()
```

!!! check "Fertig, wenn"
    Der CNN-Agent lernt dieselbe Aufgabe, die deine Raycast-Baseline bereits gelöst hat — auch wenn er dafür die 3–10× mehr Schritte braucht, vor denen diese Unit warnt. Konkret: `ep_rew_mean` in TensorBoard steigt deutlich über das anfängliche Plateau der Zufalls-Policy, und der Viz-Checkpoint (Abschnitt 10) zeigt, dass der Agent auf Objekte reagiert, sobald sie im SubViewport-Feed auftauchen. Mache es nicht zur Bedingung, die finale Belohnung des Raycast-Laufs zu erreichen — dieser Vergleich ist seed-verrauscht. Ist die Kurve am Ende deines Schritt-Budgets noch flach, prüfe zuerst auf einen eingefrorenen SubViewport oder kaputte Normalisierung (Abschnitt 8), bevor du mehr Schritte ansetzt.

Wichtige Änderungen gegenüber einem Raycast-Trainingsskript:

| Parameter | Raycast typisch | Visuelle Obs typisch |
|---|---|---|
| `policy` | `"MlpPolicy"` | `"CnnPolicy"` |
| `n_steps` | 2048 | 512 |
| `batch_size` | 64–256 | 32–64 |
| `learning_rate` | 3e-4 | 1e-4 |
| `total_timesteps` | 500k–1M | 3M–10M |

---

## 5 · Die NatureCNN-Architektur

Das eingebaute CNN von SB3 ist die Architektur aus dem DeepMind Atari DQN Nature-Paper (2015). Sie wurde für 84×84-Graustufen-Frames entworfen und ist zum Standard-Startpunkt für visuelles RL geworden:

```
Input: (C, 84, 84)
Conv2d(C,  32, kernel=8, stride=4) → (32, 20, 20)   ReLU
Conv2d(32, 64, kernel=4, stride=2) → (64,  9,  9)   ReLU
Conv2d(64, 64, kernel=3, stride=1) → (64,  7,  7)   ReLU
Flatten                            → (3136,)
Linear(3136, 512)                  → (512,)          ReLU
```

Output: 512-dimensionaler Feature-Vektor, weitergegeben an den Actor-Kopf (Policy) und den Critic-Kopf (Value Function).

**Für 64×64-Input** sind die räumlichen Dimensionen nach den drei Convolutions kleiner als für 84×84. Die geflachte Größe ist anders, aber SB3 berechnet sie automatisch. Du kannst die finale Feature-Dimension reduzieren, um zu passen:

```python
from stable_baselines3.common.torch_layers import NatureCNN

policy_kwargs = dict(
    features_extractor_class=NatureCNN,
    features_extractor_kwargs=dict(features_dim=256),  # smaller for 64×64
)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
```

Das `features_dim`-kwarg setzt die Größe der finalen linearen Schicht (hier 512 → 256). Ein kleinerer Feature-Vektor reduziert die Größe der Actor-/Critic-Köpfe und beschleunigt das Training leicht.

---

## 6 · Custom CNN für deine Umgebung (optional beim ersten Lesen)

!!! note "Erster Durchgang? Überfliege oder überspringe diesen Abschnitt."
    Die Standard-`CnnPolicy` aus Abschnitt 4 reicht für deinen ersten visuellen Trainingslauf völlig aus. Der Kernpfad durch diese Unit sind die Abschnitte 1–4 (SubViewport-Pipeline und erster CNN-Lauf), Abschnitt 7 (die VirtualCamera-Referenzszene) und Abschnitt 10 (Viz-Checkpoint). Komm hierher zurück, wenn das Standard-CNN funktioniert, dir die Iteration aber zu langsam ist.

`NatureCNN` ist für Atari dimensioniert. Wenn deine Aufgabe einfache Visualisierung hat — flache Farben, klare Formen, nicht viel Detail — trainiert ein viel leichteres CNN schneller und generalisiert genauso gut:

```python
class SimpleCNN(BaseFeaturesExtractor):
    """Lightweight CNN for simple visual tasks."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flattened size from the conv layers dynamically
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flat = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=SimpleCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
```

Der `with torch.no_grad()`-Block berechnet die Output-Größe des Conv-Stacks, indem ein Dummy-Tensor hindurchgeschickt wird. Das vermeidet hartcodierte Magic Numbers, die brechen würden, sobald du die Bildauflösung änderst.

Wann `SimpleCNN` statt `NatureCNN`:

- Bildauflösung ist 32×32 oder 64×64 (nicht 84×84)
- Szene hat einfache Geometrie (wenige Objekte, flache Texturen)
- Du willst schnellere Iteration während früher Experimente
- Speicher ist beschränkt (z. B. Training auf CPU)

---

## 7 · VirtualCamera-Beispiel aus godot_rl_agents_examples

Das offizielle `godot_rl_agents_examples`-Repository enthält eine fertige Szene mit visuellen Beobachtungen. Verwende sie als Referenzimplementierung, bevor du deine eigene baust.

### Setup

```bash
git clone https://github.com/edbeeching/godot_rl_agents_examples
```

Navigiere zu `examples/VirtualCamera/`. Die Szene enthält:

- Einen Agenten mit `SubViewport` und `Camera3D` darin
- `ai_controller.gd`, das `get_obs()` implementiert und geflachte Pixelwerte zurückgibt
- Einen kontinuierlichen Aktionsraum für Lenkung und Gas

### Sehen, was der Agent sieht

Füge ein Debug-Display-Overlay zur Szene hinzu, während du entwickelst:

```gdscript
# Debug: show the SubViewport texture in the running scene
func _ready():
    var display = TextureRect.new()
    display.texture = $SubViewport.get_texture()
    display.set_anchors_and_offsets_preset(Control.PRESET_TOP_LEFT)
    display.size = Vector2(128, 128)  # scaled up for visibility
    get_tree().root.add_child(display)
```

Das rendert den SubViewport-Feed in der Ecke des Hauptfensters. Bevor du mit dem Training startest, bestätige:

- Das Bild aktualisiert sich jeden Frame (nicht eingefroren)
- Der Kamerawinkel ist sinnvoll — der Agent sollte die Strecke/Hindernisse vor sich sehen
- Das Bild ist nicht komplett schwarz oder komplett weiß

Entferne oder verstecke das Debug-Display, bevor du das für paralleles Training verwendete Binary exportierst. Das Rendern zusätzlicher TextureRect-Knoten in 32 parallelen Instanzen verschwendet GPU-Zeit.

---

## 8 · Trainings-Tipps für visuelle Beobachtungen

| Problem | Ursache | Fix |
|---|---|---|
| Sehr langsames Training | Große Obs (64×64×3 = 12k Werte) | Verwende Graustufen (÷3) + skaliere auf 32×32, um zuerst schnell zu iterieren |
| Policy lernt nicht | Zufälliges Pixelrauschen dominiert Gradient | Normalisiere auf [0, 1]; verifiziere, dass SubViewport korrekt rendert |
| Hohe RAM-Nutzung | Großer Buffer mit Pixel-Beobachtungen | Reduziere `buffer_size` auf 100k; verwende `optimize_memory_usage=True` in DQN |
| CNN nutzt Farbe nicht | Alle Kanäle sehen identisch aus | Prüfe Bildformat; erwäge Wechsel zu explizitem Graustufen |
| `approx_kl` explodiert | Lernrate zu hoch für CNN | Senke `learning_rate` auf 1e-4; CNNs haben weit mehr Parameter als MLPs |
| Loss wird NaN | Unnormalisierte Pixel (Werte 0–255) | Stelle sicher, dass der `img.get_pixel().r`-Pfad verwendet wird, nicht rohe Byte-Werte |
| Agent ignoriert Objekte am Rand | Sichtfeld zu eng | Erweitere Kamera-FOV; positioniere Kamera weiter zurück |

!!! warning "Verifiziere, dass der SubViewport rendert, bevor du trainierst"
    Ein eingefrorener SubViewport (falsches `render_target_update_mode`) gibt dem Agenten in jedem Schritt identische Beobachtungen. Die Policy konvergiert zu einem zufälligen oder trivialen Verhalten, und der Loss sinkt nicht. Bestätige immer, dass der SubViewport live ist, bevor du einen langen Trainingslauf startest.

!!! tip "Graustufen zuerst, Farbe danach"
    Starte jede neue visuelle Aufgabe mit Graustufen bei 32×32 oder 64×64. Verifiziere, dass der Agent etwas lernt. Skaliere dann die Auflösung hoch oder füge Farbe hinzu, nur wenn du Beweise hast, dass Farbinformation hilft. Jede Erhöhung der Auflösung vervielfacht die Trainingszeit.

### BatchNorm vs LayerNorm vs VecNormalize (optional beim ersten Lesen)

!!! note "Erster Durchgang? Überfliege oder überspringe diesen Abschnitt."
    Nichts davon ist für einen ersten erfolgreichen Lauf nötig — der Kernpfad sind die Abschnitte 1–4 (SubViewport-Pipeline und erster CNN-Lauf), Abschnitt 7 (die VirtualCamera-Referenzszene) und Abschnitt 10 (Viz-Checkpoint). Komm hierher zurück, falls dein CNN-Training instabil wird.

Normalisierungsschichten sind Standard im überwachten Deep Learning, aber ihre Nutzung in RL ist nuanciert — die Nichtstationarität von RL-Daten erzeugt spezifische Probleme.

**BatchNorm** berechnet Statistiken über den Batch während des Trainings und verwendet laufende Statistiken während der Inferenz. In RL verschiebt sich die Datenverteilung, wenn die Policy sich verbessert; auf frühen Trainingsdaten berechnete Batch-Statistiken werden veraltet. Das kann Instabilität verursachen und wird in modernen Deep-RL-Netzen weitgehend aufgegeben.

**LayerNorm** normalisiert pro Sample, nicht pro Batch. Keine laufenden Statistiken, die veralten können; kompatibel mit nichtstationären Daten. Bevorzugt, wenn Normalisierung innerhalb eines Deep-RL-Netzes nötig ist.

**VecNormalize** (SB3-Wrapper) normalisiert die *Beobachtung*, bevor sie ins Netz geht — nicht innerhalb des Netzes. Operiert auf Umgebungsebene, nicht auf Schichtebene.

| Schicht | Verwendet in | Hinweis |
|-------|---------|------|
| Keine Normalisierung | SB3-Standard-MLP | Funktioniert gut für niedrigdimensionale Obs |
| BatchNorm | Frühe DQN-Paper | Weitgehend aufgegeben in modernem RL |
| LayerNorm | Transformer-basierte Policies, große Netze | Bevorzugt, wenn Normalisierung nötig ist |
| VecNormalize | SB3-Wrapper | Obs-Level-Normalisierung, nicht Schicht-Level |

**Praktische Regel:** für MLP-Policies auf niedrigdimensionalen Beobachtungen keine Normalisierung hinzufügen — verwende stattdessen `VecNormalize`. Für CNN-Policies auf Bildern: füge LayerNorm nach Conv-Schichten hinzu, falls das Training instabil ist. Beispiel:

```python
class NatureCNNWithNorm(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.GroupNorm(1, 32),          # GroupNorm(1, C) ≡ LayerNorm but input-size-agnostic
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
```

---

## 9 · Hybrid: visuelle + propriozeptive Beobachtungen

In der Praxis ist das effektivste Setup oft **nicht** reine visuelle Beobachtungen. Du gibst dem CNN den Kamera-Feed *und* gibst rohe propriozeptive Daten (Geschwindigkeit, Position, Heading) direkt an die Policy-MLP weiter — am CNN vorbei. Die beiden Ströme werden vor den Actor- und Critic-Köpfen konkateniert.

Godot-Seite — gib ein Dictionary mit zwei Schlüsseln zurück:

```gdscript
func get_obs() -> Dictionary:
    var pixels = _capture_grayscale_frame()   # flat array, 64×64 values
    var state  = [
        linear_velocity.x,
        linear_velocity.y,
        linear_velocity.z,
        rotation.y,
        distance_to_target,
        on_floor_bool,
    ]
    return {
        "image": pixels,
        "state": state,
    }
```

Python-Seite — SB3s `MultiInputPolicy` behandelt Dict-Beobachtungsräume automatisch:

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

# The Godot env must expose a dict obs space with keys "image" and "state"
env = StableBaselinesGodotEnv(env_path="./HybridAgent.x86_64", n_parallel=4)

# MultiInputPolicy: CNN for image keys, MLP for vector keys, outputs concatenated
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=2_000_000)
```

SB3 inspiziert das Dict des Beobachtungsraums beim Konstruieren. Jeder Schlüssel, dessen Space ein `Box` mit drei Dimensionen ist, wird als Bild behandelt und durch das CNN geroutet. Jeder 1D-`Box`-Schlüssel wird als flacher Vektor behandelt und durch ein kleines MLP geroutet. Die Outputs werden konkateniert und in den gemeinsamen Actor/Critic-Trunk eingespeist.

**Warum Hybride oft rein visuelle übertreffen.** Das CNN muss Geschwindigkeit und Heading aus Pixeldifferenzen ableiten — eine schwere Aufgabe selbst mit Frame Stacking. Diese Zahlen direkt als Zustand zu geben, eliminiert diese Herausforderung. Das CNN konzentriert sich dann rein auf die Erkennung von Hindernissen und Zielen, wofür es gut geeignet ist.

---

## 10 · Viz-Checkpoint

Nach dem Training führe immer eine visuelle Inspektion mit `--viz` durch:

```bash
python train.py --env_path ./VirtualCamera.x86_64 --viz
```

In der laufenden Szene:

1. Füge ein `TextureRect` hinzu, das die SubViewport-Textur anzeigt (siehe Abschnitt 7), damit du sehen kannst, was der Agent wörtlich sieht.
2. Beobachte: reagiert der Agent auf Objekte, sobald sie ins Sichtfeld der Kamera eintreten? Du solltest ausweichendes oder zielgerichtetes Verhalten sehen, sobald ein relevantes Objekt im SubViewport-Feed erscheint.
3. Vergleiche das Verhalten: wenn du einen Raycast-Agenten auf derselben Aufgabe trainiert hast, lasse beide nebeneinander laufen. Der visuelle Agent bewegt sich oft glatter, weil das CNN über Teilansichten generalisiert — der Agent beginnt zu reagieren, bevor ein Raycast das Objekt überhaupt treffen würde.

Worauf achten:

- **Gutes Zeichen**: der Agent dreht sich zu (oder weg von) Objekten, sobald sie im oberen Teil seines SubViewport-Feeds erscheinen.
- **Schlechtes Zeichen**: der Agent ignoriert Objekte, die im SubViewport klar sichtbar sind — das deutet darauf hin, dass das CNN nicht konvergiert oder die Normalisierung falsch ist.
- **Schlechtes Zeichen**: der Agent verhält sich in allen Situationen identisch — der SubViewport könnte eingefroren sein.

---

## 11 · Stretch Goals

Arbeite diese nach Abschluss der Hauptaufgabe durch. Jede isoliert eine Variable, um Intuition für visuelles RL aufzubauen:

**Auflösungs-Sweep.** Trainiere dieselbe Aufgabe bei 16×16, 32×32 und 64×64. Plotte finale Belohnung (nach gleicher Anzahl Trainingsschritte) vs Auflösung. Bei welcher Auflösung hört mehr Detail auf zu helfen? Wie skaliert die Trainingszeit mit der Auflösung?

**Graustufen vs Farbe.** Nimm eine Aufgabe, bei der Farbe nützliche Information liefert (z. B. roter Feind vs grüner Verbündeter). Trainiere einmal mit Graustufen und einmal mit RGB. Miss Sample-Effizienz (Schritte bis zu einer Zielbelohnung). Wie groß ist der Unterschied?

**Frame-Stacking-Ablation.** Erzeuge eine Umgebung, in der sich ein Objekt durch die Szene bewegt. Trainiere mit 1 Frame, 2 Frames und 4 gestapelten Frames. Wann hilft Stacking? Profitiert eine Aufgabe mit stationärem Ziel überhaupt vom Stacking?

**Transfer-Experiment.** Trainiere bis zur Konvergenz auf einer visuellen Umgebung. Speichere das Modell. Fine-tune auf einer visuell ähnlichen Umgebung (gleiche Aufgabe, andere Texturen oder Beleuchtung). Wie viele Schritte braucht Fine-Tuning im Vergleich zum Training von Grund auf? Vergleiche die CNN-Feature-Gewichte vor und nach dem Fine-Tuning.

---

## Was kommt als Nächstes?

Du hast jetzt einem Agenten das Sehen beigebracht. Die verbleibende Herausforderung ist Koordination: was passiert, wenn mehrere Agenten dieselbe Umgebung teilen?

**Unit 7 — Multi-Agent** führt kooperative und kompetitive Szenarien ein, in denen mehrere Agenten gleichzeitig handeln. Du lernst, wie godot-rl-agents Multi-Agent-Umgebungen behandelt, wie sich die Belohnungsstruktur ändert und wie unabhängiges PPO (jeder Agent trainiert seine eigene Policy) im Vergleich zu Shared-Weight-Policies abschneidet.

[→ Multi-Agent](unit-07.md)
