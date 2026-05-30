# Sim-to-Real-Transfer — Domain Randomization und die Realitätslücke

[← Ziel-bedingtes RL & HER](unit-her.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~30 min

!!! info "Drei Wege, deine KI zu beobachten"
    - **Godot** — teste deine Policy an den *Extremen* deiner Domain Randomization (max Reibung, min Masse, max Rauschen). Überlebt sie dort, hat sie eine Chance auf Hardware.
    - **TensorBoard** — beobachte `ep_rew_mean` und Belohnungs-*Varianz* über Seeds. Niedrige Varianz unter DR ist dein Proxy für Sim-to-Real-Robustheit.
    - **Deployment-Checkliste** — ein physischer Roboter verzeiht nichts. Bevor du Firmware flashst, gehe Abschnitt 7 Zeile für Zeile durch.

---

Du hast Vierbeiner, Manipulatoren und ziel-bedingte Greifer in Godot gebaut. Du hast sie mit PPO, SAC und HER trainiert. Sie funktionieren wunderschön in der Simulation.

Jetzt willst du, dass sie auf einem echten Roboter funktionieren.

Diese Unit ist die Brücke. Sie ist das praktisch wichtigste — und demütigendste — Kapitel angewandter Robotik-RL.

## 1 · Die Realitätslücke

Das mit Abstand wichtigste Konzept der angewandten Robotik-RL ist dies:

> Eine in Simulation trainierte Policy versagt oft *komplett*, wenn sie auf echter Hardware deployt wird.

Nicht „läuft etwas schlechter". Nicht „braucht etwas Fein-Tuning". Sie kippt um. Sie zerquetscht das Objekt. Sie divergiert in eine NaN-Spirale und der Notaus löst aus.

Die klassische Mahnung ist OpenAIs Dexterous Hand. Sie trainierten eine 24-DOF-Shadow-Hand, um einen Rubik-Würfel zu manipulieren, mit **rund 10 000 CPU-Kernen** und dem Äquivalent von **etwa 100 Jahren simulierter Erfahrung**. Beim ersten Deployment der ersten Version auf der echten Hand versagte sie. Die Simulation — obwohl einer der sorgfältigsten je gebauten Robotik-Simulatoren — war *nicht physikalisch genau genug*. Kontaktdynamik, Sehnensteifigkeit und Fingerreibung waren jeweils etwas falsch, und das Vertrauen der Policy war fehlplatziert.

### Woher die Lücke kommt

Die Realitätslücke ist kein einzelner Bug — sie ist ein Stapel von Diskrepanzen zwischen zwei physischen Modellen (dem Simulator und der Realität):

- **Physik-Ungenauigkeit.** Simulatoren integrieren Starrkörperdynamik mit diskreten Zeitschritten. Echte Reibung ist Stick-Slip. Echte Kontakte deformieren. Motorsteifigkeit hängt von der Temperatur ab. Sim approximiert all das.
- **Observation-Rauschen.** Echte Sensoren haben Rauschböden, Kalibrierungsdrift, gelegentliche Ausfälle und Quantisierung. Ein simulierter IMU gibt die Ground-Truth-Orientierung zurück; ein echter IMU gibt *etwas nahe der* Orientierung zurück, plus weißes Rauschen, plus einen langsamen Bias.
- **Aktuator-Lag.** Echte Motoren haben Latenz: 50–200 ms zwischen kommandiertem und produziertem Drehmoment, je nach Controller, Bus (CAN, EtherCAT, USB) und Motorelektronik. Simulatoren haben standardmäßig *null* Latenz.
- **Unmodellierte Dynamik.** Kabelführung zieht an Gliedern. Thermische Ausdehnung verschiebt Encoder-Nullpunkte. Spiel in Getrieben. Verschleiß ändert Reibung über Monate.
- **Visuelle Lücke.** Nutzt deine Policy Kameras, werden Sim-Texturen und Beleuchtung nie zu echten Photonen passen.

### Eine konkrete Konsequenz

Stell dir vor, du trainierst einen Vierbeiner in Godot zu gehen. Die Boden-`PhysicsMaterial.friction` ist 0,9. Die Policy lernt einen selbstbewussten, federnden Gang, der bei jedem Schritt fest gegen den Boden drückt.

Du deployst auf einem polierten Beton-Laborboden. Echte Reibung: 0,6. Der erste Schub rutscht. Der Roboter versucht sich zu fangen, rutscht erneut und fällt auf die Brust, bevor du den Notaus drücken kannst.

Sim-Belohnung: 980/1000. Echte Belohnung: null, plus eine verbogene Schienbeinplatte.

Der Punkt ist nicht „mach den Simulator perfekt". Perfekte Simulatoren existieren nicht, und je näher du an „perfekt" drückst, desto steiler sind die abnehmenden Erträge. Der Punkt ist der nächste Abschnitt.

### Warum „kauf einfach einen besseren Simulator" nicht funktioniert

Es ist verlockend zu denken, die Antwort sei eine bessere Physik-Engine — weichere Kontaktmodelle, FEM-basierte Deformierbare, GPU-beschleunigte Starrkörper-Solver. Die helfen. Aber drei strukturelle Gründe machen „genauere Sim" allein unzureichend:

1. **Du kannst die Realität nicht präzise genug messen.** Selbst mit perfektem Simulator musst du ihn mit der *tatsächlichen* Masse, Reibung, Motorparametern *deines konkreten* Roboters füttern. Diese Werte driften über Einheiten, Temperaturen und über die Lebenszeit des Roboters. Die Sim ist nur so genau wie die Parameter, die du reinsteckst.
2. **Manche Physik ist wirklich schwer.** Stick-Slip-Reibung, Kabelhysterese, sehnen-getriebene Kopplung, Weichgewebskontakt — das sind aktive Forschungsprobleme der numerischen Simulation, kein gelöstes Engineering.
3. **Mehr Genauigkeit heißt langsamere Sim.** Eine 100×-genauere Physik-Engine, die 100× langsamer läuft, trainiert 100× weniger Episoden pro Wand-Clock-Stunde. Der Dateneffizienz-Hit überwiegt meist den Genauigkeitsgewinn.

Domain Randomization umgeht alle drei. Du brauchst keine exakten Parameter — du brauchst *genug* Parameter in deiner Verteilung. Du musst Stick-Slip nicht exakt modellieren — du musst die Reibungsparameter genug randomisieren, um beide Regime abzudecken. Du brauchst keine langsamere Sim — Randomisierung läuft bei voller Geschwindigkeit.

---

## 2 · Domain Randomization — die Kernlösung

Die Schlüsseleinsicht, von OpenAI 2017 artikuliert und jetzt Standardpraxis in Robotik-Laboren:

> Statt zu versuchen, die Realität perfekt zu modellieren, **randomisiere Simulationsparameter so weit, dass die Realität nur ein weiteres Sample aus deiner Trainingsverteilung ist.**

Trainierst du auf Bodenreibung uniform gesampelt aus `[0,3, 1,5]`, ist ein echter Boden mit Reibung `0,6` nicht länger out-of-distribution. Es ist eine ziemlich gewöhnliche Trainingsepisode. Die Policy hat *bereits* gelernt, darauf zu gehen.

Das frameworkt das Problem komplett um. Du hörst auf, dem schwer fassbaren „akkuraten Simulator" hinterherzurennen, und jagst stattdessen einer *robusten Policy*, die über einen weiten Physikbereich überlebt.

### Was zu randomisieren ist

| Parameter | Godot-Property | Typischer Bereich |
|-----------|---------------|---------------|
| Glied-Masse | `RigidBody3D.mass` | ×0,7 bis ×1,3 |
| Gelenkreibung | `PhysicsMaterial.friction` | 0,2 bis 1,0 |
| Gelenkdämpfung | `Generic6DOFJoint3D` linear/angular damping | ×0,5 bis ×2,0 |
| Motorstärke | Skaliere angewandtes Drehmoment | ×0,8 bis ×1,2 |
| Observation-Noise-Std | `randfn(0, σ)` auf jeder Obs | 0,01 bis 0,05 |
| Action-Delay (Latenz) | Buffer letzter N Aktionen | 0 bis 3 Steps |
| Bodenreibung | Floor-`PhysicsMaterial.friction` | 0,3 bis 1,5 |
| Externe Perturbationen | Zufälliger Kraftimpuls | Alle 50–200 Steps |

Drei Faustregeln:

1. **Randomisiere breiter, als du denkst.** Ist die echte Motorstärke `1,0`, randomisiere auf `[0,7, 1,3]`, nicht `[0,95, 1,05]`. Die Kosten von „zu breit" sind langsameres Training; die Kosten von „zu eng" sind totaler Deployment-Fehlschlag.
2. **Randomisiere bei Episoden-Reset, nicht pro Step.** Der Agent sollte jede Episode als *feste Instanz* der Physik behandeln. Pro Step neu zu samplen macht die Welt nicht-markovsch und destabilisiert das Lernen.
3. **Sample aus sinnvollen Verteilungen.** Uniform ist fein für die meisten Dinge. Log-uniform für Parameter, die Größenordnungen umspannen (Steifigkeit, Dämpfung). Gauß *nur* für Sensorrauschen.

### Ein durchgearbeitetes Sizing-Beispiel

Angenommen du hast einen Vierbeiner, bei dem jedes Beinglied eine gemessene Masse von 0,5 kg hat. Die Auflösung deiner Waage ist ±10 g, also gibt es schon 2 % Messunsicherheit. Über eine Produktionscharge von Robotern beobachtest du Glied-Massen zwischen 0,46 kg und 0,54 kg (±8 % Spreizung). Temperaturänderungen verändern die effektive Trägheit leicht, Motorcontroller nullen zwischen Sitzungen neu und du kannst nicht garantieren, auf welchem von sechs identischen Robotern du deployst.

Ein sinnvoller Massenbereich ist *nicht* `[0,49, 0,51]` — das ist ein Messpräzisionsbereich. Er ist auch nicht `[0,46, 0,54]` — das ist die beobachtete Einheit-zu-Einheit-Spreizung. Er ist `[0,35, 0,65]`: etwa ±30 % um den Nennwert. Die extra Margin absorbiert unbekannte Unbekannte (eine Nutzlast, die du nicht vorhergesehen hast, ein Akkutausch, angesammelter Staub). Die Trainingskosten dieser extra Margin sind moderat; die Kosten, sie *nicht* zu haben, sind eine Policy, die auf Einheit #7 versagt.

---

## 3 · Domain Randomization in Godot implementieren

Hier ein vollständiges Muster. Das Umgebungs-Root-Skript hält die Randomisierungslogik und stellt das aktuelle Sample `get_obs` / `set_action` des Agenten zur Verfügung.

```gdscript
# In the environment root script — randomize at every episode reset
extends Node3D

@onready var robot_body  = $Robot/Body
@onready var leg_joints  = $Robot.find_children("*Joint*", "Generic6DOFJoint3D")
@onready var floor_mat   = preload("res://materials/floor.tres")

# Domain randomization ranges
const MASS_RANGE      = Vector2(0.7, 1.3)     # multiplier on base mass
const FRICTION_RANGE  = Vector2(0.3, 1.2)
const DAMPING_RANGE   = Vector2(0.5, 2.0)
const FORCE_RANGE     = Vector2(0.8, 1.2)     # motor strength multiplier
const NOISE_STD_RANGE = Vector2(0.005, 0.03)  # observation noise σ

var obs_noise_std: float = 0.01
var motor_strength: float = 1.0

func randomize_domain():
    # Randomize body mass
    robot_body.mass = base_mass * randf_range(MASS_RANGE.x, MASS_RANGE.y)

    # Randomize joint damping
    for joint in leg_joints:
        var damping = base_damping * randf_range(DAMPING_RANGE.x, DAMPING_RANGE.y)
        joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_DAMPING, damping)

    # Randomize floor friction
    var new_mat = floor_mat.duplicate()
    new_mat.friction = randf_range(FRICTION_RANGE.x, FRICTION_RANGE.y)
    $Floor.physics_material_override = new_mat

    # Store for use in get_obs() and set_action()
    obs_noise_std  = randf_range(NOISE_STD_RANGE.x, NOISE_STD_RANGE.y)
    motor_strength = randf_range(FORCE_RANGE.x, FORCE_RANGE.y)

func reset():
    randomize_domain()
    # ... rest of reset
```

Und auf der Agenten-Seite:

```gdscript
# In ai_controller.gd — use the randomized noise
func get_obs() -> Dictionary:
    var obs = []
    for value in raw_observations():
        obs.append(value + randfn(0.0, env.obs_noise_std))
    return {"obs": obs}

func set_action(action) -> void:
    for i in range(joints.size()):
        var torque = action["joints"][i] * MAX_TORQUE * env.motor_strength
        apply_joint_torque(i, torque)
```

### Implementierungs-Notizen

- **Das Material `duplicate()`-en.** Editierst du `floor_mat.friction` direkt, mutierst du die geteilte Ressource — alle parallelen Umgebungen sähen dieselbe Reibung. `duplicate()` gibt jeder Episode ihre eigene Kopie. Das ist dieselbe Immutabilitäts-Lektion, die du beim Python-Schreiben gelernt hast: mutiere nie geteilten Zustand.
- **Speichere Basiswerte einmal.** Cache `base_mass`, `base_damping` in `_ready()`. Sonst kumulierst du Randomisierung über Resets und die Parameterverteilung driftet.
- **Stelle das Sample dem Agenten zur Verfügung.** Das Agenten-Skript braucht `env.obs_noise_std` und `env.motor_strength`. Übergib den Umgebungs-Node via `@export var env: Node3D` oder hol ihn vom Parent.
- **Logge das Sample zum Debuggen.** Geht ein Trainingslauf schief, ist die erste Frage „welche DR-Samples hatten die schlechtesten Episoden?". Push `obs_noise_std`, `motor_strength` und Reibung ins Info-Dict, damit sie in TensorBoard / den Replay-Buffer-Metadaten erscheinen.
- **Reseed `RandomNumberGenerator` pro Umgebung.** Läufst du N parallele Godot-Umgebungen, kann der Default-RNG-Zustand sie korrelieren. Konstruiere einen frischen `RandomNumberGenerator` pro Env und seede ihn vom Env-Index oder von `Time.get_unix_time_from_system()`. Sonst samplen alle parallelen Envs bei jedem Reset dieselbe Reibung und deine „Vielfalt" ist falsch.

---

## 4 · Observation-Rauschen als minimale brauchbare DR

Wenn du sonst nichts aus dieser Unit machst, mach das: **füge immer Observation-Rauschen hinzu.**

```gdscript
func get_obs() -> Dictionary:
    var obs = []
    for value in raw_observations():
        obs.append(value + randfn(0.0, env.obs_noise_std))
    return {"obs": obs}
```

Drei Gründe:

1. **Es ist gratis.** Eine Zeile GDScript. Keine Physikänderungen. Keine neue Trainingsinfrastruktur.
2. **Es erzwingt Robustheit.** Die Policy muss Features lernen, die kleine Sensorstörungen überleben. Spröde Features (z. B. „Gelenkwinkel ist genau 1,4523 rad") werden beim Training rausbestraft.
3. **Es ist Voraussetzung, kein Ersatz.** Versagt deine Policy mit `σ=0,02` Rauschen in Sim, wird sie *sicher* auf echter Hardware versagen, weil echtes Sensorrauschen typischerweise größer ist.

Die Umkehrung gilt nicht — den Rauschtest in Sim zu bestehen garantiert keinen Realweltserfolg — aber ihn nicht zu bestehen garantiert Fehlschlag. Nutze es als billigen Filter.

### Wie viel Rauschen?

Eine vernünftige Starter-Heuristik ist, `σ` etwa auf die Größenordnung des erwarteten Sensorrauschens auf echter Hardware zu setzen, ausgedrückt in denselben Einheiten wie die Observation:

- Gelenk-Encoder (14-Bit absolut): Rauschboden ~`0,001` rad. Trainiere mit `σ = 0,005` für Margin.
- IMU-Orientierung (Consumer-Grade MEMS): Drift + Rauschen ~`0,02` rad. Trainiere mit `σ = 0,03`.
- IMU-Gyro: Rauschen ~`0,01` rad/s. Trainiere mit `σ = 0,02`.
- Motorstromsensorik: ~5 % der Vollskala. Wende *multiplikatives* Rauschen von σ=0,05 an, nicht additives.

Dann randomisiere `σ` selbst pro Episode innerhalb `[0,5×, 2×]` dieser Werte. Eine Policy, die nur einen Rauschpegel je gesehen hat, kann gegen ihn spröde werden; das Rauschen-*Level* zu randomisieren erzwingt Robustheit gegen die Rausch-*Charakteristiken*, nicht nur gegen die Rausch-Magnitude.

### Wo Rauschen einspeisen

Rausche die Observation, *wie der Agent sie sieht*, nicht den zugrundeliegenden Zustand. Der wahre Zustand des Simulators sollte sauber bleiben — nur der Observation-Kanal wird korrumpiert. Das zählt, weil:

1. Reward-Berechnung sollte den *wahren* Zustand nutzen, nicht die verrauschte Observation. Sonst wird der Reward zum beweglichen Ziel.
2. Terminationsbedingungen („Roboter ist umgefallen") sollten den wahren Zustand nutzen. Ein verrauschter IMU sollte nicht fälschlich Episodenende auslösen.
3. Der privilegierte Critic (Abschnitt 6) braucht den sauberen Zustand explizit.

!!! warning "Niemals deployen ohne DR-Extreme zu testen"
    Bevor du eine Policy auf Hardware flashst, lasse sie in Sim mit **jedem randomisierten Parameter auf seinem Extremwert** laufen. Max Reibung *und* min Motorstärke *und* max Rauschen *und* max Latenz, alles gleichzeitig. Kollabiert dort die Episodenbelohnung, hast du keine deploybare Policy — du hast eine Policy, die im bequemen Mittelfeld deiner Verteilung lebt. Echte Hardware wird dir das Mittelfeld *nicht* schenken.

---

## 5 · Action-Delay / Latenzsimulation

Echte Motoren reagieren nicht instantan. Es gibt eine Verzögerung zwischen „Befehl gesendet" und „Drehmoment produziert":

- USB-basierte Hobby-Servos: 20–50 ms
- CAN-Bus-Vierbeiner-Motoren (z. B. T-Motor, MIT Cheetah): 5–15 ms
- ROS2-Regelschleifen mit Networking: 50–200 ms
- Hydraulische Aktuatoren (Boston-Dynamics-Klasse): 10–40 ms

Bei 60-Hz-Physik-Step (16,7 ms pro Step) sind 50 ms Latenz **3 Steps Verzögerung** — und eine Policy, die nie Verzögerung gesehen hat, gibt korrigierende Aktionen aus, *bevor* ihre vorherigen Korrekturen überhaupt Wirkung gezeigt haben. Resultat: Oszillation und Instabilität.

Latenz zu simulieren ist trivial:

```gdscript
# Simulate actuator latency by buffering actions
var action_buffer: Array = []
const LATENCY_STEPS = 2  # randomize between 1-3 during DR

func set_action(action) -> void:
    action_buffer.append(action)
    if action_buffer.size() > LATENCY_STEPS:
        var delayed_action = action_buffer.pop_front()
        _apply_torques(delayed_action)
    # If buffer not full yet, apply zero torque (startup behavior)
```

Ein paar subtile Punkte zu dieser Implementierung:

- **Startup-Verhalten zählt.** Während der ersten `LATENCY_STEPS` Steps einer Episode ist der Buffer noch nicht voll. Null-Drehmoment anzuwenden ist eine Option; den Anfangsbefehl zu halten eine andere. Echte Systeme halten meist den letzten kommandierten Wert, also bevorzuge das für Treue.
- **Die Observation des Agenten ist von „jetzt", aber die angewandte Aktion, die er sieht, ist von „damals".** Trainierst du ohne Latenz, lernt der Agent synchron zu reagieren. Mit Latenz muss der Agent lernen zu *vorherzusagen* — Befehle auszugeben, die die Verzögerung zwischen Befehl und Effekt berücksichtigen. Das ist wirklich schwerer und braucht mehr Samples, aber das ist, was echte Hardware verlangt.
- **Wende Latenz nicht auf das Reward-Signal an.** Reward wird auf dem wahren Zustand des Simulators beim aktuellen Step berechnet. Nur die *Aktionsanwendung* ist verzögert.

Für volle DR `LATENCY_STEPS` pro Episode randomisieren:

```gdscript
var latency_steps: int = 0

func randomize_domain():
    # ... other randomizations ...
    latency_steps = randi_range(0, 3)
```

Das ist eine der wirkungsvollsten DR-Techniken und eine der am seltensten implementierten. Die meisten publizierten Sim-to-Real-Fehlschläge involvieren eine Policy, die gegen null Latenz trainiert wurde und Echt-Motor-Lag nicht handhaben konnte. Hinzufügen kostet eine Handvoll Zeilen und verbessert Transfer dramatisch.

---

## 6 · Asymmetrischer Actor-Critic

Eine Technik, von OpenAIs Dactyl-Team Pionier-eingesetzt und intensiv von ETH Zürichs ANYmal-Gruppe genutzt:

- Der **Actor** (auf dem echten Roboter deployt) sieht nur Observations, die auf echter Hardware verfügbar sind: Gelenk-Encoder, IMU, Motorströme, vielleicht eine Kamera.
- Der **Critic** (existiert nur beim Training) sieht *privilegierten* Simulationszustand: exakte Objektpositionen, wahre Physikparameter, Ground-Truth-Kontaktkräfte, das vollständige Domain-Randomization-Sample.

Beim Training produziert der Critic *viel bessere* Value-Schätzungen, weil er „schummeln" kann — er sieht den vollen Simulator-Zustand. Bessere Value-Schätzungen bedeuten bessere Policy-Gradienten (via Advantage), was bedeutet, dass der Actor schneller und auf höhere Performance lernt.

Beim Deployment wirfst du den Critic weg. Der Actor wurde gegen hochwertige Gradienten trainiert, brauchte aber immer nur das begrenzte Observation-Set.

### In einfachem Deutsch

> Trainiere einen Coach (Critic), der alles sehen kann — Windgeschwindigkeit, das Playbook des gegnerischen Teams, die exakte Muskelermüdung deines Läufers. Trainiere den Spieler (Actor), der nur sieht, was er in einem echten Spiel sehen würde. Der Coach gibt dem Spieler beim Training besseres Feedback. Am Spieltag läuft der Spieler allein.

### Skizzen-Implementierung

```python
# In Python training — custom policy with asymmetric inputs
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn

class AsymmetricPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override critic to accept larger observation
        privileged_obs_dim = 64  # sim state only available in training
        self.mlp_extractor.value_net = nn.Sequential(
            nn.Linear(self.features_dim + privileged_obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )

    def evaluate_actions(self, obs, privileged_obs, actions):
        # Concatenate privileged info to critic input
        features = self.extract_features(obs)
        critic_input = torch.cat([features, privileged_obs], dim=-1)
        values = self.value_net(critic_input)
        # ... rest of evaluation
```

Auf der Godot-Seite exportierst du zwei Observation-Dicts vom Agenten: `obs` (echt-Hardware-Äquivalent) und `privileged_obs` (Sim-Ground-Truth). Die Trainings-Schleife liest beide; die deployte ONNX-Policy liest nur die erste.

!!! tip "Asymmetrischer Actor-Critic ist ein kostenloses Upgrade"
    Stellt deine Umgebung nützliche privilegierte Info bereit (wahre Objektpose, exakte Kontaktkräfte, volles DR-Sample), kostet das Hinzufügen zum Critic fast nichts und verbessert oft Trainingsstabilität *und* Endperformance. Die deployte Policy ist unverändert — selbe Eingabedimensionen, selbes Netz — aber sie wurde besser trainiert.

---

## 7 · Real-World-Deployment-Checkliste

Drucke das aus. Klebe es neben den Notaus. Gehe es Zeile für Zeile *vor* dem ersten Hardware-Test durch.

- [ ] Domain Randomization deckt den Parameterbereich der echten Hardware ab (physikalische Eigenschaften messen — Glieder wiegen, Reibung messen, Aktuatoren timen)
- [ ] Observation-Rauschen auf allen Sensoren in Sim hinzugefügt
- [ ] Action-Delay passt zur echten Motorlatenz (mit Oszilloskop oder Zeitstempel-Logs messen)
- [ ] Gelenklimit-Verletzungen terminieren Episoden in Sim (Hardware schützen — eine Policy, die in Sim lernt, in Gelenkanschläge zu rammen, zerstört Getriebe in der Realität)
- [ ] Energieeffizienz-Belohnung verhindert Überhitzungsverhalten (`sum(|torque|)` oder `sum(torque²)` bestrafen)
- [ ] Policy an Sim-DR-Extremen getestet (min Masse, max Reibung, max Rauschen) — versagt sie hier, wird sie auf Hardware versagen
- [ ] ONNX-Export getestet: Inferenzlatenz pro Step ist weit unter dem Physik-Timestep (Ziel ≤ 5 ms für eine 60-Hz-Schleife)
- [ ] Fallback-Verhalten definiert: was passiert, wenn die Policy NaN oder Out-of-Range-Werte ausgibt? Clamp + Log + (nach N Verletzungen) Strom abschalten
- [ ] Smooth-Startup-Verhalten: erste paar Sekunden Deployment nutzen gedämpfte Befehle, nicht volle Policy-Ausgabe
- [ ] Notaus unabhängig von der Policy verifiziert — der Notaus-Schalter trennt Motorstrom direkt, nicht über Software

### Eine Anmerkung zur Messung

Bevor du randomisierst, *miss*. Der häufigste DR-Fehlschlag ist „Ich habe Reibung auf `[0,3, 1,5]` randomisiert, aber der echte Boden war 1,8." Nimm eine Zugwaage, eine Stoppuhr und ein Multimeter ins Labor:

- Wiege jedes Glied des Roboters. Setze `MASS_RANGE` auf `[0,7 × gemessen, 1,3 × gemessen]`.
- Ziehe den Fuß des Roboters mit bekannter Normalkraft und einer Zugwaage über den Boden. Berechne den Reibungskoeffizienten. Dann den Bereich um ±30 % erweitern.
- Sende einen Step-Drehmomentbefehl und zeichne den Encoder auf. Die 10–90-%-Anstiegszeit ist deine Latenz.
- Lies das Motordatenblatt für `kt` (Drehmomentkonstante). Vergleiche mit kommandiertem Drehmoment unter Last. Das Mismatch ist dein Motorstärkefaktor — randomisiere drumherum.

Messen ist unglamourös und unersetzlich.

### Der erste Hardware-Test: weniger machen, genau beobachten

Wenn du endlich eine Policy auf echter Hardware platzierst:

1. **Hänge den Roboter an.** Hänge einen Beinroboter an einem Gantry, oder befestige einen Manipulator hinter einem Sicherheitskäfig. Fang den ersten Sturz statt ihn zu reparieren.
2. **Laufe mit reduziertem Action-Gain.** Multipliziere Policy-Ausgaben für die erste Sitzung mit 0,3. Verifiziere, dass das qualitative Verhalten zur Sim passt, bevor du volles Drehmoment freigibst.
3. **Zeichne alles auf.** Gelenkpositionen, kommandierte Drehmomente, IMU, Motorströme und ein Video. Der erste Hardware-Lauf ist deine Referenz für jede zukünftige Debugging-Sitzung.
4. **Vergleiche Verteilungen, nicht einzelne Trajektorien.** Laufe 20 Episoden in Sim mit denselben Anfangsbedingungen und DR-Samples, die deine Trainingsverteilung umspannen. Laufe 20 Episoden auf Hardware. Plotte die Verteilung von Step-Rewards, Kontaktmustern und Drehmomentprofilen. Liegt die Hardware-Verteilung *außerhalb* der Sim-Verteilung, ist dein DR-Bereich zu eng — geh zurück und verbreitere.

---

## 8 · Berühmte Sim-to-Real-Erfolge und Lektionen

### OpenAI Dexterous Hand / Dactyl (2019)

- **Problem.** Eine 24-DOF-Shadow-Hand, die den Rubik-Würfel löst — eine der schwersten kontaktreichen Manipulationsaufgaben, die je in der Robotik versucht wurde.
- **Lösung.** Massive Domain Randomization über tausend-und-mehr Parameter (Objektmasse, Reibung, Sehnensteifigkeit, Motorlatenz, Observation-Rauschen, Beleuchtungs- und Kameraparameter für die Vision-Pipeline), kombiniert mit asymmetrischem Actor-Critic und einer LSTM-basierten Policy, die die aktuelle Dynamik aus einer kurzen Historie *implizit identifizieren* konnte.
- **Resultat.** In Sim für das Äquivalent von ~13 000 Jahren Erfahrung trainierte Policy übertrug direkt auf die echte Hand und löste Würfel, die mitten im Lösen physisch gestört wurden.
- **Lektion.** Im Maßstab zählt **DR-Bereich mehr als Sim-Genauigkeit.** Sie hatten keinen perfekten Simulator. Sie hatten einen Simulator, der so breite Trainingsverteilungen produzierte, dass die Realität bequem drin lebte.

### ETH Zürich ANYmal (2019)

- **Problem.** Vierbeinige Lokomotion über unstrukturiertes Naturgelände — Treppen, Felsen, Schlamm, Sand.
- **Lösung.** DR über Geländehöhenkarten, Motorstärke, Nutzlast und externe Stöße. Privilegierter Critic, der Ground-Truth-Höhenkarten und Kontaktkräfte erhielt; der deployte Actor sah nur Propriozeption (Gelenkwinkel, IMU). Eine Teacher-Student-Distillations-Pipeline ließ den Actor Wissen über privilegierte Informationen in beobachtbare Features „komprimieren".
- **Resultat.** Der echte Roboter ging Treppen rauf und über Hindernisse, die er nie im Training gesehen hatte.
- **Lektion.** **Geländerandomisierung ist kritisch für Lokomotion.** Trainierst du auf einer ebenen Fläche, bekommst du einen Ebenen-Gang. Variiere die Mesh-Höhenkarte jede Episode (Perlin-Noise, gestreute Boxen, randomisierte Stufenhöhen) und die Policy generalisiert.

### Berkeley Agility Cassie (2022)

- **Problem.** Bipedales Rennen und Springen auf variiertem Gelände — ein viel schwierigeres Balanceproblem als vierbeinige Lokomotion.
- **Lösung.** Vergleichsweise *minimale* DR. Das Team investierte stark in Simulator-Genauigkeit: präzise Motormodelle, sorgfältige Identifikation physikalischer Parameter, handgetunetes Reward-Shaping.
- **Resultat.** Cassie lief einen 5K auf echter Hardware mit einer Sim-trainierten Policy.
- **Lektion.** **DR vs. Sim-Genauigkeit ist ein Tradeoff.** Hat das System enge Stabilitätsmargin (bipedale Balance), kann eine genauere Sim mit engerer DR eine breitere DR mit schlampiger Sim übertreffen. Nutze DR — aber lass sie nicht zur Ausrede werden, deine Physik nie zu verbessern.

### Boston Dynamics Spot (kommerziell, 2020+)

- **Problem.** Industrietauglicher Vierbeiner, der über Tausende deployter Einheiten verlässlich sein muss.
- **Lösung.** Öffentliche Details sind begrenzt, aber Konsens ist ein Hybrid: klassische Model-Predictive-Control für den Kern-Gang, mit lernten Policies darüber für höhere Verhaltensebenen. Sim-to-Real ist ein Werkzeug in einer größeren Toolbox.
- **Lektion.** **Produktionsrobotik ist selten reines End-to-End-RL.** Sim-to-Real ist das richtige Werkzeug für manche Teilprobleme und das falsche für andere. Wisse, welches welches ist.

---

## 9 · Progressive Domain Randomization (Curriculum)

Breite DR ist aus dem Kalten schwer zu lernen. Eine Policy, die nie gegangen ist, kann nicht gleichzeitig Lokomotion lernen *und* über Reibung in `[0,3, 1,5]` generalisieren.

Lösung: eng beginnen, schrittweise erweitern, wie sich die Policy verbessert. Das ist nur Curriculum Learning auf Physikparameter angewendet.

```python
# Schedule DR range to grow with ep_rew_mean
def update_dr_range(env, ep_rew_mean, target_reward=800.0):
    base_noise = 0.005
    max_noise  = 0.05
    # Increase randomization as agent improves
    progress = min(ep_rew_mean / target_reward, 1.0)
    env.set_noise_std(base_noise + progress * (max_noise - base_noise))
```

Ein allgemeineres Muster:

```python
def dr_schedule(progress: float) -> dict:
    """progress in [0, 1] — interpolates DR ranges from narrow to wide."""
    def interp(narrow, wide):
        return narrow + (wide - narrow) * progress

    return {
        "mass_range":     (interp(0.95, 0.7), interp(1.05, 1.3)),
        "friction_range": (interp(0.85, 0.3), interp(0.95, 1.5)),
        "noise_std_max":  interp(0.005, 0.05),
        "latency_max":    int(interp(0, 3)),
    }
```

Pushe diese Werte über den `set_obs`-Kanal, den `godot-rl-agents` dir bereits gibt, in die Godot-Umgebung. Das Agenten-Skript liest sie bei jedem Reset.

Zwei Warnungen zu progressiver DR:

- **Kollabiere den Bereich nicht, wenn Reward fällt.** Erweiterst du DR und der Agent kämpft, gehe *nicht* sofort wieder eng — das oszilliert nur. Halte den Bereich konstant, bis sich der Agent erholt.
- **Teste immer am Endbereich vor Erfolgsanspruch.** Eine Policy, die bei progress=0,5 hohen Reward erreicht, ist nicht deploybar. Trainiere, bis `ep_rew_mean` bei progress=1,0 hoch ist.

### Wann das Curriculum voranschreiten lassen

Zwei häufige Trigger, beide nützlich:

- **Reward-basiertes Vorankommen.** `progress` voranschreiten, wann immer `ep_rew_mean` einen Schwellwert überschreitet (z. B. 80 % des bei aktuellem DR-Level erreichbaren Maximums). Das ist robust gegen Schedule-Misstuning — das Curriculum wartet auf den Agenten, statt ihn vorwärtszuzwingen.
- **Step-basiertes Vorankommen.** `progress` linear von 0 auf 1 über die ersten N Millionen Umgebungs-Steps wachsen lassen. Einfacher, aber anfällig für schlechtes Timing: ist N zu klein, holt der Agent nie auf; ist N zu groß, verschwendest du Compute.

Ein Hybrid ist am besten: linearer Schedule mit einem *Gate*, das Vorankommen pausiert, wann immer `ep_rew_mean` unter einen kürzlichen gleitenden Durchschnitt fällt. So schreitet das Curriculum standardmäßig stetig voran, weicht aber automatisch zurück, wenn das Training stockt.

---

## 10 · Stretch Goals

Willst du diese Unit vor der nächsten weitertreiben, hier drei konkrete Übungen, die echte Sim-to-Real-Arbeit spiegeln:

- **Robustheits-Sweep auf JumperHard.** Nimm deine trainierte `JumperHard`-Policy aus früheren Units. Evaluiere sie mit *null* DR. Dann evaluiere mit schwerer DR. Miss die Episoden-Reward-Varianz bei 5 verschiedenen Physikparameter-Werten (Masse × {0,7, 0,85, 1,0, 1,15, 1,3}). Plotte die Varianz. Das ist genau die Art von Robustheitskurve, die ein echtes Robotik-Team vor dem Sign-off zum Hardware-Deployment produziert.
- **Action-Delay-Studie.** Implementiere den Action-Delay-Buffer aus Abschnitt 5. Trainiere zwei Policies — eine mit null Latenz, eine mit pro Episode randomisierten `LATENCY_STEPS = 2`. Miss (a) Lerngeschwindigkeit (Steps bis `ep_rew_mean = 500`) und (b) finale Reward-Varianz bei Evaluation an Latenzen 0, 1, 2, 3. Die Null-Latenz-Policy wird wahrscheinlich schneller trainieren und schlechter transferieren.
- **Lies das OpenAI-Dactyl-Paper.** *„Learning Dexterous In-Hand Manipulation"* (Andrychowicz et al. 2019). Der Domain-Randomization-Abschnitt (und der Anhang mit jedem randomisierten Parameter) ist die Goldstandard-Referenz. Das meiste davon ist direkt auf deine Godot-Umgebungen anwendbar.

---

## Was kommt als Nächstes

Du hast jetzt den vollen angewandten-RL-Stack für Robotik: Algorithmen (PPO, SAC), Engineering (Reward-Shaping, Debugging), fortgeschrittene Techniken (Curiosity, HER) und die Brücke zur Hardware (Sim-to-Real).

Die verbleibende Grenze ist nicht mehr technisch — sie ist *Alignment*. Sobald deine Policies den Simulator verlassen und in der echten Welt handeln, werden die Fragen:

- Wie spezifizierst du, was du wirklich willst, wenn die Reward-Funktion unvollständig ist?
- Wie hältst du die Policy mit menschlicher Intention im Einklang über Verteilungsverschiebungen hinweg, die du nicht antizipiert hast?
- Wie baust du Überwachung in die Deployment-Schleife, sodass du Fehlschläge fängst, bevor sie jemanden verletzen?

Die nächste Unit behandelt **Safe RL und Constrained MDPs** — wie du harte Constraints (Gelenkgrenzen, Geofences, regulatorische Anforderungen) während des Trainings durchsetzt, sodass die Policy Sicherheit nicht für Leistung eintauschen kann. Bis dahin: miss deinen Roboter, randomisiere breit, und deploye nie ohne die Checkliste durchzugehen.

---

[← Ziel-bedingtes RL & HER](unit-her.md) · [Kursstartseite](index.md) · [→ Safe RL / Constrained MDPs](unit-safe-rl.md)
