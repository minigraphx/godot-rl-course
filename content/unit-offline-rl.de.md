# Offline-RL — Lernen aus festen Datensätzen

[← RLHF & Preference Learning](unit-rlhf.md) · [Kursstartseite](index.md) · [→ Decision Transformer](unit-decision-transformer.md)

!!! info "Zeit"
    Lesen: ~40 min · Training: ~20 min GPU / ~1,5 Std CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    - **d3rlpy-Trainings-Loss-Kurve** — beobachte, wie `critic_loss` und `actor_loss` über 100k Schritte ohne eine einzige Umgebungsinteraktion konvergieren; eine sauber abflachende Kurve bedeutet, dass das Offline-Ziel funktioniert.
    - **Verhaltensvergleich** — lass eine Zufalls-Policy, eine BC-Policy (aus [unit-09.md](unit-09.md)), eine CQL-Policy und eine IQL-Policy auf derselben MultiLevelRobot-Aufgabe laufen; nimm die Episoden-Returns für jede auf; die Reihenfolge zeigt dir genau, wie viel Offline-RL zusätzlich zum reinen Imitation Learning gebracht hat.
    - **Visualisierung der Datensatzabdeckung** — projiziere die (Beobachtung, Aktion)-Paare deines Offline-Datensatzes mit t-SNE oder PCA in 2D; spärliche Regionen offenbaren die Lücken, die CQL nicht überbrücken kann und IQL nicht abfragen darf.

---

## 1 · Warum Offline-RL?

Online-RL ist das Arbeitstier jeder früheren Unit dieses Kurses. Bei genügend Umgebungsschritten — oft in Millionenhöhe — kann ein Agent fast jedes Verhalten entdecken. Diese Annahme versteckt klammheimlich gewaltige Kosten: die Umgebung muss billig laufen, schnell genug simulieren, um Millionen Schritte zu erreichen, und sicher genug sein, damit der Agent während des Lernens wiederholt scheitern darf.

Echte Hardware bricht alle drei Bedingungen auf einmal.

Ein Roboterarm, der gegen einen Tisch kracht, kostet Geld und Zeit zum Reset. Ein chirurgischer Assistent, der frei exploriert, könnte einen Patienten verletzen. Ein selbstfahrendes Auto kann nicht „ein paar zufällige Aktionen probieren" auf einer öffentlichen Straße. Selbst wenn die Umgebung ein Simulator ist, verschwendet ein From-Scratch-Training jeder neuen Policy bei jeder geänderten Belohnungsfunktion Monate an Rechenzeit.

**Offline-RL** (offline RL) — auch Batch-RL oder datengetriebenes RL genannt — greift das Problem von der anderen Seite an. Statt mit einer Umgebung zu interagieren, ist dir ein fester Datensatz gegeben:

```
D = { (s₀, a₀, r₀, s₀'), (s₁, a₁, r₁, s₁'), … , (sₙ, aₙ, rₙ, sₙ') }
```

Dieser Datensatz wurde von **irgendeiner** Verhaltens-Policy erhoben — einem menschlichen Spieler, einem regelbasierten Controller, einem früheren RL-Agenten oder einer Mischung aller drei. Deine Aufgabe ist es, die bestmögliche Policy aus diesem festen Erfahrungs-Batch zu extrahieren — mit **null zusätzlicher Umgebungsinteraktion**.

### Drei kanonische Anwendungsfälle

**1. Roboter lernt aus menschlichen Demonstrationen.**
Ein Mensch tele-operiert den Roboter mehrere Stunden lang. Diese Daten werden zum Trainings-Korpus. Genau das Setting hast du in [unit-09.md](unit-09.md) mit Behavioral Cloning erkundet — Offline-RL geht weiter, indem es das Reward-Signal einbezieht, um *über das hinaus zu verbessern*, was der Demonstrator getan hat.

**2. Fine-Tuning einer vortrainierten Policy.**
Du hast eine Policy in Produktion ausgeliefert (siehe [unit-10.md](unit-10.md)). Nach dem Deployment hast du Real-World-Logs mit Reward-Labels gesammelt. Offline-RL lässt dich die Policy aus diesen Logs verbessern, ohne in die Simulation zurückzukehren.

**3. Sicherheitskritische Domänen, in denen Exploration gefährlich ist.**
Medizingeräte, Stromnetze, Finanzsysteme. In diesen Settings ist die Offline-Beschränkung keine Einschränkung, sondern eine Anforderung.

### Vergleichstabelle

| Eigenschaft | Online-RL | Offline-RL | Imitation Learning (BC) |
|---|---|---|---|
| Live-Umgebung nötig | Ja | Nein | Nein |
| Nutzt Reward-Signal | Ja | Ja | Nein |
| Kann Demonstrator übertreffen | Ja | Teilweise | Nein |
| Datenanforderung | Viele Env-Schritte | Fester Datensatz | Expertendemonstrationen |
| Haupt-Fehlermodus | Sample-Ineffizienz | Distributional Shift | Sich akkumulierende Fehler |
| Kanonische Algorithmen | PPO, SAC, TD3 | CQL, IQL, TD3+BC | BC, GAIL, DAgger |
| Typischer Godot-Einstieg | `--train`-Modus | JSON-Datensatzexport | HUMAN-Control-Modus |

Die Verbindung zu Alignment ist direkt: RLHF (Reinforcement Learning from Human Feedback), die Technik hinter ChatGPT und Claude, ist ein Offline-RL-Problem. Präferenzdaten werden von Menschen gesammelt, ein Reward-Modell wird darauf trainiert, dann wird eine Sprach-Policy offline fine-getunt. Die mathematische Maschinerie aus dieser Unit ist anwendbar.

---

## 2 · Das Problem des Distributional Shift

Offline-RL klingt einfach: wende Q-Learning auf den Datensatz an und extrahiere die Greedy-Policy. Das Problem ist, dass Standard-Q-Learning auf Offline-Daten **spektakulär divergiert**.

### Warum naives Q-Learning scheitert

Q-Learning verwendet das Bellman-Backup:

```
Q(s, a) ← r + γ · max_{a'} Q(s', a')
```

Das `max` über Aktionen im nächsten Zustand ist die Quelle des Problems. Beim Online-Training liegt dieses Maximum üblicherweise nahe an dem, was der Agent tatsächlich tut, weil Policy und Daten eng gekoppelt sind. Beim Offline-Training kann die maximierende Aktion `a'` etwas sein, das der Datensatz **nie enthält**.

Q-Werte für Out-of-Distribution-Aktionen (OOD) starten bei ihrer Zufallsinitialisierung. Durch Bootstrapping kann das Bellman-Backup diese Zufallsschätzungen in die Q-Werte für In-Dataset-Aktionen propagieren und überall Q aufblähen. Die Policy wählt diese aufgeblähten OOD-Aktionen greedy aus. Da es keine Umgebung gibt, die den Fehler korrigiert, akkumuliert sich der Fehler unbegrenzt.

### Konkretes Beispiel

Dein MultiLevelRobot-Datensatz enthält viele Transitions, in denen der Roboter mit moderater Geschwindigkeit **nach links** geht und Reward 1,0 für das Verbleiben auf der Plattform erhält. Der Datensatz enthält nie die Aktion „mit maximaler Geschwindigkeit nach links", weil der menschliche Demonstrator sich immer vorsichtig bewegte.

Das Q-Netzwerk, dem jegliches Negativ-Signal für Maximalgeschwindigkeit-links fehlt, weist ihr unter Umständen Q = +5 zu. Die Greedy-Policy wählt diese Aktion. In der echten Umgebung führt Maximalgeschwindigkeit dazu, dass der Roboter über die Kante rutscht und -10 erhält. Aber das Offline-Training hat keine Umgebung, die das beobachtet — die Policy wählt weiter die OOD-Aktion, und ihr Q-Wert steigt weiter.

### Das Extrapolationsfehler-Diagramm

```
                     Dataset coverage
     ┌──────────────────────────────────────┐
     │                                      │
     │   ████████████████████████████████   │  ← transitions in D
     │   ████████████████████████████████   │
     │   ████████████████████████████████   │
     │                                      │
     │        ← OOD gap →                   │
     │                             ●        │  ← OOD action (never seen)
     │                             ↑        │
     │                       Q = +5 (wrong) │
     │                       R = -10 (real) │
     └──────────────────────────────────────┘
                      action space
```

Die OOD-Aktion wird vom Datensatz nicht bestraft, weil sie nie beobachtet wird. Das Q-Netzwerk extrapoliert optimistisch in die Lücke, und die Policy nutzt diese Extrapolation aus.

Alle praktischen Offline-RL-Algorithmen sind im Wesentlichen Mechanismen, um die **gelernte Policy innerhalb der Support-Region des Datensatzes zu halten** und so das Ausnutzen von Q-Wert-Extrapolation zu verhindern.

---

## 3 · Conservative Q-Learning (CQL)

**Conservative Q-Learning** (Kumar et al., 2020) ist der am breitesten angewandte Offline-RL-Algorithmus und die direkteste Lösung für Extrapolationsfehler.

### Kernidee

CQL fügt der Standard-Bellman-Zielfunktion einen Regularisierungsterm hinzu. Es bestraft hohe Q-Werte für im Datensatz schlecht repräsentierte Aktionen und belohnt hohe Q-Werte für Aktionen, die in den Daten vorkommen. Die Policy kann nur dann von hohen Q-Werten profitieren, wenn diese Werte durch In-Dataset-Aktionen verdient sind.

### CQL-Zielfunktion (Intuition)

```
L_CQL = L_Bellman                          (standard TD loss)
       + α · E_{s~D}[ log Σ_a exp Q(s,a) ]   (penalize Q for all actions)
       − α · E_{(s,a)~D}[ Q(s, a) ]          (reward Q for dataset actions)
```

Der erste Strafterm drückt Q über den vollen Aktionsraum nach unten. Der zweite Belohnungsterm zieht Q für tatsächlich im Datensatz enthaltene Aktionen wieder hoch. Nettoeffekt: Q-Werte für OOD-Aktionen werden komprimiert; Q-Werte für Datensatz-Aktionen bleiben erhalten. Die Policy bleibt in der Datenverteilung.

### Implementierung mit d3rlpy

Die Bibliothek installieren:

```bash
pip install d3rlpy
```

Sammle deinen Datensatz aus Godot via HUMAN-Control-Modus (Details in Abschnitt 4), dann:

```python
import d3rlpy
import numpy as np

# Load a dataset collected from Godot via HUMAN control mode
# (exported as a JSON of transitions)
dataset = d3rlpy.dataset.MDPDataset(
    observations=np.array(obs_list),
    actions=np.array(act_list),
    rewards=np.array(rew_list),
    terminals=np.array(done_list),
)

cql = d3rlpy.algos.CQLConfig(
    actor_learning_rate=1e-4,
    critic_learning_rate=3e-4,
    alpha_learning_rate=1e-4,
    batch_size=256,
).create(device="cpu")

cql.fit(
    dataset,
    n_steps=100_000,
    evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
)
cql.save_model("multilevel_cql.d3")
```

### Worauf während des Trainings achten

- `critic_loss` sollte sinken und sich stabilisieren. Wächst er monoton, ist die konservative Strafe `alpha` möglicherweise zu klein.
- `actor_loss` oszilliert und setzt sich dann. Große Spitzen deuten darauf hin, dass die Policy versucht, OOD-Aktionen auszunutzen.
- Der `environment`-Evaluator meldet nach jeder Auswertung den Episoden-Return. Ein langsamer Aufwärtstrend ist gesund; eine flache Linie nach vielen Schritten bedeutet, dass die Datensatzabdeckung für die Aufgabe nicht ausreicht.

### CQL-Hyperparameter-Hinweise

| Parameter | Default | Effekt |
|---|---|---|
| `alpha_learning_rate` | 1e-4 | Steuert, wie aggressiv CQL die konservative Strafe anpasst |
| `batch_size` | 256 | Größere Batches stabilisieren die Schätzung der konservativen Strafe |
| `actor_learning_rate` | 1e-4 | Niedriger als Online-SAC — der Actor muss konservativ bleiben |

---

## 4 · Den Offline-Datensatz aus Godot sammeln

Das Datensatzformat, das d3rlpy erwartet, ist exakt das Transitions-Format, das godot-rl-agents im HUMAN-Control-Modus erzeugt. Das ist derselbe Modus, den du für Demonstrationen in [unit-09.md](unit-09.md) verwendet hast.

### HUMAN-Control-Modus aktivieren

Setze im Trainingsskript deines Godot-Projekts den Control Mode auf `HUMAN`. Der Agent-Wrapper nimmt jede Transition auf, während der Mensch spielt. Exportiere den aufgenommenen Buffer am Ende jeder Sitzung als JSON.

```
godot-rl-agents \
    --env path/to/MultiLevelRobot.x86_64 \
    --mode human \
    --export_demo demonstrations.json \
    --n_parallel 1
```

### Konvertierung ins d3rlpy-Format

```python
import json
import numpy as np
import d3rlpy

def load_godot_demos(json_path: str) -> d3rlpy.dataset.MDPDataset:
    with open(json_path, "r") as f:
        raw = json.load(f)

    obs_list = []
    act_list = []
    rew_list = []
    done_list = []

    for episode in raw["episodes"]:
        for transition in episode["transitions"]:
            obs_list.append(transition["obs"])
            act_list.append(transition["action"])
            rew_list.append(transition["reward"])
            done_list.append(float(transition["done"]))

    return d3rlpy.dataset.MDPDataset(
        observations=np.array(obs_list, dtype=np.float32),
        actions=np.array(act_list, dtype=np.float32),
        rewards=np.array(rew_list, dtype=np.float32),
        terminals=np.array(done_list, dtype=np.float32),
    )

dataset = load_godot_demos("demonstrations.json")
print(f"Loaded {len(dataset.episodes)} episodes, "
      f"{sum(len(e) for e in dataset.episodes)} transitions")
```

### Wie viele Daten sammeln?

| Datensatzgröße | Erwartetes Ergebnis |
|---|---|
| < 200 Episoden | CQL kann divergieren — Abdeckung zu spärlich |
| 200–500 Episoden | CQL trainiert, Performance ist aber durch Abdeckung begrenzt |
| 500–1000 Episoden | Guter Startpunkt; IQL performt hier gut |
| 1000+ Episoden | CQL erreicht etwa BC oder besser; IQL nähert sich Online-SAC |

**Qualität zählt mehr als Quantität.** Ein Datensatz von 300 hochwertigen Expertendemonstrationen schlägt typischerweise 3000 Zufalls-Walk-Transitions für CQL. Der Grund ist Abdeckung: CQL kann Policies nur innerhalb der Support-Region des Datensatzes verbessern. Demonstriert der Datensatz nie das entscheidende Plattform-Überquerungsmanöver, wird ihm auch keine Menge an Zufallsdaten zusätzlich beibringen.

!!! warning "Datensatzabdeckung bestimmt die Obergrenze"
    Hast du ein Verhalten beim Aufnehmen nie demonstriert, kann CQL es nicht lernen. Untersuche vor dem Training die Abdeckung deines Datensatzes (Abschnitt 9). Fehlt eine kritische Fähigkeit, nimm gezielt weitere Demonstrationen für diese Lücke auf.

---

## 5 · Implicit Q-Learning (IQL)

**Implicit Q-Learning** (Kostrikov et al., 2021) verfolgt einen sanfteren Ansatz gegen das OOD-Problem. Statt OOD-Aktionen explizit zu bestrafen, vermeidet IQL es, sie überhaupt abzufragen.

### Kernidee

Standard-Q-Learning nimmt ein Maximum über Aktionen, um das Bellman-Ziel zu berechnen, was die Auswertung der Q-Funktion an potenziell OOD-Punkten erfordert. IQL ersetzt das durch **Expectile-Regression** auf der Value-Funktion, die nur mit In-Dataset-Aktionen berechnet werden kann.

Die Intuition: statt zu fragen „was ist die bestmögliche Aktion?", fragt IQL „was ist die beste Aktion, die in den Daten vorkommt?". Das umgeht das Extrapolationsproblem ohne explizite konservative Strafe.

### IQL-Komponenten

- **Value-Funktion V(s):** Mit Expectile-Regression trainiert. Der Expectile-Parameter `τ` steuert, wie optimistisch die Value-Schätzung ist — höheres `τ` extrahiert mehr Wert aus dem Datensatz.
- **Q-Funktion Q(s, a):** Mit Standard-TD trainiert, wobei V(s') als Ziel dient, sodass sie nie OOD-Aktionen auswertet.
- **Actor:** Mit Advantage-Weighted Regression (AWR) trainiert, wobei der BC-Loss mit `exp(A(s,a))` gewichtet wird, mit `A = Q - V`. Aktionen mit hohem Advantage werden stärker imitiert.

### d3rlpy-Code für IQL

```python
iql = d3rlpy.algos.IQLConfig(
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    expectile=0.7,
    weight_temp=3.0,
    batch_size=256,
).create(device="cpu")

iql.fit(dataset, n_steps=100_000)
iql.save_model("multilevel_iql.d3")
```

### IQL-Hyperparameter-Hinweise

| Parameter | Bedeutung | Tuning |
|---|---|---|
| `expectile` | Wie optimistisch V ist (0,5 = Mittel, 1,0 = Max) | Starte bei 0,7; erhöhe für hochwertige Datensätze |
| `weight_temp` | Temperatur für Advantage-Gewichtung | Höher = aggressivere Policy-Extraktion |

### Wann IQL CQL vorziehen

| Situation | Empfehlung |
|---|---|
| Kontinuierliche Aktionsräume (Gelenkgeschwindigkeiten, Raddrehmomente) | IQL — vermeidet Diskretisierungsartefakte |
| Große Datensätze (> 1000 Episoden) | IQL — Expectile-Regression skaliert gut |
| Datensätze gemischter Qualität (Experte + Zufall) | IQL — Advantage-Gewichtung gewichtet Expertentransitions natürlich höher |
| Kleine, hochwertige Expertendatensätze | CQL — konservative Strafe ist bei spärlicher Abdeckung effektiver |
| Diskrete Aktionsräume (Plattformer-Controls) | Beide; CQL leicht bevorzugt |

Für MultiLevelRobot mit kontinuierlicher Gelenksteuerung ist IQL typischerweise die stärkere Baseline. Lass beide laufen und vergleiche.

---

## 6 · Decision Transformer (konzeptionell)

CQL und IQL erben beide das Q-Learning-Paradigma — sie verwenden weiterhin Bellman-Backups, brauchen einen Critic, hängen davon ab, dass das Reward-Signal über den Datensatz hinweg konsistent ist. **Decision Transformer** (Chen et al., 2021) wirft die Bellman-Gleichung komplett weg.

### RL als Sequence Modeling framen

Eine Trajektorie ist eine Sequenz:

```
τ = (R̂₀, s₀, a₀, R̂₁, s₁, a₁, … , R̂ₙ, sₙ, aₙ)
```

wobei R̂ₜ der **Return-to-Go** ist — die Summe zukünftiger Rewards ab Zeitschritt t. Decision Transformer trainiert einen kausalen Transformer (GPT-artig), die nächste Aktion gegeben der Historie aus Returns-to-Go, Zuständen und Aktionen vorherzusagen.

Training ist reines überwachtes Lernen: das Modell bekommt die Ground-Truth-Sequenz aus dem Datensatz und soll jede Aktion vorhersagen. Kein Bellman-Backup, kein Bootstrapping, kein Critic.

### Inferenz: Konditionierung auf gewünschten Return

Zur Testzeit setzt du R̂₀ auf welche Performance du willst — etwa den maximalen Return, den du je im Datensatz gesehen hast. Das Modell konditioniert auf dieses Ziel und generiert Aktionen, die laut seinem gelernten Sequenzmodell diesen Return erzeugen würden. Während Rewards akkumulieren, ziehst du sie von R̂ ab, um den Return-to-Go zu aktualisieren.

```
R̂ₜ₊₁ = R̂ₜ − rₜ
```

Dem Modell wird im Grunde die Frage gestellt: „Wenn ich noch so viel Reward brauche, was sollte ich als Nächstes tun?"

### Kernerkenntnis

Decision Transformer leidet nicht unter Distributional Shift im Q-Learning-Sinn, weil es nie Werte für OOD-Aktionen schätzt. Es ist ein bedingtes generatives Modell über Aktionssequenzen. Liegt der gewünschte Return innerhalb der Support-Region der Trainingsdaten, interpoliert es. Überschreitet er die beste Trajektorie im Datensatz, extrapoliert es — was funktionieren kann oder auch nicht, je nachdem, wie gut der Transformer generalisiert.

### HuggingFace-Integration

Das `huggingface/decision-transformer`-Modell enthält eine Referenz-Implementierung, die in den Standard-HuggingFace-Trainer eingebettet ist. Für große Datensätze (zehntausende Episoden) ist Pre-Training eines Decision Transformer und Fine-Tuning auf aufgabenspezifischen Daten ein zunehmend beliebtes Muster — direkt analog zum Pre-Training eines Sprachmodells auf Web-Text und Fine-Tuning auf nachgelagerte Aufgaben.

### Wann Decision Transformer einsetzen

- Sehr große Datensätze, in denen der Sequence-Modeling-Kontext hilft (> 10k Episoden)
- Pre-Training-+-Fine-Tuning-Workflows
- Situationen, in denen du zur Inferenzzeit auf unterschiedliche Ziel-Return-Niveaus konditionieren willst, ohne neu zu trainieren
- Forschungskontexte, in denen du Reward Hacking vermeiden willst — das Modell lernt, den verlangten Return zu liefern, nicht einen schlecht spezifizierten Reward auszunutzen

Für die meisten Godot-Projekte mit 1000–5000 Demonstrationen sind CQL oder IQL einfacher und schneller zu trainieren. Decision Transformer wird attraktiv, wenn du Zugriff auf große, diverse Datensätze hast oder Offline-Pre-Training mit sprachkonditioniertem Verhalten kombinieren willst.

---

## 7 · Online-Fine-Tuning nach Offline-Pre-Training

Das praktischste Deployment-Muster ist eine zweistufige Pipeline:

**Stufe 1 — Offline-Pre-Training:** Verwende CQL oder IQL auf deinem Datensatz. Das Ergebnis ist eine Policy, die etwa so gut performt wie die Demonstratoren, mit etwas Verbesserung durch das Reward-Signal.

**Stufe 2 — Online-Fine-Tuning:** Setze das Training mit einem Online-Algorithmus fort (SAC oder PPO). Die Offline-Policy gibt einen Warmstart, der die katastrophalen frühen Explorationsausfälle vermeidet, die Online-RL-from-Scratch plagen.

Diese Kombination adressiert beide Probleme: Offline-Pre-Training liefert eine Policy sicher innerhalb der Datenverteilung; Online-Fine-Tuning greift auf die echte Umgebung zu und korrigiert den Distributional Shift zwischen Datensatz und tatsächlicher Deployment-Umgebung.

### Warum die Kombination mächtig ist

- Offline-Pre-Training kollabiert die Phase „zufällige Exploration" von Millionen Schritten auf null.
- Online-Fine-Tuning greift auf die echte Umgebung zu und korrigiert Lücken in der Datensatzabdeckung.
- Der kombinierte Ansatz übertrifft oft, was reines Offline- oder reines Online-Training erreicht.

### Code-Skizze: d3rlpy offline → SB3 online

```python
import d3rlpy
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

# ── Stage 1: offline pre-train with CQL ──────────────────────────────────────
cql = d3rlpy.algos.CQLConfig(batch_size=256).create(device="cpu")
cql.fit(dataset, n_steps=100_000)

# Extract actor weights from d3rlpy (PyTorch state dict)
offline_actor_state = cql.impl.policy.state_dict()

# ── Stage 2: online fine-tune with SAC ───────────────────────────────────────
env = make_vec_env("MultiLevelRobot-v0", n_envs=4)
model = SAC("MlpPolicy", env, verbose=1, learning_starts=1_000)

# Transfer offline actor weights into SB3 policy network
# Note: layer names must match — verify architecture parity first
model.policy.actor.load_state_dict(offline_actor_state, strict=False)

# Continue training online
model.learn(total_timesteps=500_000)
model.save("multilevel_offline_then_online")
```

!!! note "Architekturparität"
    Der Offline-Actor (d3rlpy) und der Online-Actor (SB3) müssen dieselbe Netzarchitektur teilen, damit der Gewichtstransfer funktioniert. Konfiguriere beide vor dem Training mit identischen Hidden-Layer-Größen (z. B. `[256, 256]`). Unterscheiden sich die Architekturen, kannst du die Online-Policy stattdessen mit einem BC-Warmstart initialisieren statt mit CQL-Actor-Gewichten.

### Erwartete Gewinne

In typischen Godot-Umgebungen übertrifft Offline-Pre-Training + 500k Online-Schritte 2M Online-Schritte from Scratch bei Sparse-Reward-Aufgaben. Bei Dense-Reward-Aufgaben, in denen Online-RL effizient explorieren kann, ist die Lücke kleiner.

---

## 8 · Entscheidungsleitfaden

Verwende dieses Flussdiagramm, um deine Trainingsstrategie zu wählen.

```
Do you have a fixed dataset of transitions?
│
├── No → Use online RL (PPO, SAC — see earlier units)
│
└── Yes
    │
    ├── Does the dataset contain reward labels?
    │   │
    │   ├── No → Use Behavioral Cloning (unit-09.md)
    │   │
    │   └── Yes
    │       │
    │       ├── Is the dataset from a safe, cheap-to-run environment?
    │       │   │
    │       │   └── Yes → Consider offline pre-train → online fine-tune (section 7)
    │       │
    │       ├── Is your action space continuous?
    │       │   │
    │       │   ├── Yes, large dataset (> 1000 eps) → IQL (section 5)
    │       │   └── Yes, small dataset (< 500 eps)  → CQL (section 3)
    │       │
    │       ├── Discrete action space → CQL (section 3)
    │       │
    │       └── Very large dataset, want pre-train workflow → Decision Transformer (section 6)
```

### Wann Offline-RL die richtige Wahl ist

Verwende Offline-RL, wenn:

- Du einen Datensatz hast, aber **die Umgebung nicht ausführen kannst** (echte Hardware, abgeschalteter Simulator, Produktionssystem).
- **Sicherheit Exploration verbietet** — Medizingeräte, sicherheitskritische Regelung, Finanzsysteme.
- Du **vor dem Deployment vortrainieren** willst, um die zufällige Explorationsphase zu vermeiden.
- Du **geloggte Produktionsdaten** mit Reward-Signalen hast und eine bestehende Policy verbessern willst.
- Du an **goal-conditioned** Aufgaben mit Hindsight-Relabeling arbeitest — Offline-HER kombiniert HER (siehe [unit-her.md](unit-her.md)) mit Offline-RL, um die effektive Datensatzgröße dramatisch zu erhöhen, indem Transitions mit erreichten Zielen umgelabelt werden.

Verwende Online-RL, wenn du einen schnellen, billigen Simulator hast und keine vorhandenen Daten. Verwende Behavioral Cloning, wenn du Expertendemonstrationen hast, aber kein verlässliches Reward-Signal.

---

## 9 · Viz-Checkpoint

Bevor du deine Offline-Policy als deployment-bereit erklärst, führe diesen Vierfachvergleich auf der MultiLevelRobot-Aufgabe durch.

### Aufbau

Trainiere alle vier Policies auf demselben festen Datensatz:

1. **Zufalls-Policy** — kein Training, rein zufälliges Aktionssampling
2. **BC-Policy** — Behavioral Cloning aus unit-09.md, derselbe Datensatz
3. **CQL-Policy** — Conservative Q-Learning, 100k Schritte
4. **IQL-Policy** — Implicit Q-Learning, 100k Schritte

Evaluiere jede für 50 Episoden. Notiere mittleren Episoden-Return und Erfolgsrate (Erreichen der finalen Plattform).

### Erwartete Reihenfolge

```
Random < BC ≤ CQL ≈ IQL
```

CQL und IQL sollten beide BC übertreffen, da sie das Reward-Signal einbeziehen. Ob CQL oder IQL gewinnt, hängt von deiner Datensatzgröße und deinem Aktionsraum ab. Liegt CQL unter BC, ist die konservative Strafe zu aggressiv oder der Datensatz zu spärlich — senke `alpha_learning_rate` oder sammle mehr Daten.

### Wonach im Verhalten suchen

| Policy | Charakteristisches Verhalten |
|---|---|
| Zufall | Sofortige Stürze, keine Richtungsneigung |
| BC | Imitiert Demonstrationspfad, scheitert an neuen Plattform-Konfigurationen |
| CQL | Folgt demonstrierten Pfaden, passt sich aber leicht an den Reward an; vorsichtig |
| IQL | Glatter, selbstbewusster als CQL auf bekannten Plattformen; ähnliche Vorsicht bei OOD-Layouts |

### Visualisierung der Datensatzabdeckung

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

obs = dataset.observations
acts = dataset.actions

# Concatenate obs and action for joint coverage visualization
joint = np.concatenate([obs, acts], axis=1)
pca = PCA(n_components=2)
reduced = pca.fit_transform(joint)

plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.3, s=2, label="Dataset transitions")
plt.title("Dataset coverage (PCA of obs+action)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend()
plt.savefig("dataset_coverage.png", dpi=150)
```

Spärliche Regionen in diesem Plot sind die Bereiche, in denen CQL und IQL am wahrscheinlichsten scheitern. Siehst du große leere Flächen, die kritischen Aufgabenzuständen entsprechen, ziele bei der weiteren Datensammlung gezielt auf diese Zustände.

---

## 10 · Stretch Goals

**Datenseffizienzkurve.**
Sammle 500 und 5000 Demonstrationen separat im HUMAN-Control-Modus. Trainiere CQL auf jedem Datensatz für 100k Schritte. Plotte den mittleren Episoden-Return gegen die Datensatzgröße. Die Kurve sollte abnehmende Erträge zeigen — Offline-RL skaliert nicht linear mit Daten. Identifiziere das Knie in der Kurve: den Punkt, ab dem zusätzliche Daten die Performance nicht mehr nennenswert verbessern.

**D4RL-Benchmark-Baseline.**
Installiere die D4RL-Benchmark-Umgebungen:

```bash
pip install d4rl
```

Trainiere IQL auf `hopper-medium-v2` oder `halfcheetah-medium-v2` — Standard-Offline-RL-Benchmarks mit publizierten Ergebnissen. Vergleiche deinen IQM-Score (Interquartile Mean) mit den Zahlen im IQL-Paper (Kostrikov et al., 2021). Liegen deine Zahlen innerhalb von 5 %, ist deine Implementierung korrekt.

**Offline → Online-Verbesserungsmessung.**
Nimm deine beste CQL-Policy. Fine-tune sie online mit SAC für 500k Schritte (Abschnitt 7). Miss den Episoden-Return vor und nach dem Online-Fine-Tuning. Berichte die prozentuale Verbesserung. Für MultiLevelRobot mit sparsamen Rewards erwarte 15–40 % Verbesserung durch Online-Fine-Tuning oben auf der Offline-Baseline.

**Offline-HER.**
Kombiniere Offline-RL mit Hindsight Experience Replay aus [unit-her.md](unit-her.md). Nachdem du deinen Demonstrationsdatensatz gesammelt hast, label jede Transition mit dem erreichten Ziel um, als ob es das beabsichtigte Ziel gewesen wäre. Das vervielfacht die effektive Datensatzgröße um die Anzahl der Ziele in jeder Trajektorie. Trainiere CQL auf dem umgelabelten Datensatz und vergleiche gegen CQL auf den Original-Labels.

---

[← RLHF & Preference Learning](unit-rlhf.md) · [Kursstartseite](index.md) · [→ Decision Transformer](unit-decision-transformer.md)
