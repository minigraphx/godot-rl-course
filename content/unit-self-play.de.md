# Self-Play — Gegen sich selbst trainieren

[← Memory & POMDPs](unit-08.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~30 min · Training: ~30 min GPU / ~2 Std CPU

!!! info "Drei Wege, deine KI zu beobachten"
    - **Godot-Viewer**: beobachte, wie zwei KI-Paddel, gesteuert vom selben Gehirn, AirHockey gegeneinander spielen — Strategien entstehen aus dem Nichts.
    - **TensorBoard**: verfolge das ELO-Rating während des Trainings. Belohnung allein ist im Self-Play bedeutungslos (sie mittelt sich in einem Nullsummenspiel zu null); ELO ist der Proxy für absolute Spielstärke.
    - **Checkpoint-Ordner**: ein wachsendes Archiv vergangener Versionen deiner selbst. Jede `.zip` ist ein eingefrorener Gegner, den deine aktuelle Policy zu schlagen lernen muss.

---

## 1 · Was Self-Play ist und warum es funktioniert

In Standard-Reinforcement-Learning ist die Umgebung **fest**. Ein Labyrinth hat in jeder Episode dieselben Wände. Ein Cart-Pole hat immer dieselbe Physik. Wenn deine Policy sich verbessert, lernt sie, diese feste Umgebung auszunutzen — und sobald sie das getan hat, stagniert das Lernen. Die Schwierigkeitsdecke ist alles, was der menschliche Designer in den Simulator eingebacken hat.

Self-Play entfernt die Decke.

**Die Kernidee:** der Gegner IST deine aktuelle Policy. Wenn du eine Aktion gegen einen Gegner ausführst, der eine Kopie (oder ein aktueller Snapshot) deiner selbst ist, verfolgt die Schwierigkeit des Gegners automatisch dein eigenes Spielniveau. Ein Anfänger spielt gegen einen Anfänger. Ein Profi spielt gegen einen Profi. Das Match ist immer nah an deiner Grenze — genau dort, wo Lernen am schnellsten passiert.

Das macht Self-Play von Natur aus zu einem **adaptiven Curriculum**. Du musst nie einen Schwierigkeitsplan handtunen, Gegnerstatistiken hochskalieren oder Level-Progressionen entwerfen. Der Gegner wird härter, weil du besser geworden bist, und der einzige Weg, wie du besser geworden bist, war, gegen eine etwas schwächere Version des härteren Gegners zu spielen, zu dem du gerade geworden bist.

### Das berühmte Ergebnis

Der prägende Moment für Self-Play kam von DeepMind. **Silver et al., 2017 (AlphaGo Zero)** zeigten, dass eine Policy, ausgehend von komplett zufälligem Spiel, nur gegen sich selbst antretend, ohne menschliche Spieldaten, **übermenschliche Leistung in Go** erreichen konnte — einem Spiel, das Menschen 2 500 Jahre lang studiert hatten. Der frühere AlphaGo war mit Millionen menschlicher Spiele bootstrapped worden. AlphaGo Zero hatte keine und zerstörte ihn.

Dieses Ergebnis hat neu definiert, was Reinforcement Learning sein könnte. Die Lektion: mit genug Rechenleistung, dem richtigen Algorithmus und Self-Play als Curriculum kann ein Agent Strategien entdecken, die die besten, die Menschen je produziert haben, übertreffen.

### Drei wegweisende Systeme

- **AlphaGo Zero** (DeepMind, 2017) — Go von Grund auf in 3 Tagen Self-Play. Entdeckte bekannte menschliche Eröffnungen und übertraf sie dann. Entscheidend: **sah nie ein menschliches Spiel**.
- **OpenAI Five** (OpenAI, 2018) — Dota 2 auf Profiniveau. Trainiert auf dem Äquivalent von **180 Jahren Self-Play pro Tag** über Tausende GPUs hinweg. Besiegte das Weltmeisterteam OG 2019.
- **AlphaStar** (DeepMind, 2019) — StarCraft II auf Grandmaster-Niveau. Führte **League-basiertes Self-Play** ein: statt einem Gegner ein ganzer Pool vergangener Versionen, mit „Exploiter"-Agenten, die speziell darauf trainiert wurden, Schwächen zu finden. Die erste KI, die das Top-0,2 % in einem komplexen Echtzeit-Strategiespiel erreichte.

Alle drei haben dieselbe Form: zufällige Initialisierung → Self-Play → emergente Strategie, die menschliches Design übertrifft. Nichts anderes in RL hat die gleiche „Zinseszins"-Eigenschaft.

---

## 2 · Nichtstationarität und warum Self-Play hilft

Erinnerung aus [Unit 7 (Multi-Agent)](unit-07.md): wenn mehrere Agenten gleichzeitig lernen, ist die Umgebung jedes einzelnen **nichtstationär**. Die Übergangsdynamik `P(s' | s, a)` hängt von den Policies *anderer* Agenten ab — und diese Policies ändern sich ständig. Aus Sicht jedes einzelnen Agenten verschiebt sich die Welt unter seinen Füßen. Konvergenzgarantien aus dem Single-Agent-RL gelten nicht mehr.

Self-Play zähmt das auf eine spezifische, nützliche Weise:

- Der Gegner ist eine **Kopie von dir**, aber ein **Snapshot** — er ändert sich langsam, nur wenn ein neuer Checkpoint gespeichert wird.
- Zwischen Checkpoints ist die Policy des Gegners eingefroren. Die Umgebung ist aus Sicht des lernenden Agenten **vorübergehend stationär**.
- Nach einem Checkpoint-Update springt der Gegner — aber nur zu einer geringfügig besseren Version seiner selbst.

Mit anderen Worten: Self-Play ersetzt das Chaos simultanen Lernens durch eine Folge von **stationären Intervallen, getrennt durch kleine Sprünge**. Die Trainingsverteilung ist immer ungefähr „aktuelle Spielstärke ± ein Checkpoint". Stabil genug zum Lernen, frisch genug, um weiter zu verbessern.

Kontrastiere das mit naivem Multi-Agent: wenn beide Agenten bei jedem Gradientenschritt aktualisieren, jagen sie sich möglicherweise durch den Policy-Raum und konvergieren nie. Self-Play umgeht das, indem es eine **bewusste Verzögerung** zwischen Lerner und Gegner einführt.

---

## 3 · Drei Self-Play-Trainingsmodi

Es gibt nicht den einen „Self-Play-Algorithmus". Es gibt eine Familie von Trainingsschemata, jedes mit einem anderen Trade-off zwischen **Stabilität**, **Diversität der Gegner** und **Implementierungskomplexität**.

### Einfaches Self-Play (neuester Checkpoint)

Die häufigste Variante: spiele immer gegen den **neuesten** Checkpoint deiner selbst. Nach jedem Trainingsschritt ist der Gegner im Wesentlichen dasselbe Netz wie der Lerner (oder eine vor Augenblicken erstellte Kopie).

- **Pro**: trivial zu implementieren — rufe einfach `model.predict()` auf beiden Paddeln auf.
- **Contra**: hochgradig **instabil**. Der klassische Fehlermodus ist **Policy-Oszillation**: das Netz lernt eine Strategie, die das aktuelle Selbst schlägt, der Gegner (der das aktuelle Selbst ist) aktualisiert sich, um dem zu kontern, das neue Selbst kontert das, und das Netz vergisst die ursprüngliche Strategie. Rundreisen durch den Policy-Raum ohne Nettofortschritt.

!!! warning "Die Oszillationsfalle"
    Einfaches Self-Play zeigt oft eine charakteristische Pathologie: Schere-Stein-Papier-Dynamik. Die Policy lernt zu „links stürmen", lernt dann „Linksstürme blockieren", vergisst dann den Linkssturm. ELO springt hin und her. Sieh dir deine Replay-Videos an — wenn der Spielstil von Checkpoint 5 nichts wie Checkpoint 10 aussieht, aber auch nichts wie Checkpoint 15, ist Oszillation real. Der Fix ist einer der nächsten beiden Modi.

### Self-Play mit eingefrorenem Gegner

Speichere alle N Episoden (oder alle N Gradientenschritte) einen Checkpoint. Für das nächste Trainingsfenster **frier diesen Checkpoint als Gegner ein** und spiele nur gegen ihn. Nach N weiteren Schritten speichere einen neuen Checkpoint und friere wieder ein.

- **Pro**: Gegner ändert sich nicht während eines Fensters → die Umgebung ist innerhalb jedes Fensters vollständig stationär → Standard-PPO funktioniert ohne Modifikation.
- **Contra**: die Policy könnte zu **einer Gegenstrategie konvergieren, die spezifisch für diesen einen Gegner ist**. Wenn der Gegner auf einen neuen Freeze aktualisiert wird, kann die Leistung einbrechen, während die Policy sich neu ausrichtet.

### League-basiertes Self-Play (der AlphaStar-Ansatz)

Pflege einen **Pool vergangener Checkpoints** — die „League". Während des Trainings sample einen Gegner aus dem Pool für jede Episode. Der Pool wächst mit fortschreitendem Training, optional begrenzt durch Eviction der schwächsten Mitglieder.

- **Pro**: höchste Stabilität und höchste Diversität. Die aktuelle Policy muss **alle vergangenen Versionen** schlagen, was Vergessen verhindert. Der Pool enthält natürlich Mitglieder mit unterschiedlich ausnutzbaren Schwächen, sodass der Lerner generalisieren muss.
- **Contra**: mehr Buchhaltung, mehr Speicherplatz, und du musst eine Sampling-Strategie wählen (uniform? priorisiert nach ähnlicher Spielstärke? nach Exploitern?).

### Modusvergleich

| Modus | Stabilität | Diversität | Komplexität | Am besten für |
|------|-----------|-----------|------------|---------|
| Einfach (neuester) | Niedrig | Niedrig | Trivial | Prototyping, Sanity Checks |
| Eingefrorener Checkpoint | Mittel | Niedrig | Einfach | Kleine Spiele, Domänen mit einzelner Strategie |
| League | Hoch | Hoch | Moderat | Komplexe Spiele, robuste Policies, Produktion |

!!! tip "Default zu League-basiert für alles Ernsthafte"
    Für Forschung oder Produktionstraining starte mit League-basiertem Self-Play. Die extra ~50 Zeilen Buchhaltung zahlen sich in Stabilität aus. Verwende einfaches Self-Play nur für schnelle Prototypen oder um zu bestätigen, dass deine Umgebung überhaupt funktioniert.

---

## 4 · Öffne AirHockey

Unsere Umgebung für diese Unit ist das **AirHockey**-Beispiel aus [godot_rl_agents_examples](https://github.com/edbeeching/godot_rl_agents_examples), in `examples/AirHockey`. Es ist ein sauberes, minimales Zwei-Spieler-Nullsummenspiel — perfekt für Self-Play.

```bash
git clone https://github.com/edbeeching/godot_rl_agents_examples.git
cd godot_rl_agents_examples/examples/AirHockey
```

### Das Spiel

- Zwei Paddel (Agenten), ein Puck, ein Tor an jedem Ende des Tisches.
- Jeder Agent beobachtet seinen eigenen Zustand, den Puck-Zustand und den Zustand des Gegners.
- Die Aktion jedes Agenten ist ein 2D-Kontinuierlicher-Kraftvektor, der auf sein Paddel angewendet wird.

### Die Belohnung — reines Nullsummenspiel

Belohnung ist **Nullsumme**: wenn Agent 0 trifft, bekommt Agent 0 `+1` und Agent 1 bekommt `-1`. Wenn Agent 1 trifft, kehren sich die Vorzeichen um. Die Summe der Belohnungen über die beiden Agenten ist immer null (bis auf eventuelles kleines Shaping).

```text
r_agent_0 = -1 × r_agent_1
```

Das ist genau das Setting, für das Self-Play gebaut ist. Es gibt keinen Begriff von „absoluter" Belohnung, die sich verbessert — die durchschnittliche Rendite für jeden Agenten über einen langen Trainingslauf ist ungefähr null. Was sich ändert, ist *Spielstärke*, gemessen daran, wer gewinnt, was du aus der Belohnung allein nicht ablesen kannst. (Deshalb führt Abschnitt 6 ELO ein.)

### Lies den Controller

Öffne `ai_controller.gd` in der AirHockey-Szene. Die `get_obs()`-Funktion gibt ungefähr zurück:

- eigene Paddelposition `(x, y)`
- eigene Paddelgeschwindigkeit `(vx, vy)`
- Gegnerpaddelposition `(x, y)`
- Gegnerpaddelgeschwindigkeit `(vx, vy)`
- Puckposition `(x, y)`
- Puckgeschwindigkeit `(vx, vy)`

Das ist der vollständige Zustand aus Sicht des Agenten. Beide Agenten teilen dieselbe Beobachtungs*form*, sehen die Welt aber von gegenüberliegenden Seiten — per Konvention spiegelt das Beispiel das Koordinatensystem, sodass beide Agenten erleben „mein Tor ist hinter mir, das Tor des Gegners ist vor mir".

### Zwei Binaries vs ein Binary mit zwei Agenten

Du hast zwei Optionen:

- **Ein Binary, zwei Agenten innen**: ein einziger Godot-Prozess exponiert zwei `AIController`-Knoten. Die Python-Seite empfängt zwei Beobachtungs-Batches pro Schritt und sendet zwei Aktions-Batches. Das werden wir verwenden — das ist, was das Beispiel mitbringt.
- **Zwei Binaries**: spawn zwei separate Godot-Prozesse, jeder kontrolliert ein Paddel, kommunizierend über einen geteilten Puck-Server. Einfacher nachvollziehbar, aber schwerer und für Self-Play selten benötigt.

Exportiere das Binary wie in Unit 4:

```bash
# inside the Godot editor, Project → Export → Linux/macOS/Windows
# output: AirHockey.x86_64 (or .app / .exe)
```

---

## 5 · Self-Play in Python implementieren (SB3)

Wir verwenden stable-baselines3 mit dem `StableBaselinesGodotEnv`-Wrapper aus `godot-rl`. Zwei Muster folgen: ein symmetrisches Single-Policy-Setup (am einfachsten) und ein Setup mit eingefrorenem Gegner (stabiler).

### Ansatz A — Single Policy, zwei Instanzen (symmetrisches Self-Play)

Das einfachste Muster. Ein PPO-Modell kontrolliert *beide* Paddel. Das Modell sieht beide Beobachtungen pro Schritt, gibt beide Aktionen aus und bekommt beide Belohnungen. Da das Spiel symmetrisch ist und die Policy geteilt wird, trainiert jeder Gradientenschritt dasselbe Netz von beiden Seiten des Tisches.

```python
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import os, copy

env = StableBaselinesGodotEnv(
    env_path="./AirHockey.x86_64",
    n_parallel=8,
    speedup=20,
)

# Single PPO model — controls BOTH paddles with same policy
model = PPO(
    "MlpPolicy", env,
    verbose=1,
    tensorboard_log="logs/",
    n_steps=512,
    batch_size=256,
)

checkpoint_interval = 50_000   # save opponent checkpoint every 50k steps
checkpoint_dir = "self_play_checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)

# Simple self-play: save checkpoints, use latest as opponent
for iteration in range(20):
    model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
    checkpoint_path = f"{checkpoint_dir}checkpoint_{iteration}.zip"
    model.save(checkpoint_path)
    print(f"Iteration {iteration}: saved {checkpoint_path}")

model.save("airhockey_selfplay_final")
env.close()
```

Das ist „einfaches Self-Play" aus Abschnitt 3: der Gegner ist immer die aktuelle Policy (weil er buchstäblich die aktuelle Policy ist — dieselben Gewichte). Du speicherst Checkpoints zur Archivierung und Evaluation, nicht zur Verwendung während des Trainings.

Es funktioniert für AirHockey, weil das Spiel kurz ist, der Aktionsraum klein ist und Oszillation sich langsam manifestiert. Für reichere Spiele wäre es instabil.

### Ansatz B — Eingefrorener Gegner (stabiler)

Um einen stationären Gegner zu bekommen, lade einen **vergangenen** Checkpoint als Gegner-Policy. Das Paddel des Lerners verwendet das aktuelle Modell; das Paddel des Gegners verwendet ein eingefrorenes Modell, das von Festplatte geladen wird.

```python
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import os, numpy as np

env = StableBaselinesGodotEnv(
    env_path="./AirHockey.x86_64",
    n_parallel=8,
    speedup=20,
)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/",
            n_steps=512, batch_size=256)

# Bootstrap: do a few iterations of simple self-play to seed a checkpoint
model.learn(total_timesteps=50_000, reset_num_timesteps=False)
model.save("self_play_checkpoints/checkpoint_0.zip")

opponent_model = PPO.load("self_play_checkpoints/checkpoint_0.zip")

def frozen_opponent_step(obs_batch):
    """Split obs into (learner, opponent); learner acts via `model`,
       opponent acts via `opponent_model`. Re-combine actions."""
    # obs_batch shape: (2 * n_parallel, obs_dim) — interleaved agent_0, agent_1
    learner_obs   = obs_batch[0::2]
    opponent_obs  = obs_batch[1::2]
    learner_act, _   = model.predict(learner_obs,   deterministic=False)
    opponent_act, _  = opponent_model.predict(opponent_obs, deterministic=True)
    combined = np.empty((learner_act.shape[0] * 2, learner_act.shape[1]),
                        dtype=learner_act.dtype)
    combined[0::2] = learner_act
    combined[1::2] = opponent_act
    return combined

# In a real loop you'd subclass the env or callback to inject this;
# for clarity we sketch the outer training schedule:
for iteration in range(1, 20):
    # Train against the *currently frozen* opponent
    model.learn(total_timesteps=50_000, reset_num_timesteps=False)
    ckpt = f"self_play_checkpoints/checkpoint_{iteration}.zip"
    model.save(ckpt)
    # Promote the new checkpoint to opponent — the lag is intentional
    opponent_model = PPO.load(ckpt)
    print(f"Iteration {iteration}: opponent promoted to {ckpt}")

env.close()
```

Der Trick: die Policy des Gegners ist **für das gesamte 50k-Schritt-Fenster eingefroren**. Während dieses Fensters ist die Umgebung stationär; PPO hat ein wohldefiniertes Ziel zu optimieren. Am Ende des Fensters wird der Gegner upgegradet — ein diskreter, vorhersagbarer Sprung.

In der Praxis würdest du `frozen_opponent_step` entweder durch Wrapping der Umgebung implementieren (sodass der Wrapper Gegneraktionen sendet, bevor er Beobachtungen an PPO weiterleitet) oder indem du nur den Lerner-Agenten an PPO exponierst und die Godot-Umgebung manuell schrittweise ausführst. Beide Muster sind in Ordnung; der Wrapper-Ansatz ist sauberer.

---

## 6 · ELO-Rating für Self-Play-Fortschritt

Hier ist das Problem, das du fünf Minuten nach dem Start deines ersten Self-Play-Laufs treffen wirst: **wie weißt du, dass es sich verbessert?** In einem Nullsummenspiel mit einem ausgewogenen Gegner ist deine mittlere Belohnung null — per Definition. Du kannst nicht einfach `ep_rew_mean` plotten und beim Steigen zuschauen.

Die Standardantwort, ausgeliehen aus kompetitiven Spielen: **ELO-Rating**.

### ELO in 60 Sekunden

- Jeder Spieler startet mit einem Basisrating (üblicherweise **1000**).
- Nach jedem Match aktualisieren sich die Ratings basierend auf dem Ergebnis **und** dem erwarteten Ergebnis.
- Jemanden zu schlagen, der viel höher eingestuft ist als du, bringt viel Rating. Jemanden viel Schwächeren zu schlagen, bringt sehr wenig. Gegen jemand Schwächeres zu verlieren, kostet viel.
- Die erwartete Gewinnwahrscheinlichkeit wird durch die Ratinglücke bestimmt, mittels der logistischen Kurve:

```text
expected_win = 1 / (1 + 10 ^ ((opponent_elo - your_elo) / 400))
```

Eine 400-Punkte-Lücke bedeutet, dass der höher eingestufte Spieler ~91 % der Zeit gewinnt.

### Implementierung

```python
def update_elo(winner_elo, loser_elo, k=32):
    """Standard ELO update. Returns (new_winner_elo, new_loser_elo)."""
    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    new_winner = winner_elo + k * (1 - expected_winner)
    new_loser  = loser_elo  + k * (0 - (1 - expected_winner))
    return new_winner, new_loser

# After each episode:
# if agent_0 won: agent_0_elo, agent_1_elo = update_elo(agent_0_elo, agent_1_elo)
# else:           agent_1_elo, agent_0_elo = update_elo(agent_1_elo, agent_0_elo)
```

Der `k`-Faktor steuert, wie schnell sich Ratings ändern. `k=32` ist Standard für Schach. Für Self-Play mit häufigen Matches reduziert `k=16` oder sogar `k=8` das Rauschen.

### Loggen nach TensorBoard

```python
from stable_baselines3.common.callbacks import BaseCallback

class EloLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.agent_elo = 1000.0
        self.opponent_elo = 1000.0

    def _on_step(self) -> bool:
        # Detect end-of-episode and the winner from info dicts
        for info in self.locals.get("infos", []):
            if "winner" in info:  # set by your env at episode end
                if info["winner"] == 0:
                    self.agent_elo, self.opponent_elo = update_elo(
                        self.agent_elo, self.opponent_elo)
                else:
                    self.opponent_elo, self.agent_elo = update_elo(
                        self.opponent_elo, self.agent_elo)
                self.logger.record("self_play/agent_elo", self.agent_elo)
                self.logger.record("self_play/opponent_elo", self.opponent_elo)
        return True
```

Was du auf TensorBoard sehen willst:

- `self_play/agent_elo` steigt stetig über das Training.
- Eine sich weitende Lücke zwischen dem ELO des Agenten und dem eines *festen* Baseline-Gegners (einer deiner frühen Checkpoints, nie aktualisiert).
- Für League-Training: eine Leiter, in der spätere Checkpoints höher sitzen als frühere.

Wenn ELO **flach** ist, funktioniert das Training nicht — selbst wenn der Loss gut aussieht und die Belohnung nahe null sitzt (wie sie sollte).

---

## 7 · League-Training Implementierungsskizze

League-Training ist der Goldstandard. Unten ist eine minimale `League`-Klasse — weniger als 50 Zeilen — die dich den größten Teil des Weges zu AlphaStar-artigem Training bringt.

```python
import random
from stable_baselines3 import PPO

class League:
    def __init__(self, initial_model_path, max_size=10):
        self.members = [initial_model_path]
        self.max_size = max_size
        self.elos = {initial_model_path: 1000.0}

    def add(self, checkpoint_path, current_elo):
        self.members.append(checkpoint_path)
        self.elos[checkpoint_path] = current_elo
        if len(self.members) > self.max_size:
            # Remove weakest member
            weakest = min(self.members, key=lambda p: self.elos[p])
            self.members.remove(weakest)
            del self.elos[weakest]

    def sample_opponent(self, strategy="uniform"):
        if strategy == "uniform":
            return random.choice(self.members)
        elif strategy == "prioritized":
            # More likely to sample opponents close to current skill
            # (implementation varies)
            return random.choice(self.members[-3:])  # recent checkpoints

    def load_opponent(self, path):
        return PPO.load(path)

    def update_elo(self, opponent_path, agent_won, agent_elo, k=16):
        opp_elo = self.elos[opponent_path]
        if agent_won:
            new_agent, new_opp = update_elo(agent_elo, opp_elo, k)
        else:
            new_opp, new_agent = update_elo(opp_elo, agent_elo, k)
        self.elos[opponent_path] = new_opp
        return new_agent
```

### Wie die äußere Trainingsschleife sie verwendet

```python
league = League("self_play_checkpoints/checkpoint_0.zip", max_size=10)
agent_elo = 1000.0

for iteration in range(1, 50):
    # Sample a fresh opponent for this training window
    opp_path = league.sample_opponent(strategy="uniform")
    opponent_model = league.load_opponent(opp_path)
    print(f"Iteration {iteration}: opponent = {opp_path} "
          f"(elo {league.elos[opp_path]:.0f})")

    # Train against this frozen opponent for 50k steps
    model.learn(total_timesteps=50_000, reset_num_timesteps=False)

    # Evaluate to update ELO (e.g. 50 deterministic matches)
    wins = evaluate_against(model, opponent_model, n_matches=50)
    agent_elo = league.update_elo(opp_path,
                                   agent_won=(wins > 25),
                                   agent_elo=agent_elo)

    # Add a fresh checkpoint to the league
    new_ckpt = f"self_play_checkpoints/checkpoint_{iteration}.zip"
    model.save(new_ckpt)
    league.add(new_ckpt, agent_elo)
```

### Sampling-Strategien

- **Uniform**: jedes League-Mitglied gleich wahrscheinlich. Maximal divers, aber der Lerner verbringt viel Zeit damit, einfache Gegner zu zerstören.
- **Priorisiert (jüngst)**: bias zugunsten neuerer Checkpoints. Schnelleres Lernen in frühen Phasen; Risiko, ältere Gegner zu vergessen.
- **Priorisiert (nah-ELO)**: gewichte Gegner nach ELO-Ähnlichkeit. Spielt immer enge Matches → max Information pro Spiel. Das ist, was AlphaStars „Hauptagenten" verwenden.
- **Exploiter-Sampling** (fortgeschritten): halte ein paar „Exploiter"-Agenten, die speziell darauf trainiert sind, den aktuellen Hauptagenten zu schlagen. Sample sie oft. Zwingt den Hauptagenten, Robustheit zu lernen.

Für diese Unit ist uniformes Sampling ausreichend. Bau darauf in den Stretch Goals auf.

---

## 8 · Viz-Checkpoint für Self-Play

Nach dem Training lade das finale Modell und führe es mit `--viz` aus:

```bash
python train_selfplay.py --viz --resume_model_path airhockey_selfplay_final.zip
```

Beobachte, was die KI tut. Self-Play hat sehr spezifische Verhaltenssignaturen.

### Worauf zu achten ist

- **Verfolgt es den Puck oder steht es nur still?** Stillstehen — oder schlimmer, sich in eine Ecke zurückziehen und sich nie bewegen — bedeutet, dass das Training fehlgeschlagen ist. Der Agent hat gelernt, dass Nichtstun die Chance minimiert, ein Tor zu kassieren, und das Reward Shaping hat Passivität nicht ausreichend bestraft.
- **Blockiert es Schüsse oder greift es nur an?** Wenn die KI nur angreift, bestraft die Belohnung das Tor-Kassieren wahrscheinlich nicht stark genug im Vergleich zum Treffen. Wenn sie nur blockiert, das Umgekehrte. Gesunde Self-Play-KIs tun beides, oft wechselnd je nach Puck-Position.
- **Zeigt sie erkennbare Strategien?** Achte auf: Positionswinkel, um das Tor abzudecken, Fake-Moves, um den Gegner zu locken, Bankenschüsse von den Wänden, Abfangen an der Mittellinie. Emergente Strategie ist die Belohnung. **Emergente Strategie ist der Grund, warum du Self-Play ausgeführt hast.**

### Die Signatur, dass Self-Play funktioniert

Der einzelne diagnostischste Test: **lade Checkpoints aus verschiedenen Generationen und beobachte, wie sie gegeneinander spielen.**

- Generation 0 (zufällige Init): Paddel driften, gelegentliche zufällige Treffer.
- Generation 5: Paddel verfolgen den Puck zuverlässig, einfaches Blocken und Treffen.
- Generation 10: Positionsspiel entsteht, Paddel decken das Tor ab.
- Generation 20: bewusste Winkel, Fakes, Erholungen.

Wenn Checkpoint 5 identisch zu Checkpoint 10 spielt, der identisch zu Checkpoint 20 spielt — ist das Training stagniert. Wenn jede Generation **merklich anders spielt und die vorige Generation häufiger schlägt, als sie verliert** — funktioniert Self-Play.

Das ist das visuelle Analog zum Steigen des ELO. Wenn du nur Zeit für eine Diagnose hast, führe ein Round-Robin-Turnier zwischen Checkpoints durch und tabelliere Siegraten.

---

## 9 · Self-Play für kooperative Aufgaben

Self-Play wird gewöhnlich als kompetitive Technik eingeführt, funktioniert aber genauso gut für kooperative Aufgaben. Der Mechanismus ist derselbe: trainiere gegen eine eingefrorene Kopie deiner selbst.

### Kooperatives Beispiel

Stell dir zwei Roboter vor, die koordinieren müssen, um ein schweres Objekt durch eine Tür zu tragen. Keiner kann es alleine schaffen. Sie müssen synchron bewegen, gemeinsam heben, gemeinsam navigieren.

Eine geteilte Policy kontrolliert beide Roboter. Während des Trainings:

- Der **Roboter des Lerners** verwendet die Live-Policy.
- Der **Partnerroboter** verwendet einen eingefrorenen Checkpoint derselben Policy.

Beide Roboter erhalten dieselbe geteilte Belohnung (Objekt geliefert → beide gewinnen). Der Lerner verbessert sich; periodisch wird der Partner auf einen neueren Checkpoint aktualisiert.

Warum eingefroren statt live? Wegen desselben Nichtstationaritäts-Arguments aus Abschnitt 2. Mit beiden Robotern live lernend ist das Koordinationssignal rauschig — dein Partner bewegt sich bei jedem Gradientenschritt anders. Mit einem eingefrorenen Partner hast du etwas Stabiles, mit dem du **koordinieren** kannst.

### Unterschied zur Multi-Agent-Shared-Policy

Das ist subtil, aber wichtig:

- **Multi-Agent-Shared-Policy (Unit 7)**: ein Netz kontrolliert alle Agenten, alle Gradient-Updates geschehen auf Rollouts von allen Agenten gleichzeitig. Der „Partner" aktualisiert bei jedem Schritt.
- **Kooperatives Self-Play**: ein Netz ist der Lerner, eine eingefrorene Kopie ist der Partner. Der Partner aktualisiert sich nur, wenn du einen neuen Checkpoint promotest.

Das erste ist schneller, aber instabil für komplexe Koordination. Das zweite ist langsamer, produziert aber robustere kooperative Policies, weil der Lerner mit mehreren historischen Versionen seiner selbst über das Training hinweg koordinieren muss.

---

## 10 · Verbindung zu RLHF und Alignment

Self-Play ist mehr als ein Spielen-Trick — es ist ein grundlegendes Muster in moderner Alignment-Forschung.

### Constitutional AI (Anthropic, 2022)

Das Modell kritisiert seine eigenen Outputs gegen eine Menge von Prinzipien, dann überarbeitet es sie. Der Kritiker und der Schreiber sind **dasselbe Modell**, in zwei Rollen angewendet. Das ist Self-Play in der Textdomäne: das Modell verbessert sich, indem es gegen eine Kopie seiner selbst spielt, die als Richter agiert.

Die Trainingsschleife:

1. Modell schreibt eine Antwort.
2. Dasselbe Modell (oder eine eingefrorene Kopie) kritisiert die Antwort gemäß einer Verfassung.
3. Modell überarbeitet die Antwort basierend auf der Kritik.
4. RLHF-artiges Fine-Tuning belohnt Überarbeitungen, die besser mit der Verfassung übereinstimmen.

### Debatte (Irving, Christiano, Amodei, 2018)

Zwei KI-Debattierer argumentieren gegensätzliche Seiten einer Frage. Ein menschlicher Richter wählt den Gewinner. Keiner der Debattierer weiß im Voraus, welche Seite „korrekt" ist; beide müssen den stärksten Fall konstruieren, den sie können. Der Druck des Gegners erzwingt ehrliche, gut gestützte Argumentation — in der Theorie mehr als eine einzelne KI, die direkt antwortet.

Die strukturelle Parallele zu AirHockey ist exakt: zwei Policies (möglicherweise dasselbe Netz in zwei Rollen) konkurrieren in einem Nullsummenspiel; der Score wird von einem externen Richter entschieden (dem Menschen, oder schließlich einer anderen KI). Verbesserung wird durch Self-Play getrieben.

### Warum das für den Kurs wichtig ist

Der Alignment-fokussierte Folgekurs behandelt RLHF, Constitutional AI und Debatte direkt. **Self-Play ist die konzeptionelle Auffahrt** zu all dem. Sobald du verstehst, was es bedeutet, gegen eine eingefrorene Kopie deiner selbst zu trainieren, ist der Sprung zum Trainieren eines Sprachmodells, das sich selbst kritisiert, klein.

---

## 11 · Stretch Goals

Wähle eines (oder mehrere) dieser Goals, um dein Verständnis von Self-Play zu vertiefen.

### Stretch 1: Eine League bauen

Implementiere die `League`-Klasse aus Abschnitt 7 vollständig. Trainiere 10 Generationen auf AirHockey mit uniformem Sampling. Führe nach jeder Generation ein Round-Robin-Turnier innerhalb der League durch und berechne ELO für jedes Mitglied. Plotte ELO-Kurven über die Zeit.

Du solltest sehen: spätere Generationen haben monoton steigendes ELO; der ursprüngliche zufällige Checkpoint sinkt nach unten; die Kurven fächern sich in eine klare Hierarchie auf.

### Stretch 2: Asymmetrisches Self-Play

Modifiziere die AirHockey-Umgebung so, dass die beiden Paddel **unterschiedliche Beobachtungsräume** haben:

- Paddel A sieht die volle Beobachtung (eigener Zustand, Gegnerzustand, Puck-Zustand).
- Paddel B sieht nur seinen eigenen Zustand und den Puck — **nicht den Gegner**.

Trainiere mit Shared-Policy-Self-Play. Welches Paddel dominiert? Interessanter: was lernt das **schwächere Paddel (B)** zur Kompensation? Oft lernt es konservatives defensives Spiel — das Tor abzudecken, ohne zu versuchen, Gegnerzüge vorherzusagen. Das ist ein kleiner Vorgeschmack auf Dynamiken partieller Beobachtbarkeit aus [Unit 8](unit-08.md).

### Stretch 3: Self-Play auf Racer

Die Racer-Umgebung aus Unit 7 unterstützt kompetitives Multi-Agent-Rennen. Aktuell hast du sie mit unabhängigen parallelen Agenten trainiert (jedes Auto hat seine eigene Policy). Re-implementiere mit **League-basiertem Self-Play**: eine geteilte Policy, Gegner-Sampling aus einer League vergangener Checkpoints.

Vergleiche:

- Lernt die Self-Play-Version schneller als das unabhängige Training?
- Produziert sie diversere Fahrstrategien?
- Ist die finale Policy robuster gegen zurückgehaltene Test-Gegner (z. B. ein Checkpoint aus unabhängigem Training)?

Das ist das, was einem Self-Play-Experiment auf Forschungsniveau am nächsten kommt, das du an einem Abend auf einer einzigen GPU ausführen kannst.

---

## Was kommt als Nächstes

Self-Play schließt den Kreis von Phase 4. Du hast jetzt gesehen:

- **Multi-Agent** ([Unit 7](unit-07.md)) — mehrere Policies, möglicherweise kooperativ oder kompetitiv.
- **Memory & POMDPs** ([Unit 8](unit-08.md)) — Umgang mit partieller Beobachtbarkeit durch Rekurrenz.
- **Self-Play** (diese Unit) — sich selbst als Curriculum verwenden.

**Hierarchisches RL:** Long-Horizon-Aufgaben, bei denen flaches PPO stagniert — Multi-Raum-Navigation, Multi-Schritt-Assemblierung — können mit einer High-Level-Policy zerlegt werden, die Subziele setzt, und einer Low-Level-Policy, die sie erreicht.

[→ Hierarchisches RL](unit-hierarchical.md)

---

[← Memory & POMDPs](unit-08.md) · [Kursstartseite](index.md) · [→ Hierarchisches RL](unit-hierarchical.md)
