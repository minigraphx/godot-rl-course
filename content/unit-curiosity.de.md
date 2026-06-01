# Intrinsische Motivation — Neugier (Curiosity) und spärliche Belohnungen

[← Deep Q-Learning](unit-03.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~30 min · Training: ~30 min GPU / ~2 Std CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    Godot (Erkundungsabdeckung — wie viel der Karte besucht der Agent?) · TensorBoard (`rollout/ep_rew_mean` steigt früher als die Baseline ohne Neugier) · RND-Vorhersagefehler, der über das Training fällt, sobald Zustände vertraut werden

---

## 1 · Das Problem spärlicher Belohnungen

Standard-RL-Training setzt voraus, dass der Agent gelegentlich zufällig auf eine Belohnung stößt — und dieses Glücksverhalten dann wiederholen lernt. Umgebungen mit dichten Belohnungen machen das einfach: jeder kleine Schritt nach vorne gibt ein Signal ungleich null, Gradienten fließen, die Policy wird aktualisiert.

Spärliche Belohnungen brechen diese Annahme.

**Beispiel:** ein Labyrinth, in dem die einzige Belohnung +1 am Ausgang ist und 0 überall sonst. Mit einer zufälligen Policy könnte der Agent 1 Mio. Schritte laufen, ohne ein einziges positives Signal. Keine Belohnung → kein Gradient → kein Lernen. Die Policy bleibt im zufälligen Verhalten eingefroren.

Der Fehlermodus sieht in TensorBoard wie ein flacher `ep_rew_mean` aus — nicht oszillierend, einfach perfekt flach — weil der Agent für das Aufgabenziel praktisch blind ist.

**Warum die üblichen Tricks das nicht beheben:**

- Mehr Timesteps: der Agent erkundet weiter blind — ein längerer Random Walk hilft selten
- ε-greedy (DQN): in unerforschten Regionen weiterhin zufällig
- Entropie-Bonus: hält die Policy verteilt, *lenkt* die Erkundung aber nicht zu neuen Zuständen
- Reward Shaping: funktioniert, erfordert aber für jede neue Umgebung eine handgefertigte dichte Proxy-Belohnung

Gebraucht wird ein allgemeiner Mechanismus, der den Agenten zur Erkundung motiviert — unabhängig vom externen Belohnungssignal.

---

## 2 · Intrinsische Motivation: Neugier als Belohnung

Die Kernidee: dem Agenten zusätzlich eine Belohnung dafür geben, **neue Zustände** zu besuchen — unabhängig davon, was die Umgebung sagt.

```
r_total = r_ext + β · r_int
```

- `r_ext` — die externe Belohnung der Umgebung (spärlich, selten, aufgabenspezifisch)
- `r_int` — eine intrinsische Neugier-Belohnung (dicht, intern erzeugt, beruht auf Zustands-Neuheit)
- `β` — Skalierungsfaktor, der Erkundungsdrang gegen Aufgabenleistung balanciert (typisch: 0,01–1,0)

`r_int` ist hoch für Zustände, die der Agent bisher nicht besucht hat, und fällt gegen null, sobald Zustände vertraut werden. Der Agent wird intrinsisch motiviert, seine Umgebung zu erkunden — nicht weil ein Designer es ihm gesagt hat, sondern weil Neuheit selbst belohnend ist.

Das spiegelt Theorien menschlicher Motivation: Säuglinge sind „neugierig" auf neue Reize, lange bevor sie verstehen, wofür diese Reize nützlich sind.

**Schlüsseleigenschaft:** intrinsische Belohnungen wirken auch dann, wenn `r_ext = 0` für die gesamte erste Trainingsphase ist. Der Agent erkundet breit, baut ein internes Modell der Umgebung auf und beginnt *dann* zu nutzen, sobald externe Belohnungen auftreten.

---

## 3 · Random Network Distillation (RND)

RND (Burda et al. 2018) ist die einfachste effektive Neugier-Methode und der Standardstart im Deep RL.

**Der Aufbau — zwei Netze:**

| Netz | Rolle | Trainiert? |
|------|-------|------------|
| **Target-Netz** f: obs → Embedding | Erzeugt ein festes zufälliges Embedding jeder Beobachtung | Nein — Gewichte sind bei zufälliger Initialisierung eingefroren |
| **Predictor-Netz** g_θ: obs → Embedding | Versucht die Ausgabe des Target-Netzes zu treffen | Ja — wird auf jeder besuchten Beobachtung trainiert |

**Die intrinsische Belohnung:**

```
r_int = ||f(obs) - g_θ(obs)||²
```

Das ist der mittlere quadratische Vorhersagefehler zwischen den Ausgaben der beiden Netze für die aktuelle Beobachtung.

**Warum das Neuheit misst:**

- **Neuer Zustand:** der Predictor hat diese Beobachtung nie gesehen → keine gelernte Abbildung → hoher Vorhersagefehler → hohes `r_int`
- **Vertrauter Zustand:** der Predictor wurde oft auf dieser Beobachtung trainiert → er trifft das Target nahezu → niedriger Fehler → niedriges `r_int`

**Warum das Target-Netz fix ist — die Schlüsselerkenntnis:**

Das Target muss *fest* sein (zufällige, eingefrorene Gewichte). Würde es ebenfalls aktualisiert, könnte der Predictor dem Target unabhängig von Neuheit einfach „folgen" — der Vorhersagefehler würde überall auf null kollabieren. Das eingefrorene Zufalls-Target erzeugt eine stabile, konsistente Funktion, die der Predictor nur treffen kann, indem er die Beobachtung *tatsächlich während des Trainings sieht*.

**Im Vergleich zu früheren Neugier-Methoden:**

Ältere Ansätze (ICM — Intrinsic Curiosity Module, Pathak et al. 2017) nutzten ein gelerntes Forward-Modell, um `s_{t+1}` aus `(s_t, a_t)` vorherzusagen, und maßen Überraschung als Vorhersagefehler. Das erforderte das Lernen von Forward- *und* Inverse-Dynamik-Modellen — komplexer und anfällig für das „Noisy-TV-Problem" (stochastische Umgebungen wirken unendlich neu). RND umgeht das vollständig: es modelliert keine Übergänge, nur Embeddings.

---

## 3.1 · ICM vs. RND — wann welches

**ICM-Rekap (Intrinsic Curiosity Module, Pathak et al. 2017):** zwei gemeinsam trainierte Netze — ein *Inverse-Modell* sagt die ausgeführte Aktion aus `(s_t, s_{t+1})` voraus (zwingt die Features, aktionsrelevant zu sein), und ein *Forward-Modell* sagt das Embedding des nächsten Zustands aus `(s_t, a_t)` voraus. Neugier = Forward-Modellfehler im Feature-Raum.

**Das Noisy-TV-Problem:** ICM misst, wie *unvorhersehbar* der nächste Zustand ist. Ein Fernseher, der zufälliges Rauschen zeigt, ist immer unvorhersehbar — der Agent bleibt für immer davor hängen, weil jedes Bild „neu" ist. Jedes stochastische Element in der Umgebung (zufällige Partikeleffekte, prozedurales Rauschen) wird zum Neugier-Magnet.

**Warum RND das vermeidet:** das feste Zufalls-Target-Netz erzeugt für dieselbe Beobachtung jedes Mal dieselbe Ausgabe. Ein verrauschter Fernseher erzeugt dieselbe Pixelmuster-Verteilung — nach wenigen Besuchen trifft der Predictor das Target für diese Muster und `r_int` fällt fast auf null. RND misst *Unbekanntheit*, nicht *Unvorhersehbarkeit*.

| | ICM | RND |
|--|-----|-----|
| Was es misst | Forward-Modell-Vorhersagefehler | Abstand zum festen Zufallsnetz |
| Noisy-TV-Problem | Ja — stochastische Envs täuschen es | Nein — stochastische Obs haben festes Target |
| Rechenaufwand | Höher (zwei Netze) | Niedriger (ein Predictor) |
| Feature-Raum | Gelernt (Inverse-Modell) | Zufällige Projektion |
| Am besten für | Deterministische Envs, aktionsrelevante Features | Allgemein — der sichere Default |

**Wann ICM:** vollständig deterministische Umgebungen, in denen der Agent alle Zustandsänderungen steuert und du willst, dass die Neugier-Features aktionsrelevante Struktur erfassen. **Default ist RND** in allen anderen Fällen — einfacher, billiger und mit stochastischen Umgebungen korrekt.

---

## 4 · RND in der Praxis mit Stable-Baselines3

Pakete installieren:

```bash
pip install stable-baselines3 sb3-contrib
```

Der `RNDWrapper` unten umhüllt jede Godot-Umgebung, um RND-intrinsische Belohnungen hinzuzufügen. Kernidee: jeden `step()`-Aufruf abfangen, `r_int` berechnen und zu `r_ext` addieren.

```python
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv


class RNDModule:
    """Computes RND intrinsic rewards for a given observation dimension."""

    def __init__(self, obs_dim: int, embed_dim: int = 64, lr: float = 1e-3):
        # Fixed random target — never updated
        self.target = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )
        for param in self.target.parameters():
            param.requires_grad = False

        # Trained predictor — updated on every step
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)

    def compute_reward_and_train(self, obs_np: np.ndarray) -> float:
        """Return intrinsic reward for obs_np, and update the predictor."""
        obs_t = torch.tensor(obs_np, dtype=torch.float32)

        with torch.no_grad():
            target_embed = self.target(obs_t)

        pred_embed = self.predictor(obs_t)
        error = ((target_embed - pred_embed) ** 2).mean()

        # Train predictor to match target on this observation
        self.opt.zero_grad()
        error.backward()
        self.opt.step()

        return error.item()


class RNDGodotEnv:
    """
    Wraps a StableBaselinesGodotEnv to inject RND intrinsic rewards.

    Usage:
        env = RNDGodotEnv("./MyEnv.x86_64", beta=0.1)
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(500_000)
        env.close()
    """

    def __init__(self, env_path: str, beta: float = 0.1, n_parallel: int = 4, speedup: int = 20):
        self.env = StableBaselinesGodotEnv(
            env_path=env_path, n_parallel=n_parallel, speedup=speedup
        )
        self.beta = beta
        obs_dim = self.env.observation_space.shape[0]
        self.rnd = RNDModule(obs_dim=obs_dim)

        # Expose SB3-required attributes
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, r_ext, done, info = self.env.step(action)
        r_int = self.rnd.compute_reward_and_train(obs)
        r_total = r_ext + self.beta * r_int
        return obs, r_total, done, info

    def close(self):
        self.env.close()


# Training with RND
env = RNDGodotEnv("./MultiLevelRobot.x86_64", beta=0.1, n_parallel=4, speedup=20)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/")
model.learn(total_timesteps=1_000_000)
model.save("multilevel_rnd")
env.close()
```

!!! warning "β-Tuning ist umgebungsspezifisch"
    Ist β zu hoch, ignoriert der Agent die externen Belohnungen und erkundet endlos — `ep_rew_mean` bleibt selbst nach Millionen Schritten nahe null. Ist β zu niedrig, liefert Neugier kein Signal. Starte mit β = 0,1 und halbiere/verdopple, je nachdem ob der Agent genug erkundet oder die Aufgabe ignoriert. Beobachte sowohl `ep_rew_mean` (externe Aufgabe) als auch den RND-Vorhersagefehler in TensorBoard.

**RND-Vorhersagefehler loggen:**

Füge das deiner Trainingsschleife hinzu, um die Neugier-Signalstärke über die Zeit zu verfolgen:

```python
# After each update step, log mean prediction error across recent batch
# Lower error = agent is visiting more familiar states = exploration maturing
```

---

## 5 · Wo Neugier in Godot hilft

**Gute Einsatzgebiete:**

- **MultiLevelRobot (Unit 9):** der Agent muss zu Plattformen navigieren, die er nie besucht hat. Zufällige Erkundung erreicht selten höhere Plattformen. RND drängt den Agenten, neue Höhenebenen aufzusuchen.
- **FPS / RobotFPS (Unit 8):** Agenten, die hinter Wänden festhängen, müssen Türen oder Korridore entdecken. Standard-ε-greedy erkundet lokal; RND treibt globale Neuheit.
- **Jede labyrinthartige Umgebung** mit nur einer terminalen Belohnung — der klassische Fall spärlicher Belohnungen.

!!! warning "Keine Neugier in dichten Belohnungs-Umgebungen"
    In Umgebungen mit dichten, gut geformten Belohnungen (BallChase, LunarLander, CrossTheRoad mit Vorwärts-Bonus) bringt Neugier Rauschen ein. Der Agent erkundet eventuell irrelevante Zustände, statt das Aufgabensignal zu optimieren. Dichte Belohnung + Neugier = langsameres Lernen, nicht schnelleres. Wenn `ep_rew_mean` ohne Neugier glatt steigt, lass sie weg.

**Diagnose-Checkliste — wann Neugier hinzufügen:**

1. `ep_rew_mean` ist > 200 k Schritte flach?
2. Die Belohnungsfunktion hat pro Episode nur 1–2 terminale Belohnungsevents?
3. Zufällige Erkundung kann die Belohnung ohne ausgedehntes Glück nicht erreichen?

Wenn alle drei Ja → versuche RND. Sonst → Reward Shaping (siehe [Reward-Engineering-Unit](unit-reward-engineering.md)) ist vermutlich der bessere Hebel.

---

## 6 · Zählungsbasierte Erkundung (Count-based exploration)

Der älteste Erkundungs-Bonus: belohne den Agenten invers proportional dazu, wie oft er einen Zustand besucht hat.

```
r_int = 1 / sqrt(N(s))
```

wobei `N(s)` die Besuchsanzahl ist. Nie besuchte Zustände bekommen hohen Bonus; gut erkundete fast null. Das ist im tabellarischen Fall beweisbar optimal — es treibt den Agenten dazu, jeden Zustand mindestens O(√T) Mal in T Schritten zu besuchen.

**UCB (Upper Confidence Bound)** — erweitert dieselbe Idee auf die Aktionsauswahl in Bandit-Problemen:

```
a* = argmax_a [ Q(s,a) + c · sqrt(log t / N(s,a)) ]
```

Der zweite Term ist der Erkundungs-Bonus: hoch, wenn eine Aktion selten probiert wurde. In der Theorie gut untersucht; SB3 implementiert UCB für Deep RL nicht.

**SimHash / Locality-Sensitive Hashing** — approximatives Zählen in kontinuierlichen Räumen: die Beobachtung mit einer zufälligen Projektionsmatrix in einen diskreten Bucket hashen, dann Bucket-Besuche zählen. `r_int = 1 / sqrt(N(hash(s)))`. Rechnerisch günstig; funktioniert bei hochdimensionalen Beobachtungen.

**Warum exaktes Zählen nicht skaliert:** kontinuierliche Zustandsräume machen jeden Zustand einzigartig — ein Roboter bei (1,000, 2,000) und (1,001, 2,000) sind technisch unterschiedliche Zustände, beide mit N=0. SimHash aggregiert benachbarte Zustände in denselben Bucket.

| Methode | Braucht exakte Zustände? | Noisy-TV-sicher? | Typischer Einsatz |
|---------|--------------------------|------------------|-------------------|
| Count-based (exakt) | Ja | Ja | Tabellarisches FrozenLake |
| SimHash | Nein (Bucket-Counts) | Ja | Niedrig- bis mitteldimensionale kontinuierliche Obs |
| RND | Nein | Ja | Allgemein im Deep RL |
| ICM | Nein | **Nein** | Nur deterministische Envs |

**Praktische Empfehlung:** für die meisten Godot-Aufgaben RND. SimHash lohnt einen Versuch, wenn der Beobachtungsraum niedrig- bis mitteldimensional ist und du etwas Einfacheres als ein neuronales Netz willst. Count-Vergleich: FrozenLake-Q-Learning mit und ohne `1/sqrt(N)` trainieren — die Differenz der Schritte bis zur optimalen Policy ist eine sinnvolle Übung (Zusatzaufgabe, Abschnitt 9).

### Bau es · Count-Bonus auf FrozenLake

Nimm die FrozenLake-Q-Learning-Schleife aus der [Q-Learning-Unit](unit-q-learning.md) und ergänze einen `1/√N`-Neuheitsbonus. Der Punkt: sieh, wie eine zählbasierte intrinsische Belohnung ε-greedy als Treiber der Erkundung ersetzt.

```python
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
Q = np.zeros((n_states, env.action_space.n))
N = np.zeros(n_states)                       # state visit counts
alpha, gamma, beta = 0.1, 0.99, 0.1

for episode in range(5000):
    s, _ = env.reset()
    done = False
    while not done:
        a = int(np.argmax(Q[s]))             # greedy — the count bonus drives exploration
        s2, r_ext, terminated, truncated, _ = env.step(a)
        N[s2] += 1
        r_int = beta / np.sqrt(N[s2])         # intrinsic novelty bonus
        r = r_ext + r_int
        Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])
        s, done = s2, terminated or truncated

env.close()
```

!!! check "Fertig, wenn"
    Der Count-Bonus-Agent erreicht eine ~100 %ige gierige Erfolgsrate (die Evaluierung der Q-Learning-Unit) in **deutlich weniger Episoden** als die ε-greedy-Baseline. Der `1/√N`-Term, nicht ε, übernimmt jetzt die Erkundung — bestätige das, indem du prüfst, dass es mit *vollständig gieriger* Aktionswahl funktioniert (ganz ohne ε).

---

## 7 · Entropie-Bonus vs. Neugier

Diese adressieren unterschiedliche Erkundungs-Ebenen und ergänzen sich:

| Mechanismus | Steuert | Ebene |
|-------------|---------|-------|
| `ent_coef` in PPO | Hält die **Aktionsverteilung** breit | Aktions-Diversität |
| RND-Neugier-Bonus | Belohnt das Besuchen **neuer Zustände** | Zustands-Diversität |

**Entropie-Bonus** (`ent_coef=0,01` ist SB3-Default) verhindert, dass die Policy auf eine einzelne deterministische Aktion kollabiert. Er sagt: „mach nicht immer dasselbe." Er kostet nichts extra — PPO berechnet die Entropie der Policy-Verteilung ohnehin.

**Neugier-Bonus** belohnt das Entdecken neuer Regionen des Zustandsraums. Er sagt: „geh dahin, wo du noch nicht warst." Er braucht ein zusätzliches Netz und verursacht zusätzliche Trainingskosten.

!!! tip "Bei harten Erkundungsaufgaben beide kombinieren"
    Für die härtesten Erkundungsprobleme: `ent_coef` + RND:

    - Entropie-Bonus hält die Policy auf Aktionsebene variabel — weniger Risiko, in einer lokalen Schleife zu landen
    - RND lenkt die Erkundung zu global neuen Zuständen

    Setze `ent_coef=0,01–0,05` in PPO und `beta=0,05–0,2` für RND. Wenn das Verhalten im Viz-Checkpoint repetitiv wirkt, erst `ent_coef` erhöhen (billig). Wenn der Agent keine neuen Kartenregionen erreicht, `beta` erhöhen.

---

## 8 · Viz-Checkpoint für Neugier

Nach dem Training mit und ohne RND beide Policies mit `--viz` (oder `show_window=True` im Eval-Skript) laufen lassen:

**Worauf in Godot achten:**

- Besucht der mit Neugier trainierte Agent mehr Karte, bevor er das Ziel findet?
- Kehrt er wiederholt in dieselbe Ecke zurück (niedriges Neugier-Signal) oder verteilt er sich systematisch?
- Bleibt der Agent ohne Neugier am Spawn-Punkt kleben, während der Neugier-Agent entlegene Plattformen entdeckt?

**Worauf in TensorBoard achten:**

| Metrik | Ohne Neugier | Mit Neugier |
|--------|--------------|-------------|
| `rollout/ep_rew_mean` | Bleibt > 500 k Schritte flach | Beginnt bei ~100–200 k Schritten zu steigen |
| RND-Vorhersagefehler | N/A | Anfangs hoch, fällt mit zunehmender Vertrautheit |
| `rollout/ep_len_mean` | Episoden enden am Spawn (früher Tod oder Timeout) | Längere Episoden, je mehr der Agent erkundet |

Das Neugier-Signal sollte eine deutliche Abklingkurve zeigen: hoher Vorhersagefehler in den ersten 20 % des Trainings (alles ist neu), fallend, sobald der Agent die Umgebung kennt. Eine flache oder nicht abfallende Fehlerkurve heißt, der Agent besucht keine neuen Zustände — prüfe, ob er stirbt, bevor er neue Regionen erreicht.

---

## 9 · Stretch Goals

- **CrossTheRoad-Test (Unit 3):** Wende RND auf CrossTheRoad an. Hilft Neugier (spärliche Belohnung) oder schadet sie (kleiner, aber von null verschiedener Vorwärts-Bonus)? Miss die Konvergenzgeschwindigkeit von `ep_rew_mean`.
- **Paper lesen:** Burda et al. 2018, „Exploration by Random Network Distillation". ~10 Seiten, klar geschrieben, mit Atari-Resultaten. Die Ablation in Abschnitt 5 ist besonders lehrreich — sie zeigt, warum das feste zufällige Target essenziell ist.
- **β-Sweep:** Trainiere MultiLevelRobot mit β ∈ {0,01, 0,1, 0,5, 1,0}. Plotte `ep_rew_mean` über Timesteps für jedes β. Was kostet ein zu großes β?

---

## Was kommt als Nächstes

Mit Neugier kann dein Agent Umgebungen meistern, an denen Standard-PPO/DQN wegen spärlicher Belohnungen scheitern würden. Die nächste Unit behandelt Policy Gradients — die theoretische Grundlage hinter PPO und warum policy-basierte Methoden auf kontinuierliche Aktionen skalieren.

[→ Policy Gradients](unit-policy-gradients.md)
