# Ziel-bedingtes RL & Hindsight Experience Replay

[← Lokomotion-Agenten](unit-locomotion.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~35 min · Training: ~30 min GPU / ~2 h CPU

!!! info "Drei Wege, deine KI zu beobachten"
    - **Godot-Viewport** — beobachte, wie der Arm in jeder Episode auf einen zufällig gespawnten Ziel-Marker zugreift; das Ziel bewegt sich, aber dieselbe Policy bewältigt es.
    - **TensorBoard** — plotte `rollout/success_rate` vs `timesteps` für zwei Läufe nebeneinander: SAC allein (flach bei null) vs SAC+HER (steigt auf ~0,9).
    - **gymnasium-FetchReach-Render** — `env.render()` zeigt einen 3-DOF-Fetch-Arm in MuJoCo, der eine farbige Zielkugel erreicht; identischer Algorithmus, anderer Simulator.

---

## 1 · Das Manipulationsproblem

Bis hierhin im Kurs hast du Roboterumgebungen in Godot gebaut (Unit Robotik), dichte Belohnungen geformt (Unit Reward Engineering) und sogar intrinsische Neugier draufgeschraubt, um Agenten über spärliche Belohnungs-Wände zu drücken (Unit Curiosity). Manipulationsaufgaben strapazieren nun jedes dieser Werkzeuge bis zum Bruchpunkt.

Betrachte die kanonische Roboterinstruktion:

> „Heb den roten Würfel auf und stell ihn auf den Teller."

Was macht das schwer?

- **Kombinatorischer Zielraum.** Der Würfel kann irgendwo auf dem Tisch starten; der Teller kann an Tausenden gültigen Positionen sitzen. Ein 1 m × 1 m-Tisch mit 1-cm-Auflösung gesampelt gibt schon 10 000 Würfelpositionen × 10 000 Tellerpositionen — hundert Millionen verschiedene Aufgabeninstanzen.
- **Binäre Belohnung.** Entweder landet der Würfel auf dem Teller (+1) oder nicht (0). Es gibt keinen Trostpreis für „fast".
- **Exploration ist hoffnungslos.** Ein Arm mit zufälliger Policy platziert keine Objekte zufällig. Er fuchtelt, schmeißt den Würfel vom Tisch und verdient null Belohnung für jeden seiner 500 Episoden-Steps.
- **Reward Shaping skaliert nicht.** Du könntest eine dichte Belohnung handcraften — `-distance(cube, plate)` — aber jede neue Zielvariation (ein neuer Teller, eine Stapelaufgabe, ein Pin-Einfügen) erfordert eine brandneue Belohnungsfunktion.

Die Kerneinsicht, die diese Unit treibt:

> **Wir wollen eine Policy, die für JEDE Zielposition funktioniert, nicht eine separate Policy für jedes Ziel.**

Eine Policy, der gesagt wurde „erreiche (0,5, 0,3, 0,2)", sollte sich anders verhalten als dieselbe Policy mit „erreiche (-0,4, 0,1, 0,6)" — ohne Nachtraining. Der Trick ist, das Ziel Teil der Observation zu machen und einen Weg zu finden, aus spärlichen Belohnungen zu lernen.

Dieser Trick ist **Hindsight Experience Replay**.

---

## 2 · Ziel-bedingtes RL

Die erste Hälfte der Lösung ist strukturell: erweitere den Observationsvektor um das gewünschte Ziel.

### Standard-Observation

```
s = [joint_angles, joint_velocities, end_effector_pos]
```

Eine Policy `π(a | s)`, die auf dieser Observation trainiert ist, kann nur eine Aufgabe lösen — die, für die sie beim Training belohnt wurde.

### Ziel-bedingte Observation

```
s = [joint_angles, joint_velocities, end_effector_pos, goal_pos]
                                                       ^^^^^^^^
                                                       neu!
```

Eine Policy `π(a | s, g)`, die auf dieser Observation trainiert ist, kann — prinzipiell — jede als Zielvektor `g` ausdrückbare Aufgabe lösen. Dieselben Netzgewichte, mit anderem Ziel abgefragt, produzieren andere Aktionen. Wir nennen das eine **universelle Policy** oder, in der formalen Literatur, einen **Universal Value Function Approximator (UVFA, Schaul et al. 2015)**.

### Beispiel: ziel-bedingte Observation in Godot

```gdscript
# Goal-conditioned observation in Godot
func get_obs() -> Dictionary:
    # Robot state (proprioceptive)
    var robot_obs = [
        end_effector.global_position.x / reach,
        end_effector.global_position.y / reach,
        end_effector.global_position.z / reach,
        linear_velocity.x / max_vel,
        linear_velocity.y / max_vel,
        linear_velocity.z / max_vel,
    ]

    # Goal (changes every episode)
    var goal_obs = [
        goal.global_position.x / reach,
        goal.global_position.y / reach,
        goal.global_position.z / reach,
    ]

    return {"obs": robot_obs + goal_obs}

func reset():
    # Randomize goal position each episode
    goal.global_position = Vector3(
        randf_range(-reach * 0.8, reach * 0.8),
        randf_range(0.1, reach * 0.5),
        randf_range(-reach * 0.8, reach * 0.8),
    )
    _ai.needs_reset = false
```

Zwei Dinge sind hervorhebenswert:

1. **Das Ziel wird in jeder Episode frisch gesampelt.** Hältst du das Ziel fest, kollabiert der Agent zu einem Einzelaufgaben-Lerner, der den Zielkanal ignoriert — es gibt keinen Anreiz, darauf zu achten.
2. **Belohnung ist rein spärlich.** Die Belohnungsfunktion, die der Agent sieht, ist:

```gdscript
func get_reward() -> float:
    var d = end_effector.global_position.distance_to(goal.global_position)
    return 1.0 if d < 0.05 else 0.0
```

Kein Distanz-Shaping, kein Proxy-Term, keine Curriculum-Tricks. Nur Erfolg oder Fehlschlag. Hier sterben die meisten Lerner — und hier wird HER uns retten.

---

## 3 · Warum spärliches ziel-bedingtes RL ohne HER versagt

Bevor wir nach HER greifen, sind wir ehrlich, wie schlimm vanilla RL bei dieser Art Aufgabe versagt. Die Mathematik ist düster.

### Das Volumenargument

- Der Arm hat 6 DOF × 2 (Winkel + Geschwindigkeit) = 12 Obs-Dimensionen, plus 3 Zieldimensionen = **15-dim Observation**.
- Erfolg ist definiert als Endeffektor innerhalb 5 cm vom Ziel.
- Wenn die Armreichweite ≈ 1 m ist, ist das Volumen der Erfolgsregion relativ zum Arbeitsraum etwa `(0,05 / 1,0)^3 ≈ 0,0125 %`.
- Eine zufällige Policy landet in der Erfolgsregion etwa **einmal pro 10 000 Episoden**.
- Bei 500 Steps pro Episode bräuchtest du ≈ **5 Millionen Umgebungsschritte, bevor du eine einzige positive Belohnung siehst**.

### Warum Standardalgorithmen ablaufen

- **PPO** updated aus On-policy-Rollouts. Wenn jeder Rollout Belohnung 0 liefert, sind die Advantage-Schätzungen null und der Policy-Gradient null. PPO bewegt sich einfach nicht.
- **SAC** speichert Übergänge in einem Replay-Buffer, aber wenn keine Transition Reward > 0 hat, sind die Q-Targets alle null und der Critic lernt die konstante Funktion `Q(s, a) = 0`. Der Actor optimiert dann gegen eine flache Landschaft und verbessert sich nie.

### Warum Curiosity uns nicht rettet

In Unit Curiosity hast du gesehen, dass ICM/RND spärliche-Belohnung-Explorationsaufgaben wie Montezuma's Revenge löst, indem Neuheit belohnt wird. Aber Manipulation hat ein subtil anderes Problem: der Arm muss ein **spezifisches** Ziel erreichen, nicht nur neue Zustände besuchen. Curiosity treibt den Arm dazu, in interessanten Konfigurationen herumzuwedeln, von denen keine zufällig das Ziel ist, das der Supervisor heute will.

Wir brauchen eine Methode, die die Daten, die wir bereits haben — selbst wenn diese Daten Fehlschläge aufzeichnen — wiederverwendet, um dem Agenten etwas Konkretes beizubringen. Diese Methode ist HER.

---

## 4 · Hindsight Experience Replay (HER)

> **Die Schlüsseleinsicht (Andrychowicz et al., NeurIPS 2017):** sogar fehlgeschlagene Episoden enthalten nützliche Informationen.

### Ein durchgearbeitetes Beispiel

Stell dir vor, dem Arm wird gesagt, Ziel `G = (0,5, 0,3, 0,2)` zu erreichen. Nach 500 Steps landet er bei `G' = (0,4, 0,25, 0,15)`. Die Belohnungsfunktion sagt `r = 0` — Fehlschlag.

Aber halt einen Moment inne. Der Arm hat **eine** Position erreicht. Es war nur nicht die, die wir verlangt haben. Was, wenn wir den Replay-Buffer anlügen und ihm sagen, die ursprüngliche Anweisung sei tatsächlich „erreiche `(0,4, 0,25, 0,15)`" gewesen? Dann wäre dieselbe Trajektorie ein Lehrbuch-Erfolg: null Belohnung für 499 Steps, dann +1 am letzten Step.

Das ist Hindsight-Relabeling. Es ist das philosophische Äquivalent zu „Das war Absicht".

### Der Algorithmus

1. **Rolle aus** eine Episode mit dem ursprünglich gesampelten Ziel `G`. Sammle Übergänge:

   ```
   (s_0, a_0, r_0, s_1), (s_1, a_1, r_1, s_2), ..., (s_T, a_T, r_T, s_{T+1})
   ```

   Jedes `s_t` enthält bereits `G` (weil die Observation ziel-bedingt ist).

2. **Speichere wie-ist** im Replay-Buffer. Diese Übergänge haben meist `r = 0`.

3. **Hindsight-Relabel.** Wähle ein Ersatzziel `G' = achieved_goal(s_{T+1})` — d. h. den Zustand, den der Arm tatsächlich erreicht hat. Gehe durch dieselbe Trajektorie und produziere **neue Übergänge**, bei denen der Zielkanal mit `G'` ersetzt ist und die Belohnung mit derselben Belohnungsfunktion **neu berechnet** wird:

   - Steps, an denen das erreichte Ziel noch fern von `G'` ist: `r = 0`.
   - Der letzte Step (an dem per Konstruktion das erreichte Ziel `G'` entspricht): `r = 1`.

4. **Speichere die relabelten Übergänge** neben den Originalen. Trainiere SAC/DDPG/TD3 auf dem kombinierten Buffer.

Jede Episode — Erfolg oder Fehlschlag — trägt nun mindestens einen positiven-Reward-Übergang zum Lernen bei.

### Die Transformation in Pseudocode

```python
# Original failed trajectory stored as:
# (obs=[robot_state, goal_G], action, reward=0, next_obs)  × T times
# (obs=[robot_state, goal_G], action, reward=0, next_obs)  ← last step, still failed

# Hindsight relabeled as:
# (obs=[robot_state, goal_G'], action, reward=0, next_obs) × T-1 times
# (obs=[robot_state, goal_G'], action, reward=1, next_obs) ← last step, "success" on G'
```

### Warum das funktioniert

Der Agent lernt nicht aufzugeben. Er lernt eine allgemeine Fähigkeit: **„jede Position zu erreichen, die ich je erreicht habe."** Weil die relabelten Ziele aus der Erreicht-Ziel-Verteilung der eigenen Trajektorien des Agenten stammen, beginnt das Curriculum natürlicherweise leicht (die chaotischen Orte erreichen, in die die zufällige Policy stolpert) und wird härter, während sich die Policy verbessert (die Trajektorien selbst werden bewusster, also werden die relabelten Ziele bedeutsamer).

Das ist effektiv ein automatisch generiertes Curriculum — ohne menschlichen Designaufwand.

!!! tip "Kompatible Algorithmen"
    HER ist ein **Replay-Buffer-Trick**, kein eigenständiger Algorithmus. Er passt in jede Off-policy-Methode: DDPG (Originalpaper), TD3, SAC. Er kombiniert sich **nicht** mit PPO, A2C oder irgendeiner On-policy-Methode, weil On-policy-Algorithmen alte Trajektorien wegwerfen.

---

## 5 · HER-Zielauswahl-Strategien

Der Relabeling-Schritt fragt: *welcher* Zustand soll als Ersatzziel `G'` verwendet werden? Das Originalpaper hat vier Strategien gebenchmarkt:

| Strategie | Wie `G'` gewählt wird | Effekt |
|----------|--------------------|--------|
| `final` | Letzter Zustand der Episode | Einfach, oft gut genug |
| `future` | Zufälliger Zustand aus *später* in derselben Episode | Vielfältigere Ziele, meist am besten |
| `episode` | Zufälliger Zustand aus irgendwo in derselben Episode | Uniform über Episode |
| `random` | Zufälliger Zustand aus dem gesamten Replay-Buffer | Maximale Vielfalt, langsamer |

Für jeden echten Übergang erzeugt HER typischerweise **k** zusätzliche relabelte Übergänge (`k = 4` ist der Standard-Default). Mit `future` + `k = 4` spawnt jeder echte Übergang vier Hindsight-Kopien, jede mit einem anderen zufällig gewählten zukünftigen Zustand aus derselben Episode als Ersatzziel.

!!! tip "Empfohlener Default"
    Nutze `goal_selection_strategy="future"` mit `n_sampled_goal=4`. Das ist, was Stable-Baselines3, RL-Zoo und die meisten publizierten HER-Benchmarks nutzen, und es lohnt sich selten zu tunen, es sei denn, deine Aufgabe ist ungewöhnlich.

### Intuition zu jeder Strategie

- **`final`**: jedes relabelte Ziel ist das, was der Arm am Ende der Episode berührt hat. Einfach, aber verzerrt das Curriculum hin zu „bleib stehen, wo du zufällig stehst".
- **`future`**: für einen Übergang zur Zeit `t`, sample einen `t' > t` aus derselben Trajektorie und nutze `achieved_goal(s_{t'})` als Ziel. Damit wird jeder Übergang gefragt: „Mit der Aktion, die du zur Zeit `t` ausgeführt hast, hättest du Fortschritt zu diesem späteren Zustand gemacht, den du tatsächlich erreicht hast?" Die Antwort ist per Konstruktion ja — also bekommt die Value-Funktion dichtes, konsistentes Signal.
- **`episode`**: wie `future`, erlaubt aber `t' < t`. Weniger prinzipiell (du kannst kein Ziel in der Vergangenheit erreichen), funktioniert aber manchmal für nicht-temporale Zielräume.
- **`random`**: relabel mit irgendeinem Ziel aus dem ganzen Buffer. Maximale Vielfalt, aber die meisten dieser Ziele sind irrelevant für die aktuelle Trajektorie, also sind die meisten relabelten Übergänge uninformativ.

---

## 6 · Hands-on: HER mit gymnasium-robotics

Wir wärmen uns in Python auf, bevor wir Godot anfassen. Das `gymnasium-robotics`-Paket liefert die Fetch-Suite — einen MuJoCo-basierten 7-DOF-Arm, dessen Observation-Space bereits ein ziel-bedingtes Dict ist, perfekt abgestimmt auf Stable-Baselines3' `HerReplayBuffer`.

### Installation

```bash
conda activate godot_env
pip install gymnasium-robotics
```

Hast du die MuJoCo-Bindings noch nicht installiert (transitive Abhängigkeit), folge den Prompts; auf macOS und Linux geht es automatisch.

### Vollständiges funktionierendes HER-Training auf FetchReach

```python
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

# FetchReach: 3-DOF arm, reach a target position
# obs includes: observation (25-dim), achieved_goal (3-dim), desired_goal (3-dim)
env = gym.make("FetchReach-v3")

model = SAC(
    "MultiInputPolicy",    # handles dict obs with obs + goal keys
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,       # k — relabel each transition 4 times
        goal_selection_strategy="future",
    ),
    verbose=1,
    tensorboard_log="logs/",
    learning_rate=1e-3,
    buffer_size=1_000_000,
    learning_starts=1000,
    batch_size=256,
    gamma=0.95,
    tau=0.005,
)

model.learn(total_timesteps=500_000, tb_log_name="her_fetchreach")
model.save("fetchreach_her")
```

### Was in TensorBoard zu beobachten ist

Öffne TensorBoard in einem anderen Terminal:

```bash
tensorboard --logdir logs/
```

Die Metriken, die zählen:

- `rollout/success_rate` — sollte von 0 auf ≈ 0,95 über das Training steigen.
- `rollout/ep_rew_mean` — Fetch-Belohnungen sind `-1` pro Step bis zum Erfolg, also steigt das von `-50` (Episoden-Timeout) Richtung `0`.
- `train/critic_loss` — sollte *nicht* flach auf null sein. Ist sie das, produziert HERs Relabeling keine positiven Belohnungen (prüfe `compute_reward`).

### Erwartetes Ergebnis

FetchReach (3-DOF Reach) wird in ~200 k Steps mit HER gelöst. Ohne HER (setze `replay_buffer_class=None`) braucht SAC typisch 2–5× mehr Steps, und bei den härteren Aufgaben unten scheitert es komplett.

!!! check "Fertig, wenn"
    `rollout/success_rate` auf FetchReach löst sich vom Boden und erreicht ≈ 0,95 innerhalb von ~200 k Steps, während `rollout/ep_rew_mean` von `-50` Richtung `0` steigt. Lässt du zusätzlich die No-HER-Baseline (`replay_buffer_class=None`) mit demselben Budget laufen, sollte ihre Kurve sichtbar darunter liegen — ohne HER braucht SAC 2–5× mehr Steps. Eine `success_rate`, die nach dem vollen Budget immer noch flach bei null liegt, ist kein „länger trainieren"-Problem: prüfe zuerst, dass `train/critic_loss` nicht flach auf null ist (falls doch, erzeugt das Relabeling keine positiven Belohnungen), und arbeite dann die Checkliste in Abschnitt 11 durch, beginnend mit dem Bit-für-Bit-Konsistenzcheck von `compute_reward`.

### Eine härtere Aufgabe: FetchPush

```python
env = gym.make("FetchPush-v3")
# Same code, just change the env — HER handles both
model.learn(total_timesteps=1_000_000)
```

FetchPush ersetzt das Leere-Luft-Ziel durch einen Würfel, der zum Ziel **geschoben** werden muss. Der Arm muss Kontakt herstellen, Reibung muss ausgerichtet sein und der Würfel muss gleiten. Ohne HER ist diese Aufgabe in unter 10 M Steps im Wesentlichen unlernbar; mit HER löst sie sich in ≈ 1 M.

---

## 7 · HER in Godot (ziel-bedingter Wrapper)

Jetzt zurück zu deinem Godot-Roboter. Die Godot-Env aus Unit Robotik emittiert flache Float-Observations durch die godot-rl-Bridge. Um HER zu füttern, müssen wir diesen flachen Vektor in die Dict-Struktur umpacken, die `HerReplayBuffer` erwartet: `{"observation", "achieved_goal", "desired_goal"}`.

### Erforderliche Konvention auf der Godot-Seite

In deinem `get_obs()` einigst du dich, einen flachen Vektor mit bekanntem Layout zu emittieren:

```
[robot_state (obs_dim)] [achieved_goal (3)] [desired_goal (3)]
```

`achieved_goal` ist `end_effector.global_position` (normiert); `desired_goal` ist `goal.global_position` (normiert). Der Wrapper unten teilt das für SB3 auf.

### Der Wrapper

```python
# Godot → HER compatible wrapper
import gymnasium as gym
import numpy as np
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

class GoalConditionedGodotEnv(gym.Env):
    """Wraps a Godot env that returns [robot_state (n), achieved_goal (3), desired_goal (3)]."""

    def __init__(self, env_path, obs_dim=12, goal_dim=3):
        self.env = StableBaselinesGodotEnv(env_path=env_path, n_parallel=1, speedup=20)
        self.obs_dim   = obs_dim
        self.goal_dim  = goal_dim
        total_dim = obs_dim + goal_dim * 2

        self.observation_space = gym.spaces.Dict({
            "observation":    gym.spaces.Box(-np.inf, np.inf, (obs_dim,)),
            "achieved_goal":  gym.spaces.Box(-np.inf, np.inf, (goal_dim,)),
            "desired_goal":   gym.spaces.Box(-np.inf, np.inf, (goal_dim,)),
        })
        self.action_space = self.env.action_space

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (distance < 0.05).astype(np.float32) - 1.0  # -1/0 sparse reward

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        obs_dict = self._split_obs(obs)
        reward = self.compute_reward(obs_dict["achieved_goal"], obs_dict["desired_goal"], info)
        return obs_dict, reward, done, info

    def _split_obs(self, flat_obs):
        robot  = flat_obs[:self.obs_dim]
        achieved = flat_obs[self.obs_dim:self.obs_dim + self.goal_dim]
        desired  = flat_obs[self.obs_dim + self.goal_dim:]
        return {"observation": robot, "achieved_goal": achieved, "desired_goal": desired}
```

### Warum `compute_reward` eine Methode sein muss

Das ist das mit Abstand wichtigste Detail bei der Integration von HER. Der `HerReplayBuffer` ruft `env.compute_reward(achieved_goal, desired_goal, info)` von **außen** auf, mit vektorisierten Arrays relabelter Ziele — *nicht* mit dem Ziel, das der Agent tatsächlich gesehen hat. Deine Belohnung muss daher:

1. Nur von `(achieved_goal, desired_goal, info)` abhängen, nie vom verborgenen Umgebungszustand.
2. Vektorisiert sein — gebatchte Zielarrays akzeptieren und eine gebatchte Belohnung zurückgeben.
3. Genau das matchen, was die Umgebung während `step` zurückgibt — sonst sind die relabelten Übergänge inkonsistent und das Training divergiert.

### Trainings-Aufruf

```python
env = GoalConditionedGodotEnv(env_path="builds/arm.exe", obs_dim=12, goal_dim=3)
model = SAC(
    "MultiInputPolicy", env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
    verbose=1, tensorboard_log="logs/",
    learning_rate=1e-3, buffer_size=1_000_000,
    learning_starts=1000, batch_size=256, gamma=0.95, tau=0.005,
)
model.learn(total_timesteps=500_000, tb_log_name="her_godot_arm")
```

---

## 8 · Über das Reichen hinaus: FetchPush, FetchPickAndPlace, FetchSlide

Umgebungen zu wechseln, während derselbe Trainingscode bleibt, ist eine der Freuden ziel-bedingten RL. Hier die Standard-Schwierigkeitsstufung:

| Aufgabe | Schwierigkeit | Was schwer ist |
|------|------------|-------------|
| `FetchReach-v3` | Leicht | Arm erreicht leeren Raum |
| `FetchPush-v3` | Mittel | Muss ein Objekt kontaktieren und schieben |
| `FetchPickAndPlace-v3` | Schwer | Greifen (Greifer schließen) + Heben + Platzieren |
| `FetchSlide-v3` | Schwer | Objekt schieben, das auf reibungsfreier Oberfläche gleitet |

Alle vier nutzen denselben HER-Code aus §6 — nur der Umgebungsname ändert sich. Sample-Budgets variieren:

- FetchReach: ≈ 200 k Steps
- FetchPush: ≈ 1 M Steps
- FetchPickAndPlace: 1–2 M Steps
- FetchSlide: 1–2 M Steps (oft am schlimmsten, weil das Schieben eines gleitenden Pucks präzises Kontaktiming erfordert)

Diese Zahlen nehmen `n_sampled_goal=4`, `future`-Strategie und eine einzige Umgebung an. Mit vektorisierten Umgebungen (z. B. 8 parallelen Sims) sinkt die Wand-Clock-Trainingszeit ungefähr linear.

---

## 9 · Wann HER hilft (und wann nicht)

### HER funktioniert, wenn…

- **Die Aufgabe ziel-bedingt ist.** Das Ziel ändert sich pro Episode und ist Teil der Observation.
- **Belohnung spärlich und binär ist.** Erfolg oder Fehlschlag, kein Shaping.
- **Belohnung allein Funktion von `(achieved_goal, desired_goal)` ist.** Das macht Relabeling stichhaltig — jede Trajektorie, die einen Zustand erreicht, kann als Erfolg für diesen Zustand relabelt werden.

### HER hilft NICHT, wenn…

!!! warning "HER hilft bei Dichte-Reward-Aufgaben nicht"
    Hast du bereits einen nützlichen dichten Reward (z. B. `-distance(end_effector, goal)`), liefert HER wenig zusätzliches Signal — der Agent bekommt schon Gradient bei jedem Step. Schlimmer noch, Dichte-Reward mit HER-Relabeling zu mischen, kann inkonsistente Reward-Werte für dasselbe `(state, action, next_state)`-Triple über den Buffer ergeben, was den Critic destabilisiert. **Nutze HER mit spärlichen Belohnungen oder gar nicht.**

Weitere Fehlermodi:

- **Trivialer Zielraum.** Existiert nur ein Ziel, kollabiert Hindsight-Relabeling zum Originalproblem.
- **Belohnung hängt vom Pfad, nicht vom Endpunkt ab.** Enthält die Belohnungsfunktion Terme wie „Minimum Jerk" oder „entlang der Trajektorie aufgewendete Energie", kannst du sie nicht aus `(achieved_goal, desired_goal)` allein neu berechnen. HERs relabelte Belohnungen wären inkorrekt.
- **Nicht-stationäre Zielsemantik.** Bedeutet „Position X erreichen" in verschiedenen Episoden Unterschiedliches (z. B. weil sich Hindernisse bewegen), lehren die relabelten Übergänge die falsche Lektion.

---

## 10 · Stretch Goals

1. **Löse FetchPickAndPlace.** Die schwerste Standard-Fetch-Aufgabe. Budget 1–2 M Steps mit HER. Dann HER deaktivieren (`replay_buffer_class=None`) und für dasselbe Budget laufen lassen; `success_rate`-Kurven vergleichen. Der Abstand ist üblicherweise der Unterschied zwischen „fast 1,0" und „genau 0,0".
2. **Multi-Goal-Training in Godot.** Spawne 5 Kandidaten-Ziel-Marker pro Episode. Wähle einen als gewünschtes Ziel, aber zeichne die erreichte Endeffektor-Position auf. Nutze HER mit `future`-Strategie — und beobachte, dass die Policy ohne Nachtraining auf jeden der 5 Marker generalisiert.
3. **Lies das Original-HER-Paper.** [Andrychowicz et al. 2017, „Hindsight Experience Replay"](https://arxiv.org/abs/1707.01495). Nur 10 Seiten, und die Einleitungsanalogie eines Kindes, das einen Eishockey-Puck schieben lernt, ist eines der besten Motivationsbeispiele in der RL-Literatur.
4. **Implementiere HER von Grund auf.** Subclasse `ReplayBuffer` in SB3 und schreibe dein eigenes `sample()`, das relabelte Übergänge einspeist. Du wirst die Eleganz nach wenigen Zeilen schätzen.
5. **Kombiniere HER mit Curiosity.** Für wirklich spärliche + explorationsharte Aufgaben (z. B. Ziele hinter Türen erreichen), kombiniere HER mit RND aus Unit Curiosity. Beide adressieren unterschiedliche Fehlermodi.

---

## 11 · Einen HER-Lauf debuggen

Auch mit dem richtigen Algorithmus hat HER-Training charakteristische Fehlermodi. Hier eine Checkliste, wenn Training nicht konvergiert.

### Die Erfolgsrate bleibt bei null

- **Ursache #1: `compute_reward` ist inkonsistent.** Drucke die Belohnung, die `step()` zurückgibt, und vergleiche sie mit `compute_reward(achieved, desired, info)`, aufgerufen mit denselben Argumenten. Sie müssen bit-für-bit übereinstimmen. Unterscheiden sie sich, passen HERs relabelte Belohnungen nicht zu den tatsächlichen Belohnungen des Agenten, und das Lernen bricht still ab.
- **Ursache #2: Erfolgsschwelle zu eng.** Eine 1-cm-Schwelle auf einem 1-m-Arm ist grenzwertig unerreichbar. Im Training auf 5 cm lockern, bei Evaluation enger ziehen.
- **Ursache #3: Zielsampling außerhalb des erreichbaren Arbeitsraums.** Erzeugt `randf_range` Ziele jenseits der kinematischen Reichweite des Arms, kann kein Relabeling helfen.

### Der Critic-Loss explodiert

- **Ursache: Reward-Magnituden-Mismatch mit `gamma`.** Bei spärlichen `{-1, 0}`-Rewards und `gamma=0.99` kann die Q-Funktion Werte bis `-100` annehmen. Stelle sicher, dass dein Netzwerk-Ausgabeumfang und die Lernrate diese Magnitude bewältigen. Das empfohlene `gamma=0.95` in §6 deckelt die Magnitude auf `≈ -20`, was freundlicher ist.

### Der Agent lernt zu reichen, aber nie zu greifen (FetchPickAndPlace)

- **Ursache: HER kann Greifen nicht relabeln.** Das erreichte Ziel ist die Würfelposition, aber Greifen erfordert Greifer-Öffnen + Absenken + Greifer-Schließen + Heben. Hindsight-Relabeling sagt dem Agenten: „Wenn du gebeten worden wärst, den Würfel dorthin zu setzen, wo er gerade ist, hast du Erfolg gehabt" — was trivialerweise wahr ist, wenn sich der Würfel nie bewegt hat. Lösungen: `n_sampled_goal` auf 8 erhöhen, ein Curriculum nutzen, das den Würfel für die ersten 100 k Steps bereits im Greifer initialisiert, oder mit Demonstrationsdaten kombinieren (DDPG+HER+Demonstrations, Nair et al. 2018).

---

## Was kommt als Nächstes

Du hast jetzt einen Algorithmus, der eine einzige, universelle Policy für jedes erreichbare Ziel trainieren kann — spärliche Belohnungen und alles. In der nächsten Unit triffst du auf die letzte Mauer zwischen Simulation und der echten Welt: **Sim-to-Real-Transfer**. Eine Policy, die FetchPush in MuJoCo oder einer Godot-Szene löst, funktioniert selten auf einem physischen UR5-Arm out-of-the-box. Wir betrachten Domain Randomization, Systemidentifikation und Observation-Noise-Injektion, um diese Lücke zu überbrücken.

Bevor du weitermachst, stelle sicher, dass du beantworten kannst:

- Warum versagt spärliches ziel-bedingtes RL mit vanilla SAC, und warum repariert HER das?
- Welche drei Komponenten muss die `compute_reward` deiner Umgebung erfüllen, damit HER funktioniert?
- Warum ist HER inkompatibel mit PPO?
- Wann ist HER das falsche Werkzeug und wonach solltest du stattdessen greifen?

[→ Sim-to-Real-Transfer](unit-sim-to-real.md)
