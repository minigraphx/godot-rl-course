# Unit 9 — Imitation Learning

Reward Engineering komplett überspringen. Nimm einen Experten beim Spielen auf, dann trainiere eine Policy, die **dieses Verhalten kopiert**. Untersuche **MultiLevelRobot**, nimm Demonstrationen auf, führe **Behavioral Cloning (BC)** aus und erweitere optional auf **GAIL**.

[← Multi-Task-RL](unit-multitask.md) · [Kursstartseite](index.md)

!!! note "Voraussetzungen"
    - **[Unit 4](unit-04.md)** — sicherer Umgang mit PPO-End-to-End-Training (wir fine-tunen *nach* BC)
    - **[Unit 6](unit-06.md)** — kontinuierliche Aktionen und Beobachtungsnormalisierung
    - Grundlegende Intuition für überwachtes Lernen (Loss-Minimierung, Train/Val-Split)
    - Kein GAN-/Adversarial-Training-Hintergrund nötig — §7 hält GAIL praktisch

!!! info "Zeit"
    Lesen: ~30 min · Training: ~20 min GPU / ~1,5 Std CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    Godot (selbst aufnehmen, dem Klon zuschauen) · TensorBoard (`train/loss` für BC; `ep_rew_mean` für GAIL) · Demonstrations-Replay: Schritt für Schritt durch deine eigenen aufgenommenen Aktionen

---

## 1 · Warum Imitation Learning?

Reward Shaping erfordert Iteration. Imitation Learning umgeht das Problem des Reward-Designs komplett: statt dem Agenten zu sagen, *was er maximieren soll*, zeigst du ihm, *was er tun soll*.

Zwei Hauptansätze:

| Methode | Wovon sie lernt | Algorithmus | Reward nötig? |
|--------|--------------------|-----------|----|
| **Behavioral Cloning (BC)** | Experten-Trajektorien | Überwachte Klassifikation/Regression | Nein |
| **GAIL** | Experten-Trajektorien | Adversarial (Diskriminator + Generator) | Nein (intrinsisch) |
| **DAgger** | Interaktive Experten-Korrekturen | Iteratives überwachtes Lernen | Teilweise |

**BC** ist das Einfachste: behandle jedes (Beobachtung, Aktion)-Paar als überwachtes Beispiel, trainiere eine Policy, die Experten-Aktionen vorhersagt. Schnell zu trainieren, anfällig für Distribution Shift.

**GAIL** trainiert einen Diskriminator, der Experten- von Agenten-Trajektorien unterscheidet, und nutzt dessen Output als intrinsischen Reward. Langsamer, aber robuster.

!!! note "Die Verbindung zu Alignment"
    BC auf Experten-Demonstrationen (Klonen menschlicher Aktionen aus Transitions) ist das direkte Analogon zu **Supervised Fine-Tuning (SFT)** im Alignment von Sprachmodellen — der konzeptuelle Einstieg zum Folgekurs über RL aus menschlichem Feedback.

---

## 2 · MultiLevelRobot

**MultiLevelRobot** ist ein 3D-Plattformer-Roboter, der über Plattformen unterschiedlicher Höhe navigieren muss. Reward Engineering ist heikel (kleine Plattformen, lange Fallhöhen). Es ist ein idealer Imitation-Learning-Kandidat: ein Mensch kann die Route leicht demonstrieren; der Agent kann sie durch zufällige Exploration nicht effizient entdecken.

1. Klone [godot_rl_agents_examples](https://github.com/edbeeching/godot_rl_agents_examples) → `examples/MultiLevelRobot`
2. In Godot .NET öffnen, Plugin aktivieren
3. Lies `ai_controller.gd`: Beobachtungen umfassen Körpergeschwindigkeit, Bodenraycasts und Plattformdistanzen; Aktionen sind kontinuierlich (Sprungkraft + Seitwärtsbewegung)

---

## 3 · Experten-Demonstrationen aufnehmen

Godot RL Agents unterstützt einen **menschlichen Heuristik-Modus**: setze `Control Mode` am Sync-Node auf `HUMAN` und spiele das Spiel dann selbst. Der Sync-Node nimmt jedes (obs, action)-Paar auf.

**Schritt für Schritt:**

1. Öffne `training_scene.tscn`
2. Wähle den `Sync`-Node → setze `Control Mode` auf `HUMAN`
3. Starte die Szene im Godot-Editor
4. Spiele mehrere komplette Episoden durch (ziele auf 20–50 erfolgreiche Durchläufe)
5. Stoppe die Szene — Demonstrationen werden in `demonstrations.json` gespeichert (oder dem in den Sync-Eigenschaften gesetzten Pfad)

```gdscript
# No code changes needed — the Sync node handles recording automatically
# Check Sync node properties for:
#   record_demonstrations = true
#   demonstrations_path = "res://demonstrations.json"
```

!!! tip "Qualität statt Quantität"
    20 hochwertige Demonstrationen (jedes Mal das Ziel erreichen) schlagen 200 mittelmäßige. Nimm neu auf, wenn du mehr als einmal pro Lauf von der Plattform gefallen bist.

---

## 4 · Die imitation-Bibliothek installieren

```bash
conda activate godot_env
pip install imitation
```

`imitation` bietet BC, GAIL, DAgger und AIRL auf Basis von Stable Baselines 3.

---

## 5 · Behavioral Cloning

```python
import numpy as np
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from imitation.algorithms import bc
from imitation.data import rollout
import json

# Load the environment (needed to define obs/action spaces)
env = StableBaselinesGodotEnv(
    env_path="./MultiLevelRobot.x86_64",
    n_parallel=1,
    speedup=1,
)

# Load recorded demonstrations
with open("demonstrations.json") as f:
    demo_data = json.load(f)

# Convert to imitation Transitions format
obs      = np.array([d["obs"]    for d in demo_data])
acts     = np.array([d["action"] for d in demo_data])
dones    = np.array([d["done"]   for d in demo_data])
next_obs = np.roll(obs, -1, axis=0)

transitions = rollout.Transitions(
    obs=obs[:-1],
    acts=acts[:-1],
    infos=np.array([{}] * (len(obs) - 1)),
    next_obs=next_obs[:-1],
    dones=dones[:-1],
)

# Build a PPO policy to clone into
policy = PPO("MlpPolicy", env, verbose=0)

# Behavioral Cloning trainer
trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    policy=policy.policy,
    rng=np.random.default_rng(42),
)

trainer.train(n_epochs=50)
policy.policy = trainer.policy
policy.save("multilevel_bc")
env.close()
```

```bash
conda activate godot_env
python train_bc.py
```

**Was zu erwarten ist:** Der Loss fällt in den ersten 10 Epochen schnell. Nach 50 Epochen ahmt die Policy die Bewegungen des Experten nach, scheitert aber unter Umständen an leicht veränderten Plattform-Layouts (Distribution Shift).

---

## 6 · Mit PPO nach BC fine-tunen

BC liefert einen starken Ausgangspunkt. Ein kurzer PPO-Fine-Tuning-Lauf behebt Distribution Shift oft:

```python
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(
    env_path="./MultiLevelRobot.x86_64",
    n_parallel=8,
    speedup=20,
)

# Load BC-initialized policy and continue with PPO
model = PPO.load("multilevel_bc", env=env)
model.learn(total_timesteps=500_000, tensorboard_log="logs/")
model.save("multilevel_bc_finetune")
env.close()
```

In TensorBoard sollte `ep_rew_mean` deutlich höher starten als bei einem zufällig initialisierten PPO-Lauf — die BC-Policy gibt dem Agenten einen Vorsprung in den nützlichen Teil des Zustandsraums.

---

## 7 · GAIL (optional)

GAIL trainiert einen Diskriminator parallel zur Policy. Der Diskriminator sagt vorher: „Stammt diese Trajektorie vom Experten oder vom Agenten?"; der Agent erhält einen Reward dafür, ihn zu täuschen.

```python
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3 import PPO

env = StableBaselinesGodotEnv(env_path="./MultiLevelRobot.x86_64", n_parallel=4, speedup=20)

learner = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/")
reward_net = BasicRewardNet(env.observation_space, env.action_space)

gail_trainer = GAIL(
    demonstrations=transitions,   # from section 5
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

gail_trainer.train(total_timesteps=1_000_000)
learner.save("multilevel_gail")
env.close()
```

GAIL ist langsamer, generalisiert aber besser. Verwende es, wenn BC-Fine-Tuning weiterhin an neuen Plattform-Anordnungen scheitert.

---

## 8 · Viz-Checkpoint

Neu laufen lassen mit dem Godot-Editor (Sync → `ONNX_INFERENCE` nach Export, oder die Szene öffnen und das Modell manuell laden):

- Folgt der Roboter der Route, die du demonstriert hast?
- Erholt er sich, wenn er leicht vom Pfad abrutscht (GAIL / Fine-Tune)? Oder friert er ein (reines BC)?
- Vergleiche: lass das reine BC-Modell, das fine-getunte Modell und eine PPO-from-scratch-Baseline nebeneinander laufen

Ein guter BC-Agent wirkt „menschlich" — er zögert an denselben Stellen, an denen du gezögert hast, nimmt denselben Weg. Driftet er in einen untrainierten Zustand, scheitert GAIL graziös; reines BC scheitert katastrophal.

---

## 9 · Stretch Goals

- **DAgger** — iterative Imitation: lass die Policy laufen, lass den Experten die neuen besuchten Zustände labeln, trainiere neu. Behebt Distribution Shift systematisch. Verfügbar in `imitation.algorithms.dagger`.
- **Datenseffizienz vergleichen** — wie viele Demonstrationen braucht BC, um mit 500k PPO-Schritten gleichzuziehen?
- **Mixed Reward** — kombiniere den intrinsischen GAIL-Reward mit einem sparsamen Environment-Reward (Ziel erreichen), um die Policy bei der Aufgabe zu halten

---

## Was kommt als Nächstes

**RLHF & Preference Learning:** Du hast gelernt, Verhalten aus Demonstrationen zu klonen. Was, wenn du nur den *Geschmack* eines Designers hast — keinen expliziten Reward, nur paarweise Präferenzen? RLHF macht menschliches Urteil zu einem Reward-Modell, das das Fine-Tuning der Policy leitet.

!!! info "Selbstcheck, bevor du weitermachst"
    Kannst du diese in eigenen Worten beantworten?

    1. Welches Problem löst **Imitation Learning**, das PPO + Reward Shaping nicht löst, und zu welchem Preis?
    2. Was ist **Distribution Shift** in Behavioral Cloning, und warum braucht es meist entweder Fine-Tuning oder DAgger, um ihn zu beheben?
    3. Warum ist BC überwachtes Lernen, GAIL aber Reinforcement Learning?
    4. Warum hilft Fine-Tuning mit PPO nach BC meist — was fügt PPO hinzu, das reine Imitation nicht kann?
    5. Wähle eine Godot-Umgebung aus früheren Units — würden Experten-Demos im Vergleich zu PPO-from-scratch *helfen* oder *schaden*, und warum?

    Wenn du alle fünf beantworten kannst — bist du bereit.

[→ RLHF & Preference Learning](unit-rlhf.md)
