# Anwenden — SAC vs PPO auf JumperHard

[← SAC](unit-sac.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~10 min · Training: ~20 min GPU / ~1 h CPU pro Algorithmus

!!! info "Drei Wege, deine KI zu sehen"
    Godot (sieht die SAC-Policy glatter aus als die von PPO?) · TensorBoard (vergleiche die `ep_rew_mean`-Steigungen nebeneinander) · Stichprobeneffizienz (wie viele Umgebungsschritte bis zur selben Belohnung?)

!!! note "Voraussetzungen"
    - **[Einheit 4](unit-04.md)** — JumperHard läuft end-to-end mit PPO
    - **[SAC](unit-sac.md)** — Actor-Critic-mit-Entropie, Replay-Buffer, Off-Policy-Intuition (besonders §5 *PPO vs SAC — der Entscheidungsleitfaden*)

---

## 1 · Warum tauschen?

Du hast JumperHard in Einheit 4 mit PPO trainiert und im vorigen Kapitel die SAC-Theorie gelesen. Zeit, den Unterschied selbst zu sehen — auf einer Godot-Umgebung, die du bereits kennst.

PPO ist **on-policy**: Es sammelt einen Batch mit der aktuellen Policy, macht ein paar Gradientenschritte darauf, wirft den Batch weg und sammelt erneut. Einfach, robust, leicht zu parallelisieren. SAC ist **off-policy**: Jede Transition wandert in einen Replay-Buffer und wird für viele Gradienten-Updates wiederverwendet. Auf JumperHards kontinuierlichem Aktionsraum *kann* SACs Stichprobeneffizienz PPO dominieren — aber jeder Gradientenschritt ist teurer, und der Replay-Buffer frisst RAM.

Der Sinn dieses Zwischenspiels ist nicht, einen Gewinner zu küren. Es geht darum, den Trade-off an einem Projekt zu **erfühlen**, das du schon am Laufen hast.

---

## 2 · Das Trainingsskript anpassen

Öffne das Python-Trainingsskript, das du in Einheit 4 für JumperHard genutzt hast (`train_jumperhard.py` oder wie auch immer du es genannt hast). Die PPO-Version sieht ungefähr so aus:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="JumperHard.x86_64", show_window=False)
env = VecMonitor(env)

model = PPO(
    "MlpPolicy", env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.0,
    tensorboard_log="./tb_logs_ppo/",
    verbose=1,
)
model.learn(total_timesteps=200_000)
model.save("jumperhard_ppo")
```

Mach eine Kopie und tausche PPO gegen SAC:

```python
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecMonitor
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="JumperHard.x86_64", show_window=False)
env = VecMonitor(env)

model = SAC(
    "MlpPolicy", env,
    learning_rate=3e-4,
    buffer_size=200_000,        # replay buffer capacity (transitions)
    learning_starts=5_000,      # collect random data first
    batch_size=256,
    tau=0.005,                  # soft-update rate for target nets
    gamma=0.99,
    train_freq=1,               # one gradient step per env step
    gradient_steps=1,
    ent_coef="auto",            # automatic entropy temperature
    tensorboard_log="./tb_logs_sac/",
    verbose=1,
)
model.learn(total_timesteps=200_000)
model.save("jumperhard_sac")
```

Beachte, was verschwunden ist (`n_steps`, `n_epochs`, `clip_range`) und was hinzukam (`buffer_size`, `learning_starts`, `tau`, `train_freq`, `gradient_steps`, `ent_coef`). Das ist nicht derselbe Algorithmus unter anderem Namen — jede Zeile oben steht für eine andere Optimierungsgeschichte.

!!! warning "Replay-Buffer-Speicher"
    `buffer_size=200_000` mit einem kleinen Beobachtungsvektor (JumperHard hat ~ein paar Dutzend Floats) ist harmlos. Erhöhe den Buffer für Bildbeobachtungen, und du wirst es spüren: 200 k × 84×84×4 Bytes ≈ 5,6 GB. Der Buffer ist der Preis für Off-Policy.

---

## 3 · Beide trainieren

Zwei Terminal-Sitzungen, dieselbe Env-Konfiguration, anderer Algorithmus:

```bash
# Terminal 1
python train_jumperhard_ppo.py

# Terminal 2
python train_jumperhard_sac.py
```

Richte ein einzelnes TensorBoard auf beide Logverzeichnisse, damit du die Kurven überlagern kannst (erfordert TensorBoard ≥ 2.x — die Version, die SB3 aktuell pinnt):

```bash
tensorboard --logdir_spec ppo:./tb_logs_ppo,sac:./tb_logs_sac
```

Öffne jetzt `localhost:6006` und beobachte `rollout/ep_rew_mean` für beide Läufe gleichzeitig.

!!! check "Fertig, wenn"
    JumperHard hat keinen veröffentlichten Benchmark, also beurteile den Vergleich, nicht einen Score: (1) beide Läufe erscheinen als getrennte `rollout/ep_rew_mean`-Kurven unter den Tags `ppo` und `sac` in einem TensorBoard-Fenster, und (2) nachdem beide fertig sind, kannst du ein beliebiges Belohnungsniveau wählen, das beide Kurven erreicht haben, und sagen, welcher Algorithmus in weniger Env-Schritten dort ankam — und erklären warum, mit dem Replay-Buffer-Wiederverwendungs-Argument aus Abschnitt 1. Wenn einer der Läufe stirbt, bevor er eine Kurve produziert, behebe das, bevor du irgendetwas vergleichst.

---

## 4 · Was du sehen wirst

Erwartetes Verhalten auf JumperHard (das ist **erwartet**, keine Messung — deine Zahlen variieren mit Seed, Hardware und SB3-Version):

- **SACs `ep_rew_mean` steigt in weniger Umgebungsschritten.** Das ist Stichprobeneffizienz: SAC holt mehr aus jeder Transition heraus, weil der Replay-Buffer jede Transition zu vielen Updates beitragen lässt.
- **PPO gewinnt oft auf Wall-Clock-Zeit.** JumperHards Umgebungsschritt ist günstig, PPOs Gradientenschritt ist günstig, und PPOs Datenpfad ist einfacher. SACs Kosten pro Schritt (Gradientenschritt + Target-Netz-Update + Entropie-Temperatur-Update) fressen auf dieser Env seinen Stichprobeneffizienz-Vorteil in Wall-Clock-Zeit wieder auf.
- **SACs Policy kann in Godot glatter aussehen.** Kontinuierliche Aktionsverteilungen mit automatischem Entropie-Tuning produzieren oft weniger ruckelige Steuerung als eine geclippte PPO-Policy, die noch Entropie verliert.
- **SAC reagiert früh empfindlicher auf Hyperparameter.** `learning_starts` zu niedrig, und der Critic fittet Müll; `tau` zu hoch, und die Target-Netze oszillieren. PPOs Hyperparameter sind im Vergleich nachsichtig.

Wenn du nicht siehst, dass SAC dieselbe Belohnung in weniger Env-Schritten erreicht, prüfe `ent_coef` (das Auto-Tuning hat die Entropie vielleicht zu schnell verfallen lassen — probiere `ent_coef=0.2` fest) und `buffer_size` (zu klein heißt, der Buffer wird von veralteten Daten aus dem frühen Training dominiert).

---

## 5 · Wann du wirklich zu SAC greifen solltest

Ein kurzer Entscheidungsleitfaden, sobald das Experiment durch ist:

| Situation | Wähle |
|---|---|
| Kontinuierliche Aktionen, teure Simulation (echter Roboter, physiklastige Sim) | **SAC** — Stichprobeneffizienz zählt mehr als Wall-Clock |
| Günstige parallele Envs, diskrete oder kontinuierliche Aktionen | **PPO** — leichter zu skalieren, nachsichtiger |
| Du brauchst stabiles Training out of the box mit wenig Tuning | **PPO** |
| Du willst den Stand der Technik auf kontinuierlichen Steuerungs-Benchmarks vorantreiben | **SAC** (oder TD3) |
| Knappes RAM-Budget, kein Replay-Buffer möglich | **PPO** |

Du triffst SAC wieder in [Phase 6 — Fortbewegung](unit-locomotion.md) und [Sim-to-Real](unit-sim-to-real.md), wo seine Stichprobeneffizienz aufhört, eine Kuriosität zu sein, und essenziell wird.

---

## Stretch Goals

- **Wall-Clock vs Schritte.** Trainiere beide erneut mit `time` und plotte Wall-Clock-Sekunden gegen Umgebungsschritte. Übersetzt sich SACs Stichprobeneffizienz-Vorteil auf JumperHard in einen Wall-Clock-Vorteil? Warum oder warum nicht?
- **SAC auf CrossTheRoad.** Probiere das SAC-Skript auf der diskreten CrossTheRoad-Env aus Einheit 3. Es wird scheitern oder sich schlecht verhalten — finde heraus warum, bevor du die SAC-Doku liest.
- **Entropie-Temperatur-Sweep.** Trainiere SAC mit `ent_coef ∈ {0.05, 0.1, 0.2, "auto"}` und vergleiche. Worauf konvergiert der Auto-Tuner auf JumperHard?

---

## Was kommt als Nächstes?

Du hast PPO und SAC jetzt als **Anwender** gesehen — eine Algorithmusklasse wählen und der Bibliothek vertrauen. Als Nächstes schälst du eine Schicht ab: **CleanRL** reduziert PPO auf ~400 Zeilen Single-File-PyTorch, damit du jeden Gradientenschritt lesen kannst. Nützlich, wenn SB3 zu undurchsichtig zum Debuggen ist, wenn du einen eigenen Loss brauchst, oder wenn der Algorithmus aus einem Paper noch keine Bibliotheks-Implementierung hat.

[→ PPO von Grund auf (CleanRL)](unit-cleanrl.md)
