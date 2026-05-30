# GPU-beschleunigte RL-Umgebungen — forschungsskalige Alternativen zu Godot

!!! info "Zeit"
    Lesen: ~20 min

[Kursstartseite](index.md)

---

!!! info "Drei Wege, den Unterschied zu sehen"
    - **Godot** — öffne den Task Manager während BallChase mit 16 parallelen Envs trainiert. Beobachte, wie alle CPU-Cores auf 100 % geheftet sind, während die GPU nahezu untätig ist.
    - **TensorBoard** — bemerke, dass deine `rollout/ep_rew_mean`-Kurve glatt, aber langsam ist. Diese Glätte kostet Wall-Clock-Zeit.
    - **Diese Unit** — eine Karte jedes großen GPU-Env-Frameworks, wann sich der Setup-Aufwand für welches lohnt, und ein lauffähiges EnvPool + SB3-Beispiel, das du heute laufen lassen kannst.

---

## 1 · Warum CPU-Umgebungen ein Bottleneck sind (und wann nicht)

Jeder Godot-Trainingsschritt berührt mehr Maschinerie als das neuronale Netzwerk-Update.

Ein einzelner Umgebungsschritt in Godot beinhaltet:

1. Der Python-Trainingsprozess ruft `env.step(action)` über eine WebSocket-Verbindung auf
2. Die Godot-Engine führt ihren Physik-Tick aus (GodotPhysics oder Jolt)
3. Godot rendert die Szene, selbst im Headless-Modus
4. Die Beobachtung wird serialisiert, über den WebSocket zurückgesendet und in Python deserialisiert
5. Dein Python-Code batcht Beobachtungen und leitet sie für Inference an die GPU weiter
6. Die GPU gibt Aktionen zurück, die über WebSocket zu Godot zurückreisen

Die GPU ist nur in Schritt 5 beschäftigt. Alles andere ist CPU + IPC-Latenz.

**Mit godot-rl-agents' 20×-Zeit-Speedup und 16 parallelen Umgebungen kannst du auf einem modernen Desktop ungefähr 30–50 k Schritte pro Sekunde erreichen.** Das klingt schnell, bis du nachprüfst, was forschungsskaliges Training tatsächlich erfordert.

| Benchmark | Ungefähre Schritte zur Konvergenz |
|-----------|----------------------------------|
| BallChase (dieser Kurs) | 500 k – 2 M |
| Humanoid Walk (MuJoCo) | 10 M – 50 M |
| Geschickte Hand-Manipulation | 100 M – 500 M |
| NeurIPS Locomotion-Baselines | 1 Mrd. – 10 Mrd. |

Bei 40 k Schritten/Sek dauert 1 Mrd. Schritte ungefähr **7 Stunden Wall-Clock-Zeit**. Bei 5 M Schritten/Sek (Brax auf einer einzelnen GPU) dauern dieselben 1 Mrd. Schritte etwa **3 Minuten**.

!!! warning "Aber dieser Kurs braucht das nicht"
    Jeder Task in diesem Kurs konvergiert komfortabel in der linken Spalte: deutlich unter 10 M Timesteps. Bei 40 k Schritten/Sek sind das höchstens ein paar Stunden, oft unter 30 Minuten. **CPU-Umgebungen sind das richtige Werkzeug für diesen Kurs.** Diese Unit existiert, damit du weißt, wohin du gehen kannst, wenn du sie überholt hast.

### Entscheidungstabelle: Brauchst du GPU-Umgebungen?

| Situation | Empfehlung |
|-----------|------------|
| Tasks dieses Kurses trainieren | **Godot CPU — du bist okay** |
| Gesamt-Budget < 50 M Timesteps | **Godot CPU — du bist okay** |
| Ergebnisse mit einem SOTA-Paper vergleichen | Vielleicht — prüfe, welches Framework das Paper nutzte |
| Ablationen über 20+ Hyperparameter-Configs laufen lassen | EnvPool für den Parallelitäts-Boost erwägen |
| Eine NeurIPS-Submission schreiben | Fast sicher ja |
| Einen Agenten verschicken, der in einem Godot-Spiel lebt | **Godot — nichts anderes ergibt Sinn** |

---

## 2 · Die vier Hauptoptionen

### Isaac Gym / Isaac Lab (NVIDIA)

Isaac Gym simuliert Tausende von Rigid-Body-Umgebungen gleichzeitig auf einer einzelnen GPU, indem es den gesamten Physik-Zustand im GPU-Speicher hält und ihn zwischen Schritten nie zur CPU kopiert. Isaac Lab ist der neuere, modulare Nachfolger auf Basis von Isaac Sim.

**Am besten für:** Robot Manipulation, Legged Locomotion, jeden Task, der sauber auf Rigid-Body-Physik abbildet — ähnlich den Phase-6-Robotics-Units dieses Kurses.

**Hardware-Anforderung:** NVIDIA-GPU (RTX oder Data-Center-Klasse). AMD-GPUs werden nicht unterstützt.

```bash
# Isaac Gym requires a manual download from NVIDIA's developer portal.
# After downloading isaacgym-1.0rc4.tar.gz:
pip install isaacgym/
# Isaac Lab (open-source, actively maintained):
pip install isaaclab
```

Eine minimale PPO-Trainingsschleife mit Isaac Lab sieht so aus:

```python
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from stable_baselines3 import PPO

# Isaac Lab environments return GPU tensors directly.
# The VecEnv wrapper converts them to numpy for SB3.
from isaaclab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper

cfg = ManagerBasedRLEnvCfg()          # task-specific config
cfg.scene.num_envs = 2048             # 2048 envs on one GPU
isaac_env = ManagerBasedRLEnv(cfg=cfg)
env = Sb3VecEnvWrapper(isaac_env)

model = PPO("MlpPolicy", env, verbose=1, n_steps=32, batch_size=512)
model.learn(total_timesteps=100_000_000)
```

!!! warning "Limitierungen"
    - Setup ist aufwendig: spezifische CUDA-Version, spezifischer Treiber, spezifische Python-Version. Rechne mit einem Nachmittag für die Installation.
    - Headless-Rendering ist begrenzt — du kannst Metriken loggen, aber visuelles Debugging ist schwerer.
    - Isaac Gym (die ältere API) wird zugunsten von Isaac Lab deprecated. Bevorzuge Isaac Lab für neue Projekte.

---

### Brax (Google JAX)

Brax implementiert eine vollständig differenzierbare Physik-Engine in JAX. Weil JAX Berechnungsgraphen für GPU und TPU JIT-kompiliert, kannst du 4.096+ parallele Umgebungen vollständig auf dem Beschleuniger ohne CPU-Beteiligung zwischen Schritten laufen lassen.

**Am besten für:** Algorithmus-Forschung, wo End-to-End-Differenzierbarkeit zählt, oder TPU-skalige Experimente auf Google Cloud.

```bash
pip install brax
```

Eine PPO-artige Trainingsschleife mit Brax:

```python
import jax
import jax.numpy as jnp
from brax import envs
from brax.training.agents.ppo import train as ppo_train

# "ant" runs 4096 envs in parallel on GPU/TPU — all in JAX.
make_inference_fn, params, metrics = ppo_train(
    environment=envs.get_environment("ant"),
    num_timesteps=50_000_000,
    num_evals=10,
    reward_scaling=10,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=4096,
    batch_size=2048,
    seed=0,
)
```

!!! info "Brax und SB3"
    Brax liefert eigene PPO- und SAC-Implementierungen, die eng an JAX gekoppelt sind. Du kannst eine Brax-Umgebung über `brax.io.torch` für SB3 wrappen, aber du verlierst den meisten Geschwindigkeitsvorteil, weil Daten zwischen JAX (GPU) und PyTorch (GPU) bei jedem Schritt bewegt werden müssen. Für Brax sind die nativen Trainingsschleifen der vorgesehene Pfad.

!!! warning "Limitierungen"
    - Physikgenauigkeit ist vereinfacht gegenüber MuJoCo. In Brax gelernte Locomotion-Verhalten transferieren möglicherweise nicht gut auf Hardware.
    - JAX hat eine Lernkurve, wenn du reiner PyTorch-Nutzer bist.
    - Differenzierbarkeit ist mächtig, zählt aber meist nur für Gradient-Through-Sim-Forschung, nicht für standardmäßiges Model-free-RL.

---

### EnvPool

EnvPool ist keine GPU-Physik-Engine. Es ist ein **C++-multi-threaded Environment Pool**, der Pythons `multiprocessing`-basierten `SubprocVecEnv` ersetzt. Der Speedup kommt aus der Eliminierung von Python-GIL-Overhead und Inter-Prozess-Kommunikations-Serialisierung, nicht aus dem Verlagern von Physik zur GPU.

**Am besten für:** Atari-skalige Forschung, klassisches MuJoCo und DMControl — überall, wo du mehr CPU-Parallelität ohne die Komplexität eines vollen GPU-Physik-Stacks willst.

```bash
pip install envpool
```

EnvPool ist das einfachste Upgrade gegenüber dem bestehenden SB3-Workflow des Kurses. Siehe Abschnitt 6 für ein vollständiges, lauffähiges Beispiel.

!!! info "SB3-Kompatibilität"
    EnvPool-Umgebungen implementieren das Gymnasium-`VectorEnv`-Interface. Ein dünner Wrapper (in Abschnitt 6 gezeigt) macht sie vollständig kompatibel mit SB3s `learn()`-Aufruf.

---

### MJX (MuJoCo XLA)

MJX kompiliert die volle MuJoCo-Physik-Engine zu XLA (die gleiche Zwischendarstellung, die JAX nutzt), was MuJoCo-qualitative Rigid-Body-Simulation auf GPU und TPU ermöglicht. Anders als Isaac Gym erfordert es keine NVIDIA-Hardware — es läuft auf jedem XLA-Backend.

**Am besten für:** Forschung, die MuJoCos Simulations-Genauigkeit (Joint-Limits, Kontakt-Dynamik, Tendon-Constraints) in GPU-Skala braucht.

```bash
pip install mujoco mjx
```

```python
import jax
import mujoco
import mujoco.mjx as mjx

model = mujoco.MjModel.from_xml_path("humanoid.xml")
mx = mjx.put_model(model)       # upload model to GPU

# vmap over a batch of 2048 initial states
batch_step = jax.vmap(mjx.step, in_axes=(None, 0))
```

!!! info "MJX und SB3"
    Wie Brax ist MJX eine JAX-Bibliothek. Der Standard-Pfad ist, es mit einer JAX-nativen RL-Bibliothek zu paaren (z. B. Brax' Training-Utilities oder Rlax). SB3-Wrapper existieren in der Community, sind aber experimentell.

---

## 3 · Performance-Vergleich

Alle Zahlen unten sind **ungefähre Werte** und hängen stark von Task-Komplexität, Beobachtungsgröße, Netzwerk-Architektur und spezifischem GPU-Modell ab. Behandle sie als Größenordnungs-Schätzungen, nicht Benchmarks.

| Framework | Schritte/Sek (RTX 3090, ungefähr) | Typische Parallelität | Physik-Qualität | SB3-kompatibel? |
|-----------|-----------------------------------|----------------------|-----------------|------------------|
| Godot (dieser Kurs) | ~50 k | 8–32 (CPU-Cores) | Game-Physik (GodotPhysics / Jolt) | Ja, nativ |
| EnvPool — Atari | ~500 k | 64–1.024 | Emuliert (ALE) | Ja, dünner Wrapper |
| EnvPool — MuJoCo | ~200 k | 64–256 | Volles MuJoCo | Ja, dünner Wrapper |
| Isaac Gym / Isaac Lab | ~1 M | 2.048+ | Rigid Body (PhysX) | Partiell (offizieller Wrapper) |
| MJX | ~5 M | 2.048+ | Volles MuJoCo | Partiell (Community-Wrapper) |
| Brax | ~10 M | 4.096+ | Vereinfacht (Spring-basiert) | Partiell (custom loop empfohlen) |

!!! warning "Lies diese Zahlen sorgfältig"
    Die Brax- und MJX-Werte nehmen an, dass die gesamte Trainingsschleife (Umgebung + Policy + Gradient-Update) on-device in JAX läuft. Sie in SB3 zu wrappen senkt Performance typischerweise um 2–10× wegen GPU-zu-CPU-Datenbewegung.

---

## 4 · Wann Godot gewinnt

GPU-Env-Frameworks sind echt schneller beim Erzeugen von Trainings-Samples. Das heißt nicht, dass sie für jeden Use Case besser sind. Godot hat strukturelle Vorteile, die kein Physik-Engine-Rewrite ersetzen kann.

**Die Env IST das Spiel.** Wenn dein Agent in einem Godot-Spiel deployed wird, eliminiert Training in Godot die Sim-to-Real-Lücke komplett. Die Physik, das Rendering, die Level-Geometrie — alles, was der Agent zur Laufzeit sehen wird, ist exakt das, worauf er trainiert hat. Keine Domain-Randomisierung, kein Policy-Transfer, keine Überraschung.

**Reichhaltige visuelle Szenen mit Art-Assets.** Brax und Isaac Gym rendern einfache geometrische Primitive. Wenn dein Task ein gemalter Dungeon, ein stilisiertes Platformer oder ein custom-gerigter Charakter ist, ist Godot die einzige Option, die dir Training auf der tatsächlichen Kunst erlaubt.

**Iterativer Design-Loop.** Eine Level-Designerin kann eine Godot-Szene in Minuten modifizieren und das Training neu starten. GPU-Physik-Frameworks erfordern, dass URDF/MJCF-Dateien modifiziert, neu kompiliert und oft Belohnungsfunktionen neu geschrieben werden, um zur neuen Geometrie zu passen.

**Alles in diesem Kurs.** Keiner der Tasks dieses Kurses braucht mehr als ein paar Millionen Schritte. GPU-Envs würden keine bedeutsame Zeit sparen und würden signifikante Setup-Komplexität hinzufügen.

!!! tip "Faustregel"
    Wenn das Ziel ist, etwas zu verschicken, das in einem Godot-Spiel läuft, nutze Godot. Wenn das Ziel ist, ein Paper zu publizieren, das deinen Algorithmus mit MuJoCo-Baselines vergleicht, nutze dasselbe Framework, das die Baselines nutzten.

---

## 5 · Sim-to-Godot-Transfer

Manchmal ist die richtige Strategie, schnell auf einer schnellen Surrogat-Umgebung zu trainieren und dann in Godot zu fine-tunen oder zu evaluieren. Das ist eine Variante des Sim-to-Real-Transfers, angewendet auf Game-Engines.

**Was typischerweise übertragen wird:**

- High-Level-motorische Skills, die auf ähnlicher Kinematik gelernt wurden (z. B. MuJoCo HalfCheetah → Godot Biped Locomotion)
- Reward-Shaping-Intuitionen — ein Curriculum, das auf Brax Ant funktionierte, transferiert oft auf eine Godot-Quadruped
- Hyperparameter-Startpunkte — PPO-Clip-Range, GAE-Lambda und Learning-Rate-Schedules sind überraschend stabil über Physik-Engines hinweg

**Was typischerweise nicht übertragen wird:**

- Feinkörnige Kontakt-Dynamik — MuJoCo und Godot modellieren Reibung und Kontakt-Auflösung unterschiedlich
- Beobachtungs-Skala und -Range — Sensor-Messwerte bedeuten in verschiedenen Engines unterschiedliche Dinge; du wirst einen Kalibrierungs-Pass brauchen
- Visuelle Beobachtungen — eine auf MuJoCos einfachem Rendering trainierte Policy generalisiert nicht auf Godots Shader und Beleuchtung ohne Re-Training des visuellen Encoders

!!! info "Querverweis"
    Das breitere Thema des Umgangs mit der Lücke zwischen Trainings- und Deployment-Umgebungen wird in [unit-sim-to-real.md](unit-sim-to-real.md) behandelt, einschließlich Domain-Randomisierung, System-Identifikation und Real-Hardware-Deployment-Checklisten.

---

## 6 · EnvPool + SB3 Schnellstart (das einfachste Upgrade)

EnvPool ist das praktischste Upgrade vom SB3-Workflow dieses Kurses. Wenn du jemals Atari oder klassisches MuJoCo in Forschungsskala laufen lassen musst, ist das der Pfad des geringsten Widerstands.

**Warum ist EnvPool schneller als `SubprocVecEnv`?**

SB3s `SubprocVecEnv` spawnt N separate Python-Prozesse und kommuniziert mit ihnen über `multiprocessing.Pipe`. Jeder `step()`-Aufruf serialisiert das Aktions-Array mit `pickle`, sendet es durch die OS-Pipe, deserialisiert es im Worker-Prozess, läuft den Schritt, re-serialisiert die Beobachtung und sendet sie zurück. Für 64 Umgebungen sind das 128 Pickle-Round-Trips pro Trainingsschritt.

EnvPool macht all das in einem einzelnen C++-Thread-Pool. Es gibt kein Pickle, keine OS-Pipe und keine Python-GIL-Contention zwischen Umgebungen. Das Ergebnis ist ungefähr **10× höherer Durchsatz für dieselbe Anzahl paralleler Umgebungen**.

```python
import envpool
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn


class EnvPoolVecEnvWrapper(VecEnvWrapper):
    """Minimal wrapper to make an EnvPool VectorEnv compatible with SB3.

    EnvPool returns (obs, rew, terminated, truncated, info) in the Gymnasium
    step API. SB3 expects (obs, rew, done, info) in the old Gym API.
    This wrapper handles the conversion.
    """

    def reset(self) -> GymObs:
        obs, _ = self.venv.reset()
        return obs

    def step_wait(self) -> GymStepReturn:
        obs, rew, terminated, truncated, info = self.venv.step_wait()
        done = np.logical_or(terminated, truncated)
        # SB3 reads episode stats from info["terminal_observation"]
        # when done is True. EnvPool populates this automatically.
        return obs, rew, done, info


# ── Create 64 parallel Atari environments ─────────────────────────────────────
# EnvPool allocates a C++ thread pool — no subprocess overhead.
env = envpool.make(
    "Pong-v5",
    env_type="gymnasium",
    num_envs=64,
    seed=42,
    episodic_life=True,   # standard Atari training flag
    reward_clip=True,
)
env = EnvPoolVecEnvWrapper(env)

# ── Train with PPO ─────────────────────────────────────────────────────────────
# n_steps * num_envs = rollout buffer size = 128 * 64 = 8192 transitions
# This matches CleanRL's Atari PPO defaults closely.
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.01,
    learning_rate=2.5e-4,
    tensorboard_log="./runs/pong_envpool",
)
model.learn(total_timesteps=10_000_000)
model.save("ppo_pong_envpool")
```

!!! tip "Durchsatz-Sanity-Check"
    Nachdem der erste Rollout abgeschlossen ist, loggt SB3 `time/fps` in TensorBoard. Mit 64 EnvPool-Envs auf einer modernen Desktop-CPU solltest du **400 k – 600 k Schritte/Sek** für Atari sehen. Mit `SubprocVecEnv` und 64 Prozessen erreicht dieselbe Hardware typischerweise 40 k – 80 k Schritte/Sek — der EnvPool-Vorteil ist real.

!!! warning "EnvPool-Spieleliste"
    EnvPool unterstützt Atari (via ALE), klassisches MuJoCo (v4 und früher), DMControl und eine Handvoll anderer Umgebungen. Es unterstützt **keine** Custom-Umgebungen oder Godot. Für Custom-Tasks bleib bei `SubprocVecEnv` oder schau auf Isaac Lab / Brax.

---

## 7 · Stretch Goals

**Miss deinen eigenen SubprocVecEnv → EnvPool-Speedup.** Führe PPO auf `PongNoFrameskip-v4` zweimal aus — einmal mit dem SB3-`SubprocVecEnv`-Wrapper und 16 Prozessen, einmal mit `envpool.make_gymnasium("Pong-v5", num_envs=16)`. Nutze identische Hyperparameter und Step-Budget (z. B. 2 M Schritte). Erfasse `time/fps` aus TensorBoard und die Wall-Clock-Zeit zum Erreichen einer festen Belohnung. Das Verhältnis ist dein Speedup — und die Lücke zwischen der Marketing-Zahl und deiner Zahl ist das informativste Ergebnis.

**Wähle das richtige Werkzeug für einen Task.** Nimm eine *einzige* Umgebungs-Idee (deine, oder leih dir eine aus der Phase-4-Liste) und schreibe eine Ein-Absatz-Entscheidung: in welchem Framework — Godot, EnvPool, Isaac Lab oder Brax — würdest du sie trainieren, und warum? Referenziere Abschnitt 4 („Wann Godot gewinnt") explizit. Der Output ist eine Entscheidung, kein langes Memo — drei Sätze reichen.

**Profiliere einen Godot-Rollout.** Führe ein bestehendes Godot-Training (Unit 5 oder später) mit `n_parallel=8, speedup=32` aus und schaue auf `time/fps`. Dann öffne `htop` oder Activity Monitor in einem anderen Fenster. Ist der Engpass CPU, GPU oder Socket-Durchsatz? Vergleiche, was du siehst, mit den drei Failure-Modes aus Abschnitt 1. Es geht darum, *warum* deine spezifische Maschine langsam ist, bevor du nach einem anderen Framework greifst.

!!! warning "Pseudocode"
    ```python
    import envpool, time
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def make_subproc():
        import gymnasium as gym
        return SubprocVecEnv([lambda: gym.make("PongNoFrameskip-v4") for _ in range(16)])

    def make_envpool():
        return envpool.make_gymnasium("Pong-v5", num_envs=16)

    for name, factory in [("subproc", make_subproc), ("envpool", make_envpool)]:
        env = factory()
        model = PPO("CnnPolicy", env, n_steps=128, verbose=0)
        t0 = time.time()
        model.learn(total_timesteps=200_000)
        print(f"{name}: {200_000 / (time.time() - t0):.0f} steps/sec")
    ```

---

*Was kommt als Nächstes:* Wenn du neugierig bist, wie Transfer-Learning zwischen Physik-Engines in der Praxis funktioniert, lies [unit-sim-to-real.md](unit-sim-to-real.md). Wenn du die PPO-Implementierung verstehen willst, die in allen hier diskutierten Frameworks läuft, siehe [unit-cleanrl.md](unit-cleanrl.md).
