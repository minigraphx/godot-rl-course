# World Models — modellbasiertes RL mit Dreamer

!!! info "Zeit"
    Lesen: ~40 min · Training: ~45 min GPU / ~3 h CPU

[Kursstartseite](index.md)

---

!!! info "Drei Wege, deine KI zu beobachten"
    **Godot** (beobachte, wie der Agent selbstbewusst agiert, selbst nachdem er nur einen Bruchteil der Umgebungsschritte gemacht hat, die eine PPO-Baseline brauchen würde — das World Model hat in der Imagination die Schwerstarbeit erledigt) · **TensorBoard** (`train/reward_mean` in Dreamer-Logs neben World-Model-Rekonstruktions-Loss — beide sollten sinken; bleibt der Rekonstruktions-Loss hoch, hat das Modell nicht gelernt, die Szene zu repräsentieren) · **Latent-Space-Plot** (projiziere gelernte Latent-States mit t-SNE oder PCA — klare Cluster für „nahe Wand", „nahe Ziel", „freier Raum" zeigen, dass das Modell Struktur entdeckt hat, die die Belohnungsfunktion nie explizit definiert hat)

---

## 1 · Model-free vs Model-based

Jeder Algorithmus in diesem Kurs bisher war **model-free**: der Agent sammelt Transitionen aus der realen Umgebung, berechnet einen Gradienten und updated seine Policy. Die Umgebung ist eine Black Box — der Agent versucht nie vorherzusagen, was sie als Nächstes tun wird. Das funktioniert, ist aber teuer. Ein PPO-Agent, der einen mäßig schweren Godot-Task löst, braucht routinemäßig Millionen von Umgebungsschritten bis zur Konvergenz.

**Model-based RL** geht eine andere Wette ein: investiere Compute, um ein Modell der Umgebung zu lernen, und nutze dann dieses Modell, um synthetische Erfahrung zu generieren. Ist das Modell genau, kann der Agent darin planen und lernen — er braucht weit weniger Interaktionen mit der realen Umgebung.

| Eigenschaft | Model-free (PPO, SAC) | Model-based (Dreamer) |
|---|---|---|
| Lernt Umgebungsmodell | Nein | Ja — Encoder, Dynamik, Decoder, Belohnung |
| Reale Env-Schritte bis Konvergenz | Hoch (Millionen) | Niedrig (Zehntausende–Hunderttausende) |
| Compute pro Schritt | Niedrig | Hoch — World-Model-Training |
| Gesamte Wall-Clock-Zeit | Kann vergleichbar sein | Langsamer pro Schritt, weniger Schritte |
| Handhabt Sparse Rewards | Schlecht | Besser — kann Sparse-Reward-Trajektorien imaginieren |
| Engineering-Komplexität | Niedrig | Hoch — mehr bewegliche Teile |
| Modellfehler | N/A | Können „Modell-Halluzinationen" verursachen — Agent nutzt ungenaues Modell aus |

### Wann zählt der Sample-Efficiency-Gewinn?

Der Trade-off begünstigt Model-based-Methoden, wenn **Umgebungsschritte teuer sind**:

- Simulation ist langsam (physiklastige Godot-Szene, 1 parallele Instanz)
- Du trainierst auf realer Hardware, wo jeder Schritt Zeit oder Verschleiß kostet
- Belohnung ist so spärlich, dass Model-free-Methoden Millionen von Schritten zufällig explorieren

Wenn Umgebungsschritte günstig sind — leichtgewichtige Simulation, 32 parallele Instanzen — konvergieren Model-free-Methoden oft schneller in Wall-Clock-Zeit, trotz mehr benötigter Schritte. Miss in Wall-Clock-Stunden, nicht nur in Timestep-Count.

!!! warning "Mehr bewegliche Teile heißt mehr Failure-Modes"
    Ein PPO-Trainingslauf kann auf wenige Arten scheitern (schlechte Learning Rate, Belohnungs-Skala). Ein Dreamer-Trainingslauf kann auf viele weitere scheitern: schlechte Rekonstruktion, Ungenauigkeit des Dynamik-Modells, KL-Kollaps, zu langer Imaginations-Horizont. Greife zu Model-based RL, wenn du einen klaren Grund hast — Sparse Rewards, teure Simulation oder den Wunsch zu planen. Nutze es nicht als Default-Upgrade über PPO.

---

## 2 · Was ein World Model lernt

Ein World Model ist eine Sammlung gelernter Funktionen, die zusammen dem Agenten erlauben, die Umgebung intern zu simulieren. Die vier Komponenten werden gemeinsam aus realer Erfahrung trainiert:

```
Encoder:           obs_t         →  z_t          (compress raw observation to latent state)
Dynamics model:    z_t + a_t     →  z_{t+1}      (predict next latent state)
Decoder:           z_t           →  obs_t         (reconstruct observation — used as training signal)
Reward predictor:  z_t           →  r_t           (predict reward from latent state)
```

### Warum im Latent-Space arbeiten?

Direkt auf Pixel-Beobachtungen (64×64 = 12.288 Werte) zu arbeiten ist teuer. Der Encoder komprimiert jedes Frame zu einem Latent-Vektor `z` von vielleicht 32 Dimensionen. Das Dynamik-Modell propagiert dann diese kleinen Vektoren durch imaginierte zukünftige Schritte — 15 Schritte in imaginierter Zeit kosten dem Dynamik-Modell 15 winzige Forward-Passes, statt 15 voller Umgebungs-Renderings.

Der Decoder rekonstruiert die ursprüngliche Beobachtung aus `z`. Seine Aufgabe wird zur Inferenzzeit nicht genutzt — aber während des Trainings ist der Rekonstruktionsfehler das Aufsichtssignal, das `z` zwingt, alles zu erfassen, was nötig ist, um die Beobachtung zu erklären. Ohne den Decoder gibt es keine Garantie, dass der Encoder etwas Bedeutsames lernt.

```
Real trajectory:
  obs_0 ─[Encoder]─▶ z_0 ─[Dynamics + a_0]─▶ z_1 ─[Dynamics + a_1]─▶ z_2
                       │                          │                          │
                   [Decoder]                  [Decoder]                 [Decoder]
                       ▼                          ▼                          ▼
                   recon_0                    recon_1                    recon_2

Training losses:
  reconstruction:  ||obs_t - recon_t||²  (per pixel, or per feature)
  reward:          ||r_t - reward_pred_t||²
  dynamics (KL):   KL( posterior(z_t | obs_t) || prior(z_t | z_{t-1}, a_{t-1}) )
                   ↑ DreamerV1 form, shown for intuition.
                   DreamerV3 uses a balanced KL with free bits:
                   KL_loss = max(KL, free_bits) — see §4 for details.
```

Alle vier Komponenten teilen Gradienten — das Dynamik-Modell zu verbessern verbessert den Encoder, weil bessere Repräsentationen zu niedrigerem Vorhersagefehler downstream führen.

!!! info "Posterior vs Prior Unterscheidung"
    Während des Trainings sieht der Encoder die echte nächste Beobachtung und kann eine *Posterior*-Verteilung über `z_{t+1}` berechnen. Zur Inferenzzeit (Imagination) ist keine reale Beobachtung verfügbar, also muss das Dynamik-Modell nur seinen *Prior* basierend auf vorigem Latent-State und Aktion nutzen. Der KL-Loss zwingt diese beiden Verteilungen, eng beieinander zu bleiben — divergieren sie, wird die imaginierte Zukunft inkonsistent mit dem sein, was der Encoder aus einer realen Beobachtung produziert hätte.

---

## 3 · Dreamer-Architektur — RSSM

Dreamer (Hafner et al. 2019, verfeinert in DreamerV2 und DreamerV3) führt das **Recurrent State Space Model (RSSM)** als seinen World-Model-Backbone ein. Das RSSM ist die architektonische Schlüssel-Einsicht, die Dreamer von einfacheren modellbasierten Ansätzen trennt.

### Der RSSM-Latent-State: h und z zusammen

Ein reines stochastisches Latent `z` (wie ein VAE) vergisst Geschichte zwischen Schritten — es sieht nur das aktuelle Frame. Ein reiner rekurrenter State `h` (wie ein LSTM-Hidden-State) hat keine explizite Unsicherheits-Repräsentation. RSSM kombiniert beides:

```
h_t  =  GRU(h_{t-1}, z_{t-1}, a_{t-1})   — deterministic recurrent hidden state
z_t  ~  posterior(z | h_t, obs_t)          — stochastic latent (during training)
     ~  prior(z | h_t)                     — stochastic latent (during imagination)

Full state:  s_t = concat(h_t, z_t)
```

| Komponente | Typ | Rolle |
|---|---|---|
| `h_t` | Deterministisch (GRU) | Trägt Langzeit-Memory über Schritte — was vorher passierte |
| `z_t` | Stochastisch (diagonal Gauß oder kategorisch) | Repräsentiert aktuelle Unsicherheit — was gerade mehrdeutig ist |
| `s_t = [h_t, z_t]` | Kombiniert | Voller State, der an Actor, Critic und Reward-Prädiktor übergeben wird |

**Warum beides?** Der GRU-Hidden-State `h` akkumuliert Information über viele Schritte — essenziell für partiell beobachtbare Tasks, wo ein einzelnes Frame nicht reicht. Die stochastische Variable `z` erlaubt dem Modell, echte Mehrdeutigkeit zu repräsentieren: mehrere Zukünfte, die alle konsistent mit vergangenen Beobachtungen sind. DreamerV2 wechselte von Gauß-`z` zu kategorischem (Straight-Through-Gradienten), was die Trainings-Stabilität verbessert.

### Policy-Training komplett in der Imagination

Actor und Critic werden **nie aus realen Umgebungsdaten geupdatet**. Sie werden komplett innerhalb imaginierter Rollouts trainiert:

```
1. Encode a real observation → s_0 = [h_0, z_0]
2. Roll out H = 15 steps in imagination:
     s_1, s_2, ..., s_H  using dynamics model + actor actions
3. Compute imagined rewards:  r̂_1, ..., r̂_H  using reward predictor
4. Compute λ-return (advantage):  Vλ = r̂ + γ · V(s_{t+1})
5. Update actor to maximise Vλ
6. Update critic to predict Vλ
```

Reale Umgebungsschritte werden nur genutzt, um das World Model zu trainieren (Encoder, Dynamik, Decoder, Reward-Prädiktor). Ist das World Model genau, können Actor und Critic verbessern, indem sie Millionen imaginierter Trajektorien generieren — jede davon ist im Wesentlichen kostenlos.

Deshalb erreicht Dreamer hohe Sample-Efficiency: der Agent macht „Hausaufgaben" in der Imagination zwischen realen Interaktionen, statt auf den nächsten realen Umgebungsschritt zu warten, um einen Gradienten zu bekommen.

```
Outer loop (real env):
  collect 50 real steps → add to replay buffer → train world model for K steps

Inner loop (imagination):
  sample starting states from replay → roll out H steps in imagination
  → update actor and critic from imagined returns
```

---

## 4 · DreamerV3

DreamerV3 (Hafner et al. 2023) ist die aktuelle Produktionsversion von Dreamer. Sein Headline-Ergebnis: ein einzelner Satz von Hyperparametern, der über so unterschiedliche Domains wie Atari, DeepMind Control Suite (DMC), Minecraft und Robot Manipulation funktioniert — ohne per-Domain-Tuning.

### Setup

```bash
pip install dreamerv3
```

DreamerV3 erfordert JAX. Auf einer Maschine mit einer CUDA-GPU:

```bash
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install dreamerv3
```

Auf CPU (langsamer, aber funktional zum Experimentieren):

```bash
pip install jax dreamerv3
```

### DreamerV3 auf einem Standard-Benchmark laufen lassen

```python
import dreamerv3
from dreamerv3 import embodied

# Standard DreamerV3 config — works out-of-the-box on DMC tasks
config = embodied.Config(dreamerv3.configs["defaults"])
config = config.update(dreamerv3.configs["medium"])  # medium compute tier

config = config.update({
    "logdir": "logs/dreamer_cheetah",
    "run.train_ratio": 32,      # imagination steps per real step
    "run.log_every": 300,
    "batch_size": 16,
    "jax.prealloc": False,
})

# DMC benchmark: HalfCheetah-v2
import gymnasium as gym
env = gym.make("dm_control/cheetah-run-v0")

# Wrap for dreamerv3
env = dreamerv3.wrap_env(env, config)

agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
replay = embodied.replay.Uniform(config.replay_size, config.replay_online)
embodied.run.train(agent, env, replay, config)
```

### Self-tuning Hyperparameter — was DreamerV3 änderte

Frühere Dreamer-Versionen erforderten per-Domain-Tuning. DreamerV3 führt zwei Mechanismen ein, die einen einzelnen Hyperparameter-Satz über sehr unterschiedliche Belohnungs-Skalen stabil machen:

**Symlog-Transformationen.** Alle skalaren Vorhersagen (Belohnung, Value, Return) durchlaufen:

```
symlog(x) = sign(x) · log(|x| + 1)
```

Das komprimiert große Werte und expandiert kleine symmetrisch um null. Eine Belohnung von +1000 und eine Belohnung von +1 bekommen beide vernünftige Gradient-Magnituden. Ohne Symlog überwältigen große Belohnungen kleine, und Gradienten explodieren; winzige Belohnungen produzieren verschwindende Gradienten.

**Free Bits.** Der KL-Loss ist geklemmt: `KL_loss = max(KL, free_bits)`. Das verhindert, dass das Modell den stochastischen Latent `z` zu einem deterministischen Punkt kollabieren lässt (Posterior-Kollaps). Dem Modell sind bis zu `free_bits` Nats KL „frei" erlaubt — es zahlt nur einen Strafterm darüber hinaus.

| DreamerV2 | DreamerV3 |
|---|---|
| Gauß-z | Kategorisches z (stabilere Gradienten) |
| Manuelles Reward-Scaling pro Domain | Symlog-Transformationen — selbst-normalisierend |
| KL-Tuning pro Domain | Free Bits — automatische Posterior-Regulation |
| Separate Configs für Atari, DMC | Eine einzelne Config für alle Domains |

---

## 5 · Godot-Integration

DreamerV3 akzeptiert jede Gymnasium-kompatible Umgebung. Die SubViewport-Pipeline aus [Visuelle Beobachtungen](unit-visual-observations.md) exponiert Godot als Standard-Pixel-Observation-Gym-Env — DreamerV3 kann direkt darauf trainieren.

### Godot mit DreamerV3 verbinden

```python
import dreamerv3
from dreamerv3 import embodied
import numpy as np
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

# 1. Load the Godot environment (pixel observations, see unit-visual-observations.md)
#    Your Godot scene must use SubViewport + 64x64 pixel obs (Section 2 of that unit)
godot_env = StableBaselinesGodotEnv(
    env_path="./VisualAgent.x86_64",
    n_parallel=1,          # DreamerV3 typically runs one env, trains in imagination
    speedup=4,
    show_window=False,
)

# 2. Reshape obs from flat array to (H, W, C) — DreamerV3 expects channels-last
import gymnasium as gym

class GodotDreamerWrapper(gym.ObservationWrapper):
    """Reshape flat Godot pixel obs [H*W*C] → (H, W, C) for DreamerV3."""

    def __init__(self, env, height=64, width=64, channels=3):
        super().__init__(env)
        self.h, self.w, self.c = height, width, channels
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(height, width, channels),  # channels-last for DreamerV3
            dtype=np.float32,
        )

    def observation(self, obs):
        return obs.reshape(self.h, self.w, self.c)

env = GodotDreamerWrapper(godot_env, height=64, width=64, channels=3)

# 3. Configure DreamerV3 for a pixel-observation task
config = embodied.Config(dreamerv3.configs["defaults"])
config = config.update(dreamerv3.configs["small"])  # smaller model for 64x64 Godot scenes
config = config.update({
    "logdir": "logs/dreamer_godot",
    "run.train_ratio": 16,   # 16 imagination steps per real step (lower for fast Godot sim)
    "encoder.mlp_keys":  "$^",   # no MLP encoder (pure pixel obs)
    "decoder.mlp_keys":  "$^",
    "encoder.cnn_keys":  "image",
    "decoder.cnn_keys":  "image",
    "batch_size": 16,
    "batch_length": 64,
})

# 4. Train
env = dreamerv3.wrap_env(env, config)
agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
replay = embodied.replay.Uniform(config.replay_size, config.replay_online)
embodied.run.train(agent, env, replay, config)
```

!!! tip "Starte mit einer schnellen Godot-Szene"
    DreamerV3 sammelt reale Schritte nur, um das World Model zu trainieren — es braucht keine Millionen. Aber das World-Model-Training selbst ist rechenintensiv. Nutze während der Entwicklung eine einfache 2D-Szene, in der du verifizieren kannst, dass der Rekonstruktions-Loss sinkt, bevor du dich zu einer komplexen 3D-Umgebung verpflichtest. Eine Szene, in der `reconstruction_loss` nach 10k Schritten nicht sinkt, zeigt an, dass der Encoder versagt, nicht die Policy.

### Hybrid: Pixel-Obs + propriozeptiver State

Für die meisten Godot-Tasks übertrifft eine hybride Beobachtung reine Pixel. Übergib sowohl Bild als auch State durch das World Model — DreamerV3 handhabt Dict-Beobachtungen nativ:

```python
# Godot GDScript (ai_controller.gd)
func get_obs() -> Dictionary:
    return {
        "image": _capture_frame(),     # flat pixel array
        "state": [velocity.x, velocity.z, dist_to_goal, heading_to_goal],
    }

# Python config: tell DreamerV3 which keys are images vs vectors
config = config.update({
    "encoder.cnn_keys": "image",   # convolutional encoder for image key
    "encoder.mlp_keys": "state",   # MLP encoder for state key
    "decoder.cnn_keys": "image",
    "decoder.mlp_keys": "state",
})
```

Das World Model lernt, beide Streams zu rekonstruieren. State-Vektoren sind typischerweise einfacher zu rekonstruieren als Pixel, was als zusätzliche Konsistenz-Beschränkung auf den Latent-Space wirkt.

---

## 6 · Wann World Models gewinnen

World Models bieten den größten Vorteil unter spezifischen Bedingungen. Bevor du dich zu Dreamer verpflichtest, prüfe, ob dein Task zu diesen Mustern passt:

**Sparse Rewards.** Wenn Belohnungen nur ein paar Mal pro Episode erscheinen, explorieren Model-free-Methoden zufällig und warten darauf, auf eine Belohnung zu stolpern. Das World Model kann Trajektorien imaginieren, die den Belohnungs-State erreichen — sobald er mindestens ein paar Mal angetroffen wurde — und die Policy wiederholt auf diesen imaginierten Erfolgen trainieren. Du bekommst effektives Lernen aus sehr wenigen realen Belohnungs-Events.

**Teure Simulation.** Wenn jeder Umgebungsschritt Sekunden dauert (komplexe Physik, reale Hardware), zahlt sich Compute auf das World Model aus. Zehn imaginierte Schritte kosten einen winzigen Bruchteil eines realen Schritts.

**Partielle Beobachtbarkeit.** Der RSSM-rekurrente Hidden-State `h` trackt explizit Geschichte. Eine Standard-MLP-Policy, die auf einem einzelnen Frame agiert, hat kein Memory; Dreamer baut automatisch eine kompakte Geschichts-Repräsentation auf.

**Planung und Look-Ahead.** Sobald du ein World Model hast, kannst du explizite Planungs-Algorithmen (MCTS, CEM) darin laufen lassen. Dreamer nutzt standardmäßig den Actor zum Planen, aber das World Model ist für deliberatere Suche verfügbar, falls nötig.

---

## 7 · Wann World Models verlieren

Klarsichtig zu sein, wann man Model-based RL nicht nutzen sollte, spart Wochen an Debugging:

**Kontaktreiche Manipulation und komplexe Physik.** Rigid-Body-Kontakt ist notorisch schwer genau zu modellieren. Ein World Model, das auch nur leicht falsch über Objekt-Kontakt-Dynamik ist, produziert imaginierte Trajektorien, die die reale Umgebung nie generiert. Die Policy trainiert auf halluzinierter Physik und versagt in der realen Szene. Model-free-SAC ist oft schneller zu einem funktionierenden Ergebnis für kontaktreiche Tasks.

**Wenn Datensammlung günstig ist.** Wenn du 32 parallele Godot-Instanzen laufen lassen kannst, sammelst du in Stunden Millionen von realen Schritten. Der Sample-Efficiency-Vorteil von Dreamer schrumpft — und der Engineering-Overhead (World-Model-Bugs, Rekonstruktions-Debugging) bleibt konstant. Im Maßstab gewinnt PPO oft auf gesamter Wall-Clock-Zeit, trotz mehr benötigter Schritte.

**Hochgradig stochastische Umgebungen.** Dreamers Dynamik-Modell lernt, den Mittelwert des nächsten States vorherzusagen. In Umgebungen mit starker Stochastizität (zufällige Hindernisse, prozedural generierte Layouts) ist der Vorhersagefehler irreduzibel. Das World Model bleibt ungenau; Policy-Lernen in der Imagination divergiert von Real-Umgebungs-Performance.

**Short-Horizon-Tasks.** Wenn eine Episode 50 Schritte ist und dichte Belohnungen jeden Schritt verfügbar sind, konvergieren Model-free-Methoden schnell. Der Vorteil des World Models — Komprimieren von Long-Horizon-Credit-Assignment — gilt nicht.

| Situation | Empfehlung |
|---|---|
| Sparse Reward, wenige Env-Schritte verfügbar | Dreamer — starke Passung |
| Dense Reward, günstige Simulation, viele parallele Envs | PPO/SAC — einfacher und oft schneller |
| Kontaktreiche Physik | SAC Model-free — Modelle sind ungenau |
| Partielle Beobachtbarkeit mit langem Horizont | Dreamer — RSSM-Memory hilft |
| Sehr stochastische Umgebung | Vermeide Model-based — Vorhersagefehler ist irreduzibel |
| Prototyp / frühes Experiment | PPO zuerst — Baseline zuerst, dann upgraden, falls nötig |

!!! tip "Baseline zuerst"
    Trainiere stets eine Model-free-Baseline (PPO oder SAC), bevor du zu Dreamer wechselst. Die Baseline sagt dir: (1) ist der Task überhaupt lösbar? (2) ungefähr wie viele reale Schritte braucht es? (3) ist Dreamers Sample-Efficiency-Gewinn die Engineering-Kosten wert? Ein Dreamer-Lauf, der schlechter performt als deine PPO-Baseline, ist ein Signal, dass das World Model ungenau ist — nicht dass der Task schwer ist.

---

## 8 · Latent-Space-Visualisierung

Der Latent-Space `z` (oder die Kombination `[h, z]`) ist die interne Repräsentation der Umgebung des World Models. Ihn zu visualisieren sagt dir, was das Modell gelernt hat — und legt Failure-Modes offen, bevor du in lange Trainingsläufe investierst.

### t-SNE-Projektion

t-SNE projiziert hochdimensionale Latent-Vektoren in 2D, während es lokale Nachbarschafts-Struktur erhält. Klare Cluster zeigen, dass das Modell bedeutsame State-Kategorien entdeckt hat:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_latent_space(agent, replay_buffer, n_samples=2000):
    """
    Extract latent states from a trained Dreamer agent and project to 2D.

    Parameters
    ----------
    agent       : trained DreamerV3 agent (provides encode method)
    replay_buffer : collected transitions (obs, action, reward, ...)
    n_samples   : number of transitions to sample for the plot
    """
    # Sample transitions from replay
    batch = replay_buffer.sample(n_samples)
    obs   = batch["image"]           # (N, H, W, C)
    rewards = batch["reward"]        # (N,)

    # Encode observations to latent vectors
    # In DreamerV3 the latent is (h, z); use h alone for a cleaner plot
    latents = agent.encode(obs)      # (N, latent_dim)

    # Project to 2D with t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    latents_2d = tsne.fit_transform(latents)

    # Colour by reward — reveals which latent regions lead to reward
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        latents_2d[:, 0], latents_2d[:, 1],
        c=rewards, cmap="RdYlGn", alpha=0.6, s=8,
    )
    plt.colorbar(scatter, label="reward")
    plt.title("Dreamer Latent Space (t-SNE) — coloured by reward")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig("latent_tsne.png", dpi=150)
    plt.show()


# Alternative: PCA for a faster (but linear) projection
from sklearn.decomposition import PCA

def visualize_latent_pca(latents, labels, label_name="reward"):
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    explained = pca.explained_variance_ratio_.sum() * 100

    plt.figure(figsize=(9, 7))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1],
                c=labels, cmap="plasma", alpha=0.5, s=6)
    plt.title(f"Latent Space PCA — {explained:.1f}% variance explained")
    plt.colorbar(label=label_name)
    plt.tight_layout()
    plt.savefig("latent_pca.png", dpi=150)
```

### Worauf zu achten ist

**Gute Zeichen:**

- Klare Cluster, die interpretierbaren States entsprechen (nahe Ziel, nahe Wand, freier Raum)
- High-Reward-States bilden eine kompakte Region — das Modell „weiß", wie Belohnung im Latent-Space aussieht
- PCA erklärt > 50 % Varianz in den ersten beiden Komponenten — der Latent-Space ist strukturiert, nicht zufällig

**Warnzeichen:**

- Alle Punkte bilden einen einzigen Blob — der Encoder ist kollabiert; Latents sind nicht informativ
- Belohnung ist zufällig über den Raum verstreut — der Reward-Prädiktor nutzt die Latent-Struktur nicht
- t-SNE zeigt eine Ring- oder Fraktal-Struktur — das Dynamik-Modell produziert periodische oder degenerierte Trajektorien

!!! info "Färbungs-Optionen"
    Färben nach Belohnung offenbart, ob belohnungs-relevante Information codiert ist. Färben nach Episoden-Zeitschritt offenbart, ob das Modell zeitliche Progression trackt. Färben nach einer hand-gelabelten semantischen Variable (z. B. „Agent ist nahe Wand" aus deiner Godot-Szene) offenbart, ob spezifische räumliche Konzepte repräsentiert sind — selbst wenn dem Modell nie explizit davon erzählt wurde.

---

## 9 · Dreamer vs Dyna — eine kurze Geschichte

Dreamer hat die Idee, ein World Model zu lernen und es für Policy-Updates zu nutzen, nicht erfunden. Die Abstammung zu verstehen hilft zu kalibrieren, was wirklich neu ist.

**Dyna (Sutton, 1991)** ist das originale Model-based-RL-Framework. Die Kernidee ist einfach: lerne ein tabellarisches Modell `P(s', r | s, a)` aus realen Transitionen, dann nutze es, um synthetische Transitionen für Q-Learning-Updates zu generieren. Jede reale Transition wird durch `k` modellgenerierte Transitionen ergänzt, was effektive Daten k-fach multipliziert.

```
Dyna-Q algorithm:
  for each real step:
    observe (s, a, r, s')
    update Q(s, a) from real transition
    update model: P(s', r | s, a) ← observed
    for k imagined steps:
      sample (ŝ, â) from previously visited states
      ŝ', r̂ = model(ŝ, â)
      update Q(ŝ, â) from imagined transition
```

Dyna funktioniert gut in tabellarischen Settings. Es versagt in Deep RL weil: (1) ein kleines neuronales Netzwerk-Modell komplexe Beobachtungsräume nicht repräsentieren kann; (2) Q-Learning auf OOD imaginierten Transitionen zu Value-Überschätzung führt (dasselbe Distributional-Shift-Problem wie Offline RL — siehe [Offline-RL-Unit](unit-offline-rl.md)); (3) Imagination ist Single-Step, was Long-Horizon-Struktur verliert.

**Modernes Dreamer vs Dyna:**

| Dimension | Dyna (1991) | DreamerV3 (2023) |
|---|---|---|
| World Model | Tabellarisch P(s', r \| s, a) | RSSM: Encoder + GRU + stochastisches z + Decoder |
| Imaginations-Tiefe | 1 Schritt | H = 15 Schritte (volle Rollouts) |
| Policy-Update | Q-Learning auf imaginierten Transitionen | Actor-Critic komplett in Imagination |
| Beobachtung | Tabellarisch (diskrete States) | Rohe Pixel oder Vektoren |
| Latent-Space | Keiner (Modell operiert im Obs-Space) | Kompaktes z — Modell operiert im Latent-Space |
| Gradient-Fluss | Nicht differenzierbar | Differenzierbar — Backprop durch Imagination |

Der Schlüsselsprung von Dyna zu Dreamer ist **Imagination im Latent-Space über lange Horizonte mit differenzierbarer Backpropagation**. Statt Single-Step-Sampling im Beobachtungsraum läuft Dreamer 15-Schritt-imaginierte Trajektorien durch ein differenzierbares Dynamik-Modell und backpropagiert den Actor-Gradienten durch alle 15 Schritte. Das lässt den Actor aus den langreichweitigen Konsequenzen seiner Aktionen komplett innerhalb der Imagination lernen.

### Trajektorie im Latent-Space (konzeptuell)

```
Real env:     s_0 → s_1 → s_2  (3 real steps, each costing a full render)

Imagination:
  encode(obs_0) → z_0
  dynamics(z_0, a_0) → z_1
  dynamics(z_1, a_1) → z_2
  ...
  dynamics(z_13, a_13) → z_14   (15 steps, each a cheap GRU forward pass)

reward_predictor(z_0), ..., reward_predictor(z_14)
→ compute λ-return
→ backprop through all 15 dynamics steps
→ update actor weights
```

Fünfzehn imaginierte Schritte kosten einen kleinen Bruchteil eines realen Renderings. Der Actor erhält Gradient-Signal aus 15 zukünftigen Zeitschritten für den Preis einer einzelnen realen Interaktion.

---

## 10 · Stretch Goals

Arbeite diese nach dem Lesen der Haupt-Unit durch. Jedes isoliert einen Aspekt von World-Model-RL, um Intuition zu bauen, die du aus Theorie allein nicht bekommen kannst.

**Rekonstruktions-Sanity-Check.** Trainiere DreamerV3 auf einer einfachen Godot-Szene für 10k Schritte. Speichere einen Batch von Beobachtungen und ihren Rekonstruktionen. Schau sie nebeneinander an. Kannst du die Szene in der Rekonstruktion erkennen? Sind die Rekonstruktionen verschwommene Blobs, ist der Encoder nicht konvergiert — das Dynamik-Modell trainiert auf bedeutungslosen Latents. Behebe die Rekonstruktion, bevor du die Policy trainierst.

**Imagination-vs-Realität-Vergleich.** Nach einem vollen Trainingslauf rolle 15 imaginierte Schritte aus einem realen Start-State aus. Dann rolle 15 reale Schritte aus demselben State mit denselben Aktionen aus. Plotte Belohnung (real vs imaginiert) pro Schritt. Wie groß ist die Divergenz nach 5 Schritten? Nach 15? Das sagt dir den effektiven Planungs-Horizont — jenseits dessen die Vorhersagen des Modells zu ungenau sind, um ihnen zu vertrauen.

**Dreamer vs PPO Sample-Efficiency.** Trainiere beide auf demselben Godot-Task. Plotte `ep_rew_mean` vs Anzahl **realer Umgebungsschritte** (nicht Wall-Clock-Zeit). Dreamer sollte ein gegebenes Performance-Level in weniger Schritten erreichen. Jetzt plotte vs Wall-Clock-Zeit. Welcher Algorithmus erreicht zuerst dieselbe Performance? Die Antwort hängt von deiner Hardware und Szenen-Komplexität ab.

**Latent-Interpolation.** Encodiere zwei Beobachtungen — eine nahe einer Wand, eine nahe dem Ziel. Interpoliere linear zwischen den beiden Latent-Vektoren (z = α·z_wall + (1-α)·z_goal für α ∈ [0, 1]) und decodiere jedes interpolierte Latent. Zeigen die decodierten Bilder eine glatte räumliche Transition? Glatte Interpolation zeigt, dass der Latent-Space semantisch strukturiert ist; diskontinuierliche Sprünge zeigen eine kollabierte oder fragmentierte Repräsentation.

**Dyna from scratch.** Implementiere den Basis-Dyna-Q-Algorithmus (5 imaginierte Schritte pro realem Schritt) auf der FrozenLake-Umgebung aus der [Q-Learning-Unit](unit-q-learning.md). Vergleiche Konvergenz-Geschwindigkeit mit Standard-Q-Learning. Das baut direkte Intuition dafür, woher modernes Dreamer kam.

---

## Was kommt als Nächstes?

World Models repräsentieren die Frontier der Sample-Efficiency in Deep RL. Die Ideen hier — komprimierte Repräsentationen lernen, Zukünfte imaginieren, im Latent-Space planen — sind aktive Forschungsbereiche und bilden das Rückgrat einiger der fähigsten je gebauten RL-Systeme.

Für Produktions-Godot-Projekte bleibt der empfohlene Pfad: **PPO-Baseline → Curiosity bei Sparse Rewards → World Models, falls Simulation teuer ist oder Planung gebraucht wird**. Jeder Schritt fügt Macht und Komplexität hinzu; geh zum nächsten nur, wenn du Evidenz hast, dass der einfachere Ansatz plateaut hat.

Wenn du die Linse weiter aufziehen willst, untersucht die nächste Unit — **Foundation Models for Control (VLA)** — RT-2, Octo, OpenVLA und π0: eine völlig andere Wette darauf, wie man universell einsetzbare verkörperte Agenten trainiert.

[→ Foundation Models for Control](unit-foundation-models.md) · [Kursstartseite](index.md)
