# Decision Transformer — RL als Sequence Modeling

[← Offline-RL](unit-offline-rl.md) · [Kursstartseite](index.md) · [→ Ship Your Brain](unit-10.md)

!!! info "Zeit"
    Lesen: ~40 min · Training: ~20 min GPU / ~1,5 Std CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    - **Return-konditionierte Rollouts** — trainiere einen einzelnen Decision Transformer einmal, lass ihn dann mit `target_return = 50, 100, 150, 200` laufen und beobachte, wie ein Modell vier qualitativ unterschiedliche Policies produziert. Der Return-to-Go fungiert als Lenkrad für das Skill-Level.
    - **Attention-Heatmap** — extrahiere die kausalen Attention-Gewichte aus der letzten Transformer-Schicht und visualisiere, auf welche vergangenen (R, s, a)-Tokens sich das Modell stützt. Long-Horizon-Godot-Aufgaben sollten Attention-Spitzen an Verzweigungspunkten zeigen (Türöffnungen, Plattformsprünge).
    - **Skill-Ladder-Plot** — sammle drei Datensätze aus CartPole (Zufall, mittelmäßig, Experte), trainiere ein DT auf der Vereinigung, plotte dann `episode_return` gegen `target_return`. Eine Diagonale heißt: Return-Konditionierung funktioniert; eine flache Linie heißt: das Modell ist auf den Datensatzdurchschnitt kollabiert.

---

## 0 · Die große Idee: RL als Sequenzvorhersage

Alles in diesem Kurs hat bisher dieselbe Maschinerie verwendet: eine Value-Funktion oder Policy, trainiert mit Bellman-Backups, Experience Replay, Target Networks, Advantage-Schätzung und einem sorgfältigen Tanz von Hyperparametern. PPO hat seine Clip-Ratio und sein GAE-Lambda. SAC hat seine Entropie-Temperatur. CQL (behandelt in [unit-offline-rl.md](unit-offline-rl.md)) hat seinen konservativen Koeffizienten `alpha`. IQL hat sein Expectile `tau` und seine Weight-Temperature.

**Decision Transformer** (Chen et al., 2021) wirft das alles weg.

Der Pitch ist ein Satz: *sage die nächste Aktion vorher, gegeben ein Kontextfenster vergangener `(Return-to-Go, Zustand, Aktion)`-Tupel, trainiert wie ein Sprachmodell auf Offline-Daten*. Es gibt kein Bellman-Backup. Kein Target Network. Keinen Replay Buffer. Keinen Critic. Keinen Explorations-Zeitplan. Keine konservative Strafe. Die Trainingsschleife ist die Standard-Supervised-Schleife, die du für GPT-2 verwenden würdest.

Das funktioniert weit besser, als es das Recht hat. Auf den Standard-Offline-RL-Benchmarks erreicht oder schlägt Decision Transformer CQL und IQL bei den meisten Aufgaben, obwohl er ein strikt einfacherer Algorithmus ist. Und weil Architektur und Trainingsrezept identisch mit einem Sprachmodell sind, lässt sich das gesamte LLM-Ökosystem — verteiltes Training, Mixed-Precision-Kernel, Attention-Optimierungen, FlashAttention, Fine-Tuning-Bibliotheken — direkt anwenden.

Für einen Game-AI-Kurs zählt das doppelt. Erstens können deine Godot-Agenten jetzt dieselbe Compute-Infrastruktur nutzen, die dein Team bereits für Textmodelle pflegt. Zweitens verwandelt der Konditionierungstrick — dem Modell zu sagen „produziere so viel Reward" — ein trainiertes Modell in eine ganze Familie von Policies auf unterschiedlichen Skill-Levels, was genau das ist, was du für adaptive Spielschwierigkeit willst.

---

## 1 · Return-to-Go-Konditionierung

Der technische Kern von Decision Transformer ist eine Designentscheidung: statt die Policy nur auf den Zustand zu konditionieren, konditioniere sie auf `(Zustand, gewünschter zukünftiger Return)`.

### Inferenzschleife

```
At inference:
  - Set R_1 = desired_total_reward            (e.g., 200 for CartPole)
  - Generate action a_1 conditioned on (R_1, s_1)
  - Execute a_1, observe r_1, s_2
  - Set R_2 = R_1 - r_1                       (countdown)
  - Generate a_2 conditioned on (R_2, s_2, a_1, R_1, s_1)
  - Set R_3 = R_2 - r_2
  - Generate a_3 conditioned on (R_3, s_3, a_2, R_2, s_2, a_1, R_1, s_1)
  - ...
```

Der Return-to-Go startet beim Ziel und zählt herunter, während Rewards verdient werden. Das Modell sieht ein laufendes Konto „wie viel Reward ich noch schulde".

### Warum das funktioniert

Wenn der Offline-Datensatz hochwertige Trajektorien enthält, die mit hohem `R_1` gelabelt sind, lernt das Modell die bedingte Verteilung `p(action | state, return-to-go)`. Bei der Inferenz ist das Anfragen eines bestimmten Returns nur das Konditionieren des generativen Modells auf einen bestimmten Wert einer seiner Eingaben.

Das ist exakt derselbe Trick, der in der bedingten Bildgenerierung verwendet wird („generiere ein Bild einer Katze") oder in instruktionsgetunten LLMs („antworte in formalem Englisch"). Die Konditionierungsvariable lenkt die Generierung in den Teilbereich der Trainingsdaten, der dazu passt.

### Die Token-Sequenz

Die Trajektorie wird zu einer einzigen Sequenz von Tokens geflacht:

```
[R_1, s_1, a_1, R_2, s_2, a_2, R_3, s_3, a_3, ...]
```

Jeder Zeitschritt steuert genau drei Tokens in einer festen Reihenfolge bei. Ein Kontextfenster von `K = 20` Zeitschritten wird daher zu `3K = 60` Tokens. Der Transformer behandelt das als eine lange Sequenz und verwendet kausales Masking, sodass jedes Token nur frühere Tokens beachtet.

!!! tip "Return-to-Go, nicht Reward"
    Das Modell wird auf die **Summe zukünftiger Rewards** konditioniert, nicht auf den Pro-Schritt-Reward. Pro-Schritt-Rewards sind verrauscht und lokal; der Return-to-Go ist ein globales Signal, das eindeutig „das ist eine erfolgreiche Trajektorie" vs. „das ist eine mittelmäßige Trajektorie" identifiziert. Das ist dieselbe Einsicht, die Monte-Carlo-Returns in Policy-Gradient-Methoden informativer macht als One-Step-Rewards.

---

## 2 · Architektur

Decision Transformer ist strukturell identisch mit GPT-2 mit drei kleinen Modifikationen: separate Input-Embeddings für jede Modalität, eine zusätzliche Zeitschritt-Positionscodierung und einen Aktionsvorhersage-Head.

### Komponenten

- **Kausaler Transformer (GPT-Stil)** — derselbe `TransformerEncoder` mit kausaler Maske, den du fürs Sprachmodellieren verwenden würdest.
- **Input-Embeddings** — drei separate Linear-Layer: eine für Return-to-Go (1 → `d_model`), eine für Zustand (`obs_dim` → `d_model`), eine für Aktion (`act_dim` → `d_model`).
- **Zeitschritt-Positionscodierung** — eine `nn.Embedding(max_timesteps, d_model)`, die an jedem Zeitschritt zu allen drei Modalitäts-Embeddings addiert wird. Das ist zusätzlich zur (oder ersetzt die) Standard-Sinus-Position-Encoding, weil derselbe Zeitschritt `t` an drei verschiedenen Token-Positionen in der geflachten Sequenz erscheint.
- **Kontextlänge K** — typischerweise 20 bis 100 Zeitschritte. Kurze Kontexte funktionieren erstaunlich gut; das Modell braucht nicht die volle Episodenhistorie.
- **Ausgabe-Head** — ein Linear-Layer, der den Hidden State an jeder State-Token-Position nimmt und die folgende Aktion vorhersagt.
- **Loss** — Mean Squared Error für kontinuierliche Aktionen, Cross-Entropy für diskrete Aktionen.

### Architekturdiagramm

```
[R_1] [s_1] [a_1] [R_2] [s_2] [a_2] [R_3] [s_3] [?]
  |     |     |     |     |     |     |     |     |
  embed embed embed embed embed embed embed embed embed
  |     |     |     |     |     |     |     |     |
  +-------------------------------------------------+
  |         Causal Transformer (GPT-style)          |
  |              (masked self-attention)            |
  +-------------------------------------------------+
                                                    |
                                              action head
                                              -> predicted a_3
```

Das Entscheidende zu verinnerlichen: die Aktionsvorhersage an Position `s_t` beachtet nur Tokens an Positionen `<= s_t` in der geflachten Sequenz — also sieht sie alle vorherigen `(R, s, a)`-Tripel plus das aktuelle `R_t` und `s_t`, aber nicht das aktuelle `a_t`. Das erzwingt das kausale Masking.

!!! warning "Lass die Aktion nicht durchsickern"
    Ein häufiger Bug ist, `a_t` versehentlich in den Kontext aufzunehmen, der zur Vorhersage von `a_t` dient. Bei der verschachtelten Anordnung `[R_t, s_t, a_t]` musst du `a_t` aus dem Hidden State an der `s_t`-Position vorhersagen, nicht an der `a_t`-Position. Sagst du von der `a_t`-Position aus voraus, kopiert das Modell trivial den Input und erreicht near-Zero-Loss, ohne etwas Nützliches zu lernen.

---

## 3 · Training auf Offline-Daten

Das Trainingsrezept ist überwachtes Lernen von Anfang bis Ende. Keine Umgebung in der Schleife.

### Pipeline

1. **Offline-Datensatz sammeln** — aufgenommene Trajektorien mit `(obs, action, reward)` an jedem Schritt. Gleiches Format wie für CQL/IQL.
2. **Returns-to-Go berechnen** — für jede Trajektorie rückwärts laufen und `R_t = Summe der Rewards von Schritt t bis Ende` berechnen.
3. **In Kontextfenster zerlegen** — jede Trajektorie in überlappende Fenster der Länge `K` chunken.
4. **Mit Supervised-Loss trainieren** — `(R, s, a)`-Fenster durch den Transformer schicken, MSE (kontinuierlich) oder Cross-Entropy (diskret) auf den vorhergesagten Aktionen berechnen.

Es gibt nach der Datensatzsammlung keine RL-Trainingsschleife. Das Training ist `for batch in loader: loss.backward(); optimizer.step()` — identisch zum Fine-Tuning eines Sprachmodells.

### Vollständige minimale Implementierung

```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataset(Dataset):
    """Dataset of (returns-to-go, states, actions) windows."""

    def __init__(self, trajectories, context_len=20, scale=1000.0):
        # trajectories: list of dicts with 'obs', 'actions', 'rewards'
        self.context_len = context_len
        self.scale = scale
        self.windows = []  # each entry: (rtg, states, actions, timesteps)

        for traj in trajectories:
            obs = np.asarray(traj["obs"], dtype=np.float32)
            acts = np.asarray(traj["actions"], dtype=np.float32)
            rews = np.asarray(traj["rewards"], dtype=np.float32)
            T = len(rews)

            # Returns-to-go: R_t = sum of rewards from t to end
            rtg = np.zeros(T, dtype=np.float32)
            running = 0.0
            for t in reversed(range(T)):
                running += rews[t]
                rtg[t] = running
            rtg /= self.scale  # normalize for stable training

            # Slice into windows of length context_len (pad if shorter)
            for start in range(0, T):
                end = min(start + context_len, T)
                L = end - start
                pad = context_len - L

                states = np.concatenate(
                    [np.zeros((pad, obs.shape[1]), dtype=np.float32), obs[start:end]]
                )
                actions = np.concatenate(
                    [np.zeros((pad, acts.shape[1]), dtype=np.float32), acts[start:end]]
                )
                rtgs = np.concatenate([np.zeros(pad, dtype=np.float32), rtg[start:end]])
                timesteps = np.concatenate(
                    [np.zeros(pad, dtype=np.int64), np.arange(start, end, dtype=np.int64)]
                )
                self.windows.append((rtgs, states, actions, timesteps))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        rtg, states, actions, timesteps = self.windows[idx]
        return (
            torch.from_numpy(rtg),
            torch.from_numpy(states),
            torch.from_numpy(actions),
            torch.from_numpy(timesteps),
        )


class DecisionTransformer(nn.Module):
    def __init__(self, obs_dim, act_dim, d_model=128, n_heads=4, n_layers=3, context_len=20):
        super().__init__()
        self.context_len = context_len

        # Embeddings for each modality
        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_state = nn.Linear(obs_dim, d_model)
        self.embed_action = nn.Linear(act_dim, d_model)
        self.embed_timestep = nn.Embedding(1000, d_model)

        # GPT-style transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.predict_action = nn.Linear(d_model, act_dim)

    def forward(self, rtg, states, actions, timesteps):
        # rtg:        (B, T)
        # states:     (B, T, obs_dim)
        # actions:    (B, T, act_dim)
        # timesteps:  (B, T)
        B, T = states.shape[:2]

        t_emb = self.embed_timestep(timesteps)
        rtg_emb = self.embed_rtg(rtg.unsqueeze(-1)) + t_emb
        state_emb = self.embed_state(states) + t_emb
        action_emb = self.embed_action(actions) + t_emb

        # Interleave per timestep: (B, T, 3, d_model) -> (B, 3T, d_model)
        x = torch.stack([rtg_emb, state_emb, action_emb], dim=2)
        x = x.reshape(B, 3 * T, -1)
        x = self.ln(x)

        # Causal mask
        mask = torch.triu(torch.ones(3 * T, 3 * T, device=x.device), diagonal=1).bool()
        h = self.transformer(x, mask=mask)

        # Predict action from state positions (every 3rd token starting at index 1)
        state_hiddens = h[:, 1::3]
        return self.predict_action(state_hiddens)


def train_decision_transformer(dataset, obs_dim, act_dim, n_epochs=100, device="cpu"):
    model = DecisionTransformer(obs_dim, act_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(n_epochs):
        total_loss = 0.0
        for rtg, states, actions, timesteps in loader:
            rtg, states, actions, timesteps = (
                rtg.to(device),
                states.to(device),
                actions.to(device),
                timesteps.to(device),
            )
            pred_actions = model(rtg, states, actions, timesteps)
            loss = ((pred_actions - actions) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total_loss / len(loader):.4f}")
    return model
```

### Worauf während des Trainings achten

- **Loss-Kurve** — MSE-Loss sollte glatt abfallen. Eine flache Kurve ab Epoche 1 bedeutet, dass die Embeddings nicht lernen; prüfe den Zeitschritt-Embedding-Bereich gegen deine Trajektorienlängen.
- **Gradientennorm** — das `clip_grad_norm_(..., 0.25)` ist kritisch. Transformer-Gradienten schießen hoch, wenn Kontextfenster schlecht normalisierte Returns-to-Go enthalten (daher das `scale=1000.0`).
- **Validations-Rollouts** — evaluiere alle 10 Epochen in der Umgebung mit `target_return = max_return_in_dataset`. Der Trend sollte aufwärts gehen.

!!! tip "Skaliere den Return-to-Go"
    Teile Returns immer durch eine `scale`-Konstante (typisch: 1000 für CartPole/Atari, umgebungsspezifisch für Godot). Rohe Returns wie 1500 in ein lineares Embedding gefüttert produzieren riesige Aktivierungen, die Attention-Softmaxes destabilisieren. Wähle `scale` so, dass der typische Return-to-Go nach dem Teilen in `[0, 1]` liegt.

---

## 4 · Inferenz: Lenken mit Return-to-Go

Inferenz ist autoregressiv. Bei jedem Schritt hängst du das neueste `(R, s, a)` an den laufenden Kontext, kürzt auf die letzten `K` Zeitschritte und fragst das Modell nach der nächsten Aktion.

```python
import numpy as np
import torch


def evaluate_dt(model, env, target_return=200, context_len=20, scale=1000.0, max_steps=1000):
    """Run Decision Transformer in an environment."""
    obs, _ = env.reset()

    # Running context buffers (lists, trimmed each step)
    rtg_buf = [target_return / scale]
    state_buf = [obs]
    action_buf = [np.zeros(env.action_space.shape, dtype=np.float32)]
    timestep_buf = [0]

    total_reward = 0.0
    model.eval()

    for t in range(max_steps):
        start = max(0, t - context_len + 1)

        rtg = torch.tensor(rtg_buf[start:], dtype=torch.float32).unsqueeze(0)
        states = torch.tensor(np.array(state_buf[start:]), dtype=torch.float32).unsqueeze(0)
        actions = torch.tensor(np.array(action_buf[start:]), dtype=torch.float32).unsqueeze(0)
        timesteps = torch.tensor(timestep_buf[start:], dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            pred = model(rtg, states, actions, timesteps)
        action = pred[0, -1].cpu().numpy()  # last predicted action

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        rtg_buf.append(rtg_buf[-1] - reward / scale)
        state_buf.append(obs)
        action_buf.append(action)
        timestep_buf.append(t + 1)

        if terminated or truncated:
            break

    return total_reward
```

### Lenkexperiment

Trainiere das DT einmal. Lass dann `evaluate_dt(model, env, target_return=R)` für ein Raster von Ziel-Returns laufen:

```python
for target in [50, 100, 150, 200, 250]:
    returns = [evaluate_dt(model, env, target_return=target) for _ in range(10)]
    print(f"target={target:>3}  achieved={np.mean(returns):.1f} ± {np.std(returns):.1f}")
```

Erwartete Ausgabe für ein gut trainiertes DT auf CartPole-v1 (max Return 500):

```
target= 50   achieved= 53.2 ± 8.1
target=100   achieved=104.7 ± 11.4
target=150   achieved=147.9 ± 14.0
target=200   achieved=198.5 ± 18.2
target=250   achieved=241.6 ± 22.7
```

Ein Modell produziert fünf qualitativ unterschiedliche Policies. Das ist mit Standard-Policy-Gradient-Methoden unmöglich — du müsstest SAC/PPO für jedes Ziel neu trainieren.

!!! info "Adaptive Schwierigkeit"
    Das Lenkexperiment ist eine direkte Demonstration, wie man adaptive Spielschwierigkeit in ein einziges trainiertes Modell baut. Ein von einem Decision Transformer gesteuerter NPC kann auf „einfach" (niedriger Target-Return), „mittel" oder „Experte" gesetzt werden, ohne irgendein Retraining oder Per-Schwierigkeits-Checkpoints.

---

## 5 · DT vs. klassisches Offline-RL

Decision Transformer ist der am einfachsten zu *implementierende* Offline-RL-Algorithmus, dominiert aber CQL und IQL nicht bei jedem Problem. Die Unterschiede sind real und die Wahl sollte von deinem Datensatz getrieben sein.

| | BC | CQL | IQL | Decision Transformer |
|---|---|---|---|---|
| Training | Überwacht | Bellman + Konservatismus | Bellman + In-Sample | Überwacht (kein Bellman) |
| Hyperparameter | Wenige | Viele (alpha, beta, Target-Updates) | Mittel (tau, beta) | Wenige (context_len, scale, lr) |
| Stitching (suboptimale Trajektorien kombinieren) | Nein | Ja | Ja | Begrenzt (nur Kontext) |
| Return-Konditionierung | Nein | Nein | Nein | Ja |
| Architektur | MLP | MLP | MLP | Transformer |
| Inferenzkosten | Niedrig | Niedrig | Niedrig | Höher (Attention über K Schritte) |
| GPU-freundlich | Mäßig | Mäßig | Mäßig | Sehr (LLM-Stack anwendbar) |
| Am besten für | Expertendaten | Daten mit spärlicher Abdeckung | Daten gemischter Qualität | Daten mit variierten Returns |

### Der Stitching-Vorbehalt

Die wichtigste einzelne Einschränkung von Decision Transformer: **er kann nicht stitchen**.

„Stitching" heißt, Stücke zweier suboptimaler Trajektorien zu kombinieren, um eine bessere zu erzeugen. CQL und IQL stitchen, weil das Bellman-Backup Value-Information über Zustände hinweg propagiert, unabhängig davon, aus welcher Trajektorie die Daten kamen. Erreicht Trajektorie A Zustand `s*` und startet Trajektorie B in der Nähe von `s*` und erreicht ein High-Reward-Terminal, lernt die Q-Funktion, dass `s*` wertvoll ist, und `Q(s, a -> s*)` wird hoch.

Decision Transformer hat diesen Mechanismus nicht. Er ist ein bedingtes Dichtemodell über Sequenzen. Erreicht keine Trajektorie im Datensatz `target_return`, hat DT kein Lernsignal für „wie man diesen Return tatsächlich erreicht" — er kann nur innerhalb der Support-Region der gesehenen Trajektorien interpolieren.

### Praktische Konsequenz

Ist dein Offline-Datensatz ein Mix aus kurzen, suboptimalen Episoden, extrahieren CQL/IQL zuverlässig eine gestitchte Policy, die die beste Einzeltrajektorie übertrifft. DT plateaut nahe dem Return der besten Trajektorie.

Enthält dein Offline-Datensatz einen vielfältigen Mix an Returns — einschließlich einiger near-optimaler Trajektorien — glänzt DT, weil der Return-to-Go als sauberes Konditionierungssignal fungiert und du zur Inferenzzeit den High-Return-Bereich abfragen kannst.

| Dein Datensatz | Erster Algorithmus zum Probieren |
|---|---|
| Viele suboptimale Trajektorien, wenige/keine Experten | CQL oder IQL (brauchen Stitching) |
| Mix von niedrigem bis hohem Return | Decision Transformer (Return-Konditionierung funktioniert) |
| Überwiegend Experte | BC, dann DT für Steuerbarkeit |
| Winzig (< 200 Episoden) | CQL (DT überfittet) |
| Riesig (> 10k Episoden), möchtest Pre-Train + Fine-Tune | DT (LLM-artige Skalierung) |

---

## 6 · Trajectory Transformer (kurz)

**Trajectory Transformer** (Janner et al., 2021) ist das eng verwandte Geschwister des DT und es lohnt sich, ihn beim Namen zu kennen.

Die zentralen Unterschiede:

- Tokens sind `(Zustand, Aktion, Reward)`-Tripel — **Rewards sind Tokens**, keine Konditionierungsvariablen.
- Bei der Inferenz führt das Modell **Beam Search** über die Joint-Sequenz durch, um zukünftige Trajektorien zu planen, die den vorhergesagten Reward maximieren.
- Das macht ihn zu einem *Planungs*-Algorithmus, nicht nur einer *Policy*. Er kann über kontrafaktische Zukünfte schlussfolgern.

Trade-offs gegenüber Decision Transformer:

| | Decision Transformer | Trajectory Transformer |
|---|---|---|
| Inferenz | Ein Forward Pass pro Schritt | Beam Search über Horizont (teuer) |
| Return-Signal | Konditionierungs-Input | Als Token vorhergesagt |
| Planungsfähigkeit | Keine | Ja (Search) |
| Geeignet für Echtzeit-Game-AI | Ja | Grenzwertig (Beam-Search-Latenz) |
| Referenz | Chen et al., 2021 | Janner et al., 2021 |

Für Godot-Agenten, die bei jedem Physik-Tick eine Aktion produzieren müssen (60 Hz typisch), ist DT die richtige Wahl. Trajectory Transformer wird interessant, wenn du ein rundenbasiertes Spiel hast oder einen Planungshorizont lang genug, um die Beam-Search-Kosten zu amortisieren.

---

## 7 · Godot-Integration

Decision Transformer braucht denselben Offline-Datensatz wie CQL/IQL — aufgenommene `(obs, action, reward)`-Trajektorien. Der sauberste Weg, einen in diesem Kurs zu produzieren, ist, Rollouts von einem trainierten PPO-Agenten aufzunehmen (siehe [unit-07.md](unit-07.md)) und sie in eine Liste von Dicts zu dumpen.

### Trajektorien von einem trainierten PPO-Agenten sammeln

```python
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from stable_baselines3 import PPO
import numpy as np


def collect_trajectories(env_path, model_path, n_trajectories=1000):
    """Collect trajectory data from a trained PPO agent."""
    env = StableBaselinesGodotEnv(env_path=env_path, n_parallel=1)
    model = PPO.load(model_path)

    trajectories = []
    for ep in range(n_trajectories):
        obs, _ = env.reset()
        traj = {"obs": [], "actions": [], "rewards": []}
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            traj["obs"].append(np.asarray(obs, dtype=np.float32))
            traj["actions"].append(np.asarray(action, dtype=np.float32))
            obs, reward, terminated, truncated, _ = env.step(action)
            traj["rewards"].append(float(reward))
            done = terminated or truncated
        trajectories.append(traj)

    env.close()
    return trajectories
```

### Suboptimale Policies einmischen

Ein DT, das nur auf Experten-PPO-Rollouts trainiert wurde, produziert eine flache Lenkkurve — jeder Ziel-Return wird auf dasselbe Expertenverhalten abgebildet. Um die Return-Konditionierung zu üben, brauchst du *variierte* Returns im Datensatz. Der sauberste Weg:

1. Speichere PPO-Checkpoints bei 25 %, 50 %, 75 %, 100 % des Trainings.
2. Sammle 250 Trajektorien von jedem Checkpoint.
3. Konkateniere sie zu einem einzigen 1000-Trajektorien-Datensatz.

Das ergibt einen Datensatz mit Returns, die von „mittelmäßig" bis „Experte" verteilt sind — genau das Regime, in dem die DT-Konditionierung glänzt.

```python
checkpoints = ["ppo_25.zip", "ppo_50.zip", "ppo_75.zip", "ppo_100.zip"]
all_trajectories = []
for ckpt in checkpoints:
    all_trajectories.extend(collect_trajectories("MultiLevelRobot.x86_64", ckpt, n_trajectories=250))

dataset = TrajectoryDataset(all_trajectories, context_len=20, scale=1000.0)
obs_dim = all_trajectories[0]["obs"][0].shape[0]
act_dim = all_trajectories[0]["actions"][0].shape[0]
model = train_decision_transformer(dataset, obs_dim, act_dim, n_epochs=100)
```

### Headline-Vergleich

Lass nach dem Training alle drei auf der Godot-Umgebung laufen und vergleiche:

| Policy | Wie produziert | Erwarteter Episoden-Return |
|---|---|---|
| Originales PPO (Experten-Checkpoint) | Standard-Online-Training | Hoch (Baseline) |
| BC trainiert auf dem gemischten Datensatz | Überwacht auf `(s, a)` allein | Mittel — kollabiert auf Datensatzmittel |
| DT bei `target_return = max` | Rezept dieser Unit | Nahe Experten-PPO |
| DT bei `target_return = median` | Gleiches Modell, andere Konditionierung | Nahe mittelmäßigem Checkpoint |

Das DT bei `target_return = max` sollte sich dem PPO-Experten annähern. Das DT bei `target_return = median` sollte sich dem mittelmäßigen Checkpoint annähern. **Ein Modell, volle Skill-Ladder.**

---

## 8 · Verbindung zu LLMs

Ein trainierter Decision Transformer und ein trainiertes GPT-Sprachmodell sind derselbe Algorithmus, der auf verschiedenen Tokens läuft. Beide sind kausale Transformer, trainiert mit Next-Token-Prediction auf Offline-Korpora. Die einzigen Unterschiede sind Token-Typ (Aktion vs. Wortteil) und Modalität der Input-Embeddings (lineare Projektionen von `(R, s, a)` vs. ein gelerntes Vokabular-Embedding).

Das ist keine Metapher. Die konkreten Konsequenzen:

- **Gleiche Trainingsinfrastruktur** — DeepSpeed, FSDP, Mixed-Precision-Training, FlashAttention, Gradient Checkpointing gelten für DT ohne Modifikation.
- **Gleiche Fine-Tuning-Rezepte** — LoRA, Prefix Tuning, RLHF-artiges Präferenz-Fine-Tuning funktionieren alle auf DT.
- **Gleiche Skalierungsgesetze** — größere DTs, die auf größeren Trajektorien-Korpora trainiert wurden, verbessern sich weiter und spiegeln die Chinchilla-artigen Skalierungskurven wider, die man von Sprachmodellen kennt.

Zwei wichtige Folgepapiere erweitern die Analogie:

- **Prompt Decision Transformer** (Xu et al., 2022) — stellt eine kleine Menge „Demonstrations-Tokens" voran, die die Aufgabe beschreiben. Ein Modell bewältigt viele Aufgaben; Aufgabenwechsel ist Prompt-Wechsel, kein Retraining. Das ist das RL-Analogon zu System-Prompts in Chat-LLMs.
- **Hyper-Decision Transformer** (Xu et al., 2023) — verwendet ein Hypernetzwerk, um aus einer Aufgabenbeschreibung aufgabenspezifische Gewichte zu generieren. Das gleiche architektonische Muster wird heute beim Multi-Task-LLM-Serving verwendet.

Für den Game-AI-Aspekt: ein einzelner Decision Transformer kann viele Spiele auf vielen Skill-Levels spielen, indem sowohl Prompt (Aufgabe) als auch Return-to-Go (Skill) variiert werden. Genau das ist die Substanz, die du brauchst für ein vereinheitlichtes NPC-Gehirn, das jedes Minispiel in einem großen RPG mit einem Checkpoint handhabt.

---

## 9 · Stretch Goals

**Skill-Ladder-Verifikation auf CartPole.**
Sammle drei CartPole-Datensätze auf unterschiedlichen Skill-Levels: eine Zufalls-Policy (Return ~20), ein teilweise trainiertes DQN (Return ~100) und ein voll trainiertes DQN (Return ~500). Trainiere ein DT auf der Vereinigung. Führe das Lenkexperiment aus Abschnitt 4 mit `target_return in {25, 50, 100, 200, 400, 500}` aus. Plotte `achieved_return` gegen `target_return`. Eine annähernde Diagonale heißt, Return-Konditionierung funktioniert; eine flache Linie heißt, das Modell ist kollabiert und du musst untersuchen (wahrscheinlich die scale-Konstante oder Kontextlänge).

**Head-to-Head mit CQL und IQL auf einer Godot-Umgebung.**
Nimm den Vier-PPO-Checkpoint-Datensatz aus Abschnitt 7. Trainiere drei Policies darauf: CQL (mit d3rlpy), IQL (mit d3rlpy) und DT (diese Unit). Evaluiere alle drei auf der MultiLevelRobot-Aufgabe für jeweils 50 Episoden. Berichte mittleren Episoden-Return und Erfolgsquote. Erwartetes Ergebnis: DT konkurrenzfähig mit oder leicht hinter IQL bei dieser Aufgabe; weit vor CQL, wenn die Datensatzabdeckung reichhaltig ist.

**Attention-Visualisierung.**
Modifiziere die `DecisionTransformer.forward`-Methode, sodass sie auch die Attention-Gewichte der letzten Schicht zurückgibt (verwende `torch.nn.functional.scaled_dot_product_attention` mit `return_attention=True`, oder tausche ein eigenes Attention-Modul ein). Plotte für eine Rollout-Episode die Attention-Map (`3K x 3K`) bei jedem Schritt. Annotiere, auf welche vergangenen Tokens das Modell an Verzweigungspunkten der Umgebung achtet (Türöffnungen, Plattformübergänge). Du solltest Attention-Spitzen sehen, die sich mit semantisch wichtigen vergangenen Zuständen decken.

**Minimaler Prompt Decision Transformer.**
Füge ein einzelnes „Task-Token" am Anfang jeder Sequenz ein — ein gelerntes `nn.Parameter` der Form `(n_tasks, d_model)`, indiziert per Task-ID. Trainiere ein DT auf Daten aus zwei verschiedenen Godot-Umgebungen (z. B. MultiLevelRobot und eine andere Szene). Wechsle bei der Inferenz die Aufgabe, indem du die Task-ID änderst. Verifiziere, dass ein Modell beide bewältigt. Das ist eine 20-Zeilen-Modifikation, die die Kernidee von Prompt DT nachstellt.

**Phasenübergänge beim Lenken.**
Führe das Lenkexperiment mit einem feingranularen Raster aus: `target_return in [0, 10, 20, 30, ..., 500]`. Plotte das Ergebnis. Ist die Kurve glatt, oder gibt es treppenartige Phasenübergänge, bei denen kleine Änderungen in `target_return` große Sprünge im Verhalten produzieren? Phasenübergänge zeigen meist an, dass dein Trainingsdatensatz multimodal ist (ein paar diskrete Cluster von Trajektorienqualität) statt kontinuierlich verteilt. Die Lösung ist mehr Datensatzvielfalt.

---

[← Offline-RL](unit-offline-rl.md) · [Kursstartseite](index.md) · [→ Ship Your Brain](unit-10.md)
