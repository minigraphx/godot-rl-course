# RLHF — Belohnungen aus menschlichen Präferenzen lernen

[← Imitation Learning](unit-09.md) · [Kursstartseite](index.md) · [→ Offline-RL](unit-offline-rl.md)

!!! info "Zeit"
    Lesen: ~45 min · Training: ~20 min GPU / ~1,5 Std CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    - **Godot Side-by-Side-Viewer** — der einzige Ort, an dem das *echte* Signal lebt. Zwei aufgenommene Trajektorien laufen nebeneinander; ein Designer klickt diejenige an, die besser aussieht. Jede andere Metrik in dieser Unit ist ein Proxy für diese Klicks.
    - **TensorBoard** — `reward_model/val_accuracy` sollte über 70 % steigen, bevor du dem Modell traust; `policy/kl_to_ref` sollte beim PPO-Fine-Tuning *begrenzt bleiben*. Ein KL-Ausschlag ist der sichtbare Fingerabdruck von Reward Hacking.
    - **Reward-Model-Probe-Skript** — bewerte 100 handverlesene Trajektorien mit dem trainierten Reward-Modell und ordne sie. Wenn die Reihenfolge nicht dem Bauchgefühl eines Designers entspricht, ist das Modell auf Oberflächenmerkmale überangepasst und PPO wird sie ausnutzen.

---

## 0 · Warum RLHF wichtig ist (und warum dich das bei Spielen interessieren sollte)

Die Erkenntnis, die ChatGPT möglich machte, ist klein und brutal: **du kannst keine Belohnungsfunktion für „schreibe eine hilfreiche Antwort" schreiben.** Kein Designer dieser Welt kann in geschlossener Form in Python artikulieren, was einen Absatz hilfreich macht. Ein Mensch, der zwei Kandidatenabsätze liest, kann den besseren jedoch in unter einer Sekunde auswählen.

RLHF — Reinforcement Learning from Human Feedback — verwandelt diese Asymmetrie in ein Trainingssignal. Menschen vergleichen Paare von Outputs. Ein neuronales Netz lernt, ihre Präferenzen vorherzusagen. Dieses Netz *wird* die Belohnungsfunktion. PPO (das du bereits aus [unit-ppo-deep.md](unit-ppo-deep.md) kennst) optimiert die Policy dagegen.

Derselbe Trick funktioniert für Game-AI. Du hast eine ganze Unit ([unit-reward-engineering.md](unit-reward-engineering.md)) damit verbracht, Reward-Terme für Distanz, Geschwindigkeit, Energie und Überleben handzubasteln — und selbst dann bewegt sich der resultierende NPC womöglich „korrekt", ohne sich je *natürlich* anzufühlen. Natürlichkeit, Charme, Bedrohlichkeit, Fairness — nichts davon sind Funktionen von `position.xyz`. Sie sind Funktionen des Geschmacks eines Designers.

Diese Unit gibt dir die Werkzeuge, um auf Geschmack zu optimieren.

!!! tip "Die Verbindung zu Alignment (erneut)"
    [Unit 9](unit-09.md) hat die Parallele zwischen Behavioral Cloning und Supervised Fine-Tuning (SFT) aufgezeigt. RLHF ist die nächste Sprosse. SFT bringt dem Modell bei, *was es sagen soll*. RLHF bringt ihm bei, *welche von zwei Aussagen besser ist*. In einem Spiel bringt SFT-artiges BC einem NPC bei, eine aufgenommene Patrouille zu kopieren; RLHF bringt ihm bei, dass *diese* Patrouille lebendiger wirkte als *jene*.

---

## 1 · Das Problem der Belohnungsfunktion im großen Maßstab

Du weißt bereits, dass die Belohnung das einzige Signal ist, das der Agent aus der Welt erhält ([unit-reward-engineering.md, §1](unit-reward-engineering.md)). Du kennst sparsame vs. dichte Belohnungen, potenzialbasiertes Shaping und die kanonischen Reward-Hacking-Fehlermodi — Bootagenten, die um Treibstoff-Pickups kreisen, Läufer, die Physikbugs ausnutzen.

Reward Engineering funktioniert, **solange das Ziel auf eine messbare physikalische Größe reduziert werden kann**. Du kannst Belohnungen schreiben für:

- Distanz zu einem Ziel
- Geschwindigkeit in eine gewählte Richtung
- Energieverbrauch
- Zeit bis zur Fertigstellung
- Kollisionskraft

Du kannst keine Belohnungen schreiben für:

- *„Dieser NPC fühlt sich lebendig an."*
- *„Dieser Bosskampf macht Spaß."*
- *„Diese Dialogwahl passt zur Figur."*
- *„Diese Animation sieht natürlich aus."*
- *„Dieses Patrouillenmuster wirkt zielgerichtet, nicht robotisch."*

Die Grenze ist nicht technisch. Sie ist **definitorisch**. Es gibt keine `is_natural(state) -> float`-Funktion, weil es keine vereinbarte Abbildung von Weltzustand auf Natürlichkeit gibt. Natürlichkeit lebt im Kopf eines Menschen.

### Warum Bewertungen nicht funktionieren und Vergleiche schon

Ein naiver Ausweg: Designer bitten, Trajektorien auf einer 1–10-Skala zu *bewerten*. Dann das Reward-Modell auf diesen Skalar-Labels regressieren.

Das scheitert aus gut dokumentierten Gründen:

1. **Inter-Annotator-Drift.** Alice's „7" ist Bobs „5". Ohne Verankerung ist die Skala bedeutungslos.
2. **Intra-Annotator-Drift.** Alice's „7" am Montag ist ihre „5" am Freitag nach dem Mittagessen.
3. **Anker-Sensitivität.** Wonach Alice zuerst bewertet hat, wird zu ihrem unbewussten Referenzpunkt.
4. **Granularität ist Fake.** Der Unterschied zwischen „7,3" und „7,4" ist Rauschen. Der Unterschied zwischen „dieser hier" und „jener da" ist eine echte Präferenz.

Paarweise Vergleiche umgehen alle vier. Ein Mensch, dem zwei Trajektorien gezeigt werden und der gefragt wird „welche ist besser?", liefert ein robustes, varianzarmes Signal — selbst über Annotatoren hinweg, selbst über Tage hinweg. Das ist ein bekanntes Ergebnis aus der Psychometrik (Thurstone, 1927) und aus der LLM-Trainingsliteratur (Christiano et al., 2017; Ouyang et al., 2022).

| Feedback-Format | Kognitive Last | Inter-Rater-Übereinstimmung | Information pro Klick |
|---|---|---|---|
| Skalare Bewertung (1–10) | Hoch | Niedrig (~40 %) | Hoch (in der Theorie) |
| Paarweiser Vergleich | Niedrig | Hoch (~80 %) | 1 Bit (in der Theorie) |
| Ranking von 4 | Mittel | Mittel | ~5 Bits |

Paarweise gewinnt, weil die Information pro Klick ehrlich ist. Ein Klick ist ein Klick. Die Bewertung „7,3" ist meistens Rauschen, das als Signal verkleidet ist.

---

## 2 · Sammeln von Präferenzdaten

Die RLHF-Trainingsschleife ist selbst eine Schleife aus drei Schleifen:

```
┌─────────────────────────────────────────────────────────┐
│  Outer loop: alternate data collection ↔ training       │
│                                                          │
│   1. Run current policy π in Godot → trajectories       │
│   2. Sample pairs (τ_A, τ_B) for designer to compare    │
│   3. Designer clicks preferred → preference dataset D   │
│   4. Train reward model r_φ on D                        │
│   5. Fine-tune π with PPO against r_φ (+ KL penalty)    │
│   6. Go to 1                                            │
└─────────────────────────────────────────────────────────┘
```

Jede Iteration verschiebt die Policy. Neue Trajektorien decken neue Bereiche des Zustandsraums ab. Neue Präferenzen korrigieren das Reward-Modell dort, wo es falsch lag. Das System konvergiert, wenn der Designer zwei Trajektorien nicht mehr zuverlässig unterscheiden kann.

### Das Bradley-Terry-Modell

Das mathematische Rückgrat ist das **Bradley-Terry-Modell** (1952). Nimm an, jede Trajektorie τ habe eine zugrunde liegende skalare Belohnung `r(τ)`. Die Wahrscheinlichkeit, dass ein Mensch `τ_A` gegenüber `τ_B` bevorzugt, ist:

$$P(\tau_A \succ \tau_B) = \sigma\left(r(\tau_A) - r(\tau_B)\right) = \frac{1}{1 + e^{-(r(\tau_A) - r(\tau_B))}}$$

Drei Eigenschaften machen das nützlich:

1. **Invarianz gegenüber additiven Konstanten.** `r` und `r + c` erzeugen identische Präferenzen. Wir lernen Belohnungen immer nur *bis auf eine Konstante*.
2. **Logistische Form.** Kleine Belohnungslücken → 50/50-Präferenz (Menschen sehen einen Münzwurf). Große Lücken → fast sichere Präferenz.
3. **Maximum-Likelihood-Zielfunktion ist konvex** in `r(τ_A) - r(τ_B)`, sodass das Training stabil ist.

Die Trainingszielfunktion für ein Reward-Modell `r_φ` auf einem Präferenzdatensatz `D = {(τ_w, τ_l)}` (w = winner, l = loser):

$$\mathcal{L}(\varphi) = -\mathbb{E}_{(\tau_w, \tau_l) \sim D}\left[ \log \sigma(r_\varphi(\tau_w) - r_\varphi(\tau_l)) \right]$$

Das ist nur Binary-Cross-Entropy auf einer Differenz von Scores. Derselbe Loss, den du für jedes paarweise Ranking-Problem schreiben würdest.

### Wie viele Vergleiche brauchst du?

| Aufgabenkomplexität | Nötige Vergleiche | Hinweise |
|---|---|---|
| Spielzeug (CartPole „Glätte") | 200–500 | Ein einzelner Designer an einem Nachmittag |
| Einzel-NPC-Verhalten | 2k–10k | Mehrere Designer über eine Woche |
| Komplette Figur (mehrere Verhaltensweisen) | 10k–50k | Crowdsource oder Active Learning |
| LLM-Skala (ChatGPT-Klasse) | 50k–500k+ | Industrielle Annotationsoperation |

Das Christiano-et-al.-Ergebnis (2017) auf Atari nutzte ~5500 menschliche Vergleiche, um menschliches Niveau auf Pong zu erreichen. Für Game-AI solltest du **mindestens 1k Paare pro Verhalten, das du formen willst** einplanen.

### Active Learning: frage nach dem, wobei du unsicher bist

Zufalls-Sampling von Trajektorienpaaren verschwendet Designerzeit. Wenn `r_φ(τ_A) - r_φ(τ_B) = 8.0` ist, ist das Modell bereits sicher. Der Klick bestätigt, was es schon weiß, und liefert fast keinen Gradienten.

Die Lösung: bevorzuge Paare, bei denen `|r_φ(τ_A) - r_φ(τ_B)|` **klein** ist, oder bei denen ein Ensemble von Reward-Modellen *uneins* ist. Konkret:

```python
def select_pair_for_annotation(reward_models, trajectory_pool):
    """Return the pair the ensemble is most uncertain about."""
    best_pair = None
    best_disagreement = -float("inf")
    for tau_a, tau_b in sample_pairs(trajectory_pool, n=200):
        # Score under each ensemble member
        diffs = [rm(tau_a).sum() - rm(tau_b).sum() for rm in reward_models]
        disagreement = float(np.std(diffs))
        if disagreement > best_disagreement:
            best_disagreement = disagreement
            best_pair = (tau_a, tau_b)
    return best_pair
```

Active Learning reduziert die Vergleiche pro Qualitätsstufe typischerweise um den Faktor 3–5. Es ist die größte einzelne praktische Verbesserung an RLHF-Daten-Pipelines.

### Eine praktische Annotations-Oberfläche

Die Oberfläche, die ein Designer verwendet, ist wichtiger als jeder Hyperparameter. Das kanonische Layout ist denkbar einfach:

```
┌─────────────────────────────┬─────────────────────────────┐
│       Trajectory A          │       Trajectory B          │
│  (Godot recording, loop)    │  (Godot recording, loop)    │
│                             │                             │
│   [▶ Play]  [⏸ Pause]      │   [▶ Play]  [⏸ Pause]      │
└─────────────────────────────┴─────────────────────────────┘
       [  A is better  ]     [  Tie  ]     [  B is better  ]
                      [ Skip / Can't tell ]
```

Drei Regeln aus produktiven RLHF-Pipelines:

- **Verstecke alle numerischen Scores.** Sieht der Designer `r_φ`, verankert er sich daran.
- **Randomisiere die A/B-Reihenfolge pro Paar.** Sonst klicken Designer standardmäßig „A".
- **Erlaube „Unentschieden" und „Überspringen".** Bei nicht unterscheidbaren Paaren eine Wahl zu erzwingen, injiziert Label-Rauschen.

---

## 3 · Das Reward-Modell trainieren

Architektonisch ist das Reward-Modell derselbe Beobachtungs-Encoder wie deine Policy, mit einem skalaren Head statt eines Aktions-Heads.

```
obs ──► [encoder: same conv/MLP as policy] ──► scalar r_φ(obs)
```

Für eine Trajektorie `τ = (o_0, o_1, ..., o_T)` ist die Trajektorien-Belohnung die **Summe** der Pro-Schritt-Belohnungen:

$$r_\varphi(\tau) = \sum_{t=0}^{T} r_\varphi(o_t)$$

Der Bradley-Terry-Loss operiert auf diesen Summen.

### Vollständige PyTorch-Implementierung

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader


class RewardModel(nn.Module):
    """Per-timestep reward predictor. Same architecture as the policy encoder."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, T, obs_dim) -> (batch, T)
        return self.net(obs).squeeze(-1)


def preference_loss(reward_model, obs_chosen, obs_rejected):
    """Bradley-Terry loss on paired trajectory observations."""
    r_chosen = reward_model(obs_chosen).sum(dim=1)     # sum over timesteps
    r_rejected = reward_model(obs_rejected).sum(dim=1)
    # Numerically stable log-sigmoid: -softplus(-(r_w - r_l))
    return torch.nn.functional.softplus(-(r_chosen - r_rejected)).mean()


class PreferenceDataset(Dataset):
    def __init__(self, chosen_obs, rejected_obs):
        self.chosen = torch.as_tensor(chosen_obs, dtype=torch.float32)
        self.rejected = torch.as_tensor(rejected_obs, dtype=torch.float32)

    def __len__(self):
        return len(self.chosen)

    def __getitem__(self, idx):
        return self.chosen[idx], self.rejected[idx]


def train_reward_model(chosen, rejected, obs_dim, epochs=20, lr=3e-4, batch_size=64):
    """Train a reward model on preference pairs and report validation accuracy."""
    n = len(chosen)
    split = int(0.9 * n)
    train_ds = PreferenceDataset(chosen[:split], rejected[:split])
    val_ds = PreferenceDataset(chosen[split:], rejected[split:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = RewardModel(obs_dim)
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for c, r in train_loader:
            loss = preference_loss(model, c, r)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Validation: how often does the model agree with the human ranking?
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for c, r in val_loader:
                r_c = model(c).sum(dim=1)
                r_r = model(r).sum(dim=1)
                correct += int((r_c > r_r).sum())
                total += len(c)
        print(f"epoch {epoch:02d}  val_acc={correct / max(total, 1):.3f}")

    return model
```

### Validation Accuracy lesen

Die wichtigste Einzelmetrik für ein Reward-Modell ist die **Held-out-Präferenz-Genauigkeit**: wie oft stimmt das Modell auf Paaren, die es nie gesehen hat, mit dem Menschen überein?

| Val-Accuracy | Was es bedeutet | Was zu tun ist |
|---|---|---|
| ~50 % | Münzwurf — kein Signal | Mehr Daten sammeln, Labels auf Rauschen prüfen |
| 60–70 % | Schwaches Signal | Akzeptabel für frühe Iteration; erwarte verrauschtes PPO |
| 70–85 % | Gesundes Signal | Weiter zum PPO-Fine-Tuning |
| 85–95 % | Starkes Signal | Exzellent; auf Label-Leakage prüfen |
| >95 % | Verdächtig | Mit ziemlicher Sicherheit Leakage — gleiche Trajektorien in Train und Val |

Beachte die Obergrenze: der **menschliche Bayes-Fehler** liegt selten über 90–95 %. Zwei Designer sind bei knappen Entscheidungen etwa zu 5–15 % uneinig. Ein Reward-Modell, das die Inter-Annotator-Übereinstimmungsrate übersteigt, hat sich etwas gemerkt, was es nicht sollte.

!!! warning "Train/Val-Splits müssen nach *Trajektorie* erfolgen, nicht nach *Paar*"
    Erscheint Trajektorie `τ_42` sowohl in Trainings- als auch in Validierungspaaren, merkt sich das Modell trivial ihren Score. Splitte zuerst nach Trajektorien-ID, bilde dann Paare innerhalb jedes Splits.

---

## 4 · PPO mit gelernter Belohnung (die klassische RLHF-Pipeline)

Sobald du ein trainiertes Reward-Modell `r_φ` hast, fine-tunst du deine Policy mit PPO — genau dem PPO, das du in [unit-ppo-deep.md](unit-ppo-deep.md) geschrieben hast — aber mit `r_φ(obs)` als Pro-Schritt-Belohnung statt der Belohnung der Umgebung.

Implementatorisch ist das ein **Wrapper um die Godot-Umgebung**, strukturell identisch mit dem `RNDGodotEnv`-Wrapper aus [unit-curiosity.md](unit-curiosity.md):

```python
import gymnasium as gym
import numpy as np
import torch


class RLHFGodotEnv(gym.Wrapper):
    """Wrap a Godot env so its rewards come from a learned reward model
    minus a KL penalty against a frozen reference policy."""

    def __init__(self, env, reward_model, ref_policy, current_policy, kl_coef=0.05):
        super().__init__(env)
        self.reward_model = reward_model.eval()
        self.ref_policy = ref_policy.eval()       # frozen snapshot from before RLHF
        self.current_policy = current_policy      # updated by PPO
        self.kl_coef = kl_coef

    def step(self, action):
        obs, _r_env, terminated, truncated, info = self.env.step(action)

        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            r_rm = float(self.reward_model(obs_t).squeeze())

            # Per-step single-sample log-ratio between current and reference policy
            # (a one-sample MC estimator of KL — not the KL itself; use a multi-sample
            # estimator like Schulman's `(r-1) - log r` form if you want a true KL.)
            logp_cur = self.current_policy.log_prob(obs_t, action)
            logp_ref = self.ref_policy.log_prob(obs_t, action)
            log_ratio = float(logp_cur - logp_ref)

        r_total = r_rm - self.kl_coef * log_ratio
        info["r_rm"] = r_rm
        info["log_ratio_to_ref"] = log_ratio
        return obs, r_total, terminated, truncated, info
```

Diese gewrappte Umgebung übergibst du dann an SB3-PPO, genau wie in früheren Units:

```python
from stable_baselines3 import PPO

env = RLHFGodotEnv(make_godot_env("NPCGuard.x86_64"),
                   reward_model, ref_policy, current_policy,
                   kl_coef=0.05)
model = PPO("MlpPolicy", env, n_steps=2048, batch_size=64,
            learning_rate=3e-4, verbose=1, tensorboard_log="./rlhf_tb/")
model.learn(total_timesteps=500_000)
```

### Warum die KL-Strafe essenziell ist

Ohne den KL-Term wird PPO entdecken, dass das Reward-Modell Schlupflöcher hat. Es ist ein endliches neuronales Netz, trainiert auf endlichen Daten. Überall dort, wo der Score des Modells nicht dem tatsächlichen Geschmack eines Designers entspricht, wird PPO genau das finden und ausnutzen.

Die KL-Strafe `β · KL(π || π_ref)` sagt: „bleib in der Nähe der Referenz-Policy, von der du gestartet bist". Die Referenz-Policy wurde mit Reward Engineering oder Imitation Learning trainiert — sie ist *grob* sinnvoll. Die KL-Strafe begrenzt, wie weit PPO bei der Jagd nach Reward-Modell-Schlupflöchern von dieser sinnvollen Region abdriften kann.

Der volle RLHF-Reward ist:

$$r_{\text{total}}(o_t, a_t) = r_\varphi(o_t) - \beta \cdot \text{KL}\bigl(\pi(\cdot \mid o_t)\,\|\,\pi_{\text{ref}}(\cdot \mid o_t)\bigr)$$

Auswahl von `β`:

| β-Wert | Effekt | Wann verwenden |
|---|---|---|
| 0,0 | Kein Anker — sofortiges Reward Hacking | Fast nie |
| 0,01 | Lockerer Anker — Policy exploriert | Wenn Ref-Policy mittelmäßig ist |
| 0,05–0,1 | Standardbereich | Default-Startpunkt |
| 0,5 | Enger Anker — bewegt sich kaum | Wenn Ref-Policy bereits sehr gut ist |

Ein praktisches adaptives Schema (Stiennon et al., 2020): ziele auf ein festes KL-Budget (etwa 10 nats pro Episode) ab. Liegt das realisierte KL über dem Ziel, erhöhe `β`. Unter dem Ziel, senke `β`. SB3 liefert das nicht out-of-the-box — schreib es als Callback.

---

## 5 · DPO — Direct Preference Optimization

2023 veröffentlichten Rafailov et al. ein verblüffendes Ergebnis: **du kannst das Reward-Modell komplett überspringen**. Ihr Algorithmus, DPO (Direct Preference Optimization), formuliert RLHF als einen einzigen überwachten Loss direkt auf Präferenzdaten neu.

### Die Kernerkenntnis

Unter dem RLHF-Ziel mit KL-Strafe hat die optimale Policy eine geschlossene Beziehung zum Reward-Modell:

$$\pi^*(a \mid s) = \frac{1}{Z(s)} \pi_{\text{ref}}(a \mid s) \exp\!\left(\tfrac{1}{\beta} r(s, a)\right)$$

Auflösen nach `r`:

$$r(s, a) = \beta \log \frac{\pi^*(a \mid s)}{\pi_{\text{ref}}(a \mid s)} + \beta \log Z(s)$$

Wenn man diesen Ausdruck für `r` zurück in den Bradley-Terry-Präferenz-Loss einsetzt, kürzen sich die `Z(s)`-Terme zwischen der gewählten und der abgelehnten Trajektorie. Das Resultat ist ein Loss, der vollständig in `π_θ` und `π_ref` ausgedrückt ist — nirgendwo ein separates Reward-Modell:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(\tau_w, \tau_l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(\tau_w)}{\pi_{\text{ref}}(\tau_w)} - \beta \log \frac{\pi_\theta(\tau_l)}{\pi_{\text{ref}}(\tau_l)}\right)\right]$$

Das ist *überwachtes Lernen*. Kein Umgebungs-Rollout, kein PPO, kein Training eines Reward-Modells. Nur Gradient Descent auf Präferenzpaaren.

### Minimale DPO-Implementierung

```python
def dpo_loss(policy, ref_policy, obs_chosen, act_chosen, obs_rejected, act_rejected, beta=0.1):
    """DPO loss. policy is trainable; ref_policy is frozen."""
    # Sum log-probs over the trajectory
    logp_w = policy.log_prob(obs_chosen, act_chosen).sum(dim=1)
    logp_l = policy.log_prob(obs_rejected, act_rejected).sum(dim=1)

    with torch.no_grad():
        logp_w_ref = ref_policy.log_prob(obs_chosen, act_chosen).sum(dim=1)
        logp_l_ref = ref_policy.log_prob(obs_rejected, act_rejected).sum(dim=1)

    logits = beta * ((logp_w - logp_w_ref) - (logp_l - logp_l_ref))
    return torch.nn.functional.softplus(-logits).mean()
```

Das ist der ganze Algorithmus. Trainiere deine Policy, indem du diesen Loss über deinen Präferenzdatensatz minimierst. Kein Reward-Modell. Keine Umgebung.

### RLHF vs. DPO

| | RLHF + PPO | DPO |
|--|-----------|-----|
| Reward-Modell nötig? | Ja | Nein |
| Online-RL-Training? | Ja | Nein (überwacht) |
| Präferenzdaten nötig | Ja | Ja |
| Sample-Effizienz | Mittel | Hoch (keine RL-Schleife) |
| Reward-Hacking-Risiko | Ja (mit KL gemildert) | Nein |
| Kann weiter aus neuen Daten lernen? | Ja (neu sammeln, neu trainieren) | Ja (Datensatz erweitern) |
| Speicherkosten | 2 Netze (Policy + RM) | 2 Netze (Policy + Ref) |
| Compute-Kosten | Hoch (Rollouts + PPO + RM) | Niedrig (ein überwachter Durchlauf) |
| Am besten für | Kontinuierliche Online-Verbesserung | Fester Präferenzdatensatz |
| Hyperparameter-Aufwand | Hoch (β, lr, n_steps, etc.) | Niedrig (β, lr) |

DPO ist der richtige Startpunkt für die meisten Spielprojekte. Wenn dein Präferenzdatensatz fest und deine Referenz-Policy sinnvoll ist, holt DPO 80 % der RLHF-Qualität mit 20 % des Engineering-Aufwands. Greife erst zu vollem PPO-basiertem RLHF, wenn du in einer kontinuierlichen Schleife neue Trajektorien generieren und frische Präferenzen sammeln musst.

!!! tip "DPO erbt die Qualität der Referenz-Policy"
    DPO kann deine Policy nur *in die Nähe* von `π_ref` bewegen. Ist `π_ref` schlecht (z. B. eine zufällige Initialisierung), produziert DPO eine etwas-besser-als-schlechte Policy. Initialisiere DPO immer mit einem vernünftigen BC- oder PPO-Checkpoint, niemals mit zufälligen Gewichten.

---

## 6 · Reward Hacking

Reward Hacking hast du in [unit-reward-engineering.md, §7](unit-reward-engineering.md) kennengelernt. RLHF gibt dem Phänomen ein neues Gesicht: das **Reward-Modell selbst** ist das, was gehackt wird. Die Policy entdeckt Zustände, die das Modell hoch bewertet, obwohl kein Designer das würde.

### Konkrete Fehlermodi

**Sprachmodell-Klassiker.** Frühe RLHF-Chatbots lernten, jede Antwort mit *„As an AI assistant..."* zu beginnen, weil dieser String mit hohen Präferenz-Scores in den Trainingsdaten korrelierte (Annotatoren bevorzugten gut formatierte Antworten). Das Modell stellte dem Phrase dann sogar lockeren Antworten voran. Reward-Modell sagte: hilfreich. Menschen sagten: nervig.

**Godot-Beispiel.** Trainiere einen NPC-Wächter mit RLHF. Dein Reward-Modell lernte, dass Designer Trajektorien bevorzugen, bei denen der Wächter an Kreuzungen die Patrouillenroute *anschaut*. PPO entdeckt, dass der Wächter bewegungslos an einer Kreuzung stehen und sich kontinuierlich drehen kann, um die Route anzuschauen. Das Reward-Modell liebt es. Der Designer, der das Level beobachtet, sagt, es sieht kaputt aus.

**Visuelle Policy-Manipulation.** Ein bildbasierter RLHF-Agent lernte, dass das Reward-Modell Zuständen mit einer bestimmten Lichtverhältnis hohe Scores zuwies. Die Policy positionierte sich, um diese Lichtverhältnis zu maximieren, statt die Aufgabe zu erfüllen. Das Reward-Modell hatte eine spurious correlation aus den Präferenzdaten aufgenommen.

### Erkennung

| Signal | Was es anzeigt |
|---|---|
| `policy/kl_to_ref` springt beim PPO an | Policy hat sich in ein Reward-Modell-Schlupfloch bewegt |
| Reward-Modell-Score steigt, während die visuelle Qualität nachlässt | Klassisches Goodhart's Law |
| Ensemble-Uneinigkeit (`std(r_φ)`) explodiert bei Policy-Rollouts | Modelle sind sich auf Trainingsdaten einig, uneins bei neuen Exploits |
| Inverse Korrelation zwischen `r_φ` und einer Held-out-Skalarmetrik | Das Modell verfolgt einen Proxy |

Das *operationale* Signal ist `kl_to_ref` plus eine wiederkehrende qualitative Überprüfung. Richte den Trainings-Callback so ein, dass er alle 50k Schritte einen Viz-Checkpoint ablegt. Liegt `kl_to_ref` über einer Schwelle *und* sieht die Viz seltsam aus, stoppe das Training.

### Abmilderung

1. **KL-Strafe `β` ↑.** Die billigste Lösung. Engerer Anker → weniger Raum zur Ausnutzung.
2. **Reward-Modell-Ensembles.** Trainiere N Reward-Modelle mit verschiedenen Seeds. Verwende das **Minimum** (pessimistisch) oder das **Mittel minus k·std** (unsicherheitsbestraft). Die Schlupflöcher, die ein einzelnes Modell findet, sind nicht die Schlupflöcher, die ein anderes Modell findet; die Schnittmenge ist viel kleiner als die Vereinigung.
3. **Präferenzdaten aktualisieren.** Während sich die Policy verschiebt, driften ihre Trajektorien aus der Trainingsverteilung des Reward-Modells heraus. Ziehe periodisch frische Trajektorien aus der aktuellen Policy und sammle neue Präferenzen darauf.
4. **Early Stopping.** Trainiere RLHF nicht bis zur Konvergenz des *Reward-Modell-Losses*. Stoppe, wenn das **qualitative** Signal plateaut.
5. **Adversariale Review.** Einmal pro Woche versucht ein Designer gezielt, Verhalten zu finden, das das Reward-Modell hoch bewertet, das er aber hasst. Diese werden zu Hard Negatives im nächsten Präferenz-Batch.

!!! warning "RLHF beseitigt Reward Hacking nicht — es verschiebt es"
    Mit handgebauten Rewards hackt die Policy die Belohnungsfunktion. Mit RLHF hackt die Policy das Reward-*Modell*. Das neue Hacking ist im Voraus schwerer vorherzusagen (du kannst das Reward-Modell nicht so lesen wie einen GDScript-Reward), aber zur Laufzeit leichter zu erkennen (es zeigt sich als KL-Spike).

---

## 7 · Godot-Beispiel — NPC-Verhaltenspräferenzen

Konkreter Walkthrough. Die Aufgabe: ein Burgwächter-NPC, der *natürlich* patrouillieren soll. Handgebaute Reward-Terme (zurückgelegte Distanz, Zeit auf Patrouillenpfad, verbrauchte Energie) liefern einen Wächter, der technisch patrouilliert, aber robotisch wirkt. RLHF wird das richten.

### Schritt 1 — Bootstrap mit Vanilla-PPO

Verwende das Reward Engineering aus früheren Units, um eine Baseline-Policy zu erhalten:

```python
# bootstrap_guard.py
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="./NPCGuard.x86_64", n_parallel=8, speedup=8)
model = PPO("MlpPolicy", env, n_steps=2048, batch_size=128, verbose=1)
model.learn(total_timesteps=2_000_000)
model.save("guard_baseline.zip")   # this becomes π_ref
```

Der Baseline-Wächter geht die Route. Er ruckelt auch gelegentlich, wählt suboptimale Ecken und sieht aus wie ein RL-Agent. Gut genug, um eine Referenz-Policy zu sein.

### Schritt 2 — Trajektorienpaare aufnehmen

Wir brauchen diverse Trajektorien zum Vergleichen. Generiere sie, indem du Seeds und Hyperparameter variierst:

```python
# record_trajectories.py
import numpy as np
from stable_baselines3 import PPO

def record_trajectory(model_path, env, n_steps=400, seed=0):
    model = PPO.load(model_path)
    obs, _ = env.reset(seed=seed)
    obs_log, act_log, frame_log = [], [], []
    for _ in range(n_steps):
        act, _ = model.predict(obs, deterministic=False)
        obs, _, done, _, _ = env.step(act)
        obs_log.append(obs)
        act_log.append(act)
        frame_log.append(env.render(mode="rgb_array"))
        if done:
            break
    return dict(obs=np.array(obs_log),
                act=np.array(act_log),
                frames=np.array(frame_log))

# Generate 200 trajectories across different seeds and slight HP variations
trajectories = [record_trajectory("guard_baseline.zip", env, seed=s) for s in range(200)]
```

Speichere die aufgenommenen Frames für jede als `.mp4`, damit der Designer sie anschauen kann.

### Schritt 3 — GDScript-Recorder für In-Engine-Wiedergabe

Möchtest du lieber direkt in Godot aufnehmen (authentischere Vorschau für den Designer), lege das auf dein Agenten-Skript:

```gdscript
extends RigidBody3D
@onready var _ai = $AIController3D

var recording: Array = []
var record_mode: bool = false

func _physics_process(_delta):
    if record_mode:
        recording.append({
            "pos": global_position,
            "rot": global_rotation,
            "vel": linear_velocity,
            "action": _ai.last_action,
        })

func dump_recording(path: String):
    var f = FileAccess.open(path, FileAccess.WRITE)
    f.store_string(JSON.stringify(recording))
    f.close()
    recording.clear()

func play_recording(path: String):
    # Replay mode: position the agent each physics tick from file
    var f = FileAccess.open(path, FileAccess.READ)
    var data = JSON.parse_string(f.get_as_text())
    for frame in data:
        global_position = frame.pos
        global_rotation = frame.rot
        await get_tree().physics_frame
```

Zwei solcher Instanzen nebeneinander geben dem Designer eine Echtzeit-A/B-Vorschau.

### Schritt 4 — Präferenzen sammeln

Baue die einfachstmögliche Annotations-Webseite — zwei `<video>`-Tags, drei Buttons. Persistiere Klicks in einer JSON-Datei:

```python
# preference_server.py — simplified
from fastapi import FastAPI
import json, random

app = FastAPI()
prefs = []

@app.get("/next_pair")
def next_pair():
    a, b = random.sample(range(200), 2)
    return {"a": f"/clips/{a}.mp4", "b": f"/clips/{b}.mp4",
            "id_a": a, "id_b": b}

@app.post("/submit")
def submit(id_a: int, id_b: int, choice: str):
    # choice in {"a", "b", "tie", "skip"}
    if choice in ("a", "b"):
        prefs.append({"chosen": id_a if choice == "a" else id_b,
                      "rejected": id_b if choice == "a" else id_a})
        json.dump(prefs, open("preferences.json", "w"))
    return {"ok": True}
```

Lass es laufen, bitte die Designer, ~1000 Paare zu labeln. Realistische Zeit: ~3 Stunden Designerzeit, verteilt über eine Woche.

### Schritt 5 — Reward-Modell trainieren, dann PPO

Baue `chosen_obs`- und `rejected_obs`-Arrays aus `preferences.json`, übergib sie an `train_reward_model()` aus §3. Dann wrappe die Umgebung mit `RLHFGodotEnv` aus §4 und führe ein PPO-Fine-Tuning der Bootstrap-Policy aus.

### Erwartete Ergebnisse

| Policy | Patrouillen-Abschlussrate | Designer-Bewertung (1–10) | Wirkt natürlich? |
|---|---|---|---|
| Zufall | 0 % | 1,2 | Nein |
| Hand-engineered Reward (PPO-Bootstrap) | 95 % | 5,1 | „Robotisch, aber funktional" |
| Behavioral Cloning des Designers | 60 % | 6,4 | „Natürlich, aber unzuverlässig" |
| RLHF auf Bootstrap | 93 % | 8,2 | „Ja" |

Die RLHF-Policy behält die *Kompetenz* der handgebauten Baseline (weil die KL-Strafe sie dort verankert) und gewinnt gleichzeitig die *Natürlichkeit* der vom Menschen bevorzugten Trajektorien.

---

## 8 · RLHF für Game-AI vs. LLMs

Der Algorithmus ist derselbe. Die Domänen unterscheiden sich in drei praktischen Punkten.

| | LLM-RLHF | Game-AI-RLHF |
|---|---|---|
| Trajektorienformat | Token-Sequenzen | (obs, action)-Zeitschritte |
| „Episoden"-Länge | Zehn bis Tausende von Tokens | Hunderte bis Tausende Physikschritte |
| Kosten der Präferenzbeschaffung | $$$ (Crowd-Worker, langes Lesen) | $ (Designer schaut 10-Sek.-Clip) |
| Datenskala | 10k–500k Paare | 500–10k Paare |
| Referenz-Policy | SFT-Modell | BC- oder PPO-Bootstrap |
| Reward-Hack-Fehlermodus | Sycophancy, Verweigerungen, Wiederholung | Animationsartefakte, kaputte Bewegung |
| Evaluation | Held-out-Präferenz, Win-Rate vs. Baseline | Designer-Bewertungen + Viz-Checkpoint |

### Wann RLHF *nicht* zu verwenden ist

Kannst du eine funktionierende Belohnungsfunktion schreiben, **schreib sie einfach**. RLHF ist dramatisch mehr Engineering-Aufwand als Reward Engineering. Es lohnt sich nur, wenn:

1. Das Ziel echt subjektiv ist („sieht natürlich aus", „fühlt sich fair an").
2. Mehrere Stakeholder über die exakte Belohnungsfunktion uneins sind, sich aber bei Präferenzen einig sind, wenn Trajektorien gezeigt werden.
3. Der handgebaute Reward immer wieder Exploits hervorbringt, egal wie du flickst.
4. Du eine Baseline *bereits ausgeliefert hast* und sie mit echten Spielerdaten fine-tunen willst.

Für „Zeit zum Ziel minimieren" oder „aufrecht bleiben": RLHF ist Overkill. Nimm Reward Engineering.

### Wann RLHF glänzt

Der Lackmustest: *„Ich erkenne gutes Verhalten, wenn ich es sehe."* Kann ein Designer zuverlässig die bessere von zwei Trajektorien auswählen, aber die Regel nicht artikulieren, übertrifft RLHF jeden handgebauten Reward. Dieses Muster wiederholt sich bei Animationsqualität, Pacing, Schwierigkeitsgefühl, Charakterpersönlichkeit, Levelfairness und Combat-Readability.

---

## 9 · Stretch Goals

1. **CartPole-„Glätte"-Reward-Modell.** Trainiere PPO auf CartPole. Nimm 300 Trajektorien auf. Definiere eine synthetische „Präferenz": Trajektorie A wird gegenüber B bevorzugt, gdw. `mean(|action_t - action_{t-1}|)` in A kleiner ist (sanftere Kontrolle). Lass den Labeler eine Python-Funktion sein, die den Menschen simuliert. Trainiere ein Reward-Modell auf diesen Präferenzen. Fine-tune CartPole mit PPO unter Verwendung des gelernten Rewards. Vergleiche mit PPO unter dem handgebauten Smoothness-Reward `-λ · |Δa|`. Plotte beide Lernkurven. Das gelernte Reward-Modell sollte dem handgebauten Reward bis auf ~10 % Sample-Effizienz entsprechen — die Lücke ist dein Maß dafür, wie schwer es ist, das Offensichtliche zu lernen.

2. **Offline-DPO auf dem Imitations-Datensatz.** Nimm die Demonstrationen, die du in [unit-09.md](unit-09.md) aufgenommen hast. Synthetisiere „abgelehnte" Trajektorien, indem du Aktionsrauschen hinzufügst. Führe DPO auf diesen chosen/rejected-Paaren aus. Vergleiche mit normalem BC. DPO mit Rauschen-als-abgelehnt sollte eine Policy produzieren, die robuster gegen Störungen ist als BC allein — miss, indem du zur Testzeit Beobachtungsrauschen injizierst und den Return vergleichst.

3. **Minimales Annotations-Tool.** Baue eine 200-Zeilen-Python-App (Streamlit, Flask oder reines Tkinter), die zwei MP4-Clips lädt, sie nebeneinander abspielt und den Klick in eine JSONL-Datei schreibt. Verwende es, um 500 echte Präferenzen zu einem dir wichtigen Verhalten zu labeln. Begrenze das Labeln auf eine Stunde. Berichte: wie viele Paare/Stunde hast du geschafft? Wie viele wirkten wie „Unentschieden"?

4. **Effizienzkurve für Präferenzdaten.** Trainiere Reward-Modelle auf N ∈ {50, 100, 250, 500, 1000, 2500} Präferenzpaaren (subgesampelt aus demselben Gesamtdatensatz). Plotte die Held-out-Präferenzgenauigkeit gegen N. Finde den Knick — den Punkt, ab dem mehr Daten nicht mehr helfen. Das ist dein projektspezifisches Mindest-Label-Budget.

5. **Reward-Modell-Ensemble zur Hacking-Erkennung.** Trainiere 5 Reward-Modelle mit verschiedenen Seeds auf denselben Präferenzdaten. Logge beim PPO-Fine-Tuning für jeden gerollouteten Zustand die *Standardabweichung* über das Ensemble. Plotte sie über das Training. Die std sollte steigen, wenn die Policy in Regionen vordringt, bei denen das Ensemble uneins ist — das sind die verdächtigen Zustände. Inspiziere manuell 10 solche Zustände und bestätige, dass sie wie Hacking-Kandidaten aussehen.

---

## Was kommt als Nächstes

Du hast die Schleife in der Alignment-Geschichte geschlossen, die dieser Kurs erzählt hat: Reward Engineering ([unit-reward-engineering.md](unit-reward-engineering.md)) für objektive Ziele, Imitation Learning ([unit-09.md](unit-09.md)) wenn du Demonstrationen hast, und nun RLHF, wenn du nur den Geschmack eines Designers hast. Jedes erschließt eine Klasse von Aufgaben, die das vorherige nicht erreichen konnte.

Wenn du dir nur eine Sache aus dieser Unit merkst: **die Belohnung ist das Ziel, aber das Ziel sitzt in jemandes Kopf.** RLHF ist die Brücke vom Kopf zum Gradienten.

**Offline-RL:** Was, wenn du die Umgebung gar nicht ausführen kannst — nur ein statischer Datensatz vergangener Transitions? Offline-RL extrahiert eine Policy aus festen Daten ohne weitere Umgebungsinteraktion.

[← Imitation Learning](unit-09.md) · [Kursstartseite](index.md) · [→ Offline-RL](unit-offline-rl.md)
