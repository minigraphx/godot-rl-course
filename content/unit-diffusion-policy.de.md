# Diffusion Policy — Multimodale Aktionsgenerierung für Robotersteuerung

Du hast den Kurs damit verbracht, Gaußsche Policies zu bauen. PPO gibt eine Gaußverteilung über Aktionen aus. SAC gibt eine gequetschte Gaußverteilung über Aktionen aus. Selbst die durch Behaviour Cloning trainierte Imitations-Policy aus Unit 9 war ein Gaußscher Regressor. Diese Unit handelt von dem Moment, in dem diese Abstraktion bricht — und was du tun kannst, wenn das passiert. **Diffusion Policy** (Chi et al., 2023) ersetzt den Gauß-Kopf durch ein vollständiges Denoising-Diffusionsmodell und beseitigt die Einzelpeak-Limitierung, die jedem kontinuierlichen Steuerungs-Agenten, den du bisher trainiert hast, still geschadet hat. Sie ist die dominante Policy-Klasse in der modernen Manipulationsforschung, und der Rest dieser Unit zeigt dir warum.

[← SAC](unit-sac.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~40 min · Training: ~30 min GPU / ~2 h CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    Eine 2-D-bimodale Spielaufgabe, bei der SAC zum Mittelwert kollabiert und Diffusion Policy beide Modi wiederherstellt · Die DDIM-Denoising-Trajektorie live in Godot gerendert (Aktionsvektor animiert sich von Rauschen zu sauberem Griff) · Ein Side-by-Side-Video von Aktions-Chunks — zitterige SAC-Regelung vs. glatte Diffusion-Policy-Chunks auf einem 7-DoF-Arm

---

## 0 · Das Multimodalitätsproblem, das SAC nicht lösen kann

Jede Policy, die du in diesem Kurs trainiert hast, gibt eine Gaußverteilung `N(μ(s), σ(s))` aus. PPO lernt `μ` und `σ` gemeinsam. SAC lernt sie unter dem Maximum-Entropie-Ziel. Behaviour Cloning passt sie mit Maximum-Likelihood an Demonstrationen an. Alle drei teilen dieselbe fundamentale Limitierung: **eine Gaußverteilung hat genau einen Peak.**

Das ist fein, wenn die optimale Aktion bei Zustand `s` ein einzelner Punkt ist. Aber viele reale Aufgaben haben mehrere gleichwertig gute Lösungen:

- **Eine Tasse aufnehmen.** Du kannst sie von links oder von rechts greifen. Beide Griffe erzielen dieselbe Belohnung. Beide tauchen in menschlichen Demonstrationen auf.
- **Um eine Säule navigieren.** Links und rechts herum erreichen beide das Ziel. Ein Münzwurf ist eine gültige Policy.
- **Eine Go-Eröffnung wählen.** Mehrere stilistisch unterschiedliche Eröffnungen erreichen dieselbe langfristige Gewinnrate.
- **Ein Ziel mit einem 7-DoF-Arm erreichen.** Ein redundanter Manipulator hat einen unendlichdimensionalen Nullraum von Gelenkkonfigurationen, die alle dieselbe Endeffektor-Pose erzeugen.

In jedem dieser Fälle ist die Aktionsverteilung, die du lernen *solltest*, multimodal — zwei oder mehr getrennte Peaks gleich guter Aktionen. Eine Gaußverteilung kollabiert beim Anpassen einer solchen Verteilung auf den Mittelwert der Modi. Dieser Mittelwert ist oft katastrophal schlecht.

```python
# Bimodal task: go to x=+1 OR x=-1 (both equally rewarded, x=0 hits a wall)
# Demo data: half the trajectories go to +1, half go to -1.
#
# Gaussian BC policy:
#   maximum-likelihood fit -> mu = 0, sigma ~ 1.0
#   at execution: samples cluster around x=0 -> hits the wall every time
#
# Diffusion policy:
#   the denoiser learns a bimodal score field with two attractors at +/-1
#   at execution: each rollout samples from either mode -> never hits the wall
```

Die Symptome davon hast du bereits in früheren Units gesehen. Die `unit-09`-Demos (Imitation Learning), die zwei menschliche Lehrer mischten, produzierten oft einen Agenten, der keines von beidem gut machte — das war Gaußsche Modus-Mittelung. Das geschickte Greif-Beispiel in `unit-robotics` plateaute bei ~60 % Erfolg — das war Modus-Mittelung über links/rechts-Griffe. Und der Lokomotion-Stabilitätsfix in `unit-locomotion`, der ein Curriculum brauchte, um Demos in Richtung eines *einzigen* Gangs zu biasen — das war ein Workaround für dieselbe Gauß-Limitierung.

Diffusion Policy entfernt die Limitierung direkt. Der Policy-Kopf gibt nicht mehr `(μ, σ)` aus. Stattdessen gibt er eine gelernte Wahrscheinlichkeitsverteilung über den gesamten Aktionsraum aus, gesampelt durch iteratives Denoising. Welche Form die Demonstrationsdaten auch haben — unimodal, bimodal, ringförmig, bananenförmig — das Diffusionsmodell kann sie repräsentieren.

Der Preis, den du zahlst, ist Rechenleistung: statt eines einzigen Forward Passes führst du `T` Denoising-Schritte pro Aktion aus. Der Rest dieser Unit handelt davon, (a) zu verstehen, *warum* Denoising beliebige Verteilungen darstellen kann, (b) die Architektur zu implementieren und (c) `T` klein genug zu halten, damit die Policy mit 60 Hz in einer Godot-Schleife läuft.

---

## 1 · Diffusionsmodelle — von Bildern zu Aktionen (optional beim ersten Lesen)

!!! note "Erster Durchgang? Überfliege oder überspringe diesen Abschnitt."
    Der Hands-on-Pfad läuft durch Abschnitt 0 (das Multimodalitätsproblem) und die Abschnitte 2–7 (Architektur und Code, Action Chunking, DDIM-Inferenzkosten, der Vergleich mit SAC und PPO, der Godot-Trainings-Workflow und wann Diffusion die richtige Wahl ist). Dieser Abschnitt ist das *Warum* hinter dem Code: die `loss()`- und `sample()`-Methoden in Abschnitt 2 implementieren genau diese Formeln — du kannst also erst bauen und später für die Herleitung zurückkommen.

Diffusionsmodelle wurden für die Bildgenerierung erfunden (Sohl-Dickstein 2015; Ho et al. 2020, DDPM). Dieselbe Mechanik überträgt sich ohne konzeptionelle Änderung auf die Aktionsgenerierung — wir tauschen nur die Datendimension von `H×W×3` auf `act_dim` und fügen eine Observation als Konditionierungs-Input hinzu.

Die Kernidee hat zwei Hälften: einen festen **Vorwärtsprozess**, der Daten durch Rauschen zerstört, und einen gelernten **Rückwärtsprozess**, der sie wiederherstellt.

**Vorwärtsprozess (kein Lernen).** Beginne mit einem sauberen Datenpunkt `x_0` (einem echten Aktionsvektor aus einer Demonstration). Füge wiederholt eine kleine Menge Gaußsches Rauschen über `T` Schritte hinzu, bis das Sample bei `t=T` von reinem Rauschen ununterscheidbar ist. Die gesamte Trajektorie ist geschlossen-form:

```
# Forward process — add noise in closed form
# beta_t is a fixed noise schedule, alpha_t = 1 - beta_t,
# alpha_bar_t = product_{s<=t} alpha_s (cumulative product)

x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)

# At t=0:  x_0 = the real action (no noise)
# At t=T:  x_T ~ N(0, I)  (pure noise, x_0 has decayed away)
```

Du musst das nie lernen — es ist ein deterministisches Rezept.

**Rückwärtsprozess (gelernt).** Trainiere ein neuronales Netz `eps_theta(x_t, t, s)`, um das bei Schritt `t` hinzugefügte Rauschen vorherzusagen, konditioniert auf die Observation `s`. Mit dieser Vorhersage können wir einen Denoising-Schritt zurück Richtung `x_0` machen:

```
# Reverse process — one denoising step
# Subtract predicted noise to estimate x_0, then re-noise to step t-1
x_{t-1} = mu_theta(x_t, t, s) + sigma_t * z,   z ~ N(0, I)

# Training objective: predict the noise, simple regression
L = E_{x_0, t, eps} [ || eps - eps_theta(x_t, t, s) ||^2 ]
# where x_t is built from x_0 by the forward formula above
# and s is the conditioning observation
```

**Bei Inferenz.** Beginne mit `x_T ~ N(0, I)`, führe den Rückwärtsprozess für `T` Schritte aus, und heraus kommt ein Sample aus der bedingten Verteilung `p(action | obs)`. Kein Mittelwert, kein Varianz-Kopf, keine Gauß-Annahme — nur iteratives Denoising.

Warum schlägt das eine Gaußverteilung für die Aktionsgenerierung? Der Rückwärtsprozess ist eine Folge von `T` gelernten bedingten Übergängen. Komponiert können sie **beliebig komplexe Verteilungen** darstellen, einschließlich der bimodalen Tassengriff-Verteilung aus Abschnitt 0. Der Denoiser muss keinen „Modus wählen" — er lernt ein Vektorfeld (den Gradienten von `log p`, äquivalent zu einer Score-Funktion), das zwei Attraktoren hat, und welcher Attraktor ein bestimmter Rollout konvergiert, hängt vom zufälligen Anfangsrauschen `x_T` ab. Wirf am Anfang des Denoising die Münze, und du bekommst einen Links- oder Rechtsgriff.

!!! info "Score-Matching-Intuition"
    Der trainierte Rauschvorhersager `eps_theta` ist bis auf einen Skalenfaktor ein Schätzer von `-grad_x log p(x|s)` — der Score-Funktion der Datenverteilung. Iteratives Denoising ist Gradientenaufstieg auf der Log-Dichte mit kontrollierter Rauscheinspeisung. Deshalb kann Diffusion jede Verteilung darstellen: Scores können beliebig geformt sein, während Gaußverteilungen durch ihre zwei Parameter formbeschränkt sind.

---

## 2 · Diffusion-Policy-Architektur

Chi et al. (2023) haben DDPM auf Observations konditioniert und damit **Diffusion Policy** geschaffen. Sie schlugen zwei Architekturvarianten je nach Observation-Modalität vor.

| Variante | Observation-Typ | Konditionierungs-Mechanismus | Am besten für |
|---------|------------------|------------------------|----------|
| CNN U-Net | Bilder (RGB oder Tiefe) | FiLM (feature-wise linear modulation) auf U-Net-Features | Robotermanipulation von Kameras |
| Transformer | Niedrigdim. Zustandsvektor | Cross-Attention von Obs-Tokens zu Action-Tokens | Zustandsbasierte Regelung, schnelle Inferenz |
| **MLP (Kurswahl)** | Niedrigdim. Zustandsvektor | Konkatenation von `(noisy_action, obs, time_emb)` | Godot-Agenten mit Vektor-Obs |

Die MLP-Variante ist die einfachste und passt zum Standard-Observation-Setup des Kurses (die `AIController.get_obs()`-Dictionaries, die wir seit RL Essentials nutzen). Sie ist auch schnell genug, um mit DDIM-Sampling aus Abschnitt 4 in der Godot-Schleife mit 60 Hz zu laufen.

Hier eine vollständige minimale Implementierung. Sie ist klein genug, um in einer Sitzung gelesen zu werden, und funktioniert als Drop-in-Policy-Kopf für jede kontinuierliche-Aktion-Godot-Env.

```python
import torch
import torch.nn as nn
import numpy as np


class SinusoidalPosEmb(nn.Module):
    """Time-step embedding for the diffusion denoiser.

    Maps a scalar denoising step t in [0, T-1] to a `dim`-vector using
    the same sinusoidal positional encoding as in Transformers.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class DiffusionMLP(nn.Module):
    """Noise prediction network conditioned on observation and diffusion timestep.

    The model learns eps_theta(x_t, t, obs) -> predicted noise.
    Sampling iterates the learned reverse process to produce actions.
    """
    def __init__(self, obs_dim: int, act_dim: int,
                 d_model: int = 256, n_diffusion_steps: int = 100):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_steps = n_diffusion_steps

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.Mish(),
            nn.Linear(d_model * 2, d_model),
        )
        self.net = nn.Sequential(
            nn.Linear(act_dim + obs_dim + d_model, d_model),
            nn.Mish(),
            nn.Linear(d_model, d_model),
            nn.Mish(),
            nn.Linear(d_model, d_model),
            nn.Mish(),
            nn.Linear(d_model, act_dim),
        )

        # DDPM noise schedule (linear betas, the original DDPM choice)
        betas = torch.linspace(1e-4, 0.02, n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('sqrt_alphas_bar', alphas_bar.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_bar', (1 - alphas_bar).sqrt())

    def forward(self, noisy_action, t, obs):
        """Predict noise eps given noisy action x_t, timestep t, observation."""
        t_emb = self.time_emb(t)
        x = torch.cat([noisy_action, obs, t_emb], dim=-1)
        return self.net(x)

    def loss(self, action, obs):
        """DDPM training loss — predict the noise added at a random timestep."""
        B = action.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=action.device)
        eps = torch.randn_like(action)
        noisy = (self.sqrt_alphas_bar[t, None] * action
                 + self.sqrt_one_minus_alphas_bar[t, None] * eps)
        pred_eps = self.forward(noisy, t, obs)
        return ((eps - pred_eps) ** 2).mean()

    @torch.no_grad()
    def sample(self, obs, n_ddim_steps: int = 10):
        """DDIM fast sampling — generate an action conditioned on obs.

        DDIM picks a sparse subsequence of timesteps (e.g. 10 out of 100)
        and uses the deterministic update rule from Song et al. 2021.
        """
        B = obs.shape[0]
        x = torch.randn(B, self.act_dim, device=obs.device)

        timesteps = torch.linspace(self.n_steps - 1, 0, n_ddim_steps,
                                   dtype=torch.long, device=obs.device)

        for i, t in enumerate(timesteps):
            t_batch = t.expand(B)
            pred_eps = self.forward(x, t_batch, obs)

            alpha_bar = self.alphas_bar[t]
            pred_x0 = (x - (1 - alpha_bar).sqrt() * pred_eps) / alpha_bar.sqrt()
            pred_x0 = pred_x0.clamp(-1, 1)  # actions are normalised to [-1, 1]

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_bar_next = self.alphas_bar[t_next]
                x = (alpha_bar_next.sqrt() * pred_x0
                     + (1 - alpha_bar_next).sqrt() * pred_eps)
            else:
                x = pred_x0

        return x
```

Das ist das Ganze. Etwa 80 Zeilen für eine funktionierende multimodale kontinuierliche-Aktion-Policy.

!!! tip "Mish statt ReLU"
    Beachte die Aktivierung: `nn.Mish()`, nicht `nn.ReLU()`. Glatte Aktivierungen zählen für Diffusionsmodelle mehr als für Klassifikatoren — der Denoiser wird bei Inferenz wiederholt mit sich selbst komponiert, also tauchen Knicke in der Aktivierungsfunktion als sichtbare Artefakte in den gesampelten Aktionen auf. Mish, GELU und SiLU funktionieren alle. Vanilla-ReLU produziert sichtbar schlechtere Samples.

!!! warning "Normalisiere deine Aktionen auf [-1, 1]"
    Der obige Noise-Schedule nimmt Daten mit etwa Einheitsvarianz an. Gibt deine Godot-Env Aktionen in `[-100, 100]` aus (z. B. rohe Gelenkmomente), zerstört der Vorwärtsprozess das Signal bei `t=T` nicht und das Training schlägt still fehl. Normalisiere Aktions- und Observation-Kanäle immer in einen vergleichbaren Bereich, bevor du sie ins Diffusionsmodell füttest.

---

## 3 · Action Chunking

Der zweite Schlüsselbeitrag von Chi et al. (2023) ist **Action Chunking**: sage `K` konsekutive zukünftige Aktionen auf einmal voraus statt nur die nächste.

In der Welt der Gauß-Policies haben wir jeden Zeitschritt als frische Entscheidung behandelt: beobachte `s_t`, sample `a_t`, wiederhole. Das ist fein, wenn die Policy schnell und zustandslos ist, hat aber Kosten: aufeinanderfolgende Aktionen werden unabhängig gesampelt, was sichtbares Jittern im Steuersignal produziert. Auf einem Roboterarm oder Luftkissenfahrzeug sieht man es — die Gelenke zucken mit hoher Frequenz, selbst wenn die Aufgabe es nicht erfordert.

Action Chunking ändert die Policy-Schnittstelle:

```python
# Old (Gaussian / SAC / PPO):
#   action = policy(obs)            # shape (act_dim,)
#   env.step(action)
#
# New (Diffusion Policy with chunking K=8):
#   actions = policy(obs)           # shape (K, act_dim) — joint sample
#   for k in range(K):
#       env.step(actions[k])        # execute the chunk open-loop
#   # then re-plan with a fresh obs
```

Drei Vorteile ergeben sich sofort:

1. **Zeitliche Kohärenz.** Die `K` Aktionen werden *gemeinsam* aus dem Diffusionsmodell gesampelt, sind also per Konstruktion gegenseitig konsistent. Kein unabhängiges-Sample-Jittern mehr.
2. **Impliziter Planungshorizont.** Das Modell muss über die nächsten `K` Schritte schlussfolgern, um einen kohärenten Chunk zu produzieren, baut also kostenlos einen internen prädiktiven Horizont auf.
3. **Glattere Steuersignale.** Das zählt für jeden physikalischen oder pseudo-physikalischen Aktuator. Robotergelenke, Luftkissenschubdüsen und Godots `CharacterBody3D` hassen alle hochfrequentes Aktionsrauschen.

Die Kosten sind **langsamere Anpassung an plötzliche Änderungen**. Passiert bei Schritt `k=3` eines 8-Schritte-Chunks etwas Unerwartetes, führt der Agent den abgestandenen Chunk bis `k=8` weiter aus. Es gibt Abhilfen — Receding-Horizon-Ausführung (nur die ersten `K_exec < K` Aktionen ausführen, dann neu planen) ist die übliche im Originalpaper.

| `K` (Chunk-Größe) | Jitter | Reaktivität | Inferenzfrequenz |
|------------------|--------|------------|---------------------|
| 1 | Hoch (unabhängige Samples) | Maximum | Jeden Step |
| 4 | Niedrig | Gut | Alle 4 Steps |
| 8 | Sehr niedrig (Paper-Default) | Moderat | Alle 8 Steps |
| 16 | Extrem niedrig | Schlecht (sieht voraufgezeichnet aus) | Alle 16 Steps |

Um Chunking in die obige Architektur zu verdrahten, ändere `act_dim` zu `K * act_dim` und reshape die Ausgabe von `sample()` zu `(B, K, act_dim)`. Alles andere bleibt gleich.

---

## 4 · Inferenzkosten und DDIM-Beschleunigung

DDPM, wie publiziert, nutzt `T=100` (oder sogar `T=1000`) Denoising-Schritte. Das sind 100 Forward Passes durch den Denoiser pro Aktion. Bei ~0,16 ms pro Forward Pass auf einer RTX 3060 kostet die volle Sampling-Schleife ~16 ms — exakt dein gesamtes 60-Hz-Budget, ohne Platz für den Godot-Physik-Tick. Unbrauchbar.

**DDIM (Song et al., 2021)** ist der Standard-Fix. Es reinterpretiert den Rückwärtsprozess als nicht-Markovsche deterministische Update-Regel, die es erlaubt, Zeitschritte zu überspringen, ohne das Modell neu zu trainieren. Du trainierst einmal mit `T=100`, dann samplest du bei Inferenz auf einer spärlichen Teilsequenz von etwa 10 Zeitschritten.

```
# DDPM reverse step (stochastic):
#   x_{t-1} = mu_theta(x_t, t) + sigma_t * z     (z ~ N(0, I))
#   must visit every t in {T-1, T-2, ..., 1, 0}
#
# DDIM reverse step (deterministic, can skip):
#   pred_x0 = (x_t - sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)
#   x_{t_next} = sqrt(alpha_bar_{t_next}) * pred_x0
#              + sqrt(1 - alpha_bar_{t_next}) * eps_theta
#   can visit any sparse subsequence {tau_0 > tau_1 > ... > tau_S}
```

Die `sample()`-Methode in Abschnitt 2 nutzt bereits DDIM — das macht das `n_ddim_steps`-Argument. Die praktische Anleitung unten ist das, was dich als Kursteilnehmer beim Deployen in Godot wirklich interessiert.

| `T` (Denoising-Schritte) | Latenz (RTX 3060) | Aktionsqualität | Urteil für 60-Hz-Godot |
|-----------------------|--------------------|----------------|--------------------------|
| 100 (volles DDPM)       | ~16 ms             | Exzellent      | Verfehlt das Frame-Budget |
| 50 (DDIM)             | ~8 ms              | Exzellent      | Halbes Budget — riskant mit Chunking |
| 20 (DDIM)             | ~3 ms              | Sehr gut      | Sicher, empfohlener Default |
| 10 (DDIM)             | ~1,5 ms            | Gut           | Reichlich Spielraum |
| 5  (DDIM)             | ~0,8 ms            | Akzeptabel auf glatten Aufgaben | Für dichte Regelungsschleifen |
| 1  (Consistency Model)| ~0,2 ms            | Aufgabenabhängig | Erfordert Distillation, fortgeschritten |

**Faustregel für diesen Kurs.** Starte mit `n_ddim_steps=20`. Ist die Policy zu langsam für deine Env (prüfe die `policy_time_ms`-Logzeile, die dein Inferenzserver druckt), gehe auf 10 runter. Verschlechtert sich die Aktionsqualität, gehe wieder rauf. Das 60-Hz-Godot-Budget ist 16 ms gesamt; du willst, dass Diffusion höchstens ein Drittel verbraucht, damit Physik-Tick, Env-Step und Rendering je einen fairen Anteil bekommen.

!!! tip "Action Chunking und DDIM verstärken sich"
    Ein Chunk von `K=8` Aktionen mit `T=10` Denoising-Schritten heißt **ein DDIM-Aufruf pro 8 Umgebungsschritte**. Deine effektiven Inferenzkosten pro Step sind `1,5 / 8 ≈ 0,2 ms` — in vielen Fällen billiger als eine Gauß-Policy, trotz iterativem Sampling.

---

## 5 · Vergleich mit SAC und PPO

Diffusion Policy ist kein striktes Upgrade. Sie ändert die Regeln des Spiels genug, dass der richtige Vergleich Aufgabe-für-Aufgabe ist. Die untenstehende Tabelle fasst die Abwägungen zusammen, die du gegeneinander stellen solltest.

|                          | PPO                          | SAC                                  | Diffusion Policy                          |
|--------------------------|------------------------------|--------------------------------------|--------------------------------------------|
| Policy-Verteilung      | Gaußsch                     | Gequetschte Gaußsch (tanh)             | Beliebig (multimodal, jede Form)          |
| Trainings-Paradigma        | On-policy RL                 | Off-policy RL                        | Supervised (offline Behaviour Cloning)     |
| Datenanforderung         | Online-Rollouts              | Online-Rollouts + Replay-Buffer      | Offline-Datensatz aus Demonstrationen          |
| Inferenzkosten           | O(1) Forward Pass            | O(1) Forward Pass                    | O(T_ddim) Forward Passes (T_ddim = 5–20)   |
| Sample-Effizienz        | Mittel                       | Hoch                                 | N/A — hängt von Demonstrationsqualität ab     |
| Erfasst mehrere Modi  | Nein                           | Nein (einzelne Gaußverteilung)                 | Ja (der ganze Punkt)                      |
| Behandelt diskrete Aktionen | Ja                          | Diskrete-SAC-Variante existiert         | Unhandlich — braucht Softmax + Gumbel-Tricks    |
| Trainingsstabilität       | Sensibel auf Hyperparameter | Stabiler als PPO nach Tuning      | Sehr stabil (nur supervised Regression)   |
| Am besten für                 | Allgemeines RL auf Spiel-Envs      | Kontinuierliche Regelung, sample-effizient | Manipulation, multimodale Aufgaben, Demos      |
| Godot-Echtzeit?         | Ja                          | Ja                                  | Ja, mit DDIM `T <= 10` und Chunking      |

Die Schlagzeile: **PPO und SAC lernen aus Belohnung; Diffusion Policy lernt aus Demonstrationen.** Das ist kein kleiner Unterschied. Hast du keine Demonstrationen und kannst sie nicht sammeln, ist Diffusion Policy keine Option — sie ist eine Behaviour-Cloning-Methode, keine RL-Methode.

Es gibt aktive Forschung, die Diffusion-Policies mit Reward-Signalen kombiniert (Q-Score Matching, IDQL, Diffusion-QL), sodass du eine Diffusion-BC-Policy mit RL fein-tunen kannst. Diese Methoden sind hier außerhalb des Umfangs; du würdest auf `unit-offline-rl` aufbauen, um eine zu versuchen.

---

## 6 · Eine Diffusion Policy auf Godot-Demonstrationen trainieren

End-to-end-Workflow, angenommen du hast `unit-09` (Imitation Learning) absolviert und hast eine funktionierende Demonstrations-Pipeline.

**Schritt 1 — Expertendemonstrationen sammeln.** Entweder einen trainierten PPO/SAC-Agenten abspielen und `(obs, action)`-Paare loggen oder mit dem Tastatur-gesteuerten `HumanController` aus Unit 9 menschliche Demos aufzeichnen. Speichere als HDF5-Datei mit zwei Arrays der Form `(N, obs_dim)` und `(N, act_dim)`. Ziel 50k–500k Übergänge für nicht-triviale Aufgaben.

```python
# replay_pretrained.py — collect demos from a trained PPO agent
import h5py
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

env = StableBaselinesGodotEnv(env_path="builds/PickAndPlace.x86_64", n_parallel=1)
agent = PPO.load("checkpoints/pickandplace_ppo_5M.zip")

obs_log, act_log = [], []
obs, _ = env.reset()
for _ in range(100_000):
    action, _ = agent.predict(obs, deterministic=False)
    obs_log.append(obs.copy()); act_log.append(action.copy())
    obs, _, term, trunc, _ = env.step(action)
    if term.any() or trunc.any():
        obs, _ = env.reset()

with h5py.File("demos.h5", "w") as f:
    f.create_dataset("obs", data=np.array(obs_log))
    f.create_dataset("action", data=np.array(act_log))
```

**Schritt 2 — Diffusion Policy mit supervised Regression trainieren.** Keine Belohnung, keine Value-Funktion, kein Replay-Buffer — nur den Rauschvorhersage-Loss aus Abschnitt 2 minimieren.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

obs = torch.tensor(np.array(obs_log), dtype=torch.float32)
act = torch.tensor(np.array(act_log), dtype=torch.float32)
loader = DataLoader(TensorDataset(obs, act), batch_size=256, shuffle=True)

model = DiffusionMLP(obs_dim=obs.shape[1], act_dim=act.shape[1]).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)

for epoch in range(200):
    for o, a in loader:
        loss = model.loss(a.cuda(), o.cuda())
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch {epoch}: noise-MSE = {loss.item():.4f}")
```

Training ist schnell — meist unter einer Stunde auf einer einzelnen GPU für die obigen Datensatzgrößen. Die Loss-Kurve sollte glatt fallen. Es gibt keine Exploration/Exploitation-Dynamiken zu bewältigen, weil keine Exploration stattfindet: du fittest nur eine bedingte Dichte.

**Schritt 3 — In Godot deployen.** Hier wird Diffusion Policy verglichen mit PPO/SAC unhandlich. Die Denoising-Schleife ist eine Python-`for`-Schleife, und ONNX (das `godot-rl-agents` für In-Process-Inferenz nutzt) kann Python-Kontrollfluss nicht direkt tracen.

Drei Optionen, in steigender Komplexität:

a. **Nur den Einzel-Step-Denoiser nach ONNX exportieren, die Schleife in GDScript ausführen.** Kleinster Export, aber du musst die DDIM-Update-Regel in GDScript neu implementieren — fehleranfällig.

b. **Die vollständige `sample()`-Methode mit TorchScript kompilieren und von einem Python-Inferenzserver ausführen.** Sauberste Option. Die Godot-Env kommuniziert mit dem Server über das vorhandene `godot_rl`-Socket-Protokoll.

c. **ONNX-Loop-Operatoren nutzen.** Technisch möglich, sehr fragil über Runtimes hinweg. Für diesen Kurs nicht empfohlen.

Der empfohlene Pfad ist (b):

```python
# export_diffusion_policy.py
import torch

model = DiffusionMLP(obs_dim=24, act_dim=7).cuda().eval()
model.load_state_dict(torch.load("diffusion_policy.pt"))

# TorchScript can scriptify the full sampling loop including the for-loop.
scripted = torch.jit.script(model)
torch.jit.save(scripted, "diffusion_policy_scripted.pt")

# Inference server reloads and serves predictions:
#   scripted = torch.jit.load("diffusion_policy_scripted.pt").cuda().eval()
#   action = scripted.sample(obs_tensor, n_ddim_steps=10)
# Godot connects via the standard godot_rl tcp socket.
```

!!! warning "Achte beim Deploy auf die Aktionsnormalisierung"
    Das Diffusionsmodell wurde auf Aktionen in `[-1, 1]` trainiert. Deine Godot-Env erwartet fast sicher Aktionen in einem anderen Bereich (Gelenkwinkel in Radiant, Schubkräfte in Newton usw.). Der Inferenzserver ist für die inverse Normalisierung verantwortlich: `env_action = unnormalise(sampled_action)`. Das zu vergessen ist der mit Abstand häufigste Deployment-Bug.

!!! check "Fertig, wenn"
    Es gibt keinen festen Loss-Wert zu erreichen — beurteile den Lauf an den eigenen Signalen dieser Unit: (1) der noise-MSE, den die Trainingsschleife aus Schritt 2 druckt, fällt glatt über die Epochen, und (2) die deployte Policy reproduziert sichtbar das demonstrierte Verhalten im Godot-Viewer — bei dem `n_ddim_steps`, mit dem du deployt hast (das Snippet aus Schritt 3 nutzt 10; die Faustregel aus Abschnitt 4 startet bei 20). Ein noise-MSE, der sich kaum vom Startwert wegbewegt, ist die Fehlersignatur für unnormalisierte Daten — prüfe zuerst, dass Aktionen und Observations in einen vergleichbaren Bereich skaliert sind (die Warnung in Abschnitt 2), statt länger zu trainieren. Ein Modell, dessen Loss sauber fällt, das sich in Godot aber erratisch verhält, deutet auf die inverse Normalisierung im Inferenzserver (die Warnung oben), nicht auf das Modell.

---

## 7 · Wann Diffusion gewinnt vs wann du bei SAC bleibst

Die Frage, der du in dem Moment, in dem du diese Unit beendest, gegenüberstehen wirst: *Soll ich mein Projekt umschreiben, um Diffusion Policy zu nutzen?* Die ehrliche Antwort ist „wahrscheinlich nicht, aber hier sind Situationen, in denen du es solltest."

**Diffusion Policy nutzen, wenn:**

- Deine Demonstrationen **mehrere gültige Strategien** für dieselbe Situation enthalten. Das Gauß-Kollaps-Problem ist real und Diffusion Policy löst es direkt.
- Du **Manipulation oder geschickte Regelung** machst, bei der Glätte des Aktionssignals zählt. Action Chunking gibt dir freie zeitliche Kohärenz.
- Du **aus menschlichen Demonstrationen lernst** (statt aus Belohnung). Menschen sind inhärent multimodal — verschiedene Sitzungen, verschiedene Tage, verschiedene Stimmungen. Diffusion Policy kann diese Variabilität absorbieren.
- Du einen **festen Offline-Datensatz** und keinen Online-Umgebungszugriff hast. Reines BC auf einer Gauß-Policy ist Modus-Mittelung; Diffusion Policy ist es nicht.

**Bei SAC (oder PPO) bleiben, wenn:**

- Du **Online-RL** machst und frische Erfahrung sammeln kannst. Diffusion Policy ist eine BC-Methode; sie optimiert Belohnung nicht direkt.
- Die Aktionsverteilung an jedem Zustand wirklich **unimodal** ist. Ein 1-D-Gas, ein Lenkwinkel auf glatter Strecke, die meisten spielähnlichen Envs — eine Gaußverteilung ist fein und 100× schneller.
- Dein **Latenzbudget unter ~3 ms pro Aktion** liegt und DDIM mit `T=5` immer noch nicht passt. Manche Hochfrequenz-Regelschleifen (Haptikgeräte, Drohnen-Autopiloten) leben in diesem Regime.
- Du **diskrete Aktionen** brauchst. Diffusion Policy für diskrete Ausgaben erfordert kategoriale Diffusion oder Gumbel-Tricks und lohnt sich selten.
- Du **starke Sample-Effizienz aus einem kleinen Online-Budget** brauchst (der SAC-Sweet-Spot).

Eine pragmatische Mischstrategie, die in der Robotik immer häufiger wird: trainiere eine Diffusion Policy auf Demonstrationen zum Bootstrappen, dann fein-tune mit einer kleinen Menge Online-RL (Diffusion-QL / IDQL), um Belohnung über das Niveau des Demonstrators hinaus zu optimieren.

---

## 8 · Stretch Goals

Diese Übungen sind so gestaltet, dass die abstrakten Behauptungen dieser Unit konkret werden.

1. **Bau die bimodale Spielaufgabe.** Erstelle eine 2-D-Punktmassen-Env in Godot mit zwei gültigen Zielen (`x=+1` und `x=-1`), die beim Erreichen identische Belohnung geben. Sammle 1k Demonstrationen im Verhältnis 50/50 zwischen den Zielen. Trainiere SAC und eine Diffusion Policy auf denselben Daten. Visualisiere die Aktionsverteilung am Startzustand, indem du 1000 Aktionen aus jeder samplest. SAC sollte einen engen Blob um `x=0` produzieren; Diffusion Policy sollte zwei klar getrennte Cluster produzieren.

2. **Eine Manipulationsaufgabe per Tastatur demonstrieren.** Modifiziere den `AIController` einer `Reacher`- oder `PickAndPlace`-Env, sodass er auch Tastatur-Input als Aktionsquelle akzeptiert, und logge menschliche Demos während du spielst. Sammle ~1000 erfolgreiche Episoden. Trainiere Behaviour Cloning (Gauß-Regression) und Diffusion Policy auf denselben Daten. Vergleiche Erfolgsrate, Glätte (Varianz der Aktionsdifferenzen) und die visuelle Qualität der resultierenden Bewegung im Godot-Viewer.

3. **DDIM-Step-Count-Ablation.** Nimm eine trainierte Diffusion Policy und evaluiere sie bei `T_ddim ∈ {5, 10, 20, 50, 100}`. Plotte die Aufgaben-Erfolgsrate und die Inferenzlatenz pro Step im selben Chart. Die Kurve sollte von `T=5` zu `T=10` steil steigen, bei `T=20` plateauen, und die Inferenzkosten sollten linear skalieren. Nutze den Plot, um den Betriebspunkt für dein Projekt zu wählen.

4. **Action-Chunking-Ablation auf Lokomotion.** Trainiere Diffusion Policy mit Chunk-Größen `K ∈ {1, 4, 8, 16}` auf einer Vierbeiner-Lokomotion-Aufgabe aus `unit-locomotion`. Miss (a) Episoden-Return, (b) Aktionsglätte (RMS aufeinanderfolgender Aktionsdifferenzen) und (c) Erholungszeit von einer Push-Perturbation. Du solltest sehen, dass die Glätte monoton mit `K` besser wird, während die Erholungszeit schlechter wird.

5. **(Fortgeschritten) Score-Visualisierung.** Für eine 2-D-Aktions-Env plotte das gelernte Score-Feld `eps_theta(x, t=0, obs=fixed)` als Vektorfeld auf der Aktionsebene. Lege die Demonstrationsdaten darüber. Die Vektoren sollten von Bereichen niedriger Dichte zur Datenmannigfaltigkeit zeigen. Das ist die direkteste visuelle Bestätigung, dass Diffusionsmodelle Gradienten der Log-Dichte schätzen.

---

## Was kommt als Nächstes

Du hast jetzt einen Weg, Aktionsverteilungen darzustellen, die keine Gaußverteilung könnte. Diffusion Policy ist die dritte Säule moderner kontinuierlicher Regelung neben SAC und PPO — das richtige Werkzeug, wenn deine Daten multimodal sind und du Demonstrationen statt eines Reward-Signals hast.

Die nächsten Orte, dies hinzubringen:

- **Diffusion mit Belohnung kombinieren.** `unit-offline-rl` behandelt die Offline-RL-Methoden (CQL, IQL), die Q-Learning auf festen Datensätzen erlauben. Die Diffusion-RL-Hybride (Diffusion-QL, IDQL) legen Q-Learning auf ein Diffusion-BC-Backbone — lies diese Paper als Nächstes.
- **Die Demo-Schleife schließen.** `unit-curiosity` und `unit-hierarchical` geben dir Werkzeuge zum Erkunden ohne Reward-Signal — nützlich, wenn du vielfältige Demonstrationen programmatisch generieren willst, statt sie von Menschen zu sammeln.
- **Deployment-Härtung.** `unit-sim-to-real` diskutiert die Domänenlücken, die zählen, sobald du eine Policy vom Simulator wegnimmst. Diffusion-Policies erben all diese Probleme und fügen ihre eigenen hinzu (Aktionsverteilungs-Shift, wenn die Obs-Normalisierung driftet).

[← SAC](unit-sac.md) · [Kursstartseite](index.md)
