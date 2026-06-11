# Actor-Critic — Wert-Methoden mit Policy Gradients vereinen

REINFORCE hat die Policy direkt gelernt, aber um den Preis, auf ganze Episoden zu warten und verrauschte Returns zu tolerieren. DQN lernte eine Wertfunktion, aber nur für diskrete Aktionen. **Actor-Critic** vereint beides: ein **Actor** wählt Aktionen wie REINFORCE, während ein **Critic** Returns schätzt wie DQN. Diese Unit führt vom Varianzproblem in REINFORCE bis zu einer vollständigen A2C-Implementierung — dem algorithmischen Rückgrat von PPO, das du seit Unit 2 in `gdrl` laufen lässt.

[← Policy Gradients](unit-policy-gradients.md) · [Kursstartseite](index.md)

!!! note "Voraussetzungen"
    - **[Policy Gradients](unit-policy-gradients.md)** — REINFORCE, Baselines, das Varianzproblem
    - **[Q-Learning-Unit](unit-q-learning.md)** — der Critic *ist* eine Wertfunktion; Bellman sollte vertraut sein
    - **[RL Essentials](unit-01.md)** — Diskontierungsfaktor, Return, MDP-Schleife
    - PyTorch-Sicherheit (Forward Pass, Optimizer, `loss.backward()`) für §5

!!! info "Zeit"
    Lesen: ~35 min · Training: ~20 min GPU / ~1 h CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    Python-Konsole (Actor-/Critic-Loss pro Update) · matplotlib (actor_loss, critic_loss, ep_rew_mean) · `gym.make("CartPole-v1", render_mode="human")` für ein Live-CartPole-Fenster

!!! warning "Konzepte vor Code"
    Das PyTorch-Listing in Abschnitt 5 ergibt erst Sinn, wenn du die Vorteilsfunktion (advantage function) in Abschnitt 3 verstanden hast. Lies von oben nach unten — spring nicht direkt zum Code.

---

## 1 · Das Problem, das REINFORCE uns hinterlassen hat

REINFORCE funktioniert, zahlt aber einen Preis dafür, eine reine Monte-Carlo-Methode zu sein:

- **Es braucht vollständige Episoden.** Die Update-Regel nutzt den diskontierten Return `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …`. Du kannst `G_t` nicht berechnen, bevor du jede Belohnung nach Schritt `t` gesehen hast. Lange Episoden bedeuten langsames Lernen.
- **Es hat hohe Varianz.** `G_t` ist das Ergebnis eines einzelnen Rollouts. In CartPole stößt ein unglücklicher Windstoß den Stab bei Schritt 17 um, obwohl die Aktion bei Schritt 3 perfekt war — REINFORCE bestraft diese Schritt-3-Aktion trotzdem. Über Tausende Episoden mittelt sich das Rauschen heraus, aber es braucht *viele* Stichproben.
- **Es wirft Information weg.** DQN hat ein Wertenetz `Q(s,a)` gelernt, das den Return aus *jedem* Zustand schätzt, ohne eine Episode auszurollen. REINFORCE ignoriert diese Idee komplett.

Was, wenn wir DQNs Trick bitten würden, REINFORCE zu helfen? Nimm ein neuronales Netz, um den **Return aus einem Zustand zu schätzen**, damit der Actor nicht warten muss, bis die Episode endet. Dieser Schätzer heißt **Critic**.

> Mentales Modell: REINFORCE ist eine Studentin, die ihre Note erst *nach* den Abschlussprüfungen kennt. Actor-Critic ist eine Studentin, die nach jedem Quiz eine geschätzte Note bekommt — dank einer TA (dem Critic), die das ganze Semester über zugesehen hat.

---

## 2 · Zwei Netze, ein Ziel

Actor-Critic nutzt zwei Funktionsapproximatoren, die zusammenarbeiten:

| Netz | Symbol | Aufgabe | Analogon |
|---|---|---|---|
| **Actor** (Policy) | `π_θ(a \| s)` | Aktion zum Zustand wählen | REINFORCEs Policy |
| **Critic** (Wert) | `V_φ(s)` | Erwarteten Return aus `s` schätzen | DQNs `Q`, aber nur über Zustände |

- `θ` sind die Parameter des Actors, `φ` die des Critics. Jedes Netz hat sein eigenes Gradientensignal, aber in der Praxis **teilen sie sich die meisten Schichten** — ein Trunk verarbeitet die Beobachtungen, zwei kleine Heads sitzen oben drauf. Das spart Parameter und hilft dem Actor, von Features zu profitieren, die der Critic entdeckt hat (und umgekehrt).
- Der Actor wird mit einem **Policy-Gradient** trainiert, genau wie REINFORCE — nur dass das verrauschte `G_t` durch ein weniger verrauschtes Signal ersetzt wird, das den Critic nutzt.
- Der Critic wird mit **TD-Lernen** trainiert, genau dieselbe Bootstrapping-Idee aus Q-Learning und DQN: sage den Return voraus, schiebe die Vorhersage dann in Richtung der beobachteten Belohnung plus der Vorhersage des nächsten Zustands.

```
            ┌────── geteilter Backbone ──────┐
            │  (z. B. 2× Linear+Tanh, 128)   │
            └──────────────┬─────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
      Actor-Head (Logits)        Critic-Head (V(s))
         → Aktion                  → skalare Return-Schätzung
```

---

## 3 · Die Vorteilsfunktion

REINFORCEs Gradient drückt Aktionen hoch, deren Return `G_t` groß ist. Aber „groß" relativ wozu? Eine CartPole-Episode bringt vielleicht +200, weil der *Zustand* einfach war, nicht weil die *Aktion* klug war. Wir wollen die Aktion belohnen, nicht den Zustand.

Genau das ist der **Vorteil (advantage)**:

```
A(s, a) = Q(s, a) - V(s)
```

Im Klartext:

> „Wie viel besser ist Aktion `a` als die *durchschnittliche* Aktion, die ich aus Zustand `s` gewählt hätte?"

- **A > 0** — diese Aktion ist besser als der Durchschnitt → erhöhe ihre Wahrscheinlichkeit
- **A < 0** — diese Aktion ist schlechter als der Durchschnitt → senke ihre Wahrscheinlichkeit
- **A = 0** — diese Aktion ist genau durchschnittlich → lass sie in Ruhe

Wir haben keinen Zugriff auf das wahre `Q(s, a)`. Aber wir haben einen Critic, der `V(s)` schätzt, und kennen eine echte Belohnung und den nächsten Zustand aus der Interaktion mit der Umgebung. Bootstrapping (derselbe Trick wie bei DQN) ergibt:

```
Q(s, a) ≈ r + γ V(s')        ← 1-Schritt-Return
```

Eingesetzt:

```
A(s, a) ≈ r + γ V(s') - V(s) = δ      ← der TD-Fehler des Critics
```

**Das ist das Schlüsselergebnis der ganzen Unit:** der TD-Fehler des Critics ist ein (unverzerrter) Schätzer des Vorteils. Dieselbe Zahl, mit der sich der Critic selbst korrigiert, ist das Signal, mit dem der Actor seine Policy updatet. Ein Skalar pro Schritt, zwei Netze aktualisiert.

Term für Term:

| Term | Was es ist |
|---|---|
| `r` | Die Belohnung, die du nach Aktion `a` in `s` tatsächlich erhalten hast |
| `γ V(s')` | Die diskontierte Critic-Vorhersage von allem, was *nach* dem nächsten Zustand passiert |
| `V(s)` | Die Critic-Vorhersage des Gesamtreturns aus `s` *bevor* du gehandelt hast |
| `δ = r + γV(s') - V(s)` | Die „Überraschung" — besser oder schlechter als erwartet? |

---

## 4 · A2C: Advantage Actor-Critic

A2C („Advantage Actor-Critic", der synchrone Cousin von A3C) ist der sauberste Algorithmus, der auf der obigen Idee aufbaut. Hier die vollständige Schleife:

1. **Sammle ein `n`-Schritt-Rollout**, indem du die aktuelle Policy in der Umgebung ausführst:
   `(s_0, a_0, r_0), (s_1, a_1, r_1), …, (s_n, a_n, r_n)`
2. **Berechne Vorteilsschätzungen** für jeden Schritt:
   `A_t = r_t + γ V(s_{t+1}) - V(s_t)`
   (Oder die Mehrschritt-Verallgemeinerung in Abschnitt 7.)
3. **Actor-Loss** — gleiche Form wie REINFORCE, aber mit `A_t` statt `G_t`:
   `L_actor = - Σ_t A_t · log π_θ(a_t | s_t)`
4. **Critic-Loss** — quadrierter TD-Fehler, wie DQNs Regressionsziel:
   `L_critic = Σ_t (r_t + γ V(s_{t+1}) - V(s_t))²`
5. **Entropie-Bonus (entropy bonus)** — verhindert, dass die Policy zu früh kollabiert:
   `L_entropy = - β · H(π_θ) = β · Σ_t π_θ log π_θ`
6. **Gesamt-Loss**, über das Rollout aufsummiert:
   `L = L_actor + c · L_critic - β · H(π_θ)`
   mit `c = 0,5` (Critic-Gewicht) und `β = 0,01` (Entropie-Koeffizient) als üblichen Defaults.
7. **Ein Backprop** durch das geteilte Netz. Adam (oder RMSProp) updatet Actor und Critic in einem einzigen Schritt.

Verglichen mit REINFORCE ist das wild effizienter: wir updaten alle `n` Schritte statt jede Episode und nutzen eine gelernte Baseline (`V(s)`) statt des rohen Returns.

---

## 5 · Vollständige A2C-PyTorch-Implementierung

Unten ist ein eigenständiger A2C-Agent, der CartPole-v1 in wenigen hundert Updates auf CPU löst. Er nutzt einen geteilten Backbone, n-Schritt-Rollouts, Vorteilsnormalisierung, einen Entropie-Bonus und Gradient Clipping — alles Standardtricks, die du in PPO wiedersehen wirst.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128),     nn.Tanh(),
        )
        self.actor_head  = nn.Linear(128, act_dim)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.actor_head(h)
        value  = self.critic_head(h).squeeze(-1)
        return logits, value

    def get_action(self, obs):
        logits, value = self(obs)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


env = gym.make("CartPole-v1")
model = ActorCritic(obs_dim=4, act_dim=2)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

gamma       = 0.99
vf_coef     = 0.5
ent_coef    = 0.01
n_steps     = 128    # steps per update
max_updates = 500

obs, _ = env.reset()

for update in range(max_updates):
    # 1. Collect n_steps of experience
    obs_list, act_list, rew_list, val_list, logp_list, done_list = [], [], [], [], [], []

    for _ in range(n_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, log_prob, entropy, value = model.get_action(obs_t)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        obs_list.append(obs_t.squeeze(0))
        act_list.append(action)
        rew_list.append(reward)
        val_list.append(value)
        logp_list.append(log_prob)
        done_list.append(done)

        obs = next_obs if not done else env.reset()[0]

    # 2. Compute returns and advantages (reverse pass)
    returns, advantages = [], []
    G = 0.0
    for r, v, d in zip(reversed(rew_list), reversed(val_list), reversed(done_list)):
        G = r + gamma * G * (1 - d)
        adv = G - v.item()
        returns.insert(0, G)
        advantages.insert(0, adv)

    returns    = torch.tensor(returns,    dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_t  = torch.stack(obs_list)
    logp_t = torch.stack(logp_list)

    # 3. Recompute logits/values with current params (for entropy + critic loss)
    logits, values = model(obs_t)
    dist    = torch.distributions.Categorical(logits=logits)
    entropy = dist.entropy().mean()

    actor_loss  = -(advantages * logp_t).mean()
    critic_loss = (returns - values.squeeze()).pow(2).mean()
    loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy

    # 4. Single backward pass through the shared network
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    if update % 50 == 0:
        print(f"Update {update:4d} | actor_loss={actor_loss.item():+.3f} "
              f"critic_loss={critic_loss.item():.3f} entropy={entropy.item():.3f}")
```

Lass das laufen. Die Entropie sinkt von ~0,69 (das Maximum für zwei gleich wahrscheinliche Aktionen, `ln 2`) Richtung 0,3, sobald sich die Policy festlegt.

!!! check "Fertig, wenn"
    `ep_rew_mean` innerhalb von ~150 Updates über 200 klettert und weiter Richtung CartPole-v1-Obergrenze von 500 steigt. CartPole-v1 gilt als **gelöst bei einem durchschnittlichen Return ≥ 475 über 100 aufeinanderfolgende Episoden** — wenn deine Kurve weit darunter stagniert, während die Entropie nahe 0 liegt, siehst du Entropie-Kollaps (Abschnitt 8), keinen konvergierten Agenten.

!!! tip "Vergleich mit REINFORCE"
    Steck das gleiche Netz in dein REINFORCE-Skript aus der vorigen Unit (Critic-Head weglassen, rohes `G_t` statt `A_t` nutzen). Du wirst sehen, dass A2C dieselbe Belohnung in etwa einer Größenordnung weniger Umgebungsschritten erreicht — dieselbe Beobachtung, die das Feld überhaupt von REINFORCE zu A2C getrieben hat.

---

## 6 · Warum Gradient Clipping?

Die eine Zeile `torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)` leistet erstaunlich schwere Arbeit.

- Der Vorteil kann gelegentlich sehr groß sein (eine seltene große Belohnung, eine ungewöhnlich falsche Critic-Vorhersage).
- Ein großer Vorteil multipliziert mit einem `log π`-Term erzeugt einen riesigen Gradienten → Adam macht einen riesigen Schritt → die Policy-Verteilung schwingt heftig → im nächsten Rollout werden die meisten Aktionen absurd → Belohnung kollabiert → Erholung ist langsam oder unmöglich.
- Das Clippen der Gradienten-Norm auf 0,5 deckelt, wie weit sich die Policy pro Update bewegen kann.

!!! tip "Vorausschau auf PPO"
    Gradient Clipping ist die *plumpe* Version von „lass die Policy sich pro Update nicht zu weit bewegen". PPOs geclipptes Surrogat-Objektiv (nächste Unit) ist die *prinzipielle* Version: statt den Gradienten *nach* der Berechnung zu deckeln, definiert PPO den Loss so um, dass Updates, die die Policy zu sehr verschieben würden, automatisch Gradient null bekommen.

---

## 7 · N-Schritt-Returns vs. 1-Schritt-TD

Es gibt ein ganzes Spektrum, wie man den Return für den Critic und den Vorteil für den Actor schätzen kann:

| Schätzer | Formel | Bias | Varianz |
|---|---|---|---|
| 1-Schritt-TD | `r_t + γ V(s_{t+1})` | hoch (nutzt die verzerrte Critic-Schätzung) | niedrig |
| n-Schritt | `r_t + γ r_{t+1} + … + γ^{n-1} r_{t+n-1} + γ^n V(s_{t+n})` | mittel | mittel |
| Monte Carlo (REINFORCE) | `G_t = r_t + γ r_{t+1} + …` (bis Episodenende) | null (unverzerrt) | hoch |

Größeres `n` nutzt mehr echte Belohnungen und weniger Critic-Vorhersage → weniger Bias, mehr Varianz. Kleineres `n` macht das Gegenteil. `n_steps=128` im Code oben ist ein Mittelweg, den PPO-Familien-Algorithmen bevorzugen.

Das ist **genau der `n_steps`-Parameter, den du in Unit 4 getunt hast** beim Aufruf von `gdrl`. Größeres `n_steps` bedeutet längere Rollouts, weniger Updates, mehr Umgebungsdaten pro Gradientenschritt. PPOs „advantages" werden mit einer Verallgemeinerung berechnet, die **GAE (Generalized Advantage Estimation)** heißt und über einen Parameter `λ` zwischen 1-Schritt und Monte Carlo glatt interpoliert — im Geist aber identisch zu dem, was du hier siehst.

---

## 8 · Entropie-Bonus für Erkundung

Sieh dir den Loss nochmal an:

```
L = L_actor + 0.5 · L_critic - 0.01 · H(π_θ)
```

Dieser letzte Term ist der **Entropie-Bonus**. Entropie vom Loss abzuziehen ist dasselbe wie sie als Belohnung hinzuzufügen.

- `H(π_θ) = - Σ_a π(a|s) log π(a|s)` misst, wie verteilt die Aktionsverteilung ist.
- Für zwei gleich wahrscheinliche Aktionen ist `H = ln 2 ≈ 0,693`. Für eine deterministische Policy ist `H = 0`.
- Ohne diesen Bonus **kollabiert** A2C häufig: ganz früh im Training sieht eine Aktion zufällig leicht besser aus, der Actor drückt ihre Wahrscheinlichkeit auf 1,0, und der Agent hört für immer auf zu erkunden.
- `ent_coef = 0,01` ist der typische Default. Wenn du in deinem Lauf siehst, dass die Entropie in den ersten paar Updates auf 0 abstürzt, während die Belohnung noch flach ist, erhöhe ihn auf `0,05`.

!!! warning "Entropie-Kollaps sieht aus wie eine festgefahrene Belohnung"
    Eine flache Belohnungskurve mit sehr niedriger Entropie ist die klassische Signatur. Die Policy hat sich früh festgelegt und probiert nichts Neues mehr. Erhöhe `ent_coef`, senke die Lernrate, oder beides.

Das ist derselbe Regler wie das `--ent_coef`-Argument in `gdrl` aus Unit 4. Es ist keine magische Zahl — es ist das Gewicht im Loss, den du gerade gelesen hast.

---

## 9 · Geteilte vs. getrennte Netze

Zwei vernünftige Architekturen, zwei Trade-offs:

- **Geteilter Backbone, zwei Heads** (der Code in Abschnitt 5):
  - Schneller, weniger Parameter, der Actor profitiert von Features, die der Critic lernt.
  - Risiko: die riesigen Gradienten des Critics (quadrierter Fehler kann viel größer sein als der Policy-Gradient) können die des Actors überwältigen. Das Gewicht `vf_coef = 0,5` existiert, um das abzumildern.
- **Zwei getrennte Netze**:
  - Stabiler, Lernraten von Actor und Critic lassen sich unabhängig tunen.
  - Langsamer, mehr Speicher, kein Feature-Sharing.

Stable-Baselines3s PPO nutzt standardmäßig **getrennte Policy- und Wert-Heads auf einem geteilten Feature-Extractor**, was ein sinnvoller Kompromiss ist. Das Argument `policy_kwargs={"net_arch": [...]}` lässt dich umschalten.

Praktische Faustregel: nutze geteilt für niedrigdimensionale Beobachtungen (CartPole, einfache Godot-Szenen); nutze getrennt, wenn du Bild-Inputs oder stark unterschiedliche Skalen zwischen Actor und Critic hast.

---

## 10 · A2C vs. PPO: das eine verbleibende Problem

A2C ist ein vollständiger, funktionierender Algorithmus. Warum nutzt also überhaupt jemand PPO?

- A2C macht **ein Gradienten-Update pro Rollout**. Die Daten werden danach sofort verworfen.
- Mit teuren Simulatoren (Godot im Maßstab, Robotik, alles mit Bildern) ist jeder Rollout-Schritt kostbar. Wir hätten gerne **dasselbe Rollout für mehrere Gradientenschritte wiederverwendet**.
- Aber hier ist der Haken: nach dem ersten Gradientenschritt hat sich die Policy verschoben. Die Aktionen, die wir während des Rollouts genommen haben, kommen nicht mehr aus der *aktuellen* Policy — sie kamen aus der *alten* Policy. Die Vorteilsschätzungen, die für das erste Update funktionierten, werden für das zweite verzerrt.
- Naiv mehrere Epochen über das Rollout zu fahren macht A2C instabil. Die Policy kann weit von der datenerzeugenden Verteilung abdriften und alles bricht zusammen.

**PPOs geclipptes Objektiv ist der Fix.** Es führt ein Wahrscheinlichkeits-Verhältnis `r_t(θ) = π_new(a|s) / π_old(a|s)` ein und *clippt* es auf ein kleines Intervall um 1,0, sodass Updates, die die neue Policy zu weit von der alten weg drücken würden, Gradient null bekommen. Dadurch ist es sicher, **mehrere Epochen über ein einzelnes Rollout zu laufen**, was genau das ist, was `n_epochs=10` in deinem Unit-4-`gdrl`-Befehl tut.

---

## 11 · Wo A2C im Kurs auftaucht

Du hast die ganze Zeit A2C laufen lassen, verkleidet als PPO:

- **`gdrl`** nutzt unter der Haube SB3s PPO. PPO ist A2C plus ein geclipptes Surrogat-Objektiv plus Multi-Epoch-Updates plus GAE.
- Wenn du `n_steps=512` setzt, wählst du **A2Cs Rollout-Länge** aus Abschnitt 7.
- Wenn du `batch_size=256` setzt, wählst du die **Minibatch-Größe**, mit der PPO ein Rollout für mehrere Gradientenschritte zerhackt.
- Wenn du `n_epochs=10` setzt, entscheidest du, **wie oft dasselbe Rollout wiederverwendet wird** — das, was A2C nicht sicher kann, PPO aber schon.
- `ent_coef` ist das `β` aus Abschnitt 8.
- `vf_coef` ist das `c` aus Abschnitt 4.
- `clip_range` ist PPOs prinzipieller Ersatz für das Gradient Clipping aus Abschnitt 6.

Jetzt kannst du jede Zeile einer PPO-Config benennen, wenn du sie anschaust.

---

## 12 · Stretch Goals

Für Studierende, die vor PPO tiefer graben wollen:

- **Getrennte Actor- und Critic-Netze.** Refaktoriere `ActorCritic` in zwei Klassen mit zwei Optimizern. Vergleiche Trainingskurven auf CartPole. Du wirst wahrscheinlich etwas stabileres, aber langsameres Lernen sehen.
- **Probier LunarLander-v2.** Eine herausforderndere Umgebung, in der A2C typisch ~2M Schritte braucht. Beobachte die Entropie-Kurve genau — Entropie-Kollaps ist hier viel häufiger.
- **Visualisiere, was der Critic lernt.** Sample ein Gitter aus Beobachtungen, lass sie durch den Critic laufen, plotte `V(s)` als Heatmap (für 2D-Zustandsräume) oder als 1D-Kurve (für Wagenposition, Stabwinkel). Vergleiche mit den Rollout-Returns an diesen Zuständen.
- **Ersetze 1-Schritt-TD durch GAE-λ.** Implementiere Generalized Advantage Estimation mit `λ ∈ {0,9, 0,95, 1,0}` und beobachte, wie Varianz und Bias in der Praxis abgewogen werden. Das ist *genau* der Codepfad, der in SB3s PPO ausgeliefert wird.
- **Steck die Policy zurück in Godot.** Exportiere den Agenten erneut als ONNX und lade ihn in eine Godot-Szene wie in Unit 5, aber mit deinem eigenen A2C-Trainingsskript statt `gdrl`.

---

## 13 · Der Actor und Critic in SB3s PPO (optional beim ersten Lesen)

!!! note "Erster Durchgang? Überfliege oder überspringe diesen Abschnitt."
    Der Kernpfad sind die Abschnitte 1–11 — die Vorteilsfunktion, die A2C-Schleife, die CartPole-Implementierung und die Brücke zu PPO; komm hierher zurück, sobald du einen trainierten Godot-Agenten zum Inspizieren hast.

SB3s PPO ist eine Actor-Critic-Methode — sie hat genau die zwei Heads, die du in Abschnitt 5 gebaut hast. Die `ActorCritic`-Klasse, die du geschrieben hast, mappt direkt auf `model.policy` in einem trainierten SB3-Modell.

### Actor und Critic auf einem trainierten Godot-Agenten inspizieren

```python
from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import torch, numpy as np

env = StableBaselinesGodotEnv(env_path="./JumperHard.x86_64", n_parallel=1, speedup=1)
model = PPO.load("logs/sb3/jumper_baseline/best_model", env=env)

obs, _ = env.reset()
obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    # Actor: get action distribution
    dist = model.policy.get_distribution(obs_tensor)
    action_mean = dist.distribution.loc    # mean of Gaussian (continuous actions)
    action_std  = dist.distribution.scale  # std (exploration amount)

    # Critic: get value estimate
    value = model.policy.predict_values(obs_tensor)

print(f"Action mean: {action_mean.numpy()}")
print(f"Action std:  {action_std.numpy()}")
print(f"State value: {value.item():.3f}")
env.close()
```

Der `action_mean` ist das, was der Actor empfiehlt; `action_std` spiegelt wider, wie viel Unsicherheit (Erkundung) übrig ist — ein gut trainierter Agent hat niedrigeres std. Der `value` ist die Critic-Schätzung des erwarteten Returns aus diesem Zustand.

### Unit-Variablen auf SB3-Interna mappen

```
Unit-Variable             →  SB3-PPO-Äquivalent
──────────────────────────────────────────────────
actor_head               →  model.policy.action_net
critic_head              →  model.policy.value_net
geteilter Backbone       →  model.policy.mlp_extractor
Vorteil A_t              →  im Rollout-Buffer berechnet
ent_coef                 →  model.ent_coef
vf_coef                  →  model.vf_coef
n_steps                  →  model.n_steps (Rollout-Länge)
```

### TensorBoard-Verbindung

Jeder Loss-Term aus der kombinierten Loss-Formel in Abschnitt 4 hat ein TensorBoard-Pendant:

- `train/policy_gradient_loss` = L_actor aus dieser Unit — der Actor verbessert sich anhand der Vorteilsschätzungen
- `train/value_loss` = L_critic aus dieser Unit — der Critic minimiert den quadrierten TD-Fehler
- `train/entropy_loss` = L_entropy — der Entropie-Bonus hält die Erkundung am Leben

### Die Explained-Variance-Diagnose

`train/explained_variance` (in SB3s TensorBoard angezeigt) ist die nützlichste einzelne Metrik, um deinen Critic zu diagnostizieren. Sie misst, wie gut `V(s)` die tatsächlichen Returns vorhersagt:

- **Nahe 1,0** — der Critic hat eine gute Wertfunktion gelernt. Der Actor bekommt genaue Vorteilsschätzungen, und das Trainingssignal ist sauber.
- **Nahe 0 oder negativ** — der Critic ist nutzlos. Der Actor läuft im Wesentlichen REINFORCE mit hoher Varianz — genau das Problem, das diese Unit lösen sollte. Wenn du das siehst, ist der Critic untertrainiert: probier ein höheres `vf_coef`, mehr `n_steps` oder eine niedrigere Lernrate.

`explained_variance` während eines Godot-Trainingslaufs von nahe null Richtung 0,9+ klettern zu sehen, heißt den Critic in Echtzeit lernen zu sehen — derselbe Prozess, den du im CartPole-Code oben implementiert hast, nur im Maßstab.

---

## Was kommt als Nächstes

Du hast jetzt jede konzeptionelle Zutat, die PPO braucht. Die nächste Unit nimmt A2Cs Loss, tauscht `A_t · log π_θ(a_t | s_t)` gegen ein geclipptes Wahrscheinlichkeitsverhältnis, erlaubt mehrere Epochen über ein Rollout und geht das vollständige PPO-Update durch — den Algorithmus hinter jedem `gdrl`-Befehl, den du ausgeführt hast.

!!! info "Selbstcheck, bevor du weitermachst"
    Kannst du diese Fragen in eigenen Worten beantworten?

    1. Was repräsentiert der **Vorteil** A(s, a), und warum hat er niedrigere Varianz als der rohe Return G?
    2. Was geht schief, wenn der Critic viel *schlechter* ist als der Actor — wie sieht das auf TensorBoard aus?
    3. Warum updatet Actor-Critic *während* einer Episode, während REINFORCE auf das Ende warten muss?
    4. Was verhindert der **Entropie-Bonus**, und was passiert, wenn er zu groß ist?
    5. Was ist das *eine* verbleibende Problem von A2C, das PPO speziell zu beheben versucht?

    Wenn du alle fünf beantworten kannst — du bist bereit für den PPO-Deep-Dive.

??? success "Antworten zum Selbstcheck"
    1. Der **Vorteil** `A(s, a) = Q(s, a) - V(s)` misst, wie viel besser Aktion `a` ist als die *durchschnittliche* Aktion aus Zustand `s`. Er hat niedrigere Varianz als der rohe Return `G`, weil das Abziehen der Baseline `V(s)` den Anteil des Returns entfernt, der nur daher kam, dass der Zustand einfach oder schwer war — übrig bleibt allein der Beitrag der Aktion.
    2. Ein schwacher Critic liefert verrauschte Vorteilsschätzungen, der Actor läuft also faktisch wieder **REINFORCE mit hoher Varianz**. Die TensorBoard-Signatur ist `train/explained_variance` nahe 0 oder negativ — behebe es mit einem höheren `vf_coef`, mehr `n_steps` oder einer niedrigeren Lernrate.
    3. Weil der Critic **bootstrappt**: die 1-Schritt-Schätzung `A_t ≈ r + γ V(s') - V(s)` braucht nur eine echte Belohnung und die Vorhersage des nächsten Zustands. REINFORCEs `G_t` lässt sich schlicht nicht berechnen, bevor jede Belohnung nach Schritt `t` beobachtet wurde.
    4. Der **Entropie-Bonus** verhindert vorzeitigen Policy-Kollaps — dass die Wahrscheinlichkeit einer Aktion früh auf 1,0 gedrückt wird und der Agent für immer aufhört zu erkunden. Ist er zu groß, bleibt die Policy nahezu gleichverteilt (Entropie hängt nahe `ln 2` fest) und legt sich nie fest, die Belohnung bleibt also niedrig.
    5. A2C kann sicher nur **ein Gradienten-Update pro Rollout** machen — danach sind die Daten off-policy und die Vorteilsschätzungen werden verzerrt. PPOs geclipptes Wahrscheinlichkeitsverhältnis macht mehrere Epochen über dasselbe Rollout sicher, genau das, was `n_epochs=10` in `gdrl` tut.

[← Policy Gradients](unit-policy-gradients.md) · [Kursstartseite](index.md) · [→ PPO Deep Dive](unit-ppo-deep.md)
