# Policy Gradients — REINFORCE und das Policy-Gradient-Theorem

[← Deep Q-Learning](unit-03.md) · [Kursstartseite](index.md)

!!! note "Voraussetzungen"
    - **[RL Essentials](unit-01.md)** — MDP-Schleife, Return, Diskontierungsfaktor
    - **[RL Foundations Deep Dive](unit-rl-foundations-deep.md)** — On-Policy vs. Off-Policy, Monte Carlo vs. TD
    - **[Q-Learning-Unit](unit-q-learning.md)** — für den wertbasierten Kontrast in §1
    - **[Unit 3](unit-03.md)** (empfohlen) — DQN als wertbasiertes Gegenstück
    - Sicherheit mit Grund-Gradienten (`∇`, Kettenregel); den Log-Derivative-Trick leiten wir in §3 neu her

!!! info "Zeit"
    Lesen: ~40 min

---

!!! info "Drei Wege, deine KI zu beobachten"
    Python-Trainingsausgabe (Belohnung pro Episode) · matplotlib-Trainingskurve · CartPole-Rendering (`gym.make(..., render_mode="human")`)

Bisher hast du Agenten trainiert, die **Werte** lernen — Q-Learning schätzt Q(s,a) in einer Tabelle, DQN approximiert es mit einem neuronalen Netz. In beiden Fällen wird die Policy *abgeleitet*: du nimmst `argmax_a Q(s,a)`. Diese Unit geht den umgekehrten Weg. Statt Werte zu lernen und eine Policy daraus zu pressen, **parametrisieren** wir die Policy selbst und optimieren sie direkt mit Gradientenaufstieg.

Das ist die Grundlage, auf der PPO, A2C, SAC und fast jeder moderne Deep-RL-Algorithmus aufbaut. Sobald du REINFORCE verstehst, sind die anderen nur Stabilitätstricks obendrauf.

---

## 1 · Wertbasiert vs. policy-basiert: die Grundsatzentscheidung

Wertbasiertes RL kennst du schon:

- DQN lernt `Q(s,a)` — den erwarteten Return, wenn man in Zustand `s` Aktion `a` ausführt.
- Die Policy ist implizit: `π(s) = argmax_a Q(s,a)`.
- Exploration wird draufgesetzt (ε-greedy).

**Policy-basiertes RL** dreht das Bild um:

- Du lernst `π(a|s)` direkt — eine parametrisierte Funktion (meist ein neuronales Netz), die Zustände auf eine *Wahrscheinlichkeitsverteilung* über Aktionen abbildet.
- Kein `argmax`. Du samplest Aktionen aus der Verteilung.
- Exploration ist eingebaut: die Policy ist von Natur aus stochastisch, bis sie sich schärft.

### Wann policy-basiert gewinnt

- **Kontinuierliche Aktionsräume.** Wenn `a ∈ ℝ` (z. B. ein Lenkwinkel in [-1, 1]), erfordert `argmax_a Q(s,a)` ein Optimierungsproblem *in jedem Schritt*. Policy-Netze geben einfach einen Mittelwert und eine Standardabweichung aus — samplen, fertig.
- **Stochastische Policies sind erforderlich.** Denk an Schere-Stein-Papier. Jede deterministische Policy ist ausnutzbar: spielst du immer Stein, spielt der Gegner immer Papier. Die Nash-optimale Policy ist gleichverteilt zufällig. Eine Q-Tabelle kann das nicht darstellen; ein Policy-Netz schon.
- **Hochdimensionale Aktionsräume.** Bei 50 Gelenken an einem Humanoiden ist eine Enumeration von Aktionen unmöglich. Ein Policy-Netz gibt 50 Mittelwerte und 50 Standardabweichungen aus — unkompliziert.

### Wann wertbasiert gewinnt

- **Diskrete Aktionen mit kleiner Aktionsmenge.** DQN ist hier oft stichproben-effizienter.
- **Deterministische Umgebungen mit einer optimalen Aktion pro Zustand.** Eine gierige Wertmethode trifft das schnell.
- **Off-Policy-Lernen aus einem Replay-Buffer.** Vanilla REINFORCE ist on-policy — du musst frische Stichproben der *aktuellen* Policy nutzen. (PPO erweitert das mit Importance Sampling, um Daten für ein paar Epochen sicher wiederzuverwenden.)

| | Wertbasiert (DQN) | Policy-basiert (REINFORCE) |
|--|--|--|
| Lernt | `Q(s,a)` | `π(a|s)` |
| Policy | argmax (deterministisch) | Sample (stochastisch) |
| Aktionsraum | Diskret | Diskret **oder kontinuierlich** |
| Exploration | ε-greedy-Aufsatz | In die Policy eingebaut |
| Datenrffizienz | Replay-Buffer (off-policy) | Frische Stichproben (Vanilla REINFORCE ist on-policy) |

---

## 2 · Die Policy parametrisieren

Eine Policy ist eine Funktion `π_θ(a|s)` mit Parametern `θ` (den Netzgewichten). Sie nimmt einen Zustand und gibt eine Wahrscheinlichkeitsverteilung zurück.

### Diskrete Aktionen — Softmax-Head

Für `N` diskrete Aktionen gibt das Netz `N` Logits aus. Ein Softmax wandelt sie in Wahrscheinlichkeiten:

```
π_θ(a_i | s) = exp(z_i) / Σ_j exp(z_j)
```

Im Klartext: das Logit `z_i` ist der „Score" für Aktion `i`; Softmax verwandelt Scores in eine ordentliche Wahrscheinlichkeitsverteilung, die sich zu 1 summiert.

### Kontinuierliche Aktionen — Gauss-Head

Für eine kontinuierliche Aktion `a ∈ ℝ^d` gibt das Netz einen Mittelwert `μ_θ(s)` und (oft) eine logarithmische Standardabweichung `log σ_θ` aus. Du samplest aus einer Gauss-Verteilung:

```
a ~ N(μ_θ(s), σ_θ²)
```

Im Klartext: das Netz sagt grob voraus, wo die gute Aktion liegt (`μ`), und wie sicher es ist (`σ`). Kleineres `σ` = schärfer, deterministischer.

### Das Ziel

Wir wollen den **erwarteten diskontierten Return** unter unserer Policy maximieren:

```
J(θ) = E_{τ ~ π_θ} [ Σ_t γ^t r_t ]
```

Term für Term:
- `τ` ist eine Trajektorie `(s_0, a_0, r_0, s_1, a_1, r_1, …)`, gesampelt durch das Ausführen der Policy.
- `γ ∈ [0, 1)` ist der Diskontierungsfaktor — zukünftige Belohnungen sind weniger wert.
- `r_t` ist die Belohnung bei Schritt `t`.
- Der Erwartungswert läuft über die Zufälligkeit in Policy *und* Umwelt.

Wir wollen `θ* = argmax_θ J(θ)`. Bei differenzierbarem `J` würden wir einfach Gradientenaufstieg machen: `θ ← θ + α ∇_θ J(θ)`. Das Problem — adressiert im nächsten Abschnitt — ist, dass `J` die Umgebung einbezieht, und die ist **nicht** differenzierbar.

---

## 3 · Das Policy-Gradient-Theorem

### Das Problem

`J(θ)` hängt durch zwei Kanäle von `θ` ab:

1. Die Aktionswahrscheinlichkeiten `π_θ(a|s)` — differenzierbar.
2. Die Trajektorie der Zustände, die die Umgebung daraufhin erzeugt — **nicht** differenzierbar. Durch `env.step()` lässt sich nicht backpropagieren.

Wie berechnen wir also `∇_θ J(θ)`?

### Der Trick — Log-Derivative (Score-Funktion)

Eine kleine Identität aus der Analysis:

```
∇_θ π_θ(a|s) = π_θ(a|s) · ∇_θ log π_θ(a|s)
```

Das folgt aus `∇ log f = ∇f / f`. Beide Seiten mit `f` multipliziert ergibt die Zeile oben. Sieht unschuldig aus, ist aber der Schlüssel zum ganzen Feld.

Wendest du das im Erwartungswert an und machst etwas Algebra (übersprungen — siehe Sutton & Barto Kap. 13), kommst du beim **Policy-Gradient-Theorem** an:

```
∇_θ J(θ) = E_{τ ~ π_θ} [ Σ_t ∇_θ log π_θ(a_t | s_t) · G_t ]
```

Term für Term:
- `∇_θ log π_θ(a_t | s_t)` — wie man `θ` schubst, um die Aktion `a_t` wahrscheinlicher zu machen. PyTorch rechnet das per Autograd auf `log_prob` für dich.
- `G_t = Σ_{k≥t} γ^{k-t} r_k` — der **Return ab Schritt t** (auch *returns-to-go* genannt).
- Der Erwartungswert `E_{τ ~ π_θ}` — wir schätzen ihn durch Episoden-Stichproben.

### Im Klartext

> Erhöhe die Wahrscheinlichkeit von Aktionen, die zu hohem Return führten; senke die Wahrscheinlichkeit von Aktionen, die zu niedrigem Return führten. Die Größe des Schubs ist proportional zum Return.

Entscheidend: die Nicht-Differenzierbarkeit der Umgebung ist weg. Wir differenzieren nie durch `env.step()` — nur durch `log π_θ`, das ist bloß ein Forward-Pass eines neuronalen Netzes. Der Return `G_t` ist ein **skalares Gewicht**, beim Backprop wie eine Konstante behandelt.

### Warum das schätzbar ist

Wir approximieren den Erwartungswert mit einer (oder wenigen) gesampelten Trajektorien:

```
∇_θ J(θ) ≈ Σ_t ∇_θ log π_θ(a_t | s_t) · G_t
```

Eine Episode laufen lassen, `(s_t, a_t, G_t)` für jeden Schritt sammeln, Gradienten aufsummieren, einen Schritt machen. Das ist REINFORCE.

---

## 4 · REINFORCE-Algorithmus

Williams, 1992. Die einfachste mögliche Policy-Gradient-Methode.

### Algorithmus

1. Policy-Netz `π_θ` zufällig initialisieren.
2. **Rollout** einer vollständigen Episode mit `π_θ`: sammle `(s_0, a_0, r_0), …, (s_T, a_T, r_T)`.
3. **Returns-to-go berechnen**: `G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + … + γ^(T-t)·r_T` für jedes `t`.
4. **Loss berechnen**:
   ```
   L(θ) = - Σ_t G_t · log π_θ(a_t | s_t)
   ```
   Das Minuszeichen, weil PyTorch Losses *minimiert*, wir aber `J` *maximieren* wollen.
5. Backprop und Optimizer-Schritt.
6. Ab Schritt 2 wiederholen, bis Konvergenz.

### Vollständige PyTorch-Implementierung auf CartPole-v1

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

env = gym.make("CartPole-v1")
policy = PolicyNetwork(4, 2)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
gamma = 0.99

def compute_returns(rewards, gamma):
    G, returns = 0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)

for episode in range(1000):
    obs, _ = env.reset()
    log_probs, rewards = [], []
    done = False

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        probs = policy(obs_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))

        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated

    returns = compute_returns(rewards, gamma)

    # Normalize returns (variance reduction)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Policy gradient loss (negative because we do gradient ASCENT)
    loss = -torch.stack(log_probs) @ returns

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 100 == 0:
        print(f"Episode {episode}: total reward = {sum(rewards):.0f}")
```

### Was zu erwarten ist

Lass das laufen. Du siehst etwa:

```
Episode 0:   total reward = 23
Episode 100: total reward = 47
Episode 200: total reward = 89
Episode 400: total reward = 156
Episode 700: total reward = 200
```

CartPole-v1s maximale Belohnung ist **500** (das Episodenlängen-Limit). REINFORCE erreicht das Limit typisch in **500–800 Episoden**. Aber die Kurve ist **verrauscht** — manche Episoden fallen zufällig zurück auf 30. Das führt uns zur zentralen Schwäche von REINFORCE.

!!! check "Fertig, wenn"
    CartPole-v1 gilt als **gelöst bei einem durchschnittlichen Return ≥ 475 über 100 aufeinanderfolgende Episoden** (Maximum 500). Mit diesem Skript erreichen einzelne Episoden das 500er-Limit typischerweise erstmals im Bereich von 500–800 Episoden — beurteile den Erfolg aber am gleitenden Durchschnitt, nicht an einer einzelnen Glücksepisode. Hängen die Belohnungen nach ein paar hundert Episoden noch um die anfänglichen ~20, prüfe das Vorzeichen des Loss und die `compute_returns`-Logik, bevor du länger trainierst.

---

## 5 · Das Varianzproblem

!!! warning "REINFORCE ist unverzerrt, hat aber sehr hohe Varianz"
    Die Gradientenschätzung ist *im Erwartungswert* korrekt, jede einzelne Stichprobe kann aber wild daneben liegen. Das ist das größte einzelne Problem von Vanilla Policy Gradients.

Warum ist die Varianz so hoch?

- Eine einzelne Trajektorie ist eine **einzelne Stichprobe** eines hochdimensionalen Zufallsprozesses.
- Zwei Episoden derselben Policy können Returns liefern, die sich rein durch Zufall um Faktor 10 unterscheiden — zufällige Umweltdynamik, glückliche/unglückliche Erkundung.
- Der Gradient ist `∇log π · G`. Schwankt `G` zwischen Episoden stark, schwanken auch die Gradienten-Updates.

Praktische Folgen:

- Trainingskurven oszillieren stark. Du siehst Läufe auf 200 hochgehen, dann zurück auf 50 stürzen.
- Stichprobeneffizienz ist schlecht — du brauchst tausende Episoden für Probleme, die DQN in Hunderten löst.
- Bei härteren Aufgaben (lange Episoden, spärliche Belohnungen) lernt Vanilla REINFORCE oft gar nichts.

Ein gängiger billiger Fix steht im Code oben: **Returns normalisieren** auf Mittelwert null und Einheitsvarianz innerhalb der Episode. Das ist ein Hack — theoretisch nicht gerechtfertigt — hilft aber stark in der Praxis. Den prinzipiellen Fix bringt der nächste Abschnitt.

### Warum der Score-Funktion-Schätzer hohe Varianz hat

Der Gradient `∇log π(a|s) · G_t` multipliziert eine Log-Wahrscheinlichkeit mit dem Return. Der Return G_t hat hohe Varianz, weil unterschiedliche Episoden wild verschiedene Pfade durch die Umgebung nehmen. Das Produkt verstärkt diese Varianz — jeder Gradienten-Schritt ist eine verrauschte Schätzung der wahren Gradientenrichtung. Eine Baseline b(s) entfernt das „wie gut ist dieser Zustand überhaupt"-Rauschen und isoliert „wie gut war diese spezifische Aktion" — Varianzreduktion ohne den Gradienten im Erwartungswert zu ändern (denn `E[∇log π · b(s)] = 0`).

### Der Reparameterisierungs-Trick (SAC, varianzärmere Alternative) (optional beim ersten Lesen)

!!! note "Erster Durchgang? Überfliege oder überspringe diesen Abschnitt."
    Der rote Faden von REINFORCE — Theorem (§3), Algorithmus (§4), Varianz (§5), Baseline (§6) — geht direkt in §6 weiter; dieser Unterabschnitt zeigt nur vorab, wie SAC das Varianzproblem umgeht, und wird erst in der SAC-Unit wirklich gebraucht.

Statt `a ~ π(a|s)` zu samplen und *durch* den Sampling-Schritt zu differenzieren (was der Score-Funktion-Schätzer tut), schreibe:

```
z ~ N(0, 1)
a = μ_θ(s) + σ_θ(s) · z
```

Jetzt lebt die Zufälligkeit komplett in `z`, das nicht von θ abhängt. Der Gradient fließt direkt durch `μ_θ` und `σ_θ` — unter Umgehung des diskreten Sampling. Viel geringere Varianz. Deshalb hat SACs Actor selbst ohne Baseline varianzarme Gradienten. (Wir nutzen hier `z` statt `ε`, um den Konflikt mit dem ε-greedy-Erkundungssymbol in wertbasierten Methoden zu vermeiden.) Siehe [SAC-Unit](unit-sac.md) §7 für Implementierungsdetails.

| Schätzer | Genutzt in | Varianz |
|----------|------------|---------|
| Score-Funktion (REINFORCE) | REINFORCE, A2C, PPO | Hoch (braucht Baseline) |
| Reparameterisierung | SAC-Actor | Niedrig |
| Keiner (kein Policy Gradient) | DQN | N/A |

**Warum PPO die Score-Funktion nutzt, nicht Reparameterisierung:** PPOs geclipptes Ratio-Objektiv erfordert das Differenzieren durch die Log-Wahrscheinlichkeit, um zu messen, wie stark sich die Policy verändert hat. Reparameterisierung würde dieses Signal entfernen. Der Score-Funktion-Schätzer ist die richtige Wahl, wenn das Policy-Ratio Teil des Objektivs ist — und die Baseline (über GAE) behandelt das Varianzproblem.

---

## 6 · Varianzreduktion durch Baseline

!!! tip "Eine Baseline abzuziehen lässt den Gradienten unverzerrt"
    Hier die magische Identität:
    ```
    E_π [ ∇ log π_θ(a|s) · b(s) ] = 0
    ```
    für *jede* Funktion `b(s)`, die nicht von `a` abhängt. Zieh sie ab, so viel du willst.

So können wir den Policy-Gradient umschreiben:

```
∇_θ J(θ) = E_π [ ∇_θ log π_θ(a_t|s_t) · (G_t − b(s_t)) ]
```

Term für Term:
- `G_t − b(s_t)` — der Return *minus eine Baseline*. Gleicher Gradient im Erwartungswert, typisch viel kleinere Varianz.
- `b(s_t)` — alles, was nur vom Zustand abhängt, nicht von der Aktion.

### Die optimale Baseline

Die varianzminimierende Baseline ist die **Zustands-Wertfunktion** `V(s) = E_π[G_t | s_t = s]`: der erwartete Return ab Zustand `s` unter der aktuellen Policy.

Mit dieser Baseline:

```
A(s_t, a_t) = G_t − V(s_t)
```

heißt **Vorteil (advantage)**. Im Klartext:

> Der Vorteil sagt dir, wie viel besser (oder schlechter) die Aktion, die du genommen hast, war — verglichen mit der durchschnittlichen Aktion, die du in diesem Zustand normalerweise nehmen würdest.

Der modifizierte Loss wird:

```
L(θ) = − Σ_t A_t · log π_θ(a_t | s_t)
```

### Eine Baseline zum CartPole-Code hinzufügen

Du kannst `V_φ(s)` als zweites Netz fitten und es darauf trainieren, die beobachteten Returns vorherzusagen (MSE-Loss). Policy-Netz und Wertenetz teilen strukturell nichts:

```python
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

value_net = ValueNetwork(4)
value_optim = optim.Adam(value_net.parameters(), lr=1e-3)

# Inside the training loop, after collecting log_probs, rewards, obs_buffer:
obs_tensor = torch.tensor(np.array(obs_buffer), dtype=torch.float32)
returns = compute_returns(rewards, gamma)

# Critic update: fit V(s) to observed returns
values = value_net(obs_tensor)
value_loss = ((returns - values) ** 2).mean()
value_optim.zero_grad()
value_loss.backward()
value_optim.step()

# Advantage = return - baseline (detach so policy grad doesn't flow into critic)
advantages = (returns - value_net(obs_tensor).detach())
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

policy_loss = -torch.stack(log_probs) @ advantages
optimizer.zero_grad()
policy_loss.backward()
optimizer.step()
```

!!! check "Fertig, wenn"
    Lass das parallel zu Vanilla REINFORCE laufen. Die Baseline-Version erreicht eine durchschnittliche Belohnung von 200 in **deutlich weniger Episoden** als der Vanilla-Lauf, und ihre Belohnungskurve oszilliert sichtbar weniger zwischen Episoden. Sind die beiden Kurven nicht zu unterscheiden, prüfe, ob die Vorteile `.detach()` auf der Critic-Ausgabe nutzen und ob `value_loss` tatsächlich sinkt.

Du hast gerade dein erstes **Actor-Critic** gebaut — die Policy ist der Actor, `V_φ` ist der Critic.

---

## 7 · Das Credit-Assignment-Problem

In einer 500-Schritt-CartPole-Episode, die mit Versagen endete, **welcher Schritt war schuld**? Die Aktionen in den ersten 50 Schritten waren wahrscheinlich okay; der schlechte war so um Schritt 480.

Vanilla „Gesamtepisoden-Return für jeden Schritt" rechnet allen 500 Aktionen den Gesamtwert gleichermaßen zu. Das ist schlecht. Wir wollen jede Aktion nur für die Belohnungen anrechnen, die *danach* kamen.

Genau deshalb berechnen wir `G_t` als **Return ab Schritt t** — nicht als Gesamtepisoden-Return. Der frühere Code macht das schon:

```python
def compute_returns(rewards, gamma):
    G, returns = 0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)
```

Rückwärts gelesen: `G_T = r_T`, `G_{T-1} = r_{T-1} + γ·r_T` und so weiter. Aktion `a_t` wird mit dem gewichtet, was *ab t* passierte, nicht vorher.

Das ist mathematisch gerechtfertigt: Belohnungen, die *vor* einer Aktion empfangen wurden, können nicht durch diese Aktion verursacht worden sein, also tragen sie im Erwartungswert null Gradient bei. Sie wegzulassen reduziert die Varianz, ohne Verzerrung einzuführen.

Der Diskontierungsfaktor `γ` hilft ebenfalls: weit in der Zukunft liegende Belohnungen tragen weniger bei, sodass der Gradient sich auf die zeitlich nahen Konsequenzen jeder Aktion fokussiert.

---

## 8 · Von REINFORCE zu Actor-Critic

REINFORCE funktioniert, hat aber zwei strukturelle Limitierungen:

1. **Es braucht vollständige Episoden.** Returns `G_t` werden durch Aufsummieren bis zum Ende berechnet. Mitten in der Episode lässt sich nichts updaten.
2. **Varianz ist trotz Baseline weiterhin hoch**, weil `G_t` selbst eine verrauschte Monte-Carlo-Stichprobe ist.

Der Fix ist, den Monte-Carlo-Return durch eine **bootstrapped-Schätzung** aus einer gelernten Wertfunktion zu ersetzen, genau wie DQNs TD-Targets:

```
G_t ≈ r_t + γ · V_φ(s_{t+1})
```

Jetzt ist der Vorteil:

```
A_t = r_t + γ · V_φ(s_{t+1}) − V_φ(s_t)
```

Das ist der **1-Schritt-TD-Vorteil**, und du kannst ihn nach jedem einzelnen Schritt berechnen. Kein Warten auf das Episodenende.

Architektonisch:

- **Actor** = die Policy `π_θ`. Gibt Aktionen aus.
- **Critic** = die Wertfunktion `V_φ`. Gibt skalare Wertschätzungen aus.
- Der Actor wird per Policy-Gradient mit Vorteilen vom Critic trainiert.
- Der Critic wird per TD-Regression trainiert (genau wie DQN, aber sagt `V` statt `Q` voraus).

> Actor-Critic kombiniert die **Stabilität von Wert-Methoden** (varianzarme TD-Targets) mit der **Allgemeinheit von Policy-Methoden** (funktioniert für kontinuierliche Aktionen, lernt stochastische Policies).

Das ist die Familie, zu der A2C, A3C, PPO, SAC gehören. Die nächste Unit baut das vollständig aus.

---

## 9 · Verbindung zu PPO

Du hast PPO in früheren Godot-Units (RL Essentials und Unit 2) bereits genutzt. Es lohnt sich, kurz zu sehen, wie PPO zu dem passt, was du gerade gelernt hast:

PPO **ist** eine Policy-Gradient-Methode. Sein Kernobjektiv ist nach wie vor:

```
∇ log π_θ(a|s) · A(s,a)
```

Was PPO obendrauf legt:

1. **Eine geclippte Wahrscheinlichkeits-Ratio.** Statt `log π_θ` nutzt PPO die Ratio `π_θ(a|s) / π_old(a|s)` und clippt sie auf `[1−ε, 1+ε]`. Das verhindert, dass sich die Policy in einem Update zu stark ändert — eine Hauptquelle der Instabilität in Vanilla Policy Gradient.
2. **Generalized Advantage Estimation (GAE).** Statt 1-Schritt-TD oder vollem Monte Carlo interpoliert GAE zwischen beidem, gesteuert durch einen Parameter `λ`. Ein Regler zwischen high-bias-low-variance (TD) und low-bias-high-variance (Monte Carlo).
3. **Mehrere Epochen pro Batch.** PPO nutzt jeden Datenbatch für mehrere Optimizer-Schritte, was nur dank des Clippings sicher ist.
4. **Minibatching.** Große Batches werden für SGD in Minibatches geteilt.

> Mentales Modell: **PPO ist REINFORCE mit Critic, geclipptem Update, GAE-Vorteilen und Minibatch-Wiederverwendung.** Alles, was du in dieser Unit gelernt hast, gilt unverändert weiter — nur die Stabilitätsmaschinerie ist neu.

Wenn du den PPO-Loss in `stable-baselines3` liest, erkennst du jeden Term wieder.

---

## 10 · Stretch Goals

Probiere das aus, um dein Verständnis zu vertiefen:

### 10.1 PixelCopter — Varianz in der Wildnis beobachten

Ersetze `CartPole-v1` durch `PixelCopter-PLE-v0` (aus `gym-pygame` oder `pygame-learning-environment`). Episoden sind länger und Belohnungen spärlicher. Vanilla REINFORCE wird sichtbar Probleme bekommen — du siehst genau, warum wir Actor-Critic brauchen.

### 10.2 Entropie-Bonus

Füge dem Loss einen Erkundungs-Bonus hinzu:

```python
entropy = dist.entropy().mean()
loss = policy_loss - 0.01 * entropy
```

Term für Term:
- `dist.entropy()` — die Shannon-Entropie der aktuellen Policy. Höher = gleichverteilter = mehr Erkundung.
- Der `-0.01 * entropy`-Term schiebt die Policy *weg* davon, zu früh zu scharf zu werden.

Vergleiche Training mit und ohne. Mit dem Bonus hält die Policy Erkundung länger und findet bei schwereren Aufgaben oft bessere Lösungen. Genau das nutzt PPO (`ent_coef`-Parameter in stable-baselines3).

### 10.3 Trainingsstatistiken mit matplotlib plotten

Verfolge `ep_rew_mean` und `ep_len_mean` (gleitender Mittelwert über die letzten 50 Episoden) und plotte sie:

```python
import matplotlib.pyplot as plt
plt.plot(rolling_mean(episode_rewards, window=50))
plt.xlabel("Episode")
plt.ylabel("Mean reward (50-ep window)")
plt.title("REINFORCE on CartPole-v1")
plt.savefig("reinforce_curve.png")
```

Vergleiche drei Kurven auf denselben Achsen:
- Vanilla REINFORCE (mit Return-Normalisierung)
- REINFORCE + Werte-Baseline
- REINFORCE + Werte-Baseline + Entropie-Bonus

Du siehst die Varianzreduktion mit eigenen Augen entstehen.

---

## 11 · REINFORCE und die gdrl-Trainingsschleife

Jedes Mal, wenn du `gdrl --env_path=... --timesteps=...` ausführst, läuft PPO — das IST eine Policy-Gradient-Methode. REINFORCE ist die konzeptionelle Grundlage; PPO ist REINFORCE mit Stabilitätstricks obendrauf. Der Loss, den du in Abschnitt 4 minimiert hast, und der, den SB3 optimiert, sind dieselbe Gleichung.

### Mapping von REINFORCE auf das, was du in gdrl siehst

```
REINFORCE-Konzept         →  gdrl-/SB3-Pendant
─────────────────────────────────────────────────────
Episoden-Rollout          →  n_steps Rollout-Sammlung
log π(a|s)                →  policy_gradient_loss in TensorBoard
Return G_t                →  Vorteilsschätzung (mit GAE)
Policy-Update             →  n_epochs Gradientenschritte
Erkundung über Entropie   →  ent_coef-Parameter
```

### TensorBoard durch die REINFORCE-Brille lesen

Wenn du in TensorBoard auf `train/policy_gradient_loss` schaust, siehst du REINFORCEs Loss — `−Σ G_t · log π(a_t|s_t)` — der minimiert wird. Die Kurve ist früh verrauscht (hohe Varianz, wie REINFORCE) und glättet sich, sobald die Policy schärfer wird und GAE bessere Vorteilsschätzungen liefert.

Der `--ent_coef`-Flag in `gdrl` ist der Entropie-Bonus aus Abschnitt 10.2 dieser Unit — derselbe `−0,01 * entropy`-Term, den du dem CartPole-Loss hinzugefügt hast. SB3s Default ist `ent_coef=0.0` für PPO, du hast ihn aber wahrscheinlich angehoben (z. B. `--ent_coef 0.01`), um frühen Policy-Kollaps zu verhindern.

### Der zentrale Unterschied: G_t vs. GAE

REINFORCE nutzt das echte `G_t` — einen Monte-Carlo-Return aus der vollen Episode. PPO nutzt einen **GAE-Vorteil** — eine Mehrschritt-TD-Schätzung (siehe PPO-Deep-Dive-Unit). GAE ist ein Regler zwischen reinem MC (hohe Varianz, kein Bias) und reinem TD (niedrige Varianz, etwas Bias). Deshalb sind PPOs `train/policy_gradient_loss`-Kurven sichtbar glatter als ein Vanilla-REINFORCE-Lauf in derselben Umgebung.

### Praktischer Vergleich

Hast du CartPole in dieser Unit mit REINFORCE trainiert, kannst du seine TensorBoard-Loss-Kurven mit einem `gdrl`-PPO-Lauf vergleichen — gleicher Loss-Typ, glatter bei PPO dank GAE und Clipping. Die Formen sollten erkennbar ähnlich sein: ein verrauschter Loss, der tendenziell nach unten geht, während die Policy besser wird, mit einem Vorzeichenwechsel im PolicyGradientLoss, sobald der Agent konsistent positive Vorteile bekommt.

---

## Was kommt als Nächstes

Du verstehst nun:

- Warum wir Policy-basierte Methoden brauchen (kontinuierliche Aktionen, stochastische Policies, Skalierung).
- Wie das Policy-Gradient-Theorem ein nicht-differenzierbares RL-Objektiv per Log-Derivative-Trick in einen handhabbaren Gradienten verwandelt.
- Wie REINFORCE diesen Gradienten aus gesampelten Episoden schätzt.
- Warum Varianz das zentrale praktische Problem ist und wie Baselines und Credit Assignment sie reduzieren.
- Wie das Hinzufügen einer gelernten Wertfunktion REINFORCE zu Actor-Critic macht — ohne vollständige Episoden auskommen.
- Warum PPO einfach REINFORCE mit Stabilitätstricks ist.

In der nächsten Unit bauen wir einen kompletten Actor-Critic — zwei Netze, die im Gleichschritt trainieren — und sehen, wie die Varianz gegenüber Vanilla REINFORCE zusammenbricht. Danach schichten wir die PPO-Tricks Stück für Stück oben drauf und verbinden alles wieder mit Godot.

!!! info "Selbstcheck, bevor du weitermachst"
    Kannst du diese Fragen in eigenen Worten beantworten?

    1. Nenne zwei Dinge, die eine Policy-basierte Methode kann und eine wertbasierte nicht.
    2. Schreibe den Policy-Gradient in einer Zeile auf und erkläre, was jeder Faktor tut.
    3. Was macht der **Log-Derivative-Trick** zu etwas Differenzierbarem, und warum?
    4. Warum ist **Varianz** das zentrale praktische Problem von REINFORCE, und wie reduziert eine Baseline sie ohne den Gradienten zu verzerren?
    5. Wodurch ersetzt Actor-Critic den Monte-Carlo-Return — und warum ist das ein *Bias-Varianz-Trade*?

    Wenn du alle fünf beantworten kannst — du bist bereit für Actor-Critic.

??? success "Antworten zum Selbstcheck"
    1. **Kontinuierliche oder hochdimensionale Aktionsräume** handhaben (das Netz gibt einfach μ und σ aus — kein `argmax`-Optimierungsproblem pro Schritt) und **stochastische Policies** darstellen, etwa die gleichverteilt zufällige Nash-Strategie bei Schere-Stein-Papier — ein argmax über Q-Werte kann das nie ausdrücken.
    2. `∇_θ J(θ) = E[ Σ_t ∇_θ log π_θ(a_t|s_t) · G_t ]`. Der **Log-Prob-Gradient** ist die Richtung in θ, die die Aktion `a_t` wahrscheinlicher macht; der Return **G_t** ist ein skalares Gewicht, das diesen Schubs skaliert — Aktionen mit hohem nachfolgendem Return werden wahrscheinlicher, solche mit niedrigem unwahrscheinlicher.
    3. Er verwandelt den Gradienten eines Erwartungswerts in einen Erwartungswert über `∇ log π_θ` — einen schlichten **Forward-Pass eines neuronalen Netzes**, den Autograd berechnet. Durch das nicht-differenzierbare `env.step()` wird nie backpropagiert; der Return geht nur als konstantes Gewicht ein.
    4. Der Gradient wird aus einer **einzelnen gesampelten Trajektorie** geschätzt, und Returns können sich zwischen Episoden rein zufällig um Faktor 10 unterscheiden — Updates sind also verrauscht und Trainingskurven oszillieren. Eine Baseline `b(s)` abzuziehen ist im Erwartungswert gratis (`E[∇log π · b(s)] = 0`), entfernt aber das „wie gut ist dieser Zustand überhaupt"-Rauschen und lässt den varianzärmeren **Vorteil** `G_t − V(s_t)` übrig.
    5. Durch die **bootstrapped TD-Schätzung** `r_t + γ·V_φ(s_{t+1})`. Sie senkt die Varianz drastisch (kein Monte-Carlo-Rauschen über die volle Episode, Updates mitten in der Episode möglich) — um den Preis von **Bias**, weil `V_φ` selbst eine unvollkommene gelernte Schätzung ist.

[→ Actor-Critic](unit-actor-critic.md)
