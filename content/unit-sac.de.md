# SAC — Soft Actor-Critic für kontinuierliche Steuerung

PPO ist der Default in diesem Kurs, und das aus gutem Grund: es handhabt diskrete und kontinuierliche Aktionen, skaliert mit parallelen Umgebungen und ist der Algorithmus der Wahl für spielartige Godot-Umgebungen. Aber sobald du moderne Robotik- oder Continuous-Control-Paper liest, triffst du überall einen anderen Namen — **SAC (Soft Actor-Critic)**. Es dominiert Benchmarks für kontinuierliche Steuerung, Lernen auf echten Robotern und jede Domäne, in der Simulation teuer ist. Diese Unit lehrt SAC von Anfang bis Ende: die Theorie, die Architektur, die Gleichungen, und wie du PPO in deinem bestehenden Godot-Workflow durch SAC ersetzt.

[← PPO in der Praxis](unit-04.md) · [Kursstartseite](index.md)

!!! note "Voraussetzungen"
    - **[Unit 4](unit-04.md)** — sicher mit PPO; du brauchst die On-Policy- / Off-Policy-Intuition
    - **[Actor-Critic](unit-actor-critic.md)** — SAC ist „Actor-Critic mit Entropie" — die Actor-/Critic-Trennung sollte selbstverständlich sein
    - **[Unit 3](unit-03.md)** — Replay-Buffer (replay buffer) und Target-Netze (SAC nutzt beide)
    - PyTorch-Lese-Komfort (§4 ruft SB3s SAC; §3 läuft die Mathe dahinter durch)

!!! info "Zeit"
    Lesen: ~40 min · Training: ~20 min GPU / ~1,5 h CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    TensorBoard (`train/ent_coef`, `rollout/ep_rew_mean`) · Godot-Viz-Checkpoint — SAC-Policies sehen oft sichtbar glatter aus als PPO auf derselben Aufgabe · Ein Sample-Effizienz-Vergleichsplot (SAC vs. PPO bei gleicher Schrittzahl)

---

## 0 · Die PPO-Lücke bei kontinuierlicher Steuerung

Du hast vier Units mit PPO verbracht und stehst kurz davor, diese Loyalität zu brechen. Um zu verstehen, *warum*, schau genau an, was PPO mit Erfahrung anstellt.

**PPO ist on-policy.** Jede Transition, die der Agent sammelt, ist mit der Policy-Version markiert, die sie erzeugt hat. PPO nutzt jede Transition für `n_epochs` Gradienten-Durchläufe — typisch 10 — und *wirft sie dann weg*. Das nächste Rollout nutzt eine leicht andere Policy, also sind alte Transitionen unter dem Importance-Sampling-Verhältnis, auf das das geclippte Objektiv setzt, nicht mehr valide.

Für spielartige Umgebungen ist das in Ordnung: Godot kann 8, 16, sogar 32 parallele Umgebungen mit 20× Beschleunigung simulieren. Du sammelst Millionen billiger Transitionen pro Stunde, und das Verschwenderische des Wegwerfens schmerzt nicht sehr.

**Aber kontinuierliche Steuerung ist anders.**

| Domäne | Kosten pro Transition | Parallele Envs machbar? |
|--------|--------------------|------------------------|
| Godot-Spiel-Env (BallChase, JumperHard) | Mikrosekunden | 8–32 parallel, einfach |
| MuJoCo-Roboter-Simulation | Millisekunden | Wenige parallel |
| Echter Roboterarm | Sekunden | Einer |
| Teure Physik-Sim (Fluid, Kontakt) | Sekunden bis Minuten | Einer |

Sobald jede Transition echte Wall-Clock-Zeit kostet, wird es schmerzhaft, sie nach einem Rollout wegzuwerfen. Du willst **Daten wiederverwenden**.

**DQN hat das für diskrete Aktionen gelöst** mit einem Replay-Buffer: speichere Millionen vergangener Transitionen und sample Minibatches gleichverteilt. Q-Learning ist konstruktionsbedingt off-policy — die Bellman-Gleichung interessiert nicht, welche Policy die Daten erzeugt hat. Aber DQN kann mit kontinuierlichen Aktionsräumen nicht umgehen, weil die `max_a Q(s,a)`-Operation undefiniert ist, wenn Aktionen reellwertige Vektoren sind.

Die offene Frage ist also:

> Können wir **kontinuierliche Aktionen** + **einen Replay-Buffer** + **stabiles Training** haben?

Die Antwort, seit 2018, ist ja. Der Algorithmus ist **SAC — Soft Actor-Critic** (Haarnoja et al., 2018). Abschnitt 9 verfolgt die DDPG → TD3 → SAC-Linie, die ihn hervorgebracht hat.

---

## 1 · Maximum-Entropy-RL — das Framework, auf dem SAC aufbaut

Standard-RL hat ein Objektiv:

```
J(π) = E_π [ Σ_t γ^t · r_t ]
```

Maximieren des erwarteten diskontierten Returns. Nichts weiter.

**Maximum-Entropy-RL fügt einen zweiten Term hinzu** — die Entropie der Policy in jedem besuchten Zustand:

```
J(π) = E_π [ Σ_t γ^t · ( r_t + α · H(π(·|s_t)) ) ]
```

wobei:

- `H(π(·|s)) = -Σ_a π(a|s) · log π(a|s)` die **Policy-Entropie** im Zustand `s` ist — ein Maß dafür, wie zufällig die Aktionsverteilung ist
- `α` die **Temperatur** ist — ein Skalar, der steuert, wie stark wir Entropie gegenüber Belohnung gewichten
- Die Summation `Σ_t γ^t · ...` dieselbe diskontierte Summe wie in Standard-RL ist

Im Klartext wird der Agent gleichzeitig für zwei Dinge belohnt:

1. **Hohe Belohnung erzielen** (das ursprüngliche RL-Objektiv)
2. **So zufällig wie möglich bleiben** dabei

Das klingt paradox, bis du überlegst, was es tatsächlich macht:

- **Natürliche Erkundung.** Eine hochentropische Policy probiert weiter Alternativen, statt sich früh festzulegen. Vergleiche das mit PPOs `ent_coef` — einem Entropie-*Bonus*, der oben auf den Loss draufgesetzt ist. In SAC ist Entropie Teil des *Objektivs selbst*, nicht ein zusätzlicher Regularisierer.
- **Vermeidet vorzeitige Konvergenz.** Der klassische Fehlermodus in RL ist, sich auf eine suboptimale Strategie festzulegen, weil sie früh auszahlt. Maximum Entropy sagt: lege dich nicht fest, bis du *wirklich* sicher bist.
- **Findet mehrere Modi.** Wenn zwei Aktionssequenzen beide hohe Belohnung erzielen, weist eine Max-Entropy-Policy *beiden* Wahrscheinlichkeit zu, statt zu einer zu kollabieren. Das ist in der Robotik wichtig — oft gibt es mehrere gleich gute Wege, ein Objekt zu greifen.
- **Robust gegen Störungen.** Eine zufallsbehaftete Policy degradiert sanft, wenn sich die Umgebung leicht ändert; eine deterministische Policy kann katastrophal versagen.

Die Temperatur `α` ist der Regler zwischen diesen zwei Welten. `α = 0` kollabiert zu Standard-RL (deterministisches Optimum). `α = ∞` ignoriert die Belohnung komplett (gleichverteilte Zufallspolicy). SAC operiert irgendwo dazwischen — und, entscheidend, **tunt α automatisch**. Darauf kommen wir zurück.

---

## 2 · SAC-Architektur — drei Netze (plus Targets)

SAC hält fünf Netze gleichzeitig im Speicher. Das ist mehr als PPO, aber jedes hat eine klare Rolle.

**Actor `π_φ(a|s)`** — die Policy

- Input: Beobachtung `s`
- Output: Parameter einer Gauß-Verteilung über Aktionen — ein **Mittelwert `μ_φ(s)`** und eine **logarithmische Standardabweichung `log σ_φ(s)`**
- Um eine Aktion zu samplen: zieh `ε ~ N(0, 1)` und berechne `a = μ_φ(s) + σ_φ(s) · ε` (der **Reparameterisierungs-Trick (reparameterization trick)** — siehe §7)
- In der Praxis wird ein `tanh`-Squash angewandt, damit Aktionen in `[-1, 1]` bleiben

**Critic 1 `Q_θ1(s, a)`** — erstes Q-Wert-Netz

- Input: Beobachtung *und* Aktion konkateniert (oder als zwei Heads eingespeist)
- Output: ein einzelner Skalar — der geschätzte Q-Wert

**Critic 2 `Q_θ2(s, a)`** — zweites, unabhängiges Q-Wert-Netz

- Identische Architektur zu Critic 1, aber mit komplett unabhängigen Gewichten `θ2`
- Auf dasselbe Target wie Critic 1 trainiert
- Der **Twin-Critics-Trick**: bei der Berechnung des Actor-Updates oder des Bellman-Targets nimm `min(Q_θ1, Q_θ2)`
- Warum: ein einzelnes Q-Netz *überschätzt* Werte systematisch wegen der max-artigen Operation im Bellman-Backup (dasselbe Überschätzungsproblem, das DQN mit Target-Netzen bekämpft). Das Minimum aus zwei unabhängigen Schätzungen zu nehmen ist eine billige, effektive Methode, diesen Bias zu reduzieren. Die Idee kommt von TD3 (Twin Delayed DDPG), und SAC erbt sie.

**Target-Critics `Q̂_θ1`, `Q̂_θ2`** — langsam bewegte Kopien der zwei Critics

- Werden nur auf der *rechten Seite* der Bellman-Gleichung genutzt (bei der Berechnung von `y`, dem Target)
- Aktualisiert per **Soft Update** in jedem Gradientenschritt: `θ̂ ← τ · θ + (1 - τ) · θ̂`, wobei `τ = 0,005` winzig ist
- Ohne Target-Netze jagt der Critic ein bewegtes Ziel und das Training divergiert — dieselbe Lektion, die uns DQN beigebracht hat

**Replay-Buffer**

- FIFO-Queue von bis zu 1.000.000 vergangenen Transitionen `(s, a, r, s', done)`
- Jeder Gradientenschritt sampelt einen Minibatch (typisch 256) gleichverteilt zufällig
- Der Replay-Buffer ist das, was SAC *off-policy* macht: Transitionen bleiben gültig, obwohl die Policy, die sie erzeugt hat, längst veraltet ist, weil das Bellman-Update nicht von der datenerzeugenden Policy abhängt

**Diagramm eines Forward-Passes:**

```
Beobachtung s
    │
    ▼
Actor π_φ ──► (μ, log σ) ──► sample a (Reparameterisierung)
                                  │
                                  ▼
                       Critic 1 Q_θ1(s, a) ─┐
                                            ├─► min ─► Q̂(s, a)   ← Target für Actor-Update
                       Critic 2 Q_θ2(s, a) ─┘

Target-Critics Q̂_θ1, Q̂_θ2 (Soft Update von Q_θ1, Q_θ2 in jedem Schritt, τ = 0,005)
```

---

## 3 · SAC-Update-Gleichungen — ein Gradientenschritt, Term für Term

Jeden Schritt sampelt SAC einen Minibatch `{(s, a, r, s', done)}` aus dem Replay-Buffer und führt drei Updates aus: Critic, Actor, Temperatur. Wir gehen jedes sorgfältig durch.

### 3.1 · Critic-Update — den (weichen) Bellman-Fehler minimieren

Der Zielwert `y` für jede Transition ist:

```
y = r + γ · ( min( Q̂_θ1(s', ã'),  Q̂_θ2(s', ã') ) - α · log π_φ(ã' | s') )

    wobei  ã' ~ π_φ( · | s' )      (aus der *aktuellen* Policy bei s' gesampelt)
```

Term für Term:

- `r` — sofortige Belohnung, aus dem Buffer
- `γ` — Diskontierungsfaktor (typisch 0,99)
- `s'` — nächster Zustand, aus dem Buffer
- `ã'` — eine frische Aktion, aus der **aktuellen** Policy bei `s'` gesampelt. Das ist der Schlüsselgrund, warum SAC off-policy ist: wir brauchen nicht die Aktion, die ursprünglich bei `s'` genommen wurde — wir fragen die aktuelle Policy, was *sie* tun würde
- `min( Q̂_θ1, Q̂_θ2 )` — der Twin-Critic-Trick: nimm die *niedrigere* der zwei Target-Critic-Schätzungen, um Überschätzungs-Bias zu reduzieren
- `- α · log π_φ(ã' | s')` — die **Maximum-Entropy-Korrektur**. Das subtrahiert die Log-Wahrscheinlichkeit der gewählten Aktion, gewichtet mit der Temperatur. Weil `-log π` ein unverzerrter Schätzer der Entropie `H(π)` ist, addiert dieser Term effektiv den Zukunfts-Entropie-Beitrag zum Bellman-Target. Das ist es, was es zu einer *weichen* Bellman-Gleichung macht

Die zwei Critics werden trainiert, indem man den Mean-Squared-Bellman-Fehler minimiert:

```
L_critic = MSE( Q_θ1(s, a) - y )  +  MSE( Q_θ2(s, a) - y )
```

Beachte, dass `(s, a)` aus dem Buffer kommen (die Aktion, die tatsächlich genommen wurde), während `ã'` frisch gesampelt wird. Diese Trennung ist genau das, was SAC erlaubt, alte Daten wiederzuverwenden.

### 3.2 · Actor-Update — Q maximieren bei maximaler Entropie

Das Actor-Objektiv: Aktionen wählen, die hohen Q-Wert liefern *und* die Policy stochastisch halten.

```
L_actor = E_{s ~ buffer,  a ~ π_φ(·|s)}  [  α · log π_φ(a | s)  -  min( Q_θ1(s, a),  Q_θ2(s, a) )  ]
```

Wir *minimieren* `L_actor`, was äquivalent zum *Maximieren* von `Q - α · log π` ist.

- `α · log π_φ(a | s)` — bestraft zu hohes Vertrauen (niedrige Entropie). Diesen Term zu minimieren *erhöht* Entropie
- `- min(Q_θ1, Q_θ2)` — negatives Q heißt Q maximieren
- Der Erwartungswert `a ~ π_φ(·|s)` wird mit dem Reparameterisierungs-Trick berechnet, sodass Gradienten von `Q(s, a)` zurück durch `a` in die Actor-Gewichte `φ` fließen. Ohne Reparameterisierung wäre dieser Gradient durch den Sampling-Schritt blockiert

Der Actor sieht die *Aktionen* des Replay-Buffers nie direkt — er sieht nur die *Zustände* und würfelt seine eigenen frischen Aktionen durch den reparameterisierten Sampler.

### 3.3 · Temperatur-Update — automatisches Entropie-Tuning

`α` selbst ist ein lernbarer Parameter, trainiert, um die Policy-Entropie nahe an einem Zielwert `H_target` zu halten:

```
L_α = E_{a ~ π_φ}  [  -α  ·  ( log π_φ(a | s)  +  H_target )  ]

    wobei  H_target = -dim(action_space)    (ein üblicher Default)
```

Intuition:

- Wenn die aktuelle Entropie `H(π) = -E[log π]` **über** `H_target` liegt → ist der Term `(log π + H_target)` im Mittel **negativ** → Gradient *senkt* `α` → weniger Entropie-Druck
- Wenn die aktuelle Entropie **unter** `H_target` liegt → Gradient *erhöht* `α` → mehr Entropie-Druck
- Im Gleichgewicht schwebt die Entropie um `H_target` und `α` pendelt sich auf den Wert ein, der sie hält

Der Default `H_target = -dim(action_space)` ist heuristisch, funktioniert aber über eine riesige Bandbreite an Aufgaben hinweg. Für einen 4-dimensionalen kontinuierlichen Aktionsraum ist `H_target = -4`. Es skaliert natürlich: mehr Aktionsdimensionen → mehr Entropie nötig, um vielfältig zu bleiben.

---

## 4 · Hands-on — SAC mit Stable-Baselines3

Die gute Nachricht: SB3 macht SAC zu einem Drop-in-Ersatz für PPO. Du änderst einen Import und ein paar Hyperparameter.

```python
from stable_baselines3 import SAC
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

# SAC on a continuous Godot environment (FlyBy from Unit 6)
env = StableBaselinesGodotEnv(
    env_path="./FlyBy.x86_64",
    n_parallel=1,        # SAC is off-policy: no benefit from many envs
    speedup=20,
)

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/",
    learning_rate=3e-4,
    buffer_size=1_000_000,    # replay buffer size
    learning_starts=10_000,   # collect transitions before training starts
    batch_size=256,
    tau=0.005,                # soft update coefficient
    gamma=0.99,
    train_freq=1,             # update every env step
    gradient_steps=1,         # one gradient step per env step
    ent_coef="auto",          # automatic entropy tuning (α is learned)
    target_entropy="auto",    # H_target = -dim(action_space)
)

model.learn(total_timesteps=1_000_000, tb_log_name="sac_flyby")
model.save("flyby_sac")
env.close()
```

Ein paar Notizen zu den Hyperparametern:

| Parameter | Warum dieser Wert |
|-----------|---------------|
| `learning_rate=3e-4` | Standard für SAC, identisch zum PPO-Default. Adam macht den Rest |
| `buffer_size=1_000_000` | Eine Million Transitionen. Kleiner (100k) ist für kurze Aufgaben okay; größer verschwendet Speicher |
| `learning_starts=10_000` | Sammle 10k Transitionen mit Zufalls-Policy vor jedem Gradientenschritt. Vermeidet frühe Schrott-Updates |
| `batch_size=256` | Minibatch-Größe für jeden Gradientenschritt. 256 ist der universelle SAC-Default |
| `tau=0.005` | Soft-Update-Rate für Target-Netze. Kleiner = stabiler, langsamer im Nachziehen |
| `train_freq=1`, `gradient_steps=1` | Ein Env-Schritt → ein Gradientenschritt. Du kannst mehr Gradientenschritte pro Env-Schritt machen (z. B. `gradient_steps=4`), wenn Datensammlung langsam relativ zum GPU-Training ist — typisch in der Robotik |
| `ent_coef="auto"` | Aktiviert das automatische α-Tuning aus §3.3 |
| `target_entropy="auto"` | Setzt `H_target = -dim(action_space)` automatisch |

**TensorBoard-Signale zum Beobachten:**

| Signal | Bedeutung |
|--------|--------------|
| `train/ent_coef` | Der aktuelle Wert von α. Sollte sich bei einem positiven Wert stabilisieren, nicht auf 0 kollabieren (heißt Policy ging zu früh deterministisch) und nicht riesig bleiben (heißt Lernen macht die Policy gar nicht zuversichtlicher) |
| `train/actor_loss` | Sollte negativ sein (wir maximieren Q minus Entropie-Strafe) und sich langsam verbessern, wenn die Q-Werte steigen |
| `train/critic_loss` | Mean-Squared-Bellman-Fehler. Sollte sinken, wenn sich die Q-Schätzungen stabilisieren. Frühe Spikes sind normal; anhaltend hohe Werte heißen, dass etwas falsch ist |
| `train/ent_coef_loss` | Der Loss für das Temperatur-Update. Sollte im Gleichgewicht nahe null schweben |
| `rollout/ep_rew_mean` | Die Haupt-Performance-Metrik — wie bei PPO |

!!! warning "Erwarte keine Parallel-Speedups"
    SACs Update ist pro Umgebungsschritt. Mehr parallele Envs erhöhen Datensammlung, aber **nicht** proportional die Gradientenschritte. Setze `n_parallel=1` (oder 2–4, wenn deine Sim sehr schnell ist) und lass den Algorithmus arbeiten.

---

## 5 · PPO vs. SAC — die Entscheidungs-Übersicht

Das ist die Tabelle, zu der du zurückkehren solltest, wann immer du ein neues Projekt startest.

| Aspekt | PPO | SAC |
|--------|-----|-----|
| On-/Off-Policy | On-Policy (nach Gebrauch verwerfen) | Off-Policy (Replay-Buffer) |
| Aktionsraum | Diskret + kontinuierlich | Kontinuierlich (Vanilla); diskrete Varianten existieren (sb3_contrib DiscreteSAC, CleanRL sac_atari) |
| Replay-Buffer | Nein | Ja (typisch 1M Transitionen) |
| Parallele Envs | Ja — skaliert nahezu linear | Minimaler Nutzen |
| Stichprobeneffizienz | Niedriger — verwirft Daten | Höher — verwendet Transitionen wieder |
| Hyperparameter-Sensitivität | Mittel (clip range, n_steps, lr) | Niedrig — automatisches α-Tuning erledigt das meiste |
| Stabilität | Hoch (geclipptes Objektiv) | Hoch (Twin-Critics + Target-Netze) |
| Wall-Clock-Geschwindigkeit | Schnell mit vielen Envs | Langsamer pro Schritt, aber weniger Schritte nötig |
| Speicher-Footprint | Klein (kein Buffer) | Groß (1M Transitionen × Obs-Dim) |
| Am besten für | Spielartige Envs, schnelle Simulatoren, parallele Rollouts | Robotik, teure Sims, präzise kontinuierliche Steuerung |

!!! tip "Wann was wählen — die Faustregel"
    **Nutze PPO, wenn** dein Aktionsraum irgendwelche diskreten Aktionen enthält, deine Umgebung billig genug ist für 8+ parallele Kopien, und du in einer typischen Godot-Spielumgebung arbeitest.

    **Nutze SAC, wenn** der Aktionsraum rein kontinuierlich ist, die Simulation teuer ist (ein echter Roboter, eine schwere Physik-Sim, alles, das Sekunden pro Schritt braucht), Datensammlung dein Flaschenhals ist und du maximale Stichprobeneffizienz willst.

!!! warning "Vanilla SAC ist für kontinuierliche Aktionsräume entworfen"
    SB3s `SAC`-Klasse nimmt rein kontinuierliche Aktionen an. Wenn dein `AIController` irgendwelche diskreten Buttons (jump, fire, grab) neben kontinuierlichen Steuerungen exponiert, funktioniert Vanilla SAC nicht out of the box. Diskrete Varianten existieren — `sb3_contrib.DiscreteSAC` und CleanRLs `sac_atari.py` — aber für gemischte Aktionsräume ist die einfachste Wahl PPO oder `RecurrentPPO`.

---

## 6 · Der Reparameterisierungs-Trick — warum SACs Actor-Update funktioniert

Das Actor-Update in §3.2 enthält diesen Erwartungswert:

```
E_{a ~ π_φ(·|s)} [ Q(s, a) - α · log π_φ(a | s) ]
```

Wir wollen Gradienten dieses Erwartungswerts bezüglich `φ`. Es gibt ein tiefes Problem: die Aktion `a` wird aus einer von `φ` parametrisierten Verteilung *gesampelt*, und **Sampling ist nicht differenzierbar**. Das Gradientensignal von `Q(s, a)` kann nicht durch einen Sample-Schritt zurück in `μ_φ` und `σ_φ` fließen.

**Der Reparameterisierungs-Trick** löst das, indem er die Zufälligkeit *außerhalb* des Netzes bewegt. Statt `a` direkt aus `N(μ_φ(s), σ_φ(s)²)` zu samplen, samplen wir eine *Standard*-Gauß-Rauschvariable unabhängig und kombinieren sie:

```
ε ~ N(0, 1)              ← unabhängig gesampelt, keine Parameter
a = μ_φ(s) + σ_φ(s) · ε  ← deterministische Funktion von φ und ε
```

Jetzt ist `a` eine *deterministische* Funktion von `φ` (und dem zufälligen Rauschen `ε`, das nichts mit `φ` zu tun hat). Gradienten fließen sauber:

```
∂a/∂φ = ∂μ_φ/∂φ  +  ε · ∂σ_φ/∂φ
```

Dieser Gradient wird mit `∂Q/∂a` multipliziert, um den vollen Actor-Gradienten zu ergeben. Die Zufälligkeit bleibt erhalten (verschiedene `ε`-Samples geben verschiedene `a`), aber der Pfad von `φ` zu `a` ist vollständig differenzierbar.

**Vergleiche mit PPO.** PPO umgeht dieses ganze Thema mit dem Importance-Sampling-Verhältnis `r_t = π_new(a|s) / π_old(a|s)`. PPO differenziert nie *durch* das Aktions-Sample — es gewichtet *bereits gesammelte* Aktionen einfach mit dem Policy-Verhältnis neu. Der Grund ist nicht einfach, dass PPO on-policy ist (REINFORCE ist auch on-policy, nutzt aber den Score-Funktion-Schätzer). Es liegt daran, dass PPOs Update um das IS-Verhältnis herum gebaut ist, das nur `log π(a|s)`, ausgewertet an gespeicherten Aktionen, braucht — eine Ableitung durch die *Parameter*, nicht durch das Sample. SACs Actor-Objektiv ist `E_{a ~ π}[Q(s, a)]`, also muss der Gradient durch die Aktion `a` selbst fließen; Reparameterisierung macht diesen Pfad differenzierbar.

Das ist einer der tiefen mathematischen Gründe, warum SAC und PPO so verschieden aussehen, obwohl beide Actor-Critic-Methoden sind.

---

## 7 · Automatisches Entropie-Tuning — was `ent_coef="auto"` für dich erledigt

`α` von Hand zu tunen ist schmerzhaft:

- Zu hoch → der Entropie-Term dominiert → die Policy bleibt für immer zufällig und konvergiert nie
- Zu niedrig → Entropie kollabiert → die Policy wird deterministisch → keine Erkundung → festsitzen in einem lokalen Optimum
- Der richtige Wert hängt von Belohnungsskala, Aktionsdimension und davon ab, wie weit das Training fortgeschritten ist

Schlimmer noch, der *ideale* `α` ändert sich während des Trainings. Früh willst du viel Entropie (Erkundung); später weniger (Ausnutzung).

**Automatisches Entropie-Tuning** (Haarnoja et al. 2018, Follow-up-Paper) behandelt `α` als lernbaren Parameter und passt ihn online an, um eine *Ziel-Entropie* `H_target` aufrechtzuerhalten:

- Du setzt `H_target` einmal. Der Default `-dim(action_space)` ist eine Heuristik, die über die meisten Aufgaben hinweg funktioniert.
- Jeden Gradientenschritt wird der Temperatur-Loss `L_α` aus §3.3 minimiert.
- Wenn die Policy zu deterministisch wird (Entropie unter dem Ziel), steigt `α` automatisch und erhöht den Entropie-Bonus.
- Wenn die Policy zu zufällig ist (Entropie über dem Ziel), fällt `α` und lässt die Policy konvergieren.

**Praktischer Effekt auf eine Trainingskurve:**

- Frühe Schritte: `α` ist hoch (sagen wir 0,5–1,0) — die Policy wird ermutigt, breit zu erkunden
- Wenn sich die Q-Schätzungen stabilisieren: `α` driftet nach unten (oft auf 0,05–0,2) — die Policy beginnt auszunutzen
- Schließlich: `α` pendelt sich auf den Wert ein, der `H(π) ≈ H_target` für die finale Policy aufrechterhält

In SB3 ist all das durch `ent_coef="auto"` und `target_entropy="auto"` aktiviert. Du wirst `train/ent_coef` in TensorBoard sich entwickeln sehen — es sollte wie eine glatte, langsam abklingende Kurve aussehen, nicht wie eine Stufenfunktion oder ein Kollaps.

!!! tip "Was, wenn `ent_coef` auf 0 kollabiert?"
    Heißt, die Policy ist weit unter `H_target` vollständig deterministisch geworden, und der Optimizer zieht `α` verzweifelt nach unten. Meist ein Zeichen, dass das Belohnungssignal die Entropie überwältigt — probier, `target_entropy` negativer zu setzen (z. B. `-2 * dim(action_space)`), um mehr Stochastizität zu verlangen.

---

## 8 · SAC im Godot-Workflow

Der ganze Sinn dieses Kurses ist, dass du in Python trainierst und in Godot deployst. SAC passt mit einer kleinen Änderung in diese Pipeline.

**Training** — identisch zu PPO, tausch einfach die Algorithmusklasse:

```python
# Old (PPO)
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, ...)

# New (SAC)
from stable_baselines3 import SAC
model = SAC("MlpPolicy", env, ...)
```

**ONNX-Export für Inferenz** — exportiere nur den *Actor*. Die Critics existieren rein fürs Training; beim Deployment brauchst du nur `π_φ(a|s)`.

```python
import torch

obs_dim = env.observation_space.shape[0]
dummy_obs = torch.zeros(1, obs_dim)

# SB3's SAC actor is exposed as model.policy.actor
torch.onnx.export(
    model.policy.actor,
    dummy_obs,
    "flyby_sac.onnx",
    input_names=["obs"],
    output_names=["action"],
    opset_version=11,
)
```

**Godot-Seite** — exakt dasselbe wie deine PPO-Units. Setze den Sync-Node auf `ONNX_INFERENCE`, zeige auf `flyby_sac.onnx`, und der Agent läuft deterministisch im Spiel.

!!! warning "Parallele Envs und SAC in Godot"
    Wenn du den `Sync`-Node in deinem Godot-Projekt für SAC-Training einrichtest, **setze `n_parallel=1`** im Wrapper und in jedem Launcher-Skript. Mehrere parallele Envs mit SAC laufen zu lassen, verschwendet CPU und verbessert die Wall-Clock-Lerngeschwindigkeit kaum, weil SACs Flaschenhals Gradientenschritte sind, nicht Datensammlung.

---

## 9 · Die DDPG → TD3 → SAC-Linie

Zu verstehen, woher SACs Architektur kommt, macht sie weniger geheimnisvoll. Drei Algorithmen, jeder behebt das Hauptproblem des vorherigen.

**DDPG (Deep Deterministic Policy Gradient):** Off-Policy-Actor-Critic für kontinuierliche Aktionen. Der Actor gibt eine einzelne deterministische Aktion `π(s) → a` aus (keine Verteilung). Nutzt einen Replay-Buffer — der erste Continuous-Control-Algorithmus, der das effektiv tat. Problem: überschätzt Q-Werte (einzelner Critic), spröde gegenüber Hyperparametern, Erkundung erfordert manuell injiziertes Rauschen.

**TD3 (Twin Delayed DDPG, Fujimoto et al. 2018)** — drei Fixes:

1. **Twin-Critics:** trainiere zwei Q-Netze unabhängig, nimm das Minimum für die Target-Berechnung → verhindert Q-Wert-Überschätzung
2. **Verzögerte Policy-Updates:** aktualisiere den Actor alle 2 Critic-Schritte → reduziert Varianz im Actor-Gradienten, wenn Critic-Schätzungen noch verrauscht sind
3. **Target-Policy-Smoothing:** addiere kleines Gauß-Rauschen zur Target-Aktion während Critic-Updates → regularisiert die Q-Funktion, verhindert dass sie scharfe Spitzen ausnutzt

**SAC** nimmt den Twin-Critic-Trick von TD3 und fügt Maximum-Entropy-RL (Abschnitt 1) hinzu, um manuelles Rauschen-Injizieren durch prinzipielle stochastische Erkundung zu ersetzen.

| | DDPG | TD3 | SAC |
|--|------|-----|-----|
| Policy | Deterministisch | Deterministisch | Stochastisch |
| Erkundung | Rauschen-Injektion | Rauschen-Injektion | Entropie-Maximierung |
| Twin-Critics | Nein | Ja | Ja |
| Verzögerte Actor-Updates | Nein | Ja | Nein (nicht nötig) |
| Stichprobeneffizienz | Mittel | Hoch | Am höchsten |
| Hyperparameter-Sensitivität | Hoch | Mittel | Niedrig (Auto-α) |

**Wann TD3 statt SAC:** wenn du eine deterministische Policy bei Inferenz brauchst (manche Deployments auf echten Robotern erfordern wiederholbare Aktionen); wenn die Stochastizität von SAC Probleme in einer spezifischen Umgebung macht. SB3 unterstützt TD3 direkt:

```python
from stable_baselines3 import TD3
model = TD3("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

---

## 10 · Stretch Goals

Probier diese, um dein Verständnis zu vertiefen:

1. **Vergleiche SAC vs. PPO Stichprobeneffizienz auf FlyBy.** Trainiere beide Algorithmen 1M Schritte mit derselben Umgebung. Plotte `rollout/ep_rew_mean` gegen Umgebungsschritte für jeden. SAC sollte dasselbe Belohnungslevel mit deutlich weniger Transitionen erreichen — das ist das Schlüsselergebnis.

2. **Tunne `ent_coef` manuell.** Lauf SAC mit `ent_coef=0,1`, `ent_coef=0,01`, `ent_coef=0,001` und `ent_coef="auto"`. Plotte `train/ent_coef` und finale Belohnung für jeden. Die auto-getunte Kurve sollte jeden handgewählten Wert übertreffen — was bestätigt, dass automatisches Tuning sein Geld wert ist.

3. **SAC auf HovercraftRacing.** Eine schwierigere Continuous-Control-Aufgabe. SAC braucht länger (2–3M Schritte), produziert aber oft merklich glattere Steuerung als PPO — schau dir die Viz-Checkpoints nebeneinander an und sieh, ob du zustimmst.

4. **Twin-Critic-Ablation (fortgeschritten).** SB3 exponiert kein Single-Critic-SAC out of the box, aber du kannst die SAC-Klasse forken und die `min`-Operation deaktivieren. Trainiere beide Versionen und beobachte Q-Wert-Drift — die Single-Critic-Version sollte Q signifikant überschätzen, und die finale Policy-Performance sollte leiden.

5. **Replay-Buffer-Größen-Sweep.** Probier `buffer_size=10_000`, `100_000`, `1_000_000`. Kleinere Buffer zwingen den Algorithmus, frühere Transitionen zu „vergessen"; größere halten veraltete Daten herum. Finde den Sweet Spot für deine Umgebung.

---

## Was kommt als Nächstes

Du hast jetzt zwei Werkzeuge im Koffer: PPO (dein Arbeitspferd für Spielumgebungen) und SAC (dein Spezialist für kontinuierliche Steuerung). Die nächste Unit geht zurück zu PPO und stellt eine andere Frage — wie skalieren wir Datensammlung, indem wir viele Umgebungen parallel laufen lassen?

**Unit 5: Paralleles Training** deckt `n_parallel` ab, wie Rollouts über Envs zusammengenäht werden, und wie man über Durchsatz vs. Stichprobeneffizienz nachdenkt. Beachte, dass Parallelismus PPO substantiell hilft (lineare Skalierung bis zu Dutzenden Envs), SAC aber kaum hilft — eine weitere Illustration der On-Policy- / Off-Policy-Trennung.

Willst du tiefer in PPOs Code-Funktionsweise eintauchen? **PPO von Grund auf (CleanRL)** geht jede Zeile einer Single-File-PPO-Implementierung durch, bevor du skalierst.

!!! info "Selbstcheck, bevor du weitermachst"
    Kannst du diese Fragen in eigenen Worten beantworten?

    1. Warum hält SAC einen **Replay-Buffer**, während PPO das nicht tut — und was bringt dir das an Stichprobeneffizienz?
    2. Was fügt der **Maximum-Entropy**-Term dem Standard-RL-Objektiv hinzu, und welches Verhalten fördert er?
    3. Warum nutzt SAC **zwei Q-Netze** und nimmt deren `min`? Welche Pathologie behebt das?
    4. Was justiert `ent_coef="auto"`, und gegen welches Ziel?
    5. Was würde dich auf einer kontinuierlichen Godot-Env (z. B. FlyBy) zu SAC statt PPO drücken, und was zurück zu PPO?

    Wenn du alle fünf beantworten kannst — du kannst auf einer neuen Aufgabe zwischen PPO und SAC wählen, ohne zu raten.

[→ Anwenden — SAC vs PPO auf JumperHard](unit-sac-applied.md) · [→ Paralleles Training](unit-05.md)
