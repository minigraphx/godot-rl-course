# PPO Deep Dive — Geclipptes Objektiv, GAE und Hyperparameter

Das ist der Schlussstein der Theoriereihe. Du hast den langen Weg schon gegangen: **Q-Learning** lehrte dich die Bellman-Gleichung und Wert-Bootstrapping; **DQN** skalierte Werte auf neuronale Netze; **REINFORCE** kippte um zu Policy Gradients; **Actor-Critic / A2C** klebte beides mit einer Baseline zusammen. PPO ist der Algorithmus, der all diese Ideen nimmt, A2Cs zwei große Schwächen behebt und am Ende als das Arbeitspferd des modernen angewandten RL dasteht — der Algorithmus, den Godot RL Agents standardmäßig verwendet, den OpenAI für Dota 2 nutzte, den du für jedes Projekt in diesem Kurs einsetzen wirst.

Diese Unit zeigt dir nicht, wie man PPO *benutzt* — das ist die nächste. Diese Unit erklärt, **warum jede Zeile des PPO-Pseudocodes so ist, wie sie ist**.

[← Actor-Critic](unit-actor-critic.md) · [Kursstartseite](index.md)

!!! note "Voraussetzungen"
    - **[Policy Gradients](unit-policy-gradients.md)** — Log-Derivative-Trick, Baselines
    - **[Actor-Critic](unit-actor-critic.md)** — A2C-Loss; du musst spüren, warum A2C *fast* genug ist
    - **[Unit 4](unit-04.md)** (optional) — PPO einmal getunt zu haben lässt §8 und §9 härter landen
    - Komfortabel im Lesen von PyTorch (CleanRLs `ppo.py` ist die Referenz in §10)

!!! info "Zeit"
    Lesen: ~45 min

---

!!! info "Drei Wege, deine KI zu beobachten"
    TensorBoard (`train/approx_kl`, `rollout/ep_rew_mean`, `train/entropy_loss`) · CleanRL `ppo.py` im Editor geöffnet · Ein Godot-Agent, der mit genau den Hyperparametern trainiert, die du in dieser Unit getunt hast

---

## Warum es diese Unit gibt

Jedes Mal, wenn du bisher `gdrl` ausgeführt hast, war PPO der Algorithmus, der die Arbeit leistete. Du sahst `clip_range`, `gae_lambda`, `n_epochs`, `vf_coef` in den Logs vorbeiscrollen und hast den Defaults vertraut. Dieses Vertrauen endet hier. Am Ende dieser Unit solltest du folgendes können:

- Das Original-PPO-Paper (Schulman et al. 2017) lesen, ohne Gleichungen zu überspringen.
- CleanRLs Single-File-`ppo.py` öffnen und jeden Block wiedererkennen.
- Eine TensorBoard-Kurve ansehen, das Problem benennen und den richtigen Hyperparameter zum Drehen auswählen.
- Einem anderen Studenten in einfachen Worten erklären, warum die `min` und `clip` im PPO-Objektiv da sind.

Theorie-Units in diesem Kurs paaren stets eine Gleichung mit einer Geschichte. PPO hat viele Gleichungen — aber nur eine *Idee*: **lass die Policy sich so stark verbessern wie möglich, aber niemals so stark, dass die Daten, mit denen wir die Verbesserung berechnet haben, irrelevant werden**.

---

## 1 · Das A2C-Problem, das PPO löst

Erinnern wir uns zuerst daran, was A2C macht und wo es schmerzt.

**A2C in einer Schleife:**

```
loop:
    collect n_steps transitions using current policy π_θ
    compute advantages A_t = G_t - V(s_t)
    compute one gradient step:
        L = -log π_θ(a_t|s_t) · A_t  +  c1 · (V_θ(s_t) - G_t)²  -  c2 · H(π_θ)
    update θ via Adam
    throw away the rollout
```

Dieses `throw away the rollout` ist schmerzhaft. Wir haben, sagen wir, 2048 Transitionen über 8 parallele Umgebungen gesammelt — 16384 (state, action, reward)-Tupel. Wir haben *einen* Gradientenschritt damit gemacht, sie dann auf den Boden geworfen und sind zurück zum Simulator. In einer langsamen Umgebung wie Godot (jeder Schritt erfordert, dass die Engine tickt, rendert und über die WebSocket-Brücke hin- und zurückläuft) ist die Datensammlung mit Abstand der Flaschenhals. Jede Transition nur einmal zu verwenden ist Verschwendung.

**Der naheliegende Fix:** *viele* Gradientenschritte pro Rollout nehmen. Loope 10 Mal über dieselben Daten. Erhalte 10× die Policy-Verbesserung pro Umgebungsschritt. Erledigt, oder?

**Nein.** Und hier beginnen die Probleme.

### Das Instabilitätsproblem

Nach dem *ersten* Gradientenschritt ist die Policy nicht mehr π_θ — sie ist π_θ' (leicht anders). Die Vorteile, die wir berechnet haben, A_t, wurden unter der *alten* Policy berechnet. Sie sind Schätzungen von *„wie viel besser als der Durchschnitt ist Aktion a_t, unter der Annahme, dass wir uns wie π_θ verhalten?"* — nicht π_θ'.

Nach dem *zweiten* Gradientenschritt ist die Policy π_θ''. Die Vorteile sind jetzt noch falscher.

Nach zehn Schritten kann die Policy sehr verschieden von der sein, die die Daten gesammelt hat. Die Vorteile, mit denen wir weiterhin Gradienten-Updates antreiben, sind jetzt Schätzungen aus einer *völlig anderen Policy*. Wir fliegen im Wesentlichen blind und nehmen selbstbewusste Updates auf Basis veralteter Information. In der Praxis führt das zu einem von zwei Fehlermodi:

1. **Katastrophaler Kollaps:** die Policy divergiert in eine entartete Verteilung (nimmt immer dieselbe Aktion), die Entropie kollabiert, die Episodenbelohnung crasht auf Baseline.
2. **Oszillation:** die Policy ping-pongt zwischen zwei Regionen, ohne sich je zu setzen. Loss sieht verrauscht aus, Belohnung verbessert sich nie.

Das ist der zentrale Konflikt von Off-Policy-artigem Lernen:

| Wunsch | Risiko |
|--------|------|
| Daten mehrfach wiederverwenden (Stichprobeneffizienz) | Policy driftet von datensammelnder Policy weg (Vorteile werden falsch) |
| Kleine, sichere Schritte machen (Stabilität) | Selten updaten, Daten verschwenden (Stichproben-Ineffizienz) |

**PPO ist die Antwort auf genau diesen Trade-off.** Es lässt dich viele Gradientenschritte auf denselben Daten machen, *stoppt* das Update für eine einzelne Transition aber clever, sobald die Policy „zu weit" von der weggedriftet ist, die sie gesammelt hat. Jede Transition trägt nur bei, solange sie noch relevant ist.

---

## 2 · Trust-Region-Intuition

Das mentale Modell, das PPO erschließt, ist die **Trust Region (Vertrauensregion)**.

> Stell dir vor, du wanderst im dichten Nebel am Rand einer Klippe. Du siehst ein, zwei Meter um dich herum (die lokale Belohnungslandschaft). Du würdest gerne einen riesigen Sprung Richtung höheres Gelände machen — aber ein riesiger Sprung riskiert, von der Klippe zu fallen. Also verpflichtest du dich nur zu kleinen Schritten, innerhalb einer Region, der du *vertraust*.

In Policy-Optimierungssprache: die Daten, die du gesammelt hast, geben dir eine gute Schätzung, welche Aktionen besser als der Durchschnitt sind **nahe der aktuellen Policy**. Sobald die neue Policy zu verschieden von der datensammelnden Policy ist, sind diese Schätzungen nicht mehr verlässlich. Die „Trust Region" ist die Menge der Policies, die der alten nahe genug sind, dass die alten Daten weiterhin valide Vorteilsschätzungen liefern.

### TRPO — die formale Version (optional beim ersten Lesen)

!!! note "Erster Durchgang? Überfliege oder überspringe diesen Abschnitt."
    Nur dieser TRPO-Unterabschnitt ist optional — der Rest von §2, einschließlich *PPO — die praktische Version* unten, ist Kernstoff. Die Klippen-im-Nebel-Intuition oben ist alles an Trust-Region-Hintergrund, das PPO braucht; komm für TRPOs Second-Order-Maschinerie zurück, wenn du die formale Version willst.

PPOs Vorgänger, **TRPO** (Trust Region Policy Optimization, Schulman et al. 2015), formalisierte diese Idee mit einer harten mathematischen Nebenbedingung:

$$
\max_{\theta} \; \mathbb{E}_t \!\left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \, A_t \right] \quad \text{unter} \quad \mathbb{E}_t [\text{KL}(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta)] \leq \delta
$$

In Worten: *maximiere den erwarteten Vorteil, aber die KL-Divergenz zwischen neuer und alter Policy muss unter der Schwelle δ bleiben*. KL-Divergenz misst, wie verschieden zwei Wahrscheinlichkeitsverteilungen sind; sie zu begrenzen, begrenzt die Schrittweite der Policy.

TRPO funktioniert wunderschön — erfordert aber **Optimierung zweiter Ordnung** (Berechnung und Invertierung einer Approximation der Fisher-Informationsmatrix). Das ist teuer, schwer zu implementieren und verträgt sich nicht mit geteilten Actor-Critic-Netzen.

**Wie TRPO die Nebenbedingung löst:** konjugierter Gradient zur Berechnung der natürlichen Gradientenrichtung, dann Backtracking-Linesearch für den größten Schritt, der die KL-Schranke erfüllt. O(n²) in Policy-Parametern pro Update.

### PPO — die praktische Version

PPO (Schulman et al. 2017) fragte: *Können wir TRPOs Stabilität ohne die Second-Order-Mathe bekommen?* Die Antwort war peinlich einfach — **clippe das Objektiv, sodass der Gradient null wird, sobald sich die Policy „weit genug" bewegt hat**. Keine Nebenbedingung, keine KL-Berechnung, kein Lagrangian. Nur ein `min` und ein `clip` in der Loss-Funktion. First-Order-Optimierung (Vanilla Adam) reicht.

Das ist der eine Trick, der PPO von „interessant" zu „Standard" brachte.

| | TRPO | PPO |
|--|------|-----|
| Nebenbedingung | Hartes KL ≤ δ | Weiches Clip auf Ratio r_t |
| Optimizer | Konjugierter Gradient + Linesearch | Adam |
| Rechenaufwand pro Update | O(n²) | O(n) |
| Hyperparameter | δ (KL-Schwelle) | ε (`clip_range`), `target_kl` |
| Wann nutzen | Sicherheitskritische Aufgaben mit nahezu monotonen Verbesserungs-Garantien | Alles andere |

**PPOs `target_kl`** (SB3-Parameter, Default None) fügt eine optionale Early-Stopping-Prüfung hinzu: wenn die approximative KL zwischen alter und neuer Policy `target_kl` während einer Minibatch-Epoche überschreitet, stoppt das Training vorzeitig. Das gibt PPO eine weiche Version von TRPOs harter Nebenbedingung zu vernachlässigbaren Kosten.

---

## 3 · Das Wahrscheinlichkeits-Verhältnis

Um die Policy-Schrittweite zu clippen, brauchen wir zunächst einen Weg, sie zu *messen*. PPO nutzt das **Importance-Sampling-Verhältnis**:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

Lass uns das sorgfältig auspacken.

- **π_θ** ist die *aktuelle* Policy — die in diesem Gradientenschritt aktualisiert wird.
- **π_θ_old** ist die *eingefrorene* Policy, die das Rollout gesammelt hat. Ihre Parameter werden einmal zu Beginn des Rollouts gespeichert und ändern sich während der Update-Epochen nie.
- Der Zähler: Wahrscheinlichkeit der Aktion a_t unter der neuen Policy.
- Der Nenner: Wahrscheinlichkeit der Aktion a_t unter der alten Policy.

**Interpretation von r_t:**

| r_t-Wert | Bedeutung |
|-----------|---------|
| r_t = 1 | Die neue Policy weist a_t *dieselbe* Wahrscheinlichkeit zu wie die alte. (Immer wahr zu Beginn eines Updates — vor dem ersten Gradientenschritt ist θ = θ_old.) |
| r_t > 1 | Die neue Policy ist *wahrscheinlicher*, a_t zu nehmen, als die alte. Das Update hat die Wahrscheinlichkeit dieser Aktion *erhöht*. |
| r_t < 1 | Die neue Policy ist *unwahrscheinlicher*, a_t zu nehmen. Das Update hat die Wahrscheinlichkeit *gesenkt*. |
| r_t = 2 | Die neue Policy ist doppelt so wahrscheinlich, a_t zu nehmen, wie die alte. |
| r_t = 0.1 | Die neue Policy nimmt a_t fast nie mehr. |

Also ist r_t eine saubere, direkte Messung davon, *wie stark sich die Policy für diese spezifische Aktion bewegt hat*. Es ist ein Per-Transition-Trust-Region-Messgerät.

### Das ungeclippte Surrogat (CPI)

Wenn wir r_t einfach mit dem Vorteil multiplizieren und maximieren, erhalten wir das **Conservative-Policy-Iteration**-Objektiv (Kakade & Langford, 2002), das auch genau das ist, was REINFORCE+Importance-Sampling ergäbe:

$$
L^{\text{CPI}}(\theta) = \mathbb{E}_t \big[ r_t(\theta) \cdot A_t \big]
$$

Das sagt: *drücke die Wahrscheinlichkeit von Aktionen mit positivem Vorteil hoch, drücke die Wahrscheinlichkeit von Aktionen mit negativem Vorteil runter, gewichtet damit, wie weit wir uns schon bewegt haben*. Es ist mathematisch äquivalent zu REINFORCE, wenn r_t = 1 (einzelner Gradientenschritt), verallgemeinert aber via Ratio zu mehreren Schritten.

Vergleiche mit dem REINFORCE-Objektiv aus Unit 5:

$$
L^{\text{REINFORCE}}(\theta) = \mathbb{E}_t \big[ \log \pi_\theta(a_t|s_t) \cdot A_t \big]
$$

Nimm den Gradienten von beiden. Sie sind *identisch* bei θ = θ_old, weil ∇log π = (1/π)∇π und r_t bei 1 startet. Die Ratio-Form überlebt nur weiter vom Startpunkt entfernt — sie akkumuliert die Policy-Drift korrekt.

**Das Problem mit L^CPI:** es hat keine Bremsen. Wenn A_t groß und positiv ist, drückt der Gradient r_t fröhlich auf 5, 10, 100 — weit außerhalb jeder vernünftigen Trust Region. Katastrophen-Update-Territorium. Wir brauchen eine Obergrenze.

---

## 4 · Das geclippte PPO-Objektiv (DIE zentrale Gleichung)

Hier ist sie. Die eine Gleichung, die PPO definiert:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \Big[ \min\big( r_t(\theta) \cdot A_t, \; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \cdot A_t \big) \Big]
$$

Wobei:

- `clip(x, a, b)` = `max(a, min(x, b))` — also x auf das Intervall [a, b] beschränken.
- **ε** (Epsilon) ist ein Hyperparameter, typisch **0,2**. Es definiert den „Trust-Region-Radius".
- **min(·, ·)** wählt den *kleineren* der beiden Terme.

Das sieht einschüchternd aus. Ist es nicht. Gehen wir beide Fälle durch.

### Fall 1: A_t > 0 (die Aktion war besser als der Durchschnitt)

Wir wollen π_θ(a_t|s_t) *erhöhen* — diese gute Aktion wahrscheinlicher machen. Beim Update wächst r_t über 1.

- **Ohne Clip:** L = r_t · A_t. Der Gradient drückt r_t unbegrenzt hoch. Selbst wenn r_t = 5, sagt der Gradient noch „weiter hoch!" — weit außerhalb der Trust Region.
- **Mit Clip:** überlege, was `min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t)` macht.
    - Wenn r_t ≤ 1+ε: der ungeclippte Term r_t · A_t ist der *kleinere* der beiden (na ja, bis 1+ε sind sie gleich). Der Gradient fließt normal — wir verbessern die Policy.
    - Wenn r_t > 1+ε: der geclippte Term `(1+ε) · A_t` ist *konstant in θ* (der Clip flacht ihn ab). Er ist auch *kleiner* als r_t · A_t (da A_t > 0 und r_t > 1+ε). Das `min` wählt den geclippten Term. **Der Gradient bezüglich θ ist null.**

Im Klartext: *„Du hast die Wahrscheinlichkeit dieser Aktion bereits um 20% erhöht. Das reicht. Drück nicht weiter auf dieser einzelnen Transition — die Daten sind so weit draußen nicht vertrauenswürdig."*

### Fall 2: A_t < 0 (die Aktion war schlechter als der Durchschnitt)

Wir wollen π_θ(a_t|s_t) *senken*. Beim Update fällt r_t unter 1.

- **Ohne Clip:** L = r_t · A_t mit A_t negativ. Niedrigeres r_t → negativeres Produkt → „besserer" Loss (wir maximieren). Der Gradient drückt r_t Richtung 0. Katastrophal — wir können eine Aktion in einem Update effektiv aus der Policy löschen.
- **Mit Clip:**
    - Wenn r_t ≥ 1−ε: Gradient fließt, wir senken die Wahrscheinlichkeit der Aktion normal.
    - Wenn r_t < 1−ε: der geclippte Term `(1−ε) · A_t` ist konstant in θ. Mit A_t < 0 ist der *ungeclippte* r_t · A_t *negativer* (da r_t < 1−ε und mit einer negativen Zahl multipliziert kleineres r ein größeres negatives Resultat ergibt)... Moment, lass mich das nochmal machen.

    Sei vorsichtig mit dem Vorzeichen. A_t < 0, r_t < 1−ε. Dann ist r_t · A_t > (1−ε) · A_t (eine kleinere positive Zahl r mit einer negativen multipliziert ergibt ein *größeres* — weniger negatives — Resultat). Also ist der ungeclippte Term *größer*, und das `min` wählt den *geclippten* Term `(1−ε) · A_t`. Der ist konstant in θ. **Gradient erneut null.**

Im Klartext: *„Du hast die Wahrscheinlichkeit dieser Aktion bereits um 20% gesenkt. Stopp. Jede weitere Senkung auf Basis dieser einen Transition wäre fahrlässig."*

### Das ASCII-Bild

```
Vorteil A_t > 0  (gute Aktion — wir wollen r_t wachsen lassen)

  L_CLIP
    │
    │
    │                ─────────────  (geclippt: Gradient = 0)
    │              /
    │            /
    │          /  (linearer Bereich: Gradient ∝ A_t > 0)
    │        /
    │      /
    │    /
    │  /
    │/_______________________________  r_t
    0     1-ε    1     1+ε

Vorteil A_t < 0  (schlechte Aktion — wir wollen r_t schrumpfen lassen)

  L_CLIP
    │
  ──┼────                            (geclippt: Gradient = 0)
    │    \
    │      \
    │        \   (linearer Bereich: Gradient ∝ A_t < 0)
    │          \
    │            \
    │              \
    │________________\______________  r_t
    0     1-ε    1     1+ε
```

Zwei flache Regionen, ein abfallender Bereich in der Mitte. Die flachen Regionen sind, wo der Clip „greift" und der Gradient abschaltet — die Trust-Region-Grenze. Innerhalb von [1−ε, 1+ε] verhält sich PPO wie Vanilla Policy Gradient mit Importance Sampling.

### Warum `min` und nicht nur `clip`?

Ein subtiler Punkt, der jeden erwischt: das `min` ist dafür da, die Schranke **pessimistisch** zu machen. Wir wollen, dass das Objektiv *weniger* attraktiv wird, wann immer Clipping aktiv ist, niemals *attraktiver*. Ohne `min` könnte in Fall 1 der geclippte Term `(1+ε) · A_t` größer als `r_t · A_t` sein, wenn r_t < 1+ε — und der Agent könnte den Clip ausnutzen, um Updates zu *vermeiden*. Das `min` stellt sicher, dass wir immer den schlechteren der beiden nehmen und nie ein Gratisessen bekommen.

Es ist derselbe Grund, warum TRPO eine obere Schranke auf KL nutzt: optimiere das pessimistische Surrogat, erhalte ein garantiert-nicht-verschlechterndes Update.

### Der Clip-Bereich ε

`clip_range` ist der wichtigste PPO-Hyperparameter. Er setzt direkt die Trust-Region-Größe.

- **ε = 0,1** (eng): sehr konservative Updates, langsam aber stabil. Nutze bei fragilem Training.
- **ε = 0,2** (Default): der Sweet Spot, identifiziert im Original-Paper.
- **ε = 0,3+** (locker): aggressive Updates, schneller anfangs, aber instabil. Schadet oft.
- **ε = 0,0**: identisch zu A2C mit Importance Sampling — keine Trust Region überhaupt (aber dann muss `n_epochs` 1 sein, sonst divergiert es).

---

## 5 · Generalized Advantage Estimation (GAE)

Wir haben A_t geschrieben, ohne es zu definieren. Woher kommen Vorteile?

Erinnere dich an die Actor-Critic-Unit:

$$
A_t = G_t - V(s_t)
$$

Der Vorteil ist der *Return* minus die *Baseline*. Aber welcher Return? Du hast eine Wahl mit einem Spektrum.

### Der Bias-Varianz-Trade-off in der Vorteilsschätzung

| Schätzer | Formel | Bias | Varianz |
|-----------|---------|------|----------|
| 1-Schritt-TD | $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ | Hoch (hängt davon ab, dass V stimmt) | Niedrig (nur eine Zufallsbelohnung) |
| 2-Schritt-TD | $r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t)$ | Mittel | Mittel |
| Voller Monte Carlo | $G_t - V(s_t) = (\sum_{k=0}^\infty \gamma^k r_{t+k}) - V(s_t)$ | Null (unverzerrt) | Hoch (Summe vieler Zufallsbelohnungen) |

Es gibt kein Gratisessen. Kurzhorizont-Bootstrapping ist verzerrt (wir vertrauen V zu sehr). Langhorizont-Monte-Carlo ist unverzerrt, aber verrauscht (eine glückliche Episode kann dominieren). REINFORCE nutzte volles MC und litt unter Varianz. A2C nutzt 1-Schritt-TD und leidet unter Bias.

### GAE — der elegante Kompromiss

**Generalized Advantage Estimation** (Schulman et al. 2015) interpoliert zwischen allen via einem einzelnen Parameter **λ**:

$$
A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{k=0}^{\infty} (\gamma \lambda)^k \, \delta_{t+k}
$$

wobei $\delta_{t+k} = r_{t+k} + \gamma V(s_{t+k+1}) - V(s_{t+k})$ der 1-Schritt-TD-Fehler bei Schritt t+k ist.

**Regler:**

- **λ = 0**: $A_t^{\text{GAE}} = \delta_t$ — reines 1-Schritt-TD. Maximaler Bias, minimale Varianz.
- **λ = 1**: $A_t^{\text{GAE}} = \sum_k \gamma^k \delta_{t+k}$ was (durch Teleskopieren) gleich $G_t - V(s_t)$ ist — voller Monte Carlo. Null Bias, maximale Varianz.
- **λ = 0,95**: typischer Default. Etwa 95% Monte Carlo mit kleinem Bias vom Bootstrapping. Best of both worlds in der Praxis.

Du kannst λ als *„wie sehr vertraue ich meinem Critic V?"* denken — wenn V genau ist, setze λ klein und lehn dich auf Bootstrapping; wenn V Schrott ist (frühes Training), setze λ näher an 1 und vertraue den echten Returns.

### Rekursive Berechnung

Das Brillante: GAE kann in **einem Rückwärtsdurchlauf** durch das Rollout berechnet werden, mit einem einzigen Akkumulator:

```python
# Pseudo-code
advantages = zeros(T)
gae = 0
for t in reversed(range(T)):
    delta = rewards[t] + gamma * values[t+1] * not_done[t] - values[t]
    gae = delta + gamma * lam * not_done[t] * gae
    advantages[t] = gae
returns = advantages + values   # used as targets for the value function
```

Das ist genau das, was CleanRLs `ppo.py` in etwa 8 Zeilen macht. Die `not_done`-Maske setzt das Bootstrap über Episodengrenzen hinweg auf null.

**Dieses Rollout, dieser Vorteils-Tensor, wird über alle n_epochs des Updates wiederverwendet.** Das ist der Schlüssel: A_t wird *einmal* berechnet, eingefroren, und der Clip verhindert, dass die Policy so weit driftet, dass A_t bedeutungslos wird.

---

## 6 · Der volle PPO-Loss

PPO trainiert den Actor *und* den Critic *und* hält die Entropie aufrecht, alles in einem kombinierten Loss:

$$
L^{\text{PPO}}(\theta) = L^{\text{CLIP}}(\theta) - c_1 \cdot L^{\text{VF}}(\theta) + c_2 \cdot L^{\text{ENT}}(\theta)
$$

(Vorzeichen hängen davon ab, ob du minimierst oder maximierst. SB3 minimiert, also ist es intern `-L_CLIP + c1·L_VF - c2·L_ENT`. Die Mathe ist dieselbe.)

| Term | Formel | Rolle | Typischer Koeffizient |
|------|---------|------|---------------------|
| $L^{\text{CLIP}}$ | das geclippte Surrogat aus §4 | **maximieren** — Policy verbessern | 1,0 (implizit) |
| $L^{\text{VF}}$ | $\tfrac{1}{2} \mathbb{E}_t [(V_\theta(s_t) - V_t^{\text{target}})^2]$ | **minimieren** — Critic Richtung GAE-Returns trainieren | $c_1 = 0,5$ |
| $L^{\text{ENT}}$ | $H(\pi_\theta(\cdot \mid s_t)) = -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)$ | **maximieren** — Erkundung am Leben halten | $c_2 = 0,01$ |

Das Wertziel $V_t^{\text{target}}$ ist einfach `advantages[t] + values[t]` aus dem GAE-Durchlauf — also der GAE-korrigierte Return.

Ein Backward-Pass auf diesem kombinierten Loss aktualisiert den geteilten Backbone (falls vorhanden), den Policy-Head und den Wert-Head gleichzeitig. Vergleiche mit DQN, das nur einen Wert-Head trainiert; vergleiche mit REINFORCE, das überhaupt keinen Critic hat.

---

## 7 · Die PPO-Trainingsschleife (mehrere Epochen)

Hier ist das Herz von PPO, in Pseudocode:

```
initialize θ
loop forever:
    # ----- ROLLOUT PHASE -----
    θ_old ← θ                                # freeze a copy of the policy
    collect n_steps × n_envs transitions using π_{θ_old}
    for each transition store: (s_t, a_t, r_t, done_t, log π_{θ_old}(a_t|s_t), V_{θ_old}(s_t))
    compute advantages A_t with GAE  → freeze
    compute value targets V_target_t = A_t + V_{θ_old}(s_t) → freeze

    # ----- UPDATE PHASE -----
    for epoch in 1..n_epochs:                # typically 10
        shuffle the rollout indices
        for each mini-batch of size batch_size:
            log_π_new ← log π_θ(a_t | s_t)              # CURRENT θ
            r_t ← exp(log_π_new − log π_{θ_old}(a_t|s_t))
            L_CLIP ← min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t).mean()
            L_VF   ← ((V_θ(s_t) − V_target_t)²).mean() · 0.5
            L_ENT  ← entropy(π_θ(·|s_t)).mean()
            L      ← −L_CLIP + c_1 · L_VF − c_2 · L_ENT
            backward(L); clip_grad_norm_(0.5); optimizer.step()
```

Drei Feinheiten, auf die sich das Feld erst über Jahre geeinigt hat:

1. **`log π_{θ_old}` wird gespeichert, nicht neu berechnet.** Wenn wir das Rollout sammeln, speichern wir die Log-Wahrscheinlichkeit jeder Aktion unter der damaligen Policy. Das ist jetzt eine Konstante während des Updates. Die Ratio ist `exp(new_log_prob - old_log_prob)` — numerisch stabil und trivial.
2. **Vorteile werden pro Minibatch normalisiert.** Die meisten Implementierungen (CleanRL, SB3) machen `A = (A - A.mean()) / (A.std() + 1e-8)` vor der Berechnung von L_CLIP. Das stabilisiert die Skala der Gradienten über Batches mit sehr unterschiedlichen Belohnungsgrößen hinweg.
3. **Gradienten werden per globaler Norm geclippt.** `clip_grad_norm_(0.5)` verhindert, dass der seltene explodierende Gradient das Modell zerlegt. Billige Versicherung.

### Warum das funktioniert (die Pointe)

> A_t ist fix. Nur r_t hängt vom trainierbaren θ ab. Der Clip stoppt r_t davor, zu weit von 1 wegzudriften, sodass die Policy in einer Region bleibt, in der A_t weiterhin eine valide Schätzung der Aktionsqualität ist. Innerhalb dieser Region können wir es uns leisten, viele Gradientenschritte zu machen.

Das ist die gesamte Idee von PPO. Alles andere ist Engineering.

---

## 8 · Hyperparameter-Wörterbuch

Jeder PPO-Hyperparameter, was er steuert, und was sich ändert, wenn du den Regler drehst.

| Parameter | Symbol | Typisch | Mathematische Rolle |
|-----------|--------|---------|-------------------|
| `n_steps` | T | 64 – 2048 | Rollout-Länge pro Umgebung |
| `n_epochs` | K | 10 | Durchläufe über jedes Rollout |
| `batch_size` | M | 64 – 256 | Minibatch-Größe; muss `n_steps × n_envs` teilen |
| `learning_rate` | α | 3e-4 | Adam-Schrittweite |
| `clip_range` | ε | 0,2 | Trust-Region-Halbweite |
| `gae_lambda` | λ | 0,95 | GAE-Interpolation (0=TD, 1=MC) |
| `gamma` | γ | 0,99 | Diskontierungsfaktor |
| `vf_coef` | c_1 | 0,5 | Gewicht des Critic-Loss |
| `ent_coef` | c_2 | 0,0 – 0,01 | Gewicht des Entropie-Bonus |
| `max_grad_norm` | — | 0,5 | Globale Gradienten-Norm-Obergrenze |
| `n_envs` | N | 4 – 16 | Parallele Umgebungen |

### Was passiert, wenn du jeden Regler drehst

**`n_steps` ↑** : besseres Monte-Carlo-Signal in GAE (mehr Terme in der Summe), mehr Speicher, langsamere Wall-Clock pro Update. **↓** : verrauschtere Vorteile, aber häufigere Policy-Updates. Faustregel: länger für Sparse-Reward-Aufgaben, kürzer für Dense-Reward-Aufgaben.

**`n_epochs` ↑** : mehr Wiederverwendung, stichprobeneffizienter, mehr Risiko, dass die Policy aus der Trust Region driftet (beobachte `approx_kl`). **↓** : näher an A2C, sicherer aber verschwenderisch. 10 ist die magische Zahl aus dem Paper; Werte 3–20 sind vernünftig.

**`batch_size` ↑** : weniger verrauschte Gradienten pro Schritt, weniger Gradientenschritte pro Epoche. **↓** : mehr Updates, mehr Rauschen. Nebenbedingung: `n_steps × n_envs` muss durch `batch_size` teilbar sein.

**`learning_rate` ↑** : schnelleres Anfangslernen, höheres Risiko von Policy-Kollaps. **↓** : sicherer, langsamer. Viele Praktiker annealen α linear gegen 0 über das Training.

**`clip_range` ↑** : größere Trust Region, schnellere Updates, Risiko von Instabilität. **↓** : engere Trust Region, langsamer aber sicherer. Manche Implementierungen annealen ε ebenfalls über das Training nach unten.

**`gae_lambda` ↑** : näher an Monte Carlo, niedrigerer Bias, höhere Varianz. **↓** : näher an 1-Schritt-TD, höherer Bias, niedrigere Varianz. 0,95 ist robust.

**`gamma` ↑** (z. B. 0,999): Agent kümmert sich um weit entfernte Zukunftsbelohnungen; schwerer zu lernen, aber besser für Langhorizont-Aufgaben. **↓** (z. B. 0,9): Agent ist kurzsichtig; schnelleres Lernen bei Kurzhorizont-Aufgaben. Äquivalent dazu, deinen effektiven Planungshorizont ≈ 1/(1−γ) zu wählen.

**`vf_coef` ↑** : Critic-Genauigkeit priorisieren auf Kosten der Policy-Verbesserung. Nützlich, wenn der Wert-Loss riesig ist und das Gradientensignal dominiert. **↓** : die Policy führen lassen; der Critic holt langsamer auf.

**`ent_coef` ↑** : mehr Erkundung; Policy bleibt länger hochentropisch. Nutze, wenn der Agent zu früh zu einer entarteten Policy kollabiert. **↓** : schnellere Konvergenz zu deterministischem Verhalten. Auf 0 setzen ist okay für Aufgaben mit ausreichender Erkundung aus der Umweltzufälligkeit.

**`max_grad_norm` ↑** : weniger aggressives Clipping. **↓** : aggressiver. 0,5 ist ein sicherer Default; muss selten getunt werden.

**`n_envs` ↑** : weniger korrelierte Samples, mehr Wall-Clock-Parallelismus, mehr Speicher. **↓** : stärker korrelierte Rollouts, GAE-Schätzungen stärker verzerrt Richtung Einzeltrajektorien-Dynamik. In Godot RL ist das das `n_parallel`-Flag und wird vom Env-Wrapper gesetzt, nicht vom Algo.

---

## 9 · `approx_kl` als Diagnose

Du wirst mehr Zeit damit verbringen, auf `train/approx_kl` in TensorBoard zu starren, als auf jede andere PPO-Metrik. Es ist dein bestes Einzelsignal für „ist mein PPO gesund?"

Die KL-Divergenz zwischen alter und neuer Policy ist nicht gratis exakt zu berechnen (du müsstest über alle Aktionen summieren). PPO nutzt einen billigen Ein-Stichproben-Schätzer:

$$
\widehat{\text{KL}}_t = (r_t - 1) - \log r_t
$$

Du kannst auch die (ältere, einfachere) Form `−log r_t` sehen — beide approximieren dasselbe. Schulmans „Johns Blog-Post über KL" empfiehlt die erste Form, weil sie immer ≥ 0 und niedriger in der Varianz ist.

**Wie liest man das:**

| `approx_kl`-Bereich | Bedeutung | Aktion |
|-------------------|---------------|--------|
| 0,005 – 0,02 | Gesund. Policy bewegt sich, aber nicht zu schnell. | Nichts tun. |
| < 0,001 | Policy bewegt sich kaum. Vielleicht ist das Lernen fertig — oder festgefahren. | Erhöhe `learning_rate`, erhöhe `clip_range`, oder prüfe, ob die Belohnung noch steigt. |
| 0,03 – 0,05 | Grenzwertig. Beobachte die Belohnungskurve auf Instabilität. | Erwäge, `learning_rate` oder `clip_range` leicht zu senken. |
| > 0,05 | Policy bewegt sich sehr schnell. Wahrscheinlich auf dem Weg zum Kollaps. | Senke `learning_rate`, senke `clip_range`, senke `n_epochs`. |

!!! warning "Wenn `approx_kl` explodiert"
    Manche Implementierungen (SB3, CleanRL mit `--target-kl`) implementieren **Early Stopping** der Update-Epochen: wenn `approx_kl` ein Ziel überschreitet (z. B. 0,015), bricht die innere Update-Schleife sofort ab. Das ist ein Sicherheitsnetz für Fälle, in denen ein Minibatch die Policy zufällig zu weit drückt. Wenn du siehst, dass `train/n_updates` konsistent unter `n_epochs` liegt, ist das der Grund.

Verwandte Metriken, die zu verfolgen sind:

- `rollout/ep_rew_mean` — das tatsächliche Lernsignal. Alles andere dient dazu, dass das hochgeht.
- `train/entropy_loss` — sollte *langsam in der Größe abnehmen*. Plötzlicher Kollaps = Erkundungsversagen.
- `train/explained_variance` — wie gut der Critic Returns vorhersagt. Sollte Richtung 1,0 kriechen; Werte unter 0 bedeuten, der Critic ist schlechter als den Mittelwert vorherzusagen.
- `train/clip_fraction` — Anteil der Samples, bei denen der Clip aktiv wurde. 0,1–0,3 ist typisch. Nahe 0 = Clip greift nie (probier größere LR). Nahe 1 = Clip greift immer (LR oder Clip zu aggressiv).

### Bau es · Clip-Range-Ablation

Lies die Diagnose-Tabelle nicht nur — erzeuge die Daten selbst. Trainiere dieselbe Umgebung dreimal, variiere nur `clip_range`, und sieh zu, wie das Trust-Region-Argument aus §4 in den Metriken aus §9 auftaucht. CartPole-v1 hält die Schleife schnell; sobald es funktioniert, wiederhole es auf deiner Godot-Umgebung mit `gdrl --clip_range=...`.

```python
import gymnasium as gym
from stable_baselines3 import PPO

for clip in (0.1, 0.2, 0.4):
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, clip_range=clip, verbose=0,
                tensorboard_log="runs/clip_ablation")
    model.learn(total_timesteps=100_000, tb_log_name=f"clip_{clip}")
    env.close()
```

Öffne dann `tensorboard --logdir runs/clip_ablation` und lege `rollout/ep_rew_mean`, `train/approx_kl` und `train/clip_fraction` für alle drei Runs übereinander. Schreibe eine einabsätzige Erklärung dessen, was du siehst, mit Bezug auf §4 und §9.

!!! check "Fertig, wenn"
    Alle drei Runs erscheinen in TensorBoard und `rollout/ep_rew_mean` des `clip_range=0.2`-Runs klettert deutlich Richtung CartPole-v1s 500-Schritte-Return-Obergrenze. Erwarte das Muster, das §4 vorhersagt — der 0.4-Run driftet am schnellsten bei `train/approx_kl` (lockerste Trust Region), der 0.1-Run clippt am häufigsten bei `train/clip_fraction` (engster Bereich). Der Kontrast ist zwischen 0.1 und 0.4 am deutlichsten; einzelne Runs sind Seed-verrauscht — ein vertauschtes Nachbarpaar heißt also: noch einmal laufen lassen, nicht: Theorie kaputt. Wenn du das Muster in deinem Absatz erklären kannst, ohne §4 erneut zu lesen, hat es beim Clip-Mechanismus Klick gemacht.

---

## 10 · CleanRL-Referenzimplementierung (optional beim ersten Lesen)

!!! note "Erster Durchgang? Überfliege oder überspringe diesen Abschnitt."
    Die Kernlektion ist §1–§9 — das geclippte Objektiv, GAE, die Trainingsschleife und die Hyperparameter-/Diagnose-Wörterbücher; komm zu diesem 30-Minuten-Code-Walkthrough zurück, sobald du PPO mindestens einmal trainiert hast.

!!! tip "Lies `ppo.py` während du diesen Abschnitt liest"
    Öffne `https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py` in einem anderen Tab. Der gesamte Algorithmus sind ~300 Zeilen, eine Datei, keine Abstraktionen. Jedes PPO-Konzept aus dieser Unit steht da in reinem PyTorch.

CleanRLs `ppo.py` ist die *definitive* pädagogische PPO-Implementierung. Stable-Baselines3s PPO ist umfangreicher, aber über eine Klassenhierarchie verteilt; CleanRL hält alles lesbar.

### Leseleitfaden

| Abschnitt in `ppo.py` | Worauf achten | Mappt auf diese Unit |
|---------------------|------------------|-------------------|
| Zeilen ~80–100 (die `Agent`-Klasse) | Geteilter MLP-Backbone, getrennte Actor- und Critic-Heads | Actor-Critic-Rekap |
| Zeilen ~100–140 (Rollout-Storage-Tensoren `obs`, `actions`, `logprobs`, `rewards`, `dones`, `values`) | Der Rollout-Buffer | §7 — was eingefroren wird |
| Zeilen ~140–170 (die Env-Step-Schleife) | Umgebungen stepen, Transitionen speichern | §1 — On-Policy-Daten sammeln |
| Zeilen ~170–195 (der GAE-Backward-Pass) | Die exakte Rekursion aus §5 | §5 — GAE |
| Zeilen ~195–250 (die Update-Phase: `for epoch in range(args.update_epochs)`) | Die inneren Schleifen — Epochen, dann Minibatches | §7 |
| Innerhalb der Minibatch-Schleife: `ratio = (newlogprob - mb_logprobs).exp()` | Das ist r_t | §3 |
| `pg_loss1 = -mb_advantages * ratio`<br>`pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-clip, 1+clip)`<br>`pg_loss = torch.max(pg_loss1, pg_loss2).mean()` | **Der Clip!** (Anmerkung: `max`, weil sie negiert haben, äquivalent zum `min` der positiven Form) | §4 |
| `v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()` | Critic-Loss | §6 |
| `entropy_loss = entropy.mean()` | Entropie-Bonus | §6 |
| `loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef` | Der kombinierte PPO-Loss | §6 |
| `approx_kl = ((ratio - 1) - logratio).mean()` | Die KL-Diagnose | §9 |

Nach dieser Unit solltest du `ppo.py` von oben nach unten ohne Verwirrung *lesen* können. Nimm dir 30 Minuten und tu es.

---

## 11 · PPO in Godot RL

Godot RL Agents wrappt deine Godot-Umgebung in ein Gymnasium-kompatibles Interface (`StableBaselinesGodotEnv`) und reicht es an Stable-Baselines3s PPO-Implementierung weiter. Alles, was du hier gelernt hast, gilt direkt.

Ein typischer Trainingsbefehl:

```bash
gdrl --env_path=builds/MyEnv.x86_64 \
     --n_steps=512 \
     --batch_size=256 \
     --n_epochs=10 \
     --gamma=0.99 \
     --gae_lambda=0.95 \
     --clip_range=0.2 \
     --ent_coef=0.005 \
     --learning_rate=3e-4 \
     --total_timesteps=1000000
```

Jedes Flag ist ein Regler aus §8. Von links nach rechts:

- `--n_steps=512` — 512 Schritte pro Umgebung vor jedem Update sammeln.
- `--batch_size=256` — jedes Rollout in Minibatches von 256 splitten.
- `--n_epochs=10` — 10 Durchläufe über jedes Rollout (die PPO-Datenwiederverwendungs-Magie).
- `--gamma=0.99` — Diskontierungsfaktor, ~100-Schritt-effektiver-Horizont.
- `--gae_lambda=0.95` — leichter Bootstrapping-Bias, viel Varianzreduktion.
- `--clip_range=0.2` — Standard-Trust-Region.
- `--ent_coef=0.005` — leichter Erkundungs-Bonus (ein wenig unter dem SB3-Default von 0; Godot-Umgebungen brauchen oft einen Schubser).
- `--learning_rate=3e-4` — Adam-Default.

### Tuning-Workflow, wenn das Training schlecht aussieht

Die TensorBoard-getriebene Debug-Schleife in Reihenfolge:

1. **`rollout/ep_rew_mean` flach und `train/approx_kl` sehr niedrig** → Policy bewegt sich nicht. Erhöhe `learning_rate` um 3×, oder erhöhe `clip_range` auf 0,3.
2. **`approx_kl` Spikes über 0,05, Belohnung instabil** → Policy bewegt sich zu schnell. Senke `clip_range` auf 0,1 oder `learning_rate` auf 1e-4.
3. **`train/entropy_loss` kollabiert früh auf ~0** → Policy wurde deterministisch, bevor genug erkundet wurde. Erhöhe `ent_coef` auf 0,02.
4. **`train/explained_variance` bleibt nahe 0** → Critic lernt nicht. Erhöhe `vf_coef` auf 1,0 oder verlängere `n_steps`.
5. **`train/clip_fraction` nahe 0** → Clip greift nie; du machst effektiv A2C mit vielen Epochen. Möglicherweise `clip_range` zu locker für die LR.

### Das große Bild

Der `StableBaselinesGodotEnv`-Wrapper ist einfach eine Gymnasium-Umgebung. **PPO hat keine Ahnung, dass Godot auf der anderen Seite ist.** Es sieht Beobachtungen und Belohnungen eingehen, Aktionen rausgehen, genau wie CartPole. Alles, was du in dieser Unit über PPO gelernt hast, gilt identisch, egal ob die Env CartPole, Atari, MuJoCo oder dein Godot-Plattformer ist.

---

## 12 · PPO vs. Alternativen — Übersicht

| Methode | Stichprobeneffizienz | Stabilität | Kontinuierliche Aktionen | Diskrete Aktionen | Speicher |
|--------|-------------------|-----------|--------------------|------------------|--------|
| REINFORCE (Unit 5) | Niedrig | Niedrig | Ja | Ja | Niedrig |
| A2C (Actor-Critic-Unit) | Mittel | Mittel | Ja | Ja | Niedrig |
| **PPO (diese Unit)** | **Hoch** | **Hoch** | **Ja** | **Ja** | **Mittel** |
| DQN (DQN-Unit) | Hoch | Hoch | Nein | Nur ja | Hoch (Replay-Buffer) |
| SAC | Sehr hoch | Hoch | Nur ja | Umständlich | Hoch (Replay-Buffer) |
| TRPO | Hoch | Sehr hoch | Ja | Ja | Hoch (Hessian) |

PPOs Gewinner-Kombination: **gute Stichprobeneffizienz, gute Stabilität, funktioniert auf beiden Aktionstypen, einfache First-Order-Implementierung, kein Replay-Buffer nötig**. Kein anderer Algorithmus trifft alle fünf. Deshalb wurde es der Default.

---

## Was kommt als Nächstes

Du verstehst jetzt jede Zeile von PPO. Die nächste Unit, **PPO in der Praxis**, lässt die Gleichungen fallen und bringt dich zurück an die Tastatur: einen echten Godot-Agenten mit PPO trainieren, die Metriken aus §9 in TensorBoard sich entwickeln sehen und Intuition dafür entwickeln, welche Regler wann zu drehen sind.

Danach bewegt sich der Kurs von der Theorie zur Engineering: Curriculum-Design, Reward-Shaping, Sim-to-Real und Deployment. PPO wird im Hintergrund jeder verbleibenden Unit sein.

!!! info "Selbstcheck, bevor du weitermachst"
    Kannst du diese Fragen in eigenen Worten beantworten?

    1. Nenne das **geclippte PPO-Objektiv** aus dem Kopf und erkläre, was jeder `min`- / `clip`-Term verhindert.
    2. Was misst das **Wahrscheinlichkeits-Verhältnis** `r_t(θ)`, und was bedeutet `r_t = 1`?
    3. Was wägt **GAE-λ** ab, und worauf kollabieren λ = 0 und λ = 1?
    4. Was sagt dir `approx_kl > 0,05`, und welchen Hyperparameter würdest du zuerst ändern?
    5. Warum lässt PPO **mehrere Epochen** über dasselbe Rollout laufen, und warum bricht das die On-Policy-Annahme *nicht*?

    Wenn du alle fünf beantworten kannst — du verstehst PPO gut genug, das Paper zu lesen, ohne Gleichungen zu überspringen.

??? success "Antworten zum Selbstcheck"
    1. L^CLIP(θ) = 𝔼_t[ min( r_t·A_t, clip(r_t, 1−ε, 1+ε)·A_t ) ]. Der **clip** flacht das Objektiv ab, sobald r_t [1−ε, 1+ε] in der *profitablen* Richtung verlässt — kein Gradient belohnt es, die Ratio einer guten Aktion über 1+ε oder die einer schlechten unter 1−ε zu drücken. Das **min** hält die Schranke pessimistisch: Ist die Policy bereits in die *falsche* Richtung gedriftet (etwa die Ratio einer schlechten Aktion über 1+ε), ist der ungeclippte Term der kleinere und bleibt aktiv — sein Gradient zieht die Ratio wieder zurück.
    2. **r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)** — ein Per-Transition-Trust-Region-Messgerät dafür, wie weit sich die aktuelle Policy bei dieser spezifischen Aktion bewegt hat. r_t = 1 bedeutet, die neue Policy weist a_t dieselbe Wahrscheinlichkeit zu wie die alte (immer wahr vor dem ersten Gradientenschritt, wenn θ = θ_old).
    3. **GAE-λ** wägt **Bias gegen Varianz** in der Vorteilsschätzung ab. λ = 0 kollabiert zum 1-Schritt-TD-Fehler δ_t (maximaler Bias, minimale Varianz); λ = 1 teleskopiert zu vollem Monte Carlo, G_t − V(s_t) (null Bias, maximale Varianz).
    4. `approx_kl > 0,05` bedeutet, die Policy bewegt sich sehr schnell und ist wahrscheinlich auf dem Weg zum Kollaps. Ändere zuerst die **`learning_rate`** (senke sie, z. B. auf 1e-4); `clip_range` und `n_epochs` sind die nächsten Regler in der Liste aus §9.
    5. Datensammeln ist der Flaschenhals (besonders in Godot), also verwendet PPO jedes Rollout für `n_epochs` Durchläufe wieder statt für einen. Das bricht die On-Policy-Annahme nicht, weil die eingefrorenen Vorteile A_t valide bleiben, solange die Policy nahe an π_θ_old bleibt — und der **clip** nimmt jeden Anreiz, eine Ratio weiter aus [1−ε, 1+ε] hinauszuschieben (während das min Drift in die falsche Richtung weiter bestraft), und beschränkt damit jedes Update auf ungefähr diese Region.

---

## Stretch Goals

Wenn du tiefer gehen willst, in steigender Reihenfolge des Aufwands:

1. **Lies das Original-PPO-Paper.** Schulman et al. 2017, „Proximal Policy Optimization Algorithms" — 8 Seiten, sehr lesbar. Jede Gleichung im Paper mappt auf einen Abschnitt dieser Unit. Beachte, dass das Paper *zwei* Varianten vorschlägt: das geclippte Objektiv (was wir abgedeckt haben, überall genutzt) und einen adaptiven KL-Strafterm (meist vergessen).
2. **Implementiere PPO von Grund auf.** Forke CleanRLs `ppo.py`, trainiere auf CartPole-v1 (sollte in <1 Minute lösen) und LunarLander-v2 (~10 Minuten auf CPU). Nutze *nicht* SB3; der Punkt ist, jede Zeile selbst zu tippen.
3. **Reproduziere das GAE-λ-Sweep-Diagramm** (Abbildung 1 des GAE-Papers, Schulman et al. 2015). Trainiere PPO mit λ ∈ {0,0, 0,5, 0,9, 0,95, 0,99, 1,0} und beobachte, wie Varianz vs. Bias in der Praxis aussieht.
4. **Probier `target_kl`-basiertes Early Stopping.** Füge eine Prüfung hinzu, die die innere Update-Schleife bricht, wenn `approx_kl > 0,015`. Vergleiche Trainings-Stabilität mit und ohne.

Wenn du (2) schaffst, verstehst du PPO offiziell besser als 95% der Leute, die es nutzen. Das ist das wahre Ziel dieser Unit.

---

[← Actor-Critic](unit-actor-critic.md) · [Kursstartseite](index.md) · [→ PPO in der Praxis](unit-04.md)
