# RL Essentials — Vom Netzwerk zum lernenden Agenten

[← Neuronale Grundlagen 2](unit-neural-02.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~25 min · Schnelle Belohnungsanpassung: ~20 min · Training kann im Hintergrund laufen

!!! success "Was du nach dieser Unit kannst"
    - Erklären, was Verstärkungslernen zu dem Netzwerk hinzufügt, das du gerade gebaut hast
    - Die Schleifenbausteine benennen: Beobachtung, Aktion, Belohnung, nächste Beobachtung
    - Episoden, Return, Diskontierung, Policy und Exploration beschreiben
    - Zeigen, wo Godot und Python während des Trainings jeweils sitzen
    - Eine Belohnung ändern und vorhersagen, wie die Lernkurve reagieren sollte

!!! note "Voraussetzungen"
    - **Unit 0 abgeschlossen** — Conda, Godot und ein erfolgreicher BallChase-Lauf
    - **Neuronale Grundlagen 1–2 abgeschlossen** — du hast Inputs, Gewichte, Loss, Gradienten und Inference gesehen
    - Sicherer Umgang mit einem Terminal

!!! info "Drei Wege, deine KI zu beobachten"
    Godot-Editor (der Agent bewegt sich live) · TensorBoard (`rollout/ep_rew_mean` ändert sich) · Code (die Belohnungszeile, die du bearbeitest)

Du hast bereits ein kleines Netzwerk gebaut, das Zahlen in Entscheidungen umwandelt. RL fügt
ein fehlendes Stück hinzu: Das Netzwerk lernt nicht mehr aus richtigen Antworten. Es lernt
aus Aktionen, Konsequenzen und Belohnung.

---

## 1 · Was Verstärkungslernen hinzufügt

**Reinforcement Learning (RL)** bringt Software bei, gute Entscheidungen zu treffen, indem ein
**Agent** in einer **Umgebung** handelt und das Geschehene mit einer **Belohnung** bewertet wird.

Überwachtes Lernen sagt: „Dieser Input soll dieses Ziel erzeugen." RL sagt:
„Probiere eine Aktion, beobachte, was passiert ist, und nutze die Belohnung, um beim
nächsten Mal bessere Entscheidungen zu treffen."

Dieser Unterschied ist für Spiele wichtig. Ein Designer kennt vielleicht nicht die perfekte Aktion in
jeder Position, kann aber oft beschreiben, was gutes Verhalten einbringt:

- dem Ziel näher kommen;
- Gefahren vermeiden;
- schnell fertig werden;
- am Leben bleiben;
- nützliche Objekte einsammeln.

Die Belohnung ist nicht das finale Verhalten. Sie ist das Trainingssignal, das die Policy nutzt,
um Verhalten zu entdecken.

!!! info "Die Belohnungshypothese"
    RL basiert auf einer kühnen Idee: Ziele lassen sich als Maximierung der erwarteten
    kumulativen Belohnung ausdrücken. „Sanft landen" wird zu Zahlen für sichere Geschwindigkeit, aufrechten
    Winkel, Beinkontakt und keinen Absturz.

---

## 2 · Beobachtung → Aktion → Belohnung → nächste Beobachtung

Bei jedem Trainingsschritt wiederholt sich dieselbe Schleife:

1. Godot sendet die aktuelle **Beobachtung** an Python.
2. Die Policy wählt eine **Aktion**.
3. Godot wendet diese Aktion in der Szene an.
4. Godot gibt eine **Belohnung** und die **nächste Beobachtung** zurück.

<div class="diagram-scroll">

<svg class="course-diagram" viewBox="0 0 660 200" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="RL loop: agent and environment exchange actions, observations, and rewards">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 1 L10 5 L0 9 z" fill="#8892b0"/>
    </marker>
  </defs>
  <rect x="60" y="70" width="200" height="64" rx="12" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="160" y="96" text-anchor="middle" fill="#e2e8f0" font-size="18" font-weight="700">POLICY</text>
  <text x="160" y="116" text-anchor="middle" fill="#8892b0" font-size="14">the network</text>
  <rect x="400" y="70" width="200" height="64" rx="12" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="500" y="96" text-anchor="middle" fill="#e2e8f0" font-size="18" font-weight="700">GODOT</text>
  <text x="500" y="116" text-anchor="middle" fill="#8892b0" font-size="14">the environment</text>
  <path d="M260 84 C320 84, 340 84, 400 84" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#ar)"/>
  <text x="330" y="74" text-anchor="middle" fill="#6c8ef7" font-size="14" font-weight="700">action</text>
  <path d="M400 120 C340 120, 320 120, 260 120" fill="none" stroke="#8892b0" stroke-width="1.6" marker-end="url(#ar)"/>
  <text x="330" y="148" text-anchor="middle" fill="#4ecca3" font-size="14" font-weight="700">observation + reward</text>
</svg>

</div>

**Beobachtung versus Zustand**

- **Zustand** bedeutet die vollständige Wahrheit der Welt.
- **Beobachtung** bedeutet die Zahlen, die der Agent tatsächlich erhält.

In den meisten Spielen erhält der Agent Beobachtungen, nicht den vollständigen Zustand. Ein Lander kennt
vielleicht seine Geschwindigkeit und seinen Winkel, aber nicht jeden internen Physikwert. Ein Rennfahrer kennt
vielleicht Raycast-Abstände und Heading-Fehler, aber nicht die gesamte Streckenkarte.

**Aktionsraum**

| Typ | Bedeutung | Beispiel |
|------|---------|---------|
| Diskret | Aus einer festen Liste wählen | linkes Triebwerk, Haupttriebwerk oder nichts feuern |
| Kontinuierlich | Werte aus einem Bereich wählen | Lenkung und Gas zwischen -1 und 1 |

Der Aktionsraum bestimmt, welche Form die Policy-Ausgabe haben muss.

---

## 3 · Episoden, Return und Diskontierung

Eine **Episode** ist ein Versuch vom Reset bis zu einer Terminalbedingung. Eine Lander-Episode
endet, wenn er landet, abstürzt oder die Zeit abläuft. Eine Renn-Episode könnte nach einer Runde,
einer Kollision oder einem Stillstand-Timeout enden.

Die **Belohnung** ist ein Schritt Feedback. Der **Return** ist die Summe zukünftiger
Belohnungen, die der Agent maximieren will.

Zukünftige Belohnungen werden üblicherweise mit einem Wert namens Gamma (`γ`) **diskontiert**:

- `γ` nahe 1 bedeutet, dass der Agent stark auf spätere Ergebnisse achtet;
- ein niedrigeres `γ` bedeutet, dass der Agent sich mehr auf unmittelbare Belohnung konzentriert.

!!! tip "Die Maus-, Käse- und Katzen-Intuition"
    Käse in der Nähe der Katze kann wertvoll sein, aber er ist riskant und weit weg. Diskontierung
    erfasst die Idee, dass eine zukünftige Belohnung, die du vielleicht nie erreichst, weniger wert ist
    als eine Belohnung, die du jetzt zuverlässig bekommen kannst.

Alles in diesem Kurs beginnt als episodische Aufgabe. Das macht Lernkurven
leichter lesbar, weil jeder Lauf einen klaren Anfang und ein klares Ende hat.

---

## 4 · Die Policy ist das Netzwerk, das du gebaut hast

Die **Policy** ist die Entscheidungsfunktion des Agenten. Sie bildet eine Beobachtung auf eine
Aktion oder auf eine Wahrscheinlichkeitsverteilung über Aktionen ab.

In Neuronale Grundlagen 2 hat dein Netzwerk gelernt:

```text
inputs → hidden activations → outputs
```

In RL werden diese Ausgaben zu Aktionen:

```text
observation → policy network → action
```

Training ändert die Policy-Gewichte. Inference führt nur den Forward Pass aus. Diese
gleiche Trennung wird später wichtig, wenn eine Policy in Python trainiert und exportiert wird,
um in Godot zu laufen.

!!! info "Policy in einem Satz"
    Eine Policy ist das trainierte Verhalten des Agenten, gespeichert als Netzwerkgewichte.

---

## 5 · Exploration in einem Bild

Bevor ein Agent ein gutes Verhalten nutzen kann, muss er eines entdecken. Das ist
**Exploration**.

Der Kompromiss ist einfach:

- **exploit** — Aktionen nutzen, die bereits gut aussehen;
- **explore** — Aktionen ausprobieren, die etwas Besseres offenbaren könnten.

Früh im Training ist zufällig wirkende Bewegung normal. Die Policy sammelt
Erfahrung. Später sollte die Bewegung konsistenter werden, während die Belohnung das
Netzwerk zu besseren Aktionen drängt.

!!! warning "Bewerte eine Policy nicht nach einer frühen Episode"
    Eine nützliche Policy wirkt anfangs oft töricht. Beobachte Trends über viele
    Episoden hinweg, besonders die durchschnittliche Return-Kurve, nicht einen einzelnen glücklichen oder unglücklichen Lauf.

Du wirst detaillierte Explorationsmechanismen im Deep Dive nach Grundlagen
3 studieren. Vorerst lautet die praktische Frage: **Probiert der Agent genug Aktionen aus,
um belohntes Verhalten zu finden?**

---

## 6 · Godot und Python während des Trainings

Godot RL Agents führt zwei Programme nebeneinander aus:

- **Godot** ist die Umgebung: Physik, Beobachtungen, Aktionen, Belohnungen, Resets;
- **Python** ist der Trainer: er führt den RL-Algorithmus aus und aktualisiert die Policy.

<div class="diagram-scroll">

<svg class="course-diagram" viewBox="0 0 720 330" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" role="img" aria-label="Godot environment and Python training process connected by socket">
  <defs>
    <marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 1 L10 5 L0 9 z" fill="#8892b0"/>
    </marker>
  </defs>
  <rect x="20" y="30" width="290" height="210" rx="14" fill="#1a1d27" stroke="#4ecca3" stroke-width="1.5"/>
  <text x="165" y="58" text-anchor="middle" fill="#e2e8f0" font-size="16" font-weight="700">Godot</text>
  <text x="165" y="78" text-anchor="middle" fill="#8892b0" font-size="13">environment</text>
  <rect x="40" y="105" width="250" height="50" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="165" y="135" text-anchor="middle" fill="#e2e8f0" font-size="14">scene + AIController</text>
  <rect x="40" y="170" width="250" height="50" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="165" y="200" text-anchor="middle" fill="#e2e8f0" font-size="14">reward + reset logic</text>
  <rect x="410" y="30" width="290" height="210" rx="14" fill="#1a1d27" stroke="#6c8ef7" stroke-width="1.5"/>
  <text x="555" y="58" text-anchor="middle" fill="#e2e8f0" font-size="16" font-weight="700">Python</text>
  <text x="555" y="78" text-anchor="middle" fill="#8892b0" font-size="13">trainer</text>
  <rect x="430" y="105" width="250" height="50" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="555" y="135" text-anchor="middle" fill="#e2e8f0" font-size="14">godot-rl wrapper</text>
  <rect x="430" y="170" width="250" height="50" rx="8" fill="#22263a" stroke="#2e3350"/>
  <text x="555" y="200" text-anchor="middle" fill="#e2e8f0" font-size="14">Stable-Baselines3 PPO</text>
  <path d="M310 120 L410 120" fill="none" stroke="#8892b0" stroke-width="1.8" marker-end="url(#ar2)"/>
  <text x="360" y="106" text-anchor="middle" fill="#4ecca3" font-size="13" font-weight="700">obs + reward</text>
  <path d="M410 190 L310 190" fill="none" stroke="#8892b0" stroke-width="1.8" marker-end="url(#ar2)"/>
  <text x="360" y="208" text-anchor="middle" fill="#6c8ef7" font-size="13" font-weight="700">actions</text>
  <text x="360" y="260" text-anchor="middle" fill="#8892b0" font-size="13">local socket, usually port 11008</text>
</svg>

</div>

| RL-Konzept | Wo es in Godot RL lebt |
|------------|----------------------------|
| Environment | Deine Godot-Szene |
| Observation | `get_obs()` im AIController-Skript |
| Action space | `get_action_space()` im AIController-Skript |
| Reward | Belohnungsvariablen, die deine Spiellogik aktualisiert |
| Episode end | `done`, `needs_reset` oder entsprechende Reset-Flags |
| Policy | Das in Python trainierte neuronale Netz |

---

## 7 · Quick win — eine Belohnung ändern

!!! tip "Dein erster Eigentümermoment"
    In Unit 0 hast du BallChase ausgeführt. Hier änderst du das Belohnungssignal und beobachtest,
    wie sich das Trainingsverhalten anpasst.

1. Klone oder öffne [BallChase](https://github.com/edbeeching/godot_rl_agents_examples/tree/main/examples/BallChase) in Godot.
2. Finde das Skript, das `reward` für den Agenten aktualisiert.
3. Ändere einen Term, z. B. verdopple die Belohnung dafür, dem Ball näher zu kommen,
   oder füge eine kleine Strafe pro Schritt hinzu.
4. Sage das Ergebnis vor dem Training voraus: schnelleres Verfolgen, mehr Umherwandern, kürzere
   Episoden oder langsameres Lernen.

Führe eine kurze visuelle Trainingssitzung aus:

```bash
conda activate godot_env
python examples/stable_baselines3_example.py \
  --env_path=examples/godot_rl_BallChase/bin/BallChase.x86_64 \
  --experiment_name=unit1-reward-tweak --timesteps=100000 --viz
```

Beobachte drei Ansichten:

- **Godot:** Entspricht die Bewegung deiner Belohnungsänderung?
- **TensorBoard:** Steigt `ep_rew_mean` nach genügend Episoden nach oben?
- **Code:** Kannst du erklären, wie die bearbeitete Belohnung die Schleife verändert?

---

## 8 · Fertig, wenn

Du bist bereit weiterzumachen, wenn du:

- Beobachtung → Aktion → Belohnung → nächste Beobachtung ohne Notizen erklären kannst;
- beschreiben kannst, warum die Policy dieselbe Art von Netzwerk ist, die du in Grundlagen 2 gebaut hast;
- Godots Rolle und Pythons Rolle während des Trainings benennen kannst;
- eine BallChase-Belohnungsänderung vornimmst und das wahrscheinliche Symptom vorhersagst;
- eine Belohnungskurve als Trend liest, nicht als eine einzelne Episode.

!!! warning "Training kommt nicht voran?"
    Prüfe der Reihe nach: Vorzeichen und Skalierung der Belohnung, sparse Rewards, Beobachtungsfehler,
    Resets und ob der Lauf einfach mehr Episoden braucht.

---

## 9 · Stretch Goals

- Schreibe eine Random-Policy-Schleife für `CartPole-v1` und gib den Episoden-Return aus.
- Führe zwei BallChase-Belohnungsanpassungen durch: eine stärkere und eine schwächere Belohnung.
  Vergleiche die Lernkurven.
- Skizziere den Beobachtungsvektor und den Aktionsraum für eine eigene Spielidee.

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, _ = env.reset()
total_reward = 0.0

for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"Episode return: {total_reward}")
env.close()
```

---

## Was kommt als Nächstes

Du hast jetzt das operative RL-Vokabular: Beobachtungen, Aktionen, Belohnungen,
Episoden, Returns, Policies, Exploration und die Godot/Python-Trainingsschleife.
Als Nächstes macht Grundlagen 3 aus diesen Bausteinen ein kleines Belohnungs-Lernprojekt, damit
du eine Policy aus Trajektorien verbessern siehst.

!!! info "Selbstcheck, bevor du weitermachst"
    1. Was sind die vier Teile der RL-Schleife?
    2. Was ist der Unterschied zwischen einer Beobachtung und einem Zustand?
    3. Was steuert der Abzinsungsfaktor `γ`?
    4. Was ist eine Policy?
    5. Warum braucht ein Agent Exploration?
    6. Was macht Python während des Godot-RL-Trainings?
    7. Was macht Godot während des Godot-RL-Trainings?

??? success "Antworten zum Selbstcheck"
    1. Beobachtung → Aktion → Belohnung → nächste Beobachtung.
    2. Ein Zustand ist die vollständige Weltbeschreibung; eine Beobachtung ist der Teil, den der
       Agent erhält.
    3. `γ` steuert, wie stark zukünftige Belohnungen im Vergleich zu unmittelbaren
       Belohnungen zählen.
    4. Eine Policy ist die Entscheidungsfunktion, meist ein neuronales Netz, das
       Beobachtungen auf Aktionen abbildet.
    5. Exploration ermöglicht dem Agenten, Verhalten zu entdecken, von dem er noch nicht weiß,
       dass es nützlich ist.
    6. Python führt den Trainingsalgorithmus aus und aktualisiert die Policy-Gewichte.
    7. Godot simuliert die Umgebung, wendet Aktionen an, berechnet Belohnungen und
       setzt Episoden zurück.

[← Neuronale Grundlagen 2](unit-neural-02.md) · [Kursstartseite](index.md) · [→ Neuronale Grundlagen 3](unit-neural-03.md)
