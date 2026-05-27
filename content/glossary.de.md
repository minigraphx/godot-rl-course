# Glossar & Spickzettel

[Kursstartseite](index.md)

Schnellnachschlag für alle Abkürzungen, Gleichungen und Algorithmus-Parameter dieses Kurses.

---

## Teil 1 — Glossar (A–Z)

**A2C (Advantage Actor-Critic)** — Synchrone Variante von A3C, die mehrere parallele Umgebungen ausführt und Actor und Critic nach jedem Rollout aktualisiert. *Siehe:* [unit-actor-critic](unit-actor-critic.md) §3

**Actor (Akteur)** — Das neuronale Netz, das eine Wahrscheinlichkeitsverteilung (oder eine deterministische Aktion) über den Aktionsraum ausgibt; die „Policy"-Hälfte der Actor-Critic-Methoden. *Siehe:* [unit-actor-critic](unit-actor-critic.md) §1

**Advantage (A, Vorteil)** — Wie viel besser eine Aktion gegenüber dem Durchschnitt ist: `A(s,a) = Q(s,a) − V(s)`. Positiver Advantage → Aktion ist besser als erwartet; negativer → schlechter. *Siehe:* [unit-actor-critic](unit-actor-critic.md) §2

**Agent** — Die Entität, die die Umgebung beobachtet, Aktionen wählt und Belohnungen erhält. In diesem Kurs ein Godot-Node, der an einen `AIController` angebunden ist. *Siehe:* [unit-00](unit-00.md) §1

**Alpha (α in SAC)** — Entropie-Temperaturkoeffizient in SAC; steuert, wie stark der Agent für zufälliges Handeln belohnt wird. Kann fest oder automatisch gelernt sein. *Siehe:* [unit-sac](unit-sac.md) §3

**Atari** — Sammlung von Atari-2600-Spielumgebungen (über ALE/Gymnasium), kanonische Benchmark für Deep RL mit diskreten Aktionen; DQN wurde hier zuerst demonstriert. *Siehe:* [unit-q-learning](unit-q-learning.md) §4

**Baseline** — Ein Referenzwert, der vom Return abgezogen wird, um die Varianz im Policy-Gradient zu senken, ohne den erwarteten Gradienten zu ändern. Die Wertfunktion V(s) ist die Standard-Baseline. *Siehe:* [unit-policy-gradients](unit-policy-gradients.md) §3

**Batch Normalization** — Normalisiert Layer-Aktivierungen über das Mini-Batch bei jedem Gradientenschritt; stabilisiert Training, verträgt sich aber schlecht mit rekurrenten Architekturen und kleinen Batches, wie sie im RL üblich sind. *Siehe:* [unit-ppo-deep](unit-ppo-deep.md) §5

**Behavioral Cloning (BC)** — Überwachtes Imitationslernen: aus einem Datensatz von Experten-`(state, action)`-Paaren wird eine Policy via Cross-Entropy- oder MSE-Loss trainiert. Einfach und schnell, leidet aber unter Distribution Shift, wenn der Agent Zustände erreicht, die der Experte nie zeigte. *Siehe:* [unit-09](unit-09.md) §1

**Bellman-Gleichung** — Rekursive Definition der Wertfunktion: `V(s) = E[r + γ V(s')]`. Das Rückgrat aller wertbasierten RL-Algorithmen. *Siehe:* [unit-q-learning](unit-q-learning.md) §2

**Beta (β in PBT/PER)** — In PER steuert es, wie stark Importance Sampling den Bias durch nicht-uniformes Sampling korrigiert (während des Trainings 0→1 angekühlt). In PBT manchmal als Schedule-Parameter genutzt. *Siehe:* [unit-pbt](unit-pbt.md) §2

**Buffer (Replay)** — Ein FIFO-Speicher fester Größe, der `(s, a, r, s', done)`-Transitionen ablegt; Off-Policy-Algorithmen ziehen daraus Mini-Batches, um zeitliche Korrelation zu brechen. *Siehe:* [unit-q-learning](unit-q-learning.md) §5

**Clipping (PPO)** — Die `clip(ratio, 1−ε, 1+ε)`-Operation im PPO-Objektiv, die verhindert, dass die aktualisierte Policy in einem Gradientenschritt zu weit von der Behavior-Policy abweicht. *Siehe:* [unit-ppo-deep](unit-ppo-deep.md) §2

**CMDP (Constrained MDP)** — Ein MDP, erweitert um Nebenbedingungs-Funktionen `C_i(s,a)`, die unter Schwellen bleiben müssen; das Framework hinter Safe RL. *Siehe:* [unit-safe-rl](unit-safe-rl.md) §1

**CNN (Convolutional Neural Network)** — Feature-Extraktor für gitterstrukturierte Eingaben (Bilder, Tilemaps); nutzt gewichtsgeteilte Faltungs-Filter vor einem flachen Feature-Vektor. *Siehe:* [unit-visual-observations](unit-visual-observations.md) §2

**Conjugate Gradient** — Iterativer Löser in TRPO, um den Natural-Gradient-Schritt zu berechnen, ohne die volle Fisher-Informationsmatrix aufzubauen — bleibt damit rechnerisch handhabbar. *Siehe:* [unit-ppo-deep](unit-ppo-deep.md) §1

**Credit Assignment** — Das Problem zu bestimmen, welche vergangenen Aktionen für eine verzögerte Belohnung verantwortlich sind. Temporal-Difference-Methoden (TD, Q-Learning) lösen es per Bootstrapping; Eligibility Traces und n-Step-Returns erweitern das Zuordnungsfenster. Lange Horizonte mit spärlichen Belohnungen machen Credit Assignment besonders hart. *Siehe:* [unit-q-learning](unit-q-learning.md) §2

**Critic (Kritiker)** — Das neuronale Netz, das V(s) oder Q(s,a) schätzt; liefert die Baseline oder das Ziel für den Actor-Gradienten. *Siehe:* [unit-actor-critic](unit-actor-critic.md) §1

**Curriculum Learning** — Schrittweise Erhöhung der Aufgabenschwierigkeit während des Trainings (leicht → schwer), sodass der Agent immer am Rand seiner aktuellen Fähigkeit trainiert. *Siehe:* [unit-multitask](unit-multitask.md) §3

**D4RL** — Benchmark-Sammlung für Offline RL (Deep Data-Driven RL); fixierte Transitions-Datensätze aus Domänen wie HalfCheetah, AntMaze und Kitchen. *Siehe:* [unit-offline-rl](unit-offline-rl.md) §2

**Dead Neurons (tote Neuronen)** — Neuronen, die auf Null-Ausgabe festhängen (meist ReLU-Neuronen, die nie aktivieren); können die Netzkapazität still kollabieren, vor allem nach großen Gradientenschritten. *Siehe:* [unit-debugging](unit-debugging.md) §4

**Deterministische Policy** — Eine Policy, die jedem Zustand eine einzelne Aktion `a = μ(s)` zuordnet, nicht eine Verteilung; genutzt von DDPG und TD3. *Siehe:* [unit-sac](unit-sac.md) §1

**DDPG (Deep Deterministic Policy Gradient)** — Off-Policy-Actor-Critic für kontinuierliche Aktionen mit deterministischer Policy und Experience Replay; Vorläufer von TD3 und SAC. *Siehe:* [unit-sac](unit-sac.md) §2

**DPO (Direct Preference Optimization)** — Alignment-Technik, die eine Policy direkt aus menschlichen Präferenzpaaren trainiert, ohne separates Reward-Modell; verwandt mit RLHF. *Siehe:* [unit-reward-engineering](unit-reward-engineering.md) §5

**DQN (Deep Q-Network)** — Kombiniert Q-Learning mit einem tiefen neuronalen Netz, Experience Replay und einem Target-Netz, um das Training auf hochdimensionalen Eingaben zu stabilisieren. *Siehe:* [unit-03](unit-03.md) §1

**Discount Factor (γ, Diskontierungsfaktor)** — Skalar in [0, 1), der zukünftige Belohnungen im Return `G_t = Σ γ^k r_{t+k}` herunterwichtet. γ=0 macht den Agenten kurzsichtig (nur direkte Belohnung); γ→1 weitsichtig. Typische Werte: 0,99 für episodische Aufgaben, 0,999 für kontinuierliche Aufgaben mit langem Horizont. *Siehe:* [unit-00](unit-00.md) §2

**Distributional RL** — Familie von RL-Algorithmen, die die volle Verteilung der Returns Z(s,a) modellieren statt nur ihren Erwartungswert Q(s,a). C51 (das Original) repräsentiert Z als kategoriale Verteilung über 51 Atome. Verbessert Stabilität, weil der Agent lernt, *wie variabel* ein Ergebnis ist, nicht nur dessen Mittel. Wird in Rainbow DQN genutzt. *Siehe:* [unit-03](unit-03.md) §7

**Dreamer / DreamerV3** — Modellbasierte RL-Agenten, die ein kompaktes Weltmodell im latenten Raum lernen und Rollouts vollständig in diesem Modell „imaginieren"; State of the Art auf vielen Benchmarks. *Siehe:* [unit-world-models](unit-world-models.md) §4

**Dueling DQN** — DQN-Variante, die Q in getrennte Value- und Advantage-Streams zerlegt: `Q(s,a) = V(s) + A(s,a) − mean(A)`; verbessert Stabilität, wenn viele Aktionen ähnlich wertvoll sind. *Siehe:* [unit-03](unit-03.md) §4

**Entropie (Policy-Entropie)** — `H(π) = −Σ π(a|s) log π(a|s)`; misst, wie zufällig eine Policy ist. SAC maximiert Entropie explizit; PPO nutzt sie als Regularisierer über `ent_coef`. *Siehe:* [unit-sac](unit-sac.md) §3

**Episode** — Eine vollständige Folge von Transitionen vom Umgebungsreset bis zur Terminierung (done=True). Der episodische Return G ist die Summe der Belohnungen einer Episode. *Siehe:* [unit-00](unit-00.md) §2

**ε-greedy** — Erkundungsstrategie, die mit Wahrscheinlichkeit ε eine zufällige Aktion und sonst die gierige (greedy) Aktion wählt; ε wird typisch von 1,0 auf 0,05 während des Trainings angekühlt. *Siehe:* [unit-q-learning](unit-q-learning.md) §3

**Experience Replay** — Speichern vergangener Transitionen in einem Buffer und Ziehen zufälliger Mini-Batches zum Training; bricht zeitliche Korrelationen und erlaubt Datennutzung in Off-Policy-Algorithmen. *Siehe:* [unit-q-learning](unit-q-learning.md) §5

**Feature Extractor** — Das Eingangsnetz (MLP, CNN, eigener GDScript-Sensor → Python), das rohe Beobachtungen in ein Embedding fester Größe für Policy/Value-Köpfe konvertiert. *Siehe:* [unit-visual-observations](unit-visual-observations.md) §2

**Function Approximation (Funktionsapproximation)** — Nutzung eines parametrischen Modells (z. B. eines neuronalen Netzes), um V(s), Q(s,a) oder π(a|s) darzustellen, wenn der Zustandsraum zu groß für eine Tabelle ist. Deep RL ist Funktionsapproximation mit tiefen Netzen; lineare Funktionsapproximation mit Tile-Coding oder RBF-Features ist älter. *Siehe:* [unit-q-learning](unit-q-learning.md) §3

**FlyBy** — Eine der eingebauten godot-rl-agents-Beispielumgebungen; eine Drohne navigiert einen 3-D-Hindernisparcours, häufig genutzt für visuelle Beobachtungen und 3-D-Lokomotion. *Siehe:* [unit-locomotion](unit-locomotion.md) §2

**GAE (Generalized Advantage Estimation)** — Exponentiell gewichtete Summe von TD-Fehlern zur Schätzung des Advantage, gesteuert über λ ∈ [0,1]: λ=0 gibt 1-Step-TD; λ=1 ergibt vollständige Monte-Carlo-Returns. *Siehe:* [unit-ppo-deep](unit-ppo-deep.md) §3

**GDScript** — Pythonähnliche Skriptsprache, in Godot eingebaut; im Kurs für `AIController`, Beobachtungssammlung, Belohnungsfunktion und Szenenlogik genutzt. *Siehe:* [unit-00](unit-00.md) §1

**Goal-conditioned RL** — RL-Formulierung, bei der das Ziel g Teil der Beobachtung ist; der Agent lernt eine einzelne Policy `π(a | s, g)`, die über viele Ziele generalisiert. *Siehe:* [unit-her](unit-her.md) §1

**HER (Hindsight Experience Replay)** — Off-Policy-Trick, der fehlgeschlagene Episoden mit Ersatz-Zielen wiedergibt, die dem entsprechen, was der Agent tatsächlich erreichte — verwandelt Fehlschläge in Lernsignal für sparsame-Belohnungs-Manipulationsaufgaben. *Siehe:* [unit-her](unit-her.md) §2

**Hierarchical RL** — Zerlegt eine Aufgabe in High-Level-Subziel-Wahl (Manager) und Low-Level-Primitiv-Ausführung (Worker); ermöglicht Langzeit-Planung ohne Credit Assignment über Tausende Schritte. *Siehe:* [unit-hierarchical](unit-hierarchical.md) §1

**ICM (Intrinsic Curiosity Module)** — Fügt einen Vorhersagefehler zwischen dem vom Forward-Modell prognostizierten Next-State-Embedding und dem tatsächlichen als intrinsischen Belohnungsbonus hinzu, der Erkundung antreibt. *Siehe:* [unit-curiosity](unit-curiosity.md) §3

**Imitation Learning (Imitationslernen)** — Lernen einer Policy durch Beobachtung eines Experten, ohne primär auf ein Umgebungs-Belohnungssignal zu setzen. Umfasst Behavioral Cloning (überwacht), DAgger (iterativ), GAIL (adversarial) und IRL (Reward-Rückgewinnung). *Siehe:* [unit-09](unit-09.md) §1

**Importance Sampling** — Technik, mit der Transitionen einer alten Policy (Behavior-Policy) zur Gradientenschätzung unter der aktuellen Policy wiederverwendet werden, über das Verhältnis `π_θ(a|s) / π_θ_old(a|s)`. *Siehe:* [unit-ppo-deep](unit-ppo-deep.md) §2

**Intrinsische Motivation** — Belohnungssignal, das der Agent intern erzeugt (Neugier, Neuheit, Empowerment), nicht aus der Umgebung; genutzt zur Erkundung bei spärlichen Belohnungen. *Siehe:* [unit-curiosity](unit-curiosity.md) §1

**IRL (Inverse RL)** — Lernen einer Belohnungsfunktion aus Experten-Demonstrationen (die Umkehrung von RL); Grundlage für Imitations- und Alignment-Methoden. *Siehe:* [unit-reward-engineering](unit-reward-engineering.md) §4

**KL-Divergenz** — `KL(p‖q) = Σ p(x) log(p(x)/q(x))`; misst, wie sehr Verteilung q sich von der Referenz p unterscheidet; in TRPO harte Nebenbedingung, in PPO Monitoring-Metrik. *Siehe:* [unit-ppo-deep](unit-ppo-deep.md) §2

**LayerNorm** — Normalisiert Aktivierungen über Feature-Dimensionen (nicht über das Batch); stabiler als BatchNorm bei kleinen Batches im RL; Standard in vielen modernen Actor-Critic-Implementierungen. *Siehe:* [unit-ppo-deep](unit-ppo-deep.md) §5

**Lernrate (learning rate)** — Schrittweite für das Gradientenupdate `θ ← θ − α ∇L`; einer der wirkungsstärksten Hyperparameter. Typischer SB3-Bereich: `1e-4` bis `3e-4`. *Siehe:* [unit-ppo-deep](unit-ppo-deep.md) §6

**Locomotion (Fortbewegung)** — Aufgabenkategorie, in der der Agent einen Körper effizient bewegen muss (gehen, laufen, springen, fliegen); kontinuierlich, dichte Belohnung, physikalisch simuliert. *Siehe:* [unit-locomotion](unit-locomotion.md) §1

**MAML (Model-Agnostic Meta-Learning)** — Meta-Learning-Algorithmus, der eine Initialisierung lernt, von der aus eine neue Aufgabe in wenigen Gradientenschritten lösbar ist; eingesetzt im Multi-Task- und Sim-to-Real-Transfer. *Siehe:* [unit-multitask](unit-multitask.md) §4

**MDP (Markov Decision Process)** — Formaler Rahmen für RL: Tupel (S, A, P, R, γ) — S=Zustände, A=Aktionen, P=Übergangswahrscheinlichkeiten, R=Belohnungsfunktion, γ=Diskontierungsfaktor. *Siehe:* [unit-00](unit-00.md) §2

**MuJoCo** — Physik-Engine (jetzt frei dank DeepMind), weit verbreitet für kontinuierliche-Kontroll-Benchmarks (HalfCheetah, Ant, Humanoid); in Gymnasium als `gymnasium[mujoco]` integriert. *Siehe:* [unit-locomotion](unit-locomotion.md) §3

**n_steps** — Anzahl Umgebungsschritte pro Rollout pro paralleler Umgebung vor einem Gradientenupdate; Schlüssel-Hyperparameter in PPO; trägt den Bias-Varianz-Trade-off der Advantage-Schätzung. *Siehe:* [unit-ppo-deep](unit-ppo-deep.md) §6

**NatureCNN** — CNN-Architektur aus dem ursprünglichen DQN-Nature-Paper (3 Conv-Layer → flach → 512-dim FC); SB3-Default-Feature-Extractor für Bildbeobachtungen (`CnnPolicy`). *Siehe:* [unit-visual-observations](unit-visual-observations.md) §3

**Noisy Networks** — Ersetzt feste Gewichte in den FC-Layern eines DQN durch gelernte Gauß-Rausch-Parameter, um zustandsabhängige Erkundung ohne ε-greedy zu erreichen. *Siehe:* [unit-03](unit-03.md) §5

**Observation (Beobachtung)** — Die Information, die der Agent pro Schritt von der Umgebung erhält (Sensorwerte, Positionen, Geschwindigkeiten, Bildpixel etc.); je nach Umgebung partiell oder vollständig. *Siehe:* [unit-00](unit-00.md) §3

**Off-Policy** — Lernstil, bei dem die zu verbessernde Policy (Target-Policy) sich von der Datenerzeugungs-Policy (Behavior-Policy) unterscheidet; ermöglicht Replay-Buffer-Nutzung. SAC, DQN, TD3 sind Off-Policy. *Siehe:* [unit-sac](unit-sac.md) §1

**On-Policy** — Lernstil, bei dem jedes Gradientenupdate nur frische Daten der aktuellen Policy nutzt und dann verwirft. PPO, A2C, REINFORCE sind On-Policy. *Siehe:* [unit-policy-gradients](unit-policy-gradients.md) §1

**ONNX (Open Neural Network Exchange)** — Modell-Serialisierungsformat zum Export trainierter SB3-Policies für die Inferenz in Godots ONNX-Bridge ohne Python-Runtime. *Siehe:* [unit-10](unit-10.md) §2

**PBT (Population-Based Training)** — Automatisierte Hyperparameter-Suche, die eine Agenten-Population parallel laufen lässt und periodisch Gewichte besserer Agenten auf schlechtere kopiert und deren Hyperparameter stört. *Siehe:* [unit-pbt](unit-pbt.md) §1

**PER (Prioritized Experience Replay)** — Replay-Buffer-Variante, die Transitionen proportional zu ihrem TD-Fehler zieht (hoher Fehler → häufiger), korrigiert mit Importance-Sampling-Gewichten. *Siehe:* [unit-03](unit-03.md) §6

**Policy** — Eine Abbildung von Zuständen auf Aktionen, entweder stochastisch `π(a|s)` oder deterministisch `a=μ(s)`; das primäre Objekt, das RL-Algorithmen optimieren. *Siehe:* [unit-00](unit-00.md) §2

**Policy Gradient** — Familie von Algorithmen, die den erwarteten Return direkt maximieren, indem sie dem Gradienten `∇_θ J(θ) = E[∇_θ log π_θ(a|s) · A(s,a)]` folgen. *Siehe:* [unit-policy-gradients](unit-policy-gradients.md) §2

**PPO (Proximal Policy Optimization)** — On-Policy-Actor-Critic-Algorithmus, der das Wahrscheinlichkeitsverhältnis klippt, um zerstörerisch große Updates zu verhindern; Standard in godot-rl-agents. *Siehe:* [unit-ppo-deep](unit-ppo-deep.md) §2

**Q-Wert** — Aktions-Wertfunktion `Q(s,a)` = erwarteter diskontierter Return, wenn man in s Aktion a wählt und danach Policy π folgt. *Siehe:* [unit-q-learning](unit-q-learning.md) §1

**Rainbow DQN** — DQN-Variante, die sechs Verbesserungen (PER, Dueling, Noisy Nets, n-Step-Returns, Distributional RL, Double DQN) zu einem State-of-the-Art-Atari-Agenten kombiniert. *Siehe:* [unit-03](unit-03.md) §7

**Rekurrenz (LSTM / GRU)** — Architekturen, die einen Hidden State über Zeitschritte halten und der Policy Gedächtnis geben. Unverzichtbar für POMDPs (teilweise beobachtbare Umgebungen), in denen die aktuelle Beobachtung den Zustand nicht festlegt. LSTM trennt Cell- und Hidden-State mit Forget/Input/Output-Gates; GRU führt sie in Update/Reset-Gates zusammen — weniger Parameter. *Siehe:* [unit-08](unit-08.md) §2

**REINFORCE** — Monte-Carlo-Policy-Gradient: vollständige Episoden sammeln, Returns G_t berechnen, dann `θ ← θ + α ∇_θ log π_θ(a_t|s_t) · G_t`; hohe Varianz, aber konzeptionell einfach. *Siehe:* [unit-policy-gradients](unit-policy-gradients.md) §2

**Reparametrisierungs-Trick** — Drückt eine stochastische Stichprobe `a ~ π(·|s)` als deterministische Funktion einer Rauschvariablen `ε ~ N(0,1)` aus: `a = μ(s) + σ(s) · ε`; lässt Gradienten durch die Stichprobenoperation fließen. Wird in SAC genutzt. *Siehe:* [unit-sac](unit-sac.md) §4

**Replay-Buffer** — Siehe *Buffer (Replay)*. *Siehe:* [unit-q-learning](unit-q-learning.md) §5

**Return (G)** — Diskontierte Summe zukünftiger Belohnungen ab Schritt t: `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …`; das, was RL-Algorithmen letztlich maximieren. *Siehe:* [unit-00](unit-00.md) §2

**RLHF (Reinforcement Learning from Human Feedback)** — Trainingsparadigma, das ein Belohnungsmodell aus menschlichen Präferenzlabels lernt und dann eine Policy per RL feinabstimmt; eingesetzt für LLM-Alignment (ChatGPT, Claude). *Siehe:* [unit-reward-engineering](unit-reward-engineering.md) §5

**RND (Random Network Distillation)** — Intrinsische Neugier-Methode: der Agent trainiert ein Predictor-Netz, um ein festes zufälliges Target-Netz zu treffen; oft besuchte Zustände haben geringen Fehler, sodass neue Zustände eine höhere intrinsische Belohnung liefern. *Siehe:* [unit-curiosity](unit-curiosity.md) §4

**Rollout** — Eine vollständige Datenerfassungsphase: die aktuelle Policy interagiert für `n_steps` Schritte über alle parallelen Umgebungen mit der Umwelt und erzeugt einen Batch von Transitionen für das Gradientenupdate. *Siehe:* [unit-ppo-deep](unit-ppo-deep.md) §3

**SAC (Soft Actor-Critic)** — Off-Policy-Actor-Critic, der ein kombiniertes Ziel aus erwartetem Return und Policy-Entropie maximiert; von Natur aus explorativ und robust gegen Hyperparameter; dominant in der kontinuierlichen Kontrolle. *Siehe:* [unit-sac](unit-sac.md) §2

**SB3 (Stable-Baselines3)** — Python-Bibliothek mit sauberen, getesteten Implementierungen von PPO, SAC, DQN, A2C, DDPG, TD3, HER und mehr; das Trainings-Backend dieses Kurses. *Siehe:* [unit-01](unit-01.md) §2

**Score-Function-Estimator** — Anderer Name für REINFORCE bzw. den Log-Derivative-Trick: `∇_θ E[f(x)] = E[f(x) ∇_θ log p_θ(x)]`; mathematischer Kern der Policy-Gradient-Methoden. *Siehe:* [unit-policy-gradients](unit-policy-gradients.md) §2

**Self-Play** — Trainingsparadigma, bei dem Agenten gegen Kopien ihrer selbst (oder gegen einen Pool vergangener Versionen) spielen — automatisches Curriculum zunehmend stärkerer Gegner. *Siehe:* [unit-self-play](unit-self-play.md) §1

**Sim-to-Real-Transfer** — Die Herausforderung, eine in Simulation trainierte Policy auf einen echten Roboter zu übertragen, ohne die Leistung zu verlieren; angegangen über Domain Randomization, adaptive Policies und sorgfältige Sensorabstimmung. *Siehe:* [unit-sim-to-real](unit-sim-to-real.md) §1

**Spärliche Belohnung (Sparse Reward)** — Belohnungsstruktur, in der Belohnungen ungleich null extrem selten sind (z. B. +1 nur beim Aufgabenabschluss); macht Standard-RL ohne Exploration-Bonus oder Shaping ineffektiv. *Siehe:* [unit-curiosity](unit-curiosity.md) §1

**State (Zustand)** — Eine vollständige Beschreibung der Umgebung zu einem Zeitpunkt; in der Praxis erhalten Agenten oft eine partielle Beobachtung statt des wahren Zustands. *Siehe:* [unit-00](unit-00.md) §2

**Target-Netz** — Periodisch aktualisierte Kopie des Q-Netzes (oder Critic), deren Gewichte fest gehalten werden, während das Online-Netz dagegen trainiert; verhindert Rückkopplungen, die DQN destabilisieren. *Siehe:* [unit-03](unit-03.md) §2

**TD-Fehler (δ)** — Temporal-Difference-Fehler: `δ = r + γ V(s') − V(s)`; Differenz zwischen Bootstrap-Target und aktueller Wertschätzung; treibt Wertlernen *und* Policy-Gradienten. *Siehe:* [unit-q-learning](unit-q-learning.md) §2

**TD3 (Twin Delayed Deep Deterministic)** — Off-Policy-Algorithmus für kontinuierliche Aktionen, der DDPG mit Zwillings-Critic (Minimum-Q), verzögerten Policy-Updates und Ziel-Policy-Glättung stabilisiert. *Siehe:* [unit-sac](unit-sac.md) §2

**Temperatur (α)** — Siehe *Alpha (α in SAC)*. *Siehe:* [unit-sac](unit-sac.md) §3

**Timestep (Zeitschritt)** — Ein einzelner Umgebungsschritt: Agent erhält o_t, gibt a_t aus, Umgebung wechselt zu o_{t+1} und liefert r_t. Trainingsbudgets werden in Gesamt-Timesteps gemessen. *Siehe:* [unit-00](unit-00.md) §2

**Trajektorienoptimierung** — Planungsansatz, der direkt nach der Aktionsfolge `(a_0, a_1, …, a_T)` mit maximalem Return sucht, mit gradienten- oder sampling-basierten Methoden. Anders als RL setzt sie meist ein differenzierbares Umweltmodell voraus. *Siehe:* [unit-world-models](unit-world-models.md) §3

**TRPO (Trust Region Policy Optimization)** — On-Policy-Algorithmus, der jedes Policy-Update auf eine KL-Divergenz-Trust-Region beschränkt; theoretisch motiviert, rechnerisch teuer. PPO ist der praktische Nachfolger. *Siehe:* [unit-ppo-deep](unit-ppo-deep.md) §1

**Wertfunktion (V)** — `V(s)` = erwarteter diskontierter Return aus s unter π; im Actor-Critic vom Critic geschätzt; dient als Baseline zur Varianzreduktion. *Siehe:* [unit-actor-critic](unit-actor-critic.md) §2

**VecEnv** — SB3-Abstraktion für vektorisierte Umgebungen, die N parallele Umgebungs-Kopien in einem Prozess oder über Subprozesse betreibt und gebatchte Schritte für schnellere Datenerfassung liefert. *Siehe:* [unit-02](unit-02.md) §3

**VecNormalize** — SB3-Wrapper um VecEnv, der laufende Statistiken hält, um Beobachtungen und Belohnungen online zu normalisieren; essenziell für stabiles kontinuierliches Training. *Siehe:* [unit-sac](unit-sac.md) §6

**Weltmodell (World Model)** — Gelerntes neuronales Modell der Umweltdynamik: aus (s_t, a_t) wird (s_{t+1}, r_t) vorhergesagt. Erlaubt Planung, „Imagination" und dateneffizientes Lernen. *Siehe:* [unit-world-models](unit-world-models.md) §1

---

## Teil 2 — Algorithmus-Spickzettel

### PPO — Proximal Policy Optimization

| Feld | Detail |
|------|--------|
| **Zusammenfassung** | On-Policy-Actor-Critic, der das Policy-Ratio-Update klippt, um große Schritte zu verhindern |
| **On/Off-Policy** | On-Policy |
| **Aktionsraum** | Diskret & kontinuierlich |
| **Wichtige SB3-Hyperparameter** | `learning_rate=3e-4`, `n_steps=2048`, `batch_size=64`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.0`, `vf_coef=0.5` |
| **TensorBoard-Metriken** | `train/approx_kl` (< 0,02 halten), `train/clip_fraction` (< 0,1 halten), `train/entropy_loss`, `rollout/ep_rew_mean` |
| **SB3-Klasse** | `from stable_baselines3 import PPO` |
| **Am besten für** | Godot-Spielumgebungen, diskret oder kontinuierlich, parallele Envs |

---

### DQN — Deep Q-Network

| Feld | Detail |
|------|--------|
| **Zusammenfassung** | Off-Policy, wertbasiert, mit Replay-Buffer und Target-Netz für diskrete Aktionen |
| **On/Off-Policy** | Off-Policy |
| **Aktionsraum** | Nur diskret |
| **Wichtige SB3-Hyperparameter** | `learning_rate=1e-4`, `buffer_size=1_000_000`, `learning_starts=50_000`, `batch_size=32`, `tau=1.0`, `gamma=0.99`, `train_freq=4`, `target_update_interval=10_000`, `exploration_fraction=0.1`, `exploration_final_eps=0.05` |
| **TensorBoard-Metriken** | `train/loss`, `rollout/ep_rew_mean`, `train/exploration_rate` |
| **SB3-Klasse** | `from stable_baselines3 import DQN` |
| **Am besten für** | Diskrete Aktionsräume; Bildbeobachtungen mit CnnPolicy |

---

### SAC — Soft Actor-Critic

| Feld | Detail |
|------|--------|
| **Zusammenfassung** | Off-Policy-Actor-Critic, der Return + Entropie maximiert; stichprobeneffizient und robust |
| **On/Off-Policy** | Off-Policy |
| **Aktionsraum** | Nur kontinuierlich |
| **Wichtige SB3-Hyperparameter** | `learning_rate=3e-4`, `buffer_size=1_000_000`, `learning_starts=100`, `batch_size=256`, `tau=0.005`, `gamma=0.99`, `train_freq=1`, `ent_coef='auto'`, `target_entropy='auto'` |
| **TensorBoard-Metriken** | `train/ent_coef`, `train/actor_loss`, `train/critic_loss`, `rollout/ep_rew_mean` |
| **SB3-Klasse** | `from stable_baselines3 import SAC` |
| **Am besten für** | Kontinuierliche Kontrolle (Roboterarme, Lokomotion); teure Simulationen |

---

### A2C — Advantage Actor-Critic

| Feld | Detail |
|------|--------|
| **Zusammenfassung** | Synchroner On-Policy-Actor-Critic; einfacher als PPO, nützliche Lern-Baseline |
| **On/Off-Policy** | On-Policy |
| **Aktionsraum** | Diskret & kontinuierlich |
| **Wichtige SB3-Hyperparameter** | `learning_rate=7e-4`, `n_steps=5`, `gamma=0.99`, `gae_lambda=1.0`, `ent_coef=0.0`, `vf_coef=0.5`, `max_grad_norm=0.5`, `rms_prop_eps=1e-5` |
| **TensorBoard-Metriken** | `train/entropy_loss`, `train/value_loss`, `rollout/ep_rew_mean` |
| **SB3-Klasse** | `from stable_baselines3 import A2C` |
| **Am besten für** | Schnelle Experimente; Umgebungen, in denen PPOs längeres Rollout verschwendet ist |

---

### DDPG — Deep Deterministic Policy Gradient

| Feld | Detail |
|------|--------|
| **Zusammenfassung** | Off-Policy-Actor-Critic mit deterministischer Policy; grundlegender Algorithmus für kontinuierliche Kontrolle, heute meist durch TD3/SAC ersetzt |
| **On/Off-Policy** | Off-Policy |
| **Aktionsraum** | Nur kontinuierlich |
| **Wichtige SB3-Hyperparameter** | `learning_rate=1e-3`, `buffer_size=1_000_000`, `learning_starts=100`, `batch_size=100`, `tau=0.005`, `gamma=0.99`, `train_freq=1` |
| **TensorBoard-Metriken** | `train/actor_loss`, `train/critic_loss`, `rollout/ep_rew_mean` |
| **SB3-Klasse** | `from stable_baselines3 import DDPG` |
| **Am besten für** | Kontinuierliche Kontrolle; historischer Vergleich mit TD3/SAC |

---

### TD3 — Twin Delayed Deep Deterministic

| Feld | Detail |
|------|--------|
| **Zusammenfassung** | DDPG + Zwillings-Critic + verzögertes Policy-Update + Ziel-Rauschen; deutlich stabiler als DDPG |
| **On/Off-Policy** | Off-Policy |
| **Aktionsraum** | Nur kontinuierlich |
| **Wichtige SB3-Hyperparameter** | `learning_rate=1e-3`, `buffer_size=1_000_000`, `learning_starts=100`, `batch_size=100`, `tau=0.005`, `gamma=0.99`, `train_freq=1`, `policy_delay=2`, `target_policy_noise=0.2`, `target_noise_clip=0.5` |
| **TensorBoard-Metriken** | `train/actor_loss`, `train/critic_loss`, `rollout/ep_rew_mean` |
| **SB3-Klasse** | `from stable_baselines3 import TD3` |
| **Am besten für** | Kontinuierliche Kontrolle, wenn SACs stochastische Policy unerwünscht ist |

---

### REINFORCE

| Feld | Detail |
|------|--------|
| **Zusammenfassung** | Vanilla-Monte-Carlo-Policy-Gradient; ganze Episoden sammeln, mit diskontierten Returns updaten |
| **On/Off-Policy** | On-Policy |
| **Aktionsraum** | Diskret & kontinuierlich |
| **Wichtige Hyperparameter** | `learning_rate`, `gamma`; keine SB3-Klasse — im Kurs manuell implementiert |
| **TensorBoard-Metriken** | `rollout/ep_rew_mean`, `train/policy_gradient_loss` |
| **SB3-Klasse** | Keine (eigene Implementierung) |
| **Am besten für** | Policy-Gradienten von Grund auf verstehen, bevor man zu PPO geht |

---

### BC — Behavioral Cloning

| Feld | Detail |
|------|--------|
| **Zusammenfassung** | Überwachte Imitation: Cross-Entropy zwischen Experten- und vorhergesagten Aktionen minimieren; keine Umweltinteraktion nötig |
| **On/Off-Policy** | Offline (kein RL) |
| **Aktionsraum** | Diskret & kontinuierlich |
| **Wichtige Hyperparameter** | `learning_rate=1e-3`, `batch_size=64`, `n_epochs` über dem Demo-Datensatz |
| **TensorBoard-Metriken** | `train/bc_loss`, Evaluations-Erfolgsrate |
| **SB3-Klasse** | `from imitation.algorithms import bc` (imitation-Bibliothek) |
| **Am besten für** | Policy aus Demos bootstrappen, bevor mit RL feinabgestimmt wird |

---

### DT — Decision Transformer

| Feld | Detail |
|------|--------|
| **Zusammenfassung** | Offline RL als Sequenzmodellierung: ein Transformer sagt die nächste Aktion bedingt auf vergangene (Return-to-Go, State, Action)-Tokens vorher |
| **On/Off-Policy** | Offline |
| **Aktionsraum** | Diskret & kontinuierlich |
| **Wichtige Hyperparameter** | `context_length`, `n_layer`, `n_head`, `learning_rate`, Ziel-Return (RTG) bei Inferenz |
| **TensorBoard-Metriken** | `train/action_loss`, Offline-Eval-Return |
| **SB3-Klasse** | Keine (HuggingFace `decision_transformer` oder eigen) |
| **Am besten für** | Offline-Datensätze mit breiter Return-Abdeckung; ziel-bedingte Inferenz via RTG |

---

### HER — Hindsight Experience Replay

| Feld | Detail |
|------|--------|
| **Zusammenfassung** | Replay-Buffer-Wrapper, der gescheiterte Episodenziele mit erreichten Zielen umetikettiert; ermöglicht Lernen aus sparsamen-Belohnungs-Manipulationsaufgaben |
| **On/Off-Policy** | Off-Policy (wickelt SAC/TD3/DDPG ein) |
| **Aktionsraum** | Diskret & kontinuierlich |
| **Wichtige SB3-Hyperparameter** | `n_sampled_goal=4`, `goal_selection_strategy='future'`; Basis-Hyperparameter unverändert |
| **TensorBoard-Metriken** | `rollout/success_rate`, `rollout/ep_rew_mean` |
| **SB3-Klasse** | `from stable_baselines3 import HerReplayBuffer` |
| **Am besten für** | Sparsame-Belohnungs-Ziel-Aufgaben (Roboter-Manipulation, Navigation) |

---

### RND — Random Network Distillation

| Feld | Detail |
|------|--------|
| **Zusammenfassung** | Neugier-Methode: intrinsische Belohnung = Vorhersagefehler eines Netzes, das einen festen Zufalls-Encoder zu treffen lernt; skaliert auf Bildbeobachtungen |
| **On/Off-Policy** | On-Policy (typisch mit PPO kombiniert) |
| **Aktionsraum** | Diskret & kontinuierlich |
| **Wichtige Hyperparameter** | `int_coef` (intrinsische Skalierung), `ext_coef` (extrinsische Skalierung), Feature-Embedding-Größe |
| **TensorBoard-Metriken** | `train/intrinsic_reward_mean`, `train/rnd_loss`, `rollout/ep_rew_mean` |
| **SB3-Klasse** | Keine (eigen oder sb3-contrib) |
| **Am besten für** | Harte Exploration mit spärlichen Belohnungen, wo ICM nicht reicht |

---

## Teil 3 — Kerngleichungen

### Bellman-Gleichung

```
V(s) = E_a~π [ r(s,a) + γ · V(s') ]
```

Der Wert eines Zustands entspricht der erwarteten unmittelbaren Belohnung plus dem diskontierten Wert des nächsten Zustands. Jedes bootstrap-basierte Wert-Update in RL ist eine Näherung dieser Fixpunktgleichung.

---

### Policy-Gradient-Theorem (REINFORCE)

```
∇_θ J(θ) = E_{τ~π_θ} [ Σ_t ∇_θ log π_θ(a_t | s_t) · G_t ]
```

Der Gradient des erwarteten Returns entspricht dem erwarteten Log-Wahrscheinlichkeits-Gradienten, gewichtet mit dem Return. Unverzerrt, aber hohe Varianz; Abzug einer Baseline (V(s)) ergibt die Advantage-Form aus A2C/PPO.

---

### PPO-Clipped-Objektiv

```
L_CLIP(θ) = E_t [ min(
    r_t(θ) · A_t ,
    clip(r_t(θ), 1-ε, 1+ε) · A_t
) ]

with r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
```

Das Minimum aus ungeklipptem und geklipptem Surrogat; ignoriert pessimistisch Verbesserungen jenseits der Trust-Region und hält das Verhältnis nahe 1, sodass Daten on-policy bleiben.

---

### SAC-Entropie-erweitertes Objektiv

```
J(π) = E_{τ~π} [ Σ_t ( r(s_t, a_t) + α · H(π(· | s_t)) ) ]

H(π(· | s)) = -E_{a~π} [ log π(a | s) ]
```

SAC maximiert Aufgaben-Belohnung *und* Policy-Entropie. Die Temperatur α tradet Ausbeutung (hohe Belohnung) gegen Erkundung (hohe Entropie). Wird α gelernt, passt es sich auf eine Ziel-Entropie an.

---

### GAE — Generalized Advantage Estimation

```
A_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}

δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
```

Geometrisch gewichtete Summe von TD-Fehlern mit Abklingfaktor (γλ). λ=0 kollabiert zum 1-Step-TD-Advantage (niedrige Varianz, hoher Bias); λ=1 zu vollständigen MC-Returns (hohe Varianz, niedriger Bias). SB3-PPO-Default: `gae_lambda=0.95`.

---

### TD-Fehler (δ)

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
```

Der Temporal-Difference-Fehler ist das „Überraschungs"-Signal: wie viel besser oder schlechter die beobachtete Belohnung plus der bootstrap-zukunftige Wert gegenüber der aktuellen Schätzung ist. Die Minimierung von δ² ist das Trainingsziel des Critic.

---

### KL-Divergenz

```
KL(p ‖ q) = Σ_x p(x) · log( p(x) / q(x) )
```

Misst, wie viel Information verloren geht, wenn q als Näherung für p genutzt wird. Im RL: Abstand zwischen alter und neuer Policy nach einem Gradientenupdate; TRPO erzwingt KL ≤ δ als harte Bedingung; PPO überwacht `approx_kl` als weiches Signal.

---

### RND-Intrinsische Belohnung

```
r_i(s_t) = ‖ f̂(s_t ; θ_pred) - f(s_t ; θ_fixed) ‖²
```

Die intrinsische Belohnung ist der quadratische L2-Abstand zwischen Predictor- und festem Zufalls-Target-Netz. Neue Zustände haben hohen Fehler (hoher Bonus); häufig besuchte niedrigen Fehler (niedriger Bonus) und treiben den Agenten weiter.
