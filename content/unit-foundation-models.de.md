# Foundation Models für die Steuerung

Große Sprachmodelle erwiesen sich als überraschend gut auf Spiel-Benchmarks. Jetzt stellen Forschende eine schwierigere Frage: Kann ein einzelnes **Foundation Model** — trainiert auf Daten in Internet-Größenordnung — lernen, Roboter und verkörperte Agenten direkt zu steuern, ohne Reward-Engineering pro Aufgabe? RT-2, Octo, OpenVLA und π0 sind vier ernstzunehmende Antworten auf diese Frage. Diese Einheit ist eine **Literacy- und Wegweiser-Einheit**: kein praktisches Coding, aber am Ende verstehst du, was jedes Modell tut, warum sie heute schwer in Godot einzubinden sind und wo du nachschauen solltest, wenn das Ökosystem reift.

[← World Models / DreamerV3](unit-world-models.md) · [Kursstartseite](index.md)

---

!!! info "Drei Wege, deine KI zu sehen"
    Lies das Paper (RT-2s Chain-of-Thought-Reasoning-Ausgabe — das Modell druckt buchstäblich „move left to grasp") · durchstöbere die HuggingFace-Model-Card (Octo und OpenVLA haben detaillierte Cards mit Evaluations-Videos) · schau ein π0-Demo-Video (auf der Website von Physical Intelligence gibt es Zeitlupen-Clips des Roboters beim Wäschefalten — studiere, was die Hände zwischen den Keyframes tun)

---

## 1 · Was ist ein Foundation Model für die Steuerung?

Die Idee ist aus dem NLP geborgt: einmal auf einem riesigen, vielfältigen Datensatz vortrainieren, dann günstig an neue Aufgaben anpassen. In der Sprache kann ein einzelnes GPT-artiges Modell Code schreiben, Verträge zusammenfassen und das Anwaltsexamen bestehen — alles aus einem Satz Gewichte.

**Foundation Models für die Steuerung versuchen dieselbe Wette für verkörperte Agenten:**

```
Foundation model (language / vision):
  text: "put the apple in the bowl"
  image: camera frame from the robot
  →  joint torques / gripper command / Cartesian waypoint

Goal: one model, many tasks, minimal task-specific tuning.
```

Warum ist das schwer? Sprachmodelle sehen Text — eine flache Sequenz von Tokens. Ein Roboterarm sieht Kamerabilder und propriozeptiven Zustand und muss *kontinuierliche* Aktionen mit 10–50 Hz und Sub-Zentimeter-Präzision ausgeben. Diese Lücke zu überbrücken ist die zentrale Engineering-Herausforderung, die jedes Modell unten anders angeht.

---

## 2 · Die vier Modelle

### 2.1 · RT-2 — Robotic Transformer 2 (Google DeepMind, 2023)

**Was es ist.** RT-2 macht ein Fine-Tuning eines Vision-Language-Modells (PaLI-X oder PaLM-E mit 55 Mrd. Parametern) auf einer Mischung aus Webdaten und Roboter-Demonstrationen. Aktionen werden **als Text tokenisiert**: Die kontinuierlichen Gelenkwinkel werden in 256 Bins diskretisiert und als gewöhnliche Text-Tokens neben Wörtern ausgegeben.

| Eigenschaft | Detail |
|----------|--------|
| Basismodell | PaLI-X 55B (Vision-Language) |
| Trainingsdaten | 130k Roboter-Demonstrationen + Text/Bild-Daten in Web-Größenordnung |
| Aktionsraum | 7-DOF-Arm (6 Gelenke + Greifer), als Text tokenisiert |
| Schlüsselfähigkeit | Zero-Shot-Generalisierung auf neue Objekte und Anweisungen |
| Architektur-Entscheidung | Aktionen als Sprach-Tokens darstellen — die Next-Token-Maschinerie des LLM wiederverwenden |

**Chain-of-Thought-Reasoning.** RT-2 kann vor der Aktionsausgabe eine Reasoning-Spur ausgeben: `"The instruction says pick up the empty cup. I see two cups. The one on the left has liquid. I should pick the right one. Action: [move_right, close_gripper]"`. Das ist emergent aus dem Sprach-Pretraining, nicht separat entwickelt.

**Repo / Paper.** Keine öffentlichen Gewichte (Google-intern), aber das Paper liegt unter [arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818).

---

### 2.2 · Octo — An Open-Source Generalist Robot Policy (2024)

**Was es ist.** Octo ist eine **vollständig quelloffene** Transformer-Policy, trainiert auf dem Open-X-Embodiment-Datensatz — 800k+ Demonstrationen von 22 verschiedenen Roboterplattformen. Das Ziel ist, das BERT des Roboter-Lernens zu sein: ein vortrainiertes Backbone, das man feintunt statt von Grund auf zu trainieren.

| Eigenschaft | Detail |
|----------|--------|
| Basismodell | Transformer mit 90M Parametern (kein LLM-Pretraining) |
| Trainingsdaten | Open X-Embodiment — 800k+ Demonstrationen, 22 Robotertypen |
| Aktionsraum | Kontinuierlicher 7-DOF-Arm + Greifer, Diffusion-Head oder Gauß |
| Schlüsselfähigkeit | Fine-Tuning auf einen neuen Roboter mit ~100 Demonstrationen |
| Architektur-Entscheidung | Modularer Tokenizer: Sprachziel, Bildziel oder beides einsetzbar |

**Warum Octo wichtig ist.** Es ist der erste großmaßstäbliche offene Checkpoint, trainiert auf einem heterogenen Multi-Roboter-Datensatz. Die Architektur trennt *was du willst* (Task-Token: Sprach- oder Bildziel) von *wie du dich bewegst* (Action-Head), sodass das Task-Token getauscht werden kann, ohne das Backbone neu zu trainieren.

**Repo.** [github.com/octo-models/octo](https://github.com/octo-models/octo) — Gewichte auf HuggingFace unter `hf.co/rail-berkeley/octo-small` und `hf.co/rail-berkeley/octo-base`.

---

### 2.3 · OpenVLA — Open Vision-Language-Action Model (2024)

**Was es ist.** OpenVLA macht ein Fine-Tuning von **Prismatic-7B** (ein 7B-LLM mit Vision-Backbone) auf demselben Open-X-Embodiment-Datensatz wie Octo. Wie bei RT-2 werden Aktionen als Text tokenisiert. Anders als bei RT-2 sind die Gewichte öffentlich.

| Eigenschaft | Detail |
|----------|--------|
| Basismodell | Prismatic-7B (LLaVA-artiges VLM) |
| Trainingsdaten | Open X-Embodiment — 970k Demonstrationen |
| Aktionsraum | 7-DOF absolute Gelenkpositionen, 256-Bin-Tokenisierung |
| Schlüsselfähigkeit | Anweisungsbefolgung über diverse Roboterplattformen hinweg |
| Architektur-Entscheidung | Das LLM-Vokabular um Action-Tokens erweitern — kein separater Action-Head |

**Effizienz-Arbeit.** Spätere Arbeit (OpenVLA-OFT, 2024) fügt parameter-effizientes Fine-Tuning hinzu (LoRA + paralleles Dekodieren), das die Inferenz von ~6 Tokens/Sekunde auf Echtzeit auf einer einzelnen A100 bringt. Immer noch kein Embedded-System-Terrain.

**Repo / Gewichte.** [github.com/openvla/openvla](https://github.com/openvla/openvla) — Modell unter `hf.co/openvla/openvla-7b`.

---

### 2.4 · π0 — Physical Intelligence's Foundation Model (2024)

**Was es ist.** π0 („pi-zero") von Physical Intelligence trainiert einen **Flow-Matching-Diffusion-Action-Head** auf einem PaliGemma-Vision-Language-Backbone. Statt Aktionen in Tokens zu diskretisieren, lernt es eine kontinuierliche Aktionsverteilung über einen Diffusionsprozess, was glattere Trajektorien als die Bin-weise Tokenisierung erzeugt.

| Eigenschaft | Detail |
|----------|--------|
| Basismodell | PaliGemma 3B VLM (Google) |
| Trainingsdaten | Proprietäre Flotte von Roboter-Demonstrationen (geschickte Manipulation) |
| Aktionsraum | Kontinuierlich — Handgelenk-, Finger-, Basis-Geschwindigkeiten mit 50 Hz |
| Schlüsselfähigkeit | Geschickte Aufgaben: Wäsche falten, Tische abräumen, Karton-Montage |
| Architektur-Entscheidung | Flow-Matching-Diffusion-Head — kontinuierliche Aktionsverteilung, keine Tokens |

**Warum Diffusion für Aktionen?** Kontinuierliche Aktionen zu tokenisieren führt Quantisierungsfehler ein und zwingt das Modell, ein Token nach dem anderen auszugeben (langsam). Ein Diffusion-Head lernt eine vollständige Aktionsverteilung in einem einzigen Forward-Pass und kann multimodales Verhalten repräsentieren (z. B. „nach links *oder* rechts bewegen sind beide gültig" früh in der Trajektorie).

**Repo / Ankündigung.** Noch keine öffentlichen Gewichte. Blogpost und Demo-Videos unter [physicalintelligence.company/blog/pi0](https://physicalintelligence.company/blog/pi0). Paper: [arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164).

---

## 3 · Warum sie heute schwer in Godot einzubinden sind

Du könntest versucht sein, Octo oder OpenVLA in deine Godot-Umgebung zu setzen und als Policy laufen zu lassen. Drei grundlegende Diskrepanzen machen das aktuell nicht trivial.

### 3.1 · Beobachtungs-Modalitäts-Diskrepanz

Jedes Modell oben wurde auf **echten Roboter-Kameraframes** trainiert: leicht unscharfe, bewegungsverwischte 480×640-RGB-Bilder mit den subtilen Farbstichen von Leuchtstoffröhren-Beleuchtung. Godot rendert **synthetische Bilder**: scharf, aliased, übersättigt, mit perfekter Beleuchtung.

Die Domänenlücke zwischen Godot-Renderings und echten RGB-Bildern ist größer als die zwischen simulierter und realer Physik. Das visuelle Backbone dieser Modelle hat Merkmale aus echten Bildern gelernt; Godot-Bilder erzeugen Aktivierungen, die außerhalb der Trainingsverteilung liegen. Die Performance verschlechtert sich — manchmal katastrophal.

Workarounds, die Leute probiert haben:

- Style-Transfer auf Godot-Renderings anwenden (langsam, verlustbehaftet)
- Tiefenkarten oder Segmentierungsmasken statt RGB nutzen (Modalitäts-Diskrepanz, aber kleinere Domänenlücke)
- Fine-Tuning auf Godot-gerenderten Daten (erfordert einen GPU-Cluster und tausende Godot-Demonstrationen)

### 3.2 · Aktionsraum-Diskrepanz

RT-2, Octo und OpenVLA produzieren Aktionen für **7-DOF-Roboterarme** mit einer spezifischen Gelenkreihenfolge, Skalierung und physikalischen Einheiten (Radiant oder Meter bei bestimmten Frequenzen). Godot-Umgebungen haben üblicherweise einen völlig anderen Aktionsraum — einen `ContinuousAction`-Vektor mit spielspezifischer Semantik.

Remapping ist theoretisch möglich, aber nicht automatisch: Du musst verstehen, was jede Aktionsdimension im Quell-Roboter bedeutet, was sie in deinem Godot-Charakter bedeutet, und ob die Dynamik (Trägheit, Dämpfung, Übersetzungsverhältnisse) überhaupt vergleichbar ist.

### 3.3 · Keine leichtgewichtigen Godot-tauglichen Checkpoints

Der kleinste öffentliche Checkpoint ist **Octo-Small** mit 27M Parametern — schnell auszuführen, aber immer noch für CUDA-Inferenz mit spezifischen Roboter-Normalisierungsstatistiken ausgelegt. Es gibt kein `pip install octo && env.step(octo.act(obs))`, das einfach mit einer Godot-Env funktioniert. Du müsstest:

1. Einen eigenen Beobachtungs-Adapter schreiben (Godot → Roboter-Obs-Format)
2. Octos Normalisierungsstatistiken für deinen Aktionsraum überschreiben
3. Den Modell-Inferenzserver hosten (Python) und ihn aus der Godot-Trainingsschleife aufrufen
4. Eine Inferenzlatenz von ~50 ms pro Schritt akzeptieren — was deine Umgebung auf max. ~20 Hz deckelt

Für die meisten Godot-RL-Aufgaben konvergiert eine eigene PPO- oder SAC-Policy, von Grund auf trainiert, schneller und läuft mit 200 Hz ohne Infrastruktur-Overhead.

---

## 4 · Der Open-X-Embodiment-Datensatz

Viele der Modelle oben konvergieren auf denselben Trainingskorpus. Es lohnt sich zu wissen, was er ist.

**Open X-Embodiment** (Kollaboration von 33 Institutionen, Google DeepMind, 2023) aggregierte 22 öffentlich veröffentlichte Roboter-Datensätze in ein einziges standardisiertes Format — 1M+ Demonstrationen über 22 Roboter-Morphologien (Arme, mobile Manipulatoren, bimanuelle Systeme, Humanoide), 527 Fertigkeiten und mehrere Labore.

**Schlüssel-Einsicht.** Vor Open X-E trainierten die meisten Roboter-Lern-Paper auf 200–5.000 Demonstrationen vom Roboter eines einzigen Labors. Open X-E war der erste Datensatz, der groß genug war, dass Cross-Embodiment-Pretraining messbar besser war als Single-Embodiment-Training von Grund auf.

**Zugang.** [robotics-transformer-x.github.io](https://robotics-transformer-x.github.io) — Daten im RLDS-Format (TensorFlow Datasets), die meisten Teilmengen erfordern einen akademischen Google-Account.

---

## 5 · Konzeptuelle Karte — wo diese Modelle in der RL-Taxonomie sitzen

Foundation Models für die Steuerung werden oft als „kein RL" beschrieben — sie nutzen Behaviour Cloning aus Demonstrationen, keine belohnungsgetriebene Policy-Suche. Das ist teils wahr und teils irreführend.

```
Pure BC:    demonstrations → policy  (no reward, no exploration)
Pure RL:    reward function → policy  (no demonstrations, full exploration)

These models:
  Octo/OpenVLA:  demonstrations → pre-trained backbone → fine-tune with BC
  π0:            demonstrations → pre-trained backbone → optionally fine-tune with RL (π0-FAST variant)
  RT-2:          web data + demonstrations → BC → deploy (no RL fine-tuning in original paper)
```

**Warum nicht einfach RL?** Für geschickte Manipulation ist Reward-Design extrem schwer — wie schreibst du eine Belohnung für „falte das Hemd ordentlich"? Imitationslernen aus menschlichen Demonstrationen umgeht das Reward-Engineering. Der Trade-off ist, dass BC-trainierte Modelle nicht leicht Verhaltensweisen entdecken können, die nicht in den Demonstrationen vorkommen.

**Wo RL wieder eintritt.** Jüngere Arbeit (π0-FAST, 2024; RT-X mit RL-Fine-Tuning) nutzt RL, um das Foundation Model nach dem BC-Pretraining zu feintunen — ähnlich wie RLHF in der Sprache. Das kombiniert die Abdeckung menschlicher Demonstrationen mit RLs Fähigkeit, über den Demonstrator hinaus zu verbessern.

---

## 6 · Stretch Goals

### 6.1 · Ein kleines VLM als Feature-Extractor für eine Godot-Policy laden

Ein Weg, *etwas* Wert aus Foundation-Model-Pretraining zu ziehen, ohne ein volles 7B-Modell zur Inferenzzeit laufen zu lassen, ist, das visuelle Backbone als **eingefrorenen Feature-Extractor** zu nutzen und einen kleinen RL-Head darauf zu trainieren. Die Idee:

1. Nimm den Vision-Encoder aus einem kleinen offenen VLM — zum Beispiel **SigLIP** (ein 400M-Vision-Language-Encoder von Google, verfügbar unter `hf.co/google/siglip-base-patch16-224`) oder den Vision-Tower von **LLaVA-1.5-7B** (`hf.co/liuhaotian/llava-v1.5-7b`, nur der Vision-Teil).
2. Lass die Godot-SubViewport-Pipeline laufen, um 224×224-RGB-Frames zu erfassen (siehe die [Einheit Visuelle Beobachtungen](unit-visual-observations.md)).
3. Schicke jedes Frame durch den eingefrorenen Encoder, um einen 768-dim- oder 1152-dim-Embedding-Vektor zu bekommen.
4. Nutze dieses Embedding als Beobachtung für eine Standard-PPO- oder -SAC-Policy. Der RL-Head ist klein — zwei verborgene Schichten mit 256 reichen.
5. Der eingefrorene Encoder liefert semantische Merkmale (Objektidentität, räumliches Layout), die ein von Grund auf trainiertes CNN Millionen von Schritten bräuchte, um sie zu lernen.

**Was zu erwarten ist.** Diese Technik (in der Literatur „frozen visual pre-training" oder „VC-1" genannt — siehe [arxiv.org/abs/2303.04137](https://arxiv.org/abs/2303.04137)) kann die Stichprobeneffizienz bei Aufgaben verbessern, die semantisches Verständnis der Szene erfordern. Für reine Fortbewegung hilft sie oft nicht — die Aufgabe ist geometrisch, nicht semantisch.

**Infrastruktur-Hinweis.** SigLIP-base zur Inferenzzeit kostet ~20 ms auf einer CPU. Das deckelt deine Godot-Env auf ~50 Hz — akzeptabel, wenn `n_parallel` niedrig ist. Auf einer GPU sinkt es auf ~2 ms, sodass 200 Hz erreichbar sind. Nutze ONNX-Export (`optimum-cli export onnx`), um eine laufzeit-effiziente Version des Encoders zu bekommen.

### 6.2 · Sprachkonditionierte Ziele ohne ein 7B-Modell

Du brauchst kein RT-2, um sprachkonditioniertes Verhalten in Godot zu bekommen. Ein leichterer Ansatz:

1. Bette die Textanweisung mit einem eingefrorenen **Sentence-Transformer** ein (z. B. `all-MiniLM-L6-v2` — 22M Params, 384-dim-Ausgabe, CPU-schnell). Installiere mit `pip install sentence-transformers`.
2. Verkette das 384-dim-Satz-Embedding mit deinem bestehenden Beobachtungsvektor.
3. Trainiere PPO normal — die Policy lernt, auf das Sprach-Embedding zu konditionieren.
4. Zur Testzeit tauschst du den Anweisungstext, und die Policy generalisiert (soweit die Trainingsanweisungen die relevante Verteilung abgedeckt haben).

Das ist der Ansatz, der in **CLIP-Fields** und **SayCan (vereinfacht)** genutzt wird. Er kostet ~5 ms pro Anweisungs-Encode auf CPU (nicht pro Schritt — du encodierst einmal pro Episode) und fügt deinem Obs-Raum nur 384 Dimensionen hinzu.

### 6.3 · Das Octo-Fine-Tuning-Tutorial lesen

Das Octo-Repository hat ein Colab-Notebook ([github.com/octo-models/octo/blob/main/examples/02_finetune_new_observation_action.ipynb](https://github.com/octo-models/octo/blob/main/examples/02_finetune_new_observation_action.ipynb)), das zeigt, wie man Beobachtungs- und Aktionsräume für einen eigenen Roboter überschreibt. Es durchzuarbeiten — selbst ohne es auszuführen — ist der klarste Weg, zu verstehen, was ein Beobachtungs-Adapter wirklich erfordert. Achte auf den Abschnitt zu Normalisierungsstatistiken: Das ist der Schritt, den die meisten beim Portieren auf eine neue Embodiment falsch machen.

---

## Was kommt als Nächstes?

Du hast das Ende des Anleitungen-Bereichs erreicht. Kehre zur [Kursstartseite](index.md) für die vollständige Einheitenliste zurück, oder besuche eine frühere Einheit erneut, um tiefer einzusteigen.

[→ Kursstartseite](index.md)

---

[← World Models / DreamerV3](unit-world-models.md) · [Kursstartseite](index.md)
