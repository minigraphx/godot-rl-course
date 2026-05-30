# Godot RL-Kurs

Lerne Deep Reinforcement Learning durch das Bauen und Trainieren von Agenten in echten Godot-Spielumgebungen.

## Was du bauen wirst

| Phase | Inhalt | Was du lernst |
|-------|--------|---------------|
| **Phase 1 — Grundlagen** | Einrichtung · RL-Schleife · erste eigene Umgebung · Belohnungsdesign | Wie RL funktioniert, wie man Belohnungen entwirft |
| **Phase 2 — Wertbasiert** | Q-Learning · DQN · Neugier | Bellman-Gleichung, Q-Tabellen, DQN, Erkundung bei spärlichen Belohnungen |
| **Phase 3 — Richtlinienbasiert** | REINFORCE · Actor-Critic · PPO · SAC · PPO von Grund auf (CleanRL) | Policy-Gradient-Theorem, PPO-Interna, kontinuierliche Steuerung |
| **Phase 4 — Skalierung** | Parallel · 3D · Multi-Agent · Gedächtnis | Realwelttraining in großem Maßstab |
| **Phase 5 — Jenseits der Belohnung** | Multi-Task RL · Imitationslernen · ONNX/WASM · Abschlussprojekt | Alternative Lernsignale, generalistische Policies, Deployment |
| **Phase 6 — Robotik** | Robotersensoren · HER · Sim-to-Real · Safe RL | Roboter-Beobachtungs-/Aktionsdesign, zielkonditioniertes RL |
| **Anleitungen** | Debugging · Fortgeschrittene Evaluation · PBT · World Models | Systematische Diagnose, Evaluation, Hyperparameter-AutoML |

## Drei Wege, deine KI zu sehen (jede Einheit)

| Kanal | Was er zeigt |
|-------|-------------|
| **Godot** | Agentenverhalten in der Welt |
| **TensorBoard** | Lernkurven (`tensorboard --logdir=logs`) |
| **AIController-Quellcode** | Beobachtungen, Aktionen und Belohnungen, die du geändert hast |

## Einheiten

**Phase 1 — Grundlagen**

- [Einheit 0 — Einrichtung & Erster Start](unit-00.md)
- [Einheit 1 — RL-Grundlagen](unit-01.md)
- [Belohnungsdesign](unit-reward-engineering.md)
- [Einheit 2 — Erste eigene Umgebung](unit-02.md)

**Phase 2 — Wertbasierte Methoden**

- [Q-Learning](unit-q-learning.md)
- [Deep Q-Learning (DQN)](unit-03.md)
- [Intrinsische Motivation & Neugier](unit-curiosity.md)

**Phase 3 — Richtlinienbasierte Methoden**

- [Policy Gradients & REINFORCE](unit-policy-gradients.md)
- [Actor-Critic](unit-actor-critic.md)
- [PPO im Detail](unit-ppo-deep.md)
- [PPO in der Praxis (JumperHard)](unit-04.md)
- [SAC — Soft Actor-Critic](unit-sac.md)
- [PPO von Grund auf (CleanRL)](unit-cleanrl.md)

**Phase 4 — Skalierung & Komplexität**

- [Paralleles Training](unit-05.md)
- [Kontinuierliche 3D-Steuerung](unit-06.md)
- [Visuelle Beobachtungen](unit-visual-observations.md)
- [Multi-Agent](unit-07.md)
- [Gedächtnis & POMDPs](unit-08.md)
- [Self-Play](unit-self-play.md)
- [Hierarchisches RL](unit-hierarchical.md)

**Phase 5 — Jenseits der Belohnung**

- [Multi-Task RL](unit-multitask.md)
- [Imitationslernen](unit-09.md)
- [Offline RL](unit-offline-rl.md)
- [KI deployen](unit-10.md)
- [Abschlussprojekt](unit-capstone.md)

**Phase 6 — Robotik**

- [Roboterbeobachtungen & Sensoren](unit-robotics.md)
- [Fortbewegungsagenten](unit-locomotion.md)
- [Zielkonditioniertes RL & HER](unit-her.md)
- [Sim-to-Real Transfer](unit-sim-to-real.md)
- [Safe RL / Constrained MDPs](unit-safe-rl.md)

---

!!! info "Übersetzung"
    Diese Seite ist auf Deutsch verfügbar. Weitere Einheiten werden schrittweise übersetzt.
    Code-Blöcke bleiben auf Englisch — Programmiersprachen sind universell.
    Möchtest du mithelfen? Öffne ein GitHub Issue mit dem Label `translation`.
