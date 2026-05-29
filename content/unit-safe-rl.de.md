# Safe RL / Constrained MDPs

[← Sim-to-Real Transfer](unit-sim-to-real.md) · [Kursstartseite](index.md)

!!! info "Zeit"
    Lesen: ~35 min · Training: ~30 min GPU / ~2 Std CPU

!!! info "Drei Wege, deine KI zu beobachten"
    - **Godot-Viewport** — beobachte, wie der Agent zum Ziel navigiert und dabei sichtbar die Gefahrenzone meidet; zu Beginn des Trainings stolpert er stumpf hindurch; bei Konvergenz sollte er sie in nahezu jeder Episode umroutet haben.
    - **TensorBoard** — plotte zwei Kurven nebeneinander: `rollout/ep_rew_mean` (Task-Reward) und `constraint/violation_rate` (Anteil der Schritte mit Kosten). Ein funktionierender CMDP-Lauf zeigt steigenden Reward *bei gleichzeitig* fallender Verletzungsrate — penalty-basierte Methoden zeigen oft die beiden Kurven gegeneinander kämpfen.
    - **λ-Monitor** — logge bei jedem Update den Lagrange-Multiplikator `λ`. Er sollte steigen, sobald die Bedingung verletzt wird, und fallen, wenn der Agent sicher ist. Eine flache λ-Kurve heißt entweder, die Bedingung wird nie ausgelöst, oder das Dual-Update ist kaputt.

---

Du hast Policies trainiert, die springen, laufen, greifen und auf echte Hardware transferieren. In jeder Unit bisher hieß „sicher" einfach: eine `-100`-Strafe zum Reward hinzufügen und hoffen. Für Demos und Forschungsprototypen funktioniert das oft. Für Policies, die physische Roboter, OP-Assistenten, autonome Fahrzeuge oder andere Systeme steuern sollen, in denen eine Bedingung *halten muss* — funktioniert es nicht zuverlässig.

Diese Unit lehrt dich, warum, und was du stattdessen tun sollst.

## 1 · Warum Reward-Strafen nicht reichen

Der Standardansatz, einem Agenten Constraints (Bedingungen) beizubringen, ist, einen großen negativen Reward zu vergeben, sobald die Bedingung verletzt wird:

```python
# Standard penalty approach
if is_in_hazard_zone(state):
    reward -= 100.0
```

Das ist bequem und funktioniert manchmal. Aber es hat ein strukturelles Problem, das es für harte Bedingungen ungeeignet macht:

**Der Strafkoeffizient hat keine prinzipielle Interpretation.** Wenn du die Strafe auf `-100` setzt, sagst du implizit „eine Bedingungsverletzung ist 100 Reward-Punkte wert." Aber wahrscheinlich hast du diese Behauptung nie verifiziert. Was, wenn der Task-Reward +5 pro Schritt beträgt und die Episode 500 Schritte lang ist? Ein maximaler Task-Reward von +2500 heißt, dein Strafkoeffizient ist 4 % des Max-Returns — kaum spürbar, und der Agent entscheidet vielleicht, dass es sich lohnt, die Bedingung sechsmal pro Episode zu verletzen. Erhöhst du die Strafe auf `-10000`, weigert sich der Agent vielleicht ganz zu bewegen, was den Task-Reward fast auf null kollabieren lässt.

Du tunst am Ende einen Koeffizienten, dessen korrekten Wert du nicht kennen kannst, ohne das Problem zuerst gelöst zu haben. Das ist die **Penalty-Tuning-Falle**.

**Der Tradeoff ist unvorhersehbar.** Mit einem Strafterm tauscht der Agent implizit Reward gegen Sicherheit. Wie viele Verletzungen sind „akzeptabel"? Mit einer Strafe hängt die Antwort vom Verhältnis Strafgröße zu erwartetem Return ab — ein Verhältnis, das sich ändert, sobald du den Task-Reward oder die Episodenlänge modifizierst. Es gibt keinen Knopf für „null Verletzungen". Es gibt nur Knöpfe, die ungefähr in diese Richtung schieben, und die Approximationsqualität variiert je Aufgabe.

**Die Bedingung kann im Mittel erfüllt, aber in jeder Episode verletzt sein.** Angenommen, deine Strafe erzeugt einen Agenten, der die Bedingung im Mittel 0,5-mal pro Episode verletzt. Die erwartete Strafe hebt sich genau gegen den erwarteten Gewinn auf. Statistisch sieht das gut aus. Praktisch betritt der Agent die Gefahrenzone in der Hälfte aller Episoden — genau das, was du verhindern wolltest.

!!! warning "Penalty-Tuning ist ein verdeckter Tradeoff, keine Bedingung"
    Wenn du `-100` für eine Bedingungsverletzung addierst, erzwingst du keine Bedingung — du kodierst eine *Präferenz*. Der Agent verletzt die Bedingung immer dann, wenn das mehr als 100 Reward-Punkte wert ist. Brauchst du wirklich ein hartes Limit (Stromverbrauch, Gelenkmoment, Kontaktkraft, Geschwindigkeit nahe Menschen), brauchst du eine andere Formulierung.

Das richtige Framework ist der **Constrained Markov Decision Process**.

---

## 2 · Constrained-MDP-Formulierung (CMDP)

Ein Standard-MDP maximiert einen einzelnen erwarteten Return:

```
max J(π) = E[Σ γᵗ r(sₜ, aₜ)]
```

Ein **Constrained MDP (CMDP)** ergänzt eine oder mehrere explizite Constraint-Funktionen neben dem Reward:

```
max J(π) = E[Σ γᵗ r(sₜ, aₜ)]       ← task objective (unchanged)
subject to:
    C(π) ≤ d                         ← constraint: expected cost ≤ threshold
```

Wobei:

- `r(s, a)` der **Task-Reward** ist — das Signal, das du immer hattest
- `c(s, a)` eine **Kostenfunktion** ist — ein separates Signal für constraint-relevante Events (Eintritt in eine Gefahrenzone, Überschreiten eines Drehmoment-Limits usw.)
- `C(π) = E[Σ γᵗ c(sₜ, aₜ)]` die **erwarteten diskontierten Kosten** sind — die Constraint-Wertfunktion, analog zum Return
- `d` der **Constraint-Schwellwert** ist — die maximal tolerierbare erwartete Kost

Diese Formulierung ist aus drei Gründen sauberer:

1. **Der Schwellwert `d` hat eine direkte physische Interpretation.** „Die erwartete Anzahl an Schritten in der Gefahrenzone pro Episode muss unter 2,0 liegen." Du kannst `d` wählen, indem du Fachexperten fragst, was akzeptabel ist, statt einen Strafkoeffizienten zu tunen.
2. **Die Kostenfunktion ist separat von der Reward-Funktion.** Du kannst eine ändern, ohne die andere zu ändern. Du kannst eine neue Bedingung hinzufügen, ohne alle Reward-Koeffizienten neu zu tunen.
3. **Der Algorithmus erfüllt die Bedingung beweisbar während des Trainings (nicht erst bei Konvergenz).** Das ist die zentrale Behauptung von CPO, behandelt in Abschnitt 4.

### Die Kostenfunktion in Godot bauen

Kostenfunktionen haben eine einfachere Struktur als Reward-Funktionen: sie messen „wie schlimm ist dieser Schritt aus Sicherheitssicht", unabhängig vom Task-Fortschritt.

```gdscript
# In the environment script — compute cost alongside reward
func compute_step_signals() -> void:
    var reward: float = 0.0
    var cost: float   = 0.0

    # --- Task reward ---
    var dist_to_goal = global_position.distance_to(goal.global_position)
    reward += 1.0 - clampf(dist_to_goal / MAX_DIST, 0.0, 1.0)
    if dist_to_goal < GOAL_RADIUS:
        reward += 5.0
        _ai.done = true

    # --- Cost signal: hazard zone entry ---
    if _in_hazard_zone():
        cost += 1.0

    # --- Cost signal: velocity limit (e.g. near humans) ---
    if linear_velocity.length() > MAX_SAFE_SPEED:
        cost += linear_velocity.length() - MAX_SAFE_SPEED   # proportional penalty

    _ai.reward = reward
    # Store cost for the training wrapper to read
    info["cost"] = cost
```

Der Trainings-Wrapper liest `info["cost"]` und verfolgt `C(π)` über Episoden. Abschnitt 5 zeigt, wie PPO-Lagrangian dieses Signal nutzt.

---

## 3 · Lagrangian-Relaxation

Das CMDP ist ein constrained Optimierungsproblem. Die Standardmethode, constrained Optimierung in unconstrained Optimierung umzuwandeln, ist **Lagrangian-Relaxation**.

Führe einen **Lagrange-Multiplikator** `λ ≥ 0` für die Bedingung ein. Das Lagrangian-Ziel wird:

```
L(π, λ) = J(π) - λ · (C(π) - d)
```

Das Optimierungsproblem teilt sich in zwei:

- **Primal-Update** (Policy): `max_π L(π, λ)` — maximiere das Lagrangian bei gegebenem Multiplikator
- **Dual-Update** (Multiplikator): `max_λ min_π L(π, λ)` — finde den Multiplikator, der die Bedingung erzwingt

Das Dual-Update ist ein Gradient-Ascent-Schritt auf `λ`:

```
λ ← max(0, λ + α_λ · (C(π) - d))
```

Dieses Update in Klartext:

- Ist `C(π) > d` (Bedingung verletzt): `λ` steigt → die Strafe `λ · C(π)` im Primal-Ziel wächst → die Policy wird stärker gedrängt, Kosten zu reduzieren
- Ist `C(π) < d` (Bedingung mit Spielraum erfüllt): `λ` sinkt → die Strafe entspannt sich → die Policy darf den Task-Reward verbessern
- `λ` wird immer auf `≥ 0` geclampt — eine Bedingung kann nicht zum Bonus werden

Das ist die **Primal-Dual-Methode** für CMDPs, auch **Lagrangian RL** genannt. Der Multiplikator `λ` ist ein adaptiver Strafkoeffizient, der sich automatisch kalibriert, um die Bedingung zu erzwingen.

### Warum das besser ist als eine feste Strafe

Eine feste Strafe von `-100` ist äquivalent dazu, `λ = 100` zu setzen und nie zu aktualisieren. Die Lagrangian-Methode aktualisiert `λ` bei jeder Trainingsiteration basierend auf der aktuellen Constraint-Verletzung. Verbessert sich die Policy darin, die Gefahrenzone zu meiden, entspannt sich `λ` natürlich und die Policy kann sich stärker auf den Task-Reward konzentrieren. Driftet das Training Richtung Constraint-Grenze, steigt `λ`, um es zurückzudrücken.

!!! tip "λ als Diagnose"
    Plotte `λ` über Trainings-Timesteps. Ein gesunder Lauf zeigt `λ` um ein stabiles Gleichgewicht oszillierend. Ein steigendes λ, das sich nie stabilisiert, heißt: die Policy kann die Bedingung nicht erfüllen — prüfe, ob `d` erreichbar ist oder ob die Kostenfunktion zu oft auslöst.

---

## 4 · CPO — Constrained Policy Optimization

**CPO (Achiam et al. 2017)** ist die Trust-Region-Methode für CMDPs. Sie erweitert TRPO auf das constrained Setting, indem sie bei jedem Schritt ein constrained Policy-Update löst:

```
max_θ  J(π_θ) - J(π_old)          ← maximize policy improvement
subject to:
    D_KL(π_old || π_θ) ≤ δ        ← trust region: don't change policy too fast
    C(π_θ) ≤ d                    ← constraint must be satisfied after the update
```

Die zentrale Garantie: **CPO erfüllt die Bedingung bei jedem Policy-Update beweisbar**, nicht erst bei Konvergenz. Die Policy „durchquert" während des Trainings nie eine unsichere Region.

Das zählt in der Praxis, weil es heißt: du kannst CPO auf einem echten Roboter (oder in einem sicherheitskritischen Simulator) trainieren, ohne dass die Policy beim Lernen an der Constraint-Grenze hin- und herschlägt.

### Wo du CPO bekommst

CPO ist über zwei Wege verfügbar:

**Option A: safety-gymnasium + SB3-contrib**

```bash
pip install safety-gymnasium
pip install sb3-contrib
```

```python
import safety_gymnasium
from sb3_contrib import TRPOPolicy  # CPO is a constrained extension of TRPO

env = safety_gymnasium.make("SafetyPointGoal1-v0")
# safety-gymnasium envs return (obs, reward, cost, terminated, truncated, info)
# CPO implementations wrap these automatically
```

**Option B: safety-starter-agents (OpenAI Original)**

```bash
git clone https://github.com/openai/safety-starter-agents
cd safety-starter-agents && pip install -e .
```

### CPO-Tradeoffs

| Eigenschaft | CPO | PPO-Lagrangian |
|----------|-----|----------------|
| Constraint-Erfüllung während des Trainings | Beweisbar | Approximativ |
| Implementierungs-Komplexität | Hoch (zweiter Ordnung) | Niedrig (erster Ordnung) |
| Rechenaufwand pro Update | Hoch | Niedrig |
| Stabilität | Sehr stabil | Mittel |
| Wann wählen | Beim Training auf echter Hardware; strikte Sicherheit | Beim Prototyping; nur-Sim-Training |

Für die meisten Godot-Projekte ist PPO-Lagrangian (Abschnitt 5) der richtige Startpunkt. Nutze CPO, wenn die Kosten einer Constraint-Verletzung während des Trainings selbst inakzeptabel sind.

---

## 5 · PPO-Lagrangian — lauffähiges Code-Beispiel

PPO-Lagrangian ist der Arbeitspferd-Algorithmus im Safe RL: er kombiniert die Einfachheit von PPO mit dem Lagrangian-Multiplikator-Update aus Abschnitt 3. Er ist einfacher zu implementieren als CPO und funktioniert in der Praxis gut.

### Mit safety-gymnasium (Quickstart)

```python
# ppo_lagrangian_safetygymnasium.py
# Requires: pip install safety-gymnasium torch stable-baselines3

import safety_gymnasium
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class LagrangianCallback(BaseCallback):
    """
    Dual update: adjust λ based on constraint violation.
    Plugs into SB3's callback system — called after each rollout.
    """
    def __init__(
        self,
        constraint_threshold: float = 25.0,  # d: max acceptable cost per episode
        lr_lambda: float = 0.05,             # α_λ: dual learning rate
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.d = constraint_threshold
        self.lr_lambda = lr_lambda
        self.lam = 0.0          # Lagrange multiplier, initialized at 0

    def _on_rollout_end(self) -> bool:
        # Read episode costs accumulated during the rollout
        ep_costs = self.locals.get("ep_cost_mean", None)
        if ep_costs is None:
            # Fall back: try reading from info buffers
            infos = self.locals.get("infos", [{}])
            costs = [inf.get("cost", 0.0) for inf in infos if isinstance(inf, dict)]
            ep_costs = float(np.mean(costs)) if costs else 0.0

        # Dual ascent update
        self.lam = max(0.0, self.lam + self.lr_lambda * (ep_costs - self.d))

        # Inject λ into the policy's loss as a penalty coefficient
        # The policy optimizer will minimize: -J(π) + λ * C(π)
        self.model.policy.optimizer.param_groups[0]["lagrange_lambda"] = self.lam

        self.logger.record("constraint/lambda", self.lam)
        self.logger.record("constraint/ep_cost_mean", ep_costs)
        self.logger.record("constraint/threshold", self.d)
        return True


class SafetyGymWrapper(gym.Wrapper):
    """
    safety-gymnasium returns a 6-tuple including cost.
    This wrapper extracts cost into info and presents a standard 5-tuple
    to stable-baselines3.
    """
    def __init__(self, env):
        super().__init__(env)
        self._ep_cost = 0.0

    def reset(self, **kwargs):
        self._ep_cost = 0.0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        self._ep_cost += cost
        info["cost"] = cost
        info["ep_cost"] = self._ep_cost
        return obs, reward, terminated, truncated, info


def make_safe_env(env_id: str = "SafetyPointGoal1-v0"):
    raw_env = safety_gymnasium.make(env_id)
    return SafetyGymWrapper(raw_env)


if __name__ == "__main__":
    env = make_safe_env("SafetyPointGoal1-v0")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./logs/ppo_lagrangian/",
    )

    lagrangian_cb = LagrangianCallback(
        constraint_threshold=25.0,   # allow up to 25 cost units per episode
        lr_lambda=0.05,
    )

    model.learn(
        total_timesteps=2_000_000,
        callback=lagrangian_cb,
    )
    model.save("ppo_lagrangian_pointgoal")
    print("Training complete. Check TensorBoard: tensorboard --logdir logs/")
```

### Wie die Lagrangian-Strafe in den Policy-Gradienten kommt

Das Primal-Update minimiert:

```
loss = -J(π) + λ · C_batch
```

Wobei `C_batch` die mittleren Kosten über den aktuellen Rollout-Batch ist. Im Callback oben wird `lam` auf der Dual-Seite nach jedem Rollout aktualisiert. Um es in den Policy-Loss einzuhängen, erweitere die `train()`-Methode der SB3-PPO-Policy:

```python
class LagrangianPPO(PPO):
    def __init__(self, *args, lagrange_lambda: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lagrange_lambda = lagrange_lambda

    def train(self) -> None:
        # Collect batch costs from rollout buffer infos
        costs = np.array([
            info.get("cost", 0.0)
            for info in self.rollout_buffer.infos
            if isinstance(info, dict)
        ])
        mean_cost = float(np.mean(costs)) if len(costs) > 0 else 0.0

        # Standard PPO update, but add λ·C to the policy loss
        # (Override _compute_loss in a real implementation; this is the conceptual sketch)
        super().train()

        # Dual update
        self.lagrange_lambda = max(
            0.0,
            self.lagrange_lambda + 0.05 * (mean_cost - self.constraint_threshold)
        )
```

Für eine produktionsreife Implementierung siehe [safety-baselines3](https://github.com/PKU-Alignment/safety-gymnasium) oder [FSRL (Tianshou-basiert)](https://github.com/liuzuxin/FSRL), die PPO-Lagrangian mit der Rollout-Buffer-Integration bereits korrekt umsetzen.

---

## 6 · safety-gymnasium — der Standard-Benchmark

**safety-gymnasium** ist die Standard-Benchmark-Suite für Safe-RL-Forschung. Sie liefert ein Set gut verstandener Umgebungen, in denen du Algorithmen auf einem neutralen Spielfeld vergleichen kannst, bevor du auf deinen eigenen Godot-Umgebungen testest.

```bash
pip install safety-gymnasium
```

### Kanonische Aufgaben

| Umgebung | Agent | Gefahren | Aufgabe |
|-------------|-------|---------|------|
| `SafetyPointGoal1-v0` | 2D-Punkt | Zu meidende Kreise | Ziel-Sphäre erreichen |
| `SafetyPointGoal2-v0` | 2D-Punkt | Mehr/härtere Gefahren | Ziel-Sphäre erreichen |
| `SafetyCarGoal1-v0` | Auto (4 Räder) | Kreise + Säulen | Ziel-Sphäre erreichen |
| `SafetyAntGoal1-v0` | MuJoCo-Ant | Kreise | Ziel-Sphäre erreichen |
| `SafetyPointButton1-v0` | 2D-Punkt | Kreise + Gremlins | Button drücken |

Die Namenskonvention: `Safety<Agent><Task><Difficulty>-v0`.

Alle safety-gymnasium-Umgebungen geben ein 6-Tupel zurück:

```python
obs, reward, cost, terminated, truncated, info = env.step(action)
```

Wobei `cost ∈ {0.0, 1.0}` pro Schritt: 1,0, wenn der Agent in diesem Schritt eine Gefahrenregion betreten hat, sonst 0,0.

### safety-gymnasium zur Kalibrierung nutzen

Bevor du deine eigene Godot-Safe-RL-Umgebung baust, lass PPO mit fester Strafe und dann PPO-Lagrangian auf `SafetyPointGoal1-v0` laufen. Plotte die Verletzungsrate für beide. Der Abstand dazwischen ist die Lücke, die deine CMDP-Formulierung schließt. Das gibt dir eine Kalibrierungs-Baseline, bevor du in eine eigene Umgebung investierst.

```python
import safety_gymnasium
env = safety_gymnasium.make("SafetyPointGoal1-v0")
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, cost, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

---

## 7 · Godot-Implementierung

In Godot erfordert die Safe-RL-Implementierung, dass die Umgebung zwei Signale zurückgibt: den Task-Reward und das Kostensignal. Das Kostensignal ist strukturell vom Reward getrennt — es wird nicht vom Reward subtrahiert, sondern ist ein zweiter Kanal, den der Trainings-Wrapper unabhängig liest.

### Das AIController-Muster

```gdscript
# ai_controller.gd — two-signal return pattern

extends AIController3D

# Cost accumulates within the episode; training wrapper reads it at rollout end
var step_cost: float = 0.0
var episode_cost: float = 0.0

func get_obs() -> Dictionary:
    var obs = []
    # ... build observation vector as usual ...
    return {"obs": obs}

func _physics_process(delta: float) -> void:
    # Compute task reward
    var task_reward: float = _compute_task_reward()
    reward += task_reward

    # Compute cost signal (separate from reward)
    step_cost = _compute_cost()
    episode_cost += step_cost

    # Expose cost through the info dict
    # godot-rl-agents passes this back to the Python side as part of the step return
    # Access it with: info = env.step(action)[4]  (the info dict in gymnasium)
    set_heuristic("cost", step_cost)

func _compute_task_reward() -> float:
    var dist = global_position.distance_to(goal.global_position)
    var reward = 1.0 - clampf(dist / MAX_DIST, 0.0, 1.0)
    if dist < GOAL_RADIUS:
        reward += 5.0
        done = true
    return reward

func _compute_cost() -> float:
    var cost = 0.0
    # Hazard zone entry
    for hazard in hazards:
        if global_position.distance_to(hazard.global_position) < HAZARD_RADIUS:
            cost = 1.0
            break
    # Velocity limit (e.g. near a human proxy object)
    if linear_velocity.length() > MAX_SAFE_SPEED:
        cost += 0.5
    return cost
```

### Szenen-Struktur der Umgebung

```
SafeRLEnv (Node3D)
├── Agent (CharacterBody3D or RigidBody3D)
│   └── AIController3D       ← returns reward + cost
├── Goal (Node3D)             ← randomized position each episode
├── Hazards (Node3D)
│   ├── Hazard_0 (Area3D)    ← cost = 1.0 when agent overlaps
│   ├── Hazard_1 (Area3D)
│   └── ...
└── NavigableFloor (StaticBody3D)
```

### Python-seitiger Wrapper

```python
# godot_safe_wrapper.py
from godot_rl.wrappers.stable_baselines_wrapper import StableBaseline3Wrapper
import numpy as np


class GodotSafeWrapper(StableBaseline3Wrapper):
    """
    Extend the standard godot-rl SB3 wrapper to expose the cost signal
    from ai_controller.step_cost so it can be consumed by PPO-Lagrangian.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ep_cost = 0.0

    def reset(self, **kwargs):
        self._ep_cost = 0.0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # godot-rl places agent info under info["observations"]
        # Our ai_controller exposes step_cost via set_heuristic("cost", ...)
        step_cost = float(info.get("cost", 0.0))
        self._ep_cost += step_cost
        info["cost"] = step_cost
        info["ep_cost"] = self._ep_cost
        return obs, reward, terminated, truncated, info


# Training script
from godot_rl.core.godot_env import GodotEnv

base_env = GodotEnv(env_path="res://safe_rl_env.tscn")
env = GodotSafeWrapper(base_env)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/godot_safe/")
lagrangian_cb = LagrangianCallback(constraint_threshold=5.0, lr_lambda=0.02)
model.learn(total_timesteps=1_000_000, callback=lagrangian_cb)
```

### Kosten in TensorBoard verdrahten

Der `LagrangianCallback` aus Abschnitt 5 loggt bereits `constraint/lambda`, `constraint/ep_cost_mean` und `constraint/threshold`. Ergänze in deinem Godot-Wrapper außerdem:

```python
# In LagrangianCallback._on_rollout_end()
ep_lengths = [ep_info["l"] for ep_info in self.model.ep_info_buffer]
if ep_lengths:
    avg_len = np.mean(ep_lengths)
    # Violation rate: fraction of steps where cost > 0
    violation_rate = ep_costs / avg_len if avg_len > 0 else 0.0
    self.logger.record("constraint/violation_rate", violation_rate)
```

---

## 8 · Praktische Leitlinien — wann CMDP verwenden

Die CMDP-Formulierung erhöht die Komplexität. Sie führt ein zweites Signal, eine duale Variable zum Tunen und potenziell einen neuen Algorithmus ein. Hier ist, wann diese Kosten gerechtfertigt sind:

### CMDP / Safe RL verwenden, wenn

| Situation | Warum CMDP |
|-----------|----------|
| Deployment auf echter Hardware mit physikalischen Limits (Drehmoment, Geschwindigkeit, Kraft) | Physikalische Limits können Hardware beschädigen; λ liefert automatische Durchsetzung |
| Menschen in der Umgebung (Cobots, soziale Roboter) | Verletzungskosten sind inakzeptabel; du brauchst eine Garantie, keinen Mittelwert |
| Multi-Objective-Aufgaben, bei denen ein Ziel absolut halten muss | Reward-Skalarisierung kann strikte Prioritäten nicht abbilden |
| Constraint-Grenze ändert sich häufig | `d` zu justieren ist interpretierbar; einen Strafkoeffizienten zu justieren nicht |
| Du musst Compliance auditieren („nie 50 N Kontaktkraft überschritten") | Die Kostenfunktion C(π) liefert eine direkte Audit-Metrik; Penalty-Blending nicht |

### Reward-Strafen verwenden, wenn

| Situation | Warum Penalty in Ordnung ist |
|-----------|---------------------|
| Nur Sim-Training, Constraint-Verletzung hat keine reale Konsequenz | Tuning-Bequemlichkeit überwiegt prinzipielle Formulierung |
| Weiche Präferenzen („lieber auf der Straße bleiben, aber Recovery ist OK") | Präferenz-Tradeoffs sind in einer Reward-Funktion natürlich |
| Frühes Prototyping, bevor der Constraint-Schwellwert präzise definiert ist | CMDP setzt voraus, dass du `d` kennst; Penalty erlaubt dir, das Verhalten zuerst zu erkunden |
| Sehr einfache Single-Constraint-Probleme, wo Koeffizienten-Tuning handhabbar ist | Penalty fügt keinen Overhead hinzu; CMDP bringt Dual-Variablen-Bookkeeping mit |

### Den Constraint-Schwellwert `d` wählen

Der Schwellwert `d` sollte aus Fachexpertise oder physikalischen Limits kommen, nicht aus Hyperparameter-Suche:

- **Physikalisches Limit:** „Motor überhitzt, wenn das momentane Drehmoment 12 N·m länger als 0,5 s überschreitet." Übersetze in Kosten-pro-Schritt und setze `d` entsprechend.
- **Regulatorisch:** „Kontaktkraft auf Mensch darf 150 N nicht überschreiten (ISO/TS 15066)." Eine Kosteneinheit pro Verletzung, `d = 0` (null Verletzungen pro Episode erlaubt).
- **Operativ:** „Roboter darf höchstens 5 % der Episodenschritte in der Sperrzone sein." `d = 0,05 × episode_length`.

Hast du kein Domänenwissen, um `d` zu setzen, nutze die Standard-Benchmarks von safety-gymnasium als Kalibrierungswerkzeug. Der community-akzeptierte Schwellwert für `SafetyPointGoal1` ist `d = 25,0` Kosteneinheiten pro Episode.

---

## 9 · Viz-Checkpoint — Constraint-Verletzungsrate in TensorBoard

Das Training ist nicht abgeschlossen, bis du verifiziert hast, dass beide Signale sich korrekt verhalten. Geh diese Checkliste durch, bevor du eine sichere Policy beanspruchst.

**Schritt 1: Verifiziere, dass die Kostenfunktion korrekt auslöst.**

```python
# Quick sanity check — run a random policy and count cost events
env = make_safe_env()
obs, _ = env.reset()
total_steps = 0
total_cost = 0
for _ in range(10_000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_cost += info.get("cost", 0.0)
    total_steps += 1
    if terminated or truncated:
        obs, _ = env.reset()

print(f"Random policy violation rate: {total_cost / total_steps:.3f}")
# Expected: 0.1 – 0.4 for PointGoal1 with random actions
# If 0.0: cost function is not firing — check env wrapper
# If 1.0: cost fires every step — threshold or geometry is wrong
```

**Schritt 2: Starte Training und beobachte die Kurven.**

Öffne in TensorBoard `constraint/violation_rate`, `rollout/ep_rew_mean` und `constraint/lambda` auf derselben Seite. Ein korrekt konvergierender Lauf sieht so aus:

```
timesteps →    0         500k        1M         2M

ep_rew_mean    ▁▁▂▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▇▇▇  ← rises steadily
violation_rate ████▇▇▆▅▄▃▃▂▂▁▁▁▁▁▁▁▁  ← falls and stabilizes
lambda         ▁▁▂▃▃▃▃▃▂▂▂▂▁▁▁▁▁▁▁▁▁  ← peaks then settles
```

Achte auf diese Fehlermuster:

- **Reward steigt, violation_rate bleibt flach (hoch):** λ steigt nicht — prüfe, ob das Dual-Update aufgerufen wird und `lr_lambda > 0`.
- **Violation_rate fällt auf null, Reward bleibt flach:** Der Agent hat gelernt, alle Gefahren durch Stillstehen zu vermeiden. Senke `d` oder prüfe, ob der Task-Reward stark genug ist, den Agenten zum Ziel zu ziehen.
- **Lambda divergiert (steigt unbegrenzt weiter):** Die Bedingung ist nicht erreichbar. Entweder ist `d` zu klein oder Aufgabe und Bedingung stehen in fundamentalem Konflikt — der Agent kann das Ziel nicht erreichen, ohne durch Gefahren zu gehen. Umgebung neu gestalten oder `d` erhöhen.
- **Beide Kurven oszillieren ohne Trend:** `lr_lambda` ist zu groß. Halbiere und trainiere erneut.

**Schritt 3: Mit der Penalty-Baseline vergleichen.**

Lass dieselbe Umgebung mit einer festen `-100`-Strafe statt PPO-Lagrangian laufen. Plotte die finalen Verteilungen von violation_rate über 10 Eval-Episoden für beide. Der PPO-Lagrangian-Lauf sollte eine engere Verteilung und eine niedrigere mittlere Verletzungsrate bei gleichem Task-Reward-Niveau zeigen.

---

## 10 · Stretch Goals

Wenn du diese Unit vor der nächsten weiter vertiefen willst, hier vier Übungen, die echte Safe-RL-Engineering-Arbeit spiegeln:

- **Volles LagrangianPPO von Grund auf implementieren.** Statt den Callback-Hook zu nutzen, leite SB3s PPO ab und überschreibe `train()`, um die Lagrangian-Strafe direkt zum Policy-Gradienten zu addieren. Verifiziere, dass der Gradient nun zwei Terme hat — Task-Advantage und skalierten Cost-Advantage. Vergleiche die Konvergenzgeschwindigkeit mit dem Callback-Ansatz.
- **Multi-Constraint-CMDP.** Füge der Godot-Umgebung eine zweite Bedingung hinzu — etwa Gefahrenzone *und* Geschwindigkeitslimit. Führe separate `λ₁`, `λ₂` für jede Bedingung ein und aktualisiere sie unabhängig. Wie interagieren die beiden Multiplikatoren? (Sollten sie nicht — die CMDP-Formulierung hält sie entkoppelt.)
- **CPO vs. PPO-Lagrangian-Vergleich auf PointGoal1.** Installiere `safety-starter-agents` und lass beide Algorithmen jeweils 2 M Schritte auf `SafetyPointGoal1-v0` laufen. Plotte die Pareto-Front aus (Task-Reward, Verletzungsrate) am Ende des Trainings. CPO sollte unten rechts erscheinen (niedrigere Verletzungen, vergleichbarer Reward).
- **Safety-gymnasium-Transfer.** Trainiere PPO-Lagrangian auf `SafetyPointGoal1-v0`, bis violation_rate < 0,05. Baue dann dieselbe Aufgabe in Godot nach (flacher Boden, Gefahrenkreise, Ziel-Sphäre) und evaluiere, ob die Policy-Struktur generalisiert. Welche Änderungen an der Obs-Normalisierung brauchst du für einen sauberen Transfer?

---

## Was kommt als Nächstes

Du hast jetzt das mathematische Werkzeug und die Implementierungs-Tools für harte Sicherheits-Constraints in RL:

- Die CMDP-Formulierung gibt Constraints eine physikalische Interpretation, die Strafkoeffizienten nicht haben können
- Lagrangian-Relaxation wandelt constrained Optimierung in ein adaptives Dual-Update um
- PPO-Lagrangian setzt das in der Praxis mit minimalen Änderungen am SB3-Trainings-Loop um
- Die Kostenfunktion sitzt neben — nicht in — der Reward-Funktion deiner Godot-Umgebung

Safe RL ist ein aktives Forschungsfeld. Die Techniken hier (CPO, PPO-Lagrangian, Lagrangian TRPO) sind Methoden der ersten Generation. Aktuelle Forschung erkundet Offline Safe RL (Lernen aus Daten ohne Verletzungen), Garantien für sichere Erkundung und die Kombination von Ljapunow-Stabilitätstheorie mit RL. Das CMDP-Framework aus dieser Unit ist die gemeinsame Grundlage für alles davon.

---

[→ Kursstartseite](index.md)
