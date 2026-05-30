# PPO von Grund auf — CleanRL und die Implementierungsschicht

[← Anwenden — SAC vs PPO auf JumperHard](unit-sac-applied.md) · [Kursstartseite](index.md)

!!! note "Voraussetzungen"
    - **[Anwenden — SAC vs PPO auf JumperHard](unit-sac-applied.md)** — du hast den PPO-vs-SAC-Vergleich in der Praxis gesehen
    - **[PPO Deep Dive](unit-ppo-deep.md)** — das geclippte Objektiv und GAE-λ sollten sitzen
    - **[Actor-Critic](unit-actor-critic.md)** — die Actor-/Critic-Aufteilung
    - **[Unit 4](unit-04.md)** — PPO-Hyperparameter mit SB3 einmal getunt zu haben
    - PyTorch-Sicherheit (Module, Optimizer, Autograd) — CleanRL ist Single-File-PyTorch

!!! info "Zeit"
    Lesen: ~50 min · Training: ~30 min GPU / ~2 h CPU

---

!!! info "Drei Wege, deine KI zu beobachten"
    CartPole-Trainingskurve, die in unter 200k Schritten bei 500 spitzt · die echten Loss-Kurven (`policy_loss`, `value_loss`, `entropy_loss`) alle in einem TensorBoard-Lauf · die CleanRL-Quelldatei in deinem Editor offen — jede Zeile lesbar, keine Abstraktion versteckt etwas

---

## 0 · Warum den Code lesen?

Du hast Units damit verbracht, über PPO zu lesen. Du hast SB3s PPO auf einem Dutzend Godot-Umgebungen laufen lassen. Du weißt theoretisch, was `clip_range`, `gae_lambda` und `vf_coef` tun. Aber es gibt eine Lücke zwischen *die Gleichung kennen* und *die Implementierung kennen*, und diese Lücke ist wichtig, sobald Dinge schiefgehen.

**SB3s PPO erstreckt sich über 15+ Dateien.** Hier eine Teilliste:

- `on_policy_algorithm.py` — die Basis-Trainingsschleife
- `ppo.py` — der PPO-spezifische Loss
- `policies.py` — die Netzwerk-Architektur
- `buffers.py` — Rollout-Speicherung
- `type_aliases.py`, `utils.py`, `callbacks.py` — unterstützende Maschinerie

Keine dieser Dateien ist lang. Jede ist sauber und gut dokumentiert. Aber um zu verstehen, *wie* ein PPO-Update vom Rollout-Start bis zum Gradientenschritt-Abschluss passiert, musst du sie alle lesen, den Call-Graph verfolgen, verstehen, welche Klasse von was erbt, und den Daten durch sechs Typtransformationen folgen. Es kostet einen Tag.

**CleanRLs PPO ist eine einzelne `ppo.py`-Datei, rund 300 Zeilen.** Lies sie einmal — 30 Minuten — und du verstehst den gesamten Algorithmus. Jede Zeile ist entweder Datensammlung, Vorteilsberechnung oder ein Loss-Term, den du direkt auf eine Gleichung abbilden kannst.

Diese Unit baut die Brücke von der Theorie, die du in [unit-ppo-deep.md](unit-ppo-deep.md) aufgebaut hast, zu einer echten, laufenden, hackbaren Implementierung. Nach dieser Unit solltest du folgendes können:

- CleanRLs PPO auf CartPole laufen lassen und die TensorBoard-Ausgabe lesen.
- Einer Kollegin Zeile für Zeile durch jeden Block von `ppo.py` führen.
- Die Rückwärts-GAE-Schleife ohne Notizen erklären.
- CleanRL patchen, um eine Godot-Umgebung zu umhüllen.
- Drei konkrete algorithmische Modifikationen vornehmen und ihre Effekte beobachten.

---

## 1 · CleanRL-Setup

### Installation

```bash
# CleanRL is not distributed as a PyPI package — clone the repo and install from source:
git clone https://github.com/vwxyzjn/cleanrl.git
cd cleanrl
pip install -r requirements/requirements.txt
# For Atari support add:
pip install -r requirements/requirements-atari.txt
```

CleanRLs `requirements/`-Verzeichnis enthält Anforderungsdateien pro Feature. Wenn du bereits eine Kurs-Conda-Umgebung hast, installiere hinein — es gibt keine Konflikte mit deiner bestehenden SB3-Installation.

### Den CartPole-Baseline laufen lassen

```bash
python cleanrl/ppo.py \
    --env-id CartPole-v1 \
    --total-timesteps 500000 \
    --learning-rate 2.5e-4 \
    --num-envs 4 \
    --num-steps 128 \
    --num-minibatches 4 \
    --update-epochs 4
```

Du siehst eine Ausgabe wie:

```
global_step=4096,  episodic_return=23.0
global_step=8192,  episodic_return=67.0
global_step=40960, episodic_return=317.0
global_step=81920, episodic_return=500.0
```

Und TensorBoard unter `http://localhost:6006` (führe `tensorboard --logdir runs/` aus).

### CleanRL-TensorBoard auf SB3-TensorBoard abbilden

Du kennst SB3s Metriken bereits aus früheren Units. Hier die direkte Entsprechung:

| SB3-Metrik | CleanRL-Metrik | Anmerkungen |
|---|---|---|
| `train/policy_gradient_loss` | `losses/policy_loss` | Gleiche Größe, anderer Key |
| `train/value_loss` | `losses/value_loss` | Mittlerer quadrierter Fehler auf Wertvorhersagen |
| `train/entropy_loss` | `losses/entropy` | Negative Entropie (SB3 negiert sie) |
| `train/approx_kl` | `losses/approx_kl` | KL-Proxy; beobachte Spikes über 0,02 |
| `train/clip_fraction` | `losses/clipfrac` | Anteil der Transitionen, bei denen Clipping feuerte |
| `rollout/ep_rew_mean` | `charts/episodic_return` | Die, auf die du immer zuerst schaust |
| `rollout/ep_len_mean` | `charts/episodic_length` | Nützlich für Aufgaben mit variabler Episodenlänge |
| `time/fps` | `charts/SPS` | Schritte pro Sekunde-Durchsatz |

!!! tip "SPS vs. FPS"
    CleanRL loggt `SPS` (samples per second) statt `FPS`. In Godot-Umgebungen, wo jeder Schritt langsamer ist, wirst du SPS deutlich sinken sehen — das ist erwartet. Die Zahl, die du optimieren willst, ist `charts/episodic_return` pro Wall-Clock-Minute, nicht pro Schritt.

---

## 2 · Der PPO-Datei-Durchgang

Was folgt ist ein abschnittsweiser Durchgang durch CleanRLs `ppo.py`. Der gezeigte Code ist sehr nah an der echten Datei, aber leicht vereinfacht für Lesbarkeit. Die gesamte wichtige Logik ist erhalten.

### 2.1 Argument-Parsing und Seeding

```python
import argparse, random, time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id",           type=str,   default="CartPole-v1")
    parser.add_argument("--total-timesteps",  type=int,   default=500_000)
    parser.add_argument("--learning-rate",    type=float, default=2.5e-4)
    parser.add_argument("--num-envs",         type=int,   default=4)
    parser.add_argument("--num-steps",        type=int,   default=128)
    parser.add_argument("--num-minibatches",  type=int,   default=4)
    parser.add_argument("--update-epochs",    type=int,   default=4)
    parser.add_argument("--clip-coef",        type=float, default=0.2)
    parser.add_argument("--ent-coef",         type=float, default=0.01)
    parser.add_argument("--vf-coef",          type=float, default=0.5)
    parser.add_argument("--max-grad-norm",    type=float, default=0.5)
    parser.add_argument("--gae-lambda",       type=float, default=0.95)
    parser.add_argument("--gamma",            type=float, default=0.99)
    parser.add_argument("--seed",             type=int,   default=1)
    return parser.parse_args()

args = parse_args()
batch_size     = args.num_envs * args.num_steps        # total transitions per update
minibatch_size = batch_size // args.num_minibatches    # transitions per mini-batch
num_updates    = args.total_timesteps // batch_size    # how many update cycles

# Seeding — reproducibility matters for debugging
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
```

**Was zu beachten ist:** `batch_size` ist `num_envs × num_steps`. Mit 4 Envs und 128 Schritten erhältst du 512 Transitionen pro Rollout. Diese 512 Transitionen werden geshuffelt und in 4 Minibatches von 128 gesplittet. Die Update-Schleife läuft über alle 4 Minibatches, 4 Mal (`update_epochs`), insgesamt 16 Gradientenschritte pro Rollout. Das mappt exakt auf SB3s `n_steps=128`, `n_epochs=4`, `batch_size=128`.

### 2.2 Umgebungs-Setup

```python
def make_env(env_id, seed, idx, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env
    return thunk

envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, args.seed, i, run_name) for i in range(args.num_envs)]
)
```

`RecordEpisodeStatistics` umhüllt jede Umgebung, sodass `info["episode"]["r"]` und `info["episode"]["l"]` am Ende jeder Episode gefüllt werden. CleanRL liest die, um `episodic_return` und `episodic_length` zu loggen. Das ist dasselbe Muster, das wir im Godot-Wrapper-Abschnitt später nutzen.

`SyncVectorEnv` führt die Umgebungen hintereinander in einem einzelnen Prozess aus. Die Alternative, `AsyncVectorEnv`, lässt sie in parallelen Subprozessen laufen — schneller, aber schwerer zu debuggen. Beginne mit `Sync`.

### 2.3 Netzwerk-Architektur

```python
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization, a CleanRL signature choice."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = envs.single_action_space.n  # discrete case

        # Critic network: outputs a single scalar V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Actor network: outputs logits over actions
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist   = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)
```

**Wichtige Designentscheidungen:**

- **Getrennte Actor- und Critic-Heads, nichts geteilt.** Anders als viele Implementierungen, die den Trunk-MLP teilen, hält CleanRLs Default sie unabhängig. Das vermeidet Gradienten-Interferenz zwischen den Policy- und Wert-Objektiven. Siehe [unit-ppo-deep.md §8](unit-ppo-deep.md) für die Trade-off-Diskussion.
- **Orthogonale Initialisierung** mit `std=np.sqrt(2)` für versteckte Schichten, `std=0.01` für die Actor-Ausgabe (hält initiale Aktionswahrscheinlichkeiten nahe-uniform), `std=1.0` für die Critic-Ausgabe (keine Schrumpfung auf der Wert-Skala).
- **`Tanh`-Aktivierungen.** CleanRL nutzt Tanh, nicht ReLU, für Aufgaben mit diskreten Aktionen. Tanh ist beschränkt, was sich besser mit orthogonaler Init verträgt.

!!! note "Kontinuierliche Aktionen"
    Für kontinuierliche Steuerung gibt der Actor einen Mittelwert und log-std aus (entweder als Parameter oder als Netzwerk-Ausgabe). CleanRL hat `ppo_continuous_action.py` mit genau dieser Änderung. Der Rest des Algorithmus ist identisch.

### 2.4 Rollout-Speicherung

```python
# Pre-allocate buffers — fill them in the collection loop
obs     = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape)
actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape)
logprobs= torch.zeros((args.num_steps, args.num_envs))
rewards = torch.zeros((args.num_steps, args.num_envs))
dones   = torch.zeros((args.num_steps, args.num_envs))
values  = torch.zeros((args.num_steps, args.num_envs))
```

Form: `(num_steps, num_envs, ...)`. Zeit verläuft entlang Achse 0, Umgebungen entlang Achse 1. Nach dem Rollout flachen wir beide in einen einzelnen `(batch_size, ...)`-Tensor für die Update-Schleife.

### 2.5 Rollout-Sammelschleife

```python
next_obs  = torch.Tensor(envs.reset()[0])   # (num_envs, obs_dim)
next_done = torch.zeros(args.num_envs)

for update in range(1, num_updates + 1):

    # --- ROLLOUT PHASE ---
    for step in range(args.num_steps):
        obs[step]  = next_obs
        dones[step]= next_done

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)

        actions[step]  = action
        logprobs[step] = logprob
        values[step]   = value.flatten()

        next_obs_np, reward, terminated, truncated, infos = envs.step(action.numpy())
        done = np.logical_or(terminated, truncated)

        rewards[step] = torch.tensor(reward)
        next_obs      = torch.Tensor(next_obs_np)
        next_done     = torch.Tensor(done)

        # Log completed episodes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    writer.add_scalar("charts/episodic_return",
                                      info["episode"]["r"], global_step)
```

Die Rollout-Schleife ist geradlinig: speichere für jeden Schritt den aktuellen Zustand, sample eine Aktion, stepe die Umgebung, speichere die Belohnung. Beachte, dass `torch.no_grad()` den Forward-Pass umhüllt — wir sammeln nur Daten, berechnen noch keine Gradienten. Die Gradienten kommen später in der Update-Schleife.

---

## 3 · GAE-Deep-Dive

Hier verwirren sich die meisten Studierenden. Lies diesen Abschnitt langsam.

### Warum GAE?

Nach dem Sammeln eines Rollouts brauchen wir Vorteilsschätzungen A_t für jede Transition. Die einfachste Option ist der **TD-Fehler**:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
```

Das hat niedrige Varianz, aber hohen Bias — wir schauen nur einen Schritt voraus. Der volle **Monte-Carlo-Return** ist unverzerrt, aber hat hohe Varianz — wir summieren Belohnungen bis zum Episodenende, was verrauscht ist.

**GAE (Generalized Advantage Estimation)** aus [unit-ppo-deep.md §6](unit-ppo-deep.md) interpoliert zwischen beiden mit Parameter λ:

```
A_t^GAE = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

Bei λ=0 kollabiert das auf reinen TD-Fehler. Bei λ=1 approximiert es den vollen Monte-Carlo-Vorteil. λ=0,95 ist der Default — nahe an Monte Carlo, aber mit gebändigter Varianz.

### Die Rückwärtsschleife

Die obige Gleichung ist eine Summe über zukünftige TD-Fehler. Sie vorwärts zu berechnen würde verlangen, dass du alle zukünftigen Werte verfügbar hast, bevor du A_t berechnen kannst. Der Trick: **die Summe hat eine rekursive Struktur**.

```
A_t^GAE = δ_t + γλ · A_{t+1}^GAE
```

Rechts-nach-links gelesen: der Vorteil bei Schritt t ist der TD-Fehler bei Schritt t, plus der diskontierte Vorteil bei Schritt t+1. Das heißt, du kannst alle Vorteile in einem Rückwärts-Sweep berechnen — starte beim letzten Schritt und arbeite dich zurück zum ersten.

### Der echte Code

```python
with torch.no_grad():
    next_value = agent.get_value(next_obs).reshape(1, -1)  # V(s_T)
    advantages = torch.zeros_like(rewards)  # (num_steps, num_envs)
    last_gae_lam = 0.0

    # Iterate BACKWARDS: step T-1 down to 0
    for t in reversed(range(args.num_steps)):

        if t == args.num_steps - 1:
            # At the last stored step, the "next" observation is next_obs
            nextnonterminal = 1.0 - next_done        # 0 if episode just ended
            nextvalues      = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]    # 0 if episode ended at t+1
            nextvalues      = values[t + 1]

        # TD error for this step
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]

        # GAE recursion: A_t = δ_t + γλ · (1 - done_{t+1}) · A_{t+1}
        advantages[t] = last_gae_lam = (
            delta + args.gamma * args.gae_lambda * nextnonterminal * last_gae_lam
        )

    # Returns = advantages + values (used as targets for value loss)
    returns = advantages + values
```

**Zeile-für-Zeile-Erklärung:**

1. `nextnonterminal` — wenn eine Episode endet, ist der Zukunftswert null (es gibt keinen nächsten Zustand). Multiplikation mit `(1 - done)` nullt den Bootstrap-Wert an Episodengrenzen aus. Das ist der häufigste Bug, wenn man GAE selbst implementiert: vergessen, Werte an Terminalzuständen zu maskieren.

2. `delta` — der TD-Fehler: Belohnung bei Schritt t, plus diskontierter Nächstwert (falls nicht terminal), minus aktuelle Wertschätzung.

3. `last_gae_lam` — das ist A_{t+1}^GAE aus der Rekursion. Bei der ersten (rückwärts laufenden) Iteration ist es null (es gibt nichts nach Schritt T). Jede Iteration aktualisiert es auf das aktuelle A_t, das zum A_{t+1} für die nächste (rückwärts laufende) Iteration wird.

4. `returns = advantages + values` — das Wertziel. Da `advantage = return - value`, haben wir `return = advantage + value`. Diese Returns werden zum Ziel für den Wert-Loss: wir wollen, dass der Critic den Return vorhersagt, nicht nur den TD-Fehler.

!!! warning "Der häufigste GAE-Bug"
    `nextnonterminal` an Episodengrenzen vergessen. Wenn du an einem Terminalzustand von V(s_{t+1}) bootstrappst, addierst du Wert, der nicht existiert — die Episode ist vorbei. Setze den Bootstrap-Wert immer auf null, wenn `done[t+1]` True ist.

### λ=0 vs. λ=0,95: eine numerische Illustration

Betrachte eine 4-Schritt-Episode mit Belohnungen `[0, 0, 0, 1]` und allen Werten auf 0 geschätzt (ein frisches, untrainiertes Netz). γ=0,99, V(s_4)=0.

**TD-Fehler (λ=0):**
```
δ_3 = 1 + 0.99·0 - 0 = 1.0
δ_2 = 0 + 0.99·0 - 0 = 0.0
δ_1 = 0 + 0.99·0 - 0 = 0.0
δ_0 = 0 + 0.99·0 - 0 = 0.0

A_3=1.0,  A_2=0.0,  A_1=0.0,  A_0=0.0
```
Nur der letzte Schritt bekommt ein Gradientensignal. Die frühen Zustände lernen nichts über die eventuelle Belohnung.

**GAE (λ=0,95):**
```
A_3 = 1.0
A_2 = 0.0 + 0.99·0.95·1.0 = 0.940
A_1 = 0.0 + 0.99·0.95·0.940 = 0.884
A_0 = 0.0 + 0.99·0.95·0.884 = 0.831
```
Alle vier Schritte haben ein Gradientensignal. Die frühen Zustände lernen, dass sie auf einem Pfad zur Belohnung waren.

Deshalb ist λ in Sparse-Reward-Umgebungen so wichtig — wie Godot-Agenten, die nur eine Belohnung bekommen, wenn sie das Ziel erreichen. Mit λ=0 muss der Agent Glück haben und genau bei dem Schritt sein, der der Belohnung vorausgeht, damit das Update überhaupt etwas tut. Mit λ=0,95 propagiert die Anrechnung rückwärts durch die gesamte Trajektorie.

---

## 4 · Die PPO-Update-Schleife

```python
# Flatten the rollout dimensions: (num_steps, num_envs) → (batch_size,)
b_obs       = obs.reshape((-1,) + envs.single_observation_space.shape)
b_logprobs  = logprobs.reshape(-1)
b_actions   = actions.reshape((-1,) + envs.single_action_space.shape)
b_advantages= advantages.reshape(-1)
b_returns   = returns.reshape(-1)
b_values    = values.reshape(-1)

# Normalize advantages within the mini-batch (reduces variance)
b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

clipfracs = []

for epoch in range(args.update_epochs):
    # Shuffle indices to decorrelate mini-batches
    b_inds = np.random.permutation(batch_size)

    for start in range(0, batch_size, minibatch_size):
        end  = start + minibatch_size
        mb_inds = b_inds[start:end]

        # Forward pass with current (updating) policy
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
            b_obs[mb_inds], b_actions[mb_inds]
        )

        # Importance sampling ratio: π_new(a|s) / π_old(a|s)
        logratio   = newlogprob - b_logprobs[mb_inds]
        ratio      = logratio.exp()

        # Approximate KL (cheap proxy — no need for exact KL)
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

        mb_advantages = b_advantages[mb_inds]

        # --- CLIPPED POLICY LOSS (§4 of unit-ppo-deep.md) ---
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

        # --- VALUE LOSS ---
        newvalue = newvalue.view(-1)
        v_loss   = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

        # --- ENTROPY BONUS (encourages exploration) ---
        entropy_loss = entropy.mean()

        # --- COMBINED LOSS ---
        loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

# Log to TensorBoard
writer.add_scalar("losses/policy_loss",  pg_loss.item(),     global_step)
writer.add_scalar("losses/value_loss",   v_loss.item(),      global_step)
writer.add_scalar("losses/entropy",      entropy_loss.item(),global_step)
writer.add_scalar("losses/approx_kl",    approx_kl.item(),   global_step)
writer.add_scalar("losses/clipfrac",     np.mean(clipfracs), global_step)
```

**Jeden Term auf unit-ppo-deep.md mappen:**

| Code | Gleichung | Unit-Abschnitt |
|---|---|---|
| `ratio = (newlogprob - b_logprobs).exp()` | r_t(θ) = π_θ / π_{θ_old} | §3 |
| `pg_loss1 = -mb_advantages * ratio` | L^CPI (ungeclippt) | §3 |
| `pg_loss2 = -mb_advantages * clamp(ratio, ...)` | L^CLIP-Term 2 | §4 |
| `pg_loss = max(pg_loss1, pg_loss2)` | min(·, ·) von L^CLIP | §4 |
| `v_loss = 0.5 * MSE(newvalue, returns)` | Wert-Loss-Term | §7 |
| `entropy_loss = entropy.mean()` | Entropie-Bonus H(π) | §7 |
| `clip_grad_norm_(...)` | Gradient-Clipping | §8 |

!!! note "Negations-Konvention"
    CleanRL negiert `pg_loss`, um Maximierung in Minimierung zu verwandeln (Standard-PyTorch-Idiom). Der Entropie-Term wird abgezogen (`- ent_coef * entropy_loss`), weil wir Entropie *maximieren* wollen, was *negative* Entropie zu minimieren heißt. Wenn du `entropy_loss` in TensorBoard sinken siehst, steigt die Entropie — das ist gut in der frühen Trainingsphase.

---

## 5 · Mit Godot verbinden

CleanRL hat keinen Godot-Wrapper, aber `godot_rl_agents` exponiert eine Gymnasium-kompatible Umgebung, die direkt in CleanRLs `make_env`-Factory passt.

### Die Godot-Umgebung umhüllen

```python
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
import gymnasium as gym

def make_godot_env(env_path, seed, idx, run_name, port=11008):
    """
    Factory returning a thunk that constructs a single Godot environment.
    CleanRL's SyncVectorEnv expects a list of such thunks.
    """
    def thunk():
        env = StableBaselinesGodotEnv(
            env_path=env_path,
            show_window=(idx == 0),    # only show window for the first env
            port=port + idx,           # each env needs its own port
            seed=seed + idx,
        )
        # Wrap with RecordEpisodeStatistics so CleanRL can log episodic_return
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def build_godot_envs(env_path, num_envs, seed, run_name):
    """
    Build a vectorized Godot env compatible with CleanRL's rollout loop.
    """
    envs = gym.vector.SyncVectorEnv([
        make_godot_env(env_path, seed, i, run_name, port=11008)
        for i in range(num_envs)
    ])
    # Sanity check: CleanRL expects Box or Discrete
    assert isinstance(
        envs.single_action_space,
        (gym.spaces.Discrete, gym.spaces.Box)
    ), f"Unexpected action space: {envs.single_action_space}"
    return envs


# Usage — drop this in place of the CartPole make_env call
if __name__ == "__main__":
    args = parse_args()
    run_name = f"godot__{args.env_id}__{args.seed}__{int(time.time())}"
    writer   = SummaryWriter(f"runs/{run_name}")

    envs  = build_godot_envs(
        env_path="path/to/your_game.x86_64",
        num_envs=args.num_envs,
        seed=args.seed,
        run_name=run_name,
    )
    agent = Agent(envs).to(device)
    # ... rest of the CleanRL training loop unchanged
```

!!! warning "Port-Konflikte"
    Jede Godot-Instanz braucht ihren eigenen Port. Mit `num_envs=4` brauchst du die Ports 11008–11011 offen. Wenn du beim Start `ConnectionRefusedError` siehst, prüfe, ob keine vorherigen Godot-Instanzen noch laufen (`pkill -f your_game.x86_64`).

!!! tip "Beginne mit num_envs=1"
    Multi-Env-Godot-Setups zu debuggen ist schmerzhaft. Bring `num_envs=1` zum Laufen, verifiziere, dass Belohnungen fließen, skaliere dann hoch. Die CleanRL-Trainingsschleife funktioniert mit einer Umgebung identisch — sie ist nur langsamer.

### Anmerkungen zu Beobachtungs- und Aktionsraum

`StableBaselinesGodotEnv` exponiert:
- `observation_space` als `gym.spaces.Box` (standardmäßig flaches Float-Array).
- `action_space` als entweder `Discrete` (für diskrete Aktionen) oder `Box` (für kontinuierliche).

Für kontinuierliche Aktionen tausch `Agent` gegen die Variante für kontinuierliche Aktionen: ersetze `Categorical(logits=self.actor(x))` durch eine `Normal(mean, std)`-Verteilung und aktualisiere die Log-Wahrscheinlichkeits- und Entropie-Berechnungen entsprechend. CleanRLs `ppo_continuous_action.py` hat die vollständige Implementierung.

---

## 6 · PPO modifizieren

Einer der Hauptgründe, CleanRL statt SB3 zu nutzen, ist, dass jede Modifikation auf wenige Zeilen in einer Datei lokalisiert ist. Hier sind drei konkrete Hacks.

### Hack 1 — Gradienten-Norm-Diagnose

**Was:** Logge die Gradienten-Norm nach jedem Update-Schritt zu TensorBoard. Hohe Gradienten-Normen zeigen, dass die Loss-Landschaft steil ist — ein Zeichen von Instabilität.

**Vorher (in der Update-Schleife):**
```python
nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
optimizer.step()
```

**Nachher:**
```python
grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
optimizer.step()
writer.add_scalar("diagnostics/grad_norm", grad_norm.item(), global_step)
```

`clip_grad_norm_` gibt die Gesamtnorm der Gradienten *vor* dem Clipping zurück. Logge das. Wenn `grad_norm` regelmäßig `max_grad_norm` um das 10-fache überschreitet, explodiert dein Loss — senke die Lernrate oder erhöhe `max_grad_norm`.

**Was in TensorBoard zu beobachten:** `diagnostics/grad_norm` sollte nach Warm-up stabil sein, typisch zwischen 0,1 und 2,0. Ein Spike auf 50+ gefolgt von einem Belohnungs-Kollaps ist ein klares Signal einer Gradientenexplosion.

### Hack 2 — Early Stopping bei approx_kl

**Was:** SB3 hat einen `target_kl`-Parameter, der die Update-Schleife vorzeitig stoppt, wenn die KL-Divergenz eine Schwelle überschreitet. CleanRL macht das standardmäßig nicht. Füg es hinzu.

**Vorher (Anfang der Epochen-Schleife):**
```python
for epoch in range(args.update_epochs):
    b_inds = np.random.permutation(batch_size)
    for start in range(0, batch_size, minibatch_size):
        ...
```

**Nachher:**
```python
target_kl = 0.01   # add this to parse_args() as --target-kl

for epoch in range(args.update_epochs):
    b_inds = np.random.permutation(batch_size)
    for start in range(0, batch_size, minibatch_size):
        end    = start + minibatch_size
        mb_inds = b_inds[start:end]

        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
            b_obs[mb_inds], b_actions[mb_inds]
        )
        logratio = newlogprob - b_logprobs[mb_inds]
        ratio    = logratio.exp()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()

        # NEW: break inner loop if KL is too large
        if approx_kl > target_kl:
            break
        ...

    # NEW: also break outer epoch loop
    else:
        continue
    break
```

**Was zu beobachten:** `losses/approx_kl` sollte um 0,005–0,015 schweben. Wenn Early Stopping konsistent in Epoche 2 von 4 feuert, ist deine Lernrate vielleicht zu hoch für die aktuelle Umgebung. Versuch, `learning_rate` um das 2-fache zu senken.

### Hack 3 — Laufende Belohnungsnormalisierung

**Was:** Normalisiere Belohnungen durch einen laufenden Mittelwert und Standardabweichung, bevor sie in die GAE-Berechnung eingehen. Das stabilisiert das Training, wenn Belohnungsgrößen über Umgebungen oder Trainingsphasen variieren.

**Vor Imports/Setup:**
```python
class RunningMeanStd:
    """Welford online algorithm for mean and variance."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x)
        batch_mean = x.mean(axis=0)
        batch_var  = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta     = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean  = self.mean + delta * batch_count / tot_count
        m_a       = self.var   * self.count
        m_b       = batch_var  * batch_count
        m2        = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.mean  = new_mean
        self.var   = m2 / tot_count
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

# Instantiate once before the training loop
reward_rms = RunningMeanStd(shape=())
```

**In der Rollout-Schleife, nach dem Speichern der Belohnungen:**
```python
rewards[step] = torch.tensor(reward)

# NEW: update running stats and normalize
reward_rms.update(reward)
rewards[step] = (rewards[step] - reward_rms.mean) / reward_rms.std
```

**Was zu beobachten:** `charts/episodic_return` sollte früh im Training stabiler werden. Der rohe Episoden-Return wird immer noch geloggt (via `RecordEpisodeStatistics`, das Belohnungen *vor* der Normalisierung sieht). Wenn das Training schon stabil war, hilft Normalisierung vielleicht nicht — überspring sie und vermeide die zusätzliche Komplexität.

!!! tip "Belohnungsnormalisierung vs. Vorteilsnormalisierung"
    CleanRL normalisiert bereits Vorteile (`b_advantages = (b_advantages - mean) / std`). Belohnungsnormalisierung ist eine zusätzliche, frühere Normalisierung, die die Skala der *Targets* für den Wert-Loss beeinflusst. Sie adressieren verschiedene Probleme: Vorteilsnormalisierung steuert die Gradientenskala; Belohnungsnormalisierung steuert die Wertfunktionsskala in Umgebungen mit sehr großen oder sehr kleinen Belohnungen.

---

## 7 · Sample Factory — maximaler Durchsatz

Für Training im großen Maßstab — Millionen Schritte in Umgebungen, die pro Schritt langsam sind — stößt du irgendwann an die Decke von SB3 und CleanRL. Sample Factory ist die Antwort.

### Der zentrale Unterschied: asynchrone Rollout-Sammlung

CleanRL und SB3 sammeln Rollouts **synchron**: N Schritte sammeln → Gradienten berechnen → N weitere Schritte sammeln → wiederholen. Die GPU sitzt während der Sammlung untätig; die CPU sitzt während des GPU-Updates untätig. Der Durchsatz wird durch das Langsamere limitiert.

Sample Factory nutzt **asynchrone Rollout-Worker**: mehrere Prozesse sammeln gleichzeitig Erfahrung und schieben sie in einen geteilten Replay-Buffer. Der Lerner-Prozess konsumiert kontinuierlich aus dem Buffer. Die GPU ist nie untätig und wartet auf Rollout-Worker, und Worker sind nie untätig und warten auf den Lerner. Diese Architektur ist 10–100× schneller als SB3 für CPU-lastige Umgebungen.

### Installation und Schnellstart

```bash
pip install sample-factory
```

```bash
# CartPole baseline — note the different CLI style
python -m sf_examples.gym.train_gym_env \
    --env=CartPole-v1 \
    --experiment=cartpole_sf \
    --train_for_env_steps=2_000_000 \
    --num_workers=8 \
    --num_envs_per_worker=2
```

### Godot + Sample Factory

`godot_rl_agents` enthält einen Sample-Factory-Wrapper. Siehe `godot_rl/wrappers/sample_factory_wrapper.py` im Repo. Der Einzeiler:

```bash
python -m godot_rl.train \
    --backend=sample_factory \
    --env_path=path/to/game.x86_64 \
    --experiment=my_experiment \
    --num_workers=4
```

!!! note "Wann zu Sample Factory greifen"
    Wenn deine Godot-Umgebung mehr als 50 ms pro Schritt braucht (physiklastig, viele Agenten, komplexe Beobachtungen), geben dir Sample Factorys Async-Worker einen bedeutsamen Speedup. Für einfache 2D-Spiele, die mit 60Hz und 4–8 parallelen Envs laufen, ist SB3 oder CleanRL ausreichend und einfacher zu debuggen.

---

## 8 · PPO von Grund auf implementieren (Stretch)

Dieser Abschnitt ist für Studierende, die den Algorithmus schreiben wollen, nicht nur lesen. Das Skelett unten läuft auf `CartPole-v1`, wenn die TODOs ausgefüllt sind. Jedes TODO mappt auf einen Abschnitt dieser Unit und von [unit-ppo-deep.md](unit-ppo-deep.md).

```python
"""
PPO from scratch — skeleton for CartPole-v1.
Fill in every TODO. The file should run as-is when complete.
"""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

# ---------- Hyperparameters ----------
ENV_ID        = "CartPole-v1"
TOTAL_STEPS   = 200_000
NUM_ENVS      = 4
NUM_STEPS     = 128        # steps per rollout per env
UPDATE_EPOCHS = 4
MINIBATCH_SIZE= 64
LR            = 2.5e-4
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_COEF     = 0.2
ENT_COEF      = 0.01
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
BATCH_SIZE    = NUM_ENVS * NUM_STEPS

# ---------- Network ----------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # TODO: define self.actor (obs_dim → act_dim logits)
        # TODO: define self.critic (obs_dim → 1 scalar)
        raise NotImplementedError

    def get_action_and_value(self, obs, action=None):
        # TODO: forward pass through actor → Categorical distribution
        # TODO: sample action if not provided
        # TODO: compute log_prob, entropy, value
        # TODO: return action, log_prob, entropy, value
        raise NotImplementedError

    def get_value(self, obs):
        # TODO: return critic(obs)
        raise NotImplementedError


# ---------- Rollout buffer ----------
def collect_rollout(envs, agent, device):
    """Collect NUM_STEPS steps from NUM_ENVS environments."""
    obs_shape = envs.single_observation_space.shape
    obs     = torch.zeros(NUM_STEPS, NUM_ENVS, *obs_shape)
    actions = torch.zeros(NUM_STEPS, NUM_ENVS, dtype=torch.long)
    logprobs= torch.zeros(NUM_STEPS, NUM_ENVS)
    rewards = torch.zeros(NUM_STEPS, NUM_ENVS)
    dones   = torch.zeros(NUM_STEPS, NUM_ENVS)
    values  = torch.zeros(NUM_STEPS, NUM_ENVS)

    next_obs  = torch.tensor(envs.reset()[0], dtype=torch.float32)
    next_done = torch.zeros(NUM_ENVS)

    for step in range(NUM_STEPS):
        # TODO: store next_obs and next_done
        # TODO: get action, logprob, _, value from agent (no_grad)
        # TODO: step envs, store reward and done
        # TODO: update next_obs and next_done
        raise NotImplementedError

    return obs, actions, logprobs, rewards, dones, values, next_obs, next_done


# ---------- GAE ----------
def compute_gae(rewards, values, dones, next_obs, next_done, agent):
    """
    Compute GAE advantages and returns.
    Returns: advantages (NUM_STEPS, NUM_ENVS), returns (NUM_STEPS, NUM_ENVS)
    """
    advantages = torch.zeros_like(rewards)
    last_gae   = 0.0

    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        # TODO: loop backwards from NUM_STEPS-1 to 0
        # TODO: compute nextnonterminal (mask episode boundaries)
        # TODO: compute delta = r_t + gamma * V(s_{t+1}) * nextnonterminal - V(s_t)
        # TODO: compute advantages[t] using GAE recursion
        raise NotImplementedError

    returns = advantages + values
    return advantages, returns


# ---------- Update step ----------
def ppo_update(agent, optimizer, obs, actions, logprobs, advantages, returns, values):
    """Run UPDATE_EPOCHS passes over the rollout."""
    b_obs       = obs.reshape(-1, *obs.shape[2:])
    b_actions   = actions.reshape(-1)
    b_logprobs  = logprobs.reshape(-1)
    b_advantages= advantages.reshape(-1)
    b_returns   = returns.reshape(-1)

    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

    for epoch in range(UPDATE_EPOCHS):
        inds = np.random.permutation(BATCH_SIZE)
        for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
            mb = inds[start:start + MINIBATCH_SIZE]

            # TODO: forward pass → newlogprob, entropy, newvalue
            # TODO: compute ratio = exp(newlogprob - b_logprobs[mb])
            # TODO: compute clipped policy loss (pg_loss)
            # TODO: compute value loss (MSE against b_returns[mb])
            # TODO: compute entropy bonus
            # TODO: combine: loss = pg_loss - ent_coef*entropy + vf_coef*v_loss
            # TODO: backward, clip grad norm, optimizer step
            raise NotImplementedError


# ---------- Main ----------
if __name__ == "__main__":
    envs  = gym.vector.SyncVectorEnv([
        lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(ENV_ID))
        for _ in range(NUM_ENVS)
    ])
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = envs.single_action_space.n

    agent     = ActorCritic(obs_dim, act_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    for update in range(TOTAL_STEPS // BATCH_SIZE):
        rollout_data = collect_rollout(envs, agent, device="cpu")
        obs, actions, logprobs, rewards, dones, values, next_obs, next_done = rollout_data

        advantages, returns = compute_gae(rewards, values, dones, next_obs, next_done, agent)
        ppo_update(agent, optimizer, obs, actions, logprobs, advantages, returns, values)

        if update % 10 == 0:
            print(f"update {update}/{TOTAL_STEPS // BATCH_SIZE}")

    envs.close()
```

Eine korrekte Implementierung konvergiert CartPole-v1 in etwa 150–200k Schritten zu 500. Wenn sie divergiert oder sich nie verbessert, sind die GAE-Rückwärtsschleife oder die Ratio-Berechnung der wahrscheinlichste Übeltäter.

---

## 9 · Was du jetzt weißt

| Implementierungs-Konzept | Rolle in PPO | SB3-Parameter |
|---|---|---|
| `ratio = exp(new_logprob - old_logprob)` | Importance-Gewicht, das Policy-Drift misst | — (intern berechnet) |
| `clip_coef` (ε) | Trust-Region-Radius — wie weit die Policy pro Update wandern darf | `clip_range` |
| `ent_coef` | Gewicht des Entropie-Bonus — steuert Erkundungsdruck | `ent_coef` |
| `vf_coef` | Gewicht des Wert-Loss relativ zum Policy-Loss | `vf_coef` |
| `max_grad_norm` | Gradient-Clipping-Schwelle — verhindert Einzelschritt-Explosionen | `max_grad_norm` |
| `num_steps` (T) | Rollout-Länge pro Umgebung pro Update | `n_steps` |
| `update_epochs` | Wie viele Durchläufe über die Rollout-Daten | `n_epochs` |
| `gae_lambda` (λ) | Bias-Varianz-Trade-off in der Vorteilsschätzung | `gae_lambda` |
| `gamma` (γ) | Diskontierungsfaktor — wie stark Zukunftsbelohnungen diskontiert werden | `gamma` |
| `minibatch_size` | Minibatch-Größe für jeden Gradientenschritt | `batch_size` |
| `approx_kl` | Proxy für KL(π_new ∥ π_old) — beobachte für Trainingsstabilität | `target_kl` (Stoppen) |
| `clipfrac` | Anteil der Transitionen, bei denen die Ratio geclippt wurde | — (nur geloggt) |

---

## 10 · Stretch Goals

Diese sind offen. Es gibt keine vorgegebenen Lösungen — nutze CleanRLs Quellcode und die Paper als Leitfaden.

**PPO mit LSTM.** Ersetze den MLP-Backbone in `Agent` durch ein LSTM. Die Rollout-Schleife muss versteckte Zustände zwischen Schritten mitführen, und die Minibatch-Konstruktion muss die Sequenzreihenfolge respektieren (kein zufälliges Shuffeln). CleanRL hat `ppo_atari_lstm.py` als Referenz.

**Curiosity-Bonus.** Füge der CleanRL-Rollout-Schleife eine intrinsische Belohnung hinzu. Nach `rewards[step] = torch.tensor(reward)` berechne einen intrinsischen Bonus mit einem Random-Network-Distillation-Modul (RND) und addiere ihn zur gespeicherten Belohnung. Siehe [unit-curiosity.md](unit-curiosity.md) für die Theorie. Beobachte, wie sich `charts/episodic_return` in extrinsische und intrinsische Komponenten aufteilt.

**Godot Headless-Training.** Lass eine Godot-Umgebung ohne Display mit `--headless` (Godot 4) oder `--no-window` (Godot 3) Flag laufen, übergeben via `env_path`-Argumente in `StableBaselinesGodotEnv`. Profile, ob `SyncVectorEnv` mit 8 Headless-Godot-Instanzen schneller ist als 4 mit Fenster — die Antwort hängt von der CPU-Kern-Zahl deiner Maschine ab.

---

!!! info "Selbstcheck, bevor du weitermachst"
    Kannst du diese Fragen in eigenen Worten beantworten?

    1. Gehe durch ein volles CleanRL-PPO-Update: wo passiert das Rollout, wo wird GAE berechnet, wo feuert das Policy-Update?
    2. Warum reshufflet die innere Schleife in CleanRLs PPO die Minibatches jede Epoche — und was bräche, wenn sie es nicht täte?
    3. Was misst `clipfrac`, und wann ist `clipfrac = 0` tatsächlich ein *schlechtes* Zeichen?
    4. Wenn du einen intrinsischen Curiosity-Bonus hinzufügen wolltest, welche Zeile in `ppo.py` würdest du ändern?
    5. Was unterstützt CleanRL absichtlich *nicht*, was SB3 unterstützt — und warum ist das ein Feature, kein Bug?

    Wenn du alle fünf beantworten kannst — du kannst jede PPO-Implementierung in freier Wildbahn lesen.

[← Anwenden — SAC vs PPO auf JumperHard](unit-sac-applied.md) · [Kursstartseite](index.md) · [→ Paralleles Training](unit-05.md)
