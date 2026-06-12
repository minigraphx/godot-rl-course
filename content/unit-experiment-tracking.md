# Experiment Tracking — W&B, MLflow, and Hyperparameter Sweeps

[Course home](index.md) · [← Debugging](unit-debugging.md) · [→ Advanced Evaluation](unit-evaluation.md)

!!! info "Time"
    Reading: ~25 min

---

## 1 · Why TensorBoard isn't enough at scale

TensorBoard is the default logger for stable-baselines3 and it works well — until you start comparing more than two or three runs. Once you are running ten or more hyperparameter sweeps in parallel, its limitations become blockers:

- **No config logging.** TensorBoard records curves, not the hyperparameters that produced them. After a week of experiments you will have 20 reward curves and no reliable way to know which learning rate, `n_steps`, or entropy coefficient belongs to which curve.
- **No artifact tracking.** There is no built-in link between a checkpoint file on disk and the training run that produced it. The checkpoint that achieves your best result can easily become orphaned from the run that generated it.
- **Comparison is manual and fragile.** Overlaying 10+ runs in TensorBoard requires careful directory naming and still produces a cluttered, hard-to-share HTML page.
- **Collaboration is painful.** Sharing results with a teammate or reviewer means either zipping up a `logs/` directory or granting server access. Neither scales.

W&B (Weights & Biases) and MLflow both solve all of these. They log hyperparameter configs alongside metrics, store and version model artifacts, provide a hosted (or self-hosted) comparison UI, and make sharing a permanent URL instead of a zip file.

!!! tip "When to switch"
    If you are running a single training job to check that your reward function works, TensorBoard is fine. Switch to W&B or MLflow as soon as you start tuning hyperparameters or comparing algorithm variants.

---

## 2 · Weights & Biases in 10 minutes

### Installation and authentication

```bash
pip install wandb
wandb login   # paste your API key from wandb.ai/authorize
```

### Custom callback

The code below shows a minimal W&B integration written as a standard SB3 `BaseCallback`. Understanding it line-by-line is useful before switching to the built-in integration.

```python
# pip install wandb
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        if self.n_calls % 1000 == 0:
            wandb.log({
                "rollout/ep_rew_mean": self.locals.get("infos", [{}])[0].get("episode", {}).get("r", 0),
                "train/loss": self.model.logger.name_to_value.get("train/loss", 0),
            }, step=self.num_timesteps)
        return True

wandb.init(
    project="godot-rl-course",
    config={
        "algorithm": "PPO",
        "env": "FlyBy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "total_timesteps": 1_000_000,
    }
)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/")
model.learn(total_timesteps=1_000_000, callback=WandbCallback())
wandb.finish()
```

The `wandb.init(config=...)` call is the key addition over TensorBoard. Every hyperparameter passed here is recorded alongside every metric, so you can filter and group runs in the W&B UI by any config key.

### Built-in SB3 integration (recommended)

W&B ships a ready-made SB3 callback that logs everything automatically — reward, losses, learning rate schedules, and more. Use this unless you need custom metric names.

```python
from wandb.integration.sb3 import WandbCallback

run = wandb.init(
    project="godot-rl-course",
    config={
        "algorithm": "PPO",
        "env": "FlyBy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "total_timesteps": 1_000_000,
    },
    sync_tensorboard=True,  # mirrors all SB3 TensorBoard logs into W&B
)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"logs/{run.id}")
model.learn(total_timesteps=1_000_000, callback=WandbCallback(verbose=2))
wandb.finish()
```

`sync_tensorboard=True` means you do not need to change your existing logging calls — W&B captures the TensorBoard events and re-indexes them under the run.

!!! tip "Free tier is generous"
    W&B offers unlimited runs and 100 GB of artifact storage on its free personal/academic plan. For a course project this is more than enough.

---

## 3 · W&B Sweeps — hyperparameter search

A sweep defines the search space and strategy in a YAML file, then launches agents (worker processes) that each pull a hyperparameter config, run training, and report results back to the sweep controller.

### Sweep config (`sweep.yaml`)

```yaml
program: train.py
method: bayes
metric:
  name: rollout/ep_rew_mean
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  n_steps:
    values: [1024, 2048, 4096]
  ent_coef:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.1
```

`method: bayes` uses Bayesian optimisation — it models the relationship between hyperparameters and the target metric and proposes configs that are likely to improve on what it has already seen. Use `method: random` if you want a simpler baseline sweep or if your runs are very short.

### Launching

```bash
# Step 1 — register the sweep and get an ID
wandb sweep sweep.yaml

# Step 2 — start one or more agents (each runs train.py with a sampled config)
wandb agent <sweep-id>

# To run multiple agents in parallel on separate machines or tmux panes:
wandb agent <sweep-id> &
wandb agent <sweep-id> &
```

Your `train.py` should read hyperparameters from `wandb.config` so agents pick up the sampled values:

```python
import wandb

wandb.init()  # sweep agent populates wandb.config automatically
cfg = wandb.config

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=cfg.learning_rate,
    n_steps=cfg.n_steps,
    ent_coef=cfg.ent_coef,
)
model.learn(total_timesteps=500_000, callback=WandbCallback(verbose=2))
```

!!! warning "Sweep agents block until the run finishes"
    Each agent runs one config at a time. For Godot environments with slow physics, set `speedup` high (32–64×) and reduce `total_timesteps` in sweep runs — you want enough signal to rank configs, not a full production training run.

---

## 4 · What to log beyond `ep_rew_mean`

Episode reward is necessary but not sufficient for diagnosing training. The table below lists metrics that reveal problems reward alone cannot expose.

| Metric | What it tells you | How to log |
|--------|------------------|------------|
| KL divergence | Policy stability — large KL means updates are too aggressive | SB3 logs `train/approx_kl` automatically |
| Gradient norm | Exploding gradients | Log `train/explained_variance` as a proxy; low and decreasing means the critic is not learning |
| Episode length distribution | Are episodes terminating correctly? Short episodes may mean the agent dies or resets unexpectedly | Log as a histogram: `wandb.log({"ep_len": wandb.Histogram(ep_lens)})` |
| Observation statistics | Are observations in the expected range? Out-of-range obs cause silent normalisation failures | Log mean and std of the obs buffer each rollout |
| Action distribution entropy | Is the policy converging too fast? Entropy collapse early in training means the agent stops exploring | SB3 logs `train/entropy_loss` automatically |

!!! tip "Histograms in W&B"
    `wandb.Histogram` accepts a list or numpy array and renders as an interactive histogram in the W&B UI. Use it for episode lengths, action magnitudes, and observation channels — anything where the distribution shape matters, not just the mean.

---

## 5 · MLflow — self-hosted alternative

Use MLflow instead of W&B when:

- Your training machines have no internet access (air-gapped labs, cloud VPCs with egress restrictions).
- Your team has data privacy requirements that prevent sending run data to a third-party cloud.
- You already have MLflow deployed as part of a wider ML platform.

### Quick setup

```bash
pip install mlflow
mlflow server --host 0.0.0.0 --port 5000   # start the tracking server
export MLFLOW_TRACKING_URI=http://localhost:5000
```

Then open `http://localhost:5000` in your browser for the UI.

### SB3 integration

```python
import mlflow
from stable_baselines3.common.callbacks import BaseCallback

class MLflowCallback(BaseCallback):
    def _on_step(self):
        if self.n_calls % 1000 == 0:
            mlflow.log_metric(
                "ep_rew_mean",
                self.locals.get("infos", [{}])[0].get("episode", {}).get("r", 0),
                step=self.num_timesteps,
            )
        return True

with mlflow.start_run():
    mlflow.log_params({"algorithm": "PPO", "lr": 3e-4, "n_steps": 2048})
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000, callback=MLflowCallback())
    mlflow.log_artifact("flyby_ppo.zip")   # save the checkpoint into the run
```

!!! warning "mlflow.log_metric is not batched"
    Calling `mlflow.log_metric` every step will make your training slow. Log every 1 000–5 000 timesteps as shown above, or batch metrics with `mlflow.log_metrics(dict, step=n)`.

---

## 6 · W&B vs MLflow comparison

| Feature | W&B | MLflow |
|---------|-----|--------|
| Setup | 1 command (`wandb login`) | Self-hosted server |
| UI | Excellent — rich interactive charts, parallel coordinates for sweeps | Good — functional, less polished |
| Cost | Free personal/academic tier; paid for teams | Free (self-host); managed tiers available |
| Privacy | Cloud (US/EU data residency options) | On-prem — data never leaves your network |
| Sweeps | Built-in Bayesian / random / grid | Optuna integration via `mlflow.tracking` |
| Artifact storage | W&B artifact registry | MLflow artifact store (S3, GCS, local FS) |
| Best for | Individual researchers, academic projects | Teams with existing infra or privacy requirements |

For this course, W&B is the recommended default. Switch to MLflow if you hit a privacy or connectivity constraint.

---

## 7 · Artifact tracking

Logging metrics is only half the story. Without linking checkpoints to runs, you cannot reproduce a result six months later — you have curves but no policy.

### Saving checkpoints to W&B

```python
# After training finishes
model.save("flyby_ppo")               # writes flyby_ppo.zip
wandb.save("flyby_ppo.zip")           # uploads zip to W&B artifacts, linked to this run
```

### Saving at intervals with a checkpoint callback

```python
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_cb = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints/",
    name_prefix="flyby_ppo",
)

model.learn(
    total_timesteps=1_000_000,
    callback=[WandbCallback(verbose=2), checkpoint_cb],
)

# Upload all checkpoints as a versioned artifact
artifact = wandb.Artifact("flyby-checkpoints", type="model")
artifact.add_dir("./checkpoints/")
wandb.log_artifact(artifact)
```

### Reloading an artifact later

```python
run = wandb.init(project="godot-rl-course")
artifact = run.use_artifact("flyby-checkpoints:v3", type="model")
artifact_dir = artifact.download()
model = PPO.load(f"{artifact_dir}/flyby_ppo_1000000_steps.zip", env=env)
```

The `:v3` suffix pins an exact version. Artifacts are immutable — uploading a new checkpoint creates a new version rather than overwriting the old one.

!!! tip "Why this matters"
    It is common to beat a benchmark, move on, and then need to re-evaluate that policy months later for a paper or demo. Without artifact tracking you are relying on local disk, which is fragile. W&B artifact URLs are permanent.

---

## 8 · Godot-specific notes

W&B and MLflow integrate at the SB3 level, not the environment level. They work identically whether your environment is a Godot binary, a Gymnasium wrapper, or anything else — you do not need to modify your Godot scene or GDScript at all.

That said, there are a few Godot-specific parameters worth logging as run config:

```python
wandb.init(
    project="godot-rl-course",
    config={
        "algorithm": "PPO",
        "env": "FlyBy",
        "env_path": "builds/FlyBy.x86_64",
        "n_parallel": 4,          # number of parallel Godot subprocesses
        "speedup": 32,            # physics speedup factor
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "total_timesteps": 1_000_000,
    }
)
```

Logging `n_parallel` and `speedup` lets you compare wall-clock efficiency across machines: a run with `n_parallel=8, speedup=64` that achieves the same reward in half the wall time is a meaningful result.

**Additional metrics to log explicitly for common algorithms:**

- **DQN:** `rollout/exploration_rate` — tracks how quickly epsilon decays; stalling here means exploration is not annealing as expected.
- **SAC:** `train/ent_coef` — SAC learns its entropy coefficient automatically; watching it collapse early can predict reward stagnation before it appears in the reward curve.
- **PPO (Godot):** log the fraction of truncated vs terminated episodes if your Godot scene uses `is_done` vs `is_truncated` — a mismatch here is a common source of invisible bugs.

!!! warning "Parallel Godot subprocesses and logging"
    When `n_parallel > 1`, godot-rl-agents returns batched `infos`. Index correctly when extracting episode reward: `infos[0].get("episode", {}).get("r", 0)` only captures the first subprocess. Use `np.mean([i.get("episode", {}).get("r", 0) for i in infos if "episode" in i])` for a more representative mean across all parallel envs.

!!! check "Done when"
    The training run from Section 2 appears in your `godot-rl-course` project on the W&B dashboard, with the config keys (`learning_rate`, `n_steps`, `env`) visible on the run page and the familiar TensorBoard curves (`rollout/ep_rew_mean` and friends) mirrored into the charts via `sync_tensorboard=True`. If you also launched the sweep from Section 3, `wandb agent` has completed at least one trial, and it shows up as its own run under the sweep with its sampled hyperparameters logged. No internet access, or a privacy constraint? The MLflow route from Section 5 counts the same — the run, its params, and the checkpoint artifact visible at `http://localhost:5000`.

---

## 9 · Stretch Goals

**Run a 3-axis sweep.** Use the W&B sweep config in Section 3 as a starting point, then add a third axis — e.g. `n_steps ∈ {1024, 2048, 4096}`. That gives you a 3-D parallel-coordinates view in the W&B UI. Train for at least 5 runs per cell and write down which axis dominates the others. The point is to feel how quickly cost grows once a sweep is more than 1-D.

**Reproduce a run from artifacts only.** From an old W&B run (your own, from any unit), download just the config + model artifact and recreate the trained model on a fresh machine without copying any local code. Replay 10 episodes. Did you get the same reward? If not, what was missing from the artifact — was it the env binary, a seed, a code commit hash? Patch the gap in your logging template so the next run is genuinely reproducible.

**Set up MLflow alongside W&B for one run.** Wire both `MlflowOutputFormat` and `WandbCallback` into the same SB3 training script (see Sections 2 and 5). Compare the two UIs side by side on the same run. Decide — for your own use — which one you would keep if you had to pick one, and write down *why*. The answer differs by team and threat model; the exercise is to form your own.

!!! warning "Pseudocode"
    ```python
    import mlflow
    from wandb.integration.sb3 import WandbCallback
    import wandb

    wandb.init(project="godot-rl-course", sync_tensorboard=True)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("godot-rl-course")

    with mlflow.start_run():
        mlflow.log_params({"algorithm": "PPO", "env": "FlyBy"})
        model.learn(total_timesteps=200_000, callback=WandbCallback())
        mlflow.log_artifact("ppo_flyby.zip")
    wandb.finish()
    ```
