# Setup

One-time installation for every unit in this course. Do this before Unit 0.

---

## Godot 4 — .NET edition

Download the **.NET / Mono** build of Godot 4 from [godotengine.org](https://godotengine.org) (not the standard build). Tested with Godot 4.3+.

!!! warning "Use the .NET edition — not the standard build"
    The godot-rl plugin compiles native C# tasks (NuGet references) that bridge Godot to the ONNX runtime. The standard build cannot load these. You must also install the [.NET SDK](https://dotnet.microsoft.com/download).

---

## Python — Conda environment

**Why Conda?** It lets you pin Python 3.10 in an isolated folder so the ML packages don't clash with other projects. godot-rl and Stable-Baselines3 are most reliable on Python 3.10; newer versions often break package wheels.

**Install Miniconda** (skip if you already have `conda`):

Download from [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html) for your OS. After installation open a new terminal and verify:

```bash
conda --version
```

!!! tip "macOS / Linux first run"
    The installer may ask you to run `conda init` — follow the prompt, then open a new terminal.

**Create the environment** (one-time):

```bash
conda create --name godot_env python=3.10 -y
```

**Every new terminal — activate before any training command:**

```bash
conda activate godot_env
pip install "godot-rl[sb3]" tensorboard
```

Verify: `python -c "import godot_rl; print('ok')"`

!!! info "What gets installed"
    - `godot-rl[sb3]` — Python ↔ Godot socket bridge, Stable-Baselines3 wrappers, and the `gdrl` CLI
    - `tensorboard` — training-curve visualisation
    
    Keep the double quotes around `"godot-rl[sb3]"` so the shell does not expand the brackets.

---

## Godot plugin — godot-rl-agents

The Godot-side plugin is separate from the Python package.

**Option A — AssetLib (easiest)**

- Open the Godot editor → **AssetLib** tab (top center)
- Search **rl**, pick **Godot RL Agents**
- Click **Download**, unselect `LICENSE` and `README.md`, then **Install**

**Option B — Manual (always up to date)**

- Clone [github.com/edbeeching/godot_rl_agents_plugin](https://github.com/edbeeching/godot_rl_agents_plugin)
- Copy `addons/godot_rl_agents` into your project's `addons/` folder

!!! warning "Two different repos"
    `godot_rl_agents` is the *Python* package (`pip install`). The Godot plugin lives in the separate `godot_rl_agents_plugin` repo.

**Enable the plugin**

Project → Project Settings → Plugins → **Godot RL Agents** → Enabled. Wait for MSBuild to finish.

!!! warning "First-import C# error"
    If Godot reports a build error on first open, close and reopen the project — the C# assemblies build correctly on the second open.

!!! tip "Verify"
    Add Node → search `Sync` and `AIController2D`. If they appear, the plugin is working.
