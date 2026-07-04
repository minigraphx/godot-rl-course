# Setup

One-time installation for every unit in this course. Do this before Unit 0.

---

## Course repo — clone first { #course-repo }

Everything in this course — `requirements-course.txt`, the `examples/` code used in Neural Foundations, helper scripts — lives in the course repository. Clone it and work from its root:

```bash
git clone https://github.com/minigraphx/godot-rl-course.git
cd godot-rl-course
```

No git? Use **Code → Download ZIP** on [github.com/minigraphx/godot-rl-course](https://github.com/minigraphx/godot-rl-course), unpack it, and `cd` into the unpacked folder instead.

!!! info "Run all course commands from the repo root"
    Unless a unit says otherwise, every `pip`, `python -m examples.…`, and `godot --headless` command in this course assumes your terminal is in the course repo root — the folder containing `requirements-course.txt`.

---

## Godot 4 — Standard build

Download the **Standard** build of Godot 4 from [godotengine.org](https://godotengine.org). The course's native runner requires **Godot 4.5 or newer**.

!!! info "No .NET SDK, no C# — the Standard build is all you need"
    Earlier revisions of this course used a C# plugin that required the .NET/Mono edition of Godot plus the .NET SDK. The course now uses **godot-native-rl** — a pure-GDScript addon bundled with the course repo — so the Standard build suffices. If you already installed the .NET edition, it works fine too; there is just no longer any reason to install it.

---

## Godot on the command line { #godot-cli }

Several units run Godot from the terminal (`godot --headless …`). Downloading the editor does **not** put a `godot` command on your PATH — set that up once:

**macOS** — the binary lives inside the app bundle. Add an alias to your shell profile (`~/.zshrc`):

```bash
alias godot="/Applications/Godot.app/Contents/MacOS/Godot"
```

**Windows** — add the folder containing `Godot_*.exe` to your PATH (Settings → System → About → Advanced system settings → Environment Variables), or call the executable by its full path. Forward slashes work in every shell:

```bash
C:/Tools/Godot/Godot_v4.5-stable_win64.exe --version
```

**Linux** — make the downloaded binary executable and link it onto your PATH:

```bash
chmod +x Godot_v4.5-stable_linux.x86_64
sudo ln -s "$PWD/Godot_v4.5-stable_linux.x86_64" /usr/local/bin/godot
```

!!! tip "Verify"
    Open a new terminal and run `godot --version` — it should print a 4.x version string.

---

## Python — Conda environment

**Why Conda?** It lets you pin Python 3.10 in an isolated folder so the ML packages don't clash with other projects. godot-rl and Stable-Baselines3 are most reliable on Python 3.10; newer versions often break package wheels.

**Install Miniconda** (skip if you already have `conda`):

Download from [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html) for your OS. After installation open a new terminal and verify:

```bash
conda --version
```

**Create the environment** (one-time):

```bash
conda create --name godot_env python=3.10 -y
```

**Every new terminal — activate before any training command:**

```bash
conda activate godot_env
pip install -r requirements-course.txt
```

`requirements-course.txt` is at the root of the course repo you [cloned above](#course-repo) — run the command from there. It pins every package to a known-good version — see the [compatibility table](#compatibility-table) below.

!!! info "What gets installed"
    - `godot-rl` — Python ↔ Godot socket bridge, Stable-Baselines3 wrappers, and the `gdrl` CLI
    - `stable-baselines3` — PPO, SAC, and other algorithms
    - `torch` — PyTorch backend for training
    - `tensorboard` — training-curve visualisation
    - `matplotlib`, `onnx`, `onnxruntime`, `ncnn`, `opencv-python` — Neural Foundations plots, model export, and parity checks

Verify: `python -c "import godot_rl; print('ok')"`

!!! note "Native inference binaries — macOS Apple Silicon only (for now)"
    *Training* with the bundled **godot-native-rl** addon works on every platform in the compatibility table — the training bridge is pure GDScript. *Native ncnn inference* (running a trained brain inside Godot without Python, used in [Neural Foundations 3](unit-neural-03.md) and the Ship phase) currently ships binaries for **macOS Apple Silicon** only; Windows and Linux runners are on the roadmap.

!!! tip "macOS / Linux first run"
    The installer may ask you to run `conda init` — follow the prompt, then open a new terminal.

!!! tip "Windows first run"
    See [Windows first run](#windows-first-run) below for PowerShell / cmd / Git Bash activation notes and Windows-specific gotchas.

---

## Compatibility table

The table below shows the package versions that ship in `requirements-course.txt` and the Godot version they were tested with.

| Course tag | Godot | godot-native-rl | godot-rl (Python bridge) | stable-baselines3 | PyTorch | Python |
|---|---|---|---|---|---|---|
| 2026-05 | 4.5.x (Standard) | commit `4013370` (bundled) | 0.5.0 | 2.3.2 | 2.6.0 | 3.10 |

!!! warning "Do not upgrade packages mid-course"
    godot-rl, SB3, and PyTorch have broken APIs across releases. Stick to the pinned versions in `requirements-course.txt` for the duration of the course. After the course, feel free to experiment with newer releases — just create a fresh conda environment.

---

## Windows first run

The steps above work on Windows with minor differences. Read this section before your first `conda activate`.

### Shell choice

| Shell | Notes |
|-------|-------|
| **Anaconda Prompt** | Easiest — `conda activate godot_env` works out of the box. |
| **PowerShell** | Run `conda init powershell` once (as administrator), restart PowerShell, then `conda activate godot_env`. |
| **cmd** | Run `conda init cmd.exe` once, restart cmd, then `conda activate godot_env`. |
| **Git Bash** | Run `conda init bash` once (from Anaconda Prompt), restart Git Bash, then `conda activate godot_env`. |

### `--env_path` with Windows paths

godot-rl accepts forward slashes on Windows — prefer them over backslashes to avoid shell escaping issues:

```bash
# Recommended — forward slashes work everywhere, including PowerShell and cmd
gdrl --env_path=C:/Users/YourName/Projects/my_game/my_game.exe

# Also valid — backslashes, but must escape or quote
gdrl --env_path="C:\Users\YourName\Projects\my_game\my_game.exe"
```

### Windows Defender / antivirus socket issue

Windows Defender (and many third-party antivirus programs) sometimes **silently block** the local TCP port that godot-rl uses to communicate between Python and Godot (default: port 11008). Symptoms: training appears to start but Godot never connects; Python hangs waiting for the first observation.

Fix:

1. Open **Windows Security → Firewall & network protection → Allow an app through firewall**.
2. Add an exception for `python.exe` (your conda env's Python) and for the Godot executable.
3. Alternatively, try a different port: `gdrl --port=12000` (and set the same port on the Godot-side sync node).

If you use a third-party antivirus, add the conda environment folder (e.g. `C:\Users\YourName\miniconda3\envs\godot_env\`) and your Godot project folder to the exclusion list.

### `chmod +x` is not needed on Windows

The macOS/Linux commands `chmod +x godot_binary` do not apply on Windows. Godot `.exe` files are already executable by the OS.

### WSL2 vs native Windows

| Approach | Pros | Cons |
|----------|------|------|
| **Native Windows** | Simplest setup, no translation layer, Direct3D GPU | Antivirus/firewall friction; paths use backslashes |
| **WSL2 (Ubuntu)** | Full Linux toolchain, easier GPU setup via CUDA | GPU passthrough (CUDA in WSL2) requires Windows 11 + WSL2 kernel ≥ 5.15; Godot GUI cannot render inside WSL2 without an X server or WSLg |

**Recommendation for this course:** use **native Windows** unless you already have a working WSL2 + GPU setup. Godot must run on the Windows host side (or WSLg) regardless; mixing Godot on Windows with Python in WSL2 requires extra port-forwarding steps that are not covered in this course.

---

## Godot addon — godot-native-rl

The Godot-side bridge is **godot-native-rl**, a pure-GDScript addon. It ships **inside the course repo** at `examples/neural_foundations/game/addons/godot_native_rl/` — nothing to download, nothing to build.

- **Training:** the `NcnnSync` node speaks the same local-socket protocol as the `gdrl` Python side. Pure GDScript — works on every platform.
- **Inference:** a bundled ncnn GDExtension runs trained brains natively in Godot. Binaries currently ship for macOS Apple Silicon only (see the note above); training does not need them.

The addon's node classes (`NcnnSync`, `NcnnAIController2D`, `NcnnAIController3D`, sensor nodes) auto-register when the project opens. Enabling the plugin under Project → Project Settings → Plugins is **optional** — doing so only adds a clear error message if the native inference binary is missing for your platform.

!!! warning "Two different things"
    `godot-rl` is the *Python* package (installed by `requirements-course.txt`) that runs the training server. **godot-native-rl** is the *Godot* addon bundled with the course repo. They talk to each other over a local socket.

!!! tip "Verify"
    Open `examples/neural_foundations/game/project.godot` in Godot, then Add Node → search `NcnnSync` and `NcnnAIController3D`. If they appear, the addon is working.

**Using the addon in your own project:** copy `addons/godot_native_rl/` into your project's `addons/` folder. For native inference, also copy `ncnn_runner.gdextension` and the `bin/` folder.

---

## Legacy plugin — godot-rl-agents (units awaiting migration) { #godot-plugin-godot-rl-agents }

Units from [RL Essentials](unit-01.md) onward still use example environments built on the older C# plugin. **Skip this section until you reach those units** — Unit 0 and Neural Foundations need none of it.

The legacy stack additionally requires the .NET/Mono edition of Godot and the [.NET SDK](https://dotnet.microsoft.com/download). Then:

- Clone or download [github.com/edbeeching/godot_rl_agents_plugin](https://github.com/edbeeching/godot_rl_agents_plugin)
- Copy the `addons/godot_rl_agents` folder into the example project's `addons/` folder (the official example projects ship with it already)
- Project → Project Settings → Plugins → **Godot RL Agents** → Enabled; wait for MSBuild to finish
- If Godot reports a C# build error on first open, close and reopen the project — the assemblies build correctly on the second open
- Verify: Add Node → search `Sync` and `AIController2D`

These units are being migrated to the native stack one by one; this section disappears when the migration completes.
