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

## Native runtime — godot-native-rl { #native-runtime }

!!! info "Primary path, rolling out unit by unit"
    The course is migrating onto **godot-native-rl** — standard Godot, native **ncnn** inference, **no .NET**. Units move over one at a time; until a unit says otherwise, use the **[.NET fallback](#godot-net-fallback)** below. macOS **Intel** has no native runner yet, so use the fallback there too.

**Clone the plugin repo** — it holds the addon, the example environments, and the training scripts:

```bash
git clone https://github.com/minigraphx/godot-native-rl.git
```

**Standard Godot 4.5+** — download the **standard** build (not the .NET/Mono build) from [godotengine.org](https://godotengine.org).

**Get the `NcnnRunner` binary for your platform.** The Godot AssetLib package is not published yet, so download `godot-native-rl-addon-v0.3.1.zip` from the repo's [Releases](https://github.com/minigraphx/godot-native-rl/releases) and copy its `addons/godot_native_rl/bin/` into your clone — the binaries are not committed to the repo tree. Prebuilt binaries are verified for **Linux x86_64, macOS arm64, Windows x86_64, and Web/WASM**; for any other target, build from source per the repo's `docs/dev/building.md`.

!!! warning "macOS — clear the download quarantine first"
    Prebuilt `.dylib`s are not Apple-notarized, so Gatekeeper blocks them and inference silently fails. After unzipping, run once from your clone:
    ```bash
    xattr -dr com.apple.quarantine addons/godot_native_rl/bin
    ```

**Enable the plugin** — open the project in Godot 4.5+, then Project → Project Settings → Plugins → **Godot Native RL** → Enabled. (Headless training doesn't need this; it only affects editor tooling.)

**Training stack (conda).** Native training uses a newer Python stack than the [fallback](#compatibility-table) — Python 3.13 (training) + 3.14 (conversion), Stable-Baselines3 2.8, gymnasium 1.2, godot-rl 0.8.2. The authoritative setup and `train_*.sh` scripts live in your clone's `docs/guide/training.md` (run `setup_training.sh`, or the documented conda alternative: create the env and `pip install -r requirements-train.txt` / `requirements-convert.txt`). Each unit pins the exact train and deploy commands as it migrates.

---

## Godot 4 — .NET edition (fallback) { #godot-net-fallback }

!!! note "Fallback path"
    Use this if your unit hasn't moved to the [native runtime](#native-runtime) yet, or on macOS Intel (the native runner ships for arm64 only).

Download the **.NET / Mono** build of Godot 4 from [godotengine.org](https://godotengine.org) (not the standard build). Tested with Godot 4.3+.

!!! warning "Use the .NET edition — not the standard build"
    The godot-rl plugin compiles native C# tasks (NuGet references) that bridge Godot to the ONNX runtime. The standard build cannot load these. You must also install the [.NET SDK](https://dotnet.microsoft.com/download).

---

## Godot on the command line { #godot-cli }

Several units run Godot from the terminal (`godot --headless …`). Downloading the editor does **not** put a `godot` command on your PATH — set that up once:

**macOS** — the binary lives inside the app bundle. Add an alias to your shell profile (`~/.zshrc`):

```bash
alias godot="/Applications/Godot_mono.app/Contents/MacOS/Godot"
```

**Windows** — add the folder containing `Godot_*.exe` to your PATH (Settings → System → About → Advanced system settings → Environment Variables), or call the executable by its full path. Forward slashes work in every shell:

```bash
C:/Tools/Godot/Godot_v4.3-stable_mono_win64.exe --version
```

**Linux** — make the downloaded binary executable and link it onto your PATH:

```bash
chmod +x Godot_v4.3-stable_mono_linux.x86_64
sudo ln -s "$PWD/Godot_v4.3-stable_mono_linux.x86_64" /usr/local/bin/godot
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

!!! note "Neural Foundations 3 — Game path (macOS arm64 only)"
    The PPO racer in [Neural Foundations 3](unit-neural-03.md) uses a bundled **godot-native-rl** ncnn runner shipped for **macOS Apple Silicon** only. The Research path (Python REINFORCE point robot) works on every platform in the compatibility table below.

!!! tip "macOS / Linux first run"
    The installer may ask you to run `conda init` — follow the prompt, then open a new terminal.

!!! tip "Windows first run"
    See [Windows first run](#windows-first-run) below for PowerShell / cmd / Git Bash activation notes and Windows-specific gotchas.

---

## Compatibility table

The table below shows the package versions that ship in `requirements-course.txt` and the Godot version they were tested with.

| Course tag | Godot | godot-rl | stable-baselines3 | PyTorch | Python |
|---|---|---|---|---|---|
| 2026-05 | 4.3.x | 0.5.0 | 2.3.2 | 2.6.0 | 3.10 |

The [native runtime](#native-runtime) uses a separate, newer stack — Godot 4.5+ (standard), godot-rl 0.8.2, Stable-Baselines3 2.8.0, PyTorch 2.12.0, gymnasium 1.2.2, Python 3.13/3.14 — installed from the cloned godot-native-rl repo, not from `requirements-course.txt`.

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
3. Alternatively, try a different port: `gdrl --port=12000` (and set the same port in Godot's AIController).

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

## Godot plugin — godot-rl-agents (fallback)

!!! note "Fallback path"
    For units still on the .NET stack. The [native runtime](#native-runtime) above installs its plugin by cloning the godot-native-rl repo instead.

The Godot-side plugin is separate from the Python package.

!!! info "Not in the Asset Library"
    The plugin is not available in Godot's AssetLib — you must install it manually from GitHub.

- Clone or download [github.com/edbeeching/godot_rl_agents_plugin](https://github.com/edbeeching/godot_rl_agents_plugin)
- Copy the `addons/godot_rl_agents` folder into your project's `addons/` folder

!!! warning "Two different repos"
    `godot_rl_agents` is the *Python* package (`pip install`). The Godot plugin lives in the separate `godot_rl_agents_plugin` repo.

**Enable the plugin**

Project → Project Settings → Plugins → **Godot RL Agents** → Enabled. Wait for MSBuild to finish.

!!! warning "First-import C# error"
    If Godot reports a build error on first open, close and reopen the project — the C# assemblies build correctly on the second open.

!!! tip "Verify"
    Add Node → search `Sync` and `AIController2D`. If they appear, the plugin is working.
