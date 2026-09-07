# Native-RL Migration — Design & Scope (#81 / #71)

**Date:** 2026-06-15
**Status:** Design / scoping. Phase 1 (Setup) in progress.
**Strategy (#81, revised 2026-06-15):** add **godot-native-rl** as the primary path while **keeping godot-rl-agents (.NET) as a documented fallback** — a dual stack, not a hard replacement. The fallback covers macOS Intel (no native arm64-only runner there) and anyone on the validated 3.10 stack, so existing units keep working as each is migrated.

## Readiness (confirmed against plugin v0.3.1, 2026-06-14)

- The platform prerequisite #81/#71 flagged is **met**: v0.3.1's CI-verified table covers Linux x86_64, macOS arm64, Windows x86_64, and Web/WASM (Android/iOS pending). The release ships the exact binaries the `.gdextension` expects.
- The plugin now ships **native versions of the course's own environments** (`ball_chase`, `chase_the_target`, `fly_by`, `3dball`, `quadruped_walk`, `coop_collect`, `visual_chase`, `rover_3d`, `hide_and_seek`).
- Inference stack is standard Godot 4.5+ + statically-linked C++ ncnn runner — **no C#/.NET, no .NET SDK**. This half of the migration (Godot build + plugin install) is clean.

## The real cost: a course-wide Python re-platform

The native **training** stack (from the godot-native-rl repo: `docs/guide/training.md`, `scripts/setup_training.sh`, `requirements-train.txt`/`-convert.txt`/`-sf.txt`) is **not** a drop-in for the course's pinned stack:

| | Course (`requirements-course.txt`) | Native v0.3.1 training |
|---|---|---|
| Python | 3.10 | 3.13 (train) + 3.14 (convert) |
| stable-baselines3 | 2.3.2 | 2.8.0 |
| gymnasium | 0.29.1 | 1.2.2 (major break) |
| torch | 2.6.0 | 2.12.0 |
| numpy | 1.26.4 | ≥2 (major) |
| godot-rl | 0.5.0 | 0.8.2 (`--no-deps`) |
| environments | one conda `godot_env` | `.venv-train` + `.venv` + `.venv-sf` (+ ray on `.venv-train`) |

Training is driven by the native repo's scripts (e.g. `scripts/train_ball_chase.py`, **SAC**, launched by `train_ball_chase.sh` which starts the Python trainer + headless Godot training scene). Deploy is PyTorch → ncnn via `scripts/export_to_ncnn.py` (pnnx, Python 3.14).

**Impact:** every unit's training instructions and code examples (34+ units, EN+DE) are written for the 3.10 / SB3 2.3.2 / gymnasium 0.29.1 / numpy 1.x stack. gymnasium 0.29→1.2 and numpy 1→2 are breaking; a large fraction of code blocks (env `step`/`reset` API, `VecFrameStack`, wrapper signatures, etc.) must be re-checked. `setup.md` already warns against exactly these mid-course major bumps.

## Constraints this environment cannot satisfy

- **No Godot 4.5+** here — cannot open projects, enable the plugin, or run training/inference.
- **No Python 3.13/3.14** here — cannot create the venvs, resolve the new pins, or run a training script.
- Therefore **no end-to-end validation** of any migrated unit is possible from this container. All runtime validation must happen on a machine with Godot 4.5+ and the native Python stack.

## Upstream risks to confirm before student-facing changes

- **AssetLib availability.** The README describes AssetLib install ("search 'Godot Native RL'"), but `docs/guide/getting-started.md` at the v0.3.1 tag says prebuilt binaries are "not published yet — build from source." Confirm the AssetLib package is live before Setup tells students to use it.
- **macOS notarization.** Prebuilt `.dylib`s aren't Apple-notarized; Gatekeeper blocks them until `xattr -dr com.apple.quarantine …`. Setup must document this for macOS.
- **macOS Intel.** v0.3.1 verifies macOS **arm64** only — Intel Macs lose native support that the .NET stack currently provides.

## Phased plan (each phase gated on your Godot 4.5+ / Python-3.13 validation)

- **Phase 0 — Architecture doc.** ✅ `docs/architecture.md` documents the target single-stack design (PR #90).
- **Phase 1 — Foundation.** Bring the native training harness into the course (vendor or reference `setup_training.sh` + `requirements-train/convert/sf.txt` + the `train_*.sh` scripts), pin the native release, and define the course's new Python-env story (3 venvs vs. conda). Update the `setup.md` Godot-build + plugin-install surface (standard build, AssetLib, macOS quarantine), EN+DE. **Decision needed:** vendor the native scripts course-side, or have students clone the godot-native-rl repo and run its scripts?
- **Phase 2 — Unit 0** on native `ball_chase` (SAC): install → train → deploy via ncnn, EN+DE. Validate end-to-end in Godot.
- **Phase 3 — Training units (1–10)** one at a time, re-verifying each code example against gymnasium 1.2 / SB3 2.8 / numpy 2.
- **Phase 4 — Sweep + cleanup.** Retire `.NET`/MSBuild/`Sync`/`AIController` references and the Troubleshooting FAQ entries; replace the compatibility table; update `docs/curriculum.md`/`example-progression.md`.

## Decisions (locked 2026-06-15)

1. **Python-env model: conda.** Use conda env(s) with the native pins, not the native repo's `.venv` layout (the native `training.md` documents the conda alternative: create the env(s) and `pip install -r` the same requirements files).
2. **Distribution: students clone the `minigraphx/godot-native-rl` repo.** Do not vendor the addon, binaries, or training scripts into the course; do not rely on AssetLib (**not live** as of 2026-06-15). The native repo is the source of truth for the addon, the example envs, and the `setup_training.sh` / `train_*.sh` / `requirements-*.txt` training harness. Students obtain the per-platform `NcnnRunner` binary from the v0.3.1 **Release** addon zip (prebuilt) or build from source per the repo's `docs/dev/building.md`.
3. **Keep godot-rl-agents (.NET) as a fallback** — retained for macOS Intel and the existing validated stack; not deleted.
4. **Stage the gymnasium 1.2 / numpy 2 / SB3 2.8 bump per unit**, after the native path is validated end-to-end, so the just-verified 0.29.1 code keeps working under the fallback until each unit is migrated.

## Validation still required (cannot be done from this container)

Every student-facing native instruction below must be confirmed on a machine with **Godot 4.5+** and **Python 3.13/3.14**: that the cloned repo + Release binary load the `NcnnRunner` extension, that a conda env resolves the native pins, and that `train_ball_chase.sh` (or the course's conda equivalent) trains and deploys. Draft PRs stay in draft until you confirm.
