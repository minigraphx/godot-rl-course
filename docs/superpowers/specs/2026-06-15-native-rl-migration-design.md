# Native-RL Migration — Design & Scope (#81 / #71)

**Date:** 2026-06-15
**Status:** Design / scoping. No student-facing page changes yet.
**Decision recorded (#81):** full replacement of **godot-rl-agents** with **godot-native-rl**, dropping the Godot .NET edition. Distribution: install the addon from the Godot **AssetLib** (no vendored binaries in this repo).

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

## Open decisions

1. Course Python-env model: adopt the native 3-venv layout, or wrap it in a single conda env? (The native pins were chosen to coexist; conda is a documented alternative.)
2. Vendor the native training scripts/requirements into the course, or reference the godot-native-rl repo?
3. macOS Intel: accept dropping native support, or keep godot-rl-agents as an Intel-only fallback?
4. Timing of the gymnasium 1.2 / numpy 2 / SB3 2.8 bump vs. the just-completed didactic sweep, which was verified against the 0.29.1 stack.
