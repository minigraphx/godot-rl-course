# Godot Reinforcement Learning Course

Learn deep RL with **Godot 4** and **godot-rl-agents**. Train locally with Python, deploy agents to the browser via ONNX.

The course is published as a static MkDocs site. Markdown sources live in [`content/`](content/); the rendered HTML is built into `site/` (gitignored).

**Read the course:** open `mkdocs serve` locally, or visit the published site.

---

## Contributor quick start

This project uses **conda** for its Python environment. If you don't already use conda, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first; `python -m venv` will also work if you prefer, but the maintainer's environment is conda-based.

```bash
# 1. Clone
git clone https://github.com/minigraphx/godot-rl-course.git
cd godot-rl-course

# 2. Create / activate the conda env and install pinned deps
conda create -n mkdocs-env python=3.11 -y       # one time
conda activate mkdocs-env
pip install -r requirements.txt                # mkdocs, mkdocs-material, mkdocs-static-i18n

# 3. Serve with live reload at http://localhost:8000
mkdocs serve

# 4. Build the static site (outputs to site/, with site/de/ for German)
mkdocs build --strict
```

`mkdocs build --strict` is the canonical check before opening a PR — it fails on broken links, missing files, or other warnings. CI (`.github/workflows/docs-ci.yml`) runs the same command plus `linkchecker` on every PR.

> **Do not upgrade** `mkdocs`, `mkdocs-material`, or `mkdocs-static-i18n` past the versions in [`requirements.txt`](requirements.txt). The pins are intentional: newer Material releases have published breaking changes (theme rewrite, plugin removal) on a 2026 roadmap, and the i18n plugin's config schema shifts between minor versions.

If you see `Config value 'plugins': The "i18n" plugin is not installed`, your environment is missing `mkdocs-static-i18n` — re-run `pip install -r requirements.txt` inside the activated env.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `content/` | All published Markdown — units, glossary, troubleshooting, hardware-setup. This is `mkdocs.yml`'s `docs_dir`. |
| `content/*.de.md` | German translations (see [Translations](#translations) below). |
| `mkdocs.yml` | Site config, **the navigation tree**, and the `i18n` plugin config. Adding a unit or a language means editing here. |
| `requirements.txt` | Pinned MkDocs + Material + i18n plugin versions. |
| `.github/workflows/docs-ci.yml` | CI: runs `mkdocs build --strict` and a `linkchecker` pass on every PR. |
| `docs/` | Internal docs for contributors (curriculum map, architecture, conventions). **Not** published. |
| `internal/` | Working notes, gap analyses. Not published. |
| `site/` | Build output (`site/` is English, `site/de/` is German). Gitignored. |
| `CLAUDE.md` | Onboarding notes for AI coding assistants working on the repo. |

Related contributor docs:

- [`docs/curriculum.md`](docs/curriculum.md) — full syllabus and pacing.
- [`docs/neural-foundations-plan.md`](docs/neural-foundations-plan.md) — planned neuron-to-network track ending in a learning 2D vehicle.
- [`docs/architecture.md`](docs/architecture.md) — training-vs-inference architecture of the example projects.
- [`docs/example-progression.md`](docs/example-progression.md) — the example ladder the units build on.
- [`docs/html-units.md`](docs/html-units.md) — historical conventions from the pre-MkDocs HTML build (still useful for pedagogy block names).

---

## Adding or modifying a unit

### File and structure conventions

Every unit Markdown file is expected to have, in order:

1. **H1 title** — `# Unit N — Title` or `# Topic Name`.
2. **Top breadcrumb** — `[← Previous Unit](unit-prev.md) · [Course home](index.md)` immediately under the title.
3. **"Three ways to see your AI" callout** — early in the file:
   ```markdown
   !!! info "Three ways to see your AI"
       Godot (...) · TensorBoard (`metric/name`) · AIController (...)
   ```
4. **Numbered sections** — `## 1 · Topic`, `## 2 · Topic`, etc. The `·` (middle dot) is the convention, not `.`.
5. **`## N · Stretch goals`** — towards the end, a list of optional deepening exercises.
6. **`## What's next`** — final section pointing forward.
7. **Bottom breadcrumb** — `[← Prev](unit-prev.md) · [Course home](index.md) · [→ Next](unit-next.md)`.

### Wiring a new unit into the course

When you insert a unit `X` between existing units `A` and `B`, **four files** need updating:

| File | Change |
|------|--------|
| `mkdocs.yml` | Add `"X Title": unit-x.md` to the correct phase block in `nav:`. |
| `content/index.md` | Add a bullet under the right phase heading, and update the phase summary table if scope changes. |
| `content/unit-a.md` | Bottom breadcrumb: change `→ B` to `→ X`. Update the "What's next" prose. |
| `content/unit-b.md` | Top breadcrumb: change `← A` to `← X`. |

Forgetting any one of these breaks the reading flow. `mkdocs build --strict` catches dangling links but not stale `← / →` chains.

### Other authoring rules

- **GDScript correctness.** Code in the course is read and copied. Verify any GDScript you publish against the Godot 4 docs — in particular, never use a static config constant (e.g. `PhysicsServer3D.HINGE_JOINT_PARAM_*`) where a runtime value is required. (See git history for the locomotion unit fix as a worked example.)
- **Python code blocks** should be runnable as written — no pseudocode without an `!!! warning "Pseudocode"` admonition. Imports must be real and used; dead imports are flagged in review.
- **Cross-links use Markdown paths,** not URLs: `[→ SAC](unit-sac.md)`, not `/unit-sac/` or absolute links.
- **Admonitions** use the `pymdownx`/Material syntax: `!!! info`, `!!! warning`, `!!! tip`, `??? details` (collapsible).
- **Tables** are preferred for comparisons (algorithms, hyperparameters, schedules). Keep column counts ≤ 5 for readability.

---

## Translations

The site uses [`mkdocs-static-i18n`](https://github.com/ultrabug/mkdocs-static-i18n) with the **suffix** strategy. English is the default and falls back transparently for any page that hasn't been translated yet.

- Default English page: `content/unit-03.md` → `https://.../unit-03/`
- German translation: `content/unit-03.de.md` → `https://.../de/unit-03/`
- Languages, navigation labels, and per-language `site_name` / `site_description` are configured in the `plugins:` block of `mkdocs.yml`.

### Adding or updating a translation

1. Copy the source file (`unit-foo.md`) to `unit-foo.<locale>.md` (e.g. `unit-foo.de.md`).
2. Translate the prose, but **keep code blocks, file paths, library names, and TensorBoard metric names in English** — these match what students see in Godot, SB3, and the CLI.
3. **Don't translate Markdown link targets** — `[Hierarchical RL](unit-hierarchical.md)` stays as-is; the i18n plugin handles language routing.
4. Translate breadcrumb labels (`← Vorheriges` etc.) but keep their target file references English.
5. If you add a new language, extend the `languages:` list in `mkdocs.yml` (under `plugins.i18n`) and translate the `nav_translations:` block too.
6. Run `mkdocs build --strict` — the i18n plugin logs which pages fell back to English so you can see gaps.

> If a translation file references a header anchor (`unit-02.md#step-3-...`), the German heading may have a different anchor — verify the link still resolves in the built `site/de/` output.

---

## Commit and PR workflow

- **Branch naming:** topical, kebab-case (`fix-locomotion-hinge-param`, `add-multitask-unit`).
- **Commit style:** Conventional Commits.
  - `feat: add unit-X — short description` — new units.
  - `fix: ...` — corrections to existing content.
  - `chore: ...` — non-content changes (nav wiring, requirements bumps).
  - `docs: ...` — internal `docs/` or `README.md` changes.
- **PR description:** explain *why* the change exists (which gap it closes, what reader confusion it removes). Reference issues with `Closes #NN`.
- **Pre-merge checklist:**
  - [ ] `mkdocs build --strict` passes locally (CI runs the same).
  - [ ] Nav chain (`← / →` breadcrumbs) is consistent end-to-end through the touched units.
  - [ ] `mkdocs.yml`, `content/index.md`, and adjacent units are updated for any new unit.
  - [ ] Code examples compile / parse in their respective languages.
  - [ ] If touching a translated page, the `.de.md` (and other locales) is either updated to match or left alone deliberately (English fallback is fine for new sections).

---

## Reporting issues

Open a GitHub issue describing:

- The unit (or section anchor) involved.
- What's wrong or missing — be concrete (a wrong equation, a broken link, a missing prerequisite).
- A suggested fix if you have one, even a rough sketch.

For curriculum-level proposals (new units, restructured phases), open the issue against `docs/curriculum.md` rather than a unit file so the scope is clear.
