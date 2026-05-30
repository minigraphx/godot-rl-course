# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

Course material for a Godot Reinforcement Learning course. Students learn **godot-rl-agents** through a step-by-step example progression (simple → complex). Published as a static MkDocs site.

## Environments

This repo has **two** dependency sets, served by **two** conda envs — keep them separate:

| Purpose | Requirements file | Conda env |
|---------|-------------------|-----------|
| Build the docs site | `requirements.txt` (mkdocs + material + static-i18n) | `mkdocs-env` |
| Run the course training code | `requirements-course.txt` (godot-rl, stable-baselines3, torch, gymnasium) | `godot_env` |

`godot_env` is the env name students are told to create in `content/setup.md`; the whole course corpus uses it, so keep that name for the training stack. `mkdocs-env` is docs-only and never referenced by student-facing pages.

Create them once:

```bash
conda create -n mkdocs-env python=3.11 -y
conda activate mkdocs-env && pip install -r requirements.txt   # all pinned — do NOT upgrade

conda create -n godot_env python=3.10 -y   # 3.10 = the tested version per requirements-course.txt
conda activate godot_env && pip install -r requirements-course.txt
```

## Build & verify

Docs work runs in **`mkdocs-env`**:

```bash
conda activate mkdocs-env
mkdocs serve                      # live-reload at localhost:8000
mkdocs build --strict             # canonical pre-commit check; CI runs the same
```

Content lives in `content/` (Markdown). Build output goes to `site/` (English) and `site/de/` (German). Always run `mkdocs build --strict` before claiming a content change is done.

The site is multilingual via the `mkdocs-static-i18n` plugin (suffix strategy). German pages live alongside English as `unit-foo.de.md`. If `mkdocs` errors with `'i18n' plugin is not installed`, run `pip install -r requirements.txt` inside the active conda env.

## Adding or modifying a unit

When inserting unit `X` between `A` and `B`, four files must change together — `mkdocs build --strict` does **not** catch a stale breadcrumb chain:

1. `mkdocs.yml` — add `"Title": unit-x.md` under the right phase in `nav:`.
2. `content/index.md` — add the bullet under the right phase heading; update the phase summary table if scope shifts.
3. `content/unit-a.md` — change bottom breadcrumb `→ B` to `→ X` and update the "What's next" prose.
4. `content/unit-b.md` — change top breadcrumb `← A` to `← X`.

Section headings use middle dot: `## 1 · Topic`, not `## 1. Topic`. Every unit has top + bottom breadcrumbs, a "Three ways to see your AI" `!!! info` callout near the top, a Stretch Goals section near the end, and a "What's next" closing.

Pseudocode must be inside a `!!! warning "Pseudocode"` admonition. Imports in Python examples must be real and used.

## Docs

- [README.md](README.md) — Full contributor setup, conventions, PR workflow (the canonical authoring guide).
- [docs/architecture.md](docs/architecture.md) — Hybrid training/inference model and stack.
- [docs/curriculum.md](docs/curriculum.md) — Full unit syllabus and pacing notes.
- [docs/html-units.md](docs/html-units.md) — Historical conventions from the pre-MkDocs HTML build (pedagogy block names).
