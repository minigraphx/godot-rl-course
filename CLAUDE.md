# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

Course material for a Godot Reinforcement Learning course. Students learn **godot-rl-agents** through a step-by-step example progression (simple → complex).

## Build

```bash
mkdocs serve   # live-reload at localhost:8000
mkdocs build   # outputs distributable static site to site/
```

Content lives in `content/` (Markdown). Output goes to `site/` (gitignored).

## Docs

- [docs/architecture.md](docs/architecture.md) — Hybrid training/inference model and stack
- [docs/curriculum.md](docs/curriculum.md) — Full unit syllabus and pacing notes
- [docs/html-units.md](docs/html-units.md) — How to add a new unit
