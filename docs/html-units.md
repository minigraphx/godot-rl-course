# HTML Unit Conventions

All unit files share a consistent structure and scaffold.

## Numbering

| File | Unit | Content |
|------|------|---------|
| `Unit-00.html` | 0 | Setup & first run (Mac PDF + BallChase) |
| `Unit-01.html` | 1 | RL foundations |
| `Unit-02.html` | 2 | Lunar Lander build |
| `Unit-03.html` | 3 | CrossTheRoad + DQN |
| `Unit-04.html` | 4 | JumperHard + PPO *(planned)* |
| `Unit-05.html`–`Unit-10.html` | 5–10 | See [example-progression.md](example-progression.md) |
| `index.html` | — | Course home |
| `godot_rl_course_reference.html` | — | CLI / plugin appendix |

## CSS / Theme

Variables defined in `:root`:
- `--accent` #6c8ef7 (blue), `--accent2` #4ecca3 (teal), `--warn` #f7c86c, `--danger` #f76c6c
- `--bg` #0f1117, `--surface` #1a1d27, `--surface2` #22263a, `--text` #e2e8f0, `--muted` #8892b0

## Layout

- Fixed left sidebar (260 px) for in-page navigation
- Content area with `<h2 id="...">` section anchors
- Code blocks use `<pre>` inside a `.code-block` wrapper

## Pedagogy blocks (recurring)

- **First success checklist** — Unit 0 (`🎯 After this unit`)
- **Three ways to see your AI** — Godot · TensorBoard · `AIController` (index + units 0–3+)
- **Fast path / Phase A** — Unit 1 (theory ↔ tweak early); Unit 2 (SimpleReachGoal before Lander build)

## Adding a New Unit

Copy scaffold from an existing unit, then:
1. Update `<title>` and sidebar nav
2. Link from `index.html` and previous unit's "What's next"
3. Follow the example progression rhythm in [example-progression.md](example-progression.md)
