# Cliff Jumper Unit 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Unit 1's chase/retreat example with an interactive cliff-jump trigger and clarify the opening neuron notation.

**Architecture:** Keep the shared tiny-neuron implementation and add sigmoid activation support for the jumper. A self-contained Godot scene exposes laboratory sliders, guided cases, and a visual test run. English and German course pages describe the same workflow with descriptive terms before symbols.

**Tech Stack:** Godot 4 GDScript, Python unittest, MkDocs Material

---

### Task 1: Specify the jumper's decisions with tests

**Files:**
- Modify: `examples/neural_foundations/game/test/test_tiny_neuron.gd`
- Modify: `examples/neural_foundations/game/shared/tiny_neuron.gd`

- [x] Add failing assertions for sigmoid output and the three guided jumper cases.
- [x] Run the Godot headless test and confirm the missing sigmoid API fails.
- [x] Add selectable activation support while keeping tanh as the default.
- [x] Re-run the test and confirm the neuron-level assertions pass.

### Task 2: Build the interactive jumper scene

**Files:**
- Create: `examples/neural_foundations/game/unit_01_jumper/cliff_jumper.gd`
- Create: `examples/neural_foundations/game/unit_01_jumper/unit_01_jumper.tscn`
- Modify: `examples/neural_foundations/game/test/test_tiny_neuron.gd`

- [x] Point the scene test at `unit_01_jumper.tscn` and assert the visible input,
  contribution, score, output, decision, and guided-case labels.
- [x] Run the test and confirm it fails because the scene does not exist.
- [x] Implement lab sliders, parameter sliders, the `WAIT`/`JUMP` threshold,
  three-case scoring, reset, and visual test-run state.
- [x] Run the Godot tests and confirm the new scene wiring passes.

### Task 3: Clarify the fixed-number walkthrough

**Files:**
- Modify: `examples/neural_foundations/research/tests/test_neuron.py`
- Modify: `content/unit-neural-01.md`
- Modify: `content/unit-neural-01.de.md`

- [x] Replace generic `x`/`w` walkthrough labels with `speed input`,
  `closeness input`, named weights, sum, and sigmoid output.
- [x] Explain symbols only after descriptive names and remove "frozen neuron".
- [x] Replace the chase/retreat lesson with the lab-to-test-run jumper workflow
  and its three guided cases in both languages.

### Task 4: Align curriculum references and verify

**Files:**
- Modify: `docs/neural-foundations-plan.md`
- Modify: `docs/superpowers/specs/2026-06-06-neural-foundations-teaching-design.md`

- [x] Replace remaining Unit 1 chase/retreat planning references with
  cliff-jump timing.
- [x] Run Python neuron tests.
- [x] Run all Godot neural-foundation headless tests.
- [x] Run `mkdocs build --strict` in `mkdocs-env`.
- [x] Review the diff, commit all task files without staging unrelated local
  changes, and push `main`.
