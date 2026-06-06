#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GODOT_BIN="${GODOT_BIN:-godot}"

"$PYTHON_BIN" -m unittest discover \
  -s examples/neural_foundations/research/tests \
  -p 'test_*.py'

shopt -s nullglob
for test_script in examples/neural_foundations/game/test/test_*.gd; do
  "$GODOT_BIN" --headless \
    --path examples/neural_foundations/game \
    --script "res://test/$(basename "$test_script")"
done

conda run -n mkdocs-env mkdocs build --strict
