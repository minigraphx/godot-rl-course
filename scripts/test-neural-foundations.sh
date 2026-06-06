#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

GODOT_BIN="${GODOT_BIN:-godot}"

run_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    "$PYTHON_BIN" "$@"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    python "$@"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    conda run --no-capture-output -n godot_env python "$@"
    return
  fi
  echo "error: python not found. Activate godot_env or set PYTHON_BIN." >&2
  exit 1
}

run_python -m unittest discover \
  -s examples/neural_foundations/research/tests \
  -p 'test_*.py'

shopt -s nullglob
for test_script in examples/neural_foundations/game/test/test_*.gd; do
  "$GODOT_BIN" --headless \
    --path examples/neural_foundations/game \
    --script "res://test/$(basename "$test_script")"
done

conda run -n mkdocs-env mkdocs build --strict
