#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GODOT_BIN="${GODOT_BIN:-godot}"
TIMESTEPS="${TIMESTEPS:-2048}"
SPEEDUP="${SPEEDUP:-4}"
ACTION_REPEAT="${ACTION_REPEAT:-4}"
SEED="${SEED:-0}"
TRAIN_LOG="logs/foundations_racer/train.log"
NATIVE_RL_ROOT="${NATIVE_RL_ROOT:-/Users/andreas/Documents/Godot Native RL}"
CONVERT_PY="$NATIVE_RL_ROOT/scripts/export_to_ncnn.py"
CONVERT_PYTHON="$NATIVE_RL_ROOT/.venv-train/bin/python"
ONNX_PATH="examples/neural_foundations/game/unit_03_racer/models/foundations_racer.onnx"

TRAINER_PID=""

cleanup() {
  if [[ -n "$TRAINER_PID" ]] && kill -0 "$TRAINER_PID" 2>/dev/null; then
    kill "$TRAINER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

mkdir -p "$(dirname "$TRAIN_LOG")"
echo "Starting SB3 PPO trainer; logs: $TRAIN_LOG"
PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/train_foundations_racer.py \
  --timesteps "$TIMESTEPS" \
  --speedup "$SPEEDUP" \
  --action-repeat "$ACTION_REPEAT" \
  --seed "$SEED" >"$TRAIN_LOG" 2>&1 &
TRAINER_PID="$!"

for _attempt in $(seq 1 60); do
  if grep -q "waiting for remote GODOT connection" "$TRAIN_LOG"; then
    break
  fi
  if ! kill -0 "$TRAINER_PID" 2>/dev/null; then
    echo "Trainer exited before opening port 11008"
    cat "$TRAIN_LOG"
    exit 1
  fi
  sleep 1
done

if ! grep -q "waiting for remote GODOT connection" "$TRAIN_LOG"; then
  echo "Trainer did not open port 11008"
  cat "$TRAIN_LOG"
  exit 1
fi

echo "Launching Godot racer training scene"
"$GODOT_BIN" --headless \
  --path examples/neural_foundations/game \
  res://unit_03_racer/racer_train.tscn \
  "speedup=$SPEEDUP" \
  "action_repeat=$ACTION_REPEAT"

wait "$TRAINER_PID"
TRAINER_PID=""
cat "$TRAIN_LOG"

if [[ -x "$CONVERT_PYTHON" && -f "$CONVERT_PY" && -f "$ONNX_PATH" ]]; then
  "$CONVERT_PYTHON" "$CONVERT_PY" "$REPO_ROOT/$ONNX_PATH"
else
  echo "Skipping ncnn conversion; converter or ONNX file is missing."
  echo "Expected converter: $CONVERT_PY"
  echo "Expected Python: $CONVERT_PYTHON"
  echo "Expected ONNX: $ONNX_PATH"
  exit 1
fi
