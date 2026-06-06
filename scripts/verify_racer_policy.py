#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Foundations racer policy parity.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("examples/neural_foundations/game/unit_03_racer/models/foundations_racer.zip"),
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("examples/neural_foundations/game/unit_03_racer/models/foundations_racer.onnx"),
    )
    parser.add_argument(
        "--ncnn-param",
        type=Path,
        default=Path("examples/neural_foundations/game/unit_03_racer/models/foundations_racer.ncnn.param"),
    )
    parser.add_argument(
        "--ncnn-bin",
        type=Path,
        default=Path("examples/neural_foundations/game/unit_03_racer/models/foundations_racer.ncnn.bin"),
    )
    parser.add_argument("--ncnn-input", default="in0")
    parser.add_argument("--ncnn-output", default="out0")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("examples/neural_foundations/game/unit_03_racer/models/verification_report.json"),
    )
    return parser.parse_args()


def fixed_observations() -> np.ndarray:
    rng = np.random.default_rng(0)
    observations = rng.uniform(-1.0, 1.0, size=(20, 6)).astype(np.float32)
    observations[:, :3] = rng.uniform(0.0, 1.0, size=(20, 3)).astype(np.float32)
    observations[:, 5] = rng.uniform(0.0, 1.0, size=20).astype(np.float32)
    return observations


def pytorch_outputs(model_path: Path, observations: np.ndarray) -> np.ndarray:
    import torch
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    obs = torch.tensor(observations, dtype=torch.float32)
    with torch.no_grad():
        distribution = model.policy.get_distribution({"obs": obs})
        return distribution.distribution.mean.cpu().numpy()


def onnx_outputs(onnx_path: Path, observations: np.ndarray) -> tuple[np.ndarray, list[str], list[str]]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    outputs = session.run(None, {input_names[0]: observations})[0]
    return outputs, input_names, output_names


def ncnn_outputs(
    param_path: Path,
    bin_path: Path,
    observations: np.ndarray,
    input_blob: str,
    output_blob: str,
) -> np.ndarray:
    import ncnn

    net = ncnn.Net()
    net.load_param(str(param_path))
    net.load_model(str(bin_path))
    rows = []
    for observation in observations:
        extractor = net.create_extractor()
        extractor.input(input_blob, ncnn.Mat(observation))
        _code, output = extractor.extract(output_blob)
        rows.append(np.array(output, dtype=np.float32))
    return np.vstack(rows)


def main() -> None:
    args = parse_args()
    observations = fixed_observations()
    torch_values = pytorch_outputs(args.model_path, observations)
    onnx_values, input_names, output_names = onnx_outputs(args.onnx_path, observations)
    torch_onnx_error = float(np.max(np.abs(torch_values - onnx_values)))

    report: dict[str, object] = {
        "input_names": input_names,
        "output_names": output_names,
        "observations": len(observations),
        "torch_onnx_max_abs_error": torch_onnx_error,
    }
    if torch_onnx_error > 1e-5:
        report["status"] = "failed"
        report["reason"] = "PyTorch and ONNX differ by more than 1e-5"
    elif not args.ncnn_param.exists() or not args.ncnn_bin.exists():
        report["status"] = "failed"
        report["reason"] = "ncnn param/bin files are missing"
    else:
        ncnn_values = ncnn_outputs(
            args.ncnn_param,
            args.ncnn_bin,
            observations,
            args.ncnn_input,
            args.ncnn_output,
        )
        onnx_ncnn_error = float(np.max(np.abs(onnx_values - ncnn_values)))
        report["onnx_ncnn_max_abs_error"] = onnx_ncnn_error
        report["status"] = "passed" if onnx_ncnn_error <= 1e-2 else "failed"
        if onnx_ncnn_error > 1e-2:
            report["reason"] = "ONNX and ncnn differ by more than 1e-2"

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("inputs:", ", ".join(input_names))
    print("outputs:", ", ".join(output_names))
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
