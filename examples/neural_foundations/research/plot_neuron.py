import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import RadioButtons, Slider
import numpy as np

if __package__:
    from .neuron import neuron_output
else:
    from neuron import neuron_output


FEATURES = np.array(
    [
        [0.10, 0.15],
        [0.15, 0.65],
        [0.25, 0.40],
        [0.35, 0.80],
        [0.45, 0.20],
        [0.55, 0.55],
        [0.65, 0.35],
        [0.70, 0.75],
        [0.82, 0.25],
        [0.90, 0.60],
    ],
    dtype=float,
)
TARGETS = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1], dtype=int)
PROBE = np.array([0.65, 0.35], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore one neuron's boundary.")
    parser.add_argument(
        "--save",
        type=Path,
        help="Save one non-interactive frame instead of opening a window.",
    )
    return parser.parse_args()


def predict(
    features: np.ndarray,
    weights: np.ndarray,
    bias: float,
    activation: str,
) -> tuple[np.ndarray, np.ndarray]:
    outputs = np.array(
        [
            neuron_output(row, weights, bias, activation)
            for row in features
        ]
    )
    threshold = 0.5 if activation in {"step", "sigmoid"} else 0.0
    return outputs, (outputs >= threshold).astype(int)


def main() -> None:
    args = parse_args()
    weights = np.array([1.2, -0.9], dtype=float)
    bias = -0.1
    activation = "step"

    figure, axis = plt.subplots(figsize=(10.5, 7))
    figure.subplots_adjust(left=0.10, right=0.72, bottom=0.28)
    figure.canvas.manager.set_window_title("One neuron, one decision")

    weight_1_axis = figure.add_axes((0.12, 0.17, 0.54, 0.03))
    weight_2_axis = figure.add_axes((0.12, 0.12, 0.54, 0.03))
    bias_axis = figure.add_axes((0.12, 0.07, 0.54, 0.03))
    activation_axis = figure.add_axes((0.76, 0.58, 0.18, 0.18))
    weight_1_slider = Slider(weight_1_axis, "w₁", -2.0, 2.0, valinit=weights[0])
    weight_2_slider = Slider(weight_2_axis, "w₂", -2.0, 2.0, valinit=weights[1])
    bias_slider = Slider(bias_axis, "bias", -1.5, 1.5, valinit=bias)
    activation_buttons = RadioButtons(
        activation_axis,
        ("step", "sigmoid", "tanh"),
        active=0,
    )
    calculation_text = figure.text(
        0.75,
        0.46,
        "",
        family="monospace",
        fontsize=10,
        va="top",
    )
    metric_text = figure.text(0.75, 0.27, "", fontsize=11, va="top")

    background_colors = ListedColormap(["#eaf3ff", "#fff0e6"])

    def redraw(_value: object = None) -> None:
        current_weights = np.array(
            [weight_1_slider.val, weight_2_slider.val],
            dtype=float,
        )
        current_bias = bias_slider.val
        current_activation = activation_buttons.value_selected

        axis.clear()
        grid_x, grid_y = np.meshgrid(
            np.linspace(0.0, 1.0, 160),
            np.linspace(0.0, 1.0, 160),
        )
        grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        _, grid_predictions = predict(
            grid,
            current_weights,
            current_bias,
            current_activation,
        )
        axis.contourf(
            grid_x,
            grid_y,
            grid_predictions.reshape(grid_x.shape),
            levels=[-0.5, 0.5, 1.5],
            cmap=background_colors,
            alpha=0.9,
        )

        _, predictions = predict(
            FEATURES,
            current_weights,
            current_bias,
            current_activation,
        )
        point_colors = np.where(predictions == 1, "#d95f02", "#2b6cb0")
        axis.scatter(
            FEATURES[:, 0],
            FEATURES[:, 1],
            c=point_colors,
            s=90,
            edgecolors="white",
            linewidths=1.2,
            label="prediction",
        )
        mistakes = predictions != TARGETS
        if mistakes.any():
            axis.scatter(
                FEATURES[mistakes, 0],
                FEATURES[mistakes, 1],
                facecolors="none",
                edgecolors="#d62728",
                marker="o",
                s=190,
                linewidths=2.2,
                label="misclassified",
            )
        axis.scatter(
            [PROBE[0]],
            [PROBE[1]],
            color="#111111",
            marker="*",
            s=180,
            label="calculation probe",
            zorder=5,
        )

        if abs(current_weights[1]) > 1e-9:
            boundary_x = np.array([0.0, 1.0])
            boundary_y = -(
                current_weights[0] * boundary_x + current_bias
            ) / current_weights[1]
            axis.plot(
                boundary_x,
                boundary_y,
                color="#222222",
                linewidth=2,
                label="z = 0 boundary",
            )
        elif abs(current_weights[0]) > 1e-9:
            axis.axvline(
                -current_bias / current_weights[0],
                color="#222222",
                linewidth=2,
                label="z = 0 boundary",
            )

        contribution_1 = current_weights[0] * PROBE[0]
        contribution_2 = current_weights[1] * PROBE[1]
        weighted_sum = contribution_1 + contribution_2 + current_bias
        output = neuron_output(
            PROBE,
            current_weights,
            current_bias,
            current_activation,
        )
        calculation_text.set_text(
            "Current probe\n"
            f"{PROBE[0]:.2f} × {current_weights[0]:+.2f} = "
            f"{contribution_1:+.3f}\n"
            f"{PROBE[1]:.2f} × {current_weights[1]:+.2f} = "
            f"{contribution_2:+.3f}\n"
            f"bias             = {current_bias:+.3f}\n"
            f"z                = {weighted_sum:+.3f}\n"
            f"{current_activation}(z) = {output:+.3f}"
        )
        metric_text.set_text(
            f"Accuracy: {(predictions == TARGETS).mean():.0%}\n"
            "Blue = predicted safe\n"
            "Orange = predicted unsafe\n"
            "Red ring = wrong prediction"
        )
        axis.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            xlabel="Normalized temperature",
            ylabel="Normalized pressure",
            title="Safe or unsafe? Move one parameter at a time.",
        )
        axis.grid(alpha=0.18)
        axis.legend(loc="upper left", fontsize=8)
        figure.canvas.draw_idle()

    for slider in (weight_1_slider, weight_2_slider, bias_slider):
        slider.on_changed(redraw)
    activation_buttons.on_clicked(redraw)
    redraw()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=160)
        return
    plt.show()


if __name__ == "__main__":
    main()
