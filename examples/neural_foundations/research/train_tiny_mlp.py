import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .tiny_mlp import TinyMLP
else:
    from tiny_mlp import TinyMLP


def make_dataset() -> tuple[np.ndarray, np.ndarray]:
    features = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.15, 0.85],
            [0.85, 0.15],
            [0.15, 0.15],
            [0.85, 0.85],
        ],
        dtype=np.float64,
    )
    targets = np.array(
        [[-0.8], [0.8], [0.8], [-0.8], [0.8], [0.8], [-0.8], [-0.8]],
        dtype=np.float64,
    )
    return features, targets


def create_network(seed: int) -> TinyMLP:
    rng = np.random.default_rng(seed)
    return TinyMLP(
        w_hidden=rng.normal(0.0, 0.7, size=(4, 2)),
        b_hidden=np.zeros(4, dtype=np.float64),
        w_output=rng.normal(0.0, 0.7, size=(1, 4)),
        b_output=np.zeros(1, dtype=np.float64),
    )


def train_validation_split(
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    validation_indices = np.array([6, 7], dtype=np.int64)
    return (
        features[train_indices],
        targets[train_indices],
        features[validation_indices],
        targets[validation_indices],
    )


def mean_loss(network: TinyMLP, features: np.ndarray, targets: np.ndarray) -> float:
    losses = [
        network.loss(network.forward(inputs), target)
        for inputs, target in zip(features, targets)
    ]
    return float(np.mean(losses))


def accuracy(network: TinyMLP, features: np.ndarray, targets: np.ndarray) -> float:
    correct = 0
    for inputs, target in zip(features, targets):
        prediction = network.forward(inputs)
        if np.sign(prediction[0]) == np.sign(target[0]):
            correct += 1
    return correct / len(features)


def selected_gradient_signs(network: TinyMLP) -> dict[str, str]:
    features, targets = make_dataset()
    gradients = network.backward(features[2], targets[2])

    def sign_label(value: float) -> str:
        if value > 0.0:
            return "+"
        if value < 0.0:
            return "-"
        return "0"

    return {
        "w_hidden[0,0]": sign_label(float(gradients["w_hidden"][0, 0])),
        "w_output[0,0]": sign_label(float(gradients["w_output"][0, 0])),
    }


def train_once(
    epochs: int = 400,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> dict[str, object]:
    features, targets = make_dataset()
    train_features, train_targets, validation_features, validation_targets = (
        train_validation_split(features, targets)
    )
    network = create_network(seed)
    train_loss_curve = [mean_loss(network, train_features, train_targets)]
    validation_loss_curve = [
        mean_loss(network, validation_features, validation_targets)
    ]

    for _epoch in range(epochs):
        for inputs, target in zip(train_features, train_targets):
            gradients = network.backward(inputs, target)
            network.apply_gradients(gradients, learning_rate)
        train_loss_curve.append(mean_loss(network, train_features, train_targets))
        validation_loss_curve.append(
            mean_loss(network, validation_features, validation_targets)
        )

    return {
        "network": network,
        "features": features,
        "targets": targets,
        "train_loss_curve": train_loss_curve,
        "validation_loss_curve": validation_loss_curve,
        "loss_curve": train_loss_curve,
        "initial_train_loss": train_loss_curve[0],
        "final_train_loss": train_loss_curve[-1],
        "initial_validation_loss": validation_loss_curve[0],
        "final_validation_loss": validation_loss_curve[-1],
        "initial_loss": train_loss_curve[0],
        "final_loss": train_loss_curve[-1],
        "accuracy": accuracy(network, features, targets),
    }


def compare_seeds(
    seeds: tuple[int, ...],
    epochs: int,
    learning_rate: float,
) -> list[dict[str, float | int]]:
    seed_results = []
    for seed in seeds:
        result = train_once(epochs, learning_rate, seed)
        seed_results.append(
            {
                "seed": seed,
                "final_train_loss": float(result["final_train_loss"]),
                "final_validation_loss": float(result["final_validation_loss"]),
                "accuracy": float(result["accuracy"]),
            }
        )
    return seed_results


def train_demo(
    epochs: int = 400,
    learning_rate: float = 0.05,
    seed: int = 0,
    comparison_seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, object]:
    result = train_once(epochs, learning_rate, seed)
    network = result["network"]
    assert isinstance(network, TinyMLP)
    result["gradient_signs"] = selected_gradient_signs(network)
    result["seed_results"] = compare_seeds(comparison_seeds, epochs, learning_rate)
    return result


def plot_result(result: dict[str, object], output_path: Path | None) -> None:
    network = result["network"]
    assert isinstance(network, TinyMLP)
    features = result["features"]
    targets = result["targets"]
    train_loss_curve = result["train_loss_curve"]
    validation_loss_curve = result["validation_loss_curve"]
    gradient_signs = result["gradient_signs"]
    seed_results = result["seed_results"]
    assert isinstance(features, np.ndarray)
    assert isinstance(targets, np.ndarray)
    assert isinstance(train_loss_curve, list)
    assert isinstance(validation_loss_curve, list)
    assert isinstance(gradient_signs, dict)
    assert isinstance(seed_results, list)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    boundary_axis, loss_axis, gradient_axis, seed_axis = axes.ravel()
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, 1.0, 120),
        np.linspace(0.0, 1.0, 120),
    )
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    predictions = np.array([network.forward(row)[0] for row in grid])

    boundary_axis.contourf(
        grid_x,
        grid_y,
        predictions.reshape(grid_x.shape),
        levels=[-1.0, 0.0, 1.0],
        colors=["#eaf3ff", "#fff0e6"],
        alpha=0.95,
    )
    colors = np.where(targets[:, 0] > 0.0, "#d95f02", "#2b6cb0")
    boundary_axis.scatter(
        features[:, 0],
        features[:, 1],
        c=colors,
        edgecolors="white",
        s=90,
        linewidths=1.2,
    )
    boundary_axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="Input x1",
        ylabel="Input x2",
        title="Tiny network decision regions",
    )

    loss_axis.plot(train_loss_curve, color="#4e79a7", linewidth=2, label="train")
    loss_axis.plot(
        validation_loss_curve,
        color="#f28e2b",
        linewidth=2,
        label="validation",
    )
    loss_axis.set(
        xlabel="Epoch",
        ylabel="Mean squared error",
        title="Train versus validation loss",
    )
    loss_axis.legend()
    loss_axis.grid(alpha=0.25)

    gradient_axis.axis("off")
    gradient_lines = [
        f"{name}: move {'down' if sign == '+' else 'up' if sign == '-' else 'flat'}"
        for name, sign in gradient_signs.items()
    ]
    gradient_axis.text(
        0.05,
        0.75,
        "Selected gradient signs\n" + "\n".join(gradient_lines),
        fontsize=12,
        va="top",
    )

    seed_labels = [str(row["seed"]) for row in seed_results]
    seed_losses = [float(row["final_validation_loss"]) for row in seed_results]
    seed_axis.bar(seed_labels, seed_losses, color="#59a14f")
    seed_axis.set(
        xlabel="Seed",
        ylabel="Final validation loss",
        title="Seed comparison",
    )

    figure.tight_layout()
    if output_path is None:
        plt.show()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny visual MLP.")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--comparison-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--save", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_demo(
        args.epochs,
        args.learning_rate,
        args.seed,
        tuple(args.comparison_seeds),
    )
    print(
        "train_loss={initial_train_loss:.4f}->{final_train_loss:.4f} "
        "validation_loss={initial_validation_loss:.4f}->{final_validation_loss:.4f} "
        "accuracy={accuracy:.0%}".format(**result)
    )
    for row in result["seed_results"]:
        print(
            "seed={seed} validation_loss={final_validation_loss:.4f} "
            "accuracy={accuracy:.0%}".format(**row)
        )
    plot_result(result, args.save)


if __name__ == "__main__":
    main()
