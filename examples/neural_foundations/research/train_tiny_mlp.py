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


def train_demo(
    epochs: int = 400,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> dict[str, object]:
    features, targets = make_dataset()
    network = create_network(seed)
    loss_curve = [mean_loss(network, features, targets)]

    for _epoch in range(epochs):
        for inputs, target in zip(features, targets):
            gradients = network.backward(inputs, target)
            network.apply_gradients(gradients, learning_rate)
        loss_curve.append(mean_loss(network, features, targets))

    return {
        "network": network,
        "features": features,
        "targets": targets,
        "loss_curve": loss_curve,
        "initial_loss": loss_curve[0],
        "final_loss": loss_curve[-1],
        "accuracy": accuracy(network, features, targets),
    }


def plot_result(result: dict[str, object], output_path: Path | None) -> None:
    network = result["network"]
    assert isinstance(network, TinyMLP)
    features = result["features"]
    targets = result["targets"]
    loss_curve = result["loss_curve"]
    assert isinstance(features, np.ndarray)
    assert isinstance(targets, np.ndarray)
    assert isinstance(loss_curve, list)

    figure, (boundary_axis, loss_axis) = plt.subplots(1, 2, figsize=(11, 4.5))
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

    loss_axis.plot(loss_curve, color="#4e79a7", linewidth=2)
    loss_axis.set(
        xlabel="Epoch",
        ylabel="Mean squared error",
        title="Loss during training",
    )
    loss_axis.grid(alpha=0.25)

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
    parser.add_argument("--save", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_demo(args.epochs, args.learning_rate, args.seed)
    print(
        "initial_loss={initial_loss:.4f} final_loss={final_loss:.4f} "
        "accuracy={accuracy:.0%}".format(**result)
    )
    plot_result(result, args.save)


if __name__ == "__main__":
    main()
