"""Render the component-network comparison using real fits and both renderers."""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning

import htfa
from htfa.io import Subject
from htfa.tfa import component_factors


def main():
    out = Path("docs/_static/tutorials")
    out.mkdir(parents=True, exist_ok=True)
    axis = np.arange(-40.0, 41.0, 2.0)
    coords = np.stack(
        np.meshgrid(axis, axis, np.arange(-24.0, 25.0, 2.0), indexing="ij"), -1
    ).reshape(-1, 3)
    centers = np.array(
        [
            [[-18.0, -18.0, 0.0], [18.0, -18.0, 0.0]],
            [[-18.0, 18.0, 0.0], [18.0, 18.0, 0.0]],
        ]
    )
    widths = np.full((2, 2), 45.0)
    pi = np.array([[0.3, 0.7], [0.6, 0.4]])
    weights = np.random.default_rng(12).normal(size=(2, 30))
    weights[1] = 0.65 * weights[0] + 0.75 * weights[1]
    E = np.corrcoef(weights)
    np.fill_diagonal(E, 0)
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for row, kernel in enumerate(("rbf", "laplace", "mexican_hat")):
        X = component_factors(coords, centers, widths, pi, kernel) @ weights
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = htfa.TFA(
                K=2,
                kernel=kernel,
                n_components=2,
                standardize=False,
                max_iter=3,
                random_state=0,
            ).fit(X, coords, start=(centers, widths, pi))
        result = htfa.HTFAResults(model, [Subject(X, coords)])
        options = dict(
            subject=0,
            connectivity=E,
            mesh=None,
            snap=False,
            size=(600, 450),
            view="dorsal",
            node_color="darkorange",
        )
        for col, mode in enumerate(("components", "summary")):
            img = result.plot_network(**options, node_mode=mode)
            pixels = np.asarray(img)
            yy, xx = np.where(np.min(pixels, axis=2) < 245)
            pad = 20
            bounds = (
                max(0, xx.min() - pad),
                max(0, yy.min() - pad),
                min(img.width, xx.max() + pad),
                min(img.height, yy.max() + pad),
            )
            axes[row, col].imshow(img.crop(bounds))
            axes[row, col].axis("off")
        # Matching graph values come from the fitted weights, as in E.
        result.plot_connectivity(snap=False, threshold=0.0, ax=axes[row, 2])
        axes[row, 0].set_title(f"{kernel}: component lobes")
        axes[row, 1].set_title("One centroid per factor")
        axes[row, 2].set_title("Signed glass-brain projections")
        if kernel == "mexican_hat":
            result.animate_network(
                out / "12_component_networks.gif",
                subject=0,
                connectivity=E,
                n_frames=24,
                fps=8,
                spin=(0, 0, 1),
                mesh=None,
                snap=False,
                size=(480, 360),
                node_color="darkorange",
            )
    fig.subplots_adjust(wspace=0.15, hspace=0.3)
    fig.savefig(out / "12_component_networks.png", dpi=140, bbox_inches="tight")
    plt.close("all")
    print(out / "12_component_networks.png")


if __name__ == "__main__":
    main()
