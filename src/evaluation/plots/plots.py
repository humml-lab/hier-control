from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch as tc


@tc.no_grad()
def plot_gaussian(
    X_true: tc.Tensor,
    X_gen: tc.Tensor,
    title: str,
    labels: Optional[list[str]],
    n_plotted_dims: int,
    dims_to_plot: Optional[list[int]] = None,
    T: Optional[int] = None,
):
    assert dims_to_plot is not None or n_plotted_dims is not None, (
        "Either dims_to_plot or n_plotted_dims must be provided."
    )

    assert dims_to_plot is None or max(dims_to_plot) < min(X_true.shape[1], X_gen.shape[1]), (
        "dims_to_plot contains dimension indices that exceed the number of dimensions in X_true or X_gen."
    )
    if dims_to_plot is None:
        dims_to_plot = list(range(n_plotted_dims))

    if labels is None:
        labels = ["True", "Generated"]

    if T is None:
        T = min(X_true.shape[0], X_gen.shape[0])

    X_true_np = X_true.detach().cpu().numpy()
    X_gen_np = X_gen.detach().cpu().numpy()

    fig, axes = plt.subplots(
        nrows=n_plotted_dims,
        ncols=1,
        sharex=True,
    )

    if n_plotted_dims == 1:
        axes = [axes]

    legend_handles = None

    for i, (ax, dim) in enumerate(zip(axes, dims_to_plot)):
        (h_true,) = ax.plot(X_true_np[:T, dim], linewidth=1.5)
        (h_pred,) = ax.plot(X_gen_np[:T, dim], linewidth=1.5)

        ax.set_ylabel(f"Dim {dim}")

        # Capture handles once for a shared legend
        if legend_handles is None:
            legend_handles = [h_true, h_pred]

        if i < n_plotted_dims - 1:
            ax.tick_params(labelbottom=False)

    # Place title + legend, then compute the minimal top margin so they don't overlap
    # the first subplot (keeps whitespace tight and stable).
    title_artist = fig.suptitle(title, y=0.99)
    legend = fig.legend(
        legend_handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        bbox_transform=fig.transFigure,
        borderaxespad=0.0,
        frameon=False,
    )

    # Reduce unused whitespace (esp. on the right) while reserving just enough top margin
    # for the title/legend (always 2 legend entries).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    legend_bbox = inv.transform_bbox(legend.get_window_extent(renderer=renderer))
    title_bbox = inv.transform_bbox(title_artist.get_window_extent(renderer=renderer))
    header_bottom = min(legend_bbox.y0, title_bbox.y0)
    top = max(0.05, header_bottom - 0.015)
    fig.subplots_adjust(left=0.12, right=0.99, top=top, bottom=0.08, hspace=0.0)

    # Ensure all "Dim k" y-labels share the same x-position across subplots.
    if hasattr(fig, "align_ylabels"):
        fig.align_ylabels(axes)

    return fig

@tc.no_grad()
def plot_decoded_actions(
    decoded_actions: tc.Tensor, n_plotted_dim: int = 7, T: int = 300
):
    fig = plt.figure()
    plt.title("actions")
    plt.axis("off")

    n_plotted_dim = min([n_plotted_dim, decoded_actions.shape[1], decoded_actions.shape[1]])
    T = min([T, decoded_actions.shape[0], decoded_actions.shape[0]])

    for i in range(n_plotted_dim):
        fig.add_subplot(n_plotted_dim, 1, i + 1)
        plt.plot(decoded_actions[:T, i].detach().cpu().numpy())
        ax = plt.gca()
        lim = ax.get_ylim()
        ax.set_ylim(lim)
    plt.xlabel("time steps")

    return fig


@tc.no_grad()
def plot_categories(
    X_true: tc.Tensor,
    X_pred: tc.Tensor,
    is_reconstruction: bool = False,
):
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("True categories")
    plt.imshow(
        X_true.cpu().numpy().T,
        aspect="auto",
        cmap="Greys",
        interpolation="none",
    )
    plt.yticks(np.arange(X_true.shape[1]))

    plt.subplot(2, 1, 2)
    plt.title("Generated categories" if not is_reconstruction else "Reconstructed AE categories")
    plt.imshow(
        X_pred.cpu().numpy().T,
        aspect="auto",
        cmap="Greys" if not is_reconstruction else "Reds",
        interpolation="none",
        vmax=1.0,
        vmin=0.0,
    )
    plt.yticks(np.arange(X_pred.shape[1]))
    plt.xlabel("Time steps")

    plt.tight_layout()

    return fig


@tc.no_grad()
def plot_attractor(
    X_true: Optional[tc.Tensor],
    X_pred: tc.Tensor,
):
    """
    Plot the attractor of a reconstructed sequence (in 3D)
    """

    # plot
    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")

    if X_true is not None:
        ax.plot(
            X_true[:, 0].detach().cpu().numpy(),
            X_true[:, 1].detach().cpu().numpy(),
            X_true[:, 2].detach().cpu().numpy(),
            label="True",
        )

    ax.plot(
        X_pred[:, 0].detach().cpu().numpy(),
        X_pred[:, 1].detach().cpu().numpy(),
        X_pred[:, 2].detach().cpu().numpy(),
        label="Generated",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # ax.axis("off")

    # Adjust legend position closer to the plot
    ax.legend(loc="upper left", bbox_to_anchor=(0.3, 0.9))

    plt.tight_layout()

    return fig
