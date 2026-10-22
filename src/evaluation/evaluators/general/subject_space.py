from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch as tc
from matplotlib.axes import Axes
from sklearn.decomposition import PCA

from .base import BaseEvaluator, GeneralEvaluatorParams


class SubjectSpaceEvaluatorParams(GeneralEvaluatorParams):
    plot_as_2d_projections: bool
    projection_mode: list[Literal["pca", "none"]]
    max_dims_to_plot: Optional[int]

    def default() -> "SubjectSpaceEvaluatorParams":
        return {
            "name": "SubjectSpaceEvaluator",
            "plot_as_2d_projections": True,
            "projection_mode": ["none"],
            "max_dims_to_plot": None,
        }


class SubjectSpaceEvaluator(BaseEvaluator[SubjectSpaceEvaluatorParams]):
    def __init__(self, params: SubjectSpaceEvaluatorParams):
        params = {
            **SubjectSpaceEvaluatorParams.default(),
            **params,
        }
        super().__init__(params)

    @tc.no_grad()
    def compute_metrics_and_figures(
        self,
        args,
    ):
        model = args["model"]
        subject_vectors: np.ndarray = (
            model.dsr_model.latent_step_params["subject_vectors"].detach().cpu().numpy()
        )
        writer = args["summary_writer"]
        epoch = args["epoch"]
        assert writer is None or epoch is not None, (
            "If SummaryWriter is provided, epoch must be provided as well."
        )
        figures = {}
        max_dims = min(
            self.params["max_dims_to_plot"] or subject_vectors.shape[1], subject_vectors.shape[1]
        )
        modes = self.params["projection_mode"]

        vectors_to_plot = {"none": subject_vectors}
        if "pca" in modes:
            pca = PCA()
            vectors_to_plot["pca"] = pca.fit_transform(subject_vectors)
            if writer is not None:
                for dim, variance in enumerate(pca.explained_variance_ratio_):
                    writer.add_scalar(
                        f"Subject_Space/Explained_Variance_Dim{dim + 1}",
                        variance,
                        global_step=epoch,
                        display_name=f"Explained Variance Dim {dim + 1} (PCA)",
                    )

        if (
            self.params["plot_as_2d_projections"]
            and subject_vectors.shape[1] >= 2
            and max_dims <= 5
        ):
            n_rows_per_max_dim = {2: 1, 3: 2, 4: 2, 5: 3}
            n_cols_per_max_dim = {2: 1, 3: 2, 4: 3, 5: 4}

            for mode in modes:
                fig, axes = plt.subplots(
                    nrows=n_rows_per_max_dim[max_dims],
                    ncols=n_cols_per_max_dim[max_dims],
                    figsize=(4 * n_cols_per_max_dim[max_dims], 4 * n_rows_per_max_dim[max_dims]),
                )
                linear_index = 0
                vectors = vectors_to_plot[mode]
                for i in range(vectors.shape[1]):
                    for j in range(i + 1, vectors.shape[1]):
                        row, col = divmod(
                            linear_index,
                            n_cols_per_max_dim[max_dims],
                        )
                        if (
                            row >= n_rows_per_max_dim[max_dims]
                            or col >= n_cols_per_max_dim[max_dims]
                        ):
                            continue
                        ax: Axes = (
                            axes[row, col] if n_rows_per_max_dim[max_dims] > 1 else axes[col]
                        )

                        ax.scatter(
                            vectors[:, i],
                            vectors[:, j],
                        )
                        ax.set_xlabel(f"{'PCA' if mode == 'pca' else 'Dim'} {i + 1}")
                        ax.set_ylabel(f"{'PCA' if mode == 'pca' else 'Dim'} {j + 1}")

                        linear_index += 1
                plt.tight_layout()
                if writer is not None:
                    writer.add_figure(
                        tag=f"Subject_Space/Projections_{mode}",
                        figure=fig,
                        global_step=epoch,
                    )
                else:
                    figures[f"Projections_{mode}"] = fig
        else:
            max_dims = min(max_dims, 12)  # limit to first 12 dimensions
            n_rows_per_max_dim = {
                1: 1,
                2: 1,
                3: 1,
                4: 2,
                5: 2,
                6: 2,
                7: 3,
                8: 3,
                9: 3,
                10: 4,
                11: 4,
                12: 4,
            }
            n_cols_per_max_dim = {
                1: 1,
                2: 2,
                3: 3,
                4: 2,
                5: 3,
                6: 3,
                7: 3,
                8: 3,
                9: 3,
                10: 3,
                11: 3,
                12: 3,
            }

            for mode in modes:
                fig, axes = plt.subplots(
                    nrows=n_rows_per_max_dim[max_dims],
                    ncols=n_cols_per_max_dim[max_dims],
                    figsize=(4 * n_cols_per_max_dim[max_dims], 4 * n_rows_per_max_dim[max_dims]),
                )
                linear_index = 0
                vectors = vectors_to_plot[mode]
                for dim in range(vectors.shape[1]):
                    row, col = divmod(
                        linear_index,
                        n_cols_per_max_dim[max_dims],
                    )
                    if row >= n_rows_per_max_dim[max_dims] or col >= n_cols_per_max_dim[max_dims]:
                        continue
                    ax: Axes = axes[row, col] if n_rows_per_max_dim[max_dims] > 1 else axes[col]
                    ax.plot(
                        vectors[:, dim],
                        marker="o",
                        linestyle="None",
                    )
                    ax.set_xlabel("Subject Index")
                    ax.set_ylabel(f"{'PCA' if mode == 'pca' else 'Dim'} {dim + 1}")
                    linear_index += 1
                plt.tight_layout()
                if writer is not None:
                    writer.add_figure(
                        tag=f"Subject_Space/Dimensions_{mode}",
                        figure=fig,
                        global_step=epoch,
                    )
                else:
                    figures[f"Dimensions_{mode}"] = fig

        return {
            "metrics": {},
            "figures": figures,
        }
