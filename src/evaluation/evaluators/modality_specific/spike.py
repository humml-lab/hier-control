import matplotlib.pyplot as plt
import numpy as np
import torch as tc

from src.evaluation.metrics.ace import autocorrelation_nmse
from src.evaluation.metrics.crosscorr import cross_corr_nmse
from src.evaluation.metrics.spike_stats import (
    coefficient_of_variation_nmse,
    mean_spike_rate_nmse,
    zero_count_ratio_nmse,
)
from src.utils import truncate_to_common_length

from .base import BaseEvaluator, ModalitySpecificEvaluatorParams


class SpikeEvaluatorParams(ModalitySpecificEvaluatorParams):
    units_to_plot: list[int]
    sequence_length: int
    use_raster_plot: bool

    def default() -> "SpikeEvaluatorParams":
        return {
            "name": "SpikeEvaluator",
            "units_to_plot": [0, 1, 2, 3, 4],
            "use_raster_plot": False,
        }


class SpikeEvaluator(BaseEvaluator[SpikeEvaluatorParams]):
    def __init__(
        self,
        params: SpikeEvaluatorParams,
    ):
        params = {
            **SpikeEvaluatorParams.default(),
            **params,
        }
        super().__init__(params)

    @tc.no_grad()
    def get_tensorboard_figures(
        self,
        args,
    ):
        X_true = args["X_true"]
        X_generated = args["X_generated"]
        X_AE_reconstructed = args["X_AE_reconstructed"]
        """get tensorboard figure for one trial"""

        if self.params["use_raster_plot"]:
            fig1 = plt.figure(figsize=(15, 5))
            gs = plt.GridSpec(2, 3, height_ratios=[20, 1])
            max_true = X_true.max()
            max_generated = X_generated.max()
            maX_AE_reconstructed = X_AE_reconstructed.max()
            max_val = max(max_true, max_generated, maX_AE_reconstructed)
            ax1 = fig1.add_subplot(gs[0, 0])
            ax1.imshow(
                X_true.detach().cpu().numpy().T,
                aspect="auto",
                cmap="binary",
                interpolation="none",
                vmax=max_val,
                vmin=0,
            )
            ax1.set_title("True spikes")
            ax2 = fig1.add_subplot(gs[0, 1])
            ax2.imshow(
                X_AE_reconstructed.detach().cpu().numpy().T,
                aspect="auto",
                cmap="binary",
                interpolation="none",
                vmax=max_val,
                vmin=0,
            )
            ax2.set_title("AE reconstructed spikes")
            ax3 = fig1.add_subplot(gs[0, 2])
            im3 = ax3.imshow(
                X_generated.detach().cpu().numpy().T,
                aspect="auto",
                cmap="binary",
                interpolation="none",
                vmax=max_val,
                vmin=0,
            )
            ax3.set_title("Generated spikes")
            cbar_ax = fig1.add_subplot(gs[1, :])
            fig1.colorbar(im3, cax=cbar_ax, orientation="horizontal")
            plt.tight_layout()

        else:
            # plot observations
            n_units_to_plot = min(len(self.params["units_to_plot"]), X_true.shape[1])
            fig1 = plt.figure(figsize=(10, 1 * n_units_to_plot))
            plt.title(
                f"(expected or sampled) simulated neurons (n) during trial {args['timeseries_index'] + 1}"
            )
            plt.axis("off")

            for i, unit_idx in enumerate(self.params["units_to_plot"][:n_units_to_plot]):
                ax = fig1.add_subplot(n_units_to_plot, 1, i + 1)
                plt.plot(
                    X_true[:, unit_idx].detach().cpu().numpy(),
                    color="blue",
                    marker="x",
                    markersize=2,
                    alpha=0.8,
                    label="ground truth",
                )
                plt.plot(
                    X_AE_reconstructed[:, unit_idx].detach().cpu().numpy(),
                    color="green",
                    marker=".",
                    markersize=2,
                    alpha=0.5,
                    label="decoded tf signal",
                )
                plt.plot(
                    X_generated[:, unit_idx].detach().cpu().numpy(),
                    color="orange",
                    marker=".",
                    markersize=2,
                    alpha=0.5,
                    label="generated",
                )
                plt.ylabel(f"n={unit_idx + 1}")
                ax.set_xticks([])
            ax.set_xticks(
                np.linspace(
                    0, max([X_true.shape[0], X_AE_reconstructed.shape[0], X_generated.shape[0]]), 5
                )
            )
            plt.xlabel("time steps")
            plt.legend(loc="upper right")

        return {
            "neurons": fig1,
        }

    @tc.no_grad()
    def compute_metrics(
        self,
        args,
    ):
        X_generated, X_true = truncate_to_common_length(args["X_generated"], args["X_true"])

        # Permute from (T, B, :) to (B, T, :) for metric functions
        # TODO: Adjust downstream code to handle (T, B, :)
        if X_generated.dim() == 3:
            X_generated = X_generated.permute(1, 0, 2)
        if X_true.dim() == 3:
            X_true = X_true.permute(1, 0, 2)

        return {
            "cross_corr_nmse": cross_corr_nmse(
                X_generated.cpu(),
                X_true.cpu(),
                average_batch=True,
            ),
            "autocorrelation_nmse": autocorrelation_nmse(
                X_generated.cpu(),
                X_true.cpu(),
                average_batch=True,
            ),
            "mean_spike_rate_nmse": mean_spike_rate_nmse(
                X_generated.cpu(),
                X_true.cpu(),
                average_batch=True,
            ),
            "zero_count_ratio_nmse": zero_count_ratio_nmse(
                X_generated.cpu(),
                X_true.cpu(),
                average_batch=True,
            ),
            "coefficient_of_variation_nmse": coefficient_of_variation_nmse(
                X_generated.cpu(),
                X_true.cpu(),
                average_batch=True,
            ),
        }
