import torch as tc
import torch.nn.functional as F

from src.models.dsr_model.models.alrnn.base import ALRNNStep
from src.utils import l2_reg

from .linear_projection import LinearProjectionCoupledALRNN
from .linear_projection_per_region import LinearProjectionPerRegionCoupledALRNN
from .none import NoHierarchisationCoupledALRNN


class CoupledALRNNStep(ALRNNStep):
    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters):
        assert hyperparameters["latent_dim"] % 2 == 0, (
            "The latent dimension must be even for the Coupled ALRNNs step."
        )

        match hierarchisation_scheme["scheme"]:
            case "linear-projection-per-region":
                return LinearProjectionPerRegionCoupledALRNN.initialize_group_level_params(
                    hierarchisation_scheme, hyperparameters
                )
            case _:
                return super().initialize_group_level_params(
                    hierarchisation_scheme, hyperparameters
                )

    @classmethod
    def initialize_subject_level_params(
        cls, hierarchisation_scheme, hyperparameters, subject_hyperparameters
    ):
        assert hyperparameters["latent_dim"] % 2 == 0, (
            "The latent dimension must be even for the CoupledPLRNNs step."
        )

        match hierarchisation_scheme["scheme"]:
            case "linear-projection-per-region":
                return LinearProjectionPerRegionCoupledALRNN.initialize_subject_level_params(
                    hierarchisation_scheme, hyperparameters, subject_hyperparameters
                )
            case "none":
                return NoHierarchisationCoupledALRNN.initialize_subject_level_params(
                    hierarchisation_scheme, hyperparameters, subject_hyperparameters
                )
            case _:
                return super().initialize_subject_level_params(
                    hierarchisation_scheme, hyperparameters, subject_hyperparameters
                )

    @classmethod
    def construct_params(
        cls, hierarchisation_scheme, hyperparameters, group_params, subject_params
    ):
        match hierarchisation_scheme["scheme"]:
            case "linear-projection-per-region":
                return LinearProjectionPerRegionCoupledALRNN.construct_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    group_params,
                    subject_params,
                )
            case "linear-projection":
                return LinearProjectionCoupledALRNN.construct_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    group_params,
                    subject_params,
                )
            case "none":
                return NoHierarchisationCoupledALRNN.construct_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    group_params,
                    subject_params,
                )
            case _:
                raise ValueError(
                    f"Unknown hierarchisation scheme: {hierarchisation_scheme['scheme']}."
                )

    @classmethod
    def forward(cls, z, s, params, hyperparameters, **kwargs):
        z_a, z_b = tc.chunk(z, chunks=2, dim=1)

        relu_z_a = tc.cat(
            [
                z_a[:, : -hyperparameters["relu_units"]],
                F.relu(z_a[:, -hyperparameters["relu_units"] :]),
            ],
            dim=1,
        )

        relu_z_b = tc.cat(
            [
                z_b[:, : -hyperparameters["relu_units"]],
                F.relu(z_b[:, -hyperparameters["relu_units"] :]),
            ],
            dim=1,
        )

        z_a_next = (
            params["A_a"] * z_a
            + tc.einsum("bij,bj->bi", params["W_a2a"], relu_z_a)
            + tc.einsum("bij,bj->bi", params["W_b2a"], relu_z_b)
            + params["h_a"]
        )

        z_b_next = (
            params["A_b"] * z_b
            + tc.einsum("bij,bj->bi", params["W_b2b"], relu_z_b)
            + tc.einsum("bij,bj->bi", params["W_a2b"], relu_z_a)
            + params["h_b"]
        )

        if params["C_a"] is not None:
            z_a_next += tc.einsum("bij,bj->bi", params["C_a"], s)

        if params["C_b"] is not None:
            z_a_next += tc.einsum("bij,bj->bi", params["C_b"], s)

        z_next = tc.cat([z_a_next, z_b_next], dim=1)

        return z_next

    @classmethod
    def MARLoss(cls, params, regularized_units, hyperparameters):
        A_a, A_b = params["A_a"], params["A_b"]
        W_a2a, W_a2b, W_b2b, W_b2a = (
            params["W_a2a"],
            params["W_a2b"],
            params["W_b2b"],
            params["W_b2a"],
        )
        h_a, h_b = params["h_a"], params["h_b"]
        batch_sz, M = A_a.size()

        # Determine which units to regularize
        # -> start at linear units if possible, else simply regularize the last
        # regularized_units units
        forced_units = hyperparameters["dim_forcing"]
        start_idx = min(forced_units, M - regularized_units)
        end_idx = start_idx + regularized_units

        # Apply MAR to the specified units
        loss_A_a = l2_reg(A_a[:, start_idx:end_idx], target=1.0)
        loss_A_b = l2_reg(A_b[:, start_idx:end_idx], target=1.0)
        loss_W_a2a = l2_reg(W_a2a[:, start_idx:end_idx, :])
        loss_W_a2b = l2_reg(W_a2b[:, start_idx:end_idx, :])
        loss_W_b2b = l2_reg(W_b2b[:, start_idx:end_idx, :])
        loss_W_b2a = l2_reg(W_b2a[:, start_idx:end_idx, :])
        loss_h_a = l2_reg(h_a[:, start_idx:end_idx])
        loss_h_b = l2_reg(h_b[:, start_idx:end_idx])

        return (
            loss_A_a
            + loss_A_b
            + loss_W_a2a
            + loss_W_a2b
            + loss_W_b2b
            + loss_W_b2a
            + loss_h_a
            + loss_h_b
        ) / batch_sz

    @classmethod
    def jacobian(cls, z, params, hyperparameters):
        """
        Compute the Jacobian of the model at state z.

        Args:
            z: Latent states of shape (B, latent_dim)
            params: Dictionary containing model parameters (A, W)
            hyperparameters: Dictionary containing model hyperparameters
        Returns:
            J: Jacobian matrices of shape (B, latent_dim, latent_dim)
        """
        # Create activation derivative mask (1 for linear units, step function for ReLU units)
        z_a, z_b = tc.chunk(z, chunks=2, dim=1)
        D_a, D_b = tc.ones_like(z_a), tc.ones_like(z_b)
        if hyperparameters["relu_units"] > 0:
            # for region a
            D_a[:, -hyperparameters["relu_units"] :] = (
                z_a[:, -hyperparameters["relu_units"] :] > 0
            ).float()
            # for region b
            D_b[:, -hyperparameters["relu_units"] :] = (
                z_b[:, -hyperparameters["relu_units"] :] > 0
            ).float()
        D = tc.cat([D_a, D_b], dim=1)

        # Construct A and W out of region-specific A and W
        A = tc.cat([params["A_a"], params["A_b"]], dim=1)  # B x latent_dim

        W_2a = tc.cat([params["W_a2a"], params["W_b2a"]], dim=2)
        W_2b = tc.cat([params["W_a2b"], params["W_b2b"]], dim=2)
        W = tc.cat([W_2a, W_2b], dim=1)  # B x latent_dim x latent_dim

        # Combine diagonal A term and W term with activation derivatives
        return tc.diag_embed(A) + tc.einsum("ij,bi->bij", W.T, D)
