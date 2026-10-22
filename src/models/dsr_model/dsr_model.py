from typing import Callable, Literal, Optional

import torch as tc
import torch.nn as nn

from src.models.dsr_model.utils.latent_step.common_initialization import (
    CommonInitializationUtils,
)
from src.utils import index_of_nested, index_to_batched_long_tensor, not_none

from .models.alrnn.base import ALRNNHyperparameters, ALRNNOption, ALRNNStep
from .models.base import DSRHierarchisationScheme, DSRHyperparameters
from .models.clipped_shallow_plrnn.base import ClippedShallowPLRNNStep
from .models.coupled_alrnn.base import CoupledALRNNStep
from .models.coupled_clipped_shallow_plrnn.base import CoupledClippedShallowPLRNNStep
from .models.coupled_plrnn.base import CoupledPLRNNStep
from .models.coupled_shallow_plrnn.base import CoupledShallowPLRNNStep
from .models.plrnn.base import PLRNNHyperparameters, PLRNNOption, PLRNNStep
from .models.shallow_plrnn.base import (
    ShallowPLRNNHyperparameters,
    ShallowPLRNNOption,
    ShallowPLRNNStep,
)

DSRHyperparameterOption = (
    ShallowPLRNNHyperparameters | ALRNNHyperparameters | PLRNNHyperparameters | DSRHyperparameters
)

DEFAULT_DSR_HYPERPARAMETERS: DSRHyperparameters = {
    "learnable_C": False,
    "dim_forcing": None,
}


LatentStepOption = ALRNNOption | PLRNNOption | ShallowPLRNNOption

DifferentDynamicsOption = list[list[int]] | Literal["per-timeseries", "per-measurement"]
DifferentExternalInputsOption = (
    list[list[int]] | Literal["hierarchised", "constant", "per-subject"]
)


class DSRModel(nn.Module):
    def __init__(
        self,
        latent_step: LatentStepOption,
        hierarchisation_scheme: DSRHierarchisationScheme,
        latent_dim: int,
        external_input_dims: Optional[list[int]],
        cumulative_index_from_measurement_timeseries: Callable[[int, int], int],
        different_dynamics: DifferentDynamicsOption = "per-measurement",
        different_external_inputs: DifferentExternalInputsOption = "per-subject",
        num_timeseries_per_measurements: Optional[list[int]] = None,
        mar_units: int = 0,
        continuous_subject_groups: Optional[list[list[int]]] = None,
        dim_forcing: Optional[int] = None,
    ):
        """
        Wrapper for the DSR model.
        Parameters:
            latent_step: The latent step model to use.
            hierarchisation_scheme: The hierarchisation scheme to use.
            latent_dim: The latent dimension of the model.
            external_input_dims: The external input dimensions of the model. One per measurement.
            different_dynamics: Sets which parts of data get assigned a subject vector in the hierarchisation

                If list of list of integers, sets one subject vector per list of integers. Each integer corresponds to a measurement.\\
                If list of list of list of integers, sets one subject vector per list of lists. The first element of the list is the measurement index, the second element is the timeseries index (not cumulative!).\\
                If "per-timeseries", sets one subject vector per timeseries.\\
                If "per-measurement", sets one subject vector per measurement.
            different_external_inputs: Sets how external inputs' `C` matrix is assigned.

                If list of list of integers, one list gets assigned one trainable C matrix. Each integer corresponds to a measurement.\\
                If "hierarchised", the C matrix is hierarchised along with the other DSR parameters.\\
                If "constant", the C matrix is constant for all measurement.\\
                If "per-subject", each subject gets its own C matrix (which is not hierarchised).\\
            num_timeseries_per_measurements: The number of timeseries per measurement.
            mar_units: The number of units to regularize in the MAR loss.
            continuous_subject_groups: The groups of subjects whose subject vectors should be regularized to be continuous.\\
                If None, no groups are regularized.\\
                If list of list of integers, each list corresponds to a group of subjects. Each integer corresponds to a subject.
        """
        super(DSRModel, self).__init__()

        assert (
            hierarchisation_scheme["scheme"] == "none"
            or external_input_dims is None
            or (len(num_timeseries_per_measurements) == len(external_input_dims))
        ), (
            "Both the `external_input_dims`' length and the `num_timeseries_per_measurements`'s length has to correspond to the number of measurements of the dataset."
        )

        # TODO: assertions
        self.cumulative_index_from_measurement_timeseries = (
            cumulative_index_from_measurement_timeseries
        )
        self.continuous_subject_groups = continuous_subject_groups
        self.subject_index_from_timeseries_index: Optional[tc.Tensor]
        self.num_subjects: Optional[int]
        if hierarchisation_scheme["scheme"] == "none":
            # No hierarchisation, which is implemented as a single "subject"
            self.num_subjects = None
            self.subject_index_from_timeseries_index = None
        else:
            if different_dynamics == "per-measurement" or isinstance(different_dynamics, list):
                cumulative_num_timeseries_per_measurements: tc.Tensor = tc.cumsum(
                    tc.tensor(num_timeseries_per_measurements),
                    dim=0,
                )
                measurement_index_from_timeseries_index = [
                    int(tc.where(cumulative_num_timeseries_per_measurements > i)[0][0])
                    for i in range(sum(num_timeseries_per_measurements))
                ]
                if different_dynamics == "per-measurement":
                    # Each measurement gets its own subject vector
                    self.num_subjects = len(num_timeseries_per_measurements)
                    self.subject_index_from_timeseries_index = tc.tensor(
                        measurement_index_from_timeseries_index,
                        dtype=tc.long,
                    )
                else:
                    # Each group gets its own subject vector
                    self.num_subjects = len(different_dynamics)
                    if isinstance(different_dynamics[0][0], int):
                        subject_index_from_measurement_index = [
                            index_of_nested(different_dynamics, i)
                            for i in range(len(num_timeseries_per_measurements))
                        ]
                        self.subject_index_from_timeseries_index = tc.tensor(
                            [
                                subject_index_from_measurement_index[
                                    measurement_index_from_timeseries_index[i]
                                ]
                                for i in range(sum(num_timeseries_per_measurements))
                            ],
                            dtype=tc.long,
                        )
                    elif isinstance(different_dynamics[0][0], (list, tuple)):
                        self.subject_index_from_timeseries_index = [None] * sum(
                            num_timeseries_per_measurements
                        )
                        for subject_index, group in enumerate(different_dynamics):
                            for measurement_index, timeseries_index in group:
                                self.subject_index_from_timeseries_index[
                                    (
                                        cumulative_num_timeseries_per_measurements[
                                            measurement_index - 1
                                        ]
                                        if measurement_index > 0
                                        else 0
                                    )
                                    + timeseries_index
                                ] = subject_index
                        assert None not in self.subject_index_from_timeseries_index, (
                            "Not all timeseries indices are assigned a subject index."
                        )
                        self.subject_index_from_timeseries_index = tc.tensor(
                            self.subject_index_from_timeseries_index,
                            dtype=tc.long,
                        )
                    else:
                        raise ValueError(
                            f"Unknown `different_dynamics` option: {different_dynamics}."
                        )
            elif different_dynamics == "per-timeseries":
                # Each timeseries gets its own subject vector
                self.num_subjects = sum(num_timeseries_per_measurements)
                self.subject_index_from_timeseries_index = tc.arange(
                    sum(num_timeseries_per_measurements),
                    dtype=tc.long,
                )
            else:
                raise ValueError(f"Unknown `different_dynamics` option: {different_dynamics}.")

        self.external_matrix_index_from_timeseries_index: Optional[tc.Tensor]
        self.num_external_matrices: Optional[int]
        self.different_external_inputs = different_external_inputs
        if hierarchisation_scheme["scheme"] == "none":
            self.external_matrix_index_from_timeseries_index = (
                None  # C matrix is initialized as part of the subject level parameters
            )
            self.num_external_matrices = None
        else:
            if different_external_inputs == "per-subject":
                # Copying the indexing here to avoid edge case if checks later on
                self.external_matrix_index_from_timeseries_index = (
                    self.subject_index_from_timeseries_index
                )
                self.num_external_matrices = self.num_subjects
            elif different_external_inputs == "hierarchised":
                # This is not a real case as the C matrix will be hierarchised along with the other DSR parameters
                # The latent_steps will overwrite the None with the hierarchised C matrix
                self.external_matrix_index_from_timeseries_index = None
                self.num_external_matrices = None
            elif different_external_inputs == "constant":
                self.num_external_matrices = 1
                self.external_matrix_index_from_timeseries_index = tc.zeros(
                    sum(num_timeseries_per_measurements),
                    dtype=tc.long,
                )  # One C matrix for all timeseries
            elif isinstance(different_external_inputs, list):
                self.num_external_matrices = len(different_external_inputs)
                measurement_index_from_timeseries_index = [
                    int(tc.where(cumulative_num_timeseries_per_measurements > i)[0][0])
                    for i in range(sum(num_timeseries_per_measurements))
                ]
                external_matrix_index_from_measurement_index = [
                    index_of_nested(different_external_inputs, i)
                    for i in range(len(num_timeseries_per_measurements))
                ]
                self.external_matrix_index_from_timeseries_index = tc.tensor(
                    [
                        external_matrix_index_from_measurement_index[
                            measurement_index_from_timeseries_index[i]
                        ]
                        for i in range(sum(num_timeseries_per_measurements))
                    ],
                    dtype=tc.long,
                )

        self.mar_units = mar_units
        self.hierarchisation_scheme = hierarchisation_scheme
        self.num_timeseries_per_measurements = num_timeseries_per_measurements
        self.latent_step = None

        max_external_input_dim = (  # This should be unnecessary as the dataset should take care of it
            max(external_input_dims) if external_input_dims is not None else 0
        )
        max_external_input_dim = max_external_input_dim if max_external_input_dim > 0 else None
        external_input_hyperparameters = {
            "external_input_initialization": (
                "hierarchised"
                if different_external_inputs == "hierarchised"
                else (
                    "independent-per-subject"
                    if different_external_inputs == "per-subject"
                    else "none"
                )
            ),
            "external_input_dim": max_external_input_dim,
        }

        self.hyperparameters: DSRHyperparameterOption = {
            **DEFAULT_DSR_HYPERPARAMETERS,
            "latent_dim": latent_dim,
            **latent_step.get("hyperparameters", {}),
            **external_input_hyperparameters,
            "dim_forcing": dim_forcing,
        }

        match latent_step["name"]:
            case "shPLRNN":
                self.latent_step = ShallowPLRNNStep
            case "coupled_shPLRNNs":
                self.latent_step = CoupledShallowPLRNNStep
            case "clipped_shPLRNN":
                self.latent_step = ClippedShallowPLRNNStep
            case "coupled_clipped_shPLRNNs":
                self.latent_step = CoupledClippedShallowPLRNNStep
            case "PLRNN":
                self.latent_step = PLRNNStep
            case "coupled_PLRNNs":
                self.latent_step = CoupledPLRNNStep
            case "ALRNN":
                self.latent_step = ALRNNStep
            case "coupled_ALRNNs":
                self.latent_step = CoupledALRNNStep
            case _:
                raise ValueError(f"Unknown DSR Model: {latent_step['name']}.")

        if self.hierarchisation_scheme["scheme"] == "none":
            # In case of no hierarchisation a ParameterDict is returned, which can be stored directly,
            # and the hierarchisation scheme "none" is always handled as a special case.
            self.latent_step_params = nn.ParameterDict(
                self.latent_step.initialize_subject_level_params(
                    self.hierarchisation_scheme,
                    self.hyperparameters,
                    external_input_hyperparameters,
                ),
            )
            if self.latent_step_params["C"] is not None:
                self.latent_step_params["C"].requires_grad = self.hyperparameters["learnable_C"]
        else:
            self.latent_step_params = nn.ParameterDict(
                {
                    "group": self.latent_step.initialize_group_level_params(
                        self.hierarchisation_scheme, self.hyperparameters
                    )
                }
            )
            subject_vector, per_subject_C = self.latent_step.initialize_subject_level_params(
                self.hierarchisation_scheme,
                self.hyperparameters,
                {"external_input_dim": max_external_input_dim},
            )
            self.latent_step_params.update(
                {
                    "subject_vectors": tc.repeat_interleave(
                        subject_vector, self.num_subjects, dim=0
                    ),
                    # This happens if C is hiearchised or independent per subject
                    "external_matrices": (
                        None
                        if self.hyperparameters["external_input_dim"] is None
                        else (
                            None
                            if self.hyperparameters["external_input_initialization"]
                            == "hierarchised"
                            else (
                                tc.repeat_interleave(
                                    per_subject_C.unsqueeze(0), self.num_external_matrices, dim=0
                                )
                                if self.hyperparameters["external_input_initialization"]
                                == "independent-per-subject"
                                else (
                                    tc.repeat_interleave(
                                        CommonInitializationUtils.init_external_matrix(
                                            max_external_input_dim,
                                            self.hyperparameters["latent_dim"],
                                        ).unsqueeze(0),
                                        self.num_external_matrices,
                                        dim=0,
                                    )
                                )
                            )
                        )
                    ),
                }
            )
            if self.latent_step_params["external_matrices"] is not None:
                self.latent_step_params["external_matrices"].requires_grad = self.hyperparameters[
                    "learnable_C"
                ]

    def get_parameter_groups(self):
        if self.hierarchisation_scheme["scheme"] == "none":
            return list(self.parameters()), None
        else:
            return (
                list(self.latent_step_params["group"].parameters())
                + (
                    [self.latent_step_params["external_matrices"]]
                    if self.latent_step_params["external_matrices"] is not None
                    else []
                ),
                [self.latent_step_params["subject_vectors"]],
            )

    def construct_params(
        self,
        measurement_index: Optional[int] = None,
        timeseries_index: Optional[int] = None,
        cumulative_timeseries_index: Optional[tc.Tensor | int] = None,
        subject_index: Optional[tc.Tensor | int] = None,
        external_index: Optional[tc.Tensor | int] = None,
        subject_params: Optional[tc.Tensor] = None,
        external_params: Optional[tc.Tensor] = None,
    ):
        """
        Constructs the parameters of latent models in a batched manner.

        Args:
            measurement_index (int): The measurement index of the model to be constructed.
            timeseries_index (int): The timeseries index of the model to be constructed.
            cumulative_timeseries_index (int or B): The cumulative time series indices of the model(s) to be
                constructed.
            subject_index (int or B): The subject indices of the model(s) to be constructed.
            external_index (int or B): The external indices of the model(s) to be constructed.
            subject_params (B x N_feat): The batched subject vectors. Useful for interpolated parameters.
            external_params (B x N_latent x N_ext): The batched external matrices. Useful for interpolated
                parameters.

        Note that `cumulative_timeseries_index`, `subject_index` and `external_index` are mutually\
            exclusive to select the subject vectors to be used for the reconstruction. The same is true for\
            `cumulative_timeseries_index`, `external_index` and `external_params` in the case of the external\
            matrices.
        """

        if self.hierarchisation_scheme["scheme"] == "none":
            # Always route through construct_params for consistent behavior across schemes
            return self.latent_step.construct_params(
                self.hierarchisation_scheme,
                self.hyperparameters,
                None,
                self.latent_step_params,
            )
        assert (
            (measurement_index is not None and timeseries_index is not None)
            or cumulative_timeseries_index is not None
            or subject_index is not None
            or subject_params is not None
        ), (
            "Either `measurement_index` and `timeseries_index` or `cumulative_timeseries_index` or "
            "`subject_index` or `subject_params` has to be provided."
        )

        uses_external_inputs = self.hyperparameters["external_input_dim"] is not None

        assert (
            not uses_external_inputs
            or self.different_external_inputs == "hierarchised"
            or self.different_external_inputs == "constant"
            or (self.different_external_inputs == "per-subject" and subject_index is not None)
            or cumulative_timeseries_index is not None
            or external_index is not None
            or external_params is not None
        ), (
            "The model uses custom external matrix initialization, so either cumulative_timeseries_index"
            "or external_index or external_params have to be provided. Or the model uses per-subject external"
            "matrices, in which case providing subject_index is also enough."
        )

        subject_vectors = None
        external_matrices = None

        if measurement_index is not None and timeseries_index is not None:
            cumulative_timeseries_index = self.cumulative_index_from_measurement_timeseries(
                measurement_index,
                timeseries_index,
            )

        if uses_external_inputs and self.different_external_inputs == "constant":
            external_matrices = self.latent_step_params["external_matrices"]

        if cumulative_timeseries_index is not None:
            cumulative_timeseries_index = index_to_batched_long_tensor(cumulative_timeseries_index)

            subject_vectors = self.latent_step_params["subject_vectors"][
                self.subject_index_from_timeseries_index[cumulative_timeseries_index]
            ]

            if uses_external_inputs and self.different_external_inputs != "hierarchised":
                external_matrices = self.latent_step_params["external_matrices"][
                    self.external_matrix_index_from_timeseries_index[cumulative_timeseries_index]
                ]
        elif subject_index is not None:
            subject_index = index_to_batched_long_tensor(subject_index)
            subject_vectors = self.latent_step_params["subject_vectors"][subject_index]

            if uses_external_inputs and self.different_external_inputs == "per-subject":
                external_matrices = self.latent_step_params["external_matrices"][subject_index]
        elif subject_params is not None:
            subject_vectors = subject_params

        if uses_external_inputs and self.different_external_inputs != "hierarchised":
            if external_index is not None:
                external_index = index_to_batched_long_tensor(external_index)
                external_matrices = self.latent_step_params["external_matrices"][external_index]
            elif external_params is not None:
                external_matrices = external_params

        return self.latent_step.construct_params(
            self.hierarchisation_scheme,
            self.hyperparameters,
            self.latent_step_params["group"],
            {
                "subject_vectors": subject_vectors,
                "external_matrices": external_matrices,
            },
        )

    def forward(
        self,
        Z_encoded: tc.Tensor,
        cumulative_timeseries_indices: tc.Tensor,
        S: Optional[tc.Tensor] = None,
        alpha=1.0,
        n=1,
        forcing_dim: Optional[int] = None,
        z0: Optional[tc.Tensor] = None,
    ):
        """
        Forward pass with generalized (alpha) or sparse (interval n) teacher forcing.

        If z0 = None, uses first encoded latent state of Z_encoded as initial value.

        Output: predictions (!!!)
                Z_predicted = (T x B x N) tensor of next steps, such that ideally Z_predicted[t,:,:] == Z_encoded[t+1,:,:]
        """
        forcing_dim = not_none(forcing_dim, self.hyperparameters["latent_dim"])
        T, B, encoded_latent_dim = Z_encoded.size()

        assert encoded_latent_dim == self.hyperparameters["latent_dim"], (
            "The encoded latent dimension must be equal to the DSR-model's latent dimension"
        )

        # External inputs already in (T, B, M); index per time step
        if S is not None:
            S_ = S
        else:
            S_ = [None] * T

        # set initial latent state
        if z0 is None:
            z = Z_encoded[0]
        else:
            z = z0

        params = self.construct_params(cumulative_timeseries_index=cumulative_timeseries_indices)

        # Preallocate output (T, B, latent_dim)
        Z_predicted = tc.empty_like(Z_encoded)
        for t in range(T):
            if (t % n == 0) and (t > 0):
                z = self.teacher_force(z, Z_encoded[t], alpha, forcing_dim)

            z = self.latent_step.forward(z, S_[t], params, self.hyperparameters)
            Z_predicted[t] = z

        return Z_predicted

    @tc.no_grad()
    def generate_free_trajectory(
        self,
        z0: tc.Tensor,
        T: Optional[int],
        external_inputs: Optional[tc.Tensor] = None,
        measurement_index: Optional[int] = None,
        timeseries_index: Optional[int] = None,
        cumulative_timeseries_index: Optional[int | tc.Tensor] = None,
        subject_index: Optional[int | tc.Tensor] = None,
        external_index: Optional[int | tc.Tensor] = None,
        subject_params: Optional[tc.Tensor] = None,
        external_params: Optional[tc.Tensor] = None,
    ) -> tc.Tensor:
        """
        Generates a free trajectory from a given (batched) initial condition.
        The function can be used via measurement and timeseries index for a single trajectory or
        via cumulative timeseries indices, subject indices or subject parameters for batched generation.

        Note that the returned trajectory contains the initial condition in the first position.

        Args:
            z0 (N or B x N): The initial condition of the trajectory
            T: The number of steps to be generated. If not given the length of the `external_inputs` are
                used instead. `T` will be maximized to the length of the external inputs.
            external_inputs (T x N_ext or T x B x N_ext): The external inputs to use for generation.
            measurement_index: The measurement index to be used for generation.
            timeseries_index: The timeseries index to be used for generation.
            cumulative_timeseires_index (int or B): The cumulative timeseries index or indices to be used.
            subject_index (int or B): The subject index or indices to be used.
            external_index (int or B): The external matrix's index to be used.
            subject_params (B x N_features): The subject vectors of the subjects to be used
                for generation. These may be interpolated vectors as well.
            external_params (B x N_latent x N_ext): The external matrices to be used for generation.

        Note that either the `measurement_index` and `timeseries_index` or the `cumulative_timeseires_index`
            or the `subject_index` or the `subject_param` argument has to be passed.
        """

        assert (
            (measurement_index is not None and timeseries_index is not None)
            or cumulative_timeseries_index is not None
            or subject_index is not None
            or subject_params is not None
        ), (
            "Either `measurement_index` and `timeseries_index` or `cumulative_timeseries_index` or "
            "`subject_index` or `subject_params` has to be provided."
        )

        added_batch_dim = False
        if z0.ndim == 1:
            added_batch_dim = True
            z0 = z0.unsqueeze(0)  # add batch dimension
            external_inputs = external_inputs.unsqueeze(1) if external_inputs is not None else None

        params = self.construct_params(
            measurement_index,
            timeseries_index,
            cumulative_timeseries_index,
            subject_index,
            external_index,
            subject_params,
            external_params,
        )

        uses_external_inputs = external_inputs is not None and params["C"] is not None
        assert T is not None or uses_external_inputs, (
            "Either `T` or `external_inputs` has to be provided."
        )

        if uses_external_inputs:
            T = min(T or tc.inf, len(external_inputs) + 1)  # +1 because we include z0
            assert len(external_inputs) >= T - 1

        # Preallocate (T, B, latent_dim) including initial condition
        B, M = z0.size()
        Z_generated = tc.empty((T, B, M), device=z0.device, dtype=z0.dtype)
        Z_generated[0] = z0
        for t in range(T - 1):
            Z_generated[t + 1] = self.latent_step.forward(
                Z_generated[t],
                external_inputs[t] if uses_external_inputs else None,
                params,
                self.hyperparameters,
            )

        if added_batch_dim:
            Z_generated.squeeze_(1)

        return Z_generated

    @staticmethod
    def teacher_force(
        z: tc.Tensor,
        z_true: tc.Tensor,
        alpha: float,
        forcing_dim: int,
    ):
        """
        Generalized teacher forcing for the latent state vector z.
        """
        z[:, :forcing_dim] = alpha * z_true[:, :forcing_dim] + (1.0 - alpha) * z[:, :forcing_dim]
        return z

    def subject_continuity_loss(
        self,
        used_cumulative_timeseries_indices: tc.Tensor | int,
    ):
        if self.continuous_subject_groups is None:
            return tc.tensor(0.0)

        regularization = tc.tensor(0.0, device=self.latent_step_params["subject_vectors"].device)
        used_subjects = self.subject_index_from_timeseries_index[
            used_cumulative_timeseries_indices
        ].unique()

        used_mask = tc.zeros(
            self.num_subjects,
            dtype=tc.bool,
            device=used_cumulative_timeseries_indices.device,
        )
        used_mask[used_subjects] = True

        for group in self.continuous_subject_groups:
            x: tc.Tensor = self.latent_step_params["subject_vectors"][group]
            # Standardize x to make sure that the regularization is not dependent on the scale of the subject vectors
            x = x / (x.std(dim=0, keepdim=True) + 1e-8)
            mask = used_mask[group].to(x.device)
            # L2
            forward = (x[1:] - x[:-1].detach()).pow(2).sum(dim=1)
            backward = (x[:-1] - x[1:].detach()).pow(2).sum(dim=1)

            regularization_per_subject = tc.zeros(x.size(0), device=x.device)
            regularization_per_subject[1:] += forward
            regularization_per_subject[:-1] += backward

            regularization_per_subject = regularization_per_subject * mask.float()
            regularization += regularization_per_subject.sum()

        return regularization / len(self.continuous_subject_groups)

    def MAR_loss(
        self,
        cumulative_timeseries_indices: tc.Tensor | int,
    ):
        params = self.construct_params(cumulative_timeseries_index=cumulative_timeseries_indices)
        return self.latent_step.MARLoss(
            params,
            self.mar_units,
            self.hyperparameters,
        )

    def autoregressive_convergence_loss(
        self,
        cumulative_timeseries_indices: tc.Tensor | int,
    ):
        params = self.construct_params(cumulative_timeseries_index=cumulative_timeseries_indices)
        return self.latent_step.autoregressive_convergence_loss(
            params,
            self.hyperparameters,
        )

    @tc.no_grad()
    def jacobian(self, Z: tc.Tensor, cumulative_timeseries_index: int | tc.Tensor):
        """
        Computes the Jacobian of the model's dynamics.

        Args:
            Z (tc.Tensor): Either a batch/trajectory of states (B or T, M) or a batch of sequences (T, B, M). In the latter case, states are internally flattened to (T * B, M).
            cumulative_timeseries_index (int | tc.Tensor): The cumulative timeseries index/indices.

        Returns:
            The Jacobian of the model's dynamics either (B or T, M, M) or (T, B, M, M).
        """
        # local cumulative_timeseries_index
        c_idx = cumulative_timeseries_index

        # check whether cumulative_timeseries_index contains only one unique value
        single_model = False
        if isinstance(c_idx, int):
            single_model = True
        elif isinstance(c_idx, tc.Tensor):
            uniques = c_idx.unique()
            if len(uniques) == 1:
                c_idx = uniques
                single_model = True

        if Z.ndim == 2:
            B, M = Z.shape
            params = self.construct_params(cumulative_timeseries_index=c_idx)
            jacobians = self.latent_step.jacobian(Z, params, self.hyperparameters)
        elif Z.ndim == 3:
            T, B, M = Z.shape

            if single_model:
                params = self.construct_params(cumulative_timeseries_index=c_idx)
            else:
                params = self.construct_params(
                    cumulative_timeseries_index=c_idx.repeat_interleave(T)
                )

            # compute jacobians
            jacobians = self.latent_step.jacobian(
                Z.reshape(-1, M), params, self.hyperparameters
            ).view(T, B, M, M)

        return jacobians
