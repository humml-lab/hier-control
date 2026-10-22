import os
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch as tc
import torch.nn as nn

from src import utils
from src.dataset import TimeseriesDataset

from .decoder.decoder_models import Decoder
from .dsr_model.dsr_model import DSRModel
from .encoder.encoder import Encoder

if TYPE_CHECKING:
    from src.parameters import Parameters


class MTFModel(nn.Module):
    def __init__(
        self,
        dataset: TimeseriesDataset,
        args: Optional["Parameters"] = None,
        model_path: Optional[str] = None,
        current_epoch: Optional[int] = None,
        device: Optional[tc.device] = None,
    ):
        """explicite given arguments, i.e. model_path and epoch, overrule implicit given ones through args"""

        super().__init__()
        self.encoder_model = None
        self.dsr_model = None
        self.decoder_model = None
        self.dataset = dataset
        self.output_dims = dataset.modality_dimensions
        self.input_dims: list[int] = [
            sum(measurement_dimensions) for measurement_dimensions in self.output_dims
        ]
        self.external_input_dims = dataset.external_dimensions
        self.num_measurements = len(self.output_dims)
        self.num_timeseries_per_measurement: list[int] = dataset.timeseries_per_measurement
        self.current_epoch = current_epoch
        self.device = device

        # initialize model either from args (as during training) or model path (as during evalutaion)
        assert not (args is None and model_path is None), (
            "To initialize a model, either give args or model path to load parameters from."
        )

        if model_path is not None:
            self.init_from_model_path(
                model_path,
                epoch=current_epoch,
                map_location=self.device,
                args = args,
            )

        elif args is not None:
            self.args = args  # cast arguments into a dictionary via "vars" function

            # initialize submodules and possibly load saved model, as given by args
            self.init_from_args(epoch=current_epoch)

    def init_from_args(self, epoch=None):
        # possibly load model from saved parameters
        model_path = self.args["load_model_path"]
        if model_path is None:
            self.init_submodules()
        else:
            if epoch is None:
                epoch = self.args["resume_epoch"]
            self.init_from_model_path(model_path, epoch)

    def init_submodules(self):
        # set latent and forcing dimensions
        self.latent_dim = self.args["latent_dim"]
        self.dim_forcing = self.args.get("dim_forcing", None)
        if self.dim_forcing is None:
            self.dim_forcing = self.latent_dim

        # initialize models
        self.encoder_model = Encoder(
            self.args["encoder"],
            self.input_dims,
            self.dim_forcing,
            self.latent_dim,
            hierarchisation_scheme=self.args["encoder_hierarchisation"],
            num_timeseries_per_measurements=self.num_timeseries_per_measurement,
        )

        self.dsr_model = DSRModel(
            self.args["latent_step"],
            self.args["dsr_hierarchisation_scheme"],
            self.latent_dim,
            self.external_input_dims,
            different_dynamics=self.args["dsr_different_dynamics"],
            different_external_inputs=self.args["dsr_different_external_inputs"],
            num_timeseries_per_measurements=self.num_timeseries_per_measurement,
            mar_units=self.args["MARegularized_units"],
            continuous_subject_groups=self.args.get("dsr_continuous_subject_groups", None),
            dim_forcing=self.dim_forcing,
            cumulative_index_from_measurement_timeseries=self.dataset.cumulative_index_from_measurement_timeseries,
        )

        self.decoder_model = Decoder(
            self.args["decoder"],
            self.dim_forcing,
            self.output_dims,
        )

        # print all the numer of parameters in the encoder, dsr, and decoder models
        print(
            f"Encoder model parameters: {sum(p.numel() for p in self.encoder_model.parameters() if p.requires_grad)}"
        )
        print(
            f"DSR model parameters: {sum(p.numel() for p in self.dsr_model.parameters() if p.requires_grad)}"
        )
        print(
            f"Decoder model parameters: {sum(p.numel() for p in self.decoder_model.parameters() if p.requires_grad)}"
        )

    def init_from_model_path(self, model_path, epoch=None, args=None, map_location=None):
        """
        epoch: if None, start from las saved in model_path directory
        map_location: if model is loaded after saving from a GPU, this enables to load it anyway to a cpu
        """

        # load arguments from hyperparameter txt file
        args_path = os.path.join(model_path, "hypers.pkl")
        self.args: Parameters = np.load(args_path, allow_pickle=True)

        # resume model from given or else latest training epoch
        if epoch is None:
            self.current_epoch = utils.get_latest_epoch(model_path)

        # restore model parameters by loading state dict
        # backward compatibility
        state_dict = tc.load(os.path.join(model_path, f"model_{self.current_epoch}.pt"), map_location=map_location)
        # encoder linear now also contains bias
        if "encoder_model.encoder_params.subjects.0.L" in state_dict:
            for i in range(self.num_measurements):
                state_dict[f"encoder_model.encoder_params.subjects.{i}.L.0"] = state_dict.pop(f"encoder_model.encoder_params.subjects.{i}.L")
                state_dict[f"encoder_model.encoder_params.subjects.{i}.L.1"] = tc.zeros_like(state_dict[f"encoder_model.encoder_params.subjects.{i}.L.0"][:, 0])                
        
        
        if "dsr_model.latent_step_params.group.P_AW" in state_dict:
            AW = state_dict.pop("dsr_model.latent_step_params.group.P_AW")
            A = tc.diagonal(AW, dim1=-2, dim2=-1)
            W = AW - tc.diag_embed(A)
            state_dict["dsr_model.latent_step_params.group.P_A"] = A
            state_dict["dsr_model.latent_step_params.group.P_W"] = W
        # only concerns the mc_maze runs
        if state_dict.get("encoder_model.encoder_params.subjects.0.L.0", None).shape[0]==10:
            state_dict["encoder_model.encoder_params.subjects.0.L.0"] = state_dict["encoder_model.encoder_params.subjects.0.L.0"].transpose(-1, -2)
            state_dict[f"encoder_model.encoder_params.subjects.{i}.L.1"] = tc.zeros_like(state_dict[f"encoder_model.encoder_params.subjects.{i}.L.0"][:, 0])
        if state_dict.get("encoder_model.encoder_params.subjects.0.L.0", None).shape[0]==6:
            state_dict["encoder_model.encoder_params.subjects.0.L.0"] = state_dict["encoder_model.encoder_params.subjects.0.L.0"].transpose(-1, -2)
            state_dict[f"encoder_model.encoder_params.subjects.{i}.L.1"] = tc.zeros_like(state_dict[f"encoder_model.encoder_params.subjects.{i}.L.0"][:, 0])
        # drop any unexpected encoder keys that arise from fine-tuning with different number of measurements
        for key in list(state_dict.keys()):
            if "encoder_model.encoder_params.subjects." in key:
                subject_index = int(key.split("encoder_model.encoder_params.subjects.")[1].split(".")[0])
                if subject_index >= self.num_measurements:
                    state_dict.pop(key)

        if args is not None and args.get("fine_tuning_path", None) is not None:
            self.args["decoder"] = args["decoder"]
            state_dict["dsr_model.latent_step_params.subject_vectors"] = state_dict["dsr_model.latent_step_params.subject_vectors"][:self.num_measurements]

        # initialize submodules using loaded arguments
        self.init_submodules()

        self.load_state_dict(
            state_dict
        )

        if args is not None and args.get("fine_tuning_path", None) is not None:
            # all dsr group level parameters are freezed during fine-tuning
            for param in self.dsr_model.latent_step_params.group.parameters():
                param.requires_grad = False
            if self.encoder_model.encoder_params.group is not None:
                for param in self.encoder_model.encoder_params.group.parameters():
                    param.requires_grad = False
            for param in self.encoder_model.encoder_params.subjects.parameters():
                param.requires_grad = False
            # re-initialize subject vectors for fine-tuning
            #nn.init.uniform_(self.dsr_model.latent_step_params.subject_vectors, a=-1.0, b=1.0)
            #for param in self.encoder_model.encoder_params.subjects.parameters():
            #    tc.nn.init.uniform_(param, a=-1.0, b=1.0)
            self.args["optuna_trial"] = args["optuna_trial"]

    def get_parameter_groups(self):
        """Get parameter groups for optimizer"""

        shared_parameters = []
        shared_parameters += list(self.encoder_model.parameters())
        shared_parameters += list(self.decoder_model.parameters())

        dsr_group_parameters, dsr_subject_parameters = self.dsr_model.get_parameter_groups()

        shared_parameters += dsr_group_parameters

        return shared_parameters, dsr_subject_parameters

    def forward(
        self,
        subbatches: list,
        alpha: float = 1.0,
        n: int = 1,
        forcing_dim: Optional[int] = None,
    ):
        """Forward pass of assembled model.

        Args:
            X (T x B x N): encoder input with N = input_dim
            S (T x B x M): external input with M = external_input_dim
            alpha (float): teacher-forcing parameter
            n (int): force every n steps
            forcing_dim (int): dimensions to teacher-force
        """
        if forcing_dim is None:
            forcing_dim = self.latent_dim

        Z_encodeds = []
        Z_encoded_means = []
        entropies = []
        external_inputs = []
        all_cumulative_timeseries_indices = []
        for data in subbatches:
            X, _, S, cumulative_timeseries_indices, measurement_indices = data
            Z_encoded, entropy, Z_encoded_mean = self.encoder_model(
                X,
                # measurement indices are same across a subbatch
                measurement_indices[0],
                calculate_entropy=True,
            )
            Z_encodeds.append(Z_encoded)
            entropies.append(entropy)
            Z_encoded_means.append(Z_encoded_mean)
            external_inputs.append(S)
            all_cumulative_timeseries_indices.append(cumulative_timeseries_indices)

        # Concatenate along the batch dimension (dim=1) since tensors are (T, B, :)
        Z_encoded = tc.cat(Z_encodeds, dim=1)
        Z_encoded_mean = tc.cat(Z_encoded_means, dim=1)
        entropy = tc.mean(tc.stack(entropies))
        S = tc.cat(external_inputs, dim=1)
        # Concatenate cumulative indices along batch (dim=0) into a single 1D tensor
        cumulative_timeseries_indices = tc.cat(all_cumulative_timeseries_indices, dim=0)

        Z_predicted = self.dsr_model(
            Z_encoded,
            cumulative_timeseries_indices,
            S,
            alpha,
            n,
            forcing_dim,
        )

        return (
            Z_encoded,
            Z_predicted,
            entropy,
            Z_encoded_mean,
        )

    @tc.no_grad()
    def generate_free_trajectory(
        self,
        observations: tc.Tensor,
        cumulative_timeseries_index: Optional[int | tc.Tensor] = None,
        measurement_index: Optional[int] = None,
        timeseries_index: Optional[int] = None,
        external_inputs: Optional[tc.Tensor] = None,
        T: Optional[int] = None,
        stack_modalities: bool = True,
        num_samples: int = 1,
        use_expected_output: bool | list[bool] = False,
        return_decoded_teacher_forcing: bool = False,
        return_only_latent: bool = False,
        return_entropy: bool = False,
        use_latent_mean: bool = False,
    ) -> tuple[
        tc.Tensor,
        tc.Tensor,
        Optional[tc.Tensor | list[tc.Tensor]],
        Optional[tc.Tensor | list[tc.Tensor]],
        Optional[tc.Tensor],
    ]:
        """Generates free trajectory from given observations, i.e. without teacher forcing.

        Args:
            observations (T x N or T x B x N): Observed trajectory to encode.
                Only the initial latent state is used for generation.
            cumulative_timeseries_index: Cumulative index of the timeseries to generate.\
                Either this or both measurement_index and timeseries_index need to be given.\
                Only this can be used for batched generation.
            measurement_index: Measurement index of the timeseries to generate
            timeseries_index: Timeseries index of the timeseries to generate
            external_inputs (T x N_ext or T x B x N_ext): External inputs to use for generation
            T: Length of trajectory to generate. If None, uses length of observations or\
                external_inputs, whichever is shorter. Even if it is not None, the length is capped by external_inputs.
            stack_modalities: Whether to stack modalities on the last dimension when decoding or
                return them as a list.
            num_samples: Number of decoded samples to generate. If > 1, the output will have shape (num_samples, T, (B), N)
            use_expected_output: Whether to return the expected value of the decoder distribution instead of a sample. Should\
                not be used with num_samples > 1.
            return_decoded_teacher_forcing: Whether to return the decoded trajectory from the encoded trajectory.
            return_only_latent: Whether to return only the latent trajectories without decoding. This skips decoding\
                entirely.
            return_entropy: Whether to return the entropy of the encoder. Useful to compute the loss.
            use_latent_mean: Whether to use the mean of the encoder distribution as initial latent state instead of\
                a sample from it. If list, one value should be given per modality.

        Returns a tuple with the following entries:
            - predicted latent trajectory
            - encoded latent trajectory
            - predicted observations
            - decoded trajectory from encoded latent trajectory
            - entropy of the encoder
        The entries can be None depending on the arguments given.

        Note that
            - the model only receives the initial latent state from the encoder, not the entire encoded trajectory.
            - if `return_only_latent` is False, batched generation is only possible if all timeseries are from the\
                same measurement.
            - the (timeseries_index and measurement_index) or cumulative_timeseries_index actually identify the model\
                to use for generation and not the data given. Any data can be given, but the model will use the\
                dynamics of the identified model. For generating a trajectory using custom parameters\
                (eg. interpolated ones), see the `DSRModel` class's `generate_free_trajectory` method.
        """
        assert not (
            cumulative_timeseries_index is None
            and (measurement_index is None or timeseries_index is None)
        ), (
            "To generate a trajectory, either give the cumulative_timeseries_index or both measurement_index and timeseries_index."
        )

        cumulative_timeseries_index = (
            cumulative_timeseries_index
            if cumulative_timeseries_index is not None
            else self.dataset.cumulative_index_from_measurement_timeseries(
                measurement_index, timeseries_index
            )
        )

        if isinstance(cumulative_timeseries_index, int):
            measurement_index, timeseries_index = (
                self.dataset.measurement_timeseries_from_cumulative_index(
                    cumulative_timeseries_index
                )
            )
        else:
            measurement_indices = tc.tensor(
                [
                    self.dataset.measurement_timeseries_from_cumulative_index(idx)[0]
                    for idx in cumulative_timeseries_index.cpu().numpy()
                ]
            )
            assert (measurement_indices == measurement_indices[0]).all(), (
                "Batched generation is only possible if all timeseries are from the same measurement."
            )
            measurement_index = measurement_indices[0].item()

        sequence_length: int = min(
            external_inputs.shape[0] if external_inputs is not None else np.inf,
            T if T is not None else np.inf,
        )

        if sequence_length == np.inf:
            sequence_length = observations.shape[0]

        entropy: Optional[tc.Tensor]
        encoded_sample, entropy, encoded_mean = self.encoder_model(
            observations[:sequence_length],
            measurement_index,
            calculate_entropy=return_entropy,
        )

        encoded_trajectory: tc.Tensor = encoded_mean if use_latent_mean else encoded_sample
        latent_trajectory = self.dsr_model.generate_free_trajectory(
            encoded_trajectory[0],
            sequence_length,
            external_inputs[:sequence_length] if external_inputs is not None else None,
            cumulative_timeseries_index=cumulative_timeseries_index,
        )

        if return_only_latent:
            return (
                latent_trajectory,
                encoded_trajectory,
                None,
                None,
                entropy if return_entropy else None,
            )

        observed_trajectory = self.generate_decoded_trajectory(
            latent_trajectory,
            measurement_index,
            num_samples,
            stack_modalities,
            use_expected_output,
        )

        observed_tf_trajectory = None
        if return_decoded_teacher_forcing:
            observed_tf_trajectory = self.generate_decoded_trajectory(
                encoded_trajectory,
                measurement_index,
                1,
                stack_modalities,
                use_expected_output,
            )
        return (
            latent_trajectory,
            encoded_trajectory,
            observed_trajectory,
            observed_tf_trajectory,
            entropy if return_entropy else None,
        )

    @tc.no_grad()
    def generate_decoded_trajectory(
        self,
        latent_trajectory: tc.Tensor,
        measurement_index: int,
        num_samples: int = 1,
        stack_modalities: bool = True,
        use_expected_output: bool | list[bool] = False,
    ) -> tc.Tensor | list[tc.Tensor]:
        """
        Generate a decoded trajectory from the latent trajectory.

        Args:
            latent_trajectory (T x latent_dim or T x B x latent_dim): Latent trajectory to decode
            measurement_index: Measurement index of the timeseries to generate
            stack: Whether to stack modalities on the last dimension when decoding or return them as a list.
            use_expected_output: Whether to return the expected value of the decoder distribution instead
                of a sample. If list, one value should be given per modality. Should not be used with batch size > 1.
        """

        added_batch_dim = False
        if latent_trajectory.dim() == 2:
            latent_trajectory = latent_trajectory.unsqueeze(1)
            added_batch_dim = True

        if num_samples != 1:
            # automatically generates num_samples (as batches) when decoder distribution.sample() is called
            latent_trajectory = latent_trajectory.expand(
                num_samples, -1, -1, -1
            )  # num_samples x T x B x latent_dim

        observed_trajectory = self.decoder_model(
            latent_trajectory,
            measurement_index,
            stack_modalities,
            use_expected_output,
        )

        if added_batch_dim:
            if stack_modalities:
                observed_trajectory = observed_trajectory.squeeze(1)
            else:
                observed_trajectory = [modality.squeeze(1) for modality in observed_trajectory]

        return observed_trajectory

    def decoder_negative_log_likelihood(self, Y: tc.Tensor, Z: tc.Tensor, measurement_index: int):
        return self.decoder_model.log_likelihood(Y, Z, measurement_index)

    def MAR_loss(self, cumulative_timeseries_indices: tc.Tensor | int):
        return self.dsr_model.MAR_loss(cumulative_timeseries_indices)

    def autoregressive_convergence_loss(self, cumulative_timeseries_indices: tc.Tensor | int):
        return self.dsr_model.autoregressive_convergence_loss(cumulative_timeseries_indices)

    def subject_continuity_loss(self, cumulative_timeseries_indices: tc.Tensor | int):
        return self.dsr_model.subject_continuity_loss(cumulative_timeseries_indices)

    def latent_jacobians(self, Z: tc.Tensor, cumulative_trial_index: int):
        return self.dsr_model.jacobian(Z, cumulative_trial_index)  # (T, B, M, M)

    def to(self, device: tc.device):
        self.device = device
        return super().to(device)
