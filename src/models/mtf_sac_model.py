
import torch as tc
import torch.nn as nn
import os
import json
from typing import Optional, TYPE_CHECKING
from src import utils
from src.dataset import TimeseriesDataset
from src.models.mtf_model import MTFModel
from src.models.dsr_model.dsr_sac_model import DSR_SAC_Model

if TYPE_CHECKING:
    from src.parameters import SAC_Parameters


class MTF_SAC_Model(MTFModel):
    """MTF model adapted for SAC training."""

    def __init__(
            self,
            dataset: TimeseriesDataset,
            sac_args: Optional["SAC_Parameters"]= None,
            model_path: Optional[str] = None,
            current_epoch: Optional[int] = None,
            device: Optional[tc.device] = None,
            ):

        self.current_epoch = current_epoch

        # initialize model either from args (as during training) or model path (as during evalutaion)
        assert not (sac_args is None and model_path is None), (
            "To initialize a model, either give args or model path to load parameters from."
        )

        if model_path is not None:
            self.init_from_sac_model_path(
                dataset,
                model_path,
                epoch=current_epoch,
                map_location=device,
                args=sac_args,
            )

        elif sac_args is not None:
            self.sac_args = sac_args

            # initialize submodules and possibly load saved model, as given by args
            self.init_from_sac_args(dataset, device, epoch=current_epoch)

    def init_from_sac_model_path(self, dataset, model_path, epoch=None, args=None, map_location=None):
        """
        epoch: if None, start from las saved in model_path directory
        map_location: if model is loaded after saving from a GPU, this enables to load it anyway to a cpu
        """

        # load arguments from hyperparameter txt file
        self.sac_args: SAC_Parameters = json.load(open(f"{model_path}/config.json"))
        if args is not None:
            self.sac_args["fine_tuning_path"] = args.get("fine_tuning_path", None)
            self.sac_args["optuna_loss_term_scaling"] = args["optuna_loss_term_scaling"]

        # initialize submodules using loaded arguments
        self.init_sac_submodules(dataset, map_location)

        # resume model from given or else latest training epoch
        if epoch is None:
            self.current_epoch = utils.get_latest_epoch(model_path)
        else:
            self.current_epoch = epoch
        # temporary compatibility fix
        dict_ = tc.load(
                os.path.join(model_path, f"model_{self.current_epoch}.pt"),
                map_location=map_location,
            )
        
        dict_copy = dict(dict_)
        # backward compatibility
        if "encoder_model.encoder_params.subjects.0.L" in dict_:
            for i in range(self.num_measurements):
                dict_copy[f"encoder_model.encoder_params.subjects.{i}.L.0"] = dict_copy.pop(f"encoder_model.encoder_params.subjects.{i}.L")
                dict_copy[f"encoder_model.encoder_params.subjects.{i}.L.1"] = tc.zeros_like(dict_copy[f"encoder_model.encoder_params.subjects.{i}.L.0"][:, 0])   
        
        for key, param in dict_.items():
            if key.startswith("projection_matrix."):
                new_key = key.replace("projection_matrix.", "dsr_sac_model.projection_matrix.")
                dict_copy[new_key] = param
                del dict_copy[key]
        self.load_state_dict(dict_copy     
        )
        if map_location is not None:
            self.to(map_location)

        if args is not None and args["fine_tuning_path"] is not None:
            # all sac group level parameters are freezed during fine-tuning
            for param1, param2, param3, param4, param5 in zip(self.dsr_sac_model.q1_params.group.parameters(),
                                                    self.dsr_sac_model.q2_params.group.parameters(),
                                                    self.dsr_sac_model.policy_params.group.parameters(),
                                                    self.dsr_sac_model.q1_target_params.group.parameters(),
                                                    self.dsr_sac_model.q2_target_params.group.parameters()):
                param1.requires_grad = False
                param2.requires_grad = False
                param3.requires_grad = False
                param4.requires_grad = False
                param5.requires_grad = False

            # re-initialize subject vectors for fine-tuning
            nn.init.uniform_(self.dsr_sac_model.q1_params.subject_vectors, a=-1.0, b=1.0)
            nn.init.uniform_(self.dsr_sac_model.q2_params.subject_vectors, a=-1.0, b=1.0)
            nn.init.uniform_(self.dsr_sac_model.policy_params.subject_vectors, a=-1.0, b=1.0)
            nn.init.uniform_(self.dsr_sac_model.q1_target_params.subject_vectors, a=-1.0, b=1.0)
            nn.init.uniform_(self.dsr_sac_model.q2_target_params.subject_vectors, a=-1.0, b=1.0)
            self.sac_args["optuna_trial"] = args["optuna_trial"]

    def init_from_sac_args(self, dataset, device=None, epoch=None):
        # possibly load model from saved parameters
        model_path = self.sac_args["load_model_path"]
        if model_path is None:
            self.init_sac_submodules(dataset, device)

        else:
            if epoch is None:
                epoch = self.sac_args["resume_epoch"]
            self.init_from_sac_model_path(dataset, model_path, epoch, map_location=device)


    def init_sac_submodules(self, dataset, device):
        super().__init__(dataset, model_path=self.sac_args["world_model"]["path"], current_epoch=self.sac_args["world_model"]["epoch"], device=device)

        # all mtf model parameters are freezed
        for param in self.parameters():
            param.requires_grad = False

        # if subject-specific vectors are inherited from mtf model, set feature_dimension accordingly
        if self.sac_args["q_hierarchisation"].get("keep_subject_parameters", False):
            self.sac_args["q_hierarchisation"]["feature_dimension"] = self.args["dsr_hierarchisation_scheme"]["feature_dimension"]
        else:
            assert self.sac_args["q_hierarchisation"].get("feature_dimension", None) is not None, (
                "If subject-specific vectors are not inherited from mtf model, "
                "feature_dimension has to be specified in sac_args['q_hierarchisation']."
            )
        if self.sac_args["policy_hierarchisation"].get("keep_subject_parameters", False):
            self.sac_args["policy_hierarchisation"]["feature_dimension"] = self.args["dsr_hierarchisation_scheme"]["feature_dimension"]
        else:
            assert self.sac_args["policy_hierarchisation"].get("feature_dimension", None) is not None, (
                "If subject-specific vectors are not inherited from mtf model, "
                "feature_dimension has to be specified in sac_args['policy_hierarchisation']."
            )

        # add input and output dim of q function and policy ffns
        self.sac_args["policy_ffn_hidden_dims"].insert(0, self.latent_dim+int(self.dsr_model.hyperparameters["external_input_dim"] or 0))
        self.sac_args["policy_ffn_hidden_dims"].append(2 * self.dim_forcing)
        self.sac_args["q_ffn_hidden_dims"].insert(0, self.latent_dim + self.dim_forcing+int(self.dsr_model.hyperparameters["external_input_dim"] or 0))
        self.sac_args["q_ffn_hidden_dims"].append(1)

        #TODO: make this more elegant: problem is that the epoch is loaded as a string from the json
        for goal_decoders in self.sac_args["goal_decoders"]:
            for goal_decoder in goal_decoders:
                goal_decoder["goal_state"] = {float(k): v for k, v in goal_decoder["goal_state"].items()}


        self.dsr_sac_model = DSR_SAC_Model(
            self.sac_args,
            self.dsr_model,
            self.decoder_model.decoders,
            self.args["latent_step"],
            self.args["dsr_hierarchisation_scheme"],
            self.latent_dim,
            self.external_input_dims,
            different_dynamics=self.args["dsr_different_dynamics"],
            different_external_inputs=self.args["dsr_different_external_inputs"],
            num_timeseries_per_measurements=self.num_timeseries_per_measurement,
            mar_units=self.args["MARegularized_units"],
            continuous_subject_groups=self.args.get(
                "dsr_continuous_subject_groups", None
            ),
            dim_forcing=self.dim_forcing,
            cumulative_index_from_measurement_timeseries=dataset.cumulative_index_from_measurement_timeseries
        )
        if device is not None:
            self.to(device)
        
        print(
            f"SAC model parameters: {sum(p.numel() for p in self.dsr_sac_model.parameters() if p.requires_grad)}"
        )


    def reward_function(
        self,
        z: tc.Tensor,
        action: tc.Tensor,
        measurement_index: int,
        expected=False,
        epoch: Optional[int] = tc.inf,
    ) -> tc.Tensor:
        """Compute the reward for a given state and action."""

        # TODO: make reward a separate module
        reward = 0
        decoded = self.decoder_model(z, measurement_index, expected=expected)
        output_dims = self.decoder_model.decoder_output_dims[measurement_index]

        # reward for reaching goal states
        for decoder in self.sac_args["goal_decoders"][measurement_index]:
            decoder_name = decoder["name"]
            if decoder_name in self.decoder_model.decoders[measurement_index]:
                # modality index is the index of "decoder_name" in self.decoder_model.decoders[measurement_index]
                i = list(self.decoder_model.decoders[measurement_index].keys()).index(decoder_name)
                # curriculum learning
                goal_state_int = tc.argwhere(epoch >= tc.as_tensor(list(decoder["goal_state"].keys())))[-1].item()
                goal_state = tc.as_tensor(list(decoder["goal_state"].values())[goal_state_int], device=self.device)
                # where goal_state is None, ignore the corresponding dimension in the reward
                diff = decoded[:, :, sum(output_dims[:i]):sum(output_dims[:i + 1])][:, :, tc.isnan(goal_state) == False] - goal_state[tc.isnan(goal_state) == False]
                reward_weights = tc.as_tensor(decoder.get("reward_weights", [1.0] * diff.shape[-1]), device=self.device)
                reward -= ((diff*reward_weights) ** 2).sum(dim=-1)

        # sparse action penalty
        # TODO: mean, not sum
        reward -= self.sac_args.get("action_penalty", 0) * tc.nn.functional.l1_loss(action, tc.zeros_like(action), reduction='none').sum(dim=-1)
        return reward.unsqueeze(-1)

    def generate_free_latent_trajectory(
        self,
        data: tc.Tensor,
        cumulative_timeseries_index: int,
        measurement_index: int,
        S: Optional[tc.Tensor] = None,
        T: Optional[int] = None,
        t_start: int = 0,
        without_external_inputs: bool = False,
        with_entropy: bool = False,
        apply_actions: bool = True,
        expected: bool = False,
    ):
        """
        returns latent trajectory including initial condition!
        """
        if T is None:
            T = data.size(0)
        
        # get intitial latent state = encoded initial state
        _, entropy, encoded_trajectory = self.encoder_model(
            data, measurement_index, calculate_entropy=with_entropy
        )

        z0 = encoded_trajectory[..., t_start, :]

        if S is not None and self.dsr_model.hyperparameters["external_input_dim"] is not None:
            S = S[t_start:] 
            S = tc.nn.functional.pad(S, (0, 0, 0, max(0, T - S.shape[0]+1)))
            S = S[:T +1]
        else:
            S = [None] * (T + 1)

        # generate latent trajectory
        latent_trajectory, actions = self.dsr_sac_model.generate_free_trajectory(
            z0.unsqueeze(0) ,
            T,
            cumulative_timeseries_index,
            measurement_index,
            S.unsqueeze(1) if S[0] is not None else S,
            apply_actions=apply_actions,
            expected=expected
        )  # T x latent_dim

        latent_without_external = None
        if without_external_inputs:
            latent_without_external, actions_without_external = self.dsr_sac_model.generate_free_trajectory(
                z0.unsqueeze(0),
                T,
                cumulative_timeseries_index,
                measurement_index,
                [None] * (T + 1),
                apply_actions=apply_actions,
                expected=expected,
            )

        return (
            latent_trajectory,
            actions,
            encoded_trajectory,
            latent_without_external,
            actions_without_external if without_external_inputs else None,
            entropy,
        )

    def generate_free_trajectory(
        self,
        data: tc.Tensor,
        cumulative_timeseries_index: int,
        measurement_index: int,
        S=None,
        T=None,
        modalities="stacked",
        n_samples=1,
        expected=False,
        expected_decoder = False,
        t_start=0,
        without_external_inputs=False,
        apply_actions=True,
    ):
        """S: external input
        n_sampels: number of observed samples to generate
        Output: observed (stacked -> tensor or splitted -> generator), latent (freely predicted), and (latent) encoded trajectory
        """

        # generate latent trajectory
        latent_trajectory, actions, _, latent_without_external, actions_without_external, _ = (
            self.generate_free_latent_trajectory(
                data,
                cumulative_timeseries_index,
                measurement_index,
                S,
                T,
                t_start,
                without_external_inputs=without_external_inputs,
                apply_actions=apply_actions,
                expected=expected,
            )
        )

        # decode trajectory (possibly stacked on last dimension, else as generator to loop through)
        if n_samples != 1:
            # automatically generates n_samples (as batches) when decoder distribution.sample() is called
            latent_trajectory = latent_trajectory.expand(
                n_samples, -1, -1
            )  # n_samples x T x latent_dim
            latent_without_external = (
                latent_without_external.expand(n_samples, -1, -1)
                if latent_without_external is not None
                else None
            )

        observed_trajectory = self.decoder_model(
            latent_trajectory,
            measurement_index,
            modalities == "stacked",
            expected_decoder,
        )  # n_samples x T x N

        observed_without_external = self.decoder_model(
            latent_without_external,
            measurement_index,
            modalities == "stacked",
            expected_decoder,
        ) if without_external_inputs else None

        decoded_actions = self.decoder_model(
            actions,
            measurement_index,
            modalities == "stacked",
            expected_decoder,
        )  # n_samples x T x N

        decoded_actions_without_external = (
            self.decoder_model(
                actions_without_external,
                measurement_index,
                modalities == "stacked",
                expected_decoder,
            )
            if actions_without_external is not None
            else None
        )

        if modalities != "stacked":
            latent_trajectory = self.decoder_model.latent_split(
                latent_trajectory, measurement_index
            )

        return (
            observed_trajectory,
            latent_trajectory,
            actions,
            decoded_actions,
            observed_without_external,
            latent_without_external,
            actions_without_external,
            decoded_actions_without_external,
        )


    def q_step(self, state, next_state, action, cumulative_timeseries_indices, measurement_indices):
        """Perform a single step of q function optimization."""
        measurement_index = measurement_indices[0] # all indices in subbatch are the same measurement
         # compute q values for current state and action
        q1_pred, q2_pred = self.dsr_sac_model.compute_q_values(
            state, action, cumulative_timeseries_indices
        )
        # sample next action from policy
        policy_params = self.dsr_sac_model.sac_construct_params(
            cumulative_timeseries_indices, self.dsr_sac_model.policy_params, "policy")
        next_action, ll_action_next_state = self.dsr_sac_model.sample_action(
            next_state, policy_params, measurement_index
        )
        # compute q values for next state and the sampled action
        q1_next_state_pred, q2_next_state_pred = self.dsr_sac_model.compute_q_values(
            next_state, next_action, cumulative_timeseries_indices, target=True,
        )
        # Compute the minimum Q-value for the next state
        q_next_state_pred = tc.min(
            tc.stack([q1_next_state_pred, q2_next_state_pred], dim=0), dim=0).values

        return ll_action_next_state, q_next_state_pred, q1_pred, q2_pred, cumulative_timeseries_indices


    def policy_step(self, state, cumulative_timeseries_indices, measurement_indices):
        """Perform a single step of policy optimization."""
        measurement_index = measurement_indices[0] # all indices in subbatch are the same measurement
        # sample action from policy
        policy_params = self.dsr_sac_model.sac_construct_params(
            cumulative_timeseries_indices, self.dsr_sac_model.policy_params, "policy")
        sampled_action, ll_sampled_action = self.dsr_sac_model.sample_action(
            state, policy_params, measurement_index
        )
        # check that projection is correct
        # TODO:
        #decoded = self.decoder_model(sampled_action, measurement_index)
        #a = tc.mean(tc.abs(decoded[:, -4:-2]))
        #assert a < 1e-2
        q1_pred, q2_pred = self.dsr_sac_model.compute_q_values(
            state, sampled_action, cumulative_timeseries_indices
        )
        q_pred = tc.min(
            tc.stack([q1_pred, q2_pred], dim=0), dim=0).values

        return ll_sampled_action, q_pred, cumulative_timeseries_indices

    @tc.no_grad()
    def actor_forward(
        self,
        subbatches: list,
        T: int = None,
        epoch: Optional[int] = None,
    ):
        """Forward pass of assembled model
            TODO: possibly add forcing_dim != latent_dim
        """

        # pass time-flattened input through encoder and reshape afterwards
        # Z_encodeds = []
        # Z_encoded_means = []
        all_cumulative_timeseries_indices = []
        all_measurement_indices = []
        actions = []
        Z_predicteds = []
        rewards = []
        Ss = []
        
        if T is None:
            T = subbatches[0][0].shape[1]
        for data in subbatches:
            X, _, S, cumulative_timeseries_indices, measurement_indices = data
            measurement_index = measurement_indices[0]

            Z_encoded, _, Z_encoded_mean = self.encoder_model(
                X,
                measurement_indices[0],
            )

            # choose random state as initial state
            rand_idx = tc.randint(0, Z_encoded.size(0), (1,)).item()
            z0 = Z_encoded[rand_idx]
            
            # possibly broadcast external inputs
            if S is not None and self.dsr_model.hyperparameters["external_input_dim"] is not None:
                S_ = S[rand_idx:] 
                S_ = tc.nn.functional.pad(S_, (0, 0, 0, 0, 0, max(0, T - S_.shape[0]+1)))
                S_ = S_[:T +1]
                Ss.append(S_)
            else:
                S_ = [None] * (T + 1)
                S = None
            
            
            Z_predicted, action = self.dsr_sac_model.generate_free_trajectory(
                z0,
                T,
                cumulative_timeseries_indices,
                measurement_index,
                S=S_,
                expected=False,
                noise = self.sac_args.get("rollout_noise", None),
            )

            reward = self.reward_function(
                Z_predicted[:, 1:, :],  # only from t=1 onward, since reward is 0 for initial state
                action,
                measurement_index,
                epoch = epoch
            )

            Z_predicteds.append(Z_predicted)
            actions.append(action)
            rewards.append(reward)
            # Z_encodeds.append(Z_encoded)
            # Z_encoded_means.append(Z_encoded_mean)
            all_cumulative_timeseries_indices.append(
                cumulative_timeseries_indices.repeat_interleave(T)
            )
            all_measurement_indices.append(
                measurement_indices.repeat_interleave(T)
            )

        cumulative_timeseries_indices = tc.cat(
            all_cumulative_timeseries_indices, dim=0
        )  # B x T
        measurement_indices = tc.cat(all_measurement_indices, dim=0)  # B x T
        Z_predicted = tc.cat(Z_predicteds, dim=0)
        actions = tc.cat(actions, dim=0)
        rewards = tc.cat(rewards, dim=0)
        
        S = tc.cat(Ss, dim=1) if S is not None else None
        state = tc.cat([Z_predicted[:, :-1], S.permute(1, 0, 2)[:, :-1]], dim=-1) if S is not None else Z_predicted[:, :-1]
        new_state = tc.cat([Z_predicted[:, 1:], S.permute(1, 0, 2)[:, 1:]], dim=-1) if S is not None else Z_predicted[:, 1:]
        
        return (
            state,
            new_state,
            actions,
            rewards,
            cumulative_timeseries_indices,
            measurement_indices,
            Z_encoded_mean,
        )


    def q_forward(
        self,
        data,
        ):
        """Forward pass of Q functions and policy."""

        ll_action_next_states = []
        q_next_state_preds = []
        rewards = []
        q1_preds = []
        q2_preds = []
        all_cumulative_timeseries_indices = []

        for subbatch in data:
            state, next_state, action, reward, cumulative_timeseries_indices, measurement_indices = subbatch
            ll_action_next_state, q_next_state_pred, q1_pred, q2_pred, _ = self.q_step(state, next_state, action, cumulative_timeseries_indices, measurement_indices)

            ll_action_next_states.append(ll_action_next_state)
            q_next_state_preds.append(q_next_state_pred)
            rewards.append(reward)
            q1_preds.append(q1_pred)
            q2_preds.append(q2_pred)
            all_cumulative_timeseries_indices.append(cumulative_timeseries_indices)

        ll_action_next_states = tc.cat(ll_action_next_states, dim=0)
        q_next_state_preds = tc.cat(q_next_state_preds, dim=0)
        rewards = tc.cat(rewards, dim=0)
        q1_preds = tc.cat(q1_preds, dim=0)
        q2_preds = tc.cat(q2_preds, dim=0)
        cumulative_timeseries_indices = tc.cat(all_cumulative_timeseries_indices, dim=0)


        return ll_action_next_states, q_next_state_preds, rewards, q1_preds, q2_preds, cumulative_timeseries_indices


    def policy_forward(
        self,
        data,
        ):
        """Forward pass of Q functions and policy."""

        ll_sampled_actions = []
        q_preds = []
        all_cumulative_timeseries_indices = []

        for subbatch in data:
            state, next_state, action, reward, cumulative_timeseries_indices, measurement_indices = subbatch

            ll_sampled_action, q_pred, _ = self.policy_step(state, cumulative_timeseries_indices, measurement_indices)

            ll_sampled_actions.append(ll_sampled_action)
            q_preds.append(q_pred)
            all_cumulative_timeseries_indices.append(cumulative_timeseries_indices)

        ll_sampled_actions = tc.cat(ll_sampled_actions, dim=0)
        q_preds = tc.cat(q_preds, dim=0)
        cumulative_timeseries_indices = tc.cat(all_cumulative_timeseries_indices, dim=0)


        return ll_sampled_actions, q_preds, cumulative_timeseries_indices
