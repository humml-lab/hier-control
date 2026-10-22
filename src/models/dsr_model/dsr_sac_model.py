from copy import deepcopy
from typing import Optional
import torch as tc
import torch.nn as nn
from src.models.dsr_model.models.base import HierarchisedDSRModel
from src.models.dsr_model.dsr_model import DSRModel
from src.models.dsr_model.FFN.base import FFN
from src.models.decoder.models.gaussian.gaussian import Gaussian
from src.models.decoder.models.identity import Identity

class DSR_SAC_Model(DSRModel):
    """DSR model adapted for SAC training."""

    def __init__(
            self,
            sac_args: dict,
            dsr_model: DSRModel,
            decoders: nn.ModuleList,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_step_params = dsr_model.latent_step_params

        match sac_args["ffn_activation"]:
            case "ReLU":
                activation_function = nn.ReLU()
            case "Tanh":
                activation_function = nn.Tanh()
            case _:
                raise ValueError(f"Unknown activation function: {sac_args['ffn_activation']}.")

        # extend the hyperparameters of the dsr_model with the sac specific ones
        self.hyperparameters = {
            **self.hyperparameters,
            "policy_ffn_hidden_dims": sac_args["policy_ffn_hidden_dims"],
            "q_ffn_hidden_dims": sac_args["q_ffn_hidden_dims"],
            "ffn_activation": activation_function,
            "q_hierarchisation_scheme": sac_args["q_hierarchisation"],
            "policy_hierarchisation_scheme": sac_args["policy_hierarchisation"],
            "init_log_temperature": sac_args["init_log_temperature"],
            "adjustable_temperature": sac_args["adjustable_temperature"],
            "fine_tuning_path": sac_args.get("fine_tuning_path", None),
            "goal_decoders": sac_args["goal_decoders"],
            "clip_target_q_values": sac_args.get("clip_target_q_values", False),
        }

        self.log_temperature = self.init_log_temperature()
        self.policy_params = self.init_ffn_submodule(ffn_name="policy")
        self.q1_params = self.init_ffn_submodule(
            ffn_name="q")
        self.q2_params = self.init_ffn_submodule(
            ffn_name="q")
        
        self.q1_target_params = deepcopy(self.q1_params)
        self.q2_target_params = deepcopy(self.q2_params)

        # target networks require no gradients
        for param in self.q1_target_params.parameters():
            param.requires_grad = False
        for param in self.q2_target_params.parameters():
            param.requires_grad = False
            
        self.projection_matrix = self.compute_projection_matrix(decoders)
        self.use_state_dep_projection = any([decoder.get("state_dep_direction", False) for measurement_decoders in self.hyperparameters["goal_decoders"] for decoder in measurement_decoders])
        if self.use_state_dep_projection:
            self.state_dep_projection = self.compute_state_dep_projection()
        else:
            self.state_dep_projection = tc.nn.ModuleList([])

    def generate_free_trajectory(
        self: HierarchisedDSRModel,
        z0: tc.Tensor,
        T: int,
        cumulative_timeseries_indices: tc.Tensor,
        measurement_index: int,
        S: Optional[tc.Tensor] = None,
        apply_actions: Optional[bool] = True,
        expected: Optional[bool] = False,
        noise: Optional[float] = None,
    ):
        """
        Forward pass of the controlled DSR model (no teacher forcing).

        Output: predictions, actions (!!!)
                Z_predicted = (B x T x N) tensor of next steps, such that ideally Z_predicted[:,t,:] == Z_encoded[:,t+1;:]
        """

        # set initial latent state
        z = z0

        latent_step_params = self.construct_params(cumulative_timeseries_index=cumulative_timeseries_indices)
        policy_params = self.sac_construct_params(
            cumulative_timeseries_indices, self.policy_params, "policy")

        Z_predicted = tc.zeros((T+1, *z0.shape), device=z0.device)
        actions = tc.zeros((T, z0.shape[0], self.hyperparameters["dim_forcing"]), device=z0.device)

        Z_predicted[0] = z0
        if noise is not None:
            noise = tc.randn(T, *z0.shape, device=z0.device) * noise

        for t in range(T):
            state_external = tc.cat([z, S[t]], dim=-1) if S[t] is not None else z
            action, _ = self.sample_action(state_external, policy_params, measurement_index, expected
            )

            if apply_actions:
                z = z + tc.nn.functional.pad(action, (0, self.hyperparameters["latent_dim"] - action.shape[-1]), 'constant', 0) # lift the action to latent dim
            z = self.latent_step.forward(
                z, S[t], latent_step_params, self.hyperparameters)
            if noise is not None:
                z = z + noise[t]

            Z_predicted[t+1] = z
            actions[t] = action

        return Z_predicted.contiguous().permute(1, 0, 2), actions.contiguous().permute(1, 0, 2)

    def sample_action(
        self,
        state: tc.Tensor,
        policy_params: nn.ModuleList,
        measurement_index: int,
        expected: bool = False,
    ) -> tc.Tensor:
        """Sample an action from the policy given the current state."""

        mean, log_std = FFN.forward(state, policy_params, self.hyperparameters
        ).chunk(2, dim=-1)
        log_std = tc.clamp(log_std, -15, 2)
        if expected:
            u = mean
        else:
            noise = tc.randn_like(mean)
            u = mean + noise * log_std.exp()
        action = tc.tanh(u)
        normal = tc.distributions.Normal(mean, log_std.exp())
        loglikelihood = normal.log_prob(u)
        loglikelihood -= tc.log(1 - action.pow(2) + 1e-6)
        # TODO: because of the projection, this is not exact anymore
        action = self.project_action(
            action, measurement_index
        )
        if self.use_state_dep_projection:
            action = self.project_action_state_dep(action, state, measurement_index)
        loglikelihood = loglikelihood.sum(dim=-1, keepdim=True)
        return action, loglikelihood

    def project_action(self, action: tc.Tensor, measurement_index: int) -> tc.Tensor:
        """Project the action on (low rank subspace of) the null space of the decoder """
        action = self.projection_matrix[measurement_index](action)
        return action
    
    def project_action_state_dep(self, action: tc.Tensor, state: tc.Tensor, measurement_index: int) -> tc.Tensor:
        """Project the action state dependently"""
        state_dep_projection = self.state_dep_projection[measurement_index](state)
        # normalize state_dep_projection to 2 norm 1
        # state_dep_projection = state_dep_projection / (state_dep_projection.norm(dim=-1, keepdim=True) + 1e-6)
        # diagonal matrix multiplication 
        action = tc.einsum("ij, ij -> ij", state_dep_projection, action)
        return action

    def compute_q_values(
        self,
        state: tc.Tensor,
        action: tc.Tensor,
        cumulative_timeseries_indices: tc.Tensor,
        target: bool = False,
    ) -> tuple[tc.Tensor, tc.Tensor]:
        """Compute the Q-values for given state-action pairs."""
        if not target:
            q1_params = self.sac_construct_params(
                cumulative_timeseries_indices, self.q1_params, "q")
            q2_params = self.sac_construct_params(
                cumulative_timeseries_indices, self.q2_params, "q")
            # stack state and action
            state_action = tc.cat([state, action], dim=-1) 

            q1 = FFN.forward(state_action, q1_params, self.hyperparameters)
            q2 = FFN.forward(state_action, q2_params, self.hyperparameters)
        else:
            q1_params = self.sac_construct_params(
                cumulative_timeseries_indices, self.q1_target_params, "q")
            q2_params = self.sac_construct_params(
                cumulative_timeseries_indices, self.q2_target_params, "q")
            # stack state and action
            state_action = tc.cat([state, action], dim=-1) 

            q1 = FFN.forward(state_action, q1_params, self.hyperparameters)
            q2 = FFN.forward(state_action, q2_params, self.hyperparameters)
            # clip target q values at 0
            if self.hyperparameters.get("clip_target_q_values", False):
                q1 = tc.clamp(q1, max=0)
                q2 = tc.clamp(q2, max=0)

        return q1, q2


    def construct_temperature(self, cumulative_timeseries_indices: tc.Tensor):
        if self.hyperparameters["policy_hierarchisation_scheme"]["scheme"] == "none":
            return self.log_temperature.repeat(cumulative_timeseries_indices.shape[0], 1)
        else:
            return self.log_temperature[self.subject_index_from_timeseries_index[cumulative_timeseries_indices]]

    def sac_construct_params(self, cumulative_timeseries_indices: tc.Tensor | int, ffn_params: nn.ParameterDict, ffn_name: str):
        if self.hyperparameters[f"{ffn_name}_hierarchisation_scheme"]["scheme"] == "none":
            return ffn_params

        if isinstance(cumulative_timeseries_indices, int):
            cumulative_timeseries_indices = tc.tensor(
                cumulative_timeseries_indices, dtype=int
            ).unsqueeze(0)

        constructed_params = FFN.construct_params(
            self.hyperparameters[f"{ffn_name}_hierarchisation_scheme"],
            self.hyperparameters,
            ffn_params["group"],
            {
                "subject_vectors": ffn_params["subject_vectors"][
                    self.subject_index_from_timeseries_index[
                        cumulative_timeseries_indices
                    ]
                ],
            },
        )

        return constructed_params

    def reg_loss(self, cumulative_timeseries_indices: tc.Tensor, params: nn.ParameterDict, ffn_name: str) -> tc.Tensor:
        """Compute the regularization loss for the SAC model."""
        params = self.sac_construct_params(
            cumulative_timeseries_indices, params, ffn_name)
        reg_loss = FFN.reg_loss(params)
        return reg_loss

    def init_ffn_submodule(self, ffn_name: str):

        if self.hyperparameters[f"{ffn_name}_hierarchisation_scheme"]["scheme"] == "none":
            # In case of no hierarchisation a ParameterDict is returned, which can be stored directly,
            # and the hierarchisation scheme "none" is always handled as a special case.
            ffn_params = nn.ParameterDict(
                FFN.initialize_subject_level_params(
                    self.hyperparameters[f"{ffn_name}_hierarchisation_scheme"],
                    self.hyperparameters,
                    ffn_name,
                ),
            )
            return ffn_params

        else:
            ffn_params = nn.ParameterDict(
                {
                    "group": FFN.initialize_group_level_params(
                        self.hyperparameters[f"{ffn_name}_hierarchisation_scheme"], self.hyperparameters, ffn_name
                    )
                }
            )
            subject_vector = self.latent_step_params["subject_vectors"].clone().detach()
            
            if self.hyperparameters[f"{ffn_name}_hierarchisation_scheme"]["feature_dimension"] != subject_vector.shape[1]:
                print("Feature dimension of hierarchisation scheme does not match subject vector dimension, reinitializing subject vector for SAC hierarchisation")
                
                subject_vector_new = FFN.initialize_subject_level_params(
                        self.hyperparameters[f"{ffn_name}_hierarchisation_scheme"],
                        self.hyperparameters,
                        ffn_name,
                    )
                ffn_params.update(
                    {
                        "subject_vectors": tc.repeat_interleave(
                            subject_vector_new, self.num_subjects, dim=0
                        ),
                    }
                )
                
            elif not self.hyperparameters[f"{ffn_name}_hierarchisation_scheme"]["keep_subject_parameters"] or self.hyperparameters["fine_tuning_path"] is not None:
                ffn_params.update(
                    {
                        "subject_vectors": subject_vector,
                    }
                )              
                ffn_params["subject_vectors"].requires_grad = True      
            else:
                ffn_params.update(
                    {
                        "subject_vectors": subject_vector,
                    }
                )
                ffn_params["subject_vectors"].requires_grad = False

            return ffn_params

    def init_log_temperature(self) -> nn.Parameter:
        """Initialize the temperature parameter for SAC."""
        if self.hyperparameters["policy_hierarchisation_scheme"]["scheme"] == "none":
            temperature = nn.Parameter(
                tc.tensor(self.hyperparameters["init_log_temperature"], dtype=tc.float32)
            )
        else:
            temperature = nn.Parameter(
                tc.full((self.num_subjects, 1),
                        self.hyperparameters["init_log_temperature"], dtype=tc.float32)
            )

        if not self.hyperparameters.get("adjustable_temperature"):
            temperature.requires_grad = False
        return temperature


    def compute_projection_matrix(self, decoders) -> nn.ModuleList:
            """Compute the projection matrix on the null space of the decoder"""
            projection_matrix = nn.ModuleList()
            for measurement_index in range(len(self.hyperparameters["goal_decoders"])):
                stacked_mean_weights = []
                stacked_mean_biases = []
                for goal_decoder in self.hyperparameters["goal_decoders"][measurement_index]:
                    decoder_name = goal_decoder["name"]
                    # TODO: rename to goal_dimensions
                    control_dimensions = tc.as_tensor(goal_decoder["control_dimensions"])

                    decoder = decoders[measurement_index][decoder_name]

                    if isinstance(decoder, Gaussian):
                        stacked_mean_weights.append(decoder.mean.weight[control_dimensions])
                        stacked_mean_biases.append(decoder.mean.bias[control_dimensions])
                    elif isinstance(decoder, Identity):
                        stacked_mean_weights.append(tc.eye(len(control_dimensions), decoder.latent_dim))
                        stacked_mean_biases.append(tc.zeros(len(control_dimensions)))
                    else:
                        raise ValueError("Decoder for null space projection must be Gaussian or Identity, got {}".format(type(decoder)))

                stacked_mean_weights = tc.cat(stacked_mean_weights, dim=0)
                stacked_mean_biases = tc.cat(stacked_mean_biases, dim=0)
                # compute null space of stacked_mean_weights using SVD
                U, S, Vh = tc.linalg.svd(stacked_mean_weights)
                null_space = Vh[S.size()[0]:, :].T
                projection_matrix_ = null_space @ null_space.T
                # bias must satisfy stacked_mean_weights @ bias + stacked_mean_biases = 0
                projection_bias = tc.linalg.lstsq(stacked_mean_weights, -stacked_mean_biases).solution
                linear = nn.Linear(Vh.shape[0], Vh.shape[0])
                # no gradient required
                linear.weight = nn.Parameter(projection_matrix_)
                linear.bias = nn.Parameter(projection_bias)
                linear.weight.requires_grad = False
                linear.bias.requires_grad = False
                # if action dim is provided, add projection as trainable matrix on lower subspace of null space
                if goal_decoder.get("action_dim", None) is not None:
                    action_dim = goal_decoder["action_dim"]
                    # projection needs to be of order action dim, but same size as projection_matrix_
                    linear_2 = nn.Linear(Vh.shape[0], action_dim, bias=False)
                    linear_3 = nn.Linear(action_dim, Vh.shape[0], bias=False)
                    if not goal_decoder.get("state_dep_direction", False):
                        linear_2.weight.requires_grad = True
                        linear_3.weight.requires_grad = True
                    else:
                        linear_2.weight.requires_grad = False
                        linear_3.weight.requires_grad = False
                    projection_matrix.append(nn.Sequential(linear_2, linear_3, linear))
                else:
                    projection_matrix.append(linear)
            return projection_matrix
        

    def compute_state_dep_projection(self) -> nn.ModuleList:
        state_dep_projection_matrix = nn.ModuleList()
        for measurement_index in range(len(self.hyperparameters["goal_decoders"])):        
            # init a 2 layer MLP with input dimension of state and action and output dimension of action dim 
            # projection matrix has the output as the diagonal entries
            if self.hyperparameters["goal_decoders"][measurement_index][0].get("state_dep_direction", False):
                MLP = nn.Sequential(
                    nn.Linear(self.hyperparameters["latent_dim"], 128),
                    nn.ReLU(),
                    nn.Linear(128, self.hyperparameters["dim_forcing"])
                )
                #weight = tc.zeros((self.hyperparameters["dim_forcing"], self.hyperparameters["latent_dim"]))
                #weight[-2, 1] = (0.5+1)*0.5
                #weight[-1, 3] = 1*0.5
                #weight.requires_grad = False
                #MLP = nn.Linear(self.hyperparameters["latent_dim"], self.hyperparameters["dim_forcing"], bias=False)
                #MLP.weight = nn.Parameter(weight)
                
            else:
                MLP = None    
                
            state_dep_projection_matrix.append(MLP)
        return state_dep_projection_matrix
