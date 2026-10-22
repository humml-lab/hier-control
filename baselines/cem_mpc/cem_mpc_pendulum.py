"""
Cross-Entropy Method Model Predictive Control (CEM-MPC)
=======================================================

This module implements a generic CEM-MPC controller for learned latent dynamics.

Designed for the paper setup:
    - control is optimized entirely in a learned world model;
    - the world model may be an ALRNN or any other latent transition model;
    - actions can be projected into an admissible / modality-constrained subspace;
    - rewards are evaluated in latent or decoded observation space;
    - no true-system interaction is required during planning.

You need to provide:
    1. `rollout_fn(latent_state, action_sequences, context)`
       -> simulated trajectories and optional auxiliary outputs
    2. `reward_fn(trajectories, action_sequences, context)`
       -> scalar return per candidate action sequence
    3. optionally, `action_projector(actions, context)`
       -> projected admissible actions

Expected shapes
---------------
latent_state:
    (latent_dim,) or (batch, latent_dim)

action_sequences:
    (num_candidates, horizon, action_dim)

trajectories returned by rollout_fn:
    Usually (num_candidates, horizon + 1, latent_dim)

returns returned by reward_fn:
    (num_candidates,)

The controller is deliberately agnostic about how your ALRNN is represented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

import numpy as np
import json
import os
import sys
import torch as tc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
from src.models.mtf_sac_model import MTF_SAC_Model
from src.main import init_dataset
from src.environments.pendulum import Hamiltonian, ode_state_to_observation, double_pendulum_derivatives, observation_to_ode_state
from src.paths import get_results_dir
RESULTS_DIR, DATA_DIR = get_results_dir()
RESULTS_PATH = os.path.join(RESULTS_DIR, "baselines", "cem_mpc")

params_file = DATA_DIR + "/multi_task_double_pendulum/default/params.npy"

Array = np.ndarray


class RolloutFunction(Protocol):
    def __call__(
        self,
        latent_state: Array,
        action_sequences: Array,
        context: Optional[Dict[str, Any]] = None,
    ) -> Union[Array, Tuple[Array, Dict[str, Any]]]:
        """Simulate candidate action sequences in the learned world model."""


class RewardFunction(Protocol):
    def __call__(
        self,
        trajectories: Array,
        action_sequences: Array,
        context: Optional[Dict[str, Any]] = None,
    ) -> Array:
        """Return one scalar objective value for every candidate sequence."""


class ActionProjector(Protocol):
    def __call__(
        self,
        actions: Array,
        context: Optional[Dict[str, Any]] = None,
    ) -> Array:
        """Map candidate actions to the admissible action space."""


@dataclass
class CEMMPCConfig:
    """Hyperparameters for CEM-MPC."""

    horizon: int
    action_dim: int

    num_candidates: int = 1024
    num_elites: int = 64
    num_iterations: int = 6

    # Initial proposal distribution.
    init_mean: Optional[Array] = None
    init_std: Union[float, Array] = 0.5

    # Action constraints before or after projection.
    action_low: Union[float, Array] = -1.0
    action_high: Union[float, Array] = 1.0

    # Proposal-distribution stabilization.
    mean_momentum: float = 0.1
    std_momentum: float = 0.1
    min_std: Union[float, Array] = 0.05

    # Receding-horizon warm start.
    warm_start: bool = True

    # Reproducibility.
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive.")
        if self.num_elites <= 0 or self.num_elites > self.num_candidates:
            raise ValueError(
                "num_elites must be in the interval [1, num_candidates]."
            )
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive.")
        if not 0.0 <= self.mean_momentum < 1.0:
            raise ValueError("mean_momentum must be in [0, 1).")
        if not 0.0 <= self.std_momentum < 1.0:
            raise ValueError("std_momentum must be in [0, 1).")


@dataclass
class CEMMPCDiagnostics:
    """Planning diagnostics from the most recent call to `plan`."""

    best_return: float
    mean_return: float
    elite_return_mean: float
    elite_return_std: float
    proposal_mean: Array
    proposal_std: Array
    best_action_sequence: Array
    iteration_best_returns: Array = field(repr=False)
    iteration_mean_returns: Array = field(repr=False)


class CEMMPC:
    """
    Cross-Entropy Method Model Predictive Control.

    The controller optimizes a distribution over length-H action sequences:
        a_{t:t+H-1} ~ N(mu, diag(sigma^2))

    At each CEM iteration:
        1. sample candidate sequences;
        2. project and clip actions;
        3. roll out candidates in the learned world model;
        4. score them via the supplied reward function;
        5. refit the proposal to the top-performing elite sequences.

    `plan()` returns only the first action. Call it again at the next real or
    simulated state to execute MPC in receding-horizon form.
    """

    def __init__(
        self,
        config: CEMMPCConfig,
        rollout_fn: RolloutFunction,
        reward_fn: RewardFunction,
        action_projector: Optional[ActionProjector] = None,
    ) -> None:
        self.config = config
        self.rollout_fn = rollout_fn
        self.reward_fn = reward_fn
        self.action_projector = action_projector

        self.rng = np.random.default_rng(config.seed)

        self._action_low = self._broadcast_action_parameter(
            config.action_low, "action_low"
        )
        self._action_high = self._broadcast_action_parameter(
            config.action_high, "action_high"
        )
        self._min_std = self._broadcast_action_parameter(config.min_std, "min_std")
        self._init_std = self._broadcast_action_parameter(config.init_std, "init_std")

        if np.any(self._action_low >= self._action_high):
            raise ValueError("Each action_low entry must be strictly below action_high.")
        if np.any(self._min_std <= 0.0):
            raise ValueError("min_std must be strictly positive.")
        if np.any(self._init_std <= 0.0):
            raise ValueError("init_std must be strictly positive.")

        self._initial_mean = self._build_initial_mean(config.init_mean)
        self._proposal_mean = self._initial_mean.copy()
        self._proposal_std = self._init_std.copy()

        self.last_diagnostics: Optional[CEMMPCDiagnostics] = None

    def reset(self) -> None:
        """Reset warm-start state between episodes."""
        self._proposal_mean = self._initial_mean.copy()
        self._proposal_std = self._init_std.copy()
        self.last_diagnostics = None

    def plan(
        self,
        latent_state: Array,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Array, CEMMPCDiagnostics]:
        """
        Optimize an action sequence and return the first action.

        Parameters
        ----------
        latent_state:
            Current inferred ALRNN latent state, shape `(latent_dim,)`.
        context:
            Optional dictionary forwarded unchanged to rollout_fn, reward_fn,
            and action_projector. Typical contents:
                {
                    "subject_embedding": subject_vector,
                    "external_inputs": s_future,
                    "target": target_observation,
                    "decoder": decoder,
                    "action_nullspace": N_DT,
                    "discount": gamma,
                }

        Returns
        -------
        action:
            First MPC action, shape `(action_dim,)`.
        diagnostics:
            Summary of proposal fitting and achieved candidate returns.
        """
        latent_state = np.asarray(latent_state, dtype=np.float64)
        if latent_state.ndim != 1:
            raise ValueError(
                "latent_state must have shape (latent_dim,). "
                f"Received shape {latent_state.shape}."
            )

        mean = self._proposal_mean.copy()
        std = self._proposal_std.copy()

        iteration_best_returns = []
        iteration_mean_returns = []

        best_return = -np.inf
        best_actions: Optional[Array] = None

        for _ in range(self.config.num_iterations):
            candidate_actions = self._sample_action_sequences(mean, std)
            candidate_actions = self._enforce_action_constraints(
                candidate_actions, context
            )

            trajectories, rollout_info = self._run_rollout(
                latent_state, candidate_actions, context
            )
            reward_context = self._merge_context(context, rollout_info)
            returns = np.asarray(
                self.reward_fn(tc.tensor(trajectories[:, 1:]), tc.tensor(candidate_actions), reward_context["measurement_index"]).squeeze(),
                dtype=np.float64,
            )
            # discount
            returns = (returns * reward_context["discount"] ** np.arange(returns.shape[1])[None, :]).sum(axis=-1)

            if returns.shape != (self.config.num_candidates,):
                raise ValueError(
                    "reward_fn must return one return per candidate sequence. "
                    f"Expected {(self.config.num_candidates,)}, got {returns.shape}."
                )
            if not np.all(np.isfinite(returns)):
                raise ValueError(
                    "reward_fn returned non-finite values. "
                    "Check rollout stability and reward computation."
                )

            elite_indices = np.argpartition(
                returns, -self.config.num_elites
            )[-self.config.num_elites :]
            elite_actions = candidate_actions[elite_indices]
            elite_returns = returns[elite_indices]

            best_idx = int(np.argmax(returns))
            if returns[best_idx] > best_return:
                best_return = float(returns[best_idx])
                best_actions = candidate_actions[best_idx].copy()

            elite_mean = elite_actions.mean(axis=0)
            elite_std = elite_actions.std(axis=0)

            mean = (
                self.config.mean_momentum * mean
                + (1.0 - self.config.mean_momentum) * elite_mean
            )
            std = (
                self.config.std_momentum * std
                + (1.0 - self.config.std_momentum) * elite_std
            )
            std = np.maximum(std, self._min_std)

            iteration_best_returns.append(float(returns[best_idx]))
            iteration_mean_returns.append(float(returns.mean()))

        if best_actions is None:
            raise RuntimeError("CEM failed to produce any candidate action sequence.")

        # Re-evaluate final proposal mean after constraints. This often yields a
        # smoother action than the single best sampled sequence.
        final_mean_actions = self._enforce_action_constraints(
            mean[None, ...], context
        )[0]
        final_trajectories, final_rollout_info = self._run_rollout(
            latent_state, final_mean_actions[None, ...], context
        )
        final_context = self._merge_context(context, final_rollout_info)
        final_return = self.reward_fn(
                tc.tensor(final_trajectories[:, 1:]),
                tc.tensor(final_mean_actions[None, ...]),
                final_context["measurement_index"],
            ).squeeze()
        
        final_return = (final_return * reward_context["discount"] ** np.arange(final_return.shape[0])[None, :]).sum(axis=-1)

        # Select the better of final-mean and best sampled sequence.
        if final_return >= best_return:
            selected_sequence = final_mean_actions
            selected_return = final_return
        else:
            selected_sequence = best_actions
            selected_return = best_return

        self._update_warm_start(selected_sequence, std)

        diagnostics = CEMMPCDiagnostics(
            best_return=selected_return,
            mean_return=float(iteration_mean_returns[-1]),
            elite_return_mean=float(elite_returns.mean()),
            elite_return_std=float(elite_returns.std()),
            proposal_mean=mean.copy(),
            proposal_std=std.copy(),
            best_action_sequence=selected_sequence.copy(),
            iteration_best_returns=np.asarray(iteration_best_returns),
            iteration_mean_returns=np.asarray(iteration_mean_returns),
        )
        self.last_diagnostics = diagnostics

        return selected_sequence[0].copy(), diagnostics

    def _sample_action_sequences(self, mean: Array, std: Array) -> Array:
        noise = self.rng.normal(
            loc=0.0,
            scale=1.0,
            size=(
                self.config.num_candidates,
                self.config.horizon,
                self.config.action_dim,
            ),
        )
        return mean[None, ...] + noise * std[None, ...]

    def _enforce_action_constraints(
        self,
        actions: Array,
        context: Optional[Dict[str, Any]],
    ) -> Array:
        constrained = np.asarray(actions, dtype=np.float64)

        constrained = np.clip(
                constrained,
                self._action_low[None, ...],
                self._action_high[None, ...],
                )
        
        if self.action_projector is not None:
            constrained = np.asarray(
                self.action_projector(constrained, context), dtype=np.float64
            )
            if constrained.shape != actions.shape:
                raise ValueError(
                    "action_projector must preserve the action tensor shape. "
                    f"Expected {actions.shape}, got {constrained.shape}."
                )


        return constrained

    def _run_rollout(
        self,
        latent_state: Array,
        action_sequences: Array,
        context: Optional[Dict[str, Any]],
    ) -> Tuple[Array, Dict[str, Any]]:
        result = self.rollout_fn(latent_state, action_sequences, context)

        if isinstance(result, tuple):
            trajectories, rollout_info = result
            if not isinstance(rollout_info, dict):
                raise TypeError(
                    "If rollout_fn returns a tuple, its second element must be a dict."
                )
        else:
            trajectories, rollout_info = result, {}

        trajectories = np.asarray(trajectories, dtype=np.float64)
        expected_batch = self.config.num_candidates
        if trajectories.shape[0] != action_sequences.shape[0]:
            raise ValueError(
                "rollout_fn must return one trajectory per candidate sequence. "
                f"Expected first dimension {expected_batch}, "
                f"got {trajectories.shape[0]}."
            )

        return trajectories, rollout_info

    def _update_warm_start(self, selected_sequence: Array, final_std: Array) -> None:
        if not self.config.warm_start:
            self._proposal_mean = self._initial_mean.copy()
            self._proposal_std = self._init_std.copy()
            return

        shifted_mean = np.empty_like(selected_sequence)
        shifted_mean[:-1] = selected_sequence[1:]
        shifted_mean[-1] = selected_sequence[-1]

        shifted_std = np.empty_like(final_std)
        shifted_std[:-1] = final_std[1:]
        shifted_std[-1] = final_std[-1]

        self._proposal_mean = shifted_mean
        self._proposal_std = np.maximum(shifted_std, self._min_std)

    def _build_initial_mean(self, init_mean: Optional[Array]) -> Array:
        if init_mean is None:
            return np.zeros((self.config.horizon, self.config.action_dim))

        init_mean = np.asarray(init_mean, dtype=np.float64)
        target_shape = (self.config.horizon, self.config.action_dim)

        if init_mean.shape == (self.config.action_dim,):
            return np.broadcast_to(init_mean, target_shape).copy()
        if init_mean.shape == target_shape:
            return init_mean.copy()

        raise ValueError(
            "init_mean must have shape (action_dim,) or (horizon, action_dim). "
            f"Received {init_mean.shape}."
        )

    def _broadcast_action_parameter(
        self,
        parameter: Union[float, Array],
        name: str,
    ) -> Array:
        array = np.asarray(parameter, dtype=np.float64)
        target_shape = (self.config.horizon, self.config.action_dim)

        if array.ndim == 0:
            return np.full(target_shape, float(array))
        if array.shape == (self.config.action_dim,):
            return np.broadcast_to(array, target_shape).copy()
        if array.shape == target_shape:
            return array.copy()

        raise ValueError(
            f"{name} must be scalar, shape (action_dim,), or shape "
            f"(horizon, action_dim). Received {array.shape}."
        )

    @staticmethod
    def _merge_context(
        context: Optional[Dict[str, Any]],
        rollout_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        if context is not None:
            merged.update(context)
        merged.update(rollout_info)
        return merged


# ---------------------------------------------------------------------------
# ALRNN integration template
# ---------------------------------------------------------------------------

def make_alrnn_rollout_fn(
    alrnn_step_fn: Callable[[Array, Array, Optional[Dict[str, Any]]], Array],
) -> RolloutFunction:
    """
    Wrap a single-step ALRNN transition function into a batched rollout function.

    Your `alrnn_step_fn` should implement one learned transition:
        z_next = alrnn_step_fn(z_current, action_t, context)

    with:
        z_current: (batch, latent_dim)
        action_t:  (batch, action_dim)
        z_next:    (batch, latent_dim)

    The `context` can contain subject embeddings, exogenous inputs, model
    parameters, or a per-time-step input schedule.

    Example ALRNN-style transition:
        z_next = A @ (z + action) + W @ relu_masked(z + action) + C @ s_t + h

    Returns
    -------
    rollout_fn:
        Callable accepted by CEMMPC.
    """

    def rollout_fn(
        latent_state: Array,
        action_sequences: Array,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Array, Dict[str, Any]]:
        num_candidates, horizon, _ = action_sequences.shape
        latent_dim = latent_state.shape[0]

        trajectories = np.empty(
            (num_candidates, horizon + 1, latent_dim), dtype=np.float64
        )
        trajectories[:, 0] = latent_state[None, :]

        z = np.broadcast_to(latent_state, (num_candidates, latent_dim)).copy()

        for t in range(horizon):
            step_context = dict(context or {})
            step_context["planning_step"] = t

            # Optional convention for external inputs:
            # context["external_input_sequence"] has shape (horizon, input_dim)
            if (
                context is not None
                and "external_input_sequence" in context
                and context["external_input_sequence"] is not None
            ):
                external_inputs = np.asarray(
                    context["external_input_sequence"], dtype=np.float64
                )
                if external_inputs.shape[0] < horizon:
                    raise ValueError(
                        "external_input_sequence must cover the MPC horizon."
                    )
                step_context["external_input_t"] = external_inputs[t]

            z = np.asarray(
                alrnn_step_fn(z, action_sequences[:, t], step_context),
                dtype=np.float64,
            )
            if z.shape != (num_candidates, latent_dim):
                raise ValueError(
                    "alrnn_step_fn returned an invalid shape. "
                    f"Expected {(num_candidates, latent_dim)}, got {z.shape}."
                )

            trajectories[:, t + 1] = z

        return trajectories, {}

    return rollout_fn


def make_nullspace_projector(nullspace_matrix: Array) -> ActionProjector:
    """
    Build an action projector for modality-constrained latent actions.

    Parameters
    ----------
    nullspace_matrix:
        Matrix N_DT with shape (latent_action_dim, proposed_action_dim), or a
        square projection matrix with shape (action_dim, action_dim).

    Notes
    -----
    In the paper, N_DT projects action proposals into the common null space of
    target decoder weights. If candidate actions already live in latent action
    coordinates, a square projection matrix is typical.

    The CEM action dimension must equal the input dimension of this matrix.
    """

    matrix = np.asarray(nullspace_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("nullspace_matrix must be two-dimensional.")

    def projector(
        actions: Array,
        context: Optional[Dict[str, Any]] = None,
    ) -> Array:
        # actions: (num_candidates, horizon, input_action_dim)
        if actions.shape[-1] != matrix.shape[1]:
            raise ValueError(
                "Action dimension does not match nullspace projection input. "
                f"Expected {matrix.shape[1]}, got {actions.shape[-1]}."
            )
        return actions @ matrix.T

    return projector



# ---------------------------------------------------------------------------
# True-system double-pendulum evaluation (CEM-MPC in closed loop)
# ---------------------------------------------------------------------------

def evaluate_cem_on_true_double_pendulum(
    controller,
    model,
    x0,
    measurement_index,
    observation_to_ode_state,
    ode_state_to_observation,
    double_pendulum_derivatives,
    pendulum_params,
    dt,
    T,
    controller_context=None,
    apply_actions=True,
):
    """
    Evaluate CEM-MPC against the TRUE double-pendulum ODE in closed loop.

    CEM plans using the learned ALRNN via controller.plan(...). Only the first
    planned action is then applied to the true ODE. The next observation from
    the true ODE is encoded again and CEM replans.

    Assumptions matching the supplied evaluation code:
      - x0 is a 6D observation, either shape (6,) or (1, 6)
      - observation_to_ode_state(obs) returns [theta1, omega1, theta2, omega2]
      - ode_state_to_observation(x) returns the 6D model observation
      - action entries 4 and 5 act on omega1 and omega2
      - model.encoder_model(obs_tensor, measurement_index)[0] returns latents

    Returns a dictionary with:
      observations: (1, T+1, 6)
      ode_states:   (T+1, 4)
      actions:      (1, T, action_dim)
      zs:           (1, T, latent_dim)
      diagnostics:  list of CEM diagnostics
    """
    import numpy as np
    from scipy.integrate import odeint

    if T <= 0:
        raise ValueError("T must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    # Match the user's original x0 handling, but also accept a NumPy array.
    if isinstance(x0, tc.Tensor):
        obs = x0.detach().cpu().numpy()
    else:
        obs = np.asarray(x0, dtype=np.float64)

    if obs.ndim == 2:
        if obs.shape[0] != 1:
            raise ValueError(f"x0 must have batch size 1; received {obs.shape}.")
        obs = obs[0]
    if obs.shape != (6,):
        raise ValueError(f"x0 must resolve to a 6D observation; got {obs.shape}.")

    x = np.asarray(observation_to_ode_state(obs), dtype=np.float64)
    if x.shape != (4,):
        raise ValueError(
            "observation_to_ode_state must return a 4D ODE state "
            f"[theta1, omega1, theta2, omega2]; got {x.shape}."
        )

    # Infer latent dimensionality once so arrays can be allocated.
    with tc.no_grad():
        obs_tensor = tc.as_tensor(obs[None, :], device=model.device).float()
        initial_encoder_output = model.encoder_model(obs_tensor, measurement_index)[0]

    if initial_encoder_output.ndim == 1:
        initial_latent = initial_encoder_output
    else:
        initial_latent = initial_encoder_output[0]

    latent_dim = int(initial_latent.numel())
    action_dim = int(controller.config.action_dim)

    xs = tc.zeros((1, T + 1, 6), dtype=tc.float32)
    actions = tc.zeros((1, T, action_dim), dtype=tc.float32)
    zs = tc.zeros((1, T, latent_dim), dtype=tc.float32)

    xs[0, 0] = tc.as_tensor(obs, dtype=tc.float32)

    ode_states = np.zeros((T + 1, 4), dtype=np.float64)
    ode_states[0] = x
    diagnostics = []

    base_context = dict(controller_context or {})
    current_time = 0.0

    for i in range(1, T + 1):
        # 1) Current true-system observation -> ALRNN latent state.
        obs_tensor = tc.as_tensor(obs[None, :], device=model.device).float()
        with tc.no_grad():
            encoded_out = model.encoder_model(obs_tensor, measurement_index)[0]

        if encoded_out.ndim == 1:
            encoded = encoded_out.unsqueeze(0)
        else:
            encoded = encoded_out[:1]

        latent_state = encoded[0].detach().cpu().numpy()

        # Optional alignment of a provided future external-input sequence with
        # the current MPC time step.
        step_context = dict(base_context)
        if "external_input_sequence" in base_context:
            full_sequence = np.asarray(base_context["external_input_sequence"])
            horizon = controller.config.horizon
            future = full_sequence[i - 1 : i - 1 + horizon]
            if future.shape[0] == 0:
                raise ValueError("external_input_sequence is empty.")
            if future.shape[0] < horizon:
                future = np.concatenate(
                    [
                        future,
                        np.repeat(future[-1:], horizon - future.shape[0], axis=0),
                    ],
                    axis=0,
                )
            step_context["external_input_sequence"] = future

        # 2) CEM uses only the LEARNED ALRNN internally.
        proposed_action_from_controller, info = controller.plan(
            latent_state=latent_state,
            context=step_context,
        )
        action = np.asarray(proposed_action_from_controller, dtype=np.float64)

        if action.shape != (action_dim,):
            raise ValueError(
                "controller.plan must return one flat action of shape "
                f"({action_dim},); got {action.shape}."
            )

        # 3) Apply the selected first action to the TRUE physical state.
        if apply_actions:
            x = x.copy()
            x[1] += action[4]  # Action on omega1
            x[3] += action[5]  # Action on omega2

        # 4) Integrate one step on the true 4D ODE state.
        x = odeint(
            double_pendulum_derivatives,
            x,
            [current_time, current_time + dt],
            args=(
                pendulum_params["m1"],
                pendulum_params["m2"],
                pendulum_params["l1"],
                pendulum_params["l2"],
                9.81,
            ),
        )[1]
        current_time += dt

        # 5) Convert true ODE state back to the model's 6D observation.
        obs = np.asarray(ode_state_to_observation(x), dtype=np.float64)
        if obs.shape != (6,):
            raise ValueError(
                "ode_state_to_observation must return a 6D observation; "
                f"got {obs.shape}."
            )

        xs[0, i] = tc.as_tensor(obs, dtype=tc.float32)
        actions[0, i - 1] = tc.as_tensor(action, dtype=tc.float32)
        zs[0, i - 1] = encoded[0].detach().cpu().float()
        ode_states[i] = x
        diagnostics.append(info)

    return {
        "observations": xs,
        "ode_states": ode_states,
        "actions": actions,
        "zs": zs,
        "diagnostics": diagnostics,
    }


def evaluate_cem_on_world_model(
    controller,
    model,
    x0,
    measurement_index,
    cumulative_timeseries_index,
    T,
    controller_context=None,
    apply_actions=True,
):
    """
    Evaluate CEM-MPC against the loaded world model in closed loop.

    The controller is queried in the same way as in the true-environment
    evaluation, but the next state is generated by the learned latent
    transition model instead of by numerical integration of the physical
    double-pendulum dynamics.
    """

    if isinstance(x0, tc.Tensor):
        obs = x0.detach().cpu().numpy()
    else:
        obs = np.asarray(x0, dtype=np.float64)

    if obs.ndim == 2:
        if obs.shape[0] != 1:
            raise ValueError(f"x0 must have batch size 1; received {obs.shape}.")
        obs = obs[0]
    if obs.shape != (6,):
        raise ValueError(f"x0 must resolve to a 6D observation; got {obs.shape}.")

    if T <= 0:
        raise ValueError("T must be positive.")

    with tc.no_grad():
        obs_tensor = tc.as_tensor(obs[None, :], device=model.device).float()
        initial_encoder_output = model.encoder_model(obs_tensor, measurement_index)[0]

    if initial_encoder_output.ndim == 1:
        initial_latent = initial_encoder_output
    else:
        initial_latent = initial_encoder_output[0]

    latent_dim = int(initial_latent.numel())
    action_dim = int(controller.config.action_dim)
    rollout_steps = int(T)

    xs = tc.zeros((1, rollout_steps + 1, 6), dtype=tc.float32)
    actions = tc.zeros((1, rollout_steps, action_dim), dtype=tc.float32)
    zs = tc.zeros((1, rollout_steps, latent_dim), dtype=tc.float32)
    latent_states = tc.zeros((rollout_steps + 1, latent_dim), dtype=tc.float32)

    latent_states[0] = initial_latent.detach().cpu().float()
    xs[0, 0] = tc.as_tensor(obs, dtype=tc.float32)

    diagnostics = []
    base_context = dict(controller_context or {})
    step_context = dict(base_context)
    step_context["measurement_index"] = measurement_index

    latent_step_params = model.dsr_sac_model.construct_params(
        cumulative_timeseries_index=cumulative_timeseries_index
    )
    hyperparameters = model.dsr_sac_model.hyperparameters

    current_latent = initial_latent.detach().cpu().numpy()
    for i in range(1, rollout_steps + 1):
        proposed_action_from_controller, info = controller.plan(
            latent_state=current_latent,
            context=step_context,
        )
        action = np.asarray(proposed_action_from_controller, dtype=np.float64)

        if action.shape != (action_dim,):
            raise ValueError(
                "controller.plan must return one flat action of shape "
                f"({action_dim},); got {action.shape}."
            )

        zs[0, i - 1] = tc.as_tensor(current_latent, dtype=tc.float32)
        actions[0, i - 1] = tc.as_tensor(action, dtype=tc.float32)

        next_latent = current_latent.copy()
        if apply_actions:
            next_latent = next_latent + np.pad(
                action,
                (0, max(0, latent_dim - action.shape[-1])),
            )

        with tc.no_grad():
            next_latent_tensor = model.dsr_sac_model.latent_step.forward(
                tc.as_tensor(next_latent[None, :], device=model.device).float(),
                tc.zeros((1, 1), device=model.device),
                latent_step_params,
                hyperparameters,
            )

        current_latent = next_latent_tensor[0].detach().cpu().numpy()
        latent_states[i] = tc.as_tensor(current_latent, dtype=tc.float32)
        diagnostics.append(info)

    with tc.no_grad():
        decoded_observations = model.decoder_model(
            latent_states.unsqueeze(0).to(model.device),
            measurement_index,
            expected=True,
        )

    return {
        "observations": decoded_observations.detach().cpu().float(),
        "latent_states": latent_states.unsqueeze(0),
        "actions": actions,
        "zs": zs,
        "diagnostics": diagnostics,
    }


if __name__ == "__main__":
    tc.set_num_threads(10)
    os.environ["OMP_NUM_THREADS"] = "10"
    alrnn_path = os.path.join(RESULTS_DIR,"/pendulum_sac_actuated/", 
                              "params_subjects=16_q_hid1=93_pol_hid1=73_steps=29_init_temp=1p65_polyak_tau=0p00_noise=0p18_q_lr=0p03_policy_lr=0p00_q_lr_gamma=1p00_policy_lr_gamma=1p00_aq=16p22_apolicy=116p45_q_hid2=85_pol_hid2=66/001")
    
    
    parameters = json.load(open(f"{alrnn_path}/config.json"))
    mtf_parameters = json.load(open(f"{parameters["world_model"]["path"]}/config.json"))
    dataset = init_dataset(mtf_parameters, tc.device("cpu"))
    model = MTF_SAC_Model(dataset, model_path=alrnn_path, device=tc.device("cpu"))
    MEASUREMENT_INDEX_TO_TS_INDEX = {15: 750,
                                    14: 700,
                                    13: 650,
                                    12: 600,
                                    11: 550,
                                    10: 500,
                                    9: 450,
                                    8: 400,
                                    7: 350,
                                    6: 300,
                                    5: 250,
                                    4: 200,
                                    3: 150,
                                    2: 100,
                                    1: 50,
                                    0: 0}
    def alrnn_step(
        z: Array,
        action: Array,
        context: Optional[Dict[str, Any]] = None,
    ) -> Array:
        cumulative_timeseries_index = MEASUREMENT_INDEX_TO_TS_INDEX[context["measurement_index"]]
        latent_step_params = model.dsr_sac_model.construct_params(cumulative_timeseries_index=cumulative_timeseries_index)
        z = z + np.pad(action, ((0, 0), (0, model.dsr_sac_model.hyperparameters["latent_dim"] - action.shape[-1])))
        z = model.dsr_sac_model.latent_step.forward(
                tc.tensor(z).float(), tc.tensor([[0]]), latent_step_params, model.dsr_sac_model.hyperparameters)

        return z.numpy()

    horizon = model.sac_args["env_steps_per_epoch"]
    num_candidates = 2048
    num_elites = 128
    
    latent_dim = model.dsr_sac_model.hyperparameters["latent_dim"]
    rollout_fn = make_alrnn_rollout_fn(alrnn_step)
    
    rewards_all = np.zeros((5, 16, 5))
    world_model_rewards_all = np.zeros((5, 16, 5))
    for seed in range(5):
        print(f"Seed {seed}")
        config = CEMMPCConfig(
        horizon=horizon,
        action_dim=6,
        num_candidates=num_candidates,
        num_elites=num_elites,
        num_iterations=6,
        init_std=0.35,
        action_low=-1.0,
        action_high=1.0,
        min_std=0.03,
        seed=seed,
    )
        for measurement_index in range(16):
            cumulative_timeseries_index = MEASUREMENT_INDEX_TO_TS_INDEX[measurement_index]
            projection_matrix = (model.dsr_sac_model.projection_matrix[measurement_index].weight).detach().numpy()


            controller = CEMMPC(
                config=config,
                rollout_fn=rollout_fn,
                reward_fn=model.reward_function,
                action_projector=make_nullspace_projector(projection_matrix)
            )
            world_model_controller = CEMMPC(
                config=config,
                rollout_fn=rollout_fn,
                reward_fn=model.reward_function,
                action_projector=make_nullspace_projector(projection_matrix)
            )

            current_latent_state = np.zeros(latent_dim)
            context = {
                "discount": 0.99,
                "measurement_index": measurement_index,
            }

            for k in range(5):
                theta1 = np.random.uniform(-np.pi, np.pi)
                theta2 = np.random.uniform(-np.pi, np.pi)
                x0 = np.concatenate([[np.sin(theta1), np.cos(theta1), np.sin(theta2), np.cos(theta2)], np.random.randn(2)]) 
                
                result = evaluate_cem_on_true_double_pendulum(
                controller=controller,
                model=model,
                x0=x0,
                measurement_index=measurement_index,
                observation_to_ode_state=observation_to_ode_state,
                ode_state_to_observation=ode_state_to_observation,
                double_pendulum_derivatives=double_pendulum_derivatives,
                pendulum_params=np.load(params_file, allow_pickle=True)[measurement_index],
                dt=0.01,
                T=500,
                controller_context=context,
                apply_actions=True,
                )
                # sanity checks
                #assert np.diff(result["actions"][0, :, -2] / result["actions"][0, :, -1]).max()<0.01
                assert (result["actions"][0, :, 0]==0).all()
                rewards_all[seed, measurement_index, k] =model.reward_function(tc.tensor(result["observations"][0][1:]).unsqueeze(0), tc.tensor(result["actions"][0]).unsqueeze(0), measurement_index).squeeze()[15:].mean()

                world_model_result = evaluate_cem_on_world_model(
                    controller=world_model_controller,
                    model=model,
                    x0=x0,
                    measurement_index=measurement_index,
                    cumulative_timeseries_index=cumulative_timeseries_index,
                    T=500,
                    controller_context=context,
                    apply_actions=True,
                )
                assert (world_model_result["actions"][0, :, 0] == 0).all()
                #assert np.diff(world_model_result["actions"][0, :, -2] / world_model_result["actions"][0, :, -1]).max()<0.01
                world_model_rewards_all[seed, measurement_index, k] = model.reward_function(
                    tc.tensor(world_model_result["observations"][0][1:]).unsqueeze(0),
                    tc.tensor(world_model_result["actions"][0]).unsqueeze(0),
                    measurement_index,
                ).squeeze()[15:].mean()
        
    np.save(RESULTS_PATH + "/pendulum_rewards_all.npy", rewards_all)
    np.save(RESULTS_PATH + "/pendulum_world_model_rewards_all.npy", world_model_rewards_all)