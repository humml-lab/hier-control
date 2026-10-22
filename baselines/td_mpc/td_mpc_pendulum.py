"""
TD-MPC for the true double-pendulum environment
================================================

This file implements a compact, single-task TD-MPC baseline that is designed to
be used analogously to `cem_mpc_pendulum.py`, but unlike CEM-MPC it learns from
online interaction with the TRUE double-pendulum ODE.

Important protocol distinction
------------------------------
- CEM-MPC in the companion code plans in a pre-trained ALRNN and never trains
  on the true ODE.
- This TD-MPC implementation collects true ODE transitions during training:
      (observation, executed_action, reward, next_observation)
  and trains its own task-oriented latent dynamics, reward model, value
  functions, and policy prior.
- At evaluation, TD-MPC plans only in its learned latent model. The true ODE is
  advanced only after TD-MPC has selected the first action.

Action semantics match the supplied CEM implementation:
    x[1] += action[4]  # angular velocity omega_1
    x[3] += action[5]  # angular velocity omega_2
followed by one `odeint` integration step.

The action projection is applied exactly in the same order as in your CEM code:
    1. clip the policy/planner proposal to action bounds;
    2. apply the projection matrix;
    3. use the projected 6D action in both the learned model and the true ODE.

This is a practical TD-MPC (original-style) implementation:
    - task-oriented latent dynamics;
    - reward predictor;
    - twin Q functions;
    - policy prior;
    - target encoder / target critics;
    - replay buffer of real ODE interaction;
    - MPPI latent-space planning with a terminal TD value estimate.

It is intentionally not a byte-for-byte reproduction of the official TD-MPC2
codebase. TD-MPC2 adds several additional design choices (e.g. SimNorm and
discretized reward/value regression) that are not necessary to establish a
well-controlled TD-MPC baseline for the six-dimensional pendulum setting.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import copy
import random

import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import odeint


Array = np.ndarray
RewardFunction = Callable[[Array, Array], float]
InitialObservationSampler = Callable[[np.random.Generator], Array]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TDMPCConfig:
    """Hyperparameters for one single-task TD-MPC agent."""

    observation_dim: int = 6
    action_dim: int = 6
    latent_dim: int = 64
    hidden_dim: int = 256

    # True-environment interaction and replay.
    replay_capacity: int = 200_000
    batch_size: int = 256
    seed_steps: int = 2_000
    total_environment_steps: int = 100_000
    updates_per_environment_step: int = 1

    # Original TD-MPC trains on replayed trajectory segments rather than
    # independently sampled one-step transitions.
    train_horizon: int = 30
    temporal_loss_discount: float = 0.5  # rho in the TD-MPC paper

    # TD-MPC objective.
    discount: float = 0.99
    tau: float = 0.01
    learning_rate: float = 3e-4
    grad_clip_norm: float = 20.0
    latent_consistency_weight: float = 1.0
    observation_reconstruction_weight: float = 1.0
    reward_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    actor_loss_weight: float = 1.0

    # Planning.
    planning_horizon: int = 5
    cem_iterations: int = 6
    num_candidates: int = 512
    num_elites: int = 64
    num_policy_trajectories: int = 64
    init_std: float = 0.5
    min_std: float = 0.05
    mean_momentum: float = 0.1
    std_momentum: float = 0.1
    # MPPI inverse-temperature used to weight imagined trajectory returns.
    # `temperature` is retained as the user-facing name.
    temperature: float = 0.5
    warm_start: bool = True

    # Action domain. Use scalar values for the same [-1, 1] proposal bounds
    # as the CEM baseline, unless your policy uses different limits.
    action_low: float = -1.0
    action_high: float = 1.0
    exploration_std: float = 0.2

    # Reproducibility and device.
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.action_dim != 6:
            raise ValueError(
                "This pendulum implementation expects a six-dimensional action "
                "because physical actions are read from entries 4 and 5."
            )
        if self.observation_dim != 6:
            raise ValueError(
                "This pendulum implementation expects the six-dimensional "
                "sin/cos/velocity observation representation."
            )
        if self.num_elites <= 0 or self.num_elites > self.num_candidates:
            raise ValueError("num_elites must be in [1, num_candidates].")
        if self.num_policy_trajectories > self.num_candidates:
            raise ValueError(
                "num_policy_trajectories cannot exceed num_candidates."
            )
        if self.planning_horizon <= 0:
            raise ValueError("planning_horizon must be positive.")
        if self.cem_iterations <= 0:
            raise ValueError("cem_iterations must be positive.")
        if self.train_horizon <= 0:
            raise ValueError("train_horizon must be positive.")
        if not 0.0 < self.temporal_loss_discount <= 1.0:
            raise ValueError("temporal_loss_discount must lie in (0, 1].")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must lie in (0, 1].")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must lie in (0, 1].")


# ---------------------------------------------------------------------------
# Utilities and replay data
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set NumPy, Python, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tc.manual_seed(seed)
    if tc.cuda.is_available():
        tc.cuda.manual_seed_all(seed)


class ReplayBuffer:
    """
    Episodic replay buffer for original-style TD-MPC training.

    TD-MPC trains its task-oriented latent dynamics model on short replayed
    trajectory segments. Each sampled batch contains

        o_{t:t+H}, a_{t:t+H-1}, r_{t:t+H-1}, not_done_{t:t+H-1},

    rather than unrelated one-step transitions. Actions stored here are the
    projected actions actually applied to the true ODE.
    """

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        seed: int,
    ) -> None:
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self._rng = np.random.default_rng(seed)

        self._episodes: List[Dict[str, List[Any]]] = []
        self._current: Dict[str, List[Any]] = self._empty_episode()
        self._num_transitions = 0

    def _empty_episode(self) -> Dict[str, List[Any]]:
        return {
            "observations": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "not_dones": [],
        }

    def __len__(self) -> int:
        return self._num_transitions

    def add(
        self,
        observation: Array,
        action: Array,
        reward: float,
        next_observation: Array,
        done: bool = False,
    ) -> None:
        self._current["observations"].append(
            np.asarray(observation, dtype=np.float32).copy()
        )
        self._current["actions"].append(np.asarray(action, dtype=np.float32).copy())
        self._current["rewards"].append(float(reward))
        self._current["next_observations"].append(
            np.asarray(next_observation, dtype=np.float32).copy()
        )
        self._current["not_dones"].append(0.0 if done else 1.0)
        self._num_transitions += 1

        if done:
            self.end_episode()

    def end_episode(self) -> None:
        """Finalize an episode, including nonterminal truncations."""
        length = len(self._current["actions"])
        if length == 0:
            return
        self._episodes.append(self._current)
        self._current = self._empty_episode()
        self._trim_to_capacity()

    def _trim_to_capacity(self) -> None:
        while self._episodes and self._num_transitions > self.capacity:
            oldest = self._episodes.pop(0)
            self._num_transitions -= len(oldest["actions"])

    def num_valid_segments(self, horizon: int) -> int:
        return sum(max(0, len(ep["actions"]) - horizon + 1) for ep in self._episodes)

    def sample_sequences(
        self,
        batch_size: int,
        horizon: int,
        device: tc.device,
    ) -> Tuple[tc.Tensor, tc.Tensor, tc.Tensor, tc.Tensor]:
        """
        Sample fixed-length replay sequences.

        Returns:
            observations: (B, H+1, observation_dim)
            actions:      (B, H, action_dim)
            rewards:      (B, H, 1)
            not_dones:    (B, H, 1)
        """
        valid = [
            ep for ep in self._episodes
            if len(ep["actions"]) >= horizon
        ]
        if not valid:
            raise RuntimeError(
                "No completed replay episode is long enough for train_horizon="
                f"{horizon}."
            )

        obs_batch, action_batch, reward_batch, not_done_batch = [], [], [], []
        for _ in range(batch_size):
            episode = valid[self._rng.integers(len(valid))]
            max_start = len(episode["actions"]) - horizon
            start = int(self._rng.integers(max_start + 1))
            stop = start + horizon

            observations = [episode["observations"][start]]
            observations.extend(episode["next_observations"][start:stop])

            obs_batch.append(np.stack(observations, axis=0))
            action_batch.append(np.stack(episode["actions"][start:stop], axis=0))
            reward_batch.append(
                np.asarray(episode["rewards"][start:stop], dtype=np.float32)[:, None]
            )
            not_done_batch.append(
                np.asarray(episode["not_dones"][start:stop], dtype=np.float32)[:, None]
            )

        return (
            tc.as_tensor(np.stack(obs_batch), device=device),
            tc.as_tensor(np.stack(action_batch), device=device),
            tc.as_tensor(np.stack(reward_batch), device=device),
            tc.as_tensor(np.stack(not_done_batch), device=device),
        )


# ---------------------------------------------------------------------------
# Action constraints: same logical order as the provided CEM implementation
# ---------------------------------------------------------------------------

class ProjectedActionSpace:
    """
    Applies the CEM-compatible action constraint:
        raw action -> clip to proposal bounds -> projection matrix.

    The projection is represented as a differentiable PyTorch operation so the
    policy prior can be optimized through it.

    `projection_matrix` must have shape (6, 6) in the supplied pendulum setup.
    Its action convention matches the CEM code:
        projected_actions = raw_actions @ projection_matrix.T
    """

    def __init__(
        self,
        action_low: float,
        action_high: float,
        projection_matrix: Optional[Array],
        device: tc.device,
    ) -> None:
        if action_low >= action_high:
            raise ValueError("action_low must be strictly less than action_high.")
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.device = device

        if projection_matrix is None:
            self.projection_matrix: Optional[tc.Tensor] = None
        else:
            projection = tc.as_tensor(
                np.asarray(projection_matrix, dtype=np.float32),
                dtype=tc.float32,
                device=device,
            )
            if projection.shape != (6, 6):
                raise ValueError(
                    "projection_matrix must have shape (6, 6) for this "
                    f"pendulum implementation; got {tuple(projection.shape)}."
                )
            self.projection_matrix = projection

    def apply_tensor(self, raw_action: tc.Tensor) -> tc.Tensor:
        """Project an action tensor of shape (..., 6) without breaking gradients."""
        bounded = raw_action.clamp(self.action_low, self.action_high)
        if self.projection_matrix is None:
            return bounded
        return bounded @ self.projection_matrix.T

    def apply_numpy(self, raw_action: Array) -> Array:
        """NumPy version for true-environment execution."""
        bounded = np.clip(
            np.asarray(raw_action, dtype=np.float64),
            self.action_low,
            self.action_high,
        )
        if self.projection_matrix is None:
            return bounded
        matrix = self.projection_matrix.detach().cpu().numpy()
        return bounded @ matrix.T


# ---------------------------------------------------------------------------
# True double-pendulum environment
# ---------------------------------------------------------------------------

class TrueDoublePendulumEnv:
    """
    Minimal online RL environment backed by the true double-pendulum ODE.

    The initial state is a 6D model observation. Internally, the environment
    converts it to a 4D physical ODE state and applies actions exactly like
    the CEM evaluator:
        physical_state[1] += action[4]
        physical_state[3] += action[5]
    before one `odeint` step.
    """

    def __init__(
        self,
        pendulum_params: Dict[str, float],
        observation_to_ode_state: Callable[[Array], Array],
        ode_state_to_observation: Callable[[Array], Array],
        double_pendulum_derivatives: Callable[..., Array],
        reward_fn: RewardFunction,
        action_space: ProjectedActionSpace,
        *,
        dt: float = 0.01,
        episode_length: int = 500,
    ) -> None:
        required = {"m1", "m2", "l1", "l2"}
        missing = required.difference(pendulum_params)
        if missing:
            raise KeyError(f"pendulum_params is missing: {sorted(missing)}")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if episode_length <= 0:
            raise ValueError("episode_length must be positive.")

        self.pendulum_params = pendulum_params
        self.observation_to_ode_state = observation_to_ode_state
        self.ode_state_to_observation = ode_state_to_observation
        self.double_pendulum_derivatives = double_pendulum_derivatives
        self.reward_fn = reward_fn
        self.action_space = action_space
        self.dt = float(dt)
        self.episode_length = int(episode_length)

        self._state: Optional[Array] = None
        self._observation: Optional[Array] = None
        self._time = 0.0
        self._step = 0

    @property
    def observation(self) -> Array:
        if self._observation is None:
            raise RuntimeError("Call reset() before accessing observation.")
        return self._observation.copy()

    @property
    def physical_state(self) -> Array:
        if self._state is None:
            raise RuntimeError("Call reset() before accessing physical_state.")
        return self._state.copy()

    def reset(self, initial_observation: Array) -> Array:
        obs = np.asarray(initial_observation, dtype=np.float64)
        if obs.shape == (1, 6):
            obs = obs[0]
        if obs.shape != (6,):
            raise ValueError(
                "initial_observation must have shape (6,) or (1, 6), "
                f"got {obs.shape}."
            )

        state = np.asarray(self.observation_to_ode_state(obs), dtype=np.float64)
        if state.shape != (4,):
            raise ValueError(
                "observation_to_ode_state must return a physical state with "
                f"shape (4,), got {state.shape}."
            )

        self._state = state
        self._observation = obs.copy()
        self._time = 0.0
        self._step = 0
        return self.observation

    def step(
        self,
        raw_action: Array,
    ) -> Tuple[Array, float, bool, Dict[str, Any]]:
        """
        Advance the true ODE exactly one step.

        Parameters
        ----------
        raw_action:
            Action proposal from the agent. It is clipped and projected before
            physical application, exactly as in the provided CEM controller.

        Returns
        -------
        next_observation, reward, done, info
        """
        if self._state is None or self._observation is None:
            raise RuntimeError("Call reset() before step().")

        executed_action = self.action_space.apply_numpy(raw_action)
        if executed_action.shape != (6,):
            raise ValueError(
                f"Expected a 6D action after projection, got {executed_action.shape}."
            )

        state = self._state.copy()
        state[1] += executed_action[4]  # Action on omega1
        state[3] += executed_action[5]  # Action on omega2

        next_state = odeint(
            self.double_pendulum_derivatives,
            state,
            [self._time, self._time + self.dt],
            args=(
                self.pendulum_params["m1"],
                self.pendulum_params["m2"],
                self.pendulum_params["l1"],
                self.pendulum_params["l2"],
                9.81,
            ),
        )[1]
        next_state = np.asarray(next_state, dtype=np.float64)

        numerical_failure = not np.all(np.isfinite(next_state))
        if numerical_failure:
            # Keep a finite signal in the replay buffer and terminate the episode.
            next_state = np.nan_to_num(next_state, nan=0.0, posinf=1e6, neginf=-1e6)

        next_observation = np.asarray(
            self.ode_state_to_observation(next_state),
            dtype=np.float64,
        )
        if next_observation.shape != (6,):
            raise ValueError(
                "ode_state_to_observation must return shape (6,), "
                f"got {next_observation.shape}."
            )

        reward = self.reward_fn(
                    next_observation,
                    executed_action,
                )
        reward = float(reward.detach().cpu())

        self._state = next_state
        self._observation = next_observation
        self._time += self.dt
        self._step += 1

        done = numerical_failure or self._step >= self.episode_length
        info = {
            "executed_action": executed_action.copy(),
            "physical_state": next_state.copy(),
            "numerical_failure": numerical_failure,
        }
        return next_observation.copy(), reward, done, info


def make_mtf_sac_reward_adapter(
    reference_model: Any,
    measurement_index: int,
) -> RewardFunction:
    """
    Create a scalar true-environment reward function from your existing
    `model.reward_function`.

    The adapter evaluates one *executed* observation/action pair. 

    It assumes:
        reference_model.reward_function(
            observations: (batch, time, 6),
            actions:      (batch, time, 6),
            measurement_index
        ) -> per-step rewards
    """
    device = next(reference_model.parameters()).device

    def reward_fn(observation, action):
        observation_tensor = tc.as_tensor(
            observation,
            dtype=tc.float32,
            device=device,
        ).reshape(1, 1, -1)

        action_tensor = tc.as_tensor(
            action,
            dtype=tc.float32,
            device=device,
        ).reshape(1, 1, -1)

        with tc.no_grad():
            reward = reference_model.reward_function(
                observation_tensor,
                action_tensor,
                measurement_index,
            )

        return reward.reshape(())

    def batch_reward_fn(observations, actions):
        """Evaluate a complete rollout with one reference-model call."""
        observation_tensor = tc.as_tensor(
            observations, dtype=tc.float32, device=device
        )
        action_tensor = tc.as_tensor(actions, dtype=tc.float32, device=device)
        if observation_tensor.ndim == 2:
            observation_tensor = observation_tensor.unsqueeze(0)
        if action_tensor.ndim == 2:
            action_tensor = action_tensor.unsqueeze(0)
        with tc.no_grad():
            rewards = reference_model.reward_function(
                observation_tensor, action_tensor, measurement_index
            )
        return rewards.detach().cpu().numpy().reshape(-1)

    # Keep the scalar callable used by the true ODE environment, while exposing
    # a vectorized evaluator for 500-step world-model rollouts.
    reward_fn.batch = batch_reward_fn  # type: ignore[attr-defined]
    return reward_fn


# ---------------------------------------------------------------------------
# TD-MPC neural components
# ---------------------------------------------------------------------------

def mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depth: int = 2,
) -> nn.Sequential:
    """Small MLP used for encoder-adjacent latent models."""
    if depth < 1:
        raise ValueError("depth must be >= 1.")

    layers: List[nn.Module] = []
    previous = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(previous, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Mish())
        previous = hidden_dim
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class TDMPCNetworks(nn.Module):
    """
    Task-oriented latent dynamics (TOLD) model plus actor and twin critics.

    Encoder:
        observation -> latent state

    Dynamics:
        (latent state, executed action) -> next latent state

    Reward model:
        (latent state, executed action) -> immediate true-environment reward

    Q-functions:
        (latent state, executed action) -> long-horizon return estimate

    Policy:
        latent state -> raw action proposal
        The action projection is applied outside this module so it is shared
        exactly between the planner, actor loss, replay data, and true ODE.
    """

    def __init__(self, config: TDMPCConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = mlp(
            config.observation_dim,
            config.latent_dim,
            config.hidden_dim,
        )
        self.dynamics = mlp(
            config.latent_dim + config.action_dim,
            config.latent_dim,
            config.hidden_dim,
        )
        self.decoder = mlp(
            config.latent_dim,
            config.observation_dim,
            config.hidden_dim,
        )
        self.reward = mlp(
            config.latent_dim + config.action_dim,
            1,
            config.hidden_dim,
        )
        self.q1 = mlp(
            config.latent_dim + config.action_dim,
            1,
            config.hidden_dim,
        )
        self.q2 = mlp(
            config.latent_dim + config.action_dim,
            1,
            config.hidden_dim,
        )
        self.policy = mlp(
            config.latent_dim,
            config.action_dim,
            config.hidden_dim,
        )

    def encode(self, observation: tc.Tensor) -> tc.Tensor:
        return self.encoder(observation)

    def decode(self, latent: tc.Tensor) -> tc.Tensor:
        """# The first four observation coordinates are two (sin, cos) pairs.
        # Project decoder outputs back onto S^1 x S^1 so imagined observations
        # remain physically meaningful during long world-model rollouts.
        observation = self.decoder(latent)
        pair1 = F.normalize(observation[..., 0:2], p=2, dim=-1, eps=1e-6)
        pair2 = F.normalize(observation[..., 2:4], p=2, dim=-1, eps=1e-6)
        return tc.cat([pair1, pair2, observation[..., 4:6]], dim=-1)"""
        return self.decoder(latent)

    def next_latent(self, latent: tc.Tensor, action: tc.Tensor) -> tc.Tensor:
        # A residual transition stabilizes short-horizon latent rollouts.
        return latent + self.dynamics(tc.cat([latent, action], dim=-1).float())

    def predict_reward(self, latent: tc.Tensor, action: tc.Tensor) -> tc.Tensor:
        return self.reward(tc.cat([latent, action], dim=-1).float())

    def q_values(self, latent: tc.Tensor, action: tc.Tensor) -> Tuple[tc.Tensor, tc.Tensor]:
        qa_input = tc.cat([latent, action], dim=-1)
        return self.q1(qa_input), self.q2(qa_input)

    def raw_policy_action(self, latent: tc.Tensor) -> tc.Tensor:
        # The action-space object clips this proposal to the bounds.
        return tc.tanh(self.policy(latent))


# ---------------------------------------------------------------------------
# TD-MPC agent: online training + latent-space MPC inference
# ---------------------------------------------------------------------------

class TDMPC:
    """
    Single-task TD-MPC agent.

    Training:
        - collect interaction data from the true pendulum ODE;
        - learn latent transition/reward/value/policy models from replay;
        - update target encoder and target Q networks by EMA.

    Inference:
        - use MPPI planning only in the learned latent model;
        - score short-horizon trajectories with learned rewards plus a terminal
          TD value estimate;
        - execute only the first selected action in the true ODE.
    """

    def __init__(
        self,
        config: TDMPCConfig,
        projection_matrix: Optional[Array] = None,
    ) -> None:
        self.config = config
        self.device = tc.device(config.device)
        set_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)

        self.action_space = ProjectedActionSpace(
            action_low=config.action_low,
            action_high=config.action_high,
            projection_matrix=projection_matrix,
            device=self.device,
        )

        self.networks = TDMPCNetworks(config).to(self.device)
        self.target_encoder = copy.deepcopy(self.networks.encoder).to(self.device)
        self.target_q1 = copy.deepcopy(self.networks.q1).to(self.device)
        self.target_q2 = copy.deepcopy(self.networks.q2).to(self.device)

        for target_module in (self.target_encoder, self.target_q1, self.target_q2):
            target_module.requires_grad_(False)

        self.model_optimizer = tc.optim.Adam(
            list(self.networks.encoder.parameters())
            + list(self.networks.dynamics.parameters())
            + list(self.networks.decoder.parameters())
            + list(self.networks.reward.parameters())
            + list(self.networks.q1.parameters())
            + list(self.networks.q2.parameters()),
            lr=config.learning_rate,
        )
        self.actor_optimizer = tc.optim.Adam(
            self.networks.policy.parameters(),
            lr=config.learning_rate,
        )

        self.replay = ReplayBuffer(
            capacity=config.replay_capacity,
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
            seed=config.seed,
        )

        # Receding-horizon MPPI warm-start statistics.
        self._proposal_mean = np.zeros(
            (config.planning_horizon, config.action_dim),
            dtype=np.float32,
        )
        self._proposal_std = np.full(
            (config.planning_horizon, config.action_dim),
            config.init_std,
            dtype=np.float32,
        )

        self.training_updates = 0

    def reset_planner(self) -> None:
        """Reset only temporary CEM planning statistics; this does not reset learning."""
        self._proposal_mean.fill(0.0)
        self._proposal_std.fill(self.config.init_std)

    @tc.no_grad()
    def encode_observation(self, observation: Array) -> tc.Tensor:
        """Encode one six-dimensional observation to a latent state."""
        obs = tc.as_tensor(
            np.asarray(observation, dtype=np.float32),
            device=self.device,
        ).view(1, self.config.observation_dim)
        return self.networks.encode(obs)

    @tc.no_grad()
    def policy_action(
        self,
        observation: Array,
        *,
        explore: bool = False,
    ) -> Array:
        """
        Return a *raw* policy proposal. The environment performs clipping and
        action projection exactly once before applying the ODE intervention.
        """
        latent = self.encode_observation(observation)
        raw = self.networks.raw_policy_action(latent)
        if explore:
            raw = raw + self.config.exploration_std * tc.randn_like(raw)
        return raw[0].detach().cpu().numpy().astype(np.float64)

    @tc.no_grad()
    def _policy_raw_actions_in_latent_rollout(
        self,
        initial_latent: tc.Tensor,
    ) -> tc.Tensor:
        """Draw policy-prior trajectories for MPPI candidate augmentation."""
        n = self.config.num_policy_trajectories
        horizon = self.config.planning_horizon
        if n == 0:
            return tc.empty((0, horizon, self.config.action_dim), device=self.device)

        z = initial_latent.expand(n, -1).clone()
        sequence: List[tc.Tensor] = []
        for _ in range(horizon):
            raw = self.networks.raw_policy_action(z)
            raw = raw + self.config.exploration_std * tc.randn_like(raw)
            executed = self.action_space.apply_tensor(raw)
            sequence.append(raw)
            z = self.networks.next_latent(z, executed)
        return tc.stack(sequence, dim=1)

    @tc.no_grad()
    def _estimate_action_sequence_return(
        self,
        initial_latent: tc.Tensor,
        raw_action_sequences: tc.Tensor,
    ) -> tc.Tensor:
        """
        Evaluate raw candidate sequences inside the learned TOLD model.

        Projection is applied here, so the imagined dynamics and reward model
        always receive the same executed action that the true ODE receives.
        """
        action_sequences = self.action_space.apply_tensor(raw_action_sequences)
        n, horizon, _ = action_sequences.shape
        z = initial_latent.expand(n, -1)
        total_return = tc.zeros(n, device=self.device)
        discount = 1.0

        for t in range(horizon):
            action_t = action_sequences[:, t]
            total_return += discount * self.networks.predict_reward(z, action_t).squeeze(-1)
            z = self.networks.next_latent(z, action_t)
            discount *= self.config.discount

        terminal_raw = self.networks.raw_policy_action(z)
        terminal_action = self.action_space.apply_tensor(terminal_raw)
        terminal_q1, terminal_q2 = self.networks.q_values(z, terminal_action)
        total_return += discount * tc.minimum(terminal_q1, terminal_q2).squeeze(-1)
        return total_return

    @tc.no_grad()
    def plan(
        self,
        observation: Array,
        *,
        deterministic: bool = True,
        diagnostics: bool = True,
    ) -> Tuple[Array, Dict[str, Any]]:
        """
        Original-style MPPI planning in the learned latent TOLD model.

        This deliberately replaces CEM elite refitting: each iteration weights
        all sampled candidate trajectories by exponentiated predicted return.
        `plan` returns an unprojected action proposal. `TrueDoublePendulumEnv`
        projects it once before both ODE execution and replay insertion.
        """
        initial_latent = self.encode_observation(observation)
        return self.plan_from_latent(
            initial_latent, deterministic=deterministic, diagnostics=diagnostics
        )

    @tc.no_grad()
    def plan_from_latent(
        self,
        initial_latent: tc.Tensor,
        *,
        deterministic: bool = True,
        diagnostics: bool = True,
    ) -> Tuple[Array, Dict[str, Any]]:
        """GPU-native MPPI planning from an already encoded latent state.

        Unlike the previous implementation, candidate generation, return
        weighting, mean/std updates, and warm-start state all remain on the
        PyTorch device. Only the selected first action and compact diagnostics
        are copied to CPU. This removes a GPU synchronization on every MPPI
        iteration.
        """
        if initial_latent.ndim == 1:
            initial_latent = initial_latent.unsqueeze(0)
        if initial_latent.shape != (1, self.config.latent_dim):
            raise ValueError(
                "initial_latent must have shape "
                f"(1, {self.config.latent_dim}) or ({self.config.latent_dim},); "
                f"got {tuple(initial_latent.shape)}."
            )

        cfg = self.config
        initial_latent = initial_latent.to(self.device)
        mean = tc.as_tensor(self._proposal_mean, device=self.device).clone()
        std = tc.as_tensor(self._proposal_std, device=self.device).clone()
        best_sequence = mean.clone()
        best_return = tc.tensor(-tc.inf, device=self.device)
        iteration_best_returns: List[float] = []
        iteration_mean_returns: List[float] = []

        random_count = cfg.num_candidates - cfg.num_policy_trajectories
        for _ in range(cfg.cem_iterations):
            if random_count > 0:
                noise = tc.randn(
                    random_count, cfg.planning_horizon, cfg.action_dim,
                    device=self.device,
                )
                random_raw_actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
                random_raw_actions.clamp_(cfg.action_low, cfg.action_high)
            else:
                random_raw_actions = tc.empty(
                    0, cfg.planning_horizon, cfg.action_dim, device=self.device
                )

            policy_raw_actions = self._policy_raw_actions_in_latent_rollout(
                initial_latent
            )
            raw_candidates = tc.cat(
                [policy_raw_actions, random_raw_actions], dim=0
            )
            returns = self._estimate_action_sequence_return(
                initial_latent, raw_candidates
            )
            finite = tc.isfinite(returns)

            if finite.any():
                safe_returns = tc.where(
                    finite, returns, tc.full_like(returns, -tc.inf)
                )
                current_best_return, current_best = safe_returns.max(dim=0)
                improve = current_best_return > best_return
                best_return = tc.where(improve, current_best_return, best_return)
                best_sequence = tc.where(
                    improve, raw_candidates[current_best], best_sequence
                )

                finite_returns = returns[finite]
                finite_candidates = raw_candidates[finite]
                weights = tc.softmax(
                    (finite_returns - finite_returns.max())
                    / max(cfg.temperature, 1e-8),
                    dim=0,
                )
                weighted_mean = tc.einsum(
                    "n,nha->ha", weights, finite_candidates
                )
                centered = finite_candidates - weighted_mean.unsqueeze(0)
                weighted_var = tc.einsum(
                    "n,nha,nha->ha", weights, centered, centered
                )
                weighted_std = weighted_var.clamp_min(
                    cfg.min_std ** 2
                ).sqrt()
            else:
                weighted_mean, weighted_std = mean, std

            mean = cfg.mean_momentum * mean + (1.0 - cfg.mean_momentum) * weighted_mean
            std = cfg.std_momentum * std + (1.0 - cfg.std_momentum) * weighted_std
            std.clamp_(min=cfg.min_std)

            if diagnostics:
                iteration_best_returns.append(float(returns.max().item()))
                iteration_mean_returns.append(float(returns.mean().item()))

        final_return = self._estimate_action_sequence_return(
            initial_latent, mean.unsqueeze(0)
        )[0]
        use_mean = final_return >= best_return
        selected = tc.where(use_mean, mean, best_sequence)
        selected_return = tc.where(use_mean, final_return, best_return)

        if cfg.warm_start:
            shifted_mean = tc.cat([selected[1:], selected[-1:]], dim=0)
            shifted_std = tc.cat([std[1:], std[-1:]], dim=0)
            self._proposal_mean[...] = shifted_mean.cpu().numpy()
            self._proposal_std[...] = shifted_std.cpu().numpy()
        else:
            self.reset_planner()

        # Only the first action must cross the device boundary during training.
        first_action = selected[0].cpu().numpy().copy()
        if not diagnostics:
            return first_action, {}

        info = {
            "best_return": float(selected_return.item()),
            "mean_return": iteration_mean_returns[-1],
            "proposal_mean": mean.cpu().numpy(),
            "proposal_std": std.cpu().numpy(),
            "best_raw_action_sequence": selected.cpu().numpy(),
            "iteration_best_returns": np.asarray(iteration_best_returns),
            "iteration_mean_returns": np.asarray(iteration_mean_returns),
            "planner": "MPPI_GPU",
            "deterministic": deterministic,
        }
        return first_action, info

    def update(self) -> Dict[str, float]:
        """
        Original-style TD-MPC update over replayed trajectory segments.

        A target encoder supplies latent targets at all future replay steps.
        The online latent dynamics is unrolled recurrently over `train_horizon`.
        At each imagined step we jointly optimize latent consistency, reward
        prediction, and a one-step TD value target. Per-step losses are
        geometrically weighted by `temporal_loss_discount` (rho).
        """
        cfg = self.config
        observations, actions, rewards, not_dones = self.replay.sample_sequences(
            cfg.batch_size,
            cfg.train_horizon,
            self.device,
        )

        with tc.no_grad():
            target_latents = self.target_encoder(
                observations[:, 1:].reshape(-1, cfg.observation_dim)
            ).view(cfg.batch_size, cfg.train_horizon, cfg.latent_dim)

        latent = self.networks.encode(observations[:, 0])
        model_loss = tc.zeros((), device=self.device)
        latent_loss_total = tc.zeros((), device=self.device)
        observation_loss_total = tc.zeros((), device=self.device)
        reward_loss_total = tc.zeros((), device=self.device)
        q_loss_total = tc.zeros((), device=self.device)
        loss_weight_sum = 0.0

        for t in range(cfg.train_horizon):
            action_t = actions[:, t]  # projected / physically executed action
            reward_t = rewards[:, t]
            not_done_t = not_dones[:, t]
            target_next_latent = target_latents[:, t]

            predicted_reward = self.networks.predict_reward(latent, action_t)
            q1, q2 = self.networks.q_values(latent, action_t)
            predicted_next_latent = self.networks.next_latent(latent, action_t)
            predicted_next_observation = self.networks.decode(predicted_next_latent)

            with tc.no_grad():
                next_raw_action = self.networks.raw_policy_action(target_next_latent)
                next_action = self.action_space.apply_tensor(next_raw_action)
                target_q1 = self.target_q1(
                    tc.cat([target_next_latent, next_action], dim=-1)
                )
                target_q2 = self.target_q2(
                    tc.cat([target_next_latent, next_action], dim=-1)
                )
                td_target = reward_t + cfg.discount * not_done_t * tc.minimum(
                    target_q1,
                    target_q2,
                )

            latent_loss = F.mse_loss(predicted_next_latent, target_next_latent)
            observation_loss = F.mse_loss(
                predicted_next_observation,
                observations[:, t + 1],
            )
            reward_loss = F.mse_loss(predicted_reward, reward_t)
            q_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
            step_loss = (
                cfg.latent_consistency_weight * latent_loss
                + cfg.observation_reconstruction_weight * observation_loss
                + cfg.reward_loss_weight * reward_loss
                + cfg.value_loss_weight * q_loss
            )

            weight = cfg.temporal_loss_discount ** t
            model_loss = model_loss + weight * step_loss
            latent_loss_total = latent_loss_total + weight * latent_loss
            observation_loss_total = observation_loss_total + weight * observation_loss
            reward_loss_total = reward_loss_total + weight * reward_loss
            q_loss_total = q_loss_total + weight * q_loss
            loss_weight_sum += weight

            # Recurrent unroll through the learned dynamics is the key
            # difference to the previous one-step implementation.
            latent = predicted_next_latent

        model_loss = model_loss / loss_weight_sum
        latent_loss_total = latent_loss_total / loss_weight_sum
        observation_loss_total = observation_loss_total / loss_weight_sum
        reward_loss_total = reward_loss_total / loss_weight_sum
        q_loss_total = q_loss_total / loss_weight_sum

        self.model_optimizer.zero_grad(set_to_none=True)
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.networks.encoder.parameters())
            + list(self.networks.dynamics.parameters())
            + list(self.networks.decoder.parameters())
            + list(self.networks.reward.parameters())
            + list(self.networks.q1.parameters())
            + list(self.networks.q2.parameters()),
            cfg.grad_clip_norm,
        )
        self.model_optimizer.step()

        # Policy-prior update uses state representations from real replay
        # observations. The policy is then used to guide MPPI sampling.
        latent_for_actor = self.networks.encode(
            observations[:, 0]
        ).detach()
        actor_raw_action = self.networks.raw_policy_action(latent_for_actor)
        actor_action = self.action_space.apply_tensor(actor_raw_action)
        actor_q1, actor_q2 = self.networks.q_values(latent_for_actor, actor_action)
        actor_loss = -tc.minimum(actor_q1, actor_q2).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        (cfg.actor_loss_weight * actor_loss).backward()
        nn.utils.clip_grad_norm_(self.networks.policy.parameters(), cfg.grad_clip_norm)
        self.actor_optimizer.step()

        self._soft_update_targets()
        self.training_updates += 1

        return {
            "model_loss": float(model_loss.detach().cpu().item()),
            "latent_loss": float(latent_loss_total.detach().cpu().item()),
            "observation_loss": float(observation_loss_total.detach().cpu().item()),
            "reward_loss": float(reward_loss_total.detach().cpu().item()),
            "q_loss": float(q_loss_total.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "mean_reward_batch": float(rewards.mean().detach().cpu().item()),
        }

    @tc.no_grad()
    def _soft_update_targets(self) -> None:
        tau = self.config.tau
        for online, target in (
            (self.networks.encoder, self.target_encoder),
            (self.networks.q1, self.target_q1),
            (self.networks.q2, self.target_q2),
        ):
            for online_parameter, target_parameter in zip(
                online.parameters(),
                target.parameters(),
            ):
                target_parameter.lerp_(online_parameter, tau)

    def save(self, path: str | Path) -> None:
        """Save all learned TD-MPC model weights and optimizer state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tc.save(
            {
                "config": asdict(self.config),
                "networks": self.networks.state_dict(),
                "target_encoder": self.target_encoder.state_dict(),
                "target_q1": self.target_q1.state_dict(),
                "target_q2": self.target_q2.state_dict(),
                "model_optimizer": self.model_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "training_updates": self.training_updates,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load previously trained TD-MPC weights."""
        checkpoint = tc.load(path, map_location=self.device, weights_only=False)
        self.networks.load_state_dict(checkpoint["networks"])
        self.target_encoder.load_state_dict(checkpoint["target_encoder"])
        self.target_q1.load_state_dict(checkpoint["target_q1"])
        self.target_q2.load_state_dict(checkpoint["target_q2"])
        self.model_optimizer.load_state_dict(checkpoint["model_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.training_updates = int(checkpoint.get("training_updates", 0))


# ---------------------------------------------------------------------------
# Online collection / training and true-system evaluation
# ---------------------------------------------------------------------------

def default_initial_observation_sampler(rng: np.random.Generator) -> Array:
    """
    Match the initialization distribution used in the provided CEM script.

    Observation layout is left to the project's conversion functions; this
    assumes the current project convention:
        [sin(theta1), cos(theta1), sin(theta2), cos(theta2), omega1, omega2]
    """
    theta1 = rng.uniform(-np.pi, np.pi)
    theta2 = rng.uniform(-np.pi, np.pi)
    return np.array(
        [
            np.sin(theta1),
            np.cos(theta1),
            np.sin(theta2),
            np.cos(theta2),
            rng.normal(),
            rng.normal(),
        ],
        dtype=np.float64,
    )


def train_tdmpc_on_true_double_pendulum(
    agent: TDMPC,
    environment: TrueDoublePendulumEnv,
    initial_observation_sampler: InitialObservationSampler = default_initial_observation_sampler,
    *,
    total_environment_steps: Optional[int] = None,
    log_every: int = 5_000,
) -> Dict[str, List[float]]:
    """
    Train TD-MPC by interacting with the true ODE environment.

    Data-collection policy:
      - random projected actions for `seed_steps`;
      - then TD-MPC MPPI planning in the learned latent model;
      - the selected first action is applied to the true ODE;
      - transition is stored and one or more gradient updates are performed.

    The planner does not use the true ODE to score its candidates. It sees true
    dynamics only through replay data accumulated from already executed actions.
    """
    cfg = agent.config
    total_steps = total_environment_steps or cfg.total_environment_steps
    if total_steps <= 0:
        raise ValueError("total_environment_steps must be positive.")

    rng = np.random.default_rng(cfg.seed + 10_000)
    episode_return = 0.0
    episode_length = 0
    observation = environment.reset(initial_observation_sampler(rng))
    agent.reset_planner()

    logs: Dict[str, List[float]] = {
        "step": [],
        "episode_return": [],
        "episode_length": [],
        "model_loss": [],
        "latent_loss": [],
        "observation_loss": [],
        "reward_loss": [],
        "q_loss": [],
        "actor_loss": [],
    }

    latest_update: Optional[Dict[str, float]] = None

    for step in range(total_steps):
        if step < cfg.seed_steps:
            raw_action = rng.uniform(
                cfg.action_low,
                cfg.action_high,
                size=(cfg.action_dim,),
            )
        else:
            raw_action, _ = agent.plan(observation, diagnostics=False)

        next_observation, reward, done, info = environment.step(raw_action)
        executed_action = info["executed_action"]

        agent.replay.add(
            observation=observation,
            action=executed_action,
            reward=reward,
            next_observation=next_observation,
            done=done,
        )

        observation = next_observation
        episode_return += reward
        episode_length += 1

        # Original TD-MPC samples trajectory segments. Start updates only when
        # at least one completed episode can provide train_horizon segments.
        if (
            agent.replay.num_valid_segments(cfg.train_horizon) >= cfg.batch_size
        ):
            for _ in range(cfg.updates_per_environment_step):
                latest_update = agent.update()

        if done:
            logs["step"].append(float(step + 1))
            logs["episode_return"].append(float(episode_return))
            logs["episode_length"].append(float(episode_length))
            if latest_update is not None:
                for key in (
                    "model_loss",
                    "latent_loss",
                    "observation_loss",
                    "reward_loss",
                    "q_loss",
                    "actor_loss",
                ):
                    logs[key].append(latest_update[key])

            observation = environment.reset(initial_observation_sampler(rng))
            agent.reset_planner()
            episode_return = 0.0
            episode_length = 0

        if log_every > 0 and (step + 1) % log_every == 0:
            mean_recent_return = (
                float(np.mean(logs["episode_return"][-10:]))
                if logs["episode_return"]
                else float("nan")
            )
            print(
                f"[TD-MPC] step={step + 1:>7d} "
                f"replay={len(agent.replay):>7d} "
                f"mean_return(last 10)={mean_recent_return: .4f}"
            )

    # Preserve a final truncated episode for potential later continuation /
    # diagnostic use; it is not required for the completed training run.
    agent.replay.end_episode()
    return logs


@tc.no_grad()
def evaluate_tdmpc_on_true_double_pendulum(
    agent: TDMPC,
    environment: TrueDoublePendulumEnv,
    x0: Array,
    *,
    T: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained TD-MPC agent on the true pendulum in closed loop.

    This is analogous to `evaluate_cem_on_true_double_pendulum`:
        true observation -> TD-MPC plan using learned latent model only
        -> execute one action in true ODE -> replan from next observation.

    No learning occurs here. The returned fields mirror the CEM evaluator:
        observations: (1, T+1, 6)
        actions:      (1, T, 6), projected actions actually applied
        zs:           (1, T, latent_dim)
        ode_states:   (T+1, 4)
        rewards:      (T,)
        diagnostics:  list of per-step planner diagnostics
    """
    steps = T or environment.episode_length
    if steps <= 0:
        raise ValueError("T must be positive.")

    observation = environment.reset(x0)
    agent.reset_planner()

    observations = tc.zeros((1, steps + 1, 6), dtype=tc.float32)
    actions = tc.zeros((1, steps, 6), dtype=tc.float32)
    zs = tc.zeros((1, steps, agent.config.latent_dim), dtype=tc.float32)
    ode_states = np.zeros((steps + 1, 4), dtype=np.float64)
    rewards = np.zeros(steps, dtype=np.float64)
    diagnostics: List[Dict[str, Any]] = []

    observations[0, 0] = tc.as_tensor(observation, dtype=tc.float32)
    ode_states[0] = environment.physical_state

    for t in range(steps):
        latent = agent.encode_observation(observation)
        action, info = agent.plan(observation)
        next_observation, reward, done, env_info = environment.step(action)

        observations[0, t + 1] = tc.as_tensor(
            next_observation,
            dtype=tc.float32,
        )
        actions[0, t] = tc.as_tensor(
            env_info["executed_action"],
            dtype=tc.float32,
        )
        zs[0, t] = latent[0].detach().cpu()
        ode_states[t + 1] = environment.physical_state
        rewards[t] = reward
        diagnostics.append(info)

        observation = next_observation
        if done:
            # Preserve array shapes: repeat terminal state, and report the early
            # termination in diagnostics rather than silently reinitializing.
            for remaining in range(t + 1, steps):
                observations[0, remaining + 1] = tc.as_tensor(
                    observation,
                    dtype=tc.float32,
                )
                ode_states[remaining + 1] = environment.physical_state
            break

    return {
        "observations": observations,
        "actions": actions,
        "zs": zs,
        "ode_states": ode_states,
        "rewards": rewards,
        "diagnostics": diagnostics,
    }


@tc.no_grad()
def evaluate_tdmpc_on_world_model_double_pendulum(
    agent: TDMPC,
    environment: TrueDoublePendulumEnv,
    x0: Array,
    *,
    T: int,
) -> Dict[str, Any]:
    """
    Evaluate a closed-loop rollout entirely inside the learned world model.

    The learned latent dynamics generates the next latent state, the decoder
    generates the next observation, and the reference reward function scores
    that decoded observation/action pair. The true ODE is not advanced.
    """
    if T <= 0:
        raise ValueError("T must be positive.")

    obs = np.asarray(x0, dtype=np.float64)
    if obs.shape == (1, 6):
        obs = obs[0]
    if obs.shape != (6,):
        raise ValueError(
            f"x0 must have shape (6,) or (1, 6), got {obs.shape}."
        )

    agent.reset_planner()

    observations = tc.zeros((1, T + 1, 6), dtype=tc.float32)
    actions = tc.zeros((1, T, 6), dtype=tc.float32)
    latent_states = tc.zeros((1, T + 1, agent.config.latent_dim), dtype=tc.float32)
    rewards = np.zeros(T, dtype=np.float64)
    diagnostics: List[Dict[str, Any]] = []

    observations[0, 0] = tc.as_tensor(obs, dtype=tc.float32)
    latent = agent.encode_observation(obs)
    latent_states[0, 0] = latent[0].detach().cpu().float()

    for t in range(T):
        raw_action, info = agent.plan_from_latent(latent)
        executed_action = agent.action_space.apply_numpy(raw_action)
        action_tensor = tc.as_tensor(
            executed_action,
            dtype=tc.float32,
            device=agent.device,
        ).view(1, -1)

        latent = agent.networks.next_latent(latent, action_tensor)
        next_observation = agent.networks.decode(latent)
        next_observation_array = next_observation[0].detach().cpu().numpy()
        observations[0, t + 1] = tc.as_tensor(
            next_observation_array,
            dtype=tc.float32,
        )
        actions[0, t] = action_tensor[0].detach().cpu().float()
        latent_states[0, t + 1] = latent[0].detach().cpu().float()
        diagnostics.append(info)

    # Evaluate the complete 500-step imagined rollout in one call to exactly
    # the same reference_model.reward_function used for true-environment scoring.
    batch_reward = getattr(environment.reward_fn, "batch", None)
    if batch_reward is None:
        raise RuntimeError(
            "World-model evaluation requires the vectorized reference reward adapter."
        )
    rewards[:] = batch_reward(
        observations[:, 1:].numpy(),
        actions.numpy(),
    )

    return {
        "observations": observations,
        "latent_states": latent_states,
        "actions": actions,
        "rewards": rewards,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Project-specific assembly template
# ---------------------------------------------------------------------------

def projection_matrix_for_measurement(reference_model: Any, measurement_index: int) -> Array:
    """
    Extract the same 6x6 action projection used by the CEM script.

    This assumes your project structure:
        model.dsr_sac_model.projection_matrix[measurement_index][0:3].weight
    """
    layers = reference_model.dsr_sac_model.projection_matrix[measurement_index]
    return (layers.weight).detach().cpu().numpy()


def build_tdmpc_pendulum_components(
    *,
    reference_model: Any,
    measurement_index: int,
    pendulum_params: Dict[str, float],
    observation_to_ode_state: Callable[[Array], Array],
    ode_state_to_observation: Callable[[Array], Array],
    double_pendulum_derivatives: Callable[..., Array],
    config: TDMPCConfig,
    dt: float = 0.01,
    episode_length: int = 500,
) -> Tuple[TDMPC, TrueDoublePendulumEnv]:
    """
    One-call construction for the subject/session indexed by measurement_index.

    Example:
        agent, env = build_tdmpc_pendulum_components(
            reference_model=model,
            measurement_index=measurement_index,
            pendulum_params=params[measurement_index],
            observation_to_ode_state=observation_to_ode_state,
            ode_state_to_observation=ode_state_to_observation,
            double_pendulum_derivatives=double_pendulum_derivatives,
            config=TDMPCConfig(seed=seed, total_environment_steps=100_000),
        )
        logs = train_tdmpc_on_true_double_pendulum(agent, env)
        result = evaluate_tdmpc_on_true_double_pendulum(agent, env, x0, T=500)
    """
    projection_matrix = projection_matrix_for_measurement(
        reference_model,
        measurement_index,
    )
    reward_fn = make_mtf_sac_reward_adapter(
        reference_model,
        measurement_index,
    )

    agent = TDMPC(
        config=config,
        projection_matrix=projection_matrix,
    )
    environment = TrueDoublePendulumEnv(
        pendulum_params=pendulum_params,
        observation_to_ode_state=observation_to_ode_state,
        ode_state_to_observation=ode_state_to_observation,
        double_pendulum_derivatives=double_pendulum_derivatives,
        reward_fn=reward_fn,
        action_space=agent.action_space,
        dt=dt,
        episode_length=episode_length,
    )
    return agent, environment


if __name__ == "__main__":
    import json
    import os
    import sys

    tc.set_num_threads(2)
    sys.path.insert(
        0,
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."),
    )

    from src.models.mtf_sac_model import MTF_SAC_Model
    from src.main import init_dataset
    from src.environments.pendulum import (
        ode_state_to_observation,
        double_pendulum_derivatives,
        observation_to_ode_state,
    )
    from src.paths import get_results_dir
    RESULTS_DIR, DATA_DIR = get_results_dir()
    RESULTS_PATH = os.path.join(RESULTS_DIR, "baselines", "td_mpc")

    params_file = (DATA_DIR + "/multi_task_double_pendulum/default/params.npy"
    )
    alrnn_path = os.path.join(RESULTS_DIR, "pendulum_sac_actuated/", 
                              "params_subjects=16_q_hid1=93_pol_hid1=73_steps=29_init_temp=1p65_polyak_tau=0p00_noise=0p18_q_lr=0p03_policy_lr=0p00_q_lr_gamma=1p00_policy_lr_gamma=1p00_aq=16p22_apolicy=116p45_q_hid2=85_pol_hid2=66/001")
    
    

    # -----------------------------------------------------------------------
    # Load the existing model solely to reuse the same task reward function and
    # the same per-subject action projection matrices as CEM-MPC.
    # TD-MPC itself is trained from the TRUE ODE, not from this ALRNN.
    # -----------------------------------------------------------------------
    parameters = json.load(open(f"{alrnn_path}/config.json"))
    mtf_parameters = json.load(open(f"{parameters["world_model"]["path"]}/config.json"))
    dataset = init_dataset(mtf_parameters, tc.device("cpu"))
    reference_model = MTF_SAC_Model(dataset, model_path=alrnn_path, device=tc.device("cpu"))
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

    TEST_HORIZON = 500
    DT = 0.01
    TOTAL_ENVIRONMENT_STEPS = 50_000

    # TD-MPC planning budget. It should be reported alongside the CEM-MPC
    # planner settings in the paper.
    # Original TD-MPC uses short-horizon local planning, complemented by a
    # learned terminal value estimate.
    PLANNING_HORIZON = 5
    TRAIN_HORIZON = 10
    NUM_CANDIDATES = 128
    NUM_ELITES = 64
    NUM_POLICY_TRAJECTORIES = 16
    CEM_ITERATIONS = 3
    LATENT_DIM = 64
    HIDDEN_DIM = 128

    # Persist learned online agents and results separately from the existing
    # CEM output, so no file is overwritten.
    output_dir = Path(RESULTS_PATH + "/td_mpc_pendulum_results")
    os.makedirs(output_dir, exist_ok=True)

    all_pendulum_params = np.load(params_file, allow_pickle=True)

    # Same result layout as the CEM script:
    #    (seed, measurement_index / subject, test initial condition)
    rewards_all = np.full(
        (5, 16, 5),
        np.nan,
        dtype=np.float64,
    )
    world_model_rewards_all = np.full(
        (5, 16, 5),
        np.nan,
        dtype=np.float64,
    )

    for seed in range(5):
        print(f"\n========== TD-MPC seed {seed} ==========")

        for measurement_index in range(16):
            print(
                f"\n--- Training subject / measurement index "
                f"{measurement_index} (seed {seed}) ---"
            )

            config = TDMPCConfig(
                seed=seed * 10_000 + measurement_index,
                device="cuda:0",
                total_environment_steps=TOTAL_ENVIRONMENT_STEPS,
                planning_horizon=PLANNING_HORIZON,
                train_horizon=TRAIN_HORIZON,
                temporal_loss_discount=0.9,
                num_candidates=NUM_CANDIDATES,
                num_elites=NUM_ELITES,
                num_policy_trajectories=NUM_POLICY_TRAJECTORIES,
                latent_dim = LATENT_DIM,
                hidden_dim = HIDDEN_DIM,
                cem_iterations=CEM_ITERATIONS,
                latent_consistency_weight=1.0,
                action_low=-1.0,
                action_high=1.0,
                init_std=0.35,
                min_std=0.03,
                discount=0.99,
            )

            # `episode_length` is an environment argument rather than a
            # TDMPCConfig field. It is passed explicitly below.
            agent, environment = build_tdmpc_pendulum_components(
                reference_model=reference_model,
                measurement_index=measurement_index,
                pendulum_params=all_pendulum_params[measurement_index],
                observation_to_ode_state=observation_to_ode_state,
                ode_state_to_observation=ode_state_to_observation,
                double_pendulum_derivatives=double_pendulum_derivatives,
                config=config,
                dt=DT,
                episode_length=500,
            )

            # ---------------------------------------------------------------
            # Online TD-MPC learning from true ODE interaction.
            # ---------------------------------------------------------------
            training_logs = train_tdmpc_on_true_double_pendulum(
                agent=agent,
                environment=environment,
                total_environment_steps=TOTAL_ENVIRONMENT_STEPS,
                log_every=5_000,
            )

            model_file = os.path.join(
                output_dir,
                f"tdmpc_seed={seed}_measurement_index={measurement_index}.pt",
            )
            agent.save(model_file)

            np.savez(
                os.path.join(
                    output_dir,
                    f"tdmpc_training_seed={seed}_measurement_index="
                    f"{measurement_index}.npz",
                ),
                **{
                    key: np.asarray(value, dtype=np.float64)
                    for key, value in training_logs.items()
                },
            )

            # ---------------------------------------------------------------
            # Test on the true ODE, analogously to the CEM-MPC test loop.
            # No learning happens in evaluate_tdmpc_on_true_double_pendulum.
            # ---------------------------------------------------------------
            evaluation_rng = np.random.default_rng(
                1_000_000 + seed * 10_000 + measurement_index
            )

            for k in range(5):
                theta1 = evaluation_rng.uniform(-np.pi, np.pi)
                theta2 = evaluation_rng.uniform(-np.pi, np.pi)
                x0 = np.concatenate(
                    [
                        [
                            np.sin(theta1),
                            np.cos(theta1),
                            np.sin(theta2),
                            np.cos(theta2),
                        ],
                        evaluation_rng.normal(size=2),
                    ]
                ).astype(np.float64)

                result = evaluate_tdmpc_on_true_double_pendulum(
                    agent=agent,
                    environment=environment,
                    x0=x0,
                    T=TEST_HORIZON,
                )

                # Same projection-related sanity checks as in CEM-MPC:
                # - action dimensions 0 and 1 should be eliminated by your
                #   subject-specific projection;
                # - dimensions 4 and 5 should remain on the learned 1-D
                #   controlled subspace. Avoid a direct division by zero.
                applied_actions = result["actions"][0].detach().cpu().numpy()

                assert np.allclose(
                    applied_actions[:, 0],
                    0.0,
                    atol=1e-6,
                ), "Projected TD-MPC action has a non-zero first component."

                denominator = applied_actions[:, -1]
                nonzero = np.abs(denominator) > 1e-8
                """if np.any(nonzero):
                    ratio = (
                        applied_actions[nonzero, -2]
                        / denominator[nonzero]
                    )
                    assert np.max(np.abs(np.diff(ratio))) < 0.01, (
                        "The projected TD-MPC velocity actions do not retain "
                        "the expected fixed one-dimensional ratio."
                    )"""

                # Match the CEM result metric: mean reward after the initial
                # 15 steps, evaluated under the same reference reward function.
                # Use the subject's measurement_index consistently.
                observations = result["observations"][:, 1:]
                actions = result["actions"]
                with tc.no_grad():
                    per_step_reward = reference_model.reward_function(
                        observations,
                        actions,
                        measurement_index,
                    ).squeeze()

                rewards_all[seed, measurement_index, k] = float(
                    per_step_reward[15:].mean().detach().cpu().item()
                )

                print(
                    f"test rollout {k + 1}/{5}: "
                    f"mean reward after burn-in = "
                    f"{rewards_all[seed, measurement_index, k]: .6f}"
                )

                world_model_result = (
                    evaluate_tdmpc_on_world_model_double_pendulum(
                        agent=agent,
                        environment=environment,
                        x0=x0,
                        T=TEST_HORIZON,
                    )
                )
                world_model_actions = (
                    world_model_result["actions"][0].detach().cpu().numpy()
                )
                assert np.allclose(
                    world_model_actions[:, 0],
                    0.0,
                    atol=1e-6,
                ), (
                    "Projected world-model TD-MPC action has a non-zero first "
                    "component."
                )
                denominator = world_model_actions[:, -1]
                nonzero = np.abs(denominator) > 1e-8
                """if np.any(nonzero):
                    ratio = world_model_actions[nonzero, -2] / denominator[nonzero]
                    assert np.max(np.abs(np.diff(ratio))) < 0.01, (
                        "World-model projected TD-MPC velocity actions do not "
                        "retain the expected fixed one-dimensional ratio."
                    )"""

                # `rewards` was computed in one batched call to the same
                # reference_model.reward_function. Keep the identical 15-step
                # burn-in used for the true-environment score.
                world_model_rewards_all[seed, measurement_index, k] = float(
                    np.mean(world_model_result["rewards"][15:])
                )

                print(
                    f"world model rollout {k + 1}/{5}: "
                    f"mean reward after burn-in = "
                    f"{world_model_rewards_all[seed, measurement_index, k]: .6f}"
                )

            # Save progress after every trained subject so long experimental
            # runs can be resumed or inspected without losing prior outcomes.
            np.save(
                os.path.join(output_dir, "td_mpc_pendulum_rewards_all.npy"),
                rewards_all,
            )
            np.save(
                os.path.join(output_dir, "td_mpc_pendulum_world_model_rewards_all.npy"),
                world_model_rewards_all,
            )

    np.save(
        os.path.join(output_dir, "td_mpc_pendulum_rewards_all.npy"),
        rewards_all,
    )
    np.save(
        os.path.join(output_dir, "td_mpc_pendulum_world_model_rewards_all.npy"),
        world_model_rewards_all,
    )
    print("\nTD-MPC experiment completed.")
    print(
        "Saved aggregate rewards to "
        f"{os.path.join(output_dir, 'td_mpc_pendulum_rewards_all.npy')}"
    )
    print(
        "Saved world-model aggregate rewards to "
        f"{os.path.join(output_dir, 'td_mpc_pendulum_world_model_rewards_all.npy')}"
    )
