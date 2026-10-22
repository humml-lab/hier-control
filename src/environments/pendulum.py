import torch as tc
import numpy as np
from scipy.integrate import odeint
# Forward-declare MTF_SAC_Model for type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.models.mtf_sac_model import MTF_SAC_Model
    
from src.paths import get_results_dir
RESULTS_DIR, DATA_DIR = get_results_dir()

def double_pendulum_derivatives(y, t, m1, m2, l1, l2, g=9.81):
    theta1, omega1, theta2, omega2 = y
    dtheta1_dt = omega1
    dtheta2_dt = omega2
    
    delta = theta2 - theta1
    den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta) * np.cos(delta)
    den2 = (l2 / l1) * den1
    
    domega1_dt = (
        m2 * l1 * omega1 * omega1 * np.sin(delta) * np.cos(delta)
        + m2 * g * np.sin(theta2) * np.cos(delta)
        + m2 * l2 * omega2 * omega2 * np.sin(delta)
        - (m1 + m2) * g * np.sin(theta1)
    ) / den1
    
    domega2_dt = (
        -m2 * l2 * omega2 * omega2 * np.sin(delta) * np.cos(delta)
        + (m1 + m2) * g * np.sin(theta1) * np.cos(delta)
        - (m1 + m2) * l1 * omega1 * omega1 * np.sin(delta)
        - (m1 + m2) * g * np.sin(theta2)
    ) / den2
    
    return np.array([dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt], dtype=np.float32)

def Hamiltonian(trajectory, params, g=9.81):   
    m1 = params["m1"]
    m2 = params["m2"]
    l1 = params["l1"]
    l2 = params["l2"]

    sin1 = trajectory[..., 0]
    cos1 = trajectory[..., 1]
    sin2 = trajectory[..., 2]
    cos2 = trajectory[..., 3]
    omega1 = trajectory[..., 4]
    omega2 = trajectory[..., 5]

    cos_delta = cos1 * cos2 + sin1 * sin2  # cos(theta1 - theta2)

    v1_sq = (l1 * omega1) ** 2
    v2_sq = v1_sq + (l2 * omega2) ** 2 + 2.0 * l1 * l2 * omega1 * omega2 * cos_delta

    kinetic = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
    potential = -(m1 + m2) * g * l1 * cos1 - m2 * g * l2 * cos2

    return kinetic + potential, kinetic, potential

def ode_state_to_observation(y):
    theta1, omega1, theta2, omega2 = y
    return np.array(
        [np.sin(theta1), np.cos(theta1), np.sin(theta2), np.cos(theta2), omega1, omega2],
        dtype=np.float32,
    )

def observation_to_ode_state(obs):
    theta1 = np.arctan2(obs[0], obs[1])
    theta2 = np.arctan2(obs[2], obs[3])
    omega1 = obs[4]
    omega2 = obs[5]
    return np.array([theta1, omega1, theta2, omega2], dtype=np.float32)

def sample_init_cond():
    theta1 = tc.rand(1) * 2 * np.pi - np.pi
    theta2 = tc.rand(1) * 2 * np.pi - np.pi
    omega1 = float((tc.rand(1) * 2 - 1))
    omega2 = float((tc.rand(1) * 2 - 1))
    return tc.tensor([np.sin(theta1), np.cos(theta1), np.sin(theta2), np.cos(theta2), omega1, omega2])

def integrate_controlled_pendulum(
    model: "MTF_SAC_Model",
    x0: tc.Tensor,
    measurement_index: int,
    cumulative_timeseries_index: int,
    T: int,
    apply_actions=False,
    expected=False,
    dt=0.01,
    external_inputs=None, #for compatibility
    fine_tuning_params=False
):

    if not fine_tuning_params:
        # Load physical parameters for the specific pendulum instance
        pendulum_params =  np.load(DATA_DIR + "/multi_task_double_pendulum/default/params.npy", allow_pickle=True)[measurement_index]
    else:
        pendulum_params = np.load(DATA_DIR + "/multi_task_double_pendulum/default/params_ft.npy", allow_pickle=True)[measurement_index]


    # Construct policy parameters for this specific instance
    policy_params = model.dsr_sac_model.sac_construct_params(cumulative_timeseries_index, model.dsr_sac_model.policy_params, "policy")

    t = 0.0
    obs = x0.cpu().numpy()
    x = observation_to_ode_state(obs)

    xs = tc.zeros((1, T + 1, 6), device=model.device)
    zs = tc.zeros((1, T+1, model.dsr_sac_model.hyperparameters["latent_dim"]), device=model.device)
    actions = tc.zeros((1, T, model.dim_forcing), device=model.device)
    xs[0, 0] = x0

    for i in range(1, T + 1):
        obs_tensor = tc.as_tensor(obs, device=model.device).float()
        encoded = model.encoder_model(obs_tensor, measurement_index)[0].unsqueeze(0)

        # The raw action from the policy
        action, _ = model.dsr_sac_model.sample_action(encoded, policy_params, measurement_index, expected=expected)

        if apply_actions:
            # Apply action to the velocities of the ODE state
            x[1] += action[0, 4].item() # Action on omega1
            x[3] += action[0, 5].item() # Action on omega2

        # Integrate one step in the 4D physical state [theta1, omega1, theta2, omega2].
        x = odeint(
            double_pendulum_derivatives,
            x,
            [t, t + dt],
            args=(pendulum_params["m1"], pendulum_params["m2"], pendulum_params["l1"], pendulum_params["l2"], 9.81),
        )[1]
        t += dt

        obs = ode_state_to_observation(x)
        xs[0, i] = tc.as_tensor(obs)
        actions[0, i - 1] = action.squeeze()
        zs[0, i-1] = encoded
    zs[0, -1] = model.encoder_model(tc.as_tensor(obs, device=model.device).float(), measurement_index)[0].unsqueeze(0)

    # The action needs to be decoded to match the shape expected by the reward function
    decoded_actions = model.decoder_model(
        actions,
        measurement_index,
        expected=expected,
    )

    return xs, zs, decoded_actions


def transform_to_cartesian(trajectory, parameters):
    sin_theta1 = trajectory[:, 0]
    cos_theta1 = trajectory[:, 1]
    sin_theta2 = trajectory[:, 2]
    cos_theta2 = trajectory[:, 3]

    theta1 = np.arctan2(sin_theta1, cos_theta1)
    theta2 = np.arctan2(sin_theta2, cos_theta2)

    l1 = parameters["l1"]
    l2 = parameters["l2"]

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    return np.stack([x1, y1, x2, y2], axis=1)

def check_for_stabilization(trajectory, goal_state, position_tol, angle_tol, velocity_tol, final_window=100):
    trajectory = trajectory.detach().cpu().numpy()

    theta1 = np.arctan2(trajectory[:, 0], trajectory[:, 1])
    theta2 = np.arctan2(trajectory[:, 2], trajectory[:, 3])
    omega1 = trajectory[:, 4]
    omega2 = trajectory[:, 5]

    goal_theta1 = np.arctan2(goal_state[0].item(), goal_state[1].item())
    goal_theta2 = np.arctan2(goal_state[2].item(), goal_state[3].item())
    theta1_err = wrap_angle(theta1 - goal_theta1)
    theta2_err = wrap_angle(theta2 - goal_theta2)
    omega1_err = omega1 - goal_state[4].item()
    omega2_err = omega2 - goal_state[5].item()

    tail = slice(-final_window, None)
    stabilized_by_position = (
        np.mean(np.abs(theta1_err[tail])) < angle_tol
        and np.mean(np.abs(theta2_err[tail])) < angle_tol
    )
    stabilized_by_dynamics = (
        np.mean(np.abs(omega1_err[tail])) < velocity_tol
        and np.mean(np.abs(omega2_err[tail])) < velocity_tol
        and np.std(omega1[tail]) < position_tol
        and np.std(omega2[tail]) < position_tol
    )

    stabilized = stabilized_by_position and stabilized_by_dynamics
    return stabilized

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
