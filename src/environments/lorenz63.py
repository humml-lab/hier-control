import torch as tc
import numpy as np
from scipy.integrate import odeint
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.models.mtf_sac_model import MTF_SAC_Model
from src.paths import get_results_dir
RESULTS_DIR, DATA_DIR = get_results_dir()
    
def lorenz63(x, t, sigma=10.0, rho=28.0, beta=8/3):
    x1, x2, x3 = x
    return np.array([
        sigma * (x2 - x1),
        x1 * (rho - x3) - x2,
        x1 * x2 - beta * x3
    ])

def integrate_controlled_lorenz(
    model: "MTF_SAC_Model",
    x0: tc.Tensor,
    measurement_index: int,
    cumulative_timeseries_index: int,
    T:int,
    apply_actions=False,
    expected=False,
    dt=0.01,
    external_inputs=None, # for compatibility
    fine_tuning_params=False):

    if not fine_tuning_params:
        lorenz_params = np.load(DATA_DIR + "/multi_task_lorenz63/no_spirals/params.npy", allow_pickle=True)[measurement_index]
    else:
        lorenz_params = np.load(DATA_DIR + "/multi_task_lorenz63/no_spirals/params_FT.npy", allow_pickle=True)[measurement_index]
    policy_params = model.dsr_sac_model.sac_construct_params(cumulative_timeseries_index, model.dsr_sac_model.policy_params, "policy")


    scaling = tc.load(DATA_DIR + "/multi_task_lorenz63/no_spirals/std.pt").numpy().squeeze()
    shift = tc.load(DATA_DIR + "/multi_task_lorenz63/no_spirals/mean.pt").numpy().squeeze()

    t=0.0
    x = x0.cpu().numpy() * scaling + shift
    xs = tc.zeros((T+1, 3))
    zs = tc.zeros((T+1, model.dsr_sac_model.hyperparameters["latent_dim"]))
    actions = tc.zeros((T, 3))
    xs[0] = x0

    for i in range(1, T+1):
        scaled_x = (x - shift) / scaling
        encoded = model.encoder_model(tc.as_tensor(scaled_x).float(), measurement_index)[0].unsqueeze(0)
        action = model.dsr_sac_model.sample_action(encoded, policy_params, measurement_index, expected=expected)[0].squeeze()
        unscaled_action = action.detach().cpu().numpy() * scaling
        if apply_actions:
            x += unscaled_action

        # integrate one step forward
        x = odeint(lorenz63, x, [t, t+dt], args=tuple(lorenz_params.values()))[1]
        t += dt
        xs[i] = tc.as_tensor((x - shift) / scaling)
        zs[i-1] = encoded
        actions[i-1] = action
    zs[-1] = model.encoder_model(tc.as_tensor((x - shift) / scaling).float(), measurement_index)[0].unsqueeze(0)

    return xs.unsqueeze(0), zs.unsqueeze(0), actions.unsqueeze(0)