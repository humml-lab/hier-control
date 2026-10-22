import os
import json
from src.hyperparameter_search.multitasking import run_multitask
from src.hyperparameter_search.search_space.optuna_search_space import (
    OptunaCategorical,
    OptunaFloat,
    OptunaInt,
    OptunaSearchSpace,
)
from src.hyperparameter_search.search_space.search_space import SearchSpace
from src.parameters import SAC_Parameters
from src.utils import TypedDictWrapper
from src.paths import get_results_dir
RESULTS_DIR, DATA_DIR = get_results_dir()

if __name__ == "__main__":
    experiment_name = "hier_lorenz_sac_2_ft"

    # ======== General Configuration ========
    # Total number of processes to run in parallel.
    # If using GPUs, the value used will be min(this, (number of available GPUs x num_processes_per_gpu)).
    n_processes = 9

    # Whether to use GPUs for training.
    # Larger batch sizes and more complex models (latent_dim large) benefit most from GPU acceleration.
    use_gpu = True

    # Number of trainings to start in parallel on each GPU.
    # Modern GPUs can handle multiple processes, but this depends on the GPU memory.
    # Use nvidia-smi to check the GPU memory usage.
    num_processes_per_gpu = 3

    # The device ids of the GPUs to use for training.
    # If empty, the GPUs are automatically distributed using the available GPUs.
    # A GPU is available if it has at least 5GB of free memory and 0% utilization. This can be configured in the multitasking module.
    device_ids = []

    # Whether to use torch compile.
    # torch.compile just-in-time compiles the model to speed up training.
    # The most gains are seen when using a GPU.
    # On Windows, support is questionable, see torch.compile documentation.
    use_torch_compile = True

    # Whether to use Bayesian Optimization with Optuna for hyperparameter search.
    use_optuna = True

    # ======== Optuna Configuration ========
    # Note: a bunch of Optuna's options are not exposed here, but can be set in the multitasking and optuna_worker modules.

    # The total number of runs (including pruning) that Optuna will perform.
    optuna_num_runs = 50
    # Optional dictionary to scale the different loss terms in the Optuna objective function.
    # If None, the DSR loss is used as the objective.
    optuna_loss_scaling = None

    # ======== Grid Search Configuration ======== (use_optuna=False)
    # If True, performs a full grid search, testing every possible combination of the provided parameter values.
    # If False, it pairs parameters by their list index (e.g., param1[i] with param2[i]),
    # which requires all parameter lists to have the same length.
    full_grid = True

    # Number of runs for each parameter configuration.
    # Not taken into account when using Optuna.
    n_runs_per_configuration = 1

    # The parameters to be used for the experiment.
    # All parameters have to be defined inside a list. If multiple values are given,
    # grid search is performed over the values. In this case, the label is used
    # to label the different parameter settings in the results directory.

    bpe = 1
    fine_tuning_path = os.path.join(RESULTS_DIR, "hier_lorenz_sac_2/params_q_hidden1=35_policy_hidden1=27_env_steps_per_epoch=17_polyak_tau=0p00_q_lr=0p00_policy_lr=0p00_q_lr_gamma=1p00_policy_lr_gamma=1p00_aq=57p42_apolicy=63p73_areg_q=31p24_areg_policy=0p36_areg_temp=0p00_q_hidden2=30_policy_hidden2=23/001")
    n_epochs = 5000
    num_measurements = 64
    # parameters optimized by optuna
    env_steps_min, env_steps_max = 10, 30
    q_lr_min, q_lr_max = 1e-4, 1e-2
    policy_lr_min, policy_lr_max = 1e-5, 1e-2
    lr_gamma_min, lr_gamma_max = 0.999, 0.99999
    polyak_tau_min, polyak_tau_max = 0.001, 0.01
    aq_min, aq_max = 1, 1e3
    apolicy_min, apolicy_max = 1, 1e3
    aregq_min, aregq_max = 1e-2, 1e2
    areg_policy_min, areg_policy_max = 1e-2, 1e2
    atemp_min, atemp_max = 1e-4, 1e2

    config = TypedDictWrapper({
        "fine_tuning_path": fine_tuning_path,
        "world_model": {
            "path": os.path.join(RESULTS_DIR, "hier_lorenz_ft/params_dsr_lr=0p01_lr_gamma=1p00_gtf=0p42_gtf_gamma=1p00_aent=14p10_adsr=23p85_ac=0p03_ar=5p34_asc=0p00_alpha_mar=0p14/001"),
            "epoch": 5000,
        },
        "batching": {
            "subbatch_size": 512,
            "subbatch_number": num_measurements,
        },
        "batches_per_epoch": bpe,
        "env_steps_per_epoch": OptunaSearchSpace.int("env_steps_per_epoch", env_steps_min, env_steps_max),
        "init_log_temperature": -0.5,
        "adjustable_temperature": False,
        "target_entropy": -200, # common: -dim(action_space)
        "polyak_tau": OptunaSearchSpace.float("polyak_tau", polyak_tau_min, polyak_tau_max, log=True),
        "replay_buffer_max_size_per_measurement": 100000,
        "goal_decoders": [
            [
                {
                    "name": f"Identity_{i}",
                    "control_dimensions": [0],
                    "goal_state": [0.5, 0.5, -0.5],
                }
            ]
            for i in range(num_measurements)
        ],
        "evaluation_set_for_metrics": "both",
        "use_expectation_value_for_metrics": True,
        "evaluation_set_for_plots": "both",
        "use_expectation_value_for_plots": False,
        "gradient_clipping": 50,
        "learning_rate": {
            "q_functions": {
                "lr": OptunaSearchSpace.float("q_lr", q_lr_min, q_lr_max, log=True),
            },
            "policy": {
                "lr": OptunaSearchSpace.float("policy_lr", policy_lr_min, policy_lr_max, log=True),
            },
        },
        "lr_scheduling": {
            "q":
                {"name": "ExponentialLR",
                "hyperparameters": {
                    "gamma": OptunaSearchSpace.float("q_lr_gamma", lr_gamma_min, lr_gamma_max, log=True), #0.1**(1 / n_epochs)
                }
                },
            "policy":
                {
                "name": "ExponentialLR",
                "hyperparameters": {
                    "gamma": OptunaSearchSpace.float("policy_lr_gamma", lr_gamma_min, lr_gamma_max, log=True), #0.1**(1 / n_epochs)
                }
            }
        },
        "value_scheduling": {
            "alpha_q": {
                "name": "constant",
                "hyperparameters": {
                    "initial": OptunaSearchSpace.float("aq", aq_min, aq_max, log=True),
                },
            },
            "alpha_policy": {
                "name": "constant",
                "hyperparameters": {
                    "initial": OptunaSearchSpace.float("apolicy", apolicy_min, apolicy_max, log=True),
                },
            },
            "alpha_reg_q": {
                "name": "constant",
                "hyperparameters": {
                    "initial": OptunaSearchSpace.float("areg_q", aregq_min, aregq_max),
                },
            },
            "alpha_reg_policy": {
                "name": "constant",
                "hyperparameters": {
                    "initial": OptunaSearchSpace.float("areg_policy", areg_policy_min, areg_policy_max, log=True),
                },
            },
            "alpha_temperature": {
                "name": "constant",
                "hyperparameters": {
                    "initial": OptunaSearchSpace.float("areg_temp", atemp_min, atemp_max, log=True),
                },
            },
        },
        "sublosses": False,
        "n_epochs": n_epochs,
        "cheap_saving_interval": 100,
        "expensive_saving_interval": 200,
        "simulation_saving_interval": 200,
    }, SAC_Parameters)

    # reuse all parameters from old config that are not specified here
    old_config = json.load(open(os.path.join(fine_tuning_path, "config.json")))
    for name, value in old_config.items():
        if name not in config:
            config[name] = value

    # DO NOT CHANGE BELOW THIS LINE
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    run_multitask(
        RESULTS_DIR,
        experiment_name,
        n_runs_per_configuration,
        config,
        n_processes,
        "no-gpu"
        if not use_gpu
        else (device_ids if len(device_ids) > 0 else "automatic-distribution"),
        num_processes_per_gpu if use_gpu else None,
        use_torch_compile,
        use_optuna,
        optuna_num_runs,
        optuna_loss_scaling,
        full_grid,
    )
