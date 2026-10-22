import os
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
    experiment_name = "mc_maze_sac_sparse"

    # ======== General Configuration ========
    # Total number of processes to run in parallel.
    # If using GPUs, the value used will be min(this, (number of available GPUs x num_processes_per_gpu)).
    n_processes = 6

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
    optuna_num_runs = 6
    # Optional dictionary to scale the different loss terms in the Optuna objective function.
    # If None, the negative expected reward is used as the objective.
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

    bpe = 10
    n_epochs = 5000
    env_steps_min, env_steps_max = 50, 100
    hidden1_min, hidden1_max = 128, 256
    hidden2_min = 64
    q_lr_min, q_lr_max = 1e-4, 1e-2
    policy_lr_min, policy_lr_max = 1e-4, 1e-2
    lr_gamma_min, lr_gamma_max = 0.999, 1.0
    polyak_tau_min, polyak_tau_max = 0.001, 0.01
    aq_min, aq_max = 1e-2, 1e4
    apolicy_min, apolicy_max = 1e-2, 1e3
    aregq_min, aregq_max = 1e-2, 1e2
    areg_policy_min, areg_policy_max = 1e-2, 1e2
    num_measurements = 1 # needs to agree with the world model used
    feature_dimension = num_measurements
    discount_factor = 0.999
    config = TypedDictWrapper({
        "world_model": {
            "path": os.path.join(RESULTS_DIR,
                                 "mc_maze/params_ldim=209_dim_forcing=10_relu=20_mar=4_enc_lr=0p00_dec_lr=0p01_dsr_lr=0p00_gtf=0p05_gtf_gamma=0p93_aent=8p45_adsr=65p13_ac=0p08_ar=0p28_asc=0p00_alpha_mar=0p00",
                                  "001"),
            "epoch": 2400,
        },
        "batching": {
            "subbatch_size": 512,
            "subbatch_number": num_measurements,
        },
        "batches_per_epoch": bpe,
        "q_hierarchisation": {
            "scheme": "linear-projection",
            "keep_subject_parameters": True,
            "feature_dimension": feature_dimension
        },
        "policy_hierarchisation": {
            "scheme": "linear-projection",
            "keep_subject_parameters": True,
            "feature_dimension": feature_dimension
        },
        "q_ffn_hidden_dims": [OptunaSearchSpace.int("q_hidden1", hidden1_min, hidden1_max),
                              OptunaSearchSpace.int("q_hidden2", hidden2_min, "q_ffn_hidden_dims.0")],
        "policy_ffn_hidden_dims": [OptunaSearchSpace.int("policy_hidden1", hidden1_min, hidden1_max),
                                    OptunaSearchSpace.int("policy_hidden2", hidden2_min, "policy_ffn_hidden_dims.0")],
        "env_steps_per_epoch": OptunaSearchSpace.int("env_steps_per_epoch", env_steps_min, env_steps_max),
        "init_log_temperature": 0.1,
        "adjustable_temperature": False,
        "target_entropy": -10, # common: -dim(action_space)
        "polyak_tau": OptunaSearchSpace.float("polyak_tau", polyak_tau_min, polyak_tau_max, log=True),
        "replay_buffer_max_size_per_measurement": 1000000,
        "discount_factor": discount_factor,
        "goal_decoders": [
            [
                {
                    "name": f"Gaussian_{i}",
                    "control_dimensions": [0, 1],
                    "goal_state": {"1.0": [0.0, 0.0]},
                }
            ]
            for i in range(num_measurements)
        ],

        "modality_specific_evaluators": [[
            {
                "name": "action_evaluator",
                "parameters": {
                    "name": "action_evaluator",
                    "sequence_length": 80,
                    "discount_factor": discount_factor,
                    "generate_without_external_inputs": False,
                },
            },
            ] for _ in range(num_measurements)
        ],
        "evaluation_set_for_metrics": "both",
        "use_expectation_value_for_metrics": True,
        "evaluation_set_for_plots": "both",
        "use_expectation_value_for_plots": False,
        "gradient_clipping": 60,
        "action_penalty": 0.0005,
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
                    "initial": 0,
                },
            },
        },
        "sublosses": False,
        "n_epochs": n_epochs,
        "cheap_saving_interval": 100,
        "expensive_saving_interval": 200,
    }, SAC_Parameters)

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
