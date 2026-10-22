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
    experiment_name = "pendulum_sac_actuated_new"

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
    device_ids = [1, 2]

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
    optuna_loss_scaling = {
        "neg_expected_reward": 1.0,
        }

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

    bpe = 3
    n_epochs = 5000
    env_steps_min, env_steps_max = 10, 100
    hidden1_min, hidden1_max = 40, 200
    hidden2_min = 20
    lr_min, lr_max = 1e-3, 5e-2
    lr_gamma_min, lr_gamma_max = 0.999, 0.99999
    polyak_tau_min, polyak_tau_max = 0.0005, 0.005
    init_log_temp_min, init_log_temp_max = -10, 5
    aq_min, aq_max = 0.1, 1e2
    apolicy_min, apolicy_max = 1, 5e3
    #aregq_min, aregq_max = 1e-1, 1e-1
    #areg_policy_min, areg_policy_max = 1e-1, 1e-1
    noise_min, noise_max = 0, 0.2
    #atemp_min, atemp_max = 1e-4, 1e2
    num_measurements = 16
    feature_dimension = 4
    discount_factor = 0.99
    config = TypedDictWrapper({
        "world_model": {
            "path": os.path.join(RESULTS_DIR, "pendulum_different/params_ldim=108_relu=83_feature_dim=4_enc_lr=0p00_dsr_lr=0p00_lr_gamma=1p00_gtf=0p02_gtf_gamma=1p00_aent=0p37_adsr=220p62_ac=0p32_seqlen=50/001"),
            "epoch": 5000,
        },
        "batching": {
            "subbatch_size": 256,
            "subbatch_number": OptunaSearchSpace.int("subjects", 16, 16),
        },
        "batches_per_epoch": bpe,
        "q_hierarchisation": {
            "scheme": "linear-projection",
            "keep_subject_parameters": True,
            "feature_dimension": feature_dimension,
        },
        "policy_hierarchisation": {
            "scheme": "linear-projection",
            "keep_subject_parameters": True,
            "feature_dimension": feature_dimension,
        },
        "q_ffn_hidden_dims": [OptunaSearchSpace.int("q_hid1", hidden1_min, hidden1_max),
                              OptunaSearchSpace.int("q_hid2", hidden2_min, "q_ffn_hidden_dims.0")],
        "policy_ffn_hidden_dims": [OptunaSearchSpace.int("pol_hid1", hidden1_min, hidden1_max),
                                    OptunaSearchSpace.int("pol_hid2", hidden2_min, "policy_ffn_hidden_dims.0")],
        "env_steps_per_epoch": OptunaSearchSpace.int("steps", env_steps_min, env_steps_max),
        "init_log_temperature": OptunaSearchSpace.float("init_temp", init_log_temp_min, init_log_temp_max),
        "adjustable_temperature": False,
        "target_entropy": -1, # common: -dim(action_space)
        "polyak_tau": OptunaSearchSpace.float("polyak_tau", polyak_tau_min, polyak_tau_max, log=True),
        "replay_buffer_max_size_per_measurement": 5000000,
        "discount_factor": discount_factor,
        "rollout_noise": OptunaSearchSpace.float("noise", noise_min, noise_max),
        "goal_decoders": 
            [
                [
                    {
                        "name": f"Identity_{i}",
                        "control_dimensions": [0, 1, 2, 3],
                        #"action_dim": 2,
                        "state_dep_direction": False,
                        "reward_weights": [1, 10, 1, 10, 1, 1],
                        "goal_state": {1.0: [0.0, -1.0, 0.0, -1.0, 0, 0],},
                    }
                ]
                for i in range(16)
            ],
        "modality_specific_evaluators": [[
            {
                "name": "action_evaluator",
                "parameters": {
                    "name": "action_evaluator",
                    "sequence_length": 500,
                    "discount_factor": discount_factor,
                    "true_environment": "integrate_controlled_pendulum",
                },
            },
            ] for _ in range(num_measurements)
        ],
        "evaluation_set_for_metrics": "both",
        "use_expectation_value_for_metrics": True,
        "evaluation_set_for_plots": "both",
        "use_expectation_value_for_plots": False,
        "gradient_clipping": 50,
        "action_penalty": 0.005,
        # if subjects vectors are learned, one should specify learning rates for them as well, otherwise they coincide with shared parameters
        "learning_rate": {
            "q_functions": {
                "lr": OptunaSearchSpace.float("q_lr", lr_min, lr_max, log=True),
            },
            "policy": {
                "lr": OptunaSearchSpace.float("policy_lr", lr_min, lr_max, log=True),
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
                    "initial": 0.1,
                },
            },
            "alpha_reg_policy": {
                "name": "constant",
                "hyperparameters": {
                    "initial": 0.1,
                },
            },
            "alpha_temperature": {
                "name": "constant",
                "hyperparameters": {
                    "initial": 0.1,
                },
            },
        },
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
