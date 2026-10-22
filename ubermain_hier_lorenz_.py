import os
from src.hyperparameter_search.multitasking import run_multitask
from src.hyperparameter_search.search_space.optuna_search_space import (
    OptunaCategorical,
    OptunaFloat,
    OptunaInt,
    OptunaSearchSpace,
)
from src.hyperparameter_search.search_space.search_space import SearchSpace
from src.parameters import Parameters
from src.utils import TypedDictWrapper
from src.paths import get_results_dir
RESULTS_DIR, DATA_DIR = get_results_dir()

if __name__ == "__main__":
    experiment_name = "hier_lorenz_3"

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
    optuna_num_runs = 40
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

    bpe = 5
    measurements = [str(i) for i in range(64)]
    n_measurements = len(measurements)
    n_epochs = 5000
    seqlen = 30
    # parameters optimized by optuna
    feature_dimension_min, feature_dimension_max = 2, 5
    latent_dim_min, latent_dim_max = 15, 30
    relu_units_min, relu_units_max = 3, 18
    mar_min = 0
    dsr_lr_min, dsr_lr_max = 1e-4, 1e-1
    gtf_min, gtf_max = 1e-3, 1e-1
    gtf_gamma_min, gtf_gamma_max = 0.999, 0.99999
    aent_min, aent_max = 1e-1, 100
    adsr_min, adsr_max = 100, 300
    acons_min, acons_max = 1e-2, 50
    amar_min, amar_max = 0.01, 0.01
    lr_gamma_min, lr_gamma_max = 0.999, 1.0

    config = TypedDictWrapper({
        "base_path": os.path.join(DATA_DIR, "hier-lorenz_preprocessed"),
        "measurement_folders": measurements,
        "batching": {
            "subbatch_size": 512,
            "subbatch_number": n_measurements,
        },
        "batches_per_epoch": bpe,
        "test_timeseries_indices":
            [[i for i in range(4)] for _ in range(n_measurements)],
        # "train_test_split": 0,
        "latent_dim": OptunaSearchSpace.int("ldim", latent_dim_min, latent_dim_max),
        "dim_forcing": 3,
        "encoder": {
            "name": "Identity",
        },
        "encoder_hierarchisation": {
            "scheme": "none",
        },
        "latent_step": {
            "name": "ALRNN",
            "hyperparameters": {
                "relu_units": OptunaSearchSpace.int("relu", relu_units_min, relu_units_max),
            }
        },
        "dsr_hierarchisation_scheme": {
            "scheme": "linear-projection",
            "feature_dimension": OptunaSearchSpace.int("feature_dim", feature_dimension_min, feature_dimension_max),
        },
        "dsr_different_dynamics": "per-measurement",
        "dsr_different_external_inputs": "per-subject",
        "MARegularized_units": OptunaSearchSpace.int("mar", mar_min, 0),
        "decoder":
            [[
                {
                    "name": "Identity",
                }
            ] for _ in range(n_measurements)],
        "modality_specific_evaluators": [[
            {
                "name": "LorenzEvaluator",
                "parameters": {
                    "name": "Lorenz",
                },
            }
            ]
            for _ in range(n_measurements)],
        "evaluation_set_for_metrics": "both",
        "use_expectation_value_for_metrics": True,
        "evaluation_set_for_plots": "test",
        "use_expectation_value_for_plots": False,
        "evaluation_test_timeseries_indices": [[i for i in range(4)] for _ in range(n_measurements)],
        "evaluation_train_timeseries_indices": [[i for i in range(5, 6)] for _ in range(n_measurements)],
        "PE_args": {
            "compute_PE": False,
            "n_steps_ahead": 1,
            "min_context_length": 1,
            "batch_size": -1,
        },
        "n_interleave": 1,
        "gradient_clipping": 50,
        "learning_rate": {
            "encoder": {
                "lr": 0,
            },
            "decoder": {
                "lr": 0,
            },
            "dsr": {
                "lr": OptunaSearchSpace.float("dsr_lr", dsr_lr_min, dsr_lr_max, log=True),
            },
        },
        "lr_scheduling": {
            "combined": {
                "name": "ExponentialLR",
                "hyperparameters": {
                    "gamma": OptunaSearchSpace.float("lr_gamma", lr_gamma_min, lr_gamma_max, log=True), #0.1**(1 / n_epochs)
                },
            },
        },
        "value_scheduling": {
            "alpha_gtf": {
                "name": "exponential",
                "hyperparameters": {
                    "initial": OptunaSearchSpace.float("gtf", gtf_min, gtf_max, log=True),
                    "gamma": OptunaSearchSpace.float("gtf_gamma", gtf_gamma_min, gtf_gamma_max),
                },
            },
            "alpha_entropy": {
                "name": "constant",
                "hyperparameters": {
                    "initial": OptunaSearchSpace.float("aent", aent_min, aent_max, log=True),
                },
            },
            "alpha_dsr": {
                "name": "constant",
                "hyperparameters": {
                    "initial": OptunaSearchSpace.float("adsr", adsr_min, adsr_max),
                },
            },
            "alpha_consistency": {
                "name": "constant",
                "hyperparameters": {
                    "initial": OptunaSearchSpace.float("ac", acons_min, acons_max, log=True),
                },
            },
            "alpha_reconstruction": {
                "name": "constant",
                "hyperparameters": {
                    "initial": 0,
                },
            },
            "alpha_subject_continuity": {
                "name": "constant",
                "hyperparameters": {
                    "initial": 0,
                },
            },
            "alpha_MAR": {
                "name": "constant",
                "hyperparameters": {
                    "initial": OptunaSearchSpace.float("alpha_mar", amar_min, amar_max, log=True),
                },
            },
            "alpha_dsr_l2": {
                "name": "constant",
                "hyperparameters": {
                    "initial": 0,
                }
            }
        },
        "sublosses": False,
        "seq_len": seqlen,
        "n_epochs": n_epochs,
        "cheap_saving_interval": 100,
        "expensive_saving_interval": 250,
    }, Parameters)

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
