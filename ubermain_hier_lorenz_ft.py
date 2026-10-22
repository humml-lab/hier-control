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
from src.parameters import Parameters
from src.utils import TypedDictWrapper
from src.paths import get_results_dir
RESULTS_DIR, DATA_DIR = get_results_dir()

if __name__ == "__main__":
    experiment_name = "hier_lorenz_ft"

    # ======== General Configuration ========
    # Total number of processes to run in parallel.
    # If using GPUs, the value used will be min(this, (number of available GPUs x num_processes_per_gpu)).
    n_processes = 8

    # Whether to use GPUs for training.
    # Larger batch sizes and more complex models (latent_dim large) benefit most from GPU acceleration.
    use_gpu = True

    # Number of trainings to start in parallel on each GPU.
    # Modern GPUs can handle multiple processes, but this depends on the GPU memory.
    # Use nvidia-smi to check the GPU memory usage.
    num_processes_per_gpu = 4

    # The device ids of the GPUs to use for training.
    # If empty, the GPUs are automatically distributed using the available GPUs.
    # A GPU is available if it has at least 5GB of free memory and 0% utilization. This can be configured in the multitasking module.
    device_ids = [2, 3]

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
    fine_tuning_path = os.path.join(RESULTS_DIR, "hier_control/hier_lorenz_3/params_ldim=26_relu=12_feature_dim=4_dsr_lr=0p01_lr_gamma=1p00_gtf=0p06_gtf_gamma=1p00_aent=0p89_adsr=114p61_ac=4p55_ar=3p83_asc=0p32_alpha_mar=1p90_mar=0/001")
    # parameters optimized by optuna
    dsr_lr_min, dsr_lr_max = 1e-4, 1e-1
    gtf_min, gtf_max = 1e-4, 5e-1
    gtf_gamma_min, gtf_gamma_max = 0.999, 0.99999
    aent_min, aent_max = 1e-1, 40
    adsr_min, adsr_max = 1, 150
    acons_min, acons_max = 1e-2, 30
    asub_min, asub_max = 1e-4, 30
    amar_min, amar_max = 1e-4, 30
    lr_gamma_min, lr_gamma_max = 0.999, 1.0

    config = TypedDictWrapper({
        "base_path": os.path.join(DATA_DIR, "hier-lorenz_preprocessed_FT/"),
        "fine_tuning_path": fine_tuning_path,
        "measurement_folders": measurements,
        "batching": {
            "subbatch_size": 512,
            "subbatch_number": n_measurements,
        },
        "batches_per_epoch": bpe,
        "test_timeseries_indices":
            [[i for i in range(4)] for _ in range(n_measurements)],
        "evaluation_set_for_metrics": "both",
        "use_expectation_value_for_metrics": True,
        "evaluation_set_for_plots": "both",
        "use_expectation_value_for_plots": False,
        "evaluation_test_timeseries_indices": [[i for i in range(4)] for _ in range(n_measurements)],
        "evaluation_train_timeseries_indices": [[i for i in range(5, 10)] for _ in range(n_measurements)],
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
                    "initial": OptunaSearchSpace.float("asc", asub_min, asub_max, log=True),
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
        "expensive_saving_interval": 400,
    }, Parameters)

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
