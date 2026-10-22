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
    experiment_name = "mc_multitask"

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
    num_processes_per_gpu = 2

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
    optuna_num_runs = 60
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
    measurements = ['20230425', '20210715', '20230404', '20210405', '20220930', '20210412', '20230320', '20210624', '20200317', '20230405', '20230508', '20220221']
    n_measurements = len(measurements)
    n_epochs = 3000
    seqlen = 60
    feature_dimension_min, feature_dimension_max =  4, 6
    # parameters optimized by optuna
    latent_dim_min, latent_dim_max = 10, 200
    relu_units_min, relu_units_max = 0, 40
    mar_min, mar_max = 1, 4
    encoder_lr_min, encoder_lr_max = 1e-4, 1e-2
    dsr_lr_min, dsr_lr_max = 1e-4, 1e-2
    decoder_lr_min, decoder_lr_max = 1e-2, 1e-1
    gtf_min, gtf_max = 1e-2, 9e-1
    gtf_gamma_min, gtf_gamma_max = 0.8, 0.9999
    aent_min, aent_max = 1e-3, 10
    adsr_min, adsr_max = 1, 100
    acons_min, acons_max = 1e-3, 10
    arec_min, arec_max = 1e-3, 10
    asub_min, asub_max = 1e-3, 10
    amar_min, amar_max = 1e-3, 10
    lr_gamma_min, lr_gamma_max = 0.9, 1.0

    config = TypedDictWrapper({
        "base_path": os.path.join(DATA_DIR, "mc_multitask"),
        "measurement_folders": measurements,
        "batching": {
            "subbatch_size": 512,
            "subbatch_number": n_measurements,
        },
        "batches_per_epoch": bpe,
        "test_timeseries_indices":
            [[i for i in range(10)] for _ in range(n_measurements)],
        # "train_test_split": 0,
        "latent_dim": OptunaSearchSpace.int("ldim", latent_dim_min, latent_dim_max),
        "dim_forcing": OptunaSearchSpace.int("dim_forcing", 10, latent_dim_min),
        "encoder": {
            "name": "CausalStackedConvolutions",
            "hyperparameters": {
                "mean_kernel_sizes": [OptunaSearchSpace.int("mks1", 3, 15, step=2),
                                      OptunaSearchSpace.int(
                                          "mks2", 3, 15, step=2),
                                      OptunaSearchSpace.int(
                                          "mks3", 3, 15, step=2),
                                      3],
                "logvar_kernel_sizes": [11],
                "deterministic": False,
            }
        },
        "encoder_hierarchisation": {
            "scheme": "none",
        },
        "latent_step": {
            "name": "ALRNN",
            "hyperparameters": {
                "relu_units": OptunaSearchSpace.int("relu", relu_units_min, relu_units_max),
                "learnable_C": True,
            }
        },
        "dsr_hierarchisation_scheme": {
            "scheme": "linear-projection",
            "feature_dimension": OptunaSearchSpace.int("feat", feature_dimension_min, feature_dimension_max),
        },
        "dsr_different_dynamics": "per-measurement",
        "dsr_different_external_inputs": "per-subject",
        "MARegularized_units": OptunaSearchSpace.int("mar", mar_min, mar_max),
        "decoder":
            [[
                {
                    "name": "Gaussian",
                },
                {
                    "name": "Gaussian",
                },
            ] for _ in range(n_measurements)],
        "modality_specific_evaluators": [[
            {
                "name": "GaussianEvaluator",
                "parameters": {
                    "name": "Spikes",
                    "sequence_length": 5000,
                    "plot_latent_vs_encoded": False,
                    "plot_true_vs_reconstructed": False
                },
            },
            {
                "name": "GaussianEvaluator",
                "parameters": {
                    "name": "Finger Position",
                    "sequence_length": 5000,
                    "plot_latent_vs_encoded": False,
                    "plot_true_vs_reconstructed": False
                },
            }]
            for _ in range(n_measurements)],
        "evaluation_set_for_metrics": "both",
        "use_expectation_value_for_metrics": True,
        "evaluation_set_for_plots": "both",
        "use_expectation_value_for_plots": False,
        "evaluation_test_timeseries_indices": [[i for i in range(5)] for _ in range(n_measurements)],
        "evaluation_train_timeseries_indices": [[i for i in range(10, 15)] for _ in range(n_measurements)],
        "PE_args": {
            "compute_PE": True,
            "n_steps_ahead": 1,
            "min_context_length": 1,
            "batch_size": -1,
        },
        "n_interleave": 1,
        "gradient_clipping": 50,
        "learning_rate": {
            "encoder": {
                "lr": OptunaSearchSpace.float("enc_lr", encoder_lr_min, encoder_lr_max, log=True),
            },
            "decoder": {
                "lr": OptunaSearchSpace.float("dec_lr", decoder_lr_min, decoder_lr_max, log=True),
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
                    "initial": OptunaSearchSpace.float("ar", arec_min, arec_max, log=True),
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
        "seq_len": seqlen,
        "n_epochs": n_epochs,
        "cheap_saving_interval": 100,
        "expensive_saving_interval": 400,
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
