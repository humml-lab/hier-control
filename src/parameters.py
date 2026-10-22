from typing import TYPE_CHECKING, Literal, Optional, TypedDict, Union

# This makes sure that no circular imports occur, as this is purely a type definition file
if TYPE_CHECKING:
    from src.evaluation.evaluator import (
        GeneralEvaluatorOption,
        ModalityIndependentEvaluatorOption,
        ModalitySpecificEvaluatorOption,
    )
    from src.models.decoder.decoder_models import DecoderOption
    from src.models.dsr_model.dsr_model import (
        DifferentDynamicsOption,
        DifferentExternalInputsOption,
        LatentStepOption,
    )
    from src.models.dsr_model.hierarchisation_schemes.base import (
        DSRHierarchisationScheme,
    )
    from src.models.encoder.encoder import (
        EncoderHierarchisationScheme,
        EncoderOption,
    )
    from src.scheduler.composite_scheduler import LRConfig
    from src.scheduler.value_scheduler import ScheduleConfig

import os

import optuna


class PEArgsDict(TypedDict):
    """Prediction Error arguments dictionary structure."""

    compute_PE: bool
    n_steps_ahead: int
    min_context_length: int
    batch_size: int


class Parameters(TypedDict):
    """
    The hyperparameters for the training of the model.

    Args:
        experiment (str): Name of machine learning experiment.
    """

    run_directory: str
    experiment: str
    name: str
    run: int
    cheap_saving_interval: int
    expensive_saving_interval: int
    measurement_folders: list[str]
    base_path: str
    fine_tuning_path: Optional[str]
    excluded_filenames: list[str]
    test_timeseries_indices: list[int]
    train_test_split: Optional[float]
    # Encoder
    encoder: "EncoderOption"
    encoder_hierarchisation: "EncoderHierarchisationScheme"
    # DSR
    latent_step: "LatentStepOption"
    dsr_hierarchisation_scheme: "DSRHierarchisationScheme"
    dsr_different_dynamics: "DifferentDynamicsOption"
    dsr_different_external_inputs: "DifferentExternalInputsOption"
    dsr_continuous_subject_groups: Optional[list[list[int]]]
    # Decoder
    decoder: list[list["DecoderOption"]]
    decoder_scaling: list[list[float]] | list[float]  # scale each modalities loss
    latent_chunk_indices: list[
        list[set[int]]
    ]  # map of decoder -> region for each decoder when using coupled PLRNNs
    # Evaluation
    modality_specific_evaluators: list[list["ModalitySpecificEvaluatorOption"]]
    modality_independent_evaluators: list["ModalityIndependentEvaluatorOption"]
    general_evaluators: list["GeneralEvaluatorOption"]
    evaluation_set_for_metrics: Literal["train", "test", "both"]
    evaluation_set_for_plots: Literal["train", "test", "both"]
    use_expectation_value_for_metrics: bool
    n_samples_for_metrics: int
    use_expectation_value_for_plots: bool
    n_samples_for_plots: int
    plot_per_subject: bool
    evaluation_test_timeseries_indices: Optional[list[int]]
    evaluation_train_timeseries_indices: Optional[list[int]]
    # Data loading / device placement
    move_data_to_device_at_init: bool
    dataloader_num_workers: int
    # Prediction Error (PE) arguments
    PE_args: PEArgsDict

    dim_forcing: int
    latent_dim: int
    n_interleave: int
    estimate_forcing: int
    alpha: float
    gamma: float
    update_step: int
    estimation_method: str
    batching: dict
    batches_per_epoch: int
    seq_len: int
    learning_rate: Union[float, list[float], "LRConfig"]
    lr_scheduling: dict[str, str | dict[str, float | int | str | list[float]]]
    n_epochs: int
    decoder_scaling_adjustment_epochs: list[int]
    gradient_clipping: int
    value_scheduling: dict[str, "ScheduleConfig"]
    alpha_consistency: float
    alpha_reconstruction: float
    alpha_dsr: float
    alpha_entropy: float
    alpha_subject_continuity: float
    alpha_encoded_smoothness: float
    load_model_path: str
    resume_epoch: int
    alpha_MAR: float
    MARegularized_units: int
    comment: str
    optuna_trial: Optional[optuna.Trial]  # Used for optuna hyperparameter optimization
    optuna_loss_term_scaling: Optional[
        dict[str, float]
    ]  # Scaling factors for loss terms in optuna trials

    @staticmethod
    def default() -> "Parameters":
        return {
            "experiment": "default",
            "name": "default",
            "run": None,
            "cheap_saving_interval": 10,
            "expensive_saving_interval": 200,
            "base_path": os.path.join("data", "train"),
            "fine_tuning_path": None,
            "measurement_folders": [],
            "excluded_filenames": [],
            "test_timeseries_indices": None,
            "train_test_split": None,
            "latent_step": {
                "name": "shPLRNN",
            },
            "dsr_different_dynamics": "per-measurement",
            "dsr_different_external_inputs": "per-subject",
            "dsr_hierarchisation_scheme": {
                "scheme": "none",
            },
            "encoder": {
                "name": "StackedConvolutions",
            },
            "encoder_hierarchisation": {
                "scheme": "none",
            },
            "decoder": [],
            "latent_chunk_indices": [],
            "modality_specific_evaluators": None,
            "modality_independent_evaluators": [],
            "general_evaluators": [],
            "evaluation_set_for_metrics": "both",
            "evaluation_set_for_plots": "both",
            "use_expectation_value_for_metrics": True,
            "n_samples_for_metrics": 1,
            "use_expectation_value_for_plots": True,
            "n_samples_for_plots": 1,
            "plot_per_subject": True,
            "evaluation_test_timeseries_indices": None,
            "evaluation_train_timeseries_indices": None,
            "PE_args": {
                "compute_PE": False,
                "n_steps_ahead": 1,
                "min_context_length": 20,
                "batch_size": -1,
            },
            "dim_forcing": None,
            "decoder_scaling": None,
            "latent_dim": 20,
            "n_interleave": 1,
            "estimate_forcing": 0,
            "gamma": 0.999,
            "update_step": 5,
            "estimation_method": "arithmetic_mean",
            "batches_per_epoch": 25,
            "seq_len": 300,
            "learning_rate": 1e-3,
            "lr_scheduling": {
                "name": "MultiStepLR",
                "hyperparameters": {
                    "gamma": 0.1,
                    "milestones": [0.2, 0.8],
                },
            },
            "n_epochs": 5000,
            "decoder_scaling_adjustment_epochs": [2, 10],
            "gradient_clipping": 0,
            "load_model_path": None,
            "resume_epoch": None,
            "MARegularized_units": 0,
            "value_scheduling": {
                "alpha_gtf": {
                    "name": "constant",
                    "hyperparameters": {
                        "initial": 1,  # The initial alpha value for GTF.
                    },
                },
                "alpha_entropy": {
                    "name": "constant",
                    "hyperparameters": {
                        "initial": 1,
                    },
                },
                "alpha_dsr": {
                    "name": "constant",
                    "hyperparameters": {
                        "initial": 1,
                    },
                },
                "alpha_consistency": {
                    "name": "constant",
                    "hyperparameters": {
                        "initial": 1,
                    },
                },
                "alpha_reconstruction": {
                    "name": "constant",
                    "hyperparameters": {
                        "initial": 1,
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
                        "initial": 0,
                    },
                },
                "alpha_AR_convergence": {
                    "name": "constant",
                    "hyperparameters": {
                        "initial": 0,
                    },
                },
            },
            "comment": None,
            "optuna_trial": None,
            "optuna_loss_term_scaling": None,
            "move_data_to_device_at_init": True,
            "dataloader_num_workers": 0,
        }


class SAC_Parameters(TypedDict):
    """
    The hyperparameters for the training of the sac model.

    Args:
        experiment (str): Name of machine learning experiment.
    """

    run_directory: str
    experiment: str
    name: str
    run: int
    cheap_saving_interval: int
    expensive_saving_interval: int
    world_model: dict[str, Union[str, int]]
    fine_tuning_path: Optional[str]
    batches_per_epoch: int

    # Q Function and Policy
    ffn_activation: Literal["ReLU", "LeakyReLU", "ELU", "GELU", "Tanh"]
    q_ffn_out_dims: list[int]
    q_hierarchisation: "DSRHierarchisationScheme"
    policy_ffn_out_dims: list[int]
    policy_hierarchisation: "DSRHierarchisationScheme"
    clip_target_q_values: bool

    # SAC
    env_steps_per_epoch: int
    init_log_temperature: float
    adjustable_temperature: bool
    target_entropy: float
    polyak_tau: float
    replay_buffer_max_size_per_measurement: int
    goal_decoders: list[str]
    discount_factor: float
    rollout_noise: float

    # Evaluation
    modality_specific_evaluators: list[list["ModalitySpecificEvaluatorOption"]]
    modality_independent_evaluators: list["ModalityIndependentEvaluatorOption"]
    general_evaluators: list["GeneralEvaluatorOption"]
    evaluation_set_for_metrics: Literal["train", "test", "both"]
    evaluation_set_for_plots: Literal["train", "test", "both"]
    use_expectation_value_for_metrics: bool
    n_samples_for_metrics: int
    use_expectation_value_for_plots: bool
    n_samples_for_plots: int
    plot_per_subject: bool
    evaluation_test_timeseries_indices: Optional[list[int]]
    evaluation_train_timeseries_indices: Optional[list[int]]
    batching: dict
    # Data loading / device placement
    dataloader_num_workers: int
    move_data_to_device_at_init: bool
    learning_rate: Union[float, list[float], "LRConfig"]
    lr_scheduling: dict[str, str | dict[str, float | int | str | list[float]]]
    n_epochs: int
    gradient_clipping: int
    action_penalty: float
    value_scheduling: dict[str, "ScheduleConfig"]
    alpha_q: float
    alpha_policy: float
    alpha_reg_q: float
    alpha_reg_policy: float
    alpha_temperature: float
    load_model_path: str
    resume_epoch: int
    comment: str
    optuna_trial: Optional[optuna.Trial]  # Used for optuna hyperparameter optimization
    optuna_loss_term_scaling: Optional[dict[str, float]]  # Scaling factors for loss terms in optuna trials

    @staticmethod
    def default() -> "SAC_Parameters":
        return {
            "experiment": "default",
            "name": "default",
            "run": None,
            "cheap_saving_interval": 10,
            "expensive_saving_interval": 200,
            "world_model": {
                "name": "pretrained_mtf",
                "epoch": -1,
            },
            "fine_tuning_path": None,
            "batches_per_epoch": 1,
            "ffn_activation": "ReLU",
            "clip_target_q_values": True,
            "q_ffn_hidden_dims": [128, 64],
            "q_hierarchisation": {
                "scheme": "none",
                "keep_subject_parameters": False,
                "feature_dimension": 1
            },
            "policy_ffn_hidden_dims": [128, 64],
            "policy_hierarchisation": {
                "scheme": "none",
                "keep_subject_parameters": False,
                "feature_dimension": 1
            },

            "env_steps_per_epoch": 512,
            "init_log_temperature": 0,
            "adjustable_temperature": True,
            "target_entropy": -1.0,
            "polyak_tau": 0.005,
            "replay_buffer_max_size_per_measurement": 100000,
            "goal_decoders": [],
            "discount_factor": 0.99,
            "rollout_noise": 0.0,
            "modality_specific_evaluators": None,
            "modality_independent_evaluators": [],
            "general_evaluators": [],
            "evaluation_set_for_metrics": "both",
            "evaluation_set_for_plots": "both",
            "use_expectation_value_for_metrics": True,
            "n_samples_for_metrics": 1,
            "use_expectation_value_for_plots": True,
            "n_samples_for_plots": 1,
            "plot_per_subject": True,
            "evaluation_test_timeseries_indices": None,
            "evaluation_train_timeseries_indices": None,
            "dataloader_num_workers": 0,
            "move_data_to_device_at_init": False,
            "learning_rate": 1e-3,
            "lr_scheduling": {
                "name": "MultiStepLR",
                "hyperparameters": {
                    "gamma": 0.1,
                    "milestones": [0.2, 0.8],
                },
            },
            "n_epochs": 5000,
            "gradient_clipping": 0,
            "action_penalty": 0.1,
            "load_model_path": None,
            "resume_epoch": None,
            "value_scheduling": {
                "alpha_q": {
                    "name": "constant",
                    "hyperparameters": {
                        "initial": 1,  # The initial alpha value for GTF.
                    },
                },
                "alpha_policy": {
                    "name": "constant",
                    "hyperparameters": {
                        "initial": 1,
                    },
                },
                "alpha_reg_q": {
                    "name": "constant",
                    "hyperparameters": {
                        "initial": 1,
                    },
                },
                "alpha_reg_policy": {
                    "name": "constant",
                    "hyperparameters": {
                        "initial": 1,
                    },
                },
                "alpha_temperature": {
                    "name": "constant",
                    "hyperparameters": {
                        "initial": 1,
                    },
                },
                "action_penalty": {
                    "name": "constant",
                    "hyperparameters": {
                        "initial": 1,
                    },
                },
            },
            "comment": None,
            "optuna_trial": None,
            "optuna_loss_term_scaling": None,
        }


ParameterOption = Union[Parameters, SAC_Parameters]
