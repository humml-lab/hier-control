import argparse
import json
import os
import sys
import matplotlib
import optuna

# from pathlib import Path
import torch as tc
from tensorboardX import SummaryWriter

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
)  # needed for the environment
from typing import TYPE_CHECKING, Optional
from src import utils
from src.bptt import BPTT
from src.sac import SAC
from src.dataset import TimeseriesDataset
from src.hyperparameter_search.terminate_run_error import TerminateRunError

if TYPE_CHECKING:
    from src.models.dsr_model.dsr_model import DifferentDynamicsOption
    from src.models.dsr_model.hierarchisation_schemes.base import (
        DSRHierarchisationScheme,
    )
from src.parameters import ParameterOption, Parameters, SAC_Parameters

tc.set_num_threads(2) 

def get_args():
    parser = argparse.ArgumentParser(description="MTF Training")

    parser.add_argument("--run_directory", type=str)
    parser.add_argument("--device_id", type=int, default=None)
    parser.add_argument("--use_torch_compile", type=bool, default=False)

    return parser.parse_args()


def check_arg_constraints(args: "ParameterOption"):
    """
    Check if choice of given arguments satisfy constraints,
    which were assumed in model and during training
    """

    def assert_positive_or_none(arg):
        if arg is not None:
            assert arg > 0

    assert args["n_epochs"] > 0

    # TODO: add other constraints
    # if args["n_bases"] is not None:
    #     assert args["n_bases"] >= 0
    # assert args["alpha_reg"] >= 0
    # assert args["n_states_reg"] >= 0


def check_deprecated_args(args: "Parameters"):
    # check if clip_range is provided to latent_step
    lat_step = args["latent_step"]
    model_name = lat_step.get("name", None)
    lat_hypers = lat_step.get("hyperparameters", None)
    clip_range = lat_hypers.get("clip_range", None)
    if clip_range is not None:
        print(
            "Warning: clip_range is deprecated and has been removed. The argument has no effect."
        )

    layer_norm = lat_hypers.get("layer_norm", None)
    if layer_norm is not None:
        if model_name == "PLRNN":
            print(
                "Warning: layer_norm is deprecated and has been removed. The argument has no effect. Use 'mean_centering' instead."
            )
        else:
            print(
                "Warning: layer_norm is deprecated and has been removed. The argument has no effect."
            )


def check_evaluator_names(args: "ParameterOption"):
    """
    Ensure all evaluators have a 'name' parameter. If missing, generate a default name.

    Parameters:
    - args: Parameters object containing 'evaluators' and 'decoder' configurations.
    """
    evaluators = args["modality_specific_evaluators"]
    missing_names = False

    if not evaluators:
        return  # No evaluators to check

    for measurement_idx, measurement_evaluators in enumerate(evaluators):
        for modality_idx, modality_eval in enumerate(measurement_evaluators):
            # Check if 'name' is missing and generate it if necessary
            param_block = modality_eval.get("parameters") or modality_eval.get("hyperparameters")
            if "name" not in param_block:
                missing_names = True
                evaluator_name = modality_eval.get(
                    "name", f"evaluator_{measurement_idx}"
                )  # fallback name
                param_block["name"] = f"{evaluator_name}_{modality_idx}"

    if missing_names:
        print("Warning: Not all evaluator names were set. Default names have been generated.")


def calculate_can_leave_out_timeseries(
    dsr_hierarchisation_scheme: "DSRHierarchisationScheme",
    different_dynamics: "DifferentDynamicsOption",
):
    # TODO: add option to make it a list[list[int | list[int]]] with inner most list being groups of timeseries with same dynamics
    return (
        dsr_hierarchisation_scheme["scheme"] == "none" or different_dynamics != "per-timeseries"
    )  # Right now this is the only option, if TODO is implemented, additional logic is needed


def init_dataset(parameters: "ParameterOption", device: tc.device) -> TimeseriesDataset:
    """
    Initialize the dataset based on the parameters provided.
    Has its own function to be able to use it from the evaluation script as well.
    """
    measurement_folders = parameters["measurement_folders"]

    if measurement_folders is None:
        measurement_folders = os.listdir(parameters["base_path"])

    dataset_device = device if parameters.get("move_data_to_device_at_init", False) else tc.device("cpu")
    dataset = TimeseriesDataset(
        [
            os.path.join(parameters["base_path"], measurement_folder)
            for measurement_folder in measurement_folders
        ],
        parameters["seq_len"],
        parameters["batching"]["subbatch_size"],
        parameters["batching"]["subbatch_number"],
        parameters["batches_per_epoch"],
        calculate_can_leave_out_timeseries(
            parameters["dsr_hierarchisation_scheme"],
            parameters["dsr_different_dynamics"],
        ),
        excluded_filenames=parameters["excluded_filenames"],
        test_timeseries_indices=parameters["test_timeseries_indices"],
        train_test_split=parameters["train_test_split"],
        evaluator_test_timeseries_indices=parameters["evaluation_test_timeseries_indices"],
        evaluator_train_timeseries_indices=parameters["evaluation_train_timeseries_indices"],
        device=dataset_device,
        dataloader_device=device,
        dataloader_num_workers=parameters.get("dataloader_num_workers", 0),
    )
    return dataset


def check_encoder_decoder_consistency(args: "Parameters"):
    """
    Check that if the encoder is Identity, all decoders must also be Identity.
    Raises an error if this constraint is violated.
    """
    # Check if encoder is Identity
    encoder = args["encoder"]
    is_identity_encoder = isinstance(encoder, dict) and encoder.get("name") == "Identity"

    # Check if all decoders are Identity
    are_all_decoders_identity = True
    for measurement_decoders in args["decoder"]:
        for modality_decoders in measurement_decoders:
            if isinstance(modality_decoders, str):
                if modality_decoders != "Identity":
                    are_all_decoders_identity = False
                    break
            elif isinstance(modality_decoders, dict):
                if modality_decoders.get("name") != "Identity":
                    are_all_decoders_identity = False
                    break
            else:
                are_all_decoders_identity = False
                break
        if not are_all_decoders_identity:
            break

    # Enforce constraint
    if is_identity_encoder and not are_all_decoders_identity:
        raise ValueError(
            "If encoder is Identity, all decoders must also be Identity. "
            "Please update your decoder configuration to use Identity decoders only, or change "
            "your encoder to a non-Identity encoder."
        )


def start_mtf_training(
    parameters: "ParameterOption",
    device_id: Optional[int],
    use_torch_compile: bool,
):
    # get context-dependent torch.device instance
    device = utils.prepare_device(device_id)

    # create save directories & initialize writer for saving in tensorboard
    writer = SummaryWriter(parameters["run_directory"])

    # save args to tensorboard
    utils.save_parameters(parameters, parameters["run_directory"])

    dataset = init_dataset(parameters, device)
    training_algorithm = BPTT(
        parameters,
        dataset,
        writer,
        parameters["run_directory"],
        device,
        use_torch_compile,
    )

    # start training, return only interesting for optuna
    return training_algorithm.train()

def start_sac_training(
    parameters: "ParameterOption",
    device_id: Optional[int],
    use_torch_compile: bool,
):
    # get context-dependent torch.device instance
    device = utils.prepare_device(device_id)

    # create save directories & initialize writer for saving in tensorboard
    writer = SummaryWriter(parameters["run_directory"])

    # save args to tensorboard
    utils.save_parameters(parameters, parameters["run_directory"])

    mtf_parameters = json.load(open(f"{parameters['world_model']['path']}/config.json"))
    # affects the number of sequences loaded for filling the replay buffer, not the SAC batch size
    mtf_parameters["batching"]["subbatch_size"] = 512
    mtf_parameters["batching"]["subbatch_number"] = parameters["batching"]["subbatch_number"]
    mtf_dataset = init_dataset(mtf_parameters, device)

    assert len(mtf_dataset.measurement_datasets) == len(parameters["goal_decoders"]), "Number of measurements must agree between world model dataset and SAC goal decoders."
    # measurements without goal_state should not be sampled from
    mtf_dataset.dataloader.batch_sampler.all_measurement_indices = [i for i in range(len(mtf_dataset.measurement_datasets)) if not len(parameters["goal_decoders"][i][0]["goal_state"]) == 0]
    mtf_dataset.evaluator_train_samples = [train_sample for train_sample in mtf_dataset.evaluator_train_samples if train_sample.measurement_index in mtf_dataset.dataloader.batch_sampler.all_measurement_indices]
    mtf_dataset.evaluator_test_samples = [test_sample for test_sample in mtf_dataset.evaluator_test_samples if test_sample.measurement_index in mtf_dataset.dataloader.batch_sampler.all_measurement_indices]
    training_algorithm = SAC(
        parameters,
        mtf_dataset,
        writer,
        parameters["run_directory"],
        device,
        use_torch_compile,
    )

    # start training, return only interesting for optuna
    return training_algorithm.train()

def run(
    parameters: "ParameterOption",
    device_id: Optional[int] = None,
    use_torch_compile: bool = False,
):
    """
    Run the training with the given parameters.
    """
    # GPU optimizations
    tc.set_float32_matmul_precision("high")  # use tensor cores if available
    tc.backends.cudnn.benchmark = True  # scan for best convolution algorithm

    matplotlib.use("Agg")
    tc.manual_seed(parameters["run"]*43)

    if parameters.__typed_class__ is Parameters:
        check_arg_constraints(parameters)
        check_deprecated_args(parameters)
        check_evaluator_names(parameters)
        start_mtf_training(parameters, device_id, use_torch_compile)
    elif parameters.__typed_class__ is SAC_Parameters:
        start_sac_training(parameters, device_id, use_torch_compile)


def run_with_optuna(
    parameters: "ParameterOption",
    trial: optuna.Trial,
    loss_term_scaling: Optional[dict[str, float]] = None,
    device_id: Optional[int] = None,
    use_torch_compile: bool = False,
):
    """
    Run the training with Optuna for hyperparameter optimization.
    This function is a placeholder and should be implemented with actual Optuna logic.
    """

    tc.manual_seed(parameters["run"]*43)
    parameters["optuna_trial"] = trial

    if parameters.__typed_class__ is Parameters:
        parameters["optuna_loss_term_scaling"] = loss_term_scaling or {
        "dsr": 1.0,
        }
        # check parameters for various constraints
        check_arg_constraints(parameters)
        check_deprecated_args(parameters)
        check_evaluator_names(parameters)
        check_encoder_decoder_consistency(parameters)
        try:
            return start_mtf_training(parameters, device_id, use_torch_compile)
        except TerminateRunError as e:
            print(f"Optuna trial terminated with following message:\n{e.message}")
            raise optuna.TrialPruned()
    elif parameters.__typed_class__ is SAC_Parameters:
        parameters["optuna_loss_term_scaling"] = loss_term_scaling or {
        "neg_expected_reward": 1.0,
        }
        try:
            return start_sac_training(parameters, device_id, use_torch_compile)
        except TerminateRunError as e:
            print(f"Optuna trial terminated with following message:\n{e.message}")
            raise optuna.TrialPruned()
    else:
        raise ValueError(f"Unknown parameter class: {parameters.__typed_class__}")



if __name__ == "__main__":
    args = get_args()
    parameters: "ParameterOption" = json.load(open(f"{args.run_directory}/config.json"))
    parameters["run_directory"] = args.run_directory
    run(
        parameters,
        args.device_id,
        args.use_torch_compile,
    )
