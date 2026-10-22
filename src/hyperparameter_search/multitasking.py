import copy
import itertools
import json
import multiprocessing as mp
import os
import subprocess
import sys
from typing import Any, Literal, Optional

import dill
import optuna

import src.hyperparameter_search.optuna_worker
import src.main
from src import utils
from src.hyperparameter_search.search_space.search_space import SearchSpace

from .search_space.parameter import Parameter
from .utils import find_int_keyed_dicts, find_search_spaces, set_value_at_path


def get_available_gpus() -> list[int]:
    return [
        gpu
        for gpu, util in utils.get_current_gpu_utilization().items()
        if util["utilization"] < 1.0 and util["memory_free"] > 5000
    ]


def generate_named_configurations(
    base_config: dict,
    get_all_combinations: bool,
) -> list[tuple[str, dict]]:
    """
    Generates a list of (name, resolved_config) tuples from a base config
    containing SearchSpace objects.
    """
    # 1. Find all search spaces and their locations (paths)
    # The helper now returns the entire SearchSpace object
    search_spaces_with_paths = find_search_spaces(base_config)

    if not search_spaces_with_paths:
        return [("base", base_config)]

    paths, search_space_objects = zip(*search_spaces_with_paths)
    search_space_objects: list[SearchSpace]

    # 2. Generate all combinations of values from the search spaces
    if get_all_combinations:
        value_lists = [s.values for s in search_space_objects]
        value_combinations = list(itertools.product(*value_lists))
    # Or itarate through each search space's values
    # and create a tuple for each index
    else:
        length = len(search_space_objects[0].values)
        for i in range(len(search_space_objects)):
            if len(search_space_objects[i].values) != length:
                raise ValueError(
                    f"All search spaces must have the same number of values when not getting all combinations. This is not true for {search_space_objects[i].label}."
                )
        value_combinations = [
            tuple(s.values[i] for s in search_space_objects) for i in range(length)
        ]

    # 3. Generate a named configuration for each combination
    named_configs = []
    for combo in value_combinations:
        new_config = copy.deepcopy(base_config)
        name_parts = []

        # Iterate through this specific combination to build the name and config
        for i, value in enumerate(combo):
            space_obj = search_space_objects[i]
            path = paths[i]

            # Set the concrete value in the new config dict
            set_value_at_path(new_config, path, value)

            # If the space has a label, add it to the experiment name
            if space_obj.label:
                formatted_value = space_obj.value_formatter(value)
                name_parts.append(f"{space_obj.label}={formatted_value}")

        experiment_name = "params_" + ("_".join(name_parts) or "base")
        named_configs.append((experiment_name, new_config))

    return named_configs


def run_multitask(
    results_dir: str,
    experiment_name: str,
    n_runs: int,
    parameters: list[Parameter] | dict[str, Any],
    n_processes: int,
    gpu_mode: Literal["no-gpu", "automatic-distribution"] | list[int],
    n_process_per_gpu: Optional[int],
    use_torch_compile: bool,
    use_optuna: bool,
    optuna_num_trials: int,
    optuna_loss_scaling: Optional[dict[str, float]],
    get_all_gridsearch_combinations: bool,
):
    use_legacy_calculations = not isinstance(parameters, dict)
    if not isinstance(parameters, (list, dict)):
        raise ValueError("Parameters must be a list of Parameter objects or a dictionary.")

    assert not use_legacy_calculations or not use_optuna, (
        "Optuna is not compatible with the legacy parameter format."
    )
    os.makedirs(results_dir, exist_ok=True)
    if use_legacy_calculations or not use_optuna:
        multitask_gridsearch(
            results_dir,
            experiment_name,
            n_runs,
            parameters,
            n_processes,
            gpu_mode,
            n_process_per_gpu,
            use_torch_compile,
            use_legacy_calculations,
            get_all_gridsearch_combinations,
        )
    else:
        int_keyed_dict_paths = find_int_keyed_dicts(parameters)
        assert int_keyed_dict_paths == [], (
            "There are integer-keyed dictionaries in the parameters dict, which is not supported."
            " These were found at the following paths: "
            + "\n".join([".".join(map(str, p)) for p in int_keyed_dict_paths])
        )
        optuna_multitask(
            results_dir,
            experiment_name,
            parameters,
            n_processes,
            gpu_mode,
            n_process_per_gpu,
            use_torch_compile,
            optuna_num_trials,
            optuna_loss_scaling,
        )


def multitask_gridsearch(
    results_dir: str,
    experiment_name: str,
    n_runs: int,
    parameters: list[Parameter] | dict[str, Any],
    n_processes: int,
    gpu_mode: Literal["no-gpu", "automatic-distribution"] | list[int],
    n_process_per_gpu: int,
    use_torch_compile: bool,
    use_legacy_calculations: bool,
    get_all_gridsearch_combinations: bool,
):
    run_directory_paths: list[str] = []

    # Load the default configuration
    default_config = parameters.__typed_class__.default()

    if use_legacy_calculations:
        for arg in parameters:
            assert len(arg.values) > 0 and (len(arg.values) == 1 or arg.label), (
                "If an argument has multiple values, you must specify a label."
            )

        # Loop through all combinations of parameter values
        for combination in itertools.product(*[argument.values for argument in parameters]):
            config = copy.deepcopy(default_config)
            trial_name = "params"
            for arg, value in zip(parameters, combination):
                if arg.label is not None:
                    trial_name += f"_{arg.label}={arg.value_formatter(value)}"
                config[arg.name] = value
            config["experiment"] = experiment_name
            config["name"] = trial_name
            # Create one directory for each run
            for run in range(1, n_runs + 1):
                config["run"] = run
                run_dir = os.path.join(results_dir, experiment_name, trial_name, f"{run:03d}")
                os.makedirs(run_dir, exist_ok=True)
                # Build a filename that reflects the parameter combination
                file_path = os.path.join(run_dir, "config.json")

                # Write the merged configuration to a JSON file
                with open(file_path, "w") as f:
                    json.dump(config, f, indent=4)

                run_directory_paths.append(run_dir)

    else:
        for name, config in generate_named_configurations(
            parameters,
            get_all_gridsearch_combinations,
        ):
            config = {
                **copy.deepcopy(default_config),
                **config,
                "experiment": experiment_name,
                "name": name,
            }
            for run in range(1, n_runs + 1):
                config["run"] = run
                run_dir = os.path.join(results_dir, experiment_name, name, f"{run:03d}")
                os.makedirs(run_dir, exist_ok=True)
                # Build a filename that reflects the parameter combination
                file_path = os.path.join(run_dir, "config.json")

                # Write the merged configuration to a JSON file
                with open(file_path, "w") as f:
                    json.dump(config, f, indent=4)

                run_directory_paths.append(run_dir)

    # Prepare GPU assignments if needed
    device_id_per_run = [None] * len(run_directory_paths)
    workers = n_processes

    if gpu_mode != "no-gpu":
        device_ids = gpu_mode if isinstance(gpu_mode, list) else get_available_gpus()
        if not device_ids:
            raise RuntimeError("No free GPUs available.")
        assert n_process_per_gpu is not None, "n_process_per_gpu must be set for GPU mode."

        # Build GPU slots and clamp worker count
        slots = []
        per_gpu = min(n_process_per_gpu, n_processes)
        for gpu in device_ids:
            slots.extend([gpu] * per_gpu)

        workers = min(n_processes, len(slots))
        if workers < n_processes:
            print(
                f"Requested {n_processes} processes but only {len(slots)} GPU slots available."
                f" Reducing to {workers}."
            )

        # Round-robin assign device IDs for each run
        device_id_per_run = [slots[i % workers] for i in range(len(run_directory_paths))]

    # Launch processes
    pool = mp.Pool(processes=workers)
    pool.starmap(
        run_subprocess,
        [
            (run_dir, device_id_per_run[i], gpu_mode, use_torch_compile)
            for i, run_dir in enumerate(run_directory_paths)
        ],
        chunksize=1,
    )
    pool.close()
    pool.join()


def optuna_multitask(
    results_dir: str,
    experiment_name: str,
    config: dict[str, Any],
    n_processes: int,
    gpu_mode: Literal["no-gpu", "automatic-distribution"] | list[int],
    n_process_per_gpu: int,
    use_torch_compile: bool,
    optuna_num_trials: int,
    optuna_loss_scaling: Optional[dict[str, float]] = None,
):
    os.makedirs(os.path.join(results_dir, experiment_name), exist_ok=True)
    config_path = os.path.join(results_dir, experiment_name, "parameters.pkl")
    dill.dump(
        config,
        open(config_path, "wb"),
    )

    loss_scaling_path = os.path.join(results_dir, experiment_name, "optuna_loss_scaling.pkl")
    dill.dump(
        optuna_loss_scaling,
        open(loss_scaling_path, "wb"),
    )
    # Use Optuna for hyperparameter optimization
    storage_name = "optuna_study.db"
    optuna.create_study(
        storage=f"sqlite:///{os.path.join(results_dir, experiment_name, storage_name)}",
        pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=4),
        sampler=optuna.samplers.TPESampler(multivariate=True),
        study_name=experiment_name,
        load_if_exists=True,
        direction="minimize",
    )
    workers = n_processes

    if gpu_mode != "no-gpu":
        device_ids = gpu_mode if isinstance(gpu_mode, list) else get_available_gpus()
        if not device_ids:
            raise RuntimeError("No free GPUs available.")
        assert n_process_per_gpu is not None, "n_process_per_gpu must be set for GPU mode."

        # Build GPU slots and clamp worker count
        slots = []
        per_gpu = min(n_process_per_gpu, n_processes)
        for gpu in device_ids:
            slots.extend([gpu] * per_gpu)

        workers = min(n_processes, len(slots))
        if workers < n_processes:
            print(
                f"Requested {n_processes} processes but only {len(slots)} GPU slots available."
                f" Reducing to {workers}."
            )

        # Round-robin assign device IDs for each runner
        device_id_per_runner = [slots[i % workers] for i in range(workers)]
    else:
        device_id_per_runner = [None] * workers

    num_trials = optuna_num_trials // workers
    pool = mp.Pool(processes=workers)
    pool.starmap(
        run_optuna_subprocess,
        [
            (
                config_path,
                results_dir,
                experiment_name,
                use_torch_compile,
                experiment_name,
                storage_name,
                device_id_per_runner[i],
                num_trials,
                loss_scaling_path,
            )
            for i in range(workers)
        ],
        chunksize=1,
    )
    pool.close()
    pool.join()


def run_optuna_subprocess(
    config_path: str,
    results_dir: str,
    experiment_name: str,
    use_torch_compile: bool,
    study_name: str,
    storage: str,
    gpu_id: Optional[int],
    num_trials: int,
    optuna_loss_scaling_path: Optional[str],
):
    cmd = [
        os.path.abspath(sys.executable),
        os.path.abspath(src.hyperparameter_search.optuna_worker.__file__),
        "--config",
        config_path,
        "--results_dir",
        results_dir,
        "--experiment_name",
        experiment_name,
        "--study_name",
        study_name,
        "--storage",
        storage,
        "--num_trials",
        str(num_trials),
    ]
    if gpu_id is not None:
        cmd += ["--gpu_id", str(gpu_id)]

    if use_torch_compile:
        cmd += ["--use_torch_compile"]

    if optuna_loss_scaling_path is not None:
        cmd += ["--optuna_loss_scaling_path", optuna_loss_scaling_path]

    print(cmd)
    return subprocess.call(cmd)


def run_subprocess(run_dir, device_id, gpu_mode, use_torch_compile):
    cmd = [
        os.path.abspath(sys.executable),
        os.path.abspath(src.main.__file__),
        "--run_directory",
        run_dir,
    ]
    if gpu_mode != "no-gpu":
        cmd += ["--device_id", str(device_id)]

    if use_torch_compile:
        cmd += ["--use_torch_compile", str(use_torch_compile)]

    print(cmd)
    return subprocess.call(cmd)
