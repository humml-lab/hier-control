import copy
import json
import os
import sys
from argparse import ArgumentParser
from graphlib import CycleError, TopologicalSorter
from typing import Optional
# add parent parent directory to path to allow imports from src
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
from src.utils import TypedDictWrapper

import dill
import optuna

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
)  # Setting repository root directory as environment patch. Matching the behaviour in main.py

from src.hyperparameter_search.search_space.optuna_search_space import (
    Dependency,
    OptunaFloat,
    OptunaInt,
    OptunaSearchSpace,
)
from src.hyperparameter_search.utils import (
    find_search_spaces,
    get_value_at_path,
    path_exists,
    set_value_at_path,
)
from src.main import run_with_optuna


def get_true_dependency_path(config: dict, dependency_path: str):
    """
    Returns the sub-path of the dependency that exists in the configuration.

    Useful for cases where the value depended on is inside a categorical search space.
    If it turns that this is not the case, the code throws an error in a later function.

    If the dependency is not found, it will raise a ValueError.
    """
    while not path_exists(config, dependency_path):
        if "." not in dependency_path:
            raise ValueError(f"Dependency '{dependency_path}' not found in the configuration.")
        dependency_path = ".".join(dependency_path.split(".")[:-1])
    return dependency_path


def get_dependency_value(config: dict, dependency: str | dict | int | float):
    """
    Gets the value at the dependency path in the configuration.
    If a transformation is provided, it applies it to the value.
    """
    if isinstance(dependency, dict):
        dependency: Dependency[int | float] = dependency
        value = get_value_at_path(config, dependency["path"])
        if "transformation" in dependency:
            value = dependency["transformation"](value)
    elif isinstance(dependency, str):
        value = get_value_at_path(config, dependency)
    elif isinstance(dependency, int) or isinstance(dependency, float):
        value = dependency
    else:
        raise TypeError(f"Dependency must be a string or a dict, got {type(dependency).__name__}.")
    return value


def get_dependencies_from_int_or_float(
    config: dict,
    args: OptunaInt | OptunaFloat,
):
    dependencies = set()
    if isinstance(args["min"], str):
        dependency_path = get_true_dependency_path(config, args["min"])
        dependencies.add(dependency_path)
    elif isinstance(args["min"], dict):
        dependency_path = get_true_dependency_path(config, args["min"]["path"])
        dependencies.add(dependency_path)
    if isinstance(args["max"], str):
        dependency_path = get_true_dependency_path(config, args["max"])
        dependencies.add(dependency_path)
    elif isinstance(args["max"], dict):
        dependency_path = get_true_dependency_path(config, args["max"]["path"])
        dependencies.add(dependency_path)
    return dependencies


def sort_search_spaces_by_dependencies(
    paths_with_search_spaces: list[tuple[list[str | int], OptunaSearchSpace]],
    config: dict,
):
    """
    Sorts the search spaces based on their dependencies on each other using topological sorting.

    Returns the sorted paths and search spaces.

    Raises a `ValueError` if a cyclic dependency is detected.
    """
    dependencies: dict[str, set[str]] = {}
    search_space_lookup: dict[str, OptunaSearchSpace] = {}
    path_lookup: dict[str, list[str | int]] = {}

    for path, space_obj in paths_with_search_spaces:
        key = ".".join(map(str, path))
        dependencies[key] = set()
        search_space_lookup[key] = space_obj
        path_lookup[key] = path
        if space_obj.type == "int" or space_obj.type == "float":
            dependencies[key].update(get_dependencies_from_int_or_float(config, space_obj.args))
        elif space_obj.type == "list":
            # num_entries is always an OptunaInt
            dependencies[key].update(
                get_dependencies_from_int_or_float(config, space_obj.args["num_entries"])
            )

            # entry can be an Categorical, which does not support dependencies
            if space_obj.args["entry"].get("choices") is None:
                dependencies[key].update(
                    get_dependencies_from_int_or_float(config, space_obj.args["entry"])
                )

    ts = TopologicalSorter(dependencies)
    try:
        sorted_order = tuple(ts.static_order())
    except CycleError:
        raise ValueError(
            "Cyclic dependency detected in the Optuna search spaces. Please check the dependencies."
        )

    # Sort the search spaces and paths based on the topological order
    sorted_spaces = [
        search_space_lookup[key] for key in sorted_order if key in search_space_lookup
    ]
    sorted_paths = [path_lookup[key] for key in sorted_order if key in path_lookup]

    return sorted_paths, sorted_spaces


def suggest_int_or_float(
    config: dict,
    trial: optuna.trial.Trial,
    search_space: OptunaSearchSpace,
    label_suffix: str = "",
):
    min_value = get_dependency_value(config, search_space.args["min"])
    max_value = get_dependency_value(config, search_space.args["max"])
    suggest_function = trial.suggest_int if search_space.type == "int" else trial.suggest_float
    return suggest_function(
        f"{search_space.label}{label_suffix}",
        min_value,
        max_value,
        step=search_space.args.get("step", 1),
        log=search_space.args.get("log", False),
    )


def generate_optuna_configuration(
    base_config: dict, trial: optuna.trial.Trial
) -> tuple[str, dict]:
    """
    Generates a named configuration for a single Optuna trial based on the base config.
    """
    paths_with_search_spaces = find_search_spaces(base_config)

    if not paths_with_search_spaces:
        return "base", base_config

    new_config = copy.deepcopy(base_config)
    name_parts = []

    # Sort the search spaces based on dependencies
    # This ensures that when we sample a hyperparameter that depends on another,
    # it is sampled after its dependencies have been sampled.
    sorted_paths, sorted_spaces = sort_search_spaces_by_dependencies(
        paths_with_search_spaces,
        new_config,
    )

    for path_list, search_space in zip(sorted_paths, sorted_spaces):
        # Suggest the value based on the search space type
        match search_space.type:
            case "int" | "float":
                value = suggest_int_or_float(new_config, trial, search_space)
            case "list":
                num_entries = suggest_int_or_float(
                    new_config, trial, search_space.args["num_entries"], label_suffix="_num"
                )
                values = []
                for i in range(num_entries):
                    if not search_space.is_categorical():
                        suggest_int_or_float(
                            new_config,
                            trial,
                            search_space.args["entry"],
                            label_suffix=f"entry_{i}",
                        )
                    else:
                        value = trial.suggest_categorical(
                            f"{search_space.label}_entry_{i}",
                            search_space.args["entry"]["choices"],
                        )
                    values.append(value)
            case "categorical":
                value = trial.suggest_categorical(search_space.label, search_space.args["choices"])
            case _:
                raise ValueError(f"Unsupported Optuna search space type: {search_space.type}")

        # Set the suggested value in the new configuration
        set_value_at_path(new_config, path_list, value)

        # Append to the experiment name
        if search_space.label:
            formatted_value = search_space.value_formatter(value)
            name_parts.append(f"{search_space.label}={formatted_value}")

    experiment_name = "params_" + ("_".join(name_parts) or "base")

    return experiment_name, new_config


class Objective:
    def __init__(
        self,
        parameters: dict,
        results_dir: str,
        experiment_name: str,
        use_torch_compile: bool,
        gpu_id: Optional[int],
        loss_term_scaling: Optional[dict[str, float]],
    ):
        self.parameters = parameters
        self.results_dir = results_dir
        self.experiment_name = experiment_name
        self.gpu_id = gpu_id
        self.use_torch_compile = use_torch_compile
        self.loss_term_scaling = loss_term_scaling

    def __call__(self, trial: optuna.trial.Trial) -> float:
        name, config = generate_optuna_configuration(self.parameters, trial)
        run = 1
        if os.path.exists(os.path.join(self.results_dir, self.experiment_name, name)):
            # If the directory already exists, increment the run number
            existing_runs = [
                int(d)
                for d in os.listdir(os.path.join(self.results_dir, self.experiment_name, name))
                if d.isdigit()
            ]
            run = max(existing_runs) + 1 if existing_runs else 1
        parameter_class = self.parameters.__typed_class__

        config = TypedDictWrapper(
            {
            **copy.deepcopy(parameter_class.default()),
            **config,
            "run_directory": os.path.join(
                self.results_dir, self.experiment_name, name, f"{run:03d}"
            ),
            "experiment": self.experiment_name,
            "name": name,
            "run": run,

            },
            parameter_class)

        os.makedirs(config["run_directory"], exist_ok=True)
        # Write the configuration to a JSON file
        with open(os.path.join(config["run_directory"], "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        return run_with_optuna(
            config,
            trial,
            self.loss_term_scaling,
            self.gpu_id,
            self.use_torch_compile,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to the base configuration pickle file.", required=True
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results.",
        required=True,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="optuna_experiment",
        help="Name of the experiment.",
        required=True,
    )
    parser.add_argument(
        "--use_torch_compile",
        action="store_true",
        help="Whether to use torch.compile for optimization.",
        default=False,
    )
    parser.add_argument("--study_name", type=str, help="Name of the Optuna study.", required=True)
    parser.add_argument(
        "--storage",
        type=str,
        help="Name of the sqlite database for Optuna storage. It is assumed to be in the experiment directory.",
        required=True,
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU ID to use for training. If not given, no GPU is used.",
    )
    parser.add_argument("--num_trials", type=int, required=True, help="Number of trials to run.")
    parser.add_argument(
        "--optuna_loss_scaling_path",
        type=str,
        default=None,
        help="Path to the Optuna loss scaling configuration file.",
    )

    args = parser.parse_args()
    config = dill.load(open(args.config, "rb"))
    loss_scaling = None
    if args.optuna_loss_scaling_path is not None:
        loss_scaling = dill.load(open(args.optuna_loss_scaling_path, "rb"))

    study = optuna.load_study(
        study_name=args.study_name,
        storage=f"sqlite:///{os.path.join(args.results_dir, args.experiment_name, args.storage)}",
        pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=4),
        sampler=optuna.samplers.TPESampler(multivariate=True),
    )
    study.optimize(
        Objective(
            parameters=config,
            results_dir=args.results_dir,
            experiment_name=args.experiment_name,
            use_torch_compile=args.use_torch_compile,
            gpu_id=args.gpu_id,
            loss_term_scaling=loss_scaling,
        ),
        n_trials=args.num_trials,
    )
