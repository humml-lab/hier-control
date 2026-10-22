from typing import Any

from .search_space.optuna_search_space import OptunaSearchSpace
from .search_space.search_space import SearchSpace


def set_value_at_path(config: dict, path: list, value):
    """Sets a value in a nested dictionary/list at a given path."""
    d = config
    for i, key in enumerate(path[:-1]):
        # Determine if next key is an integer (list index) or string (dict key)
        next_key = path[i + 1]
        is_next_list = isinstance(next_key, int)

        if isinstance(d, list):
            # If current is a list, key must be an integer index
            if not isinstance(key, int):
                raise TypeError(f"Expected integer index for list, got {type(key).__name__}.")
            # Ensure list is large enough
            while len(d) <= key:
                d.append({} if not is_next_list else [])
            if d[key] is None:
                d[key] = {} if not is_next_list else []
            d = d[key]
        elif isinstance(d, dict):
            # If current is a dict, key must be a string
            if not isinstance(key, str):
                raise TypeError(f"Expected string key for dict, got {type(key).__name__}.")
            if d.get(key) is None:
                d[key] = {} if not is_next_list else []
            d = d[key]
        else:
            raise TypeError(f"Expected dict or list at path, got {type(d).__name__}.")

    # Set the final value
    final_key = path[-1]
    if isinstance(d, list):
        if not isinstance(final_key, int):
            raise TypeError(f"Expected integer index for list, got {type(final_key).__name__}.")
        while len(d) <= final_key:
            d.append(None)
        d[final_key] = value
    elif isinstance(d, dict):
        if not isinstance(final_key, str):
            raise TypeError(f"Expected string key for dict, got {type(final_key).__name__}.")
        d[final_key] = value
    else:
        raise TypeError(f"Expected dict or list at final path, got {type(d).__name__}.")


def get_value_at_path(config: dict, path: str) -> Any:
    """Gets a value from a nested dictionary/list at a given path."""
    d = config
    for key in path.split("."):
        if isinstance(d, list):
            d = d[int(key)]
        elif isinstance(d, dict):
            d = d.get(key)
        else:
            raise TypeError(f"Expected dict or list at path '{path}', got {type(d).__name__}.")
        if d is None:
            raise KeyError(f"Path '{path}' not found in the configuration.")
    return d


def path_exists(config: dict, path: str):
    """Checks if a path exists in the configuration."""
    try:
        get_value_at_path(config, path)
        return True
    except KeyError:
        return False


def find_search_spaces(params: Any, path: list[str | int] | None = None):
    """Recursively finds all SearchSpace instances and their paths."""
    if path is None:
        path = []

    spaces: list[tuple[list[str | int], OptunaSearchSpace | SearchSpace]] = []
    if isinstance(params, dict):
        for key, value in params.items():
            spaces.extend(find_search_spaces(value, path + [key]))
    elif isinstance(params, list):
        for i, item in enumerate(params):
            spaces.extend(find_search_spaces(item, path + [i]))
    elif isinstance(params, (SearchSpace, OptunaSearchSpace)):
        spaces.append((path, params))

    return spaces


def find_int_keyed_dicts(params: Any, path: list[str | int] | None = None):
    """Recursively finds all paths to dictionaries with integer keys."""
    if path is None:
        path = []

    paths: list[list[str | int]] = []
    if isinstance(params, dict):
        if any(isinstance(k, int) for k in params.keys()):
            paths.append(path)
        for key, value in params.items():
            paths.extend(find_int_keyed_dicts(value, path + [key]))
    elif isinstance(params, list):
        for i, item in enumerate(params):
            paths.extend(find_int_keyed_dicts(item, path + [i]))

    return paths
