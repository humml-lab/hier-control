import glob
import math
import os
import pickle
import subprocess
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

import optuna
import torch as tc

if TYPE_CHECKING:
    from src.parameters import Parameters


class TypedDictWrapper(dict):
    def __init__(self, data, typed_class):
        super().__init__(data)
        self.__typed_class__ = typed_class

def not_none[T, U](x: Optional[T], default: U) -> T | U:
    """
    Return x if it is not None, otherwise return default.
    """
    return x if x is not None else default


def index_to_batched_long_tensor(index: int | tc.Tensor):
    """
    Converts an index to a batched long tensor. Noop if the index is already a tensor.
    """
    if isinstance(index, tc.Tensor):
        return index
    return tc.tensor(index, dtype=tc.long).unsqueeze(0)


def standardize_tensor(
    X: tc.Tensor,
    dim: Optional[int | tuple[int]] = None,
    keepdim: bool = False,
):
    m = X.mean(dim=dim, keepdim=keepdim)
    s = X.std(dim=dim, keepdim=keepdim)
    return (X - m) / (s + tc.finfo(X.dtype).eps)

def reduce_tensor(
    tensor: tc.Tensor,
    reduction: Optional[Literal["mean", "sum", "median", "min", "max", "std", "var"]] = "mean",
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> tc.Tensor:
    """
    Apply reduction operation to a tensor.

    Args:
        tensor (`torch.Tensor`): Input tensor to reduce
        reduction (`str`, optional): Reduction operation to apply. Options:
            - `"mean"`: Compute mean (default)
            - `"sum"`: Compute sum
            - `"median"`: Compute median
            - `"min"`: Compute minimum
            - `"max"`: Compute maximum
            - `"std"`: Compute standard deviation
            - `"var"`: Compute variance
            - `None`: No reduction (return tensor as-is)
        dim (`int`, optional): Dimension to reduce over. If `None`, reduces over all dimensions.
        keepdim (`bool`, optional): Whether to keep the reduced dimension(s) with size 1.
            Defaults to `False`.

    Returns:
        `torch.Tensor`: Reduced tensor or original tensor if `reduction` is `None`

    Raises:
        `ValueError`: If `reduction` is not one of the supported options

    Example:
        >>> x = tc.randn(3, 4, 5)
        >>> reduce_tensor(x, "mean", dim=0)           # Reduce over first dimension
        >>> reduce_tensor(x, "sum", keepdim=True)     # Reduce over all dimensions, keep dims
        >>> reduce_tensor(x, None)                    # No reduction
    """
    if reduction is None:
        return tensor

    match reduction:
        case "mean":
            return tensor.mean(dim=dim, keepdim=keepdim)
        case "sum":
            return tensor.sum(dim=dim, keepdim=keepdim)
        case "median":
            if dim is not None:
                return tensor.median(dim=dim, keepdim=keepdim)[0]
            else:
                return tensor.median()
        case "min":
            if dim is not None:
                return tensor.min(dim=dim, keepdim=keepdim)[0]
            else:
                return tensor.min()
        case "max":
            if dim is not None:
                return tensor.max(dim=dim, keepdim=keepdim)[0]
            else:
                return tensor.max()
        case "std":
            return tensor.std(dim=dim, keepdim=keepdim)
        case "var":
            return tensor.var(dim=dim, keepdim=keepdim)
        case _:
            raise ValueError(
                f"Invalid reduction: {reduction}. "
                f"Use 'mean', 'sum', 'median', 'min', 'max', 'std', 'var', or None"
            )


def prepare_device(device_id: Optional[int]) -> tc.device:
    """
    Prepare and return a torch.device instance.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # check cuda availability
    if device_id is not None and tc.cuda.is_available():
        device = tc.device("cuda", device_id)
        name = tc.cuda.get_device_name(device_id)
        print(f"Using device {name} for training ({device}).")
    else:
        device = tc.device("cpu")
        num_threads = tc.get_num_threads()
        print(f"Training on {device} with {num_threads} threads.")

    return device


def make_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception:
            pass


def save_parameters(parameters: "Parameters", save_path: str):
    """
    Save args to a pickle file in the save_path directory.
    """

    filename = "{}/hypers.pkl".format(save_path)

    # Remove optuna trial object before saving
    if "optuna_trial" in parameters:
        new_params = {k: v for k, v in parameters.items() if k != "optuna_trial"}
    else:
        new_params = parameters

    with open(filename, "wb") as f:
        pickle.dump(new_params, f)


def get_latest_epoch(model_path: str) -> int:
    """Infer latest trainings epoch from a saved model
    Used when models are loaded from saved parameters
    """

    # get list of model checkpoints
    checkpoints = glob.glob(os.path.join(model_path, "*.pt"))
    assert checkpoints, f"No model found in {model_path}."

    # Loop over model_*.pt checkpoints
    # and find the model checkpoint of the latest epoch.
    latest_epoch = 0
    for checkpoint in checkpoints:
        epoch = int(checkpoint.split("_")[-1].strip(".pt"))
        if latest_epoch < epoch:
            latest_epoch = epoch

    return latest_epoch


def get_current_gpu_utilization():
    """
    From: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3
    Get the current gpu usage and memory information.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are dictionaries with 'utilization' (int) and 'memory_free' (int) keys.
    """
    # Get GPU utilization and memory information in a single call
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.free",
            "--format=csv,nounits,noheader",
        ],
        encoding="utf-8",
    )

    # Parse the combined output
    lines = result.strip().split("\n")
    gpu_data = []
    for line in lines:
        if line.strip():
            util, memory = line.split(", ")
            gpu_data.append((int(util), int(memory)))

    # Combine into dictionary with both utilization and memory info
    result = {}
    for i, (util, memory) in enumerate(gpu_data):
        result[i] = {"utilization": util, "memory_free": memory}

    return result


def split_tensor(
    tensor: tc.Tensor,
    split: float,
    dim: int = 0,
) -> tuple[tc.Tensor]:
    """
    Split a tensor into two parts along a given dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to be split.
    split : float
        The proportion of the first part.
    dim : int, optional
        The dimension along which to split the tensor, by default 0.

    Returns
    -------
    tuple[torch.Tensor]
        A tuple containing the two parts of the split tensor.
    """
    assert 0 < split < 1, f"split must be between 0 and 1, but got {split}"
    assert dim >= 0 and dim < tensor.dim(), (
        f"dim must be between 0 and {tensor.dim() - 1}, but got {dim}"
    )
    # Calculate the index at which to split the tensor
    split_idx = int(tensor.size(dim) * split)

    # Split the tensor into two parts
    return (
        tensor.narrow(dim, 0, split_idx),
        tensor.narrow(dim, split_idx, tensor.size(dim) - split_idx),
    )


def index_of_nested[T](lists: list[list[T]], target: T) -> int:
    """
    Find the index of the first sublist that contains the target element.
    If no such sublist exists, return -1.
    """
    for i, sublist in enumerate(lists):
        if target in sublist:
            return i
    return -1


def optuna_calculate_reported_value(
    unscaled_loss_terms: dict,
    loss_term_scales: dict,
) -> float:
    reported_value = 0
    for name, value in unscaled_loss_terms.items():
        reported_value += value * loss_term_scales.get(name, 0.0)
    return reported_value


def optuna_report(
    unscaled_loss_terms: dict,
    loss_term_scales: dict,
    trial: optuna.Trial,
    epoch: int,
):
    reported_value = optuna_calculate_reported_value(
        unscaled_loss_terms,
        loss_term_scales,
    )
    # if value is nan, report a large value to optuna
    if math.isnan(reported_value):
        reported_value = 1e10

    trial.report(
        reported_value,
        step=epoch,
    )

    if trial.should_prune():
        raise optuna.TrialPruned(
            f"Trial was pruned at epoch {epoch} with reported value {reported_value}."
        )


def init_weight_and_bias_uniform(
    shape: Union[Sequence[int], int],
    *more_shape: int,
    bias: bool = True,
    a: float = math.sqrt(5),
) -> tuple[tc.Tensor, Optional[tc.Tensor]]:
    # Normalize shape
    if isinstance(shape, int):
        dims = (shape,) + more_shape
    else:
        dims = tuple(shape)

    weight = tc.nn.init.kaiming_uniform_(tc.empty(dims), a=a)
    fan_in, _ = tc.nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in)
    bias_ = tc.nn.init.uniform_(
        tc.empty(dims[0]),
        -bound,
        bound,
    )
    if bias and fan_in != 0:
        return tc.nn.ParameterList([weight, bias_])
    else:
        return tc.nn.ParameterList([weight, tc.zeros_like(bias_)])


def l2_reg(tensor: tc.Tensor, target: Optional[tc.Tensor] = None) -> tc.Tensor:
    """
    Compute L2 regularization loss for a tensor.

    Args:
        tensor: The tensor to regularize
        target: Optional target tensor. If None, targets zero. If provided, computes L2 loss to target.

    Returns:
        L2 regularization loss: sum((tensor - target)²) if target provided, else sum(tensor²)
    """
    if target is not None:
        return tc.sum((tensor - target) ** 2)
    else:
        return tc.sum(tensor**2)


def off_diagonal(W: tc.Tensor) -> tc.Tensor:
    """
    W: ((B), M, M)
    Transform a matrix or batch of matrices to its off-diagonal part, i.e.
    set diagonal entries to 0.
    """
    # Avoid constructing a diagonal matrix; use a broadcasted mask instead.
    eye = tc.eye(W.shape[-1], dtype=tc.bool, device=W.device)
    return W.masked_fill(eye, 0)


def truncate_to_common_length(X1: tc.Tensor, X2: tc.Tensor) -> tuple[tc.Tensor, tc.Tensor]:
    """
    Truncate input tensors to common length along the time dimension.

    Args:
        X1 (tc.Tensor): First tensor to truncate (shape: (T, ...))
        X2 (tc.Tensor): Second tensor to truncate (shape: (T, ...))

    Returns:
        tuple[tc.Tensor, tc.Tensor]: Tuple containing the truncated tensors.
    """
    if X1.shape[1:] != X2.shape[1:]:
        raise ValueError(
            f"Tensors must match on non-time dimensions. Got shapes: X1={X1.shape}, X2={X2.shape}"
        )

    min_len = min(X1.shape[0], X2.shape[0])
    return X1[:min_len], X2[:min_len]
