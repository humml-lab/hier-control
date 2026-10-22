import warnings

import torch as tc
from sklearn.metrics import matthews_corrcoef


@tc.no_grad()
def mcc(X_pred: tc.tensor, X_true: tc.tensor) -> tc.tensor:
    # prepare data: only compute mcc over times for which ground-truth class probability is defined
    times_to_compare = X_true.any(dim=-1)
    X_true = X_true[times_to_compare]
    X_pred = X_pred[times_to_compare]
    # convert to binary (0/1) for mcc calculation
    X_pred = tc.nn.functional.one_hot(X_pred.argmax(dim=-1), num_classes=X_true.shape[-1])
    if X_true.dtype != X_pred.dtype:
        with warnings.catch_warnings():
            warnings.simplefilter("always", UserWarning)
            warnings.warn(
                f"X_true and X_pred have different types: {X_true.dtype} and {X_pred.dtype}. "
                "This may lead to unexpected behavior.",
                UserWarning,
                stacklevel=3,
            )
        X_true = X_true.to(dtype=X_pred.dtype)

    X_true = X_true.flatten().cpu()
    X_pred = X_pred.flatten().cpu()

    return tc.tensor(matthews_corrcoef(X_true, X_pred))
