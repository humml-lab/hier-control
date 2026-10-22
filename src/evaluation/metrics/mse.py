import torch as tc


def nmse(X_gen, X_true, average_batch=False):
    return tc.mean(tc.pow(X_gen - X_true, 2)) / tc.mean(tc.pow(X_true, 2))
