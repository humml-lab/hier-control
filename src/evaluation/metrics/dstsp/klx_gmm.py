from typing import Optional

import torch as tc


def exp_pairwise_kl_scalar_cov(
    z: tc.Tensor, mu: tc.Tensor, cov_scaling: float, batch_size: Optional[int] = None
) -> tc.Tensor:
    n1, d = z.size()
    n2 = mu.size(0)

    # If batch_size is None or if tensors are small enough, compute all at once
    if batch_size is None or (n1 * n2 * d < 1e9):
        D_kl = 0.5 * tc.cdist(z, mu) ** 2 / cov_scaling
        return tc.exp(-D_kl).mean(dim=1)

    # Initialize output tensor
    result = tc.zeros(n1, device=z.device, dtype=z.dtype)

    # Process z in batches
    for i in range(0, n1, batch_size):
        end_i = min(i + batch_size, n1)
        z_batch = z[i:end_i]

        # Compute distance matrix for this batch (full mu)
        D_kl_batch = 0.5 * tc.cdist(z_batch, mu) ** 2 / cov_scaling

        # Compute exponential and average over mu dimension
        exp_batch = tc.exp(-D_kl_batch)
        result[i:end_i] = exp_batch.mean(dim=1)

    return result


def state_space_divergence_gmm(
    X_gen: tc.Tensor,
    X_true: tc.Tensor,
    cov_scaling: float = 1.0,
    max_ts_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> tc.Tensor:
    """
    Compute the variatonal approximation of the Kullback-Leibler divergence between two Gaussian Mixture Models (GMMs)
    according to Hershey & Olsen (2007) for the special case where all Gaussian components have a `cov_scaling` * I covariance matrix.

    Used to compute the state space divergence between two ground truth and model generated attractors in state space (Koppe et al. (2017), Brenner et al. 2022), where
    attractor invariant measures are approximated by GMMs centered at trajectory points.

    References:
        - Hershey & Olsen (2007) https://ieeexplore.ieee.org/document/4218101
        - Koppe et al. (2017) https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007263
        - Brenner et al. (2022) https://proceedings.mlr.press/v162/brenner22a.html


    Args:
        X_gen (torch.Tensor): Generated samples, shape (T1, N)
        X_true (torch.Tensor): True samples, shape (T2, N)
        cov_scaling (float): Variance of Gaussian compartments. Controls sensitivity of the measure.
        max_ts_samples (int, optional): Maximum number of samples to consider. If `None`, all samples are considered, otherwise a random subset is selected.
        batch_size (int, optional): Batch size for distance matrix computation. If `None`, all distances are computed in one batch.

    Returns:
        tc.Tensor: D_stsp
    """
    assert X_true.ndim == X_gen.ndim == 2, (
        f"X_true and X_gen must be 2D tensors but have shapes {X_true.size()} and {X_gen.size()}"
    )
    assert cov_scaling > 0, f"cov_scaling must be positive but is {cov_scaling}"

    # compute device
    device = X_true.device

    mu_inf = X_true
    mu_gen = X_gen
    if max_ts_samples is not None:
        assert max_ts_samples > 0, (
            f"If max_ts_samples is not None, it must be a positive integer but is {max_ts_samples}"
        )
        # subsample both ts
        indices1 = tc.randperm(X_true.size(0), device=device)[:max_ts_samples]
        indices2 = tc.randperm(X_gen.size(0), device=device)[:max_ts_samples]
        mu_inf = X_true[indices1]
        mu_gen = X_gen[indices2]

    # denominator in variatonal approximation
    denominator = exp_pairwise_kl_scalar_cov(mu_inf, mu_gen, cov_scaling, batch_size)

    # if all zeros no need to calculate nominator at all
    zero_mask = denominator == 0
    if zero_mask.all():
        return tc.tensor(float("inf"), device=denominator.device)

    # nominator in variational approximation
    nominator = exp_pairwise_kl_scalar_cov(mu_inf, mu_inf, cov_scaling, batch_size)

    # avoid NaNs due to underflow in exp
    D_kl_gmm = tc.mean(tc.log(nominator[~zero_mask]) - tc.log(denominator[~zero_mask]))

    return D_kl_gmm
