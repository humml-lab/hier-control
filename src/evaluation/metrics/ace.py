import numpy as np
import scipy.stats
import torch as tc


# function to compute autocorrelation
def get_acf(x: np.ndarray, max_lag: int):
    result = np.zeros(max_lag + 1)
    result[0] = 1

    mean = np.mean(x)
    variance = np.var(x)

    for i in range(1, max_lag + 1):
        T_var = len(x[:-i])
        # if T_var == 0 or variance == 0.:
        #     print("NaN.")
        result[i] = np.dot((x[:-i] - mean), (x[i:] - mean)) / (variance * T_var + 1e-5)
    return result


# function to compute spearman autocorrelation
def spearman_autocorrelation(ts, lag):
    result = np.zeros(lag + 1)
    result[0] = 1
    for i in range(1, lag):
        result[i] = scipy.stats.spearmanr(ts[:-i], ts[i:]).correlation

    return result


def autocorrelation(spikes, lags=200):
    if isinstance(spikes, tc.Tensor):
        spikes = spikes.detach().cpu().numpy()

    lags = min(lags, spikes.shape[-2] - 10)
    dims = spikes.shape[-1]

    if spikes.ndim == 3:  # TODO: parallelize
        n_batches = spikes.shape[0]
        ac = np.empty((n_batches, dims, lags + 1))
        for dim_idx in range(dims):
            for batch_idx in range(n_batches):
                ac[batch_idx, dim_idx, :] = get_acf(spikes[batch_idx, :, dim_idx], lags)
    else:
        ac = np.empty((dims, lags + 1))
        for dim_idx in range(dims):
            ac[dim_idx, :] = get_acf(spikes[:, dim_idx], lags)

    return tc.from_numpy(ac)


@tc.no_grad()
def autocorrelation_error(generated, ground_truth, lags=200):
    if isinstance(generated, tc.Tensor):
        assert isinstance(ground_truth, tc.Tensor)
        generated = generated.detach().cpu().numpy()
        ground_truth = ground_truth.detach().cpu().numpy()

    lags = min(lags, generated.shape[0] - 10)
    dims = generated.shape[-1]
    error = np.zeros(dims)
    for i in range(dims):
        ac_generated = get_acf(generated[:, i], lags)
        ac_gt = get_acf(ground_truth[:, i], lags)

        error[i] = np.mean((ac_generated - ac_gt) ** 2)

    average_error = np.mean(error)

    return average_error


@tc.no_grad()
def autocorrelation_nmse(
    generated: tc.Tensor | np.ndarray,
    ground_truth: tc.Tensor | np.ndarray,
    lags: int = 200,
    average_batch: bool = False,
):  # TODO: improve for multiple samples, make fast, especially for tensors
    if isinstance(generated, tc.Tensor):
        assert isinstance(ground_truth, tc.Tensor)
        generated = generated.detach().cpu().numpy()
        ground_truth = ground_truth.detach().cpu().numpy()

    lags = min(lags, generated.shape[-2] - 10)
    dims = generated.shape[-1]

    if generated.ndim == 3:  # TODO: parallelize
        n_batches = generated.shape[0]
        ac_generated = np.empty((n_batches, dims, lags))
        ac_gt = np.empty((1, dims, lags))
        for dim_idx in range(dims):
            ac_gt[0, dim_idx, :] = get_acf(ground_truth[0, :, dim_idx], lags)[1:]
            for batch_idx in range(n_batches):
                ac_generated[batch_idx, dim_idx, :] = get_acf(
                    generated[batch_idx, :, dim_idx], lags
                )[1:]

        if not average_batch:
            ace_nmse = np.mean((ac_generated - ac_gt) ** 2, axis=(-1, -2)) / (
                np.var(ac_gt, axis=(-1, -2)) + 1e-10
            )
            return tc.from_numpy(ace_nmse)
        else:
            ace_nmse = np.mean((ac_generated - ac_gt) ** 2) / (np.var(ac_gt) + 1e-10)
            return tc.tensor(ace_nmse)

    else:
        ac_generated = np.empty((dims, lags))
        ac_gt = np.empty((dims, lags))
        for dim_idx in range(dims):
            ac_generated[dim_idx, :] = get_acf(generated[:, dim_idx], lags)[1:]
            ac_gt[dim_idx, :] = get_acf(ground_truth[:, dim_idx], lags)[1:]

        ace_nmse = np.mean((ac_generated - ac_gt) ** 2) / (np.var(ac_gt) + 1e-10)
        return tc.tensor(ace_nmse)


def spearman_autocorrelation_error(generated, ground_truth, lags):
    dims = generated.shape[-1]
    error = np.zeros(dims)
    for i in range(dims):
        ac_generated = np.nan_to_num(spearman_autocorrelation(generated[:, i], lags))
        ac_gt = spearman_autocorrelation(ground_truth[:, i], lags)

        error[i] = np.mean((ac_generated - ac_gt) ** 2)

    return np.mean(error), error
