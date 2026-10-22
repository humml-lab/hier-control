import torch as tc
from scipy.signal import correlate

from .spike_stats import nmse


def _cross_correlation(spike_train_1, spike_train_2, max_lag):
    """
    Calculate the normalized cross-correlation between two spike trains for specified lags.

    spike_train_1, spike_train_2: The two spike trains to be cross-correlated.
    max_lag: Maximum time lag to compute the cross-correlation.
    """
    # Compute full cross-correlation
    cross_corr = correlate(
        spike_train_1 - tc.mean(spike_train_1),
        spike_train_2 - tc.mean(spike_train_2),
        mode="full",
    )

    # Normalize by the product of the standard deviations
    norm_factor = tc.std(spike_train_1) * tc.std(spike_train_2) * len(spike_train_1) + 1e-10
    cross_corr = cross_corr / norm_factor

    # Select the range of lags of interest
    mid_point = len(cross_corr) // 2
    cross_corr = cross_corr[mid_point - max_lag : mid_point + max_lag + 1]

    return cross_corr


def _compute_cross_correlogram_matrix(data, max_lag):
    """Calculate pairwise cross-correlations for all neuron pairs"""
    num_neurons = data.shape[1]
    correlogram_matrix = tc.zeros((num_neurons, num_neurons, 2 * max_lag + 1))

    for i in range(num_neurons):
        for j in range(i, num_neurons):
            correlogram_matrix[i, j] = _cross_correlation(data[:, i], data[:, j], max_lag)
            correlogram_matrix[j, i] = correlogram_matrix[i, j]

    return correlogram_matrix


def _compute_inter_cross_correlogram_matrix(data_a, data_b, max_lag):
    num_neurons_a, num_neurons_b = data_a.shape[1], data_b.shape[1]
    correlogram_matrix = tc.zeros((num_neurons_a, num_neurons_b, 2 * max_lag + 1))

    for i in range(num_neurons_a):
        for j in range(i, num_neurons_b):
            correlogram_matrix[i, j] = _cross_correlation(data_a[:, i], data_b[:, j], max_lag)

    return correlogram_matrix


def _get_upper_triangle(matrix):
    m = matrix.shape[0]
    r, c = tc.triu_indices(m, m, offset=1)  # offset = 1 to exclude the diagonal
    return matrix[r, c].flatten()


def cross_corr(
    X,
    T_start=0,
    T_maxcorr=3000,
    max_lag=0,
    return_correlogram_matrix=False,
):
    # TODO: maybe instead of correlation, use cross-STTC (see elephant lib)
    # TODO: use pytroch coerrcoeff matrix?

    correlogram_matrix = _compute_cross_correlogram_matrix(
        X[T_start : T_start + T_maxcorr], max_lag
    )[:, :, max_lag]
    flatted_upper_triangle_correlogram = _get_upper_triangle(correlogram_matrix)

    if return_correlogram_matrix:
        return flatted_upper_triangle_correlogram, correlogram_matrix
    else:
        return flatted_upper_triangle_correlogram


def cross_corr_error(
    x_gen,
    x_true,
    T_start=0,
    T_maxcorr=3000,
    max_lag=0,
    average_over_units=True,
):
    flatted_upper_triangle_correlogram_gen = cross_corr(x_gen, T_start, T_maxcorr, max_lag)
    flatted_upper_triangle_correlogram_true = cross_corr(x_true, T_start, T_maxcorr, max_lag)

    flatted_upper_triangle_correlogram_err = tc.abs(
        flatted_upper_triangle_correlogram_gen - flatted_upper_triangle_correlogram_true
    )

    if average_over_units:
        return tc.mean(flatted_upper_triangle_correlogram_err), tc.std(
            flatted_upper_triangle_correlogram_err
        )
    else:
        return flatted_upper_triangle_correlogram_err


def cross_corr_nmse(
    X_gen,
    X_true,
    T_start=0,
    T_maxcorr=3000,
    max_lag=0,
    average_batch=False,
):
    if X_gen.dim() == 2:
        flatted_upper_triangle_correlograms_gen = cross_corr(X_gen, T_start, T_maxcorr, max_lag)
        flatted_upper_triangle_correlograms_true = cross_corr(X_true, T_start, T_maxcorr, max_lag)
    else:  # TODO: parallelize this, plus make sure clear that one here compares first batch of X_true with all other batches in X_gen
        flatted_upper_triangle_correlograms_gen = []
        flatted_upper_triangle_correlograms_true = []

        flatted_upper_triangle_correlogram_true = cross_corr(
            X_true[0], T_start, T_maxcorr, max_lag
        )

        for batch_idx in range(X_gen.size(0)):
            flatted_upper_triangle_correlogram_gen = cross_corr(
                X_gen[batch_idx], T_start, T_maxcorr, max_lag
            )
            flatted_upper_triangle_correlograms_gen.append(flatted_upper_triangle_correlogram_gen)

        flatted_upper_triangle_correlograms_gen = tc.stack(flatted_upper_triangle_correlograms_gen)
        flatted_upper_triangle_correlograms_true = (
            flatted_upper_triangle_correlogram_true.unsqueeze(0)
        )

    return nmse(
        flatted_upper_triangle_correlograms_gen,
        flatted_upper_triangle_correlograms_true,
        average_batch=average_batch,
    )


def inter_cross_corr_nmse(
    X_gen, X_true, T_start=0, T_maxcorr=3000, max_lag=200, average_batch=False
):
    X_gen_a, X_gen_b = X_gen
    X_true_a, X_true_b = X_true

    if X_gen_a.dim() == 3:
        assert X_gen_a.dim() == X_gen_b.dim() == X_true_a.dim() == X_true_b.dim() == 3

        # TODO: parallelize this, plus make sure clear that one here compares first batch of X_true with all other batches in X_gen
        flatted_upper_triangle_correlograms_gen = []
        flatted_upper_triangle_correlograms_true = []

        for batch_idx in range(X_gen_a.size(0)):
            inter_correlogram_matrix_gen = _compute_inter_cross_correlogram_matrix(
                X_gen_a[batch_idx, T_start : T_start + T_maxcorr],
                X_gen_b[batch_idx, T_start : T_start + T_maxcorr],
                max_lag,
            )
            inter_correlogram_matrix_true = _compute_inter_cross_correlogram_matrix(
                X_true_a[batch_idx, T_start : T_start + T_maxcorr],
                X_true_b[batch_idx, T_start : T_start + T_maxcorr],
                max_lag,
            )

            flatted_upper_triangle_correlogram_gen = inter_correlogram_matrix_gen.flatten()
            flatted_upper_triangle_correlogram_true = inter_correlogram_matrix_true.flatten()

            flatted_upper_triangle_correlograms_true.append(
                flatted_upper_triangle_correlogram_true
            )
            flatted_upper_triangle_correlograms_gen.append(flatted_upper_triangle_correlogram_gen)

        flatted_upper_triangle_correlograms_gen = tc.stack(flatted_upper_triangle_correlograms_gen)
        flatted_upper_triangle_correlograms_true = tc.stack(
            flatted_upper_triangle_correlograms_true
        )

    else:
        inter_correlogram_matrix_gen = _compute_inter_cross_correlogram_matrix(
            X_gen_a[T_start : T_start + T_maxcorr],
            X_gen_b[T_start : T_start + T_maxcorr],
            max_lag,
        )
        inter_correlogram_matrix_true = _compute_inter_cross_correlogram_matrix(
            X_true_a[T_start : T_start + T_maxcorr],
            X_true_b[T_start : T_start + T_maxcorr],
            max_lag,
        )

        flatted_upper_triangle_correlograms_gen = inter_correlogram_matrix_gen.flatten()
        flatted_upper_triangle_correlograms_true = inter_correlogram_matrix_true.flatten()

    return nmse(
        flatted_upper_triangle_correlograms_gen,
        flatted_upper_triangle_correlograms_true,
        average_batch=average_batch,
    )
