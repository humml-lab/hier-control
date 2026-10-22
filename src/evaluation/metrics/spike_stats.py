import torch as tc


def nmse(X_gen, X_true, average_batch=False):
    if average_batch and X_gen.dim() > 1:
        return tc.mean((X_gen - X_true) ** 2) / (tc.var(X_true) + 1e-10)
    else:
        return tc.mean((X_gen - X_true) ** 2, dim=-1) / (tc.var(X_true, dim=-1) + 1e-10)


def mean_spike_rate(X):
    return tc.mean(X, dim=-2)


def zero_count_ratio(X):
    return tc.sum(X == 0, dim=-2) / X.shape[-2]


def coefficient_of_variation(X):
    # avoid NaNs via +1e-10 if x_gen == 0. (along one 0. dim)
    return tc.std(X, dim=-2) / (tc.mean(X, dim=-2) + 1e-10)


def mean_spike_rate_error(x_gen, x_true, average_over_units=True):
    spike_rate_error_per_unit = tc.abs(mean_spike_rate(x_gen) - mean_spike_rate(x_true))

    if average_over_units:
        return tc.mean(spike_rate_error_per_unit), tc.std(spike_rate_error_per_unit)
    else:
        return spike_rate_error_per_unit


def mean_spike_rate_nmse(X_gen, X_true, average_batch=False):
    """normalized mean square error over units"""
    msr_true = mean_spike_rate(X_true)
    msr_gen = mean_spike_rate(X_gen)
    return nmse(msr_gen, msr_true, average_batch=average_batch)


def zero_count_ratio_error(x_gen, x_true, average_over_units=True):
    spike_rate_per_unit_gen = zero_count_ratio(x_gen)
    spike_rate_per_unit_true = zero_count_ratio(x_true)
    zero_count_ratio_error_per_unit = tc.abs(spike_rate_per_unit_gen - spike_rate_per_unit_true)

    if average_over_units:
        return tc.mean(zero_count_ratio_error_per_unit), tc.std(zero_count_ratio_error_per_unit)
    else:
        return zero_count_ratio_error_per_unit


def zero_count_ratio_nmse(x_gen, x_true, average_batch=False):
    """normalized mean square error over units"""
    zero_count_rate_per_unit_gen = zero_count_ratio(x_gen)
    zero_count_rate_per_unit_true = zero_count_ratio(x_true)
    return nmse(
        zero_count_rate_per_unit_gen,
        zero_count_rate_per_unit_true,
        average_batch=average_batch,
    )


def coefficient_of_variation_error(x_gen, x_true, average_over_units=True):
    coeff_per_unit_gen = coefficient_of_variation(x_gen)
    coeff_per_unit_true = coefficient_of_variation(x_true)
    coeff_error_per_unit = tc.abs(coeff_per_unit_gen - coeff_per_unit_true)

    if average_over_units:
        return tc.mean(coeff_error_per_unit), tc.std(coeff_error_per_unit)
    else:
        return coeff_error_per_unit


def coefficient_of_variation_nmse(x_gen, x_true, average_batch=False):
    coeff_per_unit_gen = coefficient_of_variation(x_gen)
    coeff_per_unit_true = coefficient_of_variation(x_true)
    return nmse(coeff_per_unit_gen, coeff_per_unit_true, average_batch=average_batch)
