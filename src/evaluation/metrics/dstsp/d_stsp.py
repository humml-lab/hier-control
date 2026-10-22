from .klx_bin import klx_bin
from .klx_gmm import state_space_divergence_gmm


def klx(
    data_gen,
    data_true,
    show_plot=False,
    pca=False,
    n_bins=30,
    verbose=True,
    average_batch=False,
):
    if data_gen.dim() == 3:
        # not implemented parallelized batch evaluation
        # rather draw expected trajectory from decoders for efficiency reasons
        data_gen = data_gen[0]
        data_true = data_true[0]

    T, dx = data_gen.size()
    # print("T, dx", T, dx)
    if verbose:
        print("Time steps KLx:", T)
    elif dx < 5:
        if verbose:
            print("Computing KLx-BIN")
        klx_value = klx_bin(data_gen, data_true, n_bins=n_bins, show_plot=show_plot).cpu()
    else:
        if verbose:
            print("Computing KLx-GMM")
        klx_value = state_space_divergence_gmm(data_gen, data_true, cov_scaling=1.0)

    return klx_value
