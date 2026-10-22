import torch as tc

from src.models.dsr_model.dsr_model import DSRModel


@tc.no_grad()
def lyapunov_spectrum(
    dsr_model: DSRModel,
    cumulative_timeseries_index: int,
    z0: tc.Tensor,
    T: int,
    T_transient: int = 0,
    qr_steps: int = 5,
    return_orbits: bool = False,
):
    """
    Compute the Lyapunov spectrum of the DSR model given initial condition `z0`.

    Args:
        dsr_model: DSR model
        z0: Initial condition ((B), M)
        T: Total time steps
        T_transient: Transient time
        qr_steps: interval at which to reorthogonalize

    Returns:
        Lyapunov spectra ((B), M)
    """
    added_batch_dim = False
    if z0.dim() == 1:
        z0 = z0.unsqueeze(0)
        added_batch_dim = True

    # get shape
    B, M = z0.shape

    # evolve for transient time T_transient
    if T_transient > 0:
        z = dsr_model.generate_free_trajectory(
            z0,
            T_transient,
            external_inputs=None,
            cumulative_timeseries_index=cumulative_timeseries_index,
        )[-1, :, :]
    else:
        z = z0

    # params
    params = dsr_model.construct_params(cumulative_timeseries_index=cumulative_timeseries_index)
    hyperparameters = dsr_model.hyperparameters

    # compute Lyapunov spectrum
    lyaps = tc.zeros(B, M, dtype=z.dtype, device=z.device)
    Q = tc.eye(M, dtype=z.dtype, device=z.device).unsqueeze(0).repeat(B, 1, 1)

    # dtype-aware tiny to avoid log(0) and -inf
    tiny = tc.finfo(z.dtype).tiny

    # initialize list to store orbits
    if return_orbits:
        Z = []

    # iterate over time steps
    for t in range(T):
        # forward step
        z = dsr_model.latent_step.forward(z, None, params, hyperparameters)

        # store state
        if return_orbits:
            Z.append(z)

        # compute jacobians
        jacobians = dsr_model.latent_step.jacobian(z, params, hyperparameters)

        # batched matmul (B, M, M) @ (B, M, M) -> (B, M, M)
        Q = jacobians @ Q

        # reorthogonalize every qr_steps steps and on the final step
        if ((t + 1) % qr_steps == 0) or (t == T - 1):
            Q, R = tc.linalg.qr(Q)

            # accumulate lyapunov exponents with epsilon clamping for stability
            diag_R_abs = tc.abs(tc.diagonal(R, dim1=-2, dim2=-1))
            lyaps += tc.log(tc.clamp_min(diag_R_abs, tiny))

    # remove batch dimension
    if added_batch_dim:
        lyaps.squeeze_(0)

    if return_orbits:
        return lyaps / T, tc.stack(Z)
    else:
        return lyaps / T
