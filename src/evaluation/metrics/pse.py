from typing import Literal, Optional

import torch as tc
import torch.nn.functional as F

from src.utils import reduce_tensor, standardize_tensor


def gaussian_filter1d(
    input: tc.Tensor,
    sigma: float,
    truncate: float = 4.0,
    pad_mode: Optional[Literal["zero", "reflect"]] = None,
) -> tc.Tensor:
    """
    Apply a 1D Gaussian filter to batched trajectories.

    Args:
        input (torch.Tensor): Input 3D tensor with shape (T, B, N).
        sigma (float): Standard deviation of the Gaussian kernel.
        truncate (float): Truncate the filter at this many standard deviations.
        pad_mode (str): Padding mode - "zero" for zero padding, "reflect" for reflection padding, None for no padding.

    Returns:
        torch.Tensor: Filtered tensor with same shape as input.
    """
    if sigma <= 0:
        return input

    T, B, N = input.shape
    radius = int(truncate * sigma + 0.5)

    if pad_mode == "reflect" and radius > 0 and T < 2:
        return input  # cannot reflect a length-1 signal

    x = tc.arange(-radius, radius + 1, dtype=input.dtype, device=input.device)
    kernel = tc.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    # Reshape to (B*N, 1, T) for batch processing
    input_reshaped = input.permute(1, 2, 0).contiguous().view(B * N, 1, T)

    if pad_mode == "reflect" and radius > 0:
        input_reshaped = F.pad(input_reshaped, (radius, radius), mode="reflect")
        padding = 0
    elif pad_mode == "zero":
        padding = radius if radius > 0 else 0
    elif pad_mode is None:
        padding = 0
    else:
        raise ValueError(f"Invalid pad_mode: {pad_mode}. Use 'zero', 'reflect', or None.")

    # Reshape kernel for 1D convolution: [out_channels, in_channels, kernel_size]
    kernel_reshaped = kernel.view(1, 1, -1)

    # Apply convolution to all batches at once
    output = F.conv1d(input_reshaped, kernel_reshaped, padding=padding)

    # Reshape back to (T, B, N)
    return output.view(B, N, -1).permute(2, 0, 1).contiguous()


def hellinger_distance(p: tc.Tensor, q: tc.Tensor):
    return tc.sqrt(1 - tc.sqrt(p * q).sum(dim=0))


def dimension_wise_power_spectrum(
    X: tc.Tensor,
    smoothing_sigma: float = 0.0,
    freq_cutoff: int = None,
    smoothing_pad_mode: Optional[Literal["zero", "reflect"]] = None,
):
    added_batch_dim = False
    if X.dim() == 2:
        X = X.unsqueeze(1)
        added_batch_dim = True

    ps = tc.fft.rfft(X, dim=0).abs() ** 2
    if smoothing_sigma > 0.0:
        ps = gaussian_filter1d(ps, smoothing_sigma, pad_mode=smoothing_pad_mode)
    if freq_cutoff is not None:
        ps = ps[:freq_cutoff]

    if added_batch_dim:
        ps.squeeze_(1)

    return ps


def power_spectrum_error(
    X_gen: tc.Tensor,
    X_true: tc.Tensor,
    smoothing_sigma=0.0,
    smoothing_pad_mode: Optional[Literal["zero", "reflect"]] = None,
    freq_cutoff: Optional[int] = None,
    standardize: bool = True,
    batch_reduction: Optional[str] = None,
) -> tc.Tensor:
    """
    Compute the power spectrum error defined as the Hellinger distance between power spectra
    of generated and true trajectories.

    This metric quantifies the temporal similarity between two time series by comparing their
    frequency domain representations. The Hellinger distance is computed on
    normalized power spectra, leading to bounded values between 0 (perfect match) and 1 (maximum discrepancy).

    Args:
        X_gen (`torch.Tensor`): Generated trajectories with shape `(T, (B), N)` where:
            - `T`: time steps
            - `B`: batch dimension (optional)
            - `N`: number of features/dimensions
        X_true (`torch.Tensor`): True trajectories with shape `(T, (B), N)`, same format as `X_gen`
        smoothing_sigma (`float`, optional): Standard deviation for Gaussian smoothing of power spectra.
            Defaults to `0.0` (no smoothing).
        smoothing_pad_mode (`str`, optional): Padding mode for Gaussian smoothing.
            Options: `"zero"`, `"reflect"`, or `None`. Defaults to `None`.
        freq_cutoff (`int`, optional): Frequency cutoff for the power spectrum.
            If `None`, uses all frequencies. Defaults to `None`.
        standardize (`bool`, optional): Whether to standardize trajectories before computing spectra.
            Defaults to `True`.
        batch_reduction (`str`, optional): How to reduce the batch dimension in the output.
            Options: See `reduce_tensor` for options. Defaults to `"mean"`.

    Returns:
        `torch.Tensor`: Hellinger distance between normalized power spectra of `X_gen` and `X_true`.

    Raises:
        `ValueError`: If `X_gen` and `X_true` have different shapes or dimensions
    """
    Xg, Xt = X_gen, X_true
    sz_t, sz_g = Xt.size(), Xg.size()
    assert sz_t == sz_g, (
        f"X_gen and X_true must have the same shape, got {sz_g} (gen) and {sz_t} (true)"
    )

    added_batch_dim = False
    if len(sz_t) == 2:
        Xg = Xg.unsqueeze(1)
        Xt = Xt.unsqueeze(1)
        added_batch_dim = True

    if standardize:
        Xg = standardize_tensor(Xg, dim=0, keepdim=True)
        Xt = standardize_tensor(Xt, dim=0, keepdim=True)

    # we can compute spectra in one go in this way
    spectra = dimension_wise_power_spectrum(
        tc.cat([Xg, Xt], dim=-1), smoothing_sigma, freq_cutoff, smoothing_pad_mode
    )

    # normalize by the total power
    spectra /= spectra.sum(dim=0, keepdim=True) + tc.finfo(spectra.dtype).eps

    # split
    p_g, p_t = spectra.tensor_split(2, dim=-1)

    # compute the Hellinger distance over dynamical variables
    D_h = hellinger_distance(p_g, p_t).mean(dim=-1)

    if added_batch_dim:
        D_h.squeeze_(0)
    else:
        D_h = reduce_tensor(D_h, reduction=batch_reduction)

    return D_h
