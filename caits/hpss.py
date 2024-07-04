import numpy as np
from typing import Tuple, Union, List, Any, Optional
from scipy.ndimage import median_filter
from caits.core._core_checks import dtype_r2c
from caits.core._core_typing import _IntLike_co, _FloatLike_co
from caits.fe import stft, istft


def hpss(
    y: np.ndarray, percussion_factor: Optional[float] = None, **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """Decomposes an audio time series into harmonic and percussive components.

    This function automates the STFT->HPSS->ISTFT pipeline, and ensures that
    the output waveforms have equal length to the input waveform `y`.

    Args:
        y (np.ndarray): Audio time series of shape (..., n). Multi-channel is supported.
        percussion_factor (Optional[float]): Factor to apply to the percussive component 
                                             for enhanced audio. If None, harmonic and 
                                             percussive components are returned. 
                                             Default is None.
        **kwargs (Any): Additional keyword arguments for `librosa.decompose.hpss`.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - np.ndarray: Audio time series of the harmonic elements, of shape (..., n).
            - np.ndarray: Audio time series of the percussive elements, of shape (..., n).
            - np.ndarray: Enhanced audio time series if `percussion_factor` is provided, of shape (..., n).
    """
    # Compute the STFT matrix
    stft_matrix = stft(y)

    # Decompose into harmonic and percussive components
    stft_harm, stft_perc = _hpss(stft_matrix, **kwargs)

    # Invert the STFTs. Adjust length to match the input.
    y_harm = istft(stft_harm, dtype=y.dtype, length=y.shape[-1])
    y_perc = istft(stft_perc, dtype=y.dtype, length=y.shape[-1])

    if percussion_factor is not None:
        enhanced_audio = y_harm + percussion_factor * y_perc
        return enhanced_audio

    return y_harm, y_perc


def _hpss(
    S: np.ndarray,
    *,
    kernel_size: Union[
        _IntLike_co, Tuple[_IntLike_co, _IntLike_co], List[_IntLike_co]
    ] = 31,
    power: float = 2.0,
    mask: bool = False,
    margin: Union[
        _FloatLike_co, Tuple[_FloatLike_co, _FloatLike_co], List[_FloatLike_co]
    ] = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Median-filtering harmonic percussive source separation (HPSS).

    If ``margin = 1.0``, decomposes an input spectrogram ``S = H + P``
    where ``H`` contains the harmonic components,
    and ``P`` contains the percussive components.

    If ``margin > 1.0``, decomposes an input spectrogram ``S = H + P + R``
    where ``R`` contains residual components not included in ``H`` or ``P``.

    This implementation is based upon the algorithm described by [#]_ and [#]_.

    .. [#] Fitzgerald, Derry.
        "Harmonic/percussive separation using median filtering."
        13th International Conference on Digital Audio Effects (DAFX10),
        Graz, Austria, 2010.

    .. [#] Driedger, Müller, Disch.
        "Extending harmonic-percussive separation of audio."
        15th International Society for Music Information Retrieval Conference (ISMIR 2014),
        Taipei, Taiwan, 2014.

    Parameters
    ----------
    S : np.ndarray [shape=(..., d, n)]
        input spectrogram. May be real (magnitude) or complex.
        Multi-channel is supported.

    kernel_size : int or tuple (kernel_harmonic, kernel_percussive)
        kernel size(s) for the median filters.

        - If scalar, the same size is used for both harmonic and percussive.
        - If tuple, the first value specifies the width of the
          harmonic filter, and the second value specifies the width
          of the percussive filter.

    power : float > 0 [scalar]
        Exponent for the Wiener filter when constructing soft mask matrices.

    mask : bool
        Return the masking matrices instead of components.

        Masking matrices contain non-negative real values that
        can be used to measure the assignment of energy from ``S``
        into harmonic or percussive components.

        Components can be recovered by multiplying ``S * mask_H``
        or ``S * mask_P``.

    margin : float or tuple (margin_harmonic, margin_percussive)
        margin size(s) for the masks (as described in [2]_)

        - If scalar, the same size is used for both harmonic and percussive.
        - If tuple, the first value specifies the margin of the
          harmonic mask, and the second value specifies the margin
          of the percussive mask.

    Returns
    -------
    harmonic : np.ndarray [shape=(..., d, n)]
        harmonic component (or mask)
    percussive : np.ndarray [shape=(..., d, n)]
        percussive component (or mask)
    """
    phase: Union[float, np.ndarray]

    if np.iscomplexobj(S):
        S, phase = magphase(S)
    else:
        phase = 1

    if isinstance(kernel_size, Tuple) or isinstance(kernel_size, List):
        win_harm = kernel_size[0]
        win_perc = kernel_size[1]
    else:
        win_harm = kernel_size
        win_perc = kernel_size

    if isinstance(margin, Tuple) or isinstance(margin, List):
        margin_harm = margin[0]
        margin_perc = margin[1]
    else:
        margin_harm = margin
        margin_perc = margin

    # margin minimum is 1.0
    if margin_harm < 1 or margin_perc < 1:
        raise ValueError(
            "Margins must be >= 1.0. " "A typical range is between 1 and 10."
        )

    # shape for kernels
    harm_shape: List[_IntLike_co] = [1] * S.ndim
    harm_shape[-1] = win_harm

    perc_shape: List[_IntLike_co] = [1] * S.ndim
    perc_shape[-2] = win_perc

    # Compute median filters. Pre-allocation here preserves memory layout.
    harm = np.empty_like(S)
    harm[:] = median_filter(S, size=harm_shape, mode="reflect")

    perc = np.empty_like(S)
    perc[:] = median_filter(S, size=perc_shape, mode="reflect")

    split_zeros = margin_harm == 1 and margin_perc == 1

    mask_harm = softmask(
        harm, perc * margin_harm, power=power, split_zeros=split_zeros
    )

    mask_perc = softmask(
        perc, harm * margin_perc, power=power, split_zeros=split_zeros
    )

    if mask:
        return mask_harm, mask_perc

    return ((S * mask_harm) * phase, (S * mask_perc) * phase)


def magphase(D: np.ndarray, *, power: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Separate a complex-valued spectrogram D into its magnitude (S)
    and phase (P) components, so that ``D = S * P``.

    Parameters
    ----------
    D : np.ndarray [shape=(..., d, t), dtype=complex]
        complex-valued spectrogram
    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.

    Returns
    -------
    D_mag : np.ndarray [shape=(..., d, t), dtype=real]
        magnitude of ``D``, raised to ``power``
    D_phase : np.ndarray [shape=(..., d, t), dtype=complex]
        ``exp(1.j * phi)`` where ``phi`` is the phase of ``D``
    """
    mag = np.abs(D)

    # Prevent NaNs and return magnitude 0, phase 1+0j for zero
    zeros_to_ones = mag == 0
    mag_nonzero = mag + zeros_to_ones
    # Compute real and imaginary separately, because complex division can
    # produce NaNs when denormalized numbers are involved (< ~2e-39 for
    # complex64, ~5e-309 for complex128)
    phase = np.empty_like(D, dtype=dtype_r2c(D.dtype))
    phase.real = D.real / mag_nonzero + zeros_to_ones
    phase.imag = D.imag / mag_nonzero

    mag **= power

    return mag, phase


def softmask(
    X: np.ndarray, X_ref: np.ndarray, *, power: float = 1, split_zeros: bool = False
) -> np.ndarray:
    """Robustly compute a soft-mask operation.

    `M = X**power / (X**power + X_ref**power)`

    Args:
        X: The (non-negative) input array corresponding to the positive mask elements.
        X_ref: The (non-negative) array of reference or background elements. 
               Must have the same shape as `X`.
        power: Exponent for the soft mask. If finite, returns the soft mask computed
               in a numerically stable way. If infinite, returns a hard (binary) mask
               equivalent to `X > X_ref`. Note: for hard masks, ties are always broken
               in favor of `X_ref` (mask=0). Default is 1.
        split_zeros: If `True`, entries where `X` and `X_ref` are both small (close to 0) 
                     will receive mask values of 0.5. Otherwise, the mask is set to 0 for
                     these entries. Default is False.

    Returns:
        np.ndarray: The output mask array, with the same shape as `X`.

    Raises:
        ValueError: If `X` and `X_ref` have different shapes.
        ValueError: If `X` or `X_ref` are negative anywhere.
        ValueError: If `power` is less than or equal to 0.
    """
    if X.shape != X_ref.shape:
        raise ValueError(f"Shape mismatch: {X.shape}!={X_ref.shape}")

    if np.any(X < 0) or np.any(X_ref < 0):
        raise ValueError("X and X_ref must be non-negative")

    if power <= 0:
        raise ValueError("power must be strictly positive")

    # We're working with ints, cast to float.
    dtype = X.dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float32

    # Re-scale the input arrays relative to the larger value
    Z = np.maximum(X, X_ref).astype(dtype)
    bad_idx = Z < np.finfo(dtype).tiny
    Z[bad_idx] = 1

    # For finite power, compute the softmask
    mask: np.ndarray

    if np.isfinite(power):
        mask = (X / Z) ** power
        ref_mask = (X_ref / Z) ** power
        good_idx = ~bad_idx
        mask[good_idx] /= mask[good_idx] + ref_mask[good_idx]
        # Wherever energy is below energy in both inputs, split the mask
        if split_zeros:
            mask[bad_idx] = 0.5
        else:
            mask[bad_idx] = 0.0
    else:
        # Otherwise, compute the hard mask
        mask = X > X_ref

    return mask

