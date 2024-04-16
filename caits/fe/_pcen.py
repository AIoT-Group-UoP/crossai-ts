# The code is based on the following snippet, however, with some modifications:
# https://github.com/4p0pt0Z/Audio_blind_source_separation/blob/master/pcen.py
# The implementation follows the paper by Yuxuan Wang et al.:
# Wang, Yuxuan, Pascal Getreuer, Thad Hughes, Richard F. Lyon, and Rif A.
# Saurous. "Trainable frontend for robust and far-field keyword spotting." In
# 2017 IEEE International Conference on Acoustics, Speech and Signal Processing
# (ICASSP), pp. 5670-5674. IEEE, 2017.
# https://arxiv.org/abs/1607.05666
# A part of implementation is also based on the librosa's functionality,
# however with minor modifications:
# https://librosa.github.io/librosa/generated/librosa.core.pcen.html

from typing import Optional
import numpy as np
import scipy


def pcen(
        S: np.ndarray,
        sr: int = 22050,
        hop_length: int = 512,
        gain: float = 0.98,
        bias: float = 2,
        power: float = 0.5,
        time_constant: float = 0.400,
        eps: float = 1e-6,
        b: Optional[float] = None,
        max_size: int = 1,
        ref: Optional[np.ndarray] = None,
        axis: int = -1,
        max_axis: Optional[int] = None,
        magnitude_increase: bool = False
) -> np.ndarray:
    """Generates artifacts in the output due to the zero-initialization of the
    low-pass filtered version of the input spectrogram used for normalization.
    The implementation is similar to librosa, however, unlike Librosa's
    implementation, the choice of applying a filter forward and backward or
    just forward is given. The default is to apply the filter forward only.
    For frontward-backward filter, the filtfilt function of scipy.signal cas
    been used.

    Args:
        S: Input spectrogram in np.ndarray.
        sr: Sampling rate of the input signal. Defaults to 22050.
        hop_length: The number of samples between successive frames. Defaults
            to 512.
        gain: The time constant of the IIR filter. Defaults to 0.98.
        bias: The bias term to be added to the output. Defaults to 2.
        power: The power to which the PCEN output is raised. Defaults to 0.5.
        time_constant: The time constant of the IIR filter. Defaults to 0.400.
        eps: The epsilon term to be added to the output. Defaults to 1e-6.
        b: The parameter of the IIR filter. Defaults to None.
        max_size: The size of the max-filter to be applied. Defaults to 1.
        ref: The reference value to be used for max-filtering. Defaults to
            None.
        axis: The axis along which the filter is to be applied. Defaults to -1.
        max_axis: The axis along which the max-filter is to be applied.
            Defaults to None.
        magnitude_increase: Whether to apply the filter forward and backward.
            Defaults to False.

    Returns:
        np.ndarray: The PCEN output as a 2D array.
    """

    if power <= 0:
        raise ValueError('power={} must be strictly positive'.format(power))

    if gain < 0:
        raise ValueError('gain={} must be non-negative'.format(gain))

    if bias < 0:
        raise ValueError('bias={} must be non-negative'.format(bias))

    if eps <= 0:
        raise ValueError('eps={} must be strictly positive'.format(eps))

    if time_constant <= 0:
        raise ValueError('time_constant={} must be strictly positive'.format(
            time_constant))

    if max_size < 1 or not isinstance(max_size, int):
        raise ValueError('max_size={} must be a positive integer'.format(
            max_size))

    if b is None:
        t_frames = time_constant * sr / float(hop_length)
        # By default, this solves the equation for b:
        #   b**2  + (1 - b) / t_frames  - 2 = 0
        # which approximates the full-width half-max of the
        # squared frequency response of the IIR low-pass filter

        b = (np.sqrt(1 + 4 * t_frames ** 2) - 1) / (2 * t_frames ** 2)

    if not 0 <= b <= 1:
        raise ValueError('b={} must be between 0 and 1'.format(b))

    if np.issubdtype(S.dtype, np.complexfloating):
        print('pcen was called on complex input so phase '
              'information will be discarded. To suppress this warning, '
              'call pcen(np.abs(D)) instead.')
        S = np.abs(S)

    if ref is None:
        if max_size == 1:
            ref = S
        elif S.ndim == 1:
            raise ValueError(
                'Max-filtering cannot be applied to 1-dimensional input'
            )
        else:
            if max_axis is None:
                if S.ndim != 2:
                    raise ValueError(
                        'Max-filtering a {:d}-dimensional spectrogram '
                        'requires you to specify max_axis'.format(S.ndim))
                # if axis = 0, max_axis=1
                # if axis = +- 1, max_axis = 0
                max_axis = np.mod(1 - axis, 2)

            ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=max_axis)

    if magnitude_increase:
        # forward-backward provides better results.
        S_smooth = scipy.signal.filtfilt(
            [b],
            [1, b - 1],
            ref,
            axis=axis,
            padtype=None)
    else:
        # Use of a forward only pass of the filter - but initialize it more
        # carefully.
        S_smooth, _ = scipy.signal.lfilter(
            [b], [1, b - 1], ref,
            axis=axis,
            zi=[scipy.signal.lfilter_zi(
                [b],
                [1, b - 1]
            )] * S[:, 0].shape[0]
        )

    # Working in log-space gives us some stability, and a slight speedup
    smooth = np.exp(-gain * (np.log(eps) + np.log1p(S_smooth / eps)))
    return (S * smooth + bias) ** power - bias ** power


def pcen_base(
        E: np.ndarray,
        alpha: float = 0.98,
        delta: float = 2,
        r: float = 0.5,
        s: float = 0.025,
        eps: float = 1e-6,
        magnitude_increase: bool = False
) -> np.ndarray:
    """Implements the PCEN transform to apply on a batch of np.ndarray
    instances (spectrograms).

    Implementation of the PCEN transform to operate on pytorch tensors.
    Implementation follows the description in Yuxian Wang et al. "Trainable
    Frontend For Robust and Far-Field Keyword Spotting" (2017).

        PCEN(t,f) = ( E(t, f) / (eps + M(t,f)**alpha + delta )**r - delta**r
        M(t,f) = (1-s) * M(t-1,f) + s * E(t,f)

    Args:
        E: Batch of (mel-) spectrograms in the shape of
            [instances, frequency, time].
        alpha: The 'alpha' parameter of the PCEN transform (power of
            denominator during normalization).
        delta: The 'delta' parameter of the PCEN transform (shift before
            dynamic range compression (DRC)).
        r: The 'r' parameter of the PCEN transform (parameter of the power
            function used for DRC).
        s: The 's' parameter of the PCEN transform (parameter of the filter M
            used for normalization).
        eps: The 'epsilon' parameter of the PCEN transform. (numerical
            stability of the normalization).
        magnitude_increase: Whether to make the normalization in exponential
            representation for numerical stability or not. Defaults to False.
            If True, the normalization is done in exponential representation,
            referred as "stable reformulation", due to Vincent Lostanlen (found
            at https://gist.github.com/f0k/c837bcf0bfde189ca16eab63637839cb.

    Returns:
        np.ndarray: The PCEN transform of the input bach of spectrograms
            in the shape of [instances, frequency, time])
    """

    # Compute low-pass filtered version of the spectrograms.
    M = _first_order_iir(E, s)

    if magnitude_increase:
        # Make the normalization in exponential representation for numerical
        # stability
        M = np.exp(-alpha * (np.log(eps) + np.log1p(M / eps)))
        return np.power(E * M + delta, r) - np.power(delta, r)
    else:
        # Naive implementation
        smooth = (eps + M)**(-alpha)
        return (E * smooth + delta)**r - delta**r


def _first_order_iir(
        E: np.ndarray,
        s: float
) -> np.ndarray:
    """Implements a first order Infinite Impulse Response (IIR) forward filter
    initialized using the input values. Specifically, this function implements
    the filter M defined in:

    Wang, Yuxuan, Pascal Getreuer, Thad Hughes, Richard F. Lyon, and Rif A.
    Saurous. "Trainable frontend for robust and far-field keyword spotting."
    In 2017 IEEE International Conference on Acoustics, Speech and Signal
    Processing (ICASSP), pp. 5670-5674. IEEE, 2017.

    Args:
        E: A batch of spectrograms or mel-spectrograms in the shape of
            [instances, frequency, time].
        s: Float number containing parameter of the filter.

    Returns:
        np.ndarray: M, as the batch of low-pass filtered spectrograms or
            mel-spectrograms in the shape of [instances, frequency, time].
    """
    M = np.zeros_like(E)
    M[..., 0] = E[..., 0]  # Initializes with the value of the spectrograms.
    for frame in range(1, M.shape[-1]):
        M[..., frame] = (1 - s) * M[..., frame - 1] + s * E[..., frame]
    return M
