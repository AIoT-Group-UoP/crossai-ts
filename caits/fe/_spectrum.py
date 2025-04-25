# The functionalities in this implementation are basically derived from
# librosa v0.10.1:
# https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py
import warnings
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import scipy
from numpy import fft

from caits.core.numpy_typing import ArrayLike, DTypeLike

from ..core._core_checks import dtype_c2r, dtype_r2c, is_positive_int, valid_audio
from ..core._core_fix import fix_length
from ..core._core_typing import _ComplexLike_co, _PadModeSTFT, _ScalarOrSequence, _WindowSpec
from ..core._core_window import frame, get_window, pad_center, tiny, window_sumsquare
from .core_spectrum import __overlap_add, expand_to
from .core_spectrum._utils import mel_filter

# Constrain STFT block sizes to 256 KB
MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10


def melspectrogram(
        y: Optional[np.ndarray] = None,
        sr: int = 22050,
        S: Optional[np.ndarray] = None,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = "hann",
        center: bool = True,
        pad_mode: _PadModeSTFT = "constant",
        power: float = 2.0,
        **kwargs: Any,
) -> np.ndarray:
    """Computes a mel-scaled spectrogram.

    If a spectrogram input ``S`` is provided, then it is mapped directly onto
    the mel basis by ``mel_f.dot(S)``.

    If a time-series input ``y, sr`` is provided, then its magnitude
    spectrogram ``S`` is first computed, and then mapped onto the mel scale by
    ``mel_f.dot(S**power)``.

    By default, ``power=2`` operates on a power spectrum.

    Args:
        y: Audio time-series as a numpy array. Multi-channel is supported.
        sr: The sampling rate of ``y`` as an integer.
        S: Spectrogram input, optional. If provided, ``y`` and ``sr`` are
            ignored.
        n_fft: The length of the FFT window as an integer.
        hop_length: The number of samples as integer between successive frames.
        win_length: Each frame of audio is windowed by `window()`. The window
            must be an integer <= n_fft.  The window will be of length
            `win_length` and then padded with zeros to match ``n_fft``.
            If unspecified, defaults to ``win_length = n_fft``.
        window: string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
            - a window specification (string, tuple, or number);
              see `scipy.signal.get_window`
            - a window function, such as `scipy.signal.windows.hann`
            - a vector or array of length ``n_fft``
        center: Boolean value that indicates:
            - If `True`, the signal ``y`` is padded so that frame
              ``t`` is centered at ``y[t * hop_length]``.
            - If `False`, then frame ``t`` begins at ``y[t * hop_length]``
        pad_mode: If ``center=True``, the padding mode to use at the edges of
            the signal. By default, STFT uses zero padding.
        power: A float that indicates the exponent for the magnitude
            spectrogram, e.g., 1 for energy, 2 for power, etc.
        **kwargs: Additional keyword arguments for Mel filter bank parameters
            n_mels: Number of Mel bands to generate as an integer > 0.
            fmin: The lowest frequency (in Hz), as a float >= 0

            fmax: The highest frequency (in Hz) as a float >= 0. If `None`,
                use ``fmax = sr / 2.0``
            htk: A boolean that shows the use HTK formula instead of Slaney.
            norm: Can be None, 'slaney', or number
                If 'slaney', divide the triangular mel weights by the width of
                the mel band (area normalization).
                If numeric, use `librosa.util.normalize` to normalize each
                filter by to unit l_p norm. See `librosa.util.normalize` for a
                full description of supported norm values (including `+-np.inf`).
                Otherwise, leave all the triangles aiming for a peak value of
                1.0.
            dtype : np.dtype
                The data type of the output basis.
                By default, uses 32-bit (single-precision) floating point.

    Returns:
        The Mel spectrogram as a np.ndarray in [shape=(..., n_mels, t)]

    """
    S, n_fft = spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Build a Mel filter
    mel_basis = mel_filter(sr=sr, n_fft=n_fft, **kwargs)

    melspec: np.ndarray = np.einsum("...ft,mf->...mt", S, mel_basis,
                                    optimize=True)
    return melspec


def power_to_db(
        S: _ScalarOrSequence[_ComplexLike_co],
        *,
        ref: Union[float, Callable] = 1.0,
        amin: float = 1e-10,
        top_db: Optional[float] = 80.0,
) -> np.ndarray:
    """ Converts a power spectrogram to decibel (dB) units.

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.

    This function caches at level 30.

    Note:
        The functionality in this implementation is basically derived from
        librosa v0.10.1:
        https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py

    Args:
        S: Input power which can be a numpy array.

        ref: Can be a float (scalar) or callable

            If scalar, the amplitude ``abs(S)`` is scaled relative to
            ``ref``::

                10 * log10(S / ref)

            Zeros in the output correspond to positions where ``S == ref``.

            If callable, the reference value is computed as ``ref(S)``.

        amin: A float (scalar) > 0 which indicates the minimum threshold for
        ``abs(S)`` and ``ref``

        top_db: A float >= 0. Threshold the output at ``top_db`` below the
            peak:

                ``max(10 * log10(S/ref)) - top_db``

    Returns:
        The dB-scaled spectrogram as a numpy array.

        ``S_db ~= 10 * log10(S) - 10 * log10(ref)``

    """

    S = np.asarray(S)

    if amin <= 0:
        raise ValueError("amin must be strictly positive")

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            "power_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call power_to_db(np.abs(D)**2) instead.",
            stacklevel=2,
        )
        magnitude = np.abs(S)
    else:
        magnitude = S

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec: np.ndarray = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ValueError("top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def db_to_power(
        S_db: np.ndarray,
        *,
        ref: float = 1.0
) -> np.ndarray:
    """Converts a dB-scaled spectrogram to a power spectrogram.

    This effectively inverts ``power_to_db``::

        db_to_power(S_db) ~= ref * 10.0**(S_db / 10)

    This function caches at level 30.

    Note:
        The functionality in this implementation is basically derived from
        librosa v0.10.1:
        https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py

    Args:
        S_db: Spectrogram input as a numpy array in dB-scaled values.
        ref: Reference power as a float > 0. The output will be scaled by this
            value.

    Returns:
        The power spectrogram as a numpy array.
    """
    return ref * np.power(10.0, 0.1 * S_db)


def amplitude_to_db(
    S: np.ndarray,
    *,
    ref: Union[float, Callable] = 1.0,
    amin: float = 1e-5,
    top_db: Optional[float] = 80.0
) -> np.ndarray:
    """Converts an amplitude spectrogram to dB-scaled spectrogram.

    This is equivalent to:
        ``power_to_db(S**2, ref=ref**2, amin=amin**2,top_db=top_db)``

    but is provided for convenience.

    This function caches at level 30.

    Note:
        The functionality in this implementation is basically derived from
        librosa v0.10.1:
        https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py

    Args:
        S: Spectrogram input amplitude. Can be a numpy array.

        ref: It can be a float (scalar) or callable.

            If scalar, the amplitude ``abs(S)`` is scaled relative to ``ref``:

                ``20 * log10(S / ref)``.

            Zeros in the output correspond to positions where ``S == ref``.

            If callable, the reference value is computed as ``ref(S)``.

        amin: A float (scalar) > 0. Indicates the minimum threshold for ``S``
            and ``ref``.

        top_db: A float (scalar) >= 0. It thresholds the output at ``top_db``
            below the peak:

                ``max(20 * log10(S/ref)) - top_db``

    Returns:
        The dB-scaled spectrogram as a numpy array. ``S`` measured in dB.
    """
    S = np.asarray(S)

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            "amplitude_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call amplitude_to_db(np.abs(S)) instead.",
            stacklevel=2,
        )

    magnitude = np.abs(S)

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    power = np.square(magnitude, out=magnitude)

    return power_to_db(power, ref=ref_value ** 2, amin=amin ** 2,
                       top_db=top_db)


def db_to_amplitude(
    S_db: np.ndarray,
    *,
    ref: float = 1.0
) -> np.ndarray:
    """Converts a dB-scaled spectrogram to an amplitude spectrogram.

    This effectively inverts `amplitude_to_db`::

        db_to_amplitude(S_db) ~= 10.0**(0.5 * S_db/10 + log10(ref))

    This function caches at level 30.

    Note:
        The functionality in this implementation is basically derived from
        librosa v0.10.1:
        https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py

    Args:
        S_db: Spectrogram input as a numpy array in dB-scaled values.
        ref: A float (number) > 0. Indicates Optional reference amplitude.

    Returns:
        The amplitude spectrogram in linear magnitude values as a numpy array.
    """
    return db_to_power(S_db, ref=ref ** 2) ** 0.5


def spectrogram(
    *,
    y: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    n_fft: Optional[int] = 2048,
    hop_length: Optional[int] = 512,
    power: float = 1,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
) -> Tuple[np.ndarray, int]:
    """Retrieves a magnitude spectrogram.

    This is primarily used in feature extraction functions that can operate on
    either audio time-series or spectrogram input.

    Note:
        The functionality in this implementation is basically derived from
        librosa v0.10.1:
        https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py

    Args:
        y: Time-series as a numpy array.
        S: (np.ndarray) Spectrogram input, optional.
        n_fft: STFT window size as an integer.
        hop_length: STFT hop length as an integer.
        power: A float that shows the exponent for the magnitude spectrogram,
            e.g., 1 for energy, 2 for power, etc.
            win_length: (int) Each frame of audio is windowed by `window`.
            The window will be of length `win_length` and then padded
            with zeros to match `n_fft`.  

            If unspecified, defaults to `win_length = n_fft`.

        window: (string, tuple, number, function, or np.ndarray)
            - a window specification (string, tuple, or number);
                see `scipy.signal.get_window`
            - a window function, such as `scipy.signal.windows.hann`
            - a vector or array of length `n_fft`
        center: A boolean.
            - If `True`, the signal `y` is padded so that frame
                `t` is centered at `y[t * hop_length]`.
            - If `False`, then frame `t` begins at `y[t * hop_length]`
        pad_mode: A string. If `center=True`, the padding mode to use
        at the edges of the signal. By default, STFT uses zero padding.

    Returns:
        S_out as a numpy array:
            - If `S` is provided as input, then `S_out == S`
            - Else, `S_out = |stft(y, ...)|**power`
        n_fft as an integer:
            - If `S` is provided, then `n_fft` is inferred from `S`
            - Else, copied from input
    """

    if S is not None:
        # Infer n_fft from spectrogram shape, but only if it mismatches
        if n_fft is None or n_fft // 2 + 1 != S.shape[-2]:
            n_fft = 2 * (S.shape[-2] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        if n_fft is None:
            raise ValueError(f"Unable to compute spectrogram with n_fft={n_fft}")
        if y is None:
            raise ValueError("Input signal must be provided to compute a spectrogram")
        S = (
            np.abs(
                stft(
                    y,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    center=center,
                    window=window,
                    pad_mode=pad_mode,
                )
            )
            ** power
        )

    return S, n_fft


def mfcc_stats(
    y: Optional[np.ndarray] = None,
    sr: int = 22050,
    S: Optional[np.ndarray] = None,
    n_mfcc: int = 13,
    dct_type: int = 2,
    norm: Optional[str] = "ortho",
    lifter: float = 0,
    export: str = "array",
    **kwargs: Any,
) -> Union[np.ndarray, dict]:
    mfcc_arr = mfcc(y=y, sr=sr, S=S, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm, lifter=lifter, **kwargs)
    delta_arr = delta(mfcc_arr)

    mfcc_mean = np.mean(mfcc_arr, axis=1)
    mfcc_std = np.std(mfcc_arr, axis=1)
    delta_mean = np.mean(delta_arr, axis=1)
    delta2_mean = np.mean(delta(mfcc_arr, order=2), axis=1)

    if export == "array":
        return np.concatenate([mfcc_mean, mfcc_std, delta_mean, delta2_mean], axis=1)
    elif export == "dict":
        return {
            "mfcc_mean": mfcc_mean,
            "mfcc_std": mfcc_std,
            "delta_mean": delta_mean,
            "delta2_mean": delta2_mean,
        }
    else:
        raise ValueError(f"Unsupported export={export}")


def delta(
    data: np.ndarray,
    *,
    width: int = 9,
    order: int = 1,
    axis: int = -1,
    mode: str = "interp",
    **kwargs: Any,
) -> np.ndarray:
    """Computes the delta (derivative) of an input array using the
        Savitzky-Golay filter. The operation is applied along a
        specified axis, and the degree of smoothing and differentiation order
        can be configured. The method checks for parameters validity before
        applying the transformation.

    The Savitzky-Golay filter is commonly used for smoothing and
    differentiation due to its ability to preserve higher moments in the data.

    Args:
        data: Input data as a numpy array. The array must have at least 1
            dimension.
        width: The length of the filter window must be an odd integer greater
            than or equal to 3. Defaults to 9.
        order: The order of the derivative to compute. Must be a positive
            integer. Defaults to 1.
        axis: The axis of the array along which to smooth and differentiate.
            Defaults to -1.
        mode: The mode of the symmetric boundary extension, where 'interp'
            uses an interpolation mechanism. Defaults to "interp".
        **kwargs: Additional keyword arguments passed to the
            `scipy.signal.savgol_filter` function.

    Returns:
        np.ndarray: The filtered and smoothed derivative of the input array.

    Raises:
        ValueError: If the mode is "interp" and the window width exceeds the size of the specified axis.
        ValueError: If the window width is less than 3 or is not an odd integer.
        ValueError: If the order is not a positive integer.
    """
    data = np.atleast_1d(data)

    if mode == "interp" and width > data.shape[axis]:
        raise ValueError(f"when mode='interp', width={width} " f"cannot exceed data.shape[axis]={data.shape[axis]}")

    if width < 3 or np.mod(width, 2) != 1:
        raise ValueError("width must be an odd integer >= 3")

    if order <= 0 or not isinstance(order, (int, np.integer)):
        raise ValueError("order must be a positive integer")

    kwargs.pop("deriv", None)
    kwargs.setdefault("polyorder", order)
    result: np.ndarray = scipy.signal.savgol_filter(data, width, deriv=order, axis=axis, mode=mode, **kwargs)
    return result


def mfcc(
    y: Optional[np.ndarray] = None,
    sr: int = 22050,
    S: Optional[np.ndarray] = None,
    n_mfcc: int = 20,
    dct_type: int = 2,
    norm: Optional[str] = "ortho",
    lifter: float = 0,
    **kwargs: Any,
) -> np.ndarray:
    if S is None:
        # multichannel behavior may be different due to relative noise floor
        # differences between channels
        S = power_to_db(melspectrogram(y=y, sr=sr, **kwargs))

    M: np.ndarray = scipy.fft.dct(S, axis=-2, type=dct_type, norm=norm)[..., :n_mfcc, :]

    if lifter > 0:
        # shape lifter for broadcasting
        LI = np.sin(np.pi * np.arange(1, 1 + n_mfcc, dtype=M.dtype) / lifter)
        LI = expand_to(LI, ndim=S.ndim, axes=-2)

        M *= 1 + (lifter / 2) * LI
        return M
    elif lifter == 0:
        return M
    else:
        raise ValueError(f"MFCC lifter={lifter} must be a non-negative number")


def stft(
        y: np.ndarray,
        *,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Union[str, Tuple[Any, ...], float, Callable[
            [int], np.ndarray], ArrayLike] = "hann",
        center: bool = True,
        dtype: Optional[DTypeLike] = None,
        pad_mode: Union[str, Callable[..., Any]] = "constant",
        out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Short-time Fourier transform (STFT).

    The STFT represents a signal in the time-frequency domain by
    computing discrete Fourier transforms (DFT) over short overlapping
    windows.

    This function returns a complex-valued matrix D such that

    - ``np.abs(D[..., f, t])`` is the magnitude of frequency bin ``f``
      at frame ``t``, and

    - ``np.angle(D[..., f, t])`` is the phase of frequency bin ``f``
      at frame ``t``.

    The integers ``t`` and ``f`` can be converted to physical units by means
    of the utility functions `frames_to_samples` and `fft_frequencies`.

    Note:
        The functionality in this implementation is basically derived from
        librosa v0.10.1:
        https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py

    Args:
        y: The input signal as a numpy array. Multi-channel is supported.

        n_fft: The length of the windowed signal as an integer (scalar) > 0
            after padding with zeros.

            The number of rows in the STFT matrix ``D`` is
            ``(1 + n_fft/2)``. The default value, ``n_fft=2048`` samples,
            corresponds to a physical duration of 93 milliseconds at a sample
            rate of 22050 Hz, i.e., the default sample rate in the rest of the
            spectrum transformation functions. This value is well adapted for
            music signals. However, in speech processing, the recommended
            value is 512, corresponding to 23 milliseconds at a sample rate of
            22050 Hz.

            In any case, we recommend setting ``n_fft`` to a power
            of two for optimizing the speed of the Fast Fourier Transform
            (FFT) algorithm.

        hop_length: The number of audio samples between adjacent STFT columns
            as an integer > 0.

            Smaller values increase the number of columns in ``D`` without
            affecting the frequency resolution of the STFT.

            If unspecified, defaults to ``win_length // 4`` (see in win_length
            argument description).

        win_length: Each frame of audio is windowed by ``window`` of length
            ``win_length``. The win_length is an integer <= n_fft.

            Then, it is padded with zeros to match ``n_fft``. Padding is added
            on both the left and the right sides of the window so that the
            window is centered within the frame.

            Smaller values improve the temporal resolution of the STFT (i.e.,
            the ability to discriminate impulses that are closely spaced in
            time) at the expense of frequency resolution (i.e., the ability to
            discriminate pure tones that are closely spaced in frequency).
            This effect is known as the time-frequency localization trade-off
            and needs to be adjusted according to the properties of the input
            signal ``y``. If unspecified, defaults to ``win_length = n_fft``.

        window: A window specification as a string, tuple, number, function,
            or np.ndarray in the [shape=(n_fft,)].

            Either:
                - a window specification (string, tuple, or number);
                  see `scipy.signal.get_window`
                - a window function, such as `scipy.signal.windows.hann`
                - a vector or array of length ``n_fft``

                Defaults to a raised cosine window (`'hann'`), which is
                adequate for most applications in audio signal processing.

        center: A boolean value. If ``True``, the signal ``y`` is padded so
            that frame ``D[:, t]`` is centered at ``y[t * hop_length]``. If
            ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.

        dtype: It is the form of np.dtype, Optional, and is Complex numeric
            type for ``D``.  Default is inferred to match the precision of the
            input signal.

        pad_mode: A string or function. If ``center=True``,  this argument is
            passed to `np.pad` for padding the edges of the signal ``y``.
            By default (``pad_mode="constant"``), ``y`` is padded on both
            sides with zeros.

            note:: Not all padding modes supported by `numpy.pad` are
            supported. `wrap`, `mean`, `maximum`, `median`, and `minimum` are
            not supported.

            Other modes that depend at most on input values at the edges of the
            signal (e.g., `constant`, `edge`, `linear_ramp`) are supported.

            If ``center=False``,  this argument is ignored.

            .. see also:: `numpy.pad`

        out: Can be np.ndarray or None. It is related to a pre-allocated,
            complex-valued array to store the STFT results. This must be of
            compatible shape and dtype for the given input parameters.

            If `out` is larger than necessary for the provided input signal,
            then only a prefix slice of `out` will be used.

            If not provided, a new array is allocated and returned.


    Returns:
        The STFT result as a numpy array (matrix) in [shape=(..., 1 + n_fft/2,
        n_frames), dtype=dtype]. Complex-valued matrix of short-term Fourier
        Transform coefficients.

        If a pre-allocated `out` array is provided, then `D` will be
        a reference to `out`.

        If `out` is larger than necessary, then `D` will be a sliced
        view: `D = out[..., :n_frames]`.
    """

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)
    elif not is_positive_int(hop_length):
        raise ValueError(f"hop_length={hop_length} must be a positive integer")

    # Check audio is valid
    valid_audio(y, mono=False)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, size=n_fft)

    # Reshape so that the window can be broadcast
    fft_window = expand_to(fft_window, ndim=1 + y.ndim, axes=-2)

    # Pad the time series so that frames are centered
    if center:
        if pad_mode in ("wrap", "maximum", "mean", "median", "minimum"):
            # Note: padding with a user-provided function "works", but
            # use at your own risk.
            # Since we don't passthrough kwargs here, any arguments
            # to a user-provided pad function should be encapsulated
            # by using functools.partial:
            #
            # >>> my_pad_func = functools.partial(pad_func, foo=x, bar=y)
            # >>> librosa.stft(..., pad_mode=my_pad_func)

            raise ValueError(
                f"pad_mode='{pad_mode}' is not supported by librosa.stft")

        if n_fft > y.shape[-1]:
            warnings.warn(
                f"n_fft={n_fft} is too large for input signal of length={y.shape[-1]}")

        # Set up the padding array to be empty, and we'll fix the target dimension later
        padding = [(0, 0) for _ in range(y.ndim)]

        # How many frames depend on left padding?
        start_k = int(np.ceil(n_fft // 2 / hop_length))

        # What's the first frame that depends on extra right-padding?
        tail_k = (y.shape[-1] + n_fft // 2 - n_fft) // hop_length + 1

        if tail_k <= start_k:
            # If tail and head overlap, then just copy-pad the signal and carry on
            start = 0
            extra = 0
            padding[-1] = (n_fft // 2, n_fft // 2)
            y = np.pad(y, padding, mode=pad_mode)
        else:
            # If tail and head do not overlap, then we can implement padding on each part separately
            # and avoid a full copy-pad

            # "Middle" of the signal starts here, and does not depend on head padding
            start = start_k * hop_length - n_fft // 2
            padding[-1] = (n_fft // 2, 0)

            # +1 here is to ensure enough samples to fill the window
            # fixes bug #1567
            y_pre = np.pad(
                y[..., : (start_k - 1) * hop_length - n_fft // 2 + n_fft + 1],
                padding,
                mode=pad_mode,
            )
            y_frames_pre = frame(y_pre, frame_length=n_fft,
                                 hop_length=hop_length)
            # Trim this down to the exact number of frames we should have
            y_frames_pre = y_frames_pre[..., :start_k]

            # How many extra frames do we have from the head?
            extra = y_frames_pre.shape[-1]

            # Determine if we have any frames that will fit inside the tail pad
            if tail_k * hop_length - n_fft // 2 + n_fft <= y.shape[
                -1] + n_fft // 2:
                padding[-1] = (0, n_fft // 2)
                y_post = np.pad(y[..., (tail_k) * hop_length - n_fft // 2:],
                                padding, mode=pad_mode)
                y_frames_post = frame(y_post, frame_length=n_fft,
                                      hop_length=hop_length)
                # How many extra frames do we have from the tail?
                extra += y_frames_post.shape[-1]
            else:
                # In this event, the first frame that touches tail padding would run off
                # the end of the padded array
                # We'll circumvent this by allocating an empty frame buffer for the tail
                # this keeps the subsequent logic simple
                post_shape = list(y_frames_pre.shape)
                post_shape[-1] = 0
                y_frames_post = np.empty_like(y_frames_pre, shape=post_shape)
    else:
        if n_fft > y.shape[-1]:
            raise ValueError(
                f"n_fft={n_fft} is too large for uncentered analysis of input " f"signal of length={y.shape[-1]}"
            )

        # "Middle" of the signal starts at sample 0
        start = 0
        # We have no extra frames
        extra = 0

    if dtype is None:
        dtype = dtype_r2c(y.dtype)

    # Window the time series.
    y_frames = frame(y[..., start:], frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    shape = list(y_frames.shape)

    # This is our frequency dimension
    shape[-2] = 1 + n_fft // 2

    # If there's padding, there will be extra head and tail frames
    shape[-1] += extra

    if out is None:
        stft_matrix = np.zeros(shape, dtype=dtype, order="F")
    elif not (
            np.allclose(out.shape[:-1], shape[:-1]) and out.shape[-1] >= shape[
        -1]):
        raise ValueError(
            f"Shape mismatch for provided output array out.shape={out.shape} " f"and target shape={shape}")
    elif not np.iscomplexobj(out):
        raise ValueError(
            f"output with dtype={out.dtype} is not of complex " f"type")
    else:
        if np.allclose(shape, out.shape):
            stft_matrix = out
        else:
            stft_matrix = out[..., : shape[-1]]

    # Fill in the warm-up
    if center and extra > 0:
        off_start = y_frames_pre.shape[-1]
        stft_matrix[..., :off_start] = fft.rfft(fft_window * y_frames_pre,
                                                axis=-2)

        off_end = y_frames_post.shape[-1]
        if off_end > 0:
            stft_matrix[..., -off_end:] = fft.rfft(fft_window * y_frames_post,
                                                   axis=-2)
    else:
        off_start = 0

    n_columns = int(
        MAX_MEM_BLOCK // (np.prod(y_frames.shape[:-1]) * y_frames.itemsize))
    n_columns = max(n_columns, 1)

    for bl_s in range(0, y_frames.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, y_frames.shape[-1])

        stft_matrix[..., bl_s + off_start: bl_t + off_start] = fft.rfft(
            fft_window * y_frames[..., bl_s:bl_t], axis=-2)

    return stft_matrix


def istft(
        stft_matrix: np.ndarray,
        *,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        n_fft: Optional[int] = None,
        window: Union[str, Tuple[Any, ...], float, Callable[
            [int], np.ndarray], ArrayLike] = "hann",
        center: bool = True,
        dtype: Optional[DTypeLike] = None,
        length: Optional[int] = None,
        out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Inverse short-time Fourier transform (ISTFT).

    Converts a complex-valued spectrogram ``stft_matrix`` to time-series ``y``
    by minimizing the mean squared error between ``stft_matrix`` and STFT of
    ``y`` as described in [#]_ up to Section 2 (reconstruction from MSTFT).

    In general, window function, hop length, and other parameters should be the
    sam as in stft, which mostly leads to perfect reconstruction of a signal
    from unmodified ``stft_matrix``.

    .. [#] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    This function caches at level 30.

    Note:
        The functionality in this implementation is basically derived from
        librosa v0.10.1:
        https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py

    Args:
        stft_matrix: The input STFT matrix as a numpy array. It must be
            complex-valued and have shape ((..., 1 + n_fft//2, t)).

        hop_length: The number of audio samples between adjacent STFT columns
            as an integer > 0.

            If unspecified, defaults to ``win_length // 4`` (see in win_length
            argument description).

        win_length: Each frame of audio is windowed by ``window`` of length
            ``win_length``. The win_length is an
            integer <= n_fft = 2 * (stft_matrix.shape[0] - 1).

            When reconstructing the time series, each frame is windowed,
            and each sample is normalized by the sum of squared-window
            according to the ``window`` function (see below).

            If unspecified, defaults to ``n_fft``.

        n_fft: An integer > 0 or None. It is the number of samples per frame
            in the input spectrogram. By default, this will be inferred from
            the shape of ``stft_matrix``. However, if an odd frame length was
            used, you can specify the correct length by setting ``n_fft``.

        window: A window specification as a string, tuple, number, number,
        function, or np.ndarray [shape=(n_fft,)]
            - a window specification (string, tuple, or number);
            see `scipy.signal.get_window`
            - a window function, such as `scipy.signal.windows.hann`
            - a user-specified window vector of length ``n_fft``

        center: A boolean value.
            - If ``True``, ``D`` is assumed to have centered frames.
            - If ``False``, ``D`` is assumed to have left-aligned frames.

        dtype: It is the form of np.dtype, Optional. It is a real numeric type
            for ``y``.  Defaults to match the numerical precision of the
            input spectrogram.

        length: Integer > 0, Optional. If provided, the output ``y`` is
            zero-padded or clipped to exactly ``length`` samples.

        out: Can be numpy array or None. A pre-allocated, complex-valued array
            to store the reconstructed signal ``y``.  This must be of the
            correct shape for the given input parameters.

            If not provided, a new array is allocated and returned.

    Returns:
        y : Numpy array in the [shape=(..., n)]
            - time domain signal reconstructed from ``stft_matrix``.
            - If ``stft_matrix`` contains more than two axes (e.g., from a
            stereo input signal), then ``y`` will match shape on the leading
            dimensions.
    """

    if n_fft is None:
        n_fft = 2 * (stft_matrix.shape[-2] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    ifft_window = get_window(window, win_length, fftbins=True)

    # Pad out to match n_fft, and add broadcasting axes
    ifft_window = pad_center(ifft_window, size=n_fft)
    ifft_window = expand_to(ifft_window, ndim=stft_matrix.ndim, axes=-2)

    # For efficiency, trim STFT frames according to signal length if available
    if length:
        if center:
            padded_length = length + 2 * (n_fft // 2)
        else:
            padded_length = length
        n_frames = min(stft_matrix.shape[-1],
                       int(np.ceil(padded_length / hop_length)))
    else:
        n_frames = stft_matrix.shape[-1]

    if dtype is None:
        dtype = dtype_c2r(stft_matrix.dtype)

    shape = list(stft_matrix.shape[:-2])
    expected_signal_len = n_fft + hop_length * (n_frames - 1)

    if length:
        expected_signal_len = length
    elif center:
        expected_signal_len -= 2 * (n_fft // 2)

    shape.append(expected_signal_len)

    if out is None:
        y = np.zeros(shape, dtype=dtype)
    elif not np.allclose(out.shape, shape):
        raise ValueError(
            f"Shape mismatch for provided output array " f"out.shape={out.shape} != {shape}")
    else:
        y = out
        # Since we'll be doing overlap-add here, this needs to be initialized to zero.
        y.fill(0.0)

    if center:
        # First frame that does not depend on padding
        #  k * hop_length - n_fft//2 >= 0
        # k * hop_length >= n_fft // 2
        # k >= (n_fft//2 / hop_length)

        start_frame = int(np.ceil((n_fft // 2) / hop_length))

        # Do overlap-add on the head block
        ytmp = ifft_window * fft.irfft(stft_matrix[..., :start_frame], n=n_fft,
                                       axis=-2)

        shape[-1] = n_fft + hop_length * (start_frame - 1)
        head_buffer = np.zeros(shape, dtype=dtype)

        __overlap_add(head_buffer, ytmp, hop_length)

        # If y is smaller than the head buffer, take everything
        if y.shape[-1] < shape[-1] - n_fft // 2:
            y[..., :] = head_buffer[..., n_fft // 2: y.shape[-1] + n_fft // 2]
        else:
            # Trim off the first n_fft//2 samples from the head and copy into target buffer
            y[..., : shape[-1] - n_fft // 2] = head_buffer[..., n_fft // 2:]

        # This offset compensates for any differences between frame alignment
        # and padding truncation
        offset = start_frame * hop_length - n_fft // 2

    else:
        start_frame = 0
        offset = 0

    n_columns = int(MAX_MEM_BLOCK // (
                np.prod(stft_matrix.shape[:-1]) * stft_matrix.itemsize))
    n_columns = max(n_columns, 1)

    frame = 0
    for bl_s in range(start_frame, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)

        # invert the block and apply the window function
        ytmp = ifft_window * fft.irfft(stft_matrix[..., bl_s:bl_t], n=n_fft,
                                       axis=-2)

        # Overlap-add the istft block starting at the i'th frame
        __overlap_add(y[..., frame * hop_length + offset:], ytmp, hop_length)

        frame += bl_t - bl_s

    # Normalize by sum of squared window
    ifft_window_sum = window_sumsquare(
        window=window,
        n_frames=n_frames,
        win_length=win_length,
        n_fft=n_fft,
        hop_length=hop_length,
        dtype=dtype,
    )

    if center:
        start = n_fft // 2
    else:
        start = 0

    ifft_window_sum = fix_length(ifft_window_sum[..., start:],
                                 size=y.shape[-1])

    approx_nonzero_indices = ifft_window_sum > tiny(ifft_window_sum)

    y[..., approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    return y


def fft_frequencies(
    *,
    sr: float = 22050,
    n_fft: int = 2048
) -> np.ndarray:
  """This is an alternative implementation of `np.fft.fftfreq`
  with a predefined window length and the sample spacing calculated as
  1 / sampling rate.

  Args:
      sr: Signal sampling rate as integer.
      n_fft: FFT window size as integer.

  Returns:
      np.ndarray: Frequencies ``(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)``

  Examples:
    >>> fft_frequencies(sr=22050, n_fft=16)
    array([     0.,   1400.17,   2800.24,   4200.83,
            5600.89,   7000.03,   8400.48,   9800.92,   11200.38    ])

  """
  return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
