# The functionalities in this implementation are basically derived from
# librosa v0.10.1:
# https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py
import numpy as np
import scipy
import warnings
from typing import Any, Tuple, Optional, Union, Callable
from numpy import fft
from typing import Literal
from numpy.typing import ArrayLike, DTypeLike
from .base import is_positive_int, valid_audio, dtype_r2c, dtype_c2r
from .base import frame, pad_center, expand_to, get_window, tiny
from .base import __overlap_add, window_sumsquare
from .base import fix_length
from .base._typing_base import _WindowSpec, _PadModeSTFT, _ScalarOrSequence, _ComplexLike_co
from .base._phase import phasor
from .base._utility import mel_filter


# Constrain STFT block sizes to 256 KB
MAX_MEM_BLOCK = 2**8 * 2**10


def stft(
    y: np.ndarray,
    *,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Union[str, Tuple[Any, ...], float, Callable[[int], np.ndarray], ArrayLike] = "hann",
    center: bool = True,
    dtype: Optional[DTypeLike] = None,
    pad_mode: Union[str, Callable[..., Any]] = "constant",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """

    Args:
        y:
        n_fft:
        hop_length:
        win_length:
        window:
        center:
        dtype:
        pad_mode: Could be something of the following in case of using a string
            value: "constant", "edge", "linear_ramp", "reflect", "symmetric",
            "empty".
        out:

    Returns:

    """
    # The functionality in this implementation are basically derived from
    # librosa v0.10.1:
    # https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py

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
            # Since we don't pass-through kwargs here, any arguments
            # to a user-provided pad function should be encapsulated
            # by using functools.partial:
            #
            # >>> my_pad_func = functools.partial(pad_func, foo=x, bar=y)
            # >>> librosa.stft(..., pad_mode=my_pad_func)

            raise ValueError(
                f"pad_mode='{pad_mode}' is not supported by librosa.stft"
            )

        if n_fft > y.shape[-1]:
            warnings.warn(
                f"n_fft={n_fft} is too large for input signal of length={y.shape[-1]}"
            )

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
            y_frames_pre = frame(y_pre, frame_length=n_fft, hop_length=hop_length)
            # Trim this down to the exact number of frames we should have
            y_frames_pre = y_frames_pre[..., :start_k]

            # How many extra frames do we have from the head?
            extra = y_frames_pre.shape[-1]

            # Determine if we have any frames that will fit inside the tail pad
            if tail_k * hop_length - n_fft // 2 + n_fft <= y.shape[-1] + n_fft // 2:
                padding[-1] = (0, n_fft // 2)
                y_post = np.pad(
                    y[..., (tail_k) * hop_length - n_fft // 2 :], padding, mode=pad_mode
                )
                y_frames_post = frame(
                    y_post, frame_length=n_fft, hop_length=hop_length
                )
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
                f"n_fft={n_fft} is too large for uncentered analysis of input "
                f"signal of length={y.shape[-1]}"
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
    elif not (np.allclose(out.shape[:-1], shape[:-1]) and out.shape[-1] >= shape[-1]):
        raise ValueError(
            f"Shape mismatch for provided output array out.shape={out.shape} "
            f"and target shape={shape}"
        )
    elif not np.iscomplexobj(out):
        raise ValueError(f"output with dtype={out.dtype} is not of complex "
                         f"type")
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
        MAX_MEM_BLOCK // (np.prod(y_frames.shape[:-1]) * y_frames.itemsize)
    )
    n_columns = max(n_columns, 1)

    for bl_s in range(0, y_frames.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, y_frames.shape[-1])

        stft_matrix[..., bl_s + off_start : bl_t + off_start] = fft.rfft(
            fft_window * y_frames[..., bl_s:bl_t], axis=-2
        )
    return stft_matrix


def istft(
    stft_matrix: np.ndarray,
    *,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_fft: Optional[int] = None,
    window: Union[str, Tuple[Any, ...], float, Callable[[int], np.ndarray], ArrayLike] = "hann",
    center: bool = True,
    dtype: Optional[DTypeLike] = None,
    length: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    # The functionality in this implementation are basically derived from
    # librosa v0.10.1:
    # https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py

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
        n_frames = min(stft_matrix.shape[-1], int(np.ceil(padded_length / hop_length)))
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
            f"Shape mismatch for provided output array "
            f"out.shape={out.shape} != {shape}"
        )
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
        ytmp = ifft_window * fft.irfft(stft_matrix[..., :start_frame], n=n_fft, axis=-2)

        shape[-1] = n_fft + hop_length * (start_frame - 1)
        head_buffer = np.zeros(shape, dtype=dtype)

        __overlap_add(head_buffer, ytmp, hop_length)

        # If y is smaller than the head buffer, take everything
        if y.shape[-1] < shape[-1] - n_fft // 2:
            y[..., :] = head_buffer[..., n_fft // 2 : y.shape[-1] + n_fft // 2]
        else:
            # Trim off the first n_fft//2 samples from the head and copy into target buffer
            y[..., : shape[-1] - n_fft // 2] = head_buffer[..., n_fft // 2 :]

        # This offset compensates for any differences between frame alignment
        # and padding truncation
        offset = start_frame * hop_length - n_fft // 2

    else:
        start_frame = 0
        offset = 0

    n_columns = int(
        MAX_MEM_BLOCK // (np.prod(stft_matrix.shape[:-1]) * stft_matrix.itemsize)
    )
    n_columns = max(n_columns, 1)

    frame = 0
    for bl_s in range(start_frame, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)

        # invert the block and apply the window function
        ytmp = ifft_window * fft.irfft(stft_matrix[..., bl_s:bl_t], n=n_fft, axis=-2)

        # Overlap-add the istft block starting at the i'th frame
        __overlap_add(y[..., frame * hop_length + offset :], ytmp, hop_length)

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

    ifft_window_sum = fix_length(ifft_window_sum[..., start:], size=y.shape[-1])

    approx_nonzero_indices = ifft_window_sum > tiny(ifft_window_sum)

    y[..., approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    return y


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

    if S is not None:
        # Infer n_fft from spectrogram shape, but only if it mismatches
        if n_fft is None or n_fft // 2 + 1 != S.shape[-2]:
            n_fft = 2 * (S.shape[-2] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        if n_fft is None:
            raise ValueError(
                f"Unable to compute spectrogram with n_fft={n_fft}")
        if y is None:
            raise ValueError(
                "Input signal must be provided to compute a spectrogram"
            )
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

    mfcc_arr = mfcc(y=y, sr=sr, S=S, n_mfcc=n_mfcc, dct_type=dct_type,
                        norm=norm, lifter=lifter, **kwargs)
    delta_arr = delta(mfcc_arr)

    mfcc_mean = np.mean(mfcc_arr, axis=1)
    mfcc_std = np.std(mfcc_arr, axis=1)
    delta_mean = np.mean(delta_arr, axis=1)
    delta2_mean = np.mean(delta(mfcc_arr, order=2), axis=1)

    if export == "array":
        return np.concatenate([mfcc_mean, mfcc_std, delta_mean,
                               delta2_mean],
                              axis=1)
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

    data = np.atleast_1d(data)

    if mode == "interp" and width > data.shape[axis]:
        raise ValueError(
            f"when mode='interp', width={width} "
            f"cannot exceed data.shape[axis]={data.shape[axis]}"
        )

    if width < 3 or np.mod(width, 2) != 1:
        raise ValueError("width must be an odd integer >= 3")

    if order <= 0 or not isinstance(order, (int, np.integer)):
        raise ValueError("order must be a positive integer")

    kwargs.pop("deriv", None)
    kwargs.setdefault("polyorder", order)
    result: np.ndarray = scipy.signal.savgol_filter(
        data, width, deriv=order, axis=axis, mode=mode, **kwargs
    )
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

    M: np.ndarray = scipy.fftpack.dct(S, axis=-2, type=dct_type, norm=norm)[
        ..., :n_mfcc, :
    ]

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


def melspectrogram(
        *,
        y: Optional[np.ndarray] = None,
        sr: float = 22050,
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
) -> Union[np.floating[Any], np.ndarray]:

    # The functionality in this implementation are basically derived from
    # librosa v0.10.1:
    # https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py

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


def power_to_db(
    S: np.ndarray,
    *,
    ref: Union[float, Callable] = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = 80.0,
) -> np.ndarray:
    # The functionality in this implementation is basically derived from
    # librosa v0.10.1:
    # https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.
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


def db_to_power(S_db: np.ndarray, *, ref: float = 1.0) -> np.ndarray:
    # The functionality in this implementation is basically derived from
    # librosa v0.10.1:
    # https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py
    """Convert a dB-scale spectrogram to a power spectrogram.

    This effectively inverts ``power_to_db``::

        db_to_power(S_db) ~= ref * 10.0**(S_db / 10)
    """
    return ref * np.power(10.0, 0.1 * S_db)


def amplitude_to_db(
    S: np.ndarray,
    *,
    ref: Union[float, Callable] = 1.0,
    amin: float = 1e-5,
    top_db: Optional[float] = 80.0,
) -> np.ndarray:
    # The functionality in this implementation is basically derived from
    # librosa v0.10.1:
    # https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py
    """Convert an amplitude spectrogram to dB-scaled spectrogram.

    This is equivalent to ``power_to_db(S**2, ref=ref**2, amin=amin**2, top_db=top_db)``,
    but is provided for convenience.
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

    return power_to_db(power, ref=ref_value**2, amin=amin**2, top_db=top_db)


def db_to_amplitude(S_db: np.ndarray, *, ref: float = 1.0) -> np.ndarray:
    """Convert a dB-scaled spectrogram to an amplitude spectrogram.

    This effectively inverts `amplitude_to_db`::

        db_to_amplitude(S_db) ~= 10.0**(0.5 * S_db/10 + log10(ref))
    """
    # The functionality in this implementation is basically derived from
    # librosa v0.10.1:
    # https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py
    return db_to_power(S_db, ref=ref**2) ** 0.5


def griffinlim(
    S: np.ndarray,
    *,
    n_iter: int = 32,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_fft: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    dtype: Optional[DTypeLike] = None,
    length: Optional[int] = None,
    pad_mode: _PadModeSTFT = "constant",
    momentum: float = 0.99,
    init: Optional[str] = "random",
    random_state: Optional[
        Union[int, np.random.RandomState, np.random.Generator]
    ] = None,
) -> np.ndarray:
    """Approximate magnitude spectrogram inversion using the "fast" Griffin-Lim algorithm.

    Given a short-time Fourier transform magnitude matrix (``S``), the algorithm randomly
    initializes phase estimates, and then alternates forward- and inverse-STFT
    operations. [#]_

    Note that this assumes reconstruction of a real-valued time-domain signal, and
    that ``S`` contains only the non-negative frequencies (as computed by
    `stft`).

    The "fast" GL method [#]_ uses a momentum parameter to accelerate convergence.

    .. [#] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    .. [#] Perraudin, N., Balazs, P., & Søndergaard, P. L.
        "A fast Griffin-Lim algorithm,"
        IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4),
        Oct. 2013.

    Args:
        S: np.ndarray [shape=(..., n_fft // 2 + 1, t), non-negative]
            An array of short-time Fourier transform magnitudes as produced by
            `stft`.

        n_iter: int > 0
            The number of iterations to run

        hop_length: None or int > 0
            The hop length of the STFT.  If not provided, it will default to ``n_fft // 4``

        win_length: None or int > 0
            The window length of the STFT.  By default, it will equal ``n_fft``

        n_fft: None or int > 0
            The number of samples per frame.
            By default, this will be inferred from the shape of ``S`` as an even number.
            However, if an odd frame length was used, you can explicitly set ``n_fft``.

        window: string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
            A window specification as supported by `stft` or `istft`

        center: boolean
            If ``True``, the STFT is assumed to use centered frames.
            If ``False``, the STFT is assumed to use left-aligned frames.

        dtype: np.dtype
            Real numeric type for the time-domain signal.  Default is inferred
            to match the precision of the input spectrogram.

        length: None or int > 0
            If provided, the output ``y`` is zero-padded or clipped to exactly ``length``
            samples.

        pad_mode: string
            If ``center=True``, the padding mode to use at the edges of the signal.
            By default, STFT uses zero padding.

        momentum: number >= 0
            The momentum parameter for fast Griffin-Lim.
            Setting this to 0 recovers the original Griffin-Lim method [1]_.
            Values near 1 can lead to faster convergence, but above 1 may not converge.

        init: None or 'random' [default]
            If 'random' (the default), then phase values are initialized randomly
            according to ``random_state``.  This is recommended when the input ``S`` is
            a magnitude spectrogram with no initial phase estimates.

            If `None`, then the phase is initialized from ``S``.  This is useful when
            an initial guess for phase can be provided, or when you want to resume
            Griffin-Lim from a previous output.

        random_state: None, int, np.random.RandomState, or np.random.Generator
            If int, random_state is the seed used by the random number generator
            for phase initialization.

            If `np.random.RandomState` or `np.random.Generator` instance, the random number
            generator itself.

            If `None`, defaults to the `np.random.default_rng()` object.

        Returns:
            y: np.ndarray [shape=(..., n)]
                time-domain signal reconstructed from ``S``
    """
    # The functionality in this implementation is basically derived from
    # librosa v0.10.1:
    # https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)  # type: ignore
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rng = random_state  # type: ignore
    else:
        raise ValueError(f"Unsupported random_state={random_state!r}")

    if momentum > 1:
        warnings.warn(
            f"Griffin-Lim with momentum={momentum} > 1 can be unstable. "
            "Proceed with caution!",
            stacklevel=2,
        )
    elif momentum < 0:
        raise ValueError(f"griffinlim() called with momentum={momentum} < 0")

    # Infer n_fft from the spectrogram shape
    if n_fft is None:
        n_fft = 2 * (S.shape[-2] - 1)

    # Infer the dtype from S
    angles = np.empty(S.shape, dtype=dtype_r2c(S.dtype))
    eps = tiny(angles)

    if init == "random":
        # randomly initialize the phase
        angles[:] = phasor((2 * np.pi * rng.random(size=S.shape)))
    elif init is None:
        # Initialize an all ones complex matrix
        angles[:] = 1.0
    else:
        raise ValueError(f"init={init} must either None or 'random'")

    # Place-holders for temporary data and reconstructed buffer
    rebuilt = None
    tprev = None
    inverse = None

    # Absorb magnitudes into angles
    angles *= S
    for _ in range(n_iter):
        # Invert with our current estimate of the phases
        inverse = istft(
            angles,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            window=window,
            center=center,
            dtype=dtype,
            length=length,
            out=inverse,
        )

        # Rebuild the spectrogram
        rebuilt = stft(
            inverse,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            out=rebuilt,
        )

        # Update our phase estimates
        angles[:] = rebuilt
        if tprev is not None:
            angles -= (momentum / (1 + momentum)) * tprev
        angles /= np.abs(angles) + eps
        angles *= S
        # Store
        rebuilt, tprev = tprev, rebuilt

    # Return the final phase estimates
    return istft(
        angles,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window=window,
        center=center,
        dtype=dtype,
        length=length,
        out=inverse,
    )