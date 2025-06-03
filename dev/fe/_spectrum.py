# The functionalities in this implementation are basically derived from
# librosa v0.10.1:
# https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py
import warnings
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import scipy
from numpy import fft

from caits.core.numpy_typing import ArrayLike, DTypeLike

from caits.core._core_checks import dtype_c2r, dtype_r2c, is_positive_int, valid_audio
from caits.core._core_fix import fix_length
from caits.core._core_typing import _ComplexLike_co, _PadModeSTFT, _ScalarOrSequence, _WindowSpec
from caits.core._core_window import frame, get_window, pad_center, tiny, window_sumsquare
from caits.fe.core_spectrum import __overlap_add, expand_to
from caits.fe.core_spectrum._utils import mel_filter
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import scipy
from numpy.lib.stride_tricks import as_strided

from caits.core.numpy_typing import ArrayLike, DTypeLike

from caits.core._core_typing import _FloatLike_co, _WindowSpec


MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10

import numpy as np
from numpy.lib.stride_tricks import as_strided


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
    axis: int = 0
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

    _y = y.T if axis == 1 else y

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)
    elif not is_positive_int(hop_length):
        raise ValueError(f"hop_length={hop_length} must be a positive integer")

    # Check audio is valid
    valid_audio(_y, mono=False)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, size=n_fft)

    # Reshape so that the window can be broadcast
    fft_window = expand_to(fft_window, ndim=1 + _y.ndim, axes=-2)

    # Pad the time series so that frames are centered
    if center:
        if pad_mode in ("wrap", "maximum", "mean", "median", "minimum"):
            # Note: padding with a user-provided function "works", but
            # use at your own risk.
            # Since we don't passthrough kwargs here, any arguments
            # to a user-provided pad function should be encapsulated
            # by using functools.partial:
            #
            # >>> my_pad_func = functools.partial(pad_func, foo=x, bar=_y)
            # >>> librosa.stft(..., pad_mode=my_pad_func)

            raise ValueError(f"pad_mode='{pad_mode}' is not supported by librosa.stft")

        if n_fft > _y.shape[-1]:
            warnings.warn(f"n_fft={n_fft} is too large for input signal of length={_y.shape[-1]}")

        # Set up the padding array to be empty, and we'll fix the target dimension later
        padding = [(0, 0) for _ in range(_y.ndim)]

        # How many frames depend on left padding?
        start_k = int(np.ceil(n_fft // 2 / hop_length))

        # What's the first frame that depends on extra right-padding?
        tail_k = (_y.shape[-1] + n_fft // 2 - n_fft) // hop_length + 1

        if tail_k <= start_k:
            # If tail and head overlap, then just copy-pad the signal and carry on
            start = 0
            extra = 0
            padding[-1] = (n_fft // 2, n_fft // 2)
            _y = np.pad(_y, padding, mode=pad_mode)
        else:
            # If tail and head do not overlap, then we can implement padding on each part separately
            # and avoid a full copy-pad

            # "Middle" of the signal starts here, and does not depend on head padding
            start = start_k * hop_length - n_fft // 2
            padding[-1] = (n_fft // 2, 0)

            # +1 here is to ensure enough samples to fill the window
            # fixes bug #1567
            y_pre = np.pad(
                _y[..., : (start_k - 1) * hop_length - n_fft // 2 + n_fft + 1],
                padding,
                mode=pad_mode,
            )
            y_frames_pre = frame(y_pre, frame_length=n_fft, hop_length=hop_length)
            # Trim this down to the exact number of frames we should have
            y_frames_pre = y_frames_pre[..., :start_k]

            # How many extra frames do we have from the head?
            extra = y_frames_pre.shape[-1]

            # Determine if we have any frames that will fit inside the tail pad
            if tail_k * hop_length - n_fft // 2 + n_fft <= _y.shape[-1] + n_fft // 2:
                padding[-1] = (0, n_fft // 2)
                y_post = np.pad(_y[..., (tail_k) * hop_length - n_fft // 2:], padding, mode=pad_mode)
                y_frames_post = frame(y_post, frame_length=n_fft, hop_length=hop_length)
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
        if n_fft > _y.shape[-1]:
            raise ValueError(
                f"n_fft={n_fft} is too large for uncentered analysis of input " f"signal of length={_y.shape[-1]}"
            )

        # "Middle" of the signal starts at sample 0
        start = 0
        # We have no extra frames
        extra = 0

    if dtype is None:
        dtype = dtype_r2c(_y.dtype)

    # Window the time series.
    y_frames = frame(_y[..., start:], frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    shape = list(y_frames.shape)

    # This is our frequency dimension
    shape[-2] = 1 + n_fft // 2

    # If there's padding, there will be extra head and tail frames
    shape[-1] += extra

    if out is None:
        stft_matrix = np.zeros(shape, dtype=dtype, order="F")
    elif not (np.allclose(out.shape[:-1], shape[:-1]) and out.shape[-1] >= shape[-1]):
        raise ValueError(f"Shape mismatch for provided output array out.shape={out.shape} " f"and target shape={shape}")
    elif not np.iscomplexobj(out):
        raise ValueError(f"output with dtype={out.dtype} is not of complex " f"type")
    else:
        if np.allclose(shape, out.shape):
            stft_matrix = out
        else:
            stft_matrix = out[..., : shape[-1]]

    # Fill in the warm-up
    if center and extra > 0:
        off_start = y_frames_pre.shape[-1]
        stft_matrix[..., :off_start] = fft.rfft(fft_window * y_frames_pre, axis=-2)

        off_end = y_frames_post.shape[-1]
        if off_end > 0:
            stft_matrix[..., -off_end:] = fft.rfft(fft_window * y_frames_post, axis=-2)
    else:
        off_start = 0

    n_columns = int(MAX_MEM_BLOCK // (np.prod(y_frames.shape[:-1]) * y_frames.itemsize))
    n_columns = max(n_columns, 1)

    for bl_s in range(0, y_frames.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, y_frames.shape[-1])

        stft_matrix[..., bl_s + off_start: bl_t + off_start] = fft.rfft(fft_window * y_frames[..., bl_s:bl_t], axis=-2)
    return stft_matrix

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
    axis = 1,
) -> Tuple[np.ndarray, int]:
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
                    axis=axis,
                )
            )
            ** power
        )

    return S, n_fft