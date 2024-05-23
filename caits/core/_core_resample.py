# This implementation is basically derived from librosa v0.10.1
# https://github.com/librosa/librosa/blob/main/librosa/core/audio.py
from typing import Any

import numpy as np
import resampy
import samplerate
import scipy
import soxr

from ._core_fix import fix_length


def resample(
    y: np.ndarray,
    *,
    orig_sr: float,
    target_sr: float,
    res_type: str = "soxr_hq",
    fix: bool = True,
    scale: bool = False,
    axis: int = -1,
    **kwargs: Any,
) -> np.ndarray:
    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr

    n_samples = int(np.ceil(y.shape[axis] * ratio))

    if res_type in ("scipy", "fft"):
        y_hat = scipy.signal.resample(y, n_samples, axis=axis)
    elif res_type == "polyphase":
        if int(orig_sr) != orig_sr or int(target_sr) != target_sr:
            raise ValueError("polyphase resampling is only supported for integer-valued " "sampling rates.")

        # For polyphase resampling, we need up- and down-sampling ratios
        # We can get those from the greatest common divisor of the rates
        # as long as the rates are integrable
        orig_sr = int(orig_sr)
        target_sr = int(target_sr)
        gcd = np.gcd(orig_sr, target_sr)
        y_hat = scipy.signal.resample_poly(y, target_sr // gcd, orig_sr // gcd, axis=axis)
    elif res_type in (
        "linear",
        "zero_order_hold",
        "sinc_best",
        "sinc_fastest",
        "sinc_medium",
    ):
        # Use numpy to vectorize the resampler along the target axis
        # This is because samplerate does not support ndim>2 generally.
        y_hat = np.apply_along_axis(samplerate.resample, axis=axis, arr=y, ratio=ratio, converter_type=res_type)
    elif res_type.startswith("soxr"):
        # Use numpy to vectorize the resampler along the target axis
        # This is because soxr does not support ndim>2 generally.
        y_hat = np.apply_along_axis(
            soxr.resample,
            axis=axis,
            arr=y,
            in_rate=orig_sr,
            out_rate=target_sr,
            quality=res_type,
        )
    else:
        y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=axis)

    if fix:
        y_hat = fix_length(y_hat, size=n_samples, axis=axis, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    # Match dtypes
    return np.asarray(y_hat, dtype=y.dtype)
