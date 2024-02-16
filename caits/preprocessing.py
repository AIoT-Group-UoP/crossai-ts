import numpy as np
from scipy import signal


def normalize_signal(
        sig: np.ndarray
) -> np.ndarray:
    """Normalizes a signal to the range [-1, 1].

    Args:
        sig: An array-like input signal.

    Returns:
        array-like: The normalized signal.
    """
    try:
        intinfo = np.iinfo(sig.dtype)
        return sig / max(intinfo.max, -intinfo.min)

    except ValueError:  # array is not integer dtype
        return sig / max(sig.max(), -sig.min())


def resample_signal(sig, native_sr, g_sr, d_type=np.float32):
    """Resamples an input audio buffer to the goal sampling rate. Linear
    resampling using numpy is significantly faster than Librosa's default
    technique.

    Args:
        sig: The input signal as a numpy.ndarray.
        native_sr: The native sampling rate of the input signal as integer.
        g_sr(int): The goal sampling rate as integer.
        d_type: The data type of the resampled audio buffer.

    Returns:

    """
    duration = len(sig) / native_sr
    n_target_samples = int(duration * g_sr)
    time_x_source = np.linspace(0, duration, len(sig),
                                dtype=d_type)
    time_x = np.linspace(0, duration, n_target_samples,
                         dtype=d_type)
    resampled_buffer = np.interp(time_x, time_x_source, sig)
    return resampled_buffer
