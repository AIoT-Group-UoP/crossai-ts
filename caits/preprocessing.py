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


def resample_signal(
        sig: np.ndarray,
        target: int
) -> np.ndarray:
    """Resamples a signal to a new length using linear interpolation.

    Args:
        sig: An array-like input signal.
        target: Integer length of the desired resampled signal.

    Returns:
        array-like: The resampled signal.
    """
    old_length = sig.shape[0]
    old_indices = np.arange(old_length)
    new_indices = np.linspace(0, old_length - 1, target)

    if sig.ndim == 1:
        return np.interp(new_indices, old_indices, sig)
    else:
        resampled_signal = np.zeros((target, signal.shape[1]))
        for i in range(signal.shape[1]):
            resampled_signal[:, i] = np.interp(new_indices, old_indices,
                                               signal[:, i])

        return resampled_signal
