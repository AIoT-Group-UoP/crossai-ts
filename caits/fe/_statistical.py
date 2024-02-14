import numpy as np
import scipy.signal
import scipy.stats


def std_value(signal: np.ndarray) -> float:
    """Computes the standard deviation of an audio signal.

    Args:
        signal: The input signal as a numpy.ndarray.

    Returns:
        float: The standard deviation of the audio signal.
    """
    return np.std(signal)


def mean_value(signal: np.ndarray) -> float:
    """Computes the mean of an audio signal.

    Args:
        signal: The input signal as a numpy.ndarray.

    Returns:
        float: The mean of the audio signal.
    """
    return np.mean(signal)


def max_value(signal: np.ndarray) -> float:
    """Computes the maximum value of an audio signal.

    Args:
        signal: The input signal as a numpy.ndarray.

    Returns:
        float: The maximum value of the audio signal.
    """
    return np.max(signal)


def min_value(signal: np.ndarray) -> float:
    """Computes the minimum value of a signal.

    Args:
        signal: The input signal as a numpy.ndarray.

    Returns:
        float: The minimum value of the audio signal.
    """
    return np.min(signal)


def kurtosis(signal: np.ndarray) -> float:
    """Computes the kurtosis of an audio signal.

    Args:
        signal: The input signal as a numpy.ndarray.

    Returns:
        float: The kurtosis of the audio signal.
    """
    return scipy.stats.kurtosis(signal)


def root_mean_square(signal: np.ndarray) -> float:
    """Computes the Root Mean Square (RMS) of an audio signal.

    Args:
        signal: The input signal as a numpy.ndarray.

    Returns:
        float: The RMS of the audio signal.
    """
    return np.sqrt(np.mean(np.square(signal)))
