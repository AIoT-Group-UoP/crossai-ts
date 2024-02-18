import numpy as np
import scipy.signal
import scipy.stats
import math


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


def rms_value(sig: np.ndarray) -> float:
    """Computes the Root Mean Square value of a signal.

    Args:
        sig: Input signal as a numpy.ndarray.

    Returns:
        float: The Root Mean Square value of the signal.
    """
    square = 0
    n = len(sig)
    # Calculate square
    for i in range(0, n):
        square += (sig[i] ** 2)
    # Calculate Mean
    mean = (square / float(n))
    # Calculate Root
    root = math.sqrt(mean)

    return root


def signal_stats(
        arr: np.ndarray,
        name: str
) -> dict:
    """Computes the basic statistical information of signal.

    Args:
        arr: A 2D NumPy array.
        name: A string with the name of the input array.

    Returns:
        Dict: A dictionary containing the mean, max, min, and STD calculations
        of the signal.
    """

    return {
        f"{name}_mean": np.mean(arr),
        f"{name}_max": np.max(arr),
        f"{name}_min": np.min(arr),
        f"{name}_std": np.std(arr)
    }
