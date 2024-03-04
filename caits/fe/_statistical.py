import numpy as np
from scipy.stats import kurtosis, moment, skew
import math


def std_value(
        array: np.ndarray,
        axis: int = 0
) -> float:
    """Computes the standard deviation of an audio signal.

    Args:
        array: The input signal as a numpy.ndarray.
        axis: The axis along which to compute the standard deviation.
            Defaults to 0.

    Returns:
        float: The standard deviation of the audio signal.
    """
    return np.std(array, axis=axis)


def mean_value(
        array: np.ndarray,
        axis: int = 0
) -> float:
    """Computes the mean of an audio signal.

    Args:
        array: The input signal as a numpy.ndarray.
        axis: The axis along which to compute the mean value.
            Defaults to 0.

    Returns:
        float: The mean of the audio signal.
    """
    return np.mean(array, axis=axis)


def max_value(
        array: np.ndarray,
        axis: int = 0
) -> float:
    """Computes the maximum value of an audio signal.

    Args:
        array: The input signal as a numpy.ndarray.
        axis: The axis along which to compute the maximum value.
            Defaults to 0.

    Returns:
        float: The maximum value of the audio signal.
    """
    return np.max(array, axis=axis)


def min_value(
        array: np.ndarray,
        axis: int = 0
) -> float:
    """Computes the minimum value of a signal.

    Args:
        array: The input signal as a numpy.ndarray.
        axis: The axis along which to compute the minimum value.
            Defaults to 0.

    Returns:
        float: The minimum value of the audio signal.
    """
    return np.min(array, axis=axis)


def kurtosis_value(array: np.ndarray) -> float:
    """Computes the kurtosis of an audio signal.

    Args:
        array: The input signal as a numpy.ndarray.

    Returns:
        float: The kurtosis of the audio signal.
    """
    return kurtosis(array)


def sample_skewness(array):
    """
    Calculate the sample skewness of an array using scipy.

    Args:
        array (numpy.ndarray): Input array.

    Returns:
        float: Sample skewness of the array.

    Raises:
        ValueError: If the input array has less than 3 elements.

    Examples:
        >>> arr = np.array([1, 2, 3, 4, 5])
        >>> sample_skewness(arr)
        0.0
    """
    if len(array) < 3:
        raise ValueError("Input array must have at least 3 elements")

    return skew(array, bias=False)


def rms_value(array: np.ndarray) -> float:
    """Computes the Root Mean Square value of a signal.

    Args:
        array: Input signal as a numpy.ndarray.

    Returns:
        float: The Root Mean Square value of the signal.
    """
    square = 0
    n = len(array)
    # Calculate square
    for i in range(0, n):
        square += (array[i] ** 2)
    # Calculate Mean
    mean = (square / float(n))
    # Calculate Root
    root = math.sqrt(mean)

    return root


def central_moments(array):
    """
    Calculate the 0th, 1st, 2nd, 3rd, and 4th central moments of an array using scipy.

    Args:
        array (numpy.ndarray): Input array.

    Returns:
        tuple: A tuple containing the 0th, 1st, 2nd, 3rd, and 4th central moments.

    Raises:
        ValueError: If the input array is empty.

    Examples:
        >>> arr = np.array([1, 2, 3, 4, 5])
        >>> central_moments(arr)
        (1.0, 0.0, 2.5, 0.0, 26.0)
    """
    if len(array) == 0:
        raise ValueError("Input array is empty")

    moment0 = moment(array, moment=0)
    moment1 = moment(array, moment=1)
    moment2 = moment(array, moment=2)
    moment3 = moment(array, moment=3)
    moment4 = moment(array, moment=4)

    return moment0, moment1, moment2, moment3, moment4


def signal_stats(
        arr: np.ndarray,
        name: str,
        axis: int = 0
) -> dict:
    """Computes the basic statistical information of signal.

    Args:
        arr: A 2D NumPy array.
        name: A string with the name of the input array.
        axis: The axis along which to compute the statistics. Defaults to 0.

    Returns:
        Dict: A dictionary containing the mean, max, min, and STD calculations
        of the signal.
    """

    return {

        f"{name}_max": np.max(arr, axis=axis),
        f"{name}_min": np.min(arr, axis=axis),
        f"{name}_mean": np.mean(arr, axis=axis),
        f"{name}_median": np.median(arr, axis=axis),
        f"{name}_std": np.std(arr, axis=axis),
        f"{name}_var": np.var(arr, axis=axis),
    }
