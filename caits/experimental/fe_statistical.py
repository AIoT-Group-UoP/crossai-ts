import math
import numpy as np


def rms_dbfs(arr: np.ndarray) -> float:
    """Calculates the root mean square of an audio signal using math module.

    Args:
        arr: The input audio signal as a numpy.ndarray.

    Returns:
        float: The root mean square of the audio signal.
    """
    square = 0
    mean = 0.0
    root = 0.0

    # Calculate square
    n = len(arr)
    for i in range(0, n):
        square += (arr[i] ** 2)
    # Calculate Mean
    mean = (square / (float)(n))
    # Calculate Root
    root = math.sqrt(mean)

    return root
