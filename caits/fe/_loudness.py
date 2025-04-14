import math

import numpy as np

from ._statistical import rms_value


def dBFS(
    array: np.ndarray,
    sample_width: int
) -> float:
    """Calculates the decibels relative to full scale (dBFS) of an audio.

    Args:
        array: The audio data (signal) as a numpy array.
        sample_width: The sample width as an integer.

    Returns:
        float: The decibels relative to full scale (dBFS) of an audio signal.
    """
    rms = rms_value(array)
    if not rms:
        return -float("infinity")
    return ratio_to_db(rms / max_possible_amplitude(sample_width))


def max_possible_amplitude(sample_width: int) -> float:
    """Calculates the maximum possible amplitude for a given sample width.

    Args:
        sample_width: The sample width as an integer.

    Returns:
        float: The maximum possible amplitude of the signal.
    """
    bits = sample_width * 8
    max_possible_val = 2**bits

    # since half is above 0 and half is below the max amplitude is divided
    return max_possible_val / 2


def ratio_to_db(
    ratio,
    val2=None,
    using_amplitude=True
) -> float:
    """Converts the input float to db, which represents the equivalent
        to the ratio in power represented by the multiplier passed in.

    Args:
        ratio: The ratio to convert to dB.
        val2: The second value to use in the ratio calculation.
        using_amplitude: Whether to use amplitude or power in the calculation.

    Returns:
        float: The ratio in dB.
    """
    ratio = float(ratio)

    # accept 2 values and use the ratio of val1 to val2
    if val2 is not None:
        ratio = ratio / val2

    # special case for multiply-by-zero (convert to silence)
    if ratio == 0:
        return -float("inf")

    if using_amplitude:
        return 20 * math.log(ratio, 10)
    else:  # using power
        return 10 * math.log(ratio, 10)
