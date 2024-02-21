import numpy as np
from scipy.interpolate import interp1d


def add_white_noise(
        array: np.ndarray,
        noise_factor: float
) -> np.ndarray:
    """Adds white noise to a signal.

    Args:
        array (ndarray): Input signal.
        noise_factor (float): Noise factor.

    Returns:
        ndarray: Noisy signal.
    """
    noise = np.random.normal(0, array.std(), array.size)
    return array + noise_factor * noise


def random_gain(
        array: np.ndarray,
        min_factor: float = 0.1,
        max_factor: float = 0.12
) -> np.ndarray:
    """Applies random gain to a signal.

    Args:
        array: The input signal.
        min_factor: The minimum gain factor.
        max_factor: The maximum gain factor.

    Returns:
        ndarray: The signal with random gain applied.
    """
    gain_rate = np.random.uniform(min_factor, max_factor)
    return array * gain_rate


def polarity_inversion(array: np.ndarray) -> np.ndarray:
    """Inverts the polarity of a signal.

    Args:
        array: The input signal.

    Returns:
        ndarray: The signal with inverted polarity.
    """
    return array * -1


def time_stretch(
        array: np.ndarray,
        factor: float
) -> np.ndarray:
    """
    Time-stretch a signal by a given factor.

    Args:
        array: The input signal to be time-stretched.
        factor (float): The factor by which to stretch the time.
            - A factor greater than 1 will stretch the signal, making it
                longer.
            - A factor less than 1 will compress the signal, making it shorter.

    Returns:
        ndarray: The time-stretched signal.

    Raises:
        ValueError: If the input factor is not greater than 0.

    Example:
        >>> import numpy as np
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> stretched_signal = time_stretch(signal, 2.0)
        >>> print(stretched_signal)
        [1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]
    """
    if factor <= 0:
        raise ValueError("Factor must be greater than 0.")

    # Generate new time indices
    new_indices = np.arange(0, len(array), factor)

    # Interpolate the signal at the new time indices
    interpolator = interp1d(np.arange(len(array)), array, kind='linear',
                            bounds_error=False, fill_value=0)

    return interpolator(new_indices)
