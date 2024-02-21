import numpy as np


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
