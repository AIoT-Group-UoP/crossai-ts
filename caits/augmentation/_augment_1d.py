import numpy as np


def add_noise(
        array: np.ndarray,
        noise_factor: float
) -> np.ndarray:
    """Adds noise to a signal.

    Args:
        array (ndarray): Input signal.
        noise_factor (float): Noise factor.

    Returns:
        ndarray: Noisy signal.
    """
    noise = np.random.normal(0, array.std(), array.size)
    return array + noise_factor * noise
