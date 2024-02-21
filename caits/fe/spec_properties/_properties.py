import numpy as np


def spec_to_power(
        spec: np.ndarray
) -> np.ndarray:
    """Transforms a complex-valued spectrogram to a power spectrogram.

    Args: Input complex-valued spectrogram in np.ndarray.

    Returns:
        mp.ndarray: The power spectrogram.
    """
    return np.abs(spec)**2


def power_to_db(
        power_spectrogram: np.ndarray,
        ref=1.0
) -> np.ndarray:
    """Converts a power spectrogram to decibel (dB) units.
    Args:
        power_spectrogram: Input power spectrogram in np.ndarray.
        ref: Reference power level (in amplitude squared) for dB calculation.
            Defaults to 1.0.

    Returns:
        np.ndarray: The power spectrogram in dB.
    """
    return 10 * np.log10(power_spectrogram / ref)