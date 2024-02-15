import numpy as np
from scipy.signal import hilbert


def amplitude_envelope_hbt(signal: np.ndarray) -> np.ndarray:
    """Calculates the envelope of a signal by computing first the analytic
    signal using the Hilbert transform.

    Args:
        signal: The input signal as a numpy.ndarray.

    Returns:
        numpy.ndarray: The envelope of the input signal.
    """
    analytic_signal = hilbert(signal)
    ae = np.abs(analytic_signal)
    return ae


def instantaneous_frequency_hbt(signal: np.ndarray, fs: int) -> np.ndarray:
    """Calculates the instantaneous frequency of a signal by computing first
    the analytic signal using the Hilbert transform.

    Args:
        signal: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the input signal.

    Returns:
        numpy.ndarray: The instantaneous frequency of the input signal.
    """
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instant_freq = (np.diff(instantaneous_phase) / (2.0 * np.pi) * fs)

    return instant_freq


def instantaneous_amplitude_hbt(signal: np.ndarray) -> np.ndarray:
    """Calculates the instantaneous amplitude of a signal by computing first
    the analytic signal using the Hilbert transform.

    Args:
        signal: The input signal as a numpy.ndarray.

    Returns:
        numpy.ndarray: The instantaneous amplitude of the input signal.
    """
    analytic_signal = hilbert(signal)
    ia = np.abs(analytic_signal)
    return ia
