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


def instantaneous_frequency_hbt(
        signal: np.ndarray,
        fs: int
) -> np.ndarray:
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


def rolling_rms(
        signal: np.ndarray,
        frame_length: float,
        hop_length: float,
        padding_mode: str = "constant"
) -> np.ndarray:
    """Calculates the rolling Root Mean Square (RMS) of a signal in
    time-domain.

    Args:
        signal: The input signal as a numpy.ndarray.
        frame_length: The length of the frame in samples.
        hop_length: The number of samples to advance between frames (overlap).
        padding_mode: A string with the padding mode to use when padding the
            signal. Defaults to "constant". Check numpy.pad for more
            information about the relevant padding modes.
            https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    Returns:
        numpy.ndarray: The RMS of the input signal.
    """
    # Pad the signal on both sides
    pad_width = frame_length // 2
    padded_signal = np.pad(signal, pad_width, mode=padding_mode)

    # Calculate the number of frames
    num_frames = 1 + (len(padded_signal) - frame_length) // hop_length

    # Initialize an array to store the RMS values
    rms_values = np.zeros(num_frames)

    # Calculate RMS for each frame
    for i in range(num_frames):
        frame = padded_signal[i * hop_length:i * hop_length + frame_length]
        rms_values[i] = np.sqrt(np.mean(frame**2))

    return rms_values


def sma_signal(signal) -> np.ndarray:
    """Calculates the rolling Simple Moving Average (SMA) between the axes
    of a multi-axis signal in time-domain.
        Formula: sum(abs(signal))

    Args:
        signal: The input signal as a numpy.ndarray.

    Returns:
        numpy.ndarray: The SMA of the input signal.
    """
    return np.sum(np.abs(signal), axis=1)


def magnitude_signal(signal: np.ndarray) -> np.ndarray:
    """Calculates the Magnitude between the axes of a multi-axis signal in
    time-domain.
        Formula: sqrt(sum(signal^2))

    Args:
        signal: The input signal as a numpy.ndarray.

    Returns:
        numpy.ndarray: The magnitude of the input signal.
    """
    return np.sqrt(np.sum(signal**2, axis=1))
