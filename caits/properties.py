from typing import Any, Optional

import numpy as np
from scipy.signal import hilbert

from .windowing import frame_signal


def amplitude_envelope_hbt(
    signal: np.ndarray, 
    N: Optional[int] = None, 
    axis: Optional[int] = -1
) -> np.ndarray:
    """Calculates the envelope of a signal by computing first the analytic
    signal using the Hilbert transform.

    Args:
        signal: The input signal as a numpy.ndarray.
        N: Number of Fourier components. Default: signal.shape[axis]
        axis: Axis along which to do the transformation. Default: -1.

    Returns:
        numpy.ndarray: The envelope of the input signal.
    """
    analytic_signal = hilbert(x=signal, N=N, axis=axis)
    ae = np.abs(analytic_signal)
    return ae


def instantaneous_frequency_hbt(
    signal: np.ndarray,
    fs: int,
    N: Optional[int] = None,
    axis: Optional[int] = -1
) -> np.ndarray:
    """Calculates the instantaneous frequency of a signal by computing first
    the analytic signal using the Hilbert transform.

    Args:
        signal: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the input signal.
        N: Number of Fourier components. Default: signal.shape[axis]
        axis: Axis along which to do the transformation. Default: -1.

    Returns:
        numpy.ndarray: The instantaneous frequency of the input signal.
    """
    analytic_signal = hilbert(x=signal, N=N, axis=axis)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal), axis=axis)
    instant_freq = np.diff(instantaneous_phase, axis=axis) / (2.0 * np.pi) * fs

    return instant_freq


def instantaneous_amplitude_hbt(signal: np.ndarray, axis: int = 0) -> np.ndarray:
    """Calculates the instantaneous amplitude of a signal by computing first
    the analytic signal using the Hilbert transform.

    Args:
        signal: The input signal as a numpy.ndarray.

    Returns:
        numpy.ndarray: The instantaneous amplitude of the input signal.
    """
    analytic_signal = hilbert(signal, axis=axis)
    ia = np.abs(analytic_signal)
    return ia


def sma_signal(signal, axis: int = 0) -> np.ndarray:
    """Calculates the rolling Simple Moving Average (SMA) between the axes
    of a multi-axis signal in time-domain.
        Formula: sum(abs(signal))

    Args:
        signal: The input signal as a numpy.ndarray.

    Returns:
        numpy.ndarray: The SMA of the input signal.
    """
    return np.sum(np.abs(signal), axis=axis)


def magnitude_signal(signal: np.ndarray, axis: int = 0) -> np.ndarray:
    """Calculates the Magnitude between the axes of a multi-axis signal in
    time-domain.
        Formula: sqrt(sum(signal^2))

    Args:
        signal: The input signal as a numpy.ndarray.

    Returns:
        numpy.ndarray: The magnitude of the input signal.
    """
    return np.sqrt(np.sum(signal**2, axis=axis))


def rolling_rms(
    signal: np.ndarray,
    frame_length: float,
    hop_length: float,
    padding_mode: str = "constant",
    axis: int = 0
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

    if signal.ndim > 1:
        if axis == 0:
            padded_signal = np.pad(signal, ((pad_width, pad_width), (0, 0)), mode=padding_mode)
        else:
            padded_signal = np.pad(signal, ((0, 0), (pad_width, pad_width)), mode=padding_mode)
    else:
        padded_signal = np.pad(signal, pad_width, mode=padding_mode)

    # Calculate the number of frames
    num_frames = 1 + (len(padded_signal) - frame_length) // hop_length

    # Initialize an array to store the RMS values
    if signal.ndim == 1:
        rms_values = np.zeros(num_frames)
    else:
        if axis == 0:
            # rms_values = np.zeros((signal.shape[axis], num_frames))
            rms_values = np.zeros((num_frames, signal.shape[1]))
        else:
            rms_values = np.zeros((signal.shape[axis], num_frames))

    # Calculate RMS for each frame
    for i in range(int(num_frames)):
        if signal.ndim == 1:
            frame = padded_signal[i * hop_length : i * hop_length + frame_length]
            rms_values[i] = np.sqrt(np.mean(frame ** 2))
        else:
            if axis == 0:
                frame = padded_signal[i * hop_length : i * hop_length + frame_length, :]
                rms_values[i, :] = np.sqrt(np.mean(frame ** 2, axis=axis))
            else:
                frame = padded_signal[:, i * hop_length : i * hop_length + frame_length]
                rms_values[:, i] = np.sqrt(np.mean(frame ** 2, axis=axis))


    return rms_values


def rolling_zcr(
    array: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    padding_mode: str = "edge",
    axis: int = 0
) -> np.ndarray:
    """Calculates the rolling Zero Crossing Rate (ZCR) of a signal in
    time-domain. Implementation based on:
    - https://www.sciencedirect.com/topics/engineering/zero-crossing-rate

    Args:
        array: The input signal as a numpy.ndarray.
        frame_length: The length of the frame in samples.
        hop_length: The number of samples to advance between frames (overlap).
        center: If True, the signal is padded on both sides to center the
            frames.
        padding_mode: A string with the padding mode to use when padding the
            signal. Defaults to "edge". Check numpy.pad for more
            information about the relevant padding modes.
            https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    Returns:
        numpy.ndarray: The rolling ZCR of the input signal.
    """
    sig = None
    if center:
        # Reflect padding on both sides for centering frames
        pad_length = frame_length // 2

        if array.ndim > 1:
            if axis == 0:
                sig = np.pad(array, ((pad_length, pad_length), (0, 0)), mode=padding_mode)
            else:
                sig = np.pad(array, ((0, 0), (pad_length, pad_length)), mode=padding_mode)
        else:
            sig = np.pad(array, pad_length, mode=padding_mode)

    frames = frame_signal(sig, frame_length, hop_length)

    # Calculate zero crossings
    # Check where adjacent samples in the frame have different signs and
    # count these occurrences.
    zero_crossings = np.abs(np.diff(np.signbit(frames), axis=axis))
    zcr = np.sum(zero_crossings, axis=axis) / float(frame_length)

    return zcr
