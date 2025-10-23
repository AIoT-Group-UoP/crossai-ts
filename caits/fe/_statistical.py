from typing import Dict, Union, Any

import numpy as np
import scipy
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import kurtosis, moment, skew

from ..properties import rolling_rms, rolling_zcr
from ._spectrum import mfcc

# --- STATISTICAL ---

def std_value(
    array: np.ndarray,
    ddof: int = 0,
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
    return np.std(array, ddof=ddof, axis=axis)


def variance_value(
    array: np.ndarray,
    ddof: int = 0,
    axis: int = 0
) -> float:
    """Computes the variance of an audio signal.

    Args:
        array: The input signal as a numpy.ndarray.
        axis: The axis along which to compute the variance.
            Defaults to 0.

    Returns:
        float: The variance of the audio signal.
    """
    return np.var(array, ddof=ddof, axis=axis)


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


def median_value(
    array: np.ndarray,
    axis: int = 0
) -> float:
    """Computes the median of an audio signal.

    Args:
        array: The input signal as a numpy.ndarray.
        axis: The axis along which to compute the median value.
            Defaults to 0.

    Returns:
        float: The median of the audio signal.
    """
    return np.median(array, axis=axis)


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


def kurtosis_value(
    array: np.ndarray,
    axis: int = 0
) -> float:
    """Computes the kurtosis of an audio signal.

    Args:
        array: The input signal as a numpy.ndarray.
        axis: The axis along which to compute the kurtosis. Defaults to 0.

    Returns:
        float: The kurtosis of the audio signal.
    """
    return kurtosis(array, axis=axis)


def sample_skewness(
    array: np.ndarray,
    axis: int = 0
) -> float:
    """
    Calculate the sample skewness of an array using scipy.

    Args:
        array (numpy.ndarray): Input array.
        axis (int): Axis along which to compute the skewness. Defaults to 0.

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

    return skew(array, axis=axis, bias=False)


def signal_length(
    array: np.ndarray,
    fs: float,
    time_mode: str = "time",
    axis: int = 0
) -> float:
    """Computes the length of a signal in seconds.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal as a float.
        time_mode: The export format. Can be "time" or "samples". Defaults to
            "time".

    Returns:
        float: The length of the signal in seconds.
    """
    if array.ndim == 1:
        axis = 0

    if time_mode == "time":
        return array.shape[axis] / fs
    elif time_mode == "samples":
        return array.shape[axis]
    else:
        raise ValueError(f"Unsupported export={time_mode}")


def central_moments(
    array: np.ndarray,
    export: str = "array",
    axis: int = 0
) -> Union[np.ndarray, Dict[str, float]]:
    """
    Calculate the 0th, 1st, 2nd, 3rd, and 4th central moments of an array using
    scipy.

    Args:
        array: The input signal as a numpy.ndarray.
        export: The export format. Can be "array" or "dict". Defaults to
            "array".

    Returns:
        Union[np.ndarray, dict]: The central moments of the input array.

    Raises:
        ValueError: If the input array is empty.

    Examples:
        >>> arr = np.array([1, 2, 3, 4, 5])
        >>> central_moments(arr)
        (1.0, 0.0, 2.5, 0.0, 26.0)
    """
    if len(array) == 0:
        raise ValueError("Input array is empty")

    moment0 = moment(array, moment=0, axis=axis)
    moment1 = moment(array, moment=1, axis=axis)
    moment2 = moment(array, moment=2, axis=axis)
    moment3 = moment(array, moment=3, axis=axis)
    moment4 = moment(array, moment=4, axis=axis)
    if export == "array":
        return np.array([moment0, moment1, moment2, moment3, moment4])
    elif export == "dict":
        return {
            "moment0": moment0,
            "moment1": moment1,
            "moment2": moment2,
            "moment3": moment3,
            "moment4": moment4
        }
    else:
        raise ValueError(f"Unsupported export={export}")

# --- ENERGY ---

def rms_value(
    array: np.ndarray,
    axis: int = 0
) -> float:
    """Computes the RMS Power value of a signal.

    Args:
        array: The input signal as a numpy.ndarray.

    Returns:
        float: The RMS Power of the signal.
    """
    return np.sqrt(np.mean(np.square(array), axis=axis))


def rms_max(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int,
    axis: int = 0,
    **kwargs: Any
) -> np.ndarray:
    """Computes the maximum of the rolling Root Mean Square (RMS) values of a
    signal.

    Args:
        signal: The input signal.
        frame_length: The length of the frame in samples.
        hop_length: The number of samples to advance between frames (overlap).
        **kwargs: Additional keyword arguments passed to `rolling_rms`.

    Returns:
        float: The maximum RMS value.
    """
    rms_values = rolling_rms(signal, frame_length, hop_length, **kwargs)
    return np.max(rms_values, axis=axis)


def rms_mean(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int,
    axis: int = 0,
    **kwargs: Any
) -> np.ndarray:
    """Computes the mean of the rolling Root Mean Square (RMS) values of a
    signal.

    Args:
        signal: The input signal.
        frame_length: The length of the frame in samples.
        hop_length: The number of samples to advance between frames (overlap).
        **kwargs: Additional keyword arguments passed to `rolling_rms`.

    Returns:
        float: The mean RMS value.
    """
    rms_values = rolling_rms(signal, frame_length, hop_length, **kwargs)
    return np.mean(rms_values, axis=axis)


def rms_min(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int,
    axis: int = 0,
    **kwargs: Any
) -> np.ndarray:
    """Computes the minimum of the rolling Root Mean Square (RMS) values of a
    signal.

    Args:
        signal: The input signal.
        frame_length: The length of the frame in samples.
        hop_length: The number of samples to advance between frames (overlap).
        **kwargs: Additional keyword arguments passed to `rolling_rms`.

    Returns:
        float: The minimum RMS value.
    """
    rms_values = rolling_rms(signal, frame_length, hop_length, **kwargs)
    if rms_values.ndim == 1:
        return np.min(rms_values)
    else:
        return np.min(rms_values, axis=axis)


def zcr_value(
    array: np.ndarray,
    axis: int = 0
) -> float:
    """Computes the zero crossing rate of a signal.

    Args:
        array: The input signal as a numpy.ndarray.
        axis: The axis to compute the zcr for. Default is 0.

    Returns:
        float: The zero crossing rate of the signal.
    """
    if array.ndim == 1:
        return float(np.sum(np.multiply(array[0:-1], array[1:]) < 0) / (len(array) - 1))
    elif axis == 1:
        return np.sum(np.multiply(array[:, :-1], array[:, 1:]) < 0, axis=axis) / (array.shape[1] - 1)
    else:
        return np.sum(np.multiply(array[:-1, :], array[1:, :]) < 0, axis=axis) / (array.shape[0] - 1)


def zcr_max(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int,
    axis: int = 0,
    **kwargs: Any
) -> np.ndarray:
    """Computes the maximum value of the rolling zero crossing rate of a
    signal.

    Args:
        signal: The input signal as a numpy.ndarray.
        frame_length: The length of the frame in samples.
        hop_length: The number of samples to advance between frames (overlap).
        axis: The axis to compute the zcr for. Default is 0.
        **kwargs: Additional keyword arguments passed to `rolling_zcr`.

    Returns:
        float: The maximum of the rolling RMS of the input signal.
    """
    zcr_values = rolling_zcr(signal, frame_length, hop_length, **kwargs)
    if zcr_values.ndim == 1:
        return np.max(zcr_values)
    else:
        return np.max(zcr_values, axis=axis)


def zcr_mean(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int,
    axis: int = 0,
    **kwargs: Any
) -> np.ndarray:
    """Computes the mean value of the rolling zero crossing rate of a signal.

    Args:
        signal: The input signal as a numpy.ndarray.
        frame_length: The length of the frame in samples.
        hop_length: The number of samples to advance between frames (overlap).
        axis: The axis to compute the zcr for. Default is 0.
        **kwargs: Additional keyword arguments passed to `rolling_zcr`.

    Returns:
        float: The mean of the rolling RMS of the input signal.
    """
    zcr_values = rolling_zcr(signal, frame_length, hop_length, **kwargs)
    if zcr_values.ndim == 1:
        return np.mean(zcr_values)
    else:
        return np.mean(zcr_values, axis=axis)


def zcr_min(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int,
    axis: int = 0,
    **kwargs: Any
) -> np.ndarray:
    """Computes the minimum of the rolling zero crossing rate of a signal.

    Args:
        signal: The input signal as a numpy.ndarray.
        frame_length: The length of the frame in samples.
        hop_length: The number of samples to advance between frames (overlap).
        axis: The axis to compute the zcr for. Default is 0.
        **kwargs: Additional keyword arguments passed to `rolling_zcr`.

    Returns:
        float: The minimum of the rolling RMS of the input signal.
    """
    zcr_values = rolling_zcr(signal, frame_length, hop_length, **kwargs)
    if zcr_values.ndim == 1:
        return np.min(zcr_values)
    else:
        return np.min(zcr_values, axis=axis)


def energy(
    array: np.ndarray,
    axis: int = 0
) -> np.ndarray:
    """Computes the energy of a signal:
    https://dsp.stackexchange.com/questions/3377/calculating-the-total-energy-of-a-signal

    Args:
        array: The input signal in 1D or 2D numpy.ndarray.
        axis: The axis along which to compute the energy. Defaults to 0.

    Returns:
        np.ndarray: The energy of the signal.
    """
    return np.sum(np.square(array), axis=axis)


def average_power(
    array: np.ndarray,
    axis: int = 0
) -> np.ndarray:
    """Computes the average power of a signal:
    https://dsp.stackexchange.com/questions/3377/calculating-the-total-energy-of-a-signal

    Args:
        array: The input signal in 1D or 2D numpy.ndarray.
        axis: The axis along which to compute the average power. Defaults to 0.

    Returns:
        np.ndarray: The average power of the signal.
    """
    return np.sum(np.square(array), axis=axis) / array.shape[axis]


def crest_factor(
    array: np.ndarray,
    axis: int = 0
) -> np.ndarray:
    """Computes the crest factor of the signal

    Args:
        array: The input signal as a numpy.ndarray.
        axis: The axis along which to compute the crest factor. Defaults to 0.

    Returns:
        np.ndarray: The crest factor of the signal.
    """
    peak = np.amax(np.absolute(array), axis=axis)
    rms = rms_value(array, axis=axis)
    return peak / rms


def envelope_energy_peak_detection(
    array: np.ndarray,
    fs: Union[int, float],
    start: int = 50,
    stop: int = 1000,
    freq_step: int = 50,
    fcl_add: int = 50,
    export: str = "array",
    axis: int = 0,
) -> Union[np.ndarray, dict]:
    """Computes the Envelope Energy Peak Detection of a signal within
    frequency bands.

    Args:
        array: The input time-domain signal.
        fs: The sampling frequency of the signal (Hz) as float or integer.
        start: The lower frequency bound of the first band (Hz). Default: 50.
        stop: The upper frequency bound of the last band (Hz). Default: 1000.
        freq_step: The width of each frequency band (Hz). Default: 50.
        fcl_add: Additional width added to the upper bound of each band.
                 Default: 50.
        export: The desired output format ("array" for NumPy array, "dict" for
                dictionary). Default: "array".

    Returns:
        Union[np.ndarray, dict]: The number of peaks detected in each
                                 frequency band.

    Raises:
        ValueError: If an unsupported export format is provided.
    """

    if axis == 1:
        _array = array.T
    else:
        _array = array

    f_nyq = fs / 2  # Nyquist frequency
    names = []
    n_peaks = []

    for fcl in range(start, stop, freq_step):  # Iterate over frequency bands
        names.append(f"EEPD{fcl}_{fcl + freq_step}")

        # Bandpass filtering
        fc = [fcl / f_nyq, (fcl + fcl_add) / f_nyq]
        b, a = butter(1, fc, btype="bandpass")
        bp_filter = filtfilt(b, a, _array)

        # Lowpass filtering for envelope energy
        b, a = butter(2, 10 / f_nyq, btype="lowpass")
        eed = filtfilt(b, a, bp_filter**2)
        eed /= np.max(eed + 1e-17)  # Normalize envelope energy

        peaks, _ = find_peaks(eed)  # Peak detection
        n_peaks.append(peaks.shape[0])

    if export == "array":
        return np.array(n_peaks)
    elif export == "dict":
        return dict(zip(names, n_peaks))
    else:
        raise ValueError(f"Unsupported export={export}")

# --- SPECTRAL ---

def dominant_frequency(
    array: np.ndarray,
    fs: int,
    axis: int = 0
) -> float:
    """Computes the dominant frequency of a signal.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal.
        axis: The axis along which to compute the dominant frequency. Defaults
            to 0.

    Returns:
        float: The dominant frequency of the signal.
    """


    if array.ndim == 1:
        nperseg = array.shape[0]
        freqs, psd = scipy.signal.welch(x=array, fs=fs, nperseg=nperseg)
        return freqs[np.argmax(psd)]
    else:
        nperseg = array.shape[0]
        freqs, psd = scipy.signal.welch(x=array, fs=fs, nperseg=nperseg, axis=axis)

        if axis == 0:
            return np.array([freqs[np.argmax(psd[:, i])] for i in range(psd.shape[1])])
        else:
            return np.array([freqs[np.argmax(psd[i, :])] for i in range(psd.shape[0])])


def mfcc_mean(
    y: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 20,
    axis: int = 0,
    **kwargs: Any
) -> np.ndarray:
    """Calculates the mean of each MFCC coefficient over time.

    Args:
        y: Audio time series.
        sr: Sampling rate of y. Default: 22050 Hz.
        n_mfcc: Number of MFCCs to return. Default: 20.
        axis: The axis along which to compute the mean. Defaults to 0.
        **kwargs: Additional keyword arguments passed to `mfcc`.

    Returns:
        np.ndarray: Mean MFCC values (n_mfcc,).
    """
    mfcc_features = mfcc(y, sr=sr, n_mfcc=n_mfcc, axis=axis, **kwargs)
    return np.mean(mfcc_features, axis=axis)


def signal_stats(
    arr: np.ndarray,
    name: str,
    axis: int = 0,
    fs: int = 44100,
    time_mode: str = "time"
) -> dict:
    """Computes the basic statistical information of signal.

    Args:
        arr: A 2D NumPy array.
        name: A string with the name of the input array.
        axis: The axis along which to compute the statistics. Defaults to 0.
        fs: The sampling frequency of the signal. Defaults to 44100.
        time_mode: The export format. Can be "time" or "samples". Defaults to
            "time".

    Returns:
        Dict: A dictionary containing the mean, max, min, and STD calculations
        of the signal.
    """

    return {
        f"{name}_max": max_value(arr, axis=axis),
        f"{name}_min": min_value(arr, axis=axis),
        f"{name}_mean": mean_value(arr, axis=axis),
        f"{name}_median": median_value(arr, axis=axis),
        f"{name}_std": std_value(arr, axis=axis),
        f"{name}_var": variance_value(arr, axis=axis),
        f"{name}_kurtosis": kurtosis_value(arr, axis=axis),
        f"{name}_skewness": sample_skewness(arr, axis=axis),
        f"{name}_rms": rms_value(arr, axis=axis),
        # f"{name}_zcr": zcr_value(arr),
        f"{name}_dominant_frequency": dominant_frequency(arr, fs, axis=axis),
        f"{name}_crest_factor": crest_factor(arr, axis=axis),
        f"{name}_signal_length": signal_length(arr, fs=fs, time_mode=time_mode, axis=axis),
    }
