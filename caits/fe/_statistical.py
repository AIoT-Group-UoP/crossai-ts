import numpy as np
import scipy
from scipy.stats import kurtosis, moment, skew
from scipy.signal import butter, filtfilt, find_peaks
from typing import Union


def std_value(
        array: np.ndarray,
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
    return np.std(array, axis=axis)


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


def kurtosis_value(array: np.ndarray) -> float:
    """Computes the kurtosis of an audio signal.

    Args:
        array: The input signal as a numpy.ndarray.

    Returns:
        float: The kurtosis of the audio signal.
    """
    return kurtosis(array)


def sample_skewness(array) -> float:
    """
    Calculate the sample skewness of an array using scipy.

    Args:
        array (numpy.ndarray): Input array.

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

    return skew(array, bias=False)


def rms_value(array: np.ndarray) -> float:
    """Computes the RMS Power value of a signal.

    Args:
        array: The input signal as a numpy.ndarray.

    Returns:
        float: The RMS Power of the signal.
    """
    return np.sqrt(np.mean(np.square(array)))


def zcr_value(array: np.ndarray) -> float:
    """Computes the zero crossing rate of a signal.

    Args:
        array: The input signal as a numpy.ndarray.

    Returns:
        float: The zero crossing rate of the signal.
    """
    return np.sum(np.multiply(array[0:-1], array[1:]) < 0) / (len(array) - 1)


def dominant_frequency(
    array: np.ndarray,
    fs: int
) -> float:
    """Computes the dominant frequency of a signal.

    Args:
        array: The input signal as a numpy.ndarray.

    Returns:
        float: The dominant frequency of the signal.
    """

    nperseg = array.shape[0]
    freqs, psd = scipy.signal.welch(x=array, fs=fs, nperseg=nperseg)

    return freqs[np.argmax(psd)]


def central_moments(
    array: np.ndarray,
    export: str = "array"
) -> Union[np.ndarray, dict]:
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

    moment0 = moment(array, moment=0)
    moment1 = moment(array, moment=1)
    moment2 = moment(array, moment=2)
    moment3 = moment(array, moment=3)
    moment4 = moment(array, moment=4)
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


def signal_length(
        array: np.ndarray,
        fs: int
) -> float:
    """Computes the length of a signal in seconds.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal.

    Returns:
        float: The length of the signal in seconds.
    """
    return len(array) / fs


def crest_factor(
    array: np.ndarray
) -> float:
    """Computes the crest factor of the signal

    Args:
        array: The input signal as a numpy.ndarray.

    Returns:
        float: The crest factor of the signal.
    """

    peak = np.amax(np.absolute(array))
    rms = rms_value(array)

    return peak / rms


def envelope_energy_peak_detection(
    array: np.ndarray,
    fs: int,
    start: int = 50,
    stop: int = 1000,
    freq_step: int = 50,
    fcl_add: int = 50,
    export: str = "array"
) -> Union[np.ndarray, dict]:
    """Computes the Envelope Energy Peak Detection of a signal.

    Args:
        array:
        fs:
        start:
        stop:
        freq_step:
        fcl_add:
        export:

    Returns:

    """
    names = []

    f_nyq = fs/2
    n_peaks = []
    for fcl in range(start, stop, freq_step):
        names = names + ['EEPD'+str(fcl)+'_'+str(fcl+freq_step)]
        fc = [fcl/f_nyq, (fcl + fcl_add)/f_nyq]
        b, a = butter(1, fc, btype='bandpass')
        bp_filter = filtfilt(b, a, array)
        b, a = butter(2, 10/f_nyq, btype='lowpass')
        eed = filtfilt(b, a, bp_filter**2)
        eed = eed/np.max(eed+1e-17)
        peaks,_ = find_peaks(eed)
        n_peaks.append(peaks.shape[0])
    if export == "array":
        return np.array(n_peaks)
    elif export == "dict":
        return dict(zip(names, n_peaks))
    else:
        raise ValueError(f"Unsupported export={export}")


def signal_stats(
        arr: np.ndarray,
        name: str,
        axis: int = 0,
        fs: int = 44100
) -> dict:
    """Computes the basic statistical information of signal.

    Args:
        arr: A 2D NumPy array.
        name: A string with the name of the input array.
        axis: The axis along which to compute the statistics. Defaults to 0.
        fs: The sampling frequency of the signal. Defaults to 44100.

    Returns:
        Dict: A dictionary containing the mean, max, min, and STD calculations
        of the signal.
    """

    return {

        f"{name}_max": np.max(arr, axis=axis),
        f"{name}_min": np.min(arr, axis=axis),
        f"{name}_mean": np.mean(arr, axis=axis),
        f"{name}_median": np.median(arr, axis=axis),
        f"{name}_std": np.std(arr, axis=axis),
        f"{name}_var": np.var(arr, axis=axis),
        f"{name}_kurtosis": kurtosis_value(arr),
        f"{name}_skewness": sample_skewness(arr),
        f"{name}_rms": rms_value(arr),
        f"{name}_zcr": zcr_value(arr),
        f"{name}_dominant_frequency": dominant_frequency(arr),
        f"{name}_crest_factor": crest_factor(arr),
        f"{name}_signal_length": signal_length(arr, fs=fs)
    }
