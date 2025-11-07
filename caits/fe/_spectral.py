from typing import Dict, List, Tuple, Union

import numpy as np
import scipy
from scipy.integrate import simpson


def spectral_centroid(
    array: np.ndarray,
    fs: Union[int, float],
) -> float:
    """Computes the spectral centroid of a signal.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal.

    Returns:
        float: The spectral centroid of the signal.
    """
    magnitudes, freqs, sum_mag = underlying_spectral(array, fs)

    return (magnitudes.T @ freqs) / sum_mag


def spectral_rolloff(
    array: np.ndarray,
    fs: Union[int, float],
    perc: float = 0.95,
    axis: int = 0
) -> float:
    """Computes the spectral rolloff of a signal, meaning the frequency below
    which a certain percentage of the total spectral energy, e.g. 85%, is
    contained.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal.
        perc: The percentage of the total spectral energy.

    Returns:
        float: The spectral rolloff of the signal.
    """
    magnitudes, _, sum_mag = underlying_spectral(array, fs, axis=axis)
    cumsum_mag = np.cumsum(magnitudes, axis=axis)

    if array.ndim == 1:
        return np.min(np.where(cumsum_mag >= perc * sum_mag)).astype(float)
    else:
        idxs = np.where(cumsum_mag >= perc * sum_mag)
        tmp = [[] for i in range(array.shape[1 if axis == 0 else 0])]
        for row, col in zip(*idxs):
            tmp[col].append(row)

        return np.array([min(l) for l in tmp]).astype(float)


def spectral_spread(
    array: np.ndarray,
    fs: Union[int, float],
    axis: int = 0
) -> float:
    """Computes the spectral spread of a signal, meaning the weighted
    standard deviation of frequencies wrt FFT value.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal.

    Returns:
        float: The spectral spread of the signal.
    """
    magnitudes, freqs, sum_mag = underlying_spectral(array, fs, axis=axis)
    spec_centroid = spectral_centroid(array, fs)

    if array.ndim == 1:
        return np.sqrt(np.sum(((freqs - spec_centroid) ** 2) * magnitudes, axis=axis)
                       / sum_mag)
    else:
        tmp = np.array([freqs - c for c in spec_centroid])
        return np.sqrt(np.sum((tmp ** 2).T * magnitudes, axis=axis)
                   / sum_mag)


def spectral_skewness(
    array: np.ndarray,
    fs: Union[int, float],
    axis: int = 0
) -> float:
    """Computes the spectral skewness of a signal, meaning the distribution
    of the spectrum around its mean.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal.

    Returns:
        float: The spectral skewness of the signal.
    """
    magnitudes, freqs, sum_mag = underlying_spectral(array, fs, axis=axis)
    spec_centroid = spectral_centroid(array, fs)
    spec_spread = spectral_spread(array, fs, axis=axis)

    if array.ndim == 1:
        return (np.sum(((freqs - spec_centroid) ** 3) * magnitudes) /
                ((spec_spread**3) * sum_mag))
    else:
        tmp = np.array([freqs - c for c in spec_centroid])
        return (np.sum((tmp ** 3).T * magnitudes, axis=axis) /
                ((spec_spread**3) * sum_mag))



def spectral_kurtosis(
    array: np.ndarray,
    fs: Union[int, float],
    axis: int = 0
) -> float:
    """Computes the spectral kurtosis of a signal, meaning the distribution
    of the spectrum around its mean.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal.

    Returns:
        float: The spectral kurtosis of the signal.
    """
    magnitudes, freqs, sum_mag = underlying_spectral(array, fs, axis=axis)
    spec_centroid = spectral_centroid(array, fs)
    spec_spread = spectral_spread(array, fs, axis=axis)

    if array.ndim == 1:
        return (np.sum(((freqs - spec_centroid) ** 4) * magnitudes)
                / ((spec_spread**4) * sum_mag))
    else:
        tmp = np.array([freqs - c for c in spec_centroid])
        return (np.sum((tmp ** 4).T * magnitudes, axis=axis)
                / ((spec_spread**4) * sum_mag))


def underlying_spectral(
    array: np.ndarray,
    fs: Union[int, float],
    axis: int = 0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Calculates the magnitudes and frequencies of the positive side of the
    Fourier Transform of a signal, along with the total sum of the magnitudes.

    Args:
        array (np.ndarray): The input signal.
        fs (int): Sampling frequency of the signal in Hz as an integer or
            float.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: A tuple containing:
            * magnitudes: Array of magnitudes of the positive frequencies.
            * freqs: Array of positive frequencies.
            * sum_mag: Sum of all magnitudes.

    Raises:
        TypeError: If the input array is not of type float32 or float64.
    """

    if array.dtype not in (np.float32, np.float64):
        raise TypeError("Input array must be of type np.float32 or np.float64")

    # Ensure correct data type for output arrays
    magnitudes = np.abs(np.fft.rfft(array, axis=axis)).astype(array.dtype)
    length = array.shape[axis]

    freqs = np.abs(np.fft.fftfreq(length, 1.0 / fs)[: length // 2 + 1]).astype(array.dtype)

    sum_mag = np.sum(magnitudes, axis=axis)
    return magnitudes, freqs, sum_mag


def spectral_bandwidth(
    array: np.ndarray,
    fs: Union[int, float],
    axis: int = 0
) -> float:
    """Calculates the spectral bandwidth of a given signal using its p
    ower spectrum.

    Args:
        array: 1-D numpy.ndarray containing the time-domain samples of the
            signal.
        fs: An integter that defines the sampling frequency of the signal in
            Hz.

    Returns:
        float: Spectral bandwidth of the signal.

    Note:
        The spectral bandwidth is calculated based on the standard deviation
        of the power spectrum, providing a measure of its spread in the
        frequency domain.

    Example:
        >>> fs = 1000  # Sampling frequency in Hz
        >>> t = np.arange(0, 1, 1/fs)  # Time vector
        >>> signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
        >>> bw = spectral_bandwidth(signal, fs)
        >>> print(f"Spectral Bandwidth: {bw} Hz")
    """
    # Compute the FFT
    fft_result = np.fft.fft(array, axis=axis)

    # Calculate the two-sided power spectrum
    power_spectrum = np.abs(fft_result) ** 2

    # Consider only the positive frequencies (single-sided spectrum)
    if array.ndim == 1:
        if array.size % 2 == 0:
            power_spectrum = power_spectrum[: array.size // 2] * 2
        else:
            power_spectrum = power_spectrum[: (array.size - 1) // 2] * 2
    else:
        if array.shape[axis] % 2 == 0:
            if axis == 0:
                power_spectrum = power_spectrum[:array.shape[axis] // 2, :] * 2
            else:
                power_spectrum = power_spectrum[:, :array.shape[axis] // 2] * 2
        else:
            if axis == 0:
                power_spectrum = power_spectrum[:(array.shape[axis] - 1) // 2, :] * 2
            else:
                power_spectrum = power_spectrum[:, :(array.shape[axis] - 1) // 2] * 2


    # Normalize power spectrum
    power_spectrum /= np.sum(power_spectrum, axis=axis)

    # Frequency vector
    freqs = np.fft.fftfreq(array.shape[axis], d=1 / fs)[: array.shape[axis] // 2]

    print(freqs.shape, power_spectrum.shape)


    if array.ndim == 1:
        # Mean frequency (center of gravity)
        mean_freq = np.sum(freqs * power_spectrum, axis=axis)

        # Spectral bandwidth (standard deviation)
        spectral_bw = np.sqrt(np.sum(((freqs - mean_freq) ** 2) * power_spectrum))
    else:
        if axis == 0:
            tmp = np.tile(freqs[..., np.newaxis], (1, array.shape[1]))
        else:
            tmp = np.tile(freqs[np.newaxis, ...], (array.shape[0], 1))

        mean_freq = np.sum(tmp * power_spectrum, axis=axis)
        spectral_bw = np.sqrt(np.sum(((tmp - mean_freq) ** 2) * power_spectrum, axis=axis))

    return spectral_bw


def spectral_flatness(
    array: np.ndarray,
    fs: Union[int, float],
    nperseg_th: int = 900,
    noverlap_th: int = 600,
    axis: int = 0
) -> float:
    """Calculates the spectral flatness of a signal.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal as an integer or float.
        nperseg_th: The theoretical length of each segment for the Welch
            method.
        noverlap_th: The theoretical number of points to overlap between
            segments.

    Returns:
        float: The spectral flatness of the signal.
    """
    nperseg = min(nperseg_th, array.shape[axis])
    noverlap = min(noverlap_th, int(nperseg / 2))
    freqs, psd = scipy.signal.welch(array, fs, nperseg=nperseg, noverlap=noverlap, axis=axis)
    psd_len = psd.shape[axis]
    gmean = np.exp((1 / psd_len) * np.sum(np.log(psd + 1e-17), axis=axis))
    amean = (1 / psd_len) * np.sum(psd, axis=axis)

    return gmean / amean


def spectral_std(
    array: np.ndarray,
    fs: Union[int, float],
    nperseg_th: int = 900,
    noverlap_th: int = 600,
    ddof: int = 1,
    axis: int = 0
) -> float:
    """Calculates the standard deviation of the power spectral density (PSD)

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal as an integer or float.
        nperseg_th: The theoretical length of each segment for the Welch
            method.
        noverlap_th: The theoretical number of points to overlap between
            segments.

    Returns:
        float: The standard deviation of the PSD of the signal.
    """
    nperseg = min(nperseg_th, array.shape[axis])
    noverlap = min(noverlap_th, int(nperseg / 2))
    _, psd = scipy.signal.welch(array, fs, nperseg=nperseg, noverlap=noverlap, axis=axis)

    return np.std(psd, ddof=ddof, axis=axis)


def spectral_slope(
    array: np.ndarray,
    fs: Union[int, float],
    b1_th: int = 0,
    b2_th: int = 8000,
    axis: int = 0
) -> float:
    """Calculates the spectral slope of a signal.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal as an integer or float.
        b1_th: The lower bound of the frequency range.
        b2_th: The upper bound of the frequency range.

    Returns:
        float: The spectral slope of the signal.
    """
    b1 = b1_th
    b2 = b2_th

    s = np.absolute(np.fft.fft(array, axis=axis))
    if array.ndim == 1:
        s = s[: s.shape[axis] // 2]
    else:
        if axis == 0:
            s = s[:s.shape[0] // 2, :]
        else:
            s = s[:, :s.shape[1] // 2]
    muS = np.mean(s, axis=axis)
    f = np.linspace(0, fs / 2, s.shape[axis])
    muF = np.mean(f)

    bidx = np.where(np.logical_and(b1 <= f, f <= b2))
    # print(f.shape, muF.shape, s.shape, muS.shape, len(bidx))

    if array.ndim == 1:
        slope = np.sum(((f - muF) * (s - muS))[bidx]) / np.sum((f[bidx] - muF) ** 2)
    else:
        if axis == 0:
            tmp1 = f - muF
            tmp1 = np.tile(tmp1[:, np.newaxis], (1, s.shape[1]))
            tmp2 = s - np.tile(muS[np.newaxis, ...], (s.shape[0], 1))
            tmp = tmp1 * tmp2
            tmp = tmp[bidx[0], :]
        else:
            tmp1 = f - muF
            tmp1 = np.tile(tmp1[:, np.newaxis], (1, s.shape[0]))
            tmp2 = s - np.tile(muS[np.newaxis, ...], (s.shape[1], 1))
            tmp = tmp1 * tmp2
            tmp = tmp[:, bidx[0]]

        slope = np.sum(tmp, axis=axis) / np.sum((f[bidx] - muF) ** 2, axis=axis)

    return slope


def spectral_decrease(
    array: np.ndarray,
    fs: Union[int, float],
    b1_th: int = 0,
    b2_th: int = 8000,
    axis: int = 0
) -> float:
    """Calculates the spectral decrease of a signal.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal as an integer or float.
        b1_th: The lower bound of the frequency range.
        b2_th: The upper bound of the frequency range.

    Returns:
        float: The spectral decrease of the signal.
    """
    b1 = b1_th
    b2 = b2_th

    s = np.absolute(np.fft.fft(array, axis=axis))
    if array.ndim == 1:
        s = s[: s.shape[0] // 2]
    else:
        if axis == 0:
            s = s[:s.shape[0] // 2, :]
        else:
            s = s[:, :s.shape[1] // 2]

    f = np.linspace(0, fs / 2, s.shape[axis])

    bidx = np.where(np.logical_and(b1 <= f, f <= b2))[0]
    k = bidx[1:]

    if array.ndim == 1:
        sb1 = s[bidx[0]]
        decrease = np.sum((s[k] - sb1) / (f[k] - 1 + 1e-17)) / (np.sum(s[k]) + 1e-17)
    else:
        if axis == 0:
            sb1 = s[bidx[0], :]
            tmp1 = s[k, :]
            tmp2 = np.tile(f[k][..., np.newaxis], (1, s.shape[1]))

        else:
            sb1 = s[:, bidx[0]]
            tmp1 = s[:, k]
            tmp2 = np.tile(f[k][np.newaxis, ...], (s.shape[0], 1))

        decrease = np.sum((tmp1 - sb1) / (tmp2 - 1 + 1e-17), axis=axis) / (np.sum(tmp1, axis=axis) + 1e-17)


    return decrease


def power_spectral_density(
    array: np.ndarray,
    fs: Union[int, float],
    nperseg_th: int = 900,
    noverlap_th: int = 600,
    freq_cuts: List[Tuple[int, int]] = [
        (0, 200),
        (300, 425),
        (500, 650),
        (950, 1150),
        (1400, 1800),
        (2300, 2400),
        (2850, 2950),
        (3800, 3900),
    ],
    axis: int = 0,
    export: str = "array",
) -> Union[np.ndarray, Dict[str, float]]:
    """Calculates the power spectral density (PSD) of a signal and returns
    the relative power in specific frequency bands.

    Args:
        array: The input time-domain signal.
        fs: The sampling frequency of the signal (Hz).
        nperseg_th: The theoretical length of each segment for the Welch
                    method. Default: 900.
        noverlap_th: The theoretical number of points to overlap between
                     segments. Default: 600.
        freq_cuts: A list of tuples defining the frequency bands of interest.
                   Default: Predefined bands.
        axis: Axis along which the periodogram is computed; the default is
            over the first axis (i.e. ``axis=0``).
        export: The desired output format ("array" for NumPy array, "dict" for
                      dictionary). Default: "array".

    Returns:
        Union[np.ndarray, dict[str, float]]: The relative power in each
            frequency band.
            - If `export` is "array", returns a NumPy array of relative powers.
            - If `export` is "dict", returns a dictionary mapping band names
                to their relative powers.

    Raises:
        ValueError: If an unsupported export format is provided.
    """
    nperseg = min(nperseg_th, len(array))
    noverlap = min(noverlap_th, nperseg / 2)

    freqs, psd = scipy.signal.welch(
        array, fs, nperseg=nperseg, noverlap=noverlap, axis=axis
    )
    dx_freq = freqs[1] - freqs[0]
    total_power = simpson(psd, dx=dx_freq)

    band_powers = []
    for lf, hf in freq_cuts:
        idx_band = np.logical_and(freqs >= lf, freqs <= hf)
        band_power = simpson(psd[idx_band], dx=dx_freq)
        band_powers.append(band_power / total_power)

    # Handle export format
    if export == "array":
        return np.array(band_powers)
    elif export == "dict":
        feat_names = [f"PSD_{lf}-{hf}" for lf, hf in freq_cuts]
        return dict(zip(feat_names, band_powers))
    else:
        raise ValueError(f"Unsupported export={export}")


def spectral_values(
    array: np.ndarray,
    fs: Union[int, float],
    perc: float = 0.95,
    nperseg_th: int = 900,
    noverlap_th: int = 600,
    b1_th: int = 0,
    b2_th: int = 8000,
    axis: int = 0
) -> Dict[str, float]:
    """Computes the underlying spectral values of a signal.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal.
        perc: The percentage of the total spectral energy.
        nperseg_th: The theoretical length of each segment for the Welch
                    method. Default: 900.
        noverlap_th: The theoretical number of points to overlap between
                     segments. Default: 600.
        b1_th: The lower bound of the frequency range.
        b2_th: The upper bound of the frequency range.

    Returns:
        dict: A dictionary containing the spectral centroid, spectral rolloff,
        spectral spread, spectral skewness, spectral kurtosis, and spectral
        bandwidth of the signal.
    """
    return {
        "spectral_centroid": spectral_centroid(array, fs),
        "spectral_rolloff": spectral_rolloff(array, fs, perc, axis=axis),
        "spectral_spread": spectral_spread(array, fs, axis=axis),
        "spectral_skewness": spectral_skewness(array, fs, axis=axis),
        "spectral_kurtosis": spectral_kurtosis(array, fs, axis=axis),
        "spectral_bandwidth": spectral_bandwidth(array, fs, axis=axis),
        "spectral_flatness": spectral_flatness(array, fs, nperseg_th,
                                               noverlap_th, axis=axis),
        "spectral_std": spectral_std(array, fs, nperseg_th, noverlap_th, axis=axis),
        "spectral_slope": spectral_slope(array, fs, b1_th, b2_th, axis=axis),
        "spectral_decrease": spectral_decrease(array, fs, b1_th, b2_th, axis=axis),
    }
