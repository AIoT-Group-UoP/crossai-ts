import numpy as np
import scipy


def spectral_centroid(array: np.ndarray, fs: int) -> float:
    """Computes the spectral centroid of a signal.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal.

    Returns:
        float: The spectral centroid of the signal.
    """
    magnitudes, freqs, sum_mag = underlying_spectral(array, fs)

    return np.sum(magnitudes * freqs) / sum_mag


def spectral_rolloff(array: np.ndarray, fs: int, perc: 0.95) -> float:
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
    magnitudes, _, sum_mag = underlying_spectral(array, fs)
    cumsum_mag = np.cumsum(magnitudes)
    return np.min(np.where(cumsum_mag >= perc * sum_mag)[0])


def spectral_spread(array: np.ndarray, fs: int) -> float:
    """Computes the spectral spread of a signal, meaning the weighted
    standard deviation of frequencies wrt FFT value.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal.

    Returns:
        float: The spectral spread of the signal.
    """
    magnitudes, freqs, sum_mag = underlying_spectral(array, fs)
    spec_centroid = spectral_centroid(array, fs)

    return np.sqrt(np.sum(((freqs - spec_centroid) ** 2) * magnitudes) /
                   sum_mag)


def spectral_skewness(array: np.ndarray, fs: int) -> float:
    """Computes the spectral skewness of a signal, meaning the distribution
    of the spectrum around its mean.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal.

    Returns:
        float: The spectral skewness of the signal.
    """
    magnitudes, freqs, sum_mag = underlying_spectral(array, fs)
    spec_centroid = spectral_centroid(array, fs)
    spec_spread = spectral_spread(array, fs)

    return (np.sum(((freqs - spec_centroid)**3) * magnitudes) /
            ((spec_spread**3) * sum_mag))


def spectral_kurtosis(array: np.ndarray, fs: int) -> float:
    """Computes the spectral kurtosis of a signal, meaning the distribution
    of the spectrum around its mean.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal.

    Returns:
        float: The spectral kurtosis of the signal.
    """
    magnitudes, freqs, sum_mag = underlying_spectral(array, fs)
    spec_centroid = spectral_centroid(array, fs)
    spec_spread = spectral_spread(array, fs)

    return (np.sum(((freqs - spec_centroid)**4) * magnitudes) /
            ((spec_spread**4) * sum_mag))


def underlying_spectral(
    array: np.ndarray,
    fs: int
) -> tuple[np.ndarray, np.ndarray, float]:

    magnitudes = np.abs(
        np.fft.rfft(array))  # magnitudes of positive frequencies
    length = len(array)
    freqs = np.abs(np.fft.fftfreq(length, 1.0 / fs)[
                   :length // 2 + 1])  # positive frequencies

    sum_mag = np.sum(magnitudes)

    return magnitudes, freqs, sum_mag


def spectral_bandwidth(
        array: np.ndarray,
        fs: int
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
    fft_result = np.fft.fft(array)
    
    # Calculate the two-sided power spectrum
    power_spectrum = np.abs(fft_result) ** 2
    
    # Consider only the positive frequencies (single-sided spectrum)
    if array.size % 2 == 0:
        power_spectrum = power_spectrum[:array.size // 2] * 2
    else:
        power_spectrum = power_spectrum[:(array.size - 1) // 2] * 2
    
    # Normalize power spectrum
    power_spectrum /= np.sum(power_spectrum)
    
    # Frequency vector
    freqs = np.fft.fftfreq(array.size, d=1/fs)[:array.size // 2]
    
    # Mean frequency (center of gravity)
    mean_freq = np.sum(freqs * power_spectrum)
    
    # Spectral bandwidth (standard deviation)
    spectral_bw = np.sqrt(np.sum(((freqs - mean_freq) ** 2) * power_spectrum))
    
    return spectral_bw


def spectral_flatness(
    array: np.ndarray,
    fs: int,
    nperseg_th: int = 900,
    noverlap_th: int = 600
) -> float:

    nperseg = min(nperseg_th, len(array))
    noverlap = min(noverlap_th, int(nperseg/2))
    freqs, psd = scipy.signal.welch(array, fs, nperseg=nperseg,
                                    noverlap=noverlap)
    psd_len = len(psd)
    gmean = np.exp((1/psd_len)*np.sum(np.log(psd + 1e-17)))
    amean = (1/psd_len)*np.sum(psd)

    return gmean/amean


def spectral_std(
    array: np.ndarray,
    fs: int,
    nperseg_th: int = 900,
    noverlap_th: int = 600
) -> float:

    nperseg = min(nperseg_th, len(array))
    noverlap = min(noverlap_th, int(nperseg / 2))
    _, psd = scipy.signal.welch(array, fs, nperseg=nperseg, noverlap=noverlap)

    return np.std(psd)


def spectral_slope(
    array: np.ndarray,
    fs: int,
    b1_th: int = 0,
    b2_th: int = 8000
) -> float:

    b1 = b1_th
    b2 = b2_th

    s = np.absolute(np.fft.fft(array))
    s = s[:s.shape[0] // 2]
    muS = np.mean(s)
    f = np.linspace(0, fs / 2, s.shape[0])
    muF = np.mean(f)

    bidx = np.where(np.logical_and(b1 <= f, f <= b2))
    slope = np.sum(((f - muF) * (s - muS))[bidx]) / np.sum(
        (f[bidx] - muF) ** 2)

    return slope


def spectral_decrease(
    array: np.ndarray,
    fs: int,
    b1_th: int = 0,
    b2_th: int = 8000
) -> float:

    b1 = b1_th
    b2 = b2_th

    s = np.absolute(np.fft.fft(array))
    s = s[:s.shape[0] // 2]
    f = np.linspace(0, fs / 2, s.shape[0])

    bidx = np.where(np.logical_and(b1 <= f, f <= b2))

    k = bidx[0][1:]
    sb1 = s[bidx[0][0]]
    decrease = np.sum((s[k] - sb1) / (f[k] - 1 + 1e-17)) / (
                np.sum(s[k]) + 1e-17)

    return decrease


def power_spectral_density(
    array: np.ndarray,
    fs: int,
    nperseg_th: int = 900,
    noverlap_th: int = 600,
    freq_cuts: list[tuple[int, int]] = [(0,200),(300,425),(500,650),(950,1150),
                                        (1400,1800),(2300,2400),(2850,2950),
                                        (3800,3900)]
) -> dict[str, float]:
    from scipy.integrate import simps

    feat = []
    nperseg = min(nperseg_th, len(array))
    noverlap=min(noverlap_th, int(nperseg/2))
    freqs, psd = scipy.signal.welch(array, fs, nperseg=nperseg,
                                    noverlap=noverlap)
    dx_freq = freqs[1]-freqs[0]
    total_power = simps(psd, dx=dx_freq)
    for lf, hf in freq_cuts:
        idx_band = np.logical_and(freqs >= lf, freqs <= hf)
        band_power = simps(psd[idx_band], dx=dx_freq)
        feat.append(band_power/total_power)
    feat = np.array(feat)
    feat_names = [f'PSD_{lf}-{hf}' for lf, hf in freq_cuts]

    return dict(zip(feat_names, feat))


def zcr_mean(
    array: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    padding_mode: str = "edge"
) -> float:
    """Calculates the mean zero-crossing rate of a signal based on the rolling
    zero-crossing rate.

    Args:
        array: The input signal as a numpy.ndarray.
        frame_length: The length of the frame.
        hop_length: The number of samples to advance between frames (overlap).
        center: If True, the signal is padded on both sides to center the
            frames.
        padding_mode: A string with the padding mode to use when padding the
            signal. Defaults to "edge". Check numpy.pad for more
            information about the relevant padding modes.
            https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    Returns:
        float: The mean zero-crossing rate of the input signal.
    """
    from caits.properties import rolling_zcr

    return rolling_zcr(array, frame_length, hop_length, center,
                       padding_mode).mean()


def spectral_values(
        array: np.ndarray,
        fs: int,
        perc: float = 0.95,
        p: int = 2
) -> dict:
    """Computes the underlying spectral values of a signal.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal.
        perc: The percentage of the total spectral energy.
        p: The exponent of the Minkowski distance.

    Returns:
        dict: A dictionary containing the spectral centroid, spectral rolloff,
        spectral spread, spectral skewness, spectral kurtosis, and spectral
        bandwidth of the signal.
    """
    return {
        "spectral_centroid": spectral_centroid(array, fs),
        "spectral_rolloff": spectral_rolloff(array, fs, perc),
        "spectral_spread": spectral_spread(array, fs),
        "spectral_skewness": spectral_skewness(array, fs),
        "spectral_kurtosis": spectral_kurtosis(array, fs),
        "spectral_bandwidth": spectral_bandwidth(array, fs),
        "spectral_flatness": spectral_flatness(array, fs),
        "spectral_std": spectral_std(array, fs),
        "spectral_slope": spectral_slope(array, fs),
        "spectral_decrease": spectral_decrease(array, fs)
    }
