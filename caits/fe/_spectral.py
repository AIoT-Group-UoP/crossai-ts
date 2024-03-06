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


def spectral_bandwidth(
        array: np.ndarray,
        fs: int,
        p: int = 2
) -> float:
    """Computes the spectral bandwith of a signal, meaning the width of the
    spectrum.

    Args:
        array: The input signal as a numpy.ndarray.
        fs: The sampling frequency of the signal.
        p: The exponent of the Minkowski distance.

    Returns:
        float: The spectral bandwith of the signal.
    """
    magnitudes, freqs, _ = underlying_spectral(array, fs)
    spec_centroid = spectral_centroid(array, fs)

    return (np.sum(magnitudes*(freqs - spec_centroid)**p))**(1/p)


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
    fs: int
) -> float:

    b1 = 0
    b2 = 8000

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
    fs: int
) -> float:

    b1 = 0
    b2 = 8000

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
        "spectral_bandwidth": spectral_bandwidth(array, fs, p),
        "spectral_flatness": spectral_flatness(array, fs),
        "spectral_std": spectral_std(array, fs),
        "spectral_slope": spectral_slope(array, fs),
        "spectral_decrease": spectral_decrease(array, fs)
    }


