import numpy as np
import scipy.signal
from scipy.signal import stft
from scipy.signal import get_window


def compute_spectrogram(
        signal: np.ndarray,
        fs: int,
        window: str = "hann",
        nperseg: int = 256,
        noverlap: int = None,
        nfft: int =None,
        fmin: float = None,
        fmax: float = None
):
    """Computes the spectrogram of a signal.

    Args:
        signal: The input signal in np.ndarray.
        fs: Integer with the sampling frequency of the signal in float.
        window: String value containing the desired window to use.
            Default is 'hann'.
        nperseg: Integer with the length of each segment. Defaults to 256.
        noverlap: Integer with the number of points to overlap between
            segments. If None, `nperseg // 2` is used.
        nfft: Integer with the length of the FFT used, if a zero-padded FFT is
            desired. If None, it defaults to `nperseg`.
        fmin: Float with the lowest frequency to include in the spectrogram
            (in Hz). If None, it defaults to 0.
        fmax: Float with the highest frequency to include in the spectrogram
            (in Hz).  If None, it defaults to `fs / 2.0`.

    Returns:
        f: np.ndarray of sample frequencies.
        t: np.ndarray of segment times.
        spec: 2D np.ndarray Spectrogram of the `signal`.
    """
    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = fs / 2.0

    f, t, spec = stft(signal, fs=fs, window=window, nperseg=nperseg,
                      noverlap=noverlap, nfft=nfft, boundary=None)
    freq_mask = (f >= fmin) & (f <= fmax)
    return f[freq_mask], t, np.abs(spec[freq_mask, :])


def compute_power_spectrogram(
        signal: np.ndarray,
        fs: int,
        window: str = 'hann',
        nperseg: int = 256,
        noverlap: int = None,
        nfft: int = None,
        fmin: float = None,
        fmax: float = None
):
    """Computes the power spectrogram of a signal.

    Args:
        signal: np.ndarray of shape (n_samples,) with the input signal.
        fs: The sampling frequency of the signal as integer.
        window: String value containing the desired window to use. Default
            is 'hann'.
        nperseg: Integer with the length of each segment. Defaults to 256.
        noverlap: Integer with the number of points to overlap between If
            None, `nperseg // 2` is used.
        nfft: Integer with the length of the FFT used, if a zero-padded FFT is
            desired. If None, it defaults to `nperseg`.
        fmin: Float with the lowest frequency to include in the spectrogram
            (in Hz). If None, it defaults to 0.
        fmax: Float with the highest frequency to include in the spectrogram
            (in Hz).  If None, it defaults to `fs / 2.0`.

    Returns:
        f: np.ndarray of sample frequencies.
        t: np.ndarray of segment times.
        Pxx: np.ndarray of the power spectrogram of the signal.
    """
    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = fs / 2.0

    f, t, Zxx = stft(signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    freq_mask = (f >= fmin) & (f <= fmax)
    f = f[freq_mask]
    spec = np.abs(Zxx[freq_mask, :])
    Pxx = np.abs(spec)**2
    return f, t, Pxx


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


def compute_mel_spectrogram(
        signal: np.ndarray,
        sr: int,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = None,
        fmax: float = None
) -> np.ndarray:
    """Computes the Mel spectrogram of a signal.

    Args:
        signal: np.ndarray of shape (n_samples,) with the input signal.
        sr: Integer indicating the sampling rate of the signal.
        n_fft: Integer with the length of the FFT window. Defaults to 2048.
        hop_length: Integer with the number of samples between successive
            frames. Defaults to 512.
        n_mels: Integer with the number of Mel bands to generate. Defaults
            to 128.
        fmin: Float to indicate the lowest frequency to include in the
            spectrogram (in Hz). If None, it defaults to 0.
        fmax: Float to indicate the highest frequency to include in the
            spectrogram (in Hz). If None, it defaults to `sr / 2.0`.

    Returns:
        mel_spectrogram: np.ndarray with the mel spectrogram of the signal.
    """
    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = sr / 2.0

    # Compute power spectrogram
    window = get_window('hann', n_fft)
    _, _, Sxx = scipy.signal.stft(signal, fs=sr, window=window, nperseg=n_fft,
                                  noverlap=n_fft - hop_length)
    power_spectrogram = np.abs(Sxx) ** 2

    # Compute Mel filterbanks
    mel_filters = _compute_mel_filterbanks(sr, n_fft, n_mels, fmin, fmax)

    # Apply Mel filterbanks to power spectrogram
    mel_spectrogram = np.dot(mel_filters, power_spectrogram)

    return mel_spectrogram


def _compute_mel_filterbanks(
        sr: int,
        n_fft: int,
        n_mels: int,
        fmin: float,
        fmax: float
) -> np.ndarray:
    """Computes Mel filterbanks.

    Args:
        sr: Integer with the sampling rate of the signal.
        n_fft: Integer with the length of the FFT window.
        n_mels: Integer with the number of Mel bands to generate.
        fmin: Float with the lowest frequency to include in the spectrogram
            (in Hz).
        fmax: Float with the highest frequency to include in the spectrogram
            (in Hz).

    Returns:
        mel_filters: np.ndarray with the Mel filterbanks.
    """
    mel_min = 0 if fmin == 0 else hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(1, n_mels + 1):
        filters[i - 1, bin_points[i - 1]:bin_points[i]] = (
                (np.arange(bin_points[i - 1], bin_points[i]) - bin_points[
                    i - 1]) /
                (bin_points[i] - bin_points[i - 1]))
        filters[i - 1, bin_points[i]:bin_points[i + 1]] = (
                1 - (np.arange(bin_points[i], bin_points[i + 1]) - bin_points[
            i]) /
                (bin_points[i + 1] - bin_points[i]))

    return filters


def hz_to_mel(freq: float) -> float:
    """Converts Hz to Mel scale.

    Args:
        freq: Float with the frequency value in Hz.

    Returns:
        mel: Float with the frequency value in Mel scale.
    """
    return 2595 * np.log10(1 + freq / 700)


def mel_to_hz(mel: float) -> float:
    """Converts Mel scale to Hz.

    Args:
        mel: Float with the frequency value in Mel scale.

    Returns:
        freq: Float with the frequency value in Hz.
    """
    return 700 * (10 ** (mel / 2595) - 1)
