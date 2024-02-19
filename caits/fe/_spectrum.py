import numpy as np
from scipy.signal import stft


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
        signal: The input signal np.ndarray.
        fs: The sampling frequency of the signal in float.
        window: Desired window to use. Default is 'hann'.
        nperseg (Optional): The length of each segment. Defaults to 256.
        noverlap (Optional): The number of points to overlap between segments.
            If None, `nperseg // 2` is used.
        nfft (Optional): The length of the FFT used, if a zero-padded FFT is
            desired. If None, it defaults to `nperseg`.
        fmin (Optional): The lowest frequency to include in the spectrogram
            (in Hz). If None, it defaults to 0.
        fmax (Optional): The highest frequency to include in the spectrogram
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
        signal,
        fs,
        window='hann',
        nperseg=256,
        noverlap=None,
        nfft=None,
        fmin: float = None,
        fmax: float = None
):
    """Computes the power spectrogram of a signal.

    Args:
        signal: array_like
            Input signal.
        fs: float
            Sampling frequency of the signal.
        window: str or tuple or array_like, optional
            Desired window to use. Default is 'hann'.
        nperseg: int, optional
            Length of each segment. Defaults to 256.
        noverlap: int, optional
            Number of points to overlap between segments. If None, `nperseg // 2` is used.
        nfft: int, optional
            Length of the FFT used, if a zero-padded FFT is desired. If None, it defaults to `nperseg`.
        fmin (Optional): The lowest frequency to include in the spectrogram
            (in Hz). If None, it defaults to 0.
        fmax (Optional): The highest frequency to include in the spectrogram
            (in Hz).  If None, it defaults to `fs / 2.0`.

    Returns:
        f: ndarray
            Array of sample frequencies.
        t: ndarray
            Array of segment times.
        Pxx: ndarray
            Power spectrogram of the signal.
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


def normal_to_power(
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
        ref:  Reference power level (in amplitude squared) for dB calculation.
            Defaults to 1.0.

    Returns:
        np.ndarray: The power spectrogram in dB.
    """
    return 10 * np.log10(power_spectrogram / ref)
