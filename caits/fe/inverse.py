from typing import Any, Optional
import numpy as np
from numpy.typing import DTypeLike
from ..base._typing_base import _WindowSpec, _PadModeSTFT
from ..base._utility import nnls
from ..fe._mel_lib import mel_filter
from ..fe._spectrum_lib import griffinlim


def mel_to_stft(
    M: np.ndarray,
    *,
    sr: float = 22050,
    n_fft: int = 2048,
    power: float = 2.0,
    **kwargs: Any,
) -> np.ndarray:
    """Approximate STFT magnitude from a Mel power spectrogram.

    Args:
        M: The spectrogram as produced by `fe.melspectrogram`.
        sr: The sampling rate of the underlying signal.
        n_fft: The number of FFT components in the resulting STFT.
        power: Exponent for the magnitude melspectrogram.
        **kwargs: Additional keyword arguments for Mel filter bank parameters
        fmin: Lowest frequency (in Hz)
        fmax: Highest frequency (in Hz). If `None`, use ``fmax = sr / 2.0``
        htk: Use HTK formula instead of Slaney
        norm: If 'slaney', divide the triangular mel weights by the width of
            the mel band (area normalization).
            If numeric, use `librosa.util.normalize` to normalize each filter
            by to unit l_p norm. See `librosa.util.normalize` for a full
            description of supported norm values (including `+-np.inf`).
            Otherwise, leave all linear_spectrogram = mel_to_stft(
    feat_extr_2d_dataset.X[0].values, sr=SAMPLE_RATE, n_fft=2048, power=2
)

y = griffinlim(
    linear_spectrogram, n_fft=2048, hop_length=512, win_length=None,
    window='hann',
    momentum=.99,
    center=True,
    pad_mode='reflect',
    n_iter=32
)

ipd.Audio(y, rate=SAMPLE_RATE)the triangles aiming for a peak value of 1.0
        dtype : np.dtype
            The data type of the output basis.
            By default, uses 32-bit (single-precision) floating point.

    Returns
        S: An approximate linear magnitude spectrogram
    """
    # Construct a mel basis with dtype matching the input data
    mel_basis = mel_filter(
        sr=sr, n_fft=n_fft, n_mels=M.shape[-2], dtype=M.dtype, **kwargs
    )

    # Find the non-negative least squares solution, and apply
    # the inverse exponent.
    # We'll do the exponentiation in-place.
    inverse = nnls(mel_basis, M)
    return np.power(inverse, 1.0 / power, out=inverse)


def mel_to_audio(
    M: np.ndarray,
    *,
    sr: float = 22050,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
    power: float = 2.0,
    n_iter: int = 32,
    length: Optional[int] = None,
    dtype: DTypeLike = np.float32,
    **kwargs: Any,
) -> np.ndarray:
    """Invert a mel power spectrogram to audio using Griffin-Lim.

    Args:
        M: np.ndarray [shape=(..., n_mels, n), non-negative]
            The spectrogram as produced by `feature.melspectrogram`
        sr: number > 0 [scalar]
            sampling rate of the underlying signal
        n_fft: int > 0 [scalar]
            number of FFT components in the resulting STFT
        hop_length: None or int > 0
            The hop length of the STFT.  If not provided, it will default to ``n_fft // 4``
        win_length: None or int > 0
            The window length of the STFT.  By default, it will equal ``n_fft``
        window: string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
            A window specification as supported by `stft` or `istft`
        center: boolean
            If `True`, the STFT is assumed to use centered frames.
            If `False`, the STFT is assumed to use left-aligned frames.
        pad_mode: string
            If ``center=True``, the padding mode to use at the edges of the signal.
            By default, STFT uses zero padding.
        power: float > 0 [scalar]
            Exponent for the magnitude melspectrogram
        n_iter: int > 0
            The number of iterations for Griffin-Lim
        length: None or int > 0
            If provided, the output ``y`` is zero-padded or clipped to exactly ``length``
            samples.
        dtype: np.dtype
            Real numeric type for the time-domain signal.  Default is 32-bit float.
        **kwargs: additional keyword arguments for Mel filter bank parameters
        fmin: float >= 0 [scalar]
            lowest frequency (in Hz)
        fmax: float >= 0 [scalar]
            highest frequency (in Hz).
            If `None`, use ``fmax = sr / 2.0``
        htk: bool [scalar]
            use HTK formula instead of Slaney
        norm: {None, 'slaney', or number} [scalar]
            If 'slaney', divide the triangular mel weights by the width of
            the mel band (area normalization).
            If numeric, use `librosa.util.normalize` to normalize each filter
            by to unit l_p norm. See `librosa.util.normalize` for a full
            description of supported norm values (including `+-np.inf`).
            Otherwise, leave all the triangles aiming for a peak value of 1.0

    Returns:
        y: np.ndarray [shape(..., n,)]
            time-domain signal reconstructed from ``M``
    """
    stft = mel_to_stft(M, sr=sr, n_fft=n_fft, power=power, **kwargs)

    return griffinlim(
        stft,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window=window,
        center=center,
        dtype=dtype,
        length=length,
        pad_mode=pad_mode,
    )