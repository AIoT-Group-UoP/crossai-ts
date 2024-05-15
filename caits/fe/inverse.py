import warnings
import numpy as np
from typing import Any, Optional, Union
from numpy.typing import DTypeLike
from caits.core._core_typing import _WindowSpec, _PadModeSTFT
from .core_spectrum._utils import nnls, mel_filter
from caits.core._core_window import tiny
from .core_spectrum._phase import phasor
from ._spectrum import istft, stft
from caits.core._core_checks import dtype_r2c


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


def griffinlim(
    S: np.ndarray,
    *,
    n_iter: int = 32,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_fft: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    dtype: Optional[DTypeLike] = None,
    length: Optional[int] = None,
    pad_mode: _PadModeSTFT = "constant",
    momentum: float = 0.99,
    init: Optional[str] = "random",
    random_state: Optional[
        Union[int, np.random.RandomState, np.random.Generator]
    ] = None,
) -> np.ndarray:
    """Approximate magnitude spectrogram inversion using the "fast" Griffin-Lim
    algorithm.

    Given a short-time Fourier transform magnitude matrix (``S``), the
    algorithm randomly initializes phase estimates, and then alternates
    forward- and inverse-STFT operations. [#]_

    Note that this assumes reconstruction of a real-valued time-domain signal,
    and that ``S`` contains only the non-negative frequencies (as computed by
    `stft`).

    The "fast" GL method [#]_ uses a momentum parameter to accelerate
    convergence.

    [#] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    [#] Perraudin, N., Balazs, P., & Søndergaard, P. L.
        "A fast Griffin-Lim algorithm,"
        IEEE Workshop on Applications of Signal Processing to Audio and
        Acoustics (pp. 1-4),
        Oct. 2013.

    Args:
        S: np.ndarray [shape=(..., n_fft // 2 + 1, t), non-negative]
            An array of short-time Fourier transform magnitudes as produced by
            `stft`.

        n_iter: int > 0
            The number of iterations to run

        hop_length: None or int > 0
            The hop length of the STFT.  If not provided, it will default to
                ``n_fft // 4``

        win_length: None or int > 0
            The window length of the STFT.  By default, it will equal ``n_fft``

        n_fft: None or int > 0
            The number of samples per frame.
            By default, this will be inferred from the shape of ``S`` as an
                even number. However, if an odd frame length was used, you can
                explicitly set ``n_fft``.

        window: string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
            A window specification as supported by `stft` or `istft`

        center: boolean
            If ``True``, the STFT is assumed to use centered frames.
            If ``False``, the STFT is assumed to use left-aligned frames.

        dtype: np.dtype
            Real numeric type for the time-domain signal.  Default is inferred
            to match the precision of the input spectrogram.

        length: None or int > 0
            If provided, the output ``y`` is zero-padded or clipped to exactly
            ``length`` samples.

        pad_mode: string
            If ``center=True``, the padding mode to use at the edges of the
            signal. By default, STFT uses zero padding.

        momentum: number >= 0
            The momentum parameter for fast Griffin-Lim.
            Setting this to 0 recovers the original Griffin-Lim method [1]_.
            Values near 1 can lead to faster convergence, but above 1 may not
            converge.

        init: None or 'random' [default]
            If 'random' (the default), then phase values are initialized
            randomly according to ``random_state``.  This is recommended when
            the input ``S`` isa magnitude spectrogram with no initial phase
            estimates.

            If `None`, then the phase is initialized from ``S``.  This is
            useful when an initial guess for phase can be provided, or when
            you want to resume Griffin-Lim from a previous output.

        random_state: None, int, np.random.RandomState, or np.random.Generator
            If int, random_state is the seed used by the random number
            generator for phase initialization.

            If `np.random.RandomState` or `np.random.Generator` instance, the
            random number generator itself.

            If `None`, defaults to the `np.random.default_rng()` object.

        Returns:
            y: np.ndarray [shape=(..., n)]
                time-domain signal reconstructed from ``S``
    """
    # The functionality in this implementation is basically derived from
    # librosa v0.10.1:
    # https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)  # type: ignore
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rng = random_state  # type: ignore
    else:
        raise ValueError(f"Unsupported random_state={random_state!r}")

    if momentum > 1:
        warnings.warn(
            f"Griffin-Lim with momentum={momentum} > 1 can be unstable. "
            "Proceed with caution!",
            stacklevel=2,
        )
    elif momentum < 0:
        raise ValueError(f"griffinlim() called with momentum={momentum} < 0")

    # Infer n_fft from the spectrogram shape
    if n_fft is None:
        n_fft = 2 * (S.shape[-2] - 1)

    # Infer the dtype from S
    angles = np.empty(S.shape, dtype=dtype_r2c(S.dtype))
    eps = tiny(angles)

    if init == "random":
        # randomly initialize the phase
        angles[:] = phasor((2 * np.pi * rng.random(size=S.shape)))
    elif init is None:
        # Initialize an all ones complex matrix
        angles[:] = 1.0
    else:
        raise ValueError(f"init={init} must either None or 'random'")

    # Place-holders for temporary data and reconstructed buffer
    rebuilt = None
    tprev = None
    inverse = None

    # Absorb magnitudes into angles
    angles *= S
    for _ in range(n_iter):
        # Invert with our current estimate of the phases
        inverse = istft(
            angles,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            window=window,
            center=center,
            dtype=dtype,
            length=length,
            out=inverse,
        )

        # Rebuild the spectrogram
        rebuilt = stft(
            inverse,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            out=rebuilt,
        )

        # Update our phase estimates
        angles[:] = rebuilt
        if tprev is not None:
            angles -= (momentum / (1 + momentum)) * tprev
        angles /= np.abs(angles) + eps
        angles *= S
        # Store
        rebuilt, tprev = tprev, rebuilt

    # Return the final phase estimates
    return istft(
        angles,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window=window,
        center=center,
        dtype=dtype,
        length=length,
        out=inverse,
    )
    
    
def mel_to_audio(
    M: np.ndarray,
    *,
    sr: float = 22050,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    momentum: float = 0.99,
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
            The hop length of the STFT.  If not provided, it will default to
            ``n_fft // 4``
        win_length: None or int > 0
            The window length of the STFT.  By default, it will equal ``n_fft``
        window: string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
            A window specification as supported by `stft` or `istft`
        center: boolean
            If `True`, the STFT is assumed to use centered frames.
            If `False`, the STFT is assumed to use left-aligned frames.
        pad_mode: string
            If ``center=True``, the padding mode to use at the edges of the
            signal. By default, STFT uses zero padding.
        power: float > 0 [scalar]
            Exponent for the magnitude melspectrogram
        n_iter: int > 0
            The number of iterations for Griffin-Lim
        length: None or int > 0
            If provided, the output ``y`` is zero-padded or clipped to exactly
            ``length`` samples.
        dtype: np.dtype
            Real numeric type for the time-domain signal.  Default is 32-bit
            float.
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
        momentum=momentum,
        center=center,
        dtype=dtype,
        length=length,
        pad_mode=pad_mode,
    )
