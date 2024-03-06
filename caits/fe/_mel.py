# The functionalities in this implementation are basically derived from
# librosa v0.10.1:
# https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py
import warnings
from typing import Any, Tuple, Optional, Sequence, Union, TypeVar, Callable
import numpy as np
import scipy
from numpy.typing import ArrayLike, DTypeLike
from typing_extensions import Literal
from ..base._utility import expand_to, normalize
from ..base._frequency import _spectrogram, power_to_db

_T = TypeVar("_T")

_STFTPad = Literal[
    "constant",
    "edge",
    "linear_ramp",
    "reflect",
    "symmetric",
    "empty",
]
_Real = Union[float, "np.integer[Any]", "np.floating[Any]"]
_Number = Union[complex, "np.number[Any]"]
_PadModeSTFT = Union[_STFTPad, Callable[..., Any]]
_WindowSpec = Union[str, Tuple[Any, ...], float, Callable[[int], np.ndarray], ArrayLike]
_SequenceLike = Union[Sequence[_T], np.ndarray]
_ScalarOrSequence = Union[_T, _SequenceLike[_T]]
_BoolLike_co = Union[bool, np.bool_]
_IntLike_co = Union[_BoolLike_co, int, "np.integer[Any]"]
_FloatLike_co = Union[_IntLike_co, float, "np.floating[Any]"]
_ComplexLike_co = Union[_FloatLike_co, complex, "np.complexfloating[Any, Any]"]


def mfcc_stat(
    y: Optional[np.ndarray] = None,
    sr: int = 22050,
    S: Optional[np.ndarray] = None,
    n_mfcc: int = 20,
    dct_type: int = 2,
    norm: Optional[str] = "ortho",
    lifter: float = 0,
    export: str = "dict",
    **kwargs: Any,
) -> Union[Tuple[float, float], dict]:
    mfcc_arr = mfcc(y=y, sr=sr, S=S, n_mfcc=n_mfcc, dct_type=dct_type,
                    norm=norm, lifter=lifter, **kwargs)

    mfcc_mean = np.mean(mfcc_arr, axis=1)
    mfcc_std = np.std(mfcc_arr, axis=1)

    if export == "values":
        return mfcc_mean, mfcc_std
    elif export == "dict":
        return {
            "mfcc_mean": mfcc_mean,
            "mfcc_std": mfcc_std
        }
    else:
        raise ValueError(f"Unsupported export={export}")


def mfcc(
    y: Optional[np.ndarray] = None,
    sr: int = 22050,
    S: Optional[np.ndarray] = None,
    n_mfcc: int = 20,
    dct_type: int = 2,
    norm: Optional[str] = "ortho",
    lifter: float = 0,
    **kwargs: Any,
) -> np.ndarray:
    if S is None:
    # multichannel behavior may be different due to relative noise floor
    # differences between channels
        S = power_to_db(melspectrogram(y=y, sr=sr, **kwargs))

    M: np.ndarray = scipy.fftpack.dct(S, axis=-2, type=dct_type, norm=norm)[
        ..., :n_mfcc, :
    ]

    if lifter > 0:
        # shape lifter for broadcasting
        LI = np.sin(np.pi * np.arange(1, 1 + n_mfcc, dtype=M.dtype) / lifter)
        LI = expand_to(LI, ndim=S.ndim, axes=-2)

        M *= 1 + (lifter / 2) * LI
        return M
    elif lifter == 0:
        return M
    else:
        raise ValueError(f"MFCC lifter={lifter} must be a non-negative number")


def filter_mel(
    *,
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
    norm: Optional[Union[Literal["slaney"], float]] = "slaney",
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if isinstance(norm, str):
        if norm == "slaney":
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
            weights *= enorm[:, np.newaxis]
        else:
            raise ValueError(f"Unsupported norm={norm}")
    else:
        weights = normalize(weights, norm=norm, axis=-1)

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn(
            "Empty filters detected in mel frequency basis. "
            "Some channels will produce empty responses. "
            "Try increasing your sampling rate (and fmax) or "
            "reducing n_mels.",
            stacklevel=2,
        )

    return weights


def melspectrogram(
        *,
        y: Optional[np.ndarray] = None,
        sr: float = 22050,
        S: Optional[np.ndarray] = None,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = "hann",
        center: bool = True,
        pad_mode: _PadModeSTFT = "constant",
        power: float = 2.0,
        **kwargs: Any,
) -> np.ndarray:
    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Build a Mel filter
    mel_basis = filter_mel(sr=sr, n_fft=n_fft, **kwargs)

    melspec: np.ndarray = np.einsum("...ft,mf->...mt", S, mel_basis,
                                    optimize=True)
    return melspec


def fft_frequencies(*, sr: float = 22050, n_fft: int = 2048) -> np.ndarray:

    return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)


def mel_frequencies(
    n_mels: int = 128, *, fmin: float = 0.0, fmax: float = 11025.0, htk: bool = False
) -> np.ndarray:

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    hz: np.ndarray = mel_to_hz(mels, htk=htk)
    return hz


def hz_to_mel(
    frequencies: _ScalarOrSequence[_FloatLike_co], *, htk: bool = False
) -> Union[np.floating[Any], np.ndarray]:

    frequencies = np.asanyarray(frequencies)

    if htk:
        mels: np.ndarray = 2595.0 * np.log10(1.0 + frequencies / 700.0)
        return mels

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(
            frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(
    mels: _ScalarOrSequence[_FloatLike_co], *, htk: bool = False
) -> Union[np.floating[Any], np.ndarray]:

    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs
