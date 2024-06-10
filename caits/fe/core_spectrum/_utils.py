# The functionalities in this implementation are basically derived from
# librosa v0.10.1:
# https://github.com/librosa/librosa/blob/main/librosa/util/utils.py
# https://github.com/librosa/librosa/blob/main/librosa/filters.py
# https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py
import warnings
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import scipy

from caits.core._core_typing import _FloatLike_co, _ScalarOrSequence
from caits.core._core_window import normalize
from caits.core.numpy_typing import DTypeLike

# Constrain STFT block sizes to 256 KB
MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10


def expand_to(x: np.ndarray, *, ndim: int, axes: Union[int, slice, Sequence[int], Sequence[slice]]) -> np.ndarray:
    # Force axes into a tuple
    axes_tup: Tuple[int]
    try:
        axes_tup = tuple(axes)  # type: ignore
    except TypeError:
        axes_tup = tuple([axes])  # type: ignore

    if len(axes_tup) != x.ndim:
        raise ValueError(f"Shape mismatch between axes={axes_tup} and input " f"x.shape={x.shape}")

    if ndim < x.ndim:
        raise ValueError(f"Cannot expand x.shape={x.shape} to fewer dimensions ndim={ndim}")

    shape: List[int] = [1] * ndim
    for i, axi in enumerate(axes_tup):
        shape[axi] = x.shape[i]

    return x.reshape(shape)


def __overlap_add(y, ytmp, hop_length):
    # numba-accelerated overlap add for inverse stft
    # y is the pre-allocated output buffer
    # ytmp is the windowed inverse-stft frames
    # hop_length is the hop-length of the STFT analysis

    n_fft = ytmp.shape[-2]
    N = n_fft
    for frame in range(ytmp.shape[-1]):
        sample = frame * hop_length
        if N > y.shape[-1] - sample:
            N = y.shape[-1] - sample

        y[..., sample: (sample + N)] += ytmp[..., :N, frame]


def _nnls_obj(x: np.ndarray, shape: Sequence[int], A: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray]:
    """Computes the objective and gradient for NNLS"""
    # Scipy's lbfgs flattens all arrays, so we first reshape
    # the iteration x
    x = x.reshape(shape)

    # Compute the difference matrix
    diff = np.einsum("mf,...ft->...mt", A, x, optimize=True) - B

    # Compute the objective value
    value = (1 / B.size) * 0.5 * np.sum(diff ** 2)

    # And the gradient
    grad = (1 / B.size) * np.einsum("mf,...mt->...ft", A, diff, optimize=True)

    # Flatten the gradient
    return value, grad.flatten()


def _nnls_lbfgs_block(A: np.ndarray, B: np.ndarray, x_init: Optional[np.ndarray] = None, **kwargs: Any) -> np.ndarray:
    """Solves the constrained problem over a single block.

    Parameters
    ----------
    A : np.ndarray [shape=(m, d)]
        The basis matrix
    B : np.ndarray [shape=(m, N)]
        The regression targets
    x_init : np.ndarray [shape=(d, N)]
        An initial guess
    **kwargs
        Additional keyword arguments to `scipy.optimize.fmin_l_bfgs_b`

    Returns
    -------
    x : np.ndarray [shape=(d, N)]
        Non-negative matrix such that Ax ~= B
    """
    # If we don't have an initial point, start at the projected
    # least squares solution
    if x_init is None:
        # Suppress type checks because mypy can't find pinv
        x_init = np.einsum("fm,...mt->...ft", np.linalg.pinv(A), B, optimize=True)
        np.clip(x_init, 0, None, out=x_init)

    # Adapt the hessian approximation to the dimension of the problem
    kwargs.setdefault("m", A.shape[1])

    # Construct non-negative bounds
    bounds = [(0, None)] * x_init.size
    shape = x_init.shape

    # optimize
    x: np.ndarray
    x, obj_value, diagnostics = scipy.optimize.fmin_l_bfgs_b(
        _nnls_obj, x_init, args=(shape, A, B), bounds=bounds, **kwargs
    )
    # reshape the solution
    return x.reshape(shape)


def nnls(A: np.ndarray, B: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Non-negative least squares.

    Given two matrices A and B, find a non-negative matrix X
    that minimizes the sum squared error::

        err(X) = sum_i,j ((AX)[i,j] - B[i, j])^2

    Args:
        A: np.ndarray [shape=(m, n)]
            The basis matrix
        B: np.ndarray [shape=(..., m, N)]
            The target array.  Additional leading dimensions are supported.
        **kwargs
            Additional keyword arguments to `scipy.optimize.fmin_l_bfgs_b`

    Returns
        X: np.ndarray [shape=(..., n, N), non-negative]
        A minimizing solution to ``|AX - B|^2``
    """
    # If B is a single vector, punt up to the scipy method
    if B.ndim == 1:
        return scipy.optimize.nnls(A, B)[0]  # type: ignore

    n_columns = int(MAX_MEM_BLOCK // (np.prod(B.shape[:-1]) * A.itemsize))
    n_columns = max(n_columns, 1)

    # Process in blocks:
    if B.shape[-1] <= n_columns:
        return _nnls_lbfgs_block(A, B, **kwargs).astype(A.dtype)

    x: np.ndarray
    x = np.einsum("fm,...mt->...ft", np.linalg.pinv(A), B, optimize=True)
    np.clip(x, 0, None, out=x)
    x_init = x

    for bl_s in range(0, x.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, B.shape[-1])
        x[..., bl_s:bl_t] = _nnls_lbfgs_block(A, B[..., bl_s:bl_t], x_init=x_init[..., bl_s:bl_t], **kwargs)
    return x


def mel_filter(
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
            enorm = 2.0 / (mel_f[2: n_mels + 2] - mel_f[:n_mels])
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


def fft_frequencies(*, sr: float = 22050, n_fft: int = 2048) -> np.ndarray:
    return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)


def mel_frequencies(n_mels: int = 128, *, fmin: float = 0.0, fmax: float = 11025.0, htk: bool = False) -> np.ndarray:
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    hz: np.ndarray = mel_to_hz(mels, htk=htk)
    return hz


def hz_to_mel(
        frequencies: _ScalarOrSequence[_FloatLike_co], *, htk: bool = False
) -> Union[np.floating, np.ndarray]:
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
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels: _ScalarOrSequence[_FloatLike_co], *, htk: bool = False) -> Union[np.floating, np.ndarray]:
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
