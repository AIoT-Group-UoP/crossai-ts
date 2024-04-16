# The functionalities in this implementation are basically derived from
# librosa v0.10.1:
# https://github.com/librosa/librosa/blob/main/librosa/util/utils.py
# https://github.com/librosa/librosa/blob/main/librosa/filters.py
# https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py
from typing import Optional, Union, Callable, Tuple, Any, Sequence, List, Dict
import numpy as np
import scipy
from numpy.typing import ArrayLike, DTypeLike
from numpy.lib.stride_tricks import as_strided
from ._typing_base import _FloatLike_co, _WindowSpec

# Constrain STFT block sizes to 256 KB
MAX_MEM_BLOCK = 2**8 * 2**10


def frame(
    x: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    axis: int = -1,
    writeable: bool = False,
    subok: bool = False,
) -> np.ndarray:
    
    x = np.array(x, copy=False, subok=subok)

    if x.shape[axis] < frame_length:
        raise ValueError(
            f"Input is too short (n={x.shape[axis]:d}) for "
            f"frame_length={frame_length:d}"
        )

    if hop_length < 1:
        raise ValueError(f"Invalid hop_length: {hop_length:d}")

    # put our new within-frame axis at the end for now
    out_strides = x.strides + tuple([x.strides[axis]])

    # Reduce the shape on the framing axis
    x_shape_trimmed = list(x.shape)
    x_shape_trimmed[axis] -= frame_length - 1

    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok,
        writeable=writeable
    )

    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1

    xw = np.moveaxis(xw, -1, target_axis)

    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    return xw[tuple(slices)]


def get_window(
    window: Union[str, Tuple[Any, ...], float, Callable[[int], np.ndarray], ArrayLike],
    Nx: int,
    *,
    fftbins: Optional[bool] = True,
) -> np.ndarray:
    if callable(window):
        return window(Nx)

    elif isinstance(window, (str, tuple)) or np.isscalar(window):

        win: np.ndarray = scipy.signal.get_window(window, Nx, fftbins=fftbins)
        return win

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise ValueError(f"Window size mismatch: {len(window):d} != {Nx:d}")
    else:
        raise ValueError(f"Invalid window specification: {window!r}")


def pad_center(
    data: np.ndarray,
    *,
    size: int,
    axis: int = -1,
    **kwargs: Any
) -> np.ndarray:

    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ValueError(
            f"Target size ({size:d}) must be at least input size ({n:d})"
        )

    return np.pad(data, lengths, **kwargs)


def expand_to(
    x: np.ndarray,
    *,
    ndim: int,
    axes: Union[int, slice, Sequence[int], Sequence[slice]]
) -> np.ndarray:
    # Force axes into a tuple
    axes_tup: Tuple[int]
    try:
        axes_tup = tuple(axes)  # type: ignore
    except TypeError:
        axes_tup = tuple([axes])  # type: ignore

    if len(axes_tup) != x.ndim:
        raise ValueError(
            f"Shape mismatch between axes={axes_tup} and input "
            f"x.shape={x.shape}"
        )

    if ndim < x.ndim:
        raise ValueError(
            f"Cannot expand x.shape={x.shape} to fewer dimensions ndim={ndim}"
        )

    shape: List[int] = [1] * ndim
    for i, axi in enumerate(axes_tup):
        shape[axi] = x.shape[i]

    return x.reshape(shape)


def normalize(
    S: np.ndarray,
    *,
    norm: Optional[float] = np.inf,
    axis: Optional[int] = 0,
    threshold: Optional[_FloatLike_co] = None,
    fill: Optional[bool] = None,
) -> np.ndarray:
    # Avoid div-by-zero
    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        raise ValueError(f"threshold={threshold} must be strictly positive")

    if fill not in [None, False, True]:
        raise ValueError(f"fill={fill} must be None or boolean")

    if not np.all(np.isfinite(S)):
        raise ValueError("Input must be finite")

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm is None:
        return S

    elif norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise ValueError("Cannot normalize with norm=0 and fill=True")

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True) ** (1.0 / norm)

        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)

    else:
        raise ValueError(f"Unsupported norm: {repr(norm)}")

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm


def tiny(
        x: Union[float, np.ndarray]
) -> _FloatLike_co:
    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.dtype(np.float32)

    return np.finfo(dtype).tiny


def window_sumsquare(
    *,
    window: _WindowSpec,
    n_frames: int,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    n_fft: int = 2048,
    dtype: DTypeLike = np.float32,
    norm: Optional[float] = None,
) -> np.ndarray:

    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length)
    win_sq = normalize(win_sq, norm=norm) ** 2
    win_sq = pad_center(win_sq, size=n_fft)

    # Fill the envelope
    __window_ss_fill(x, win_sq, n_frames, hop_length)

    return x


def __window_ss_fill(x, win_sq, n_frames, hop_length):  # pragma: no cover
    """Compute the sum-square envelope of a window."""
    n = len(x)
    n_fft = len(win_sq)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]


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


def _nnls_obj(
    x: np.ndarray, shape: Sequence[int], A: np.ndarray, B: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Compute the objective and gradient for NNLS"""
    # Scipy's lbfgs flattens all arrays, so we first reshape
    # the iterate x
    x = x.reshape(shape)

    # Compute the difference matrix
    diff = np.einsum("mf,...ft->...mt", A, x, optimize=True) - B

    # Compute the objective value
    value = (1 / B.size) * 0.5 * np.sum(diff**2)

    # And the gradient
    grad = (1 / B.size) * np.einsum("mf,...mt->...ft", A, diff, optimize=True)

    # Flatten the gradient
    return value, grad.flatten()


def _nnls_lbfgs_block(
    A: np.ndarray, B: np.ndarray, x_init: Optional[np.ndarray] = None, **kwargs: Any
) -> np.ndarray:
    """Solve the constrained problem over a single block

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
        x[..., bl_s:bl_t] = _nnls_lbfgs_block(
            A, B[..., bl_s:bl_t], x_init=x_init[..., bl_s:bl_t], **kwargs
        )
    return x
