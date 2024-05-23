# The functionalities in this implementation are basically derived from
# librosa v0.10.1:
# https://github.com/librosa/librosa/blob/main/librosa/util/utils.py
# https://github.com/librosa/librosa/blob/main/librosa/util/deprecation.py
from typing import Dict, Optional, Union

import numpy as np
from numpy.typing import DTypeLike


class Deprecated(object):
    """A placeholder class to catch usage of deprecated variable names"""

    def __repr__(self) -> str:
        """Pretty-print display for deprecated objects"""
        return "<DEPRECATED parameter>"


def is_positive_int(x: float) -> bool:
    """Check that x is a positive integer, i.e. 1 or greater.

    Parameters
    ----------
    x : number

    Returns
    -------
    positive : bool
    """
    # Check type first to catch None values.
    return isinstance(x, (int, np.integer)) and (x > 0)


def valid_audio(y: np.ndarray, *, mono: Union[bool, Deprecated] = Deprecated()) -> bool:
    if not isinstance(y, np.ndarray):
        raise ValueError("Audio data must be of type numpy.ndarray")

    if not np.issubdtype(y.dtype, np.floating):
        raise ValueError("Audio data must be floating-point")

    if y.ndim == 0:
        raise ValueError(f"Audio data must be at least one-dimensional, given y.shape={y.shape}")

    if isinstance(mono, Deprecated):
        mono = False

    if mono and y.ndim != 1:
        raise ValueError(f"Invalid shape for monophonic audio: ndim={y.ndim:d}, shape={y.shape}")

    if not np.isfinite(y).all():
        raise ValueError("Audio buffer is not finite everywhere")

    return True


def dtype_r2c(d: DTypeLike, *, default: Optional[type] = np.complex64) -> DTypeLike:
    mapping: Dict[DTypeLike, type] = {
        np.dtype(np.float32): np.complex64,
        np.dtype(np.float64): np.complex128,
        np.dtype(float): np.dtype(complex).type,
    }

    # If we're given a complex type already, return it
    dt = np.dtype(d)
    if dt.kind == "c":
        return dt

    # Otherwise, try to map the dtype.
    # If no match is found, return the default.
    return np.dtype(mapping.get(dt, default))


def dtype_c2r(d: DTypeLike, *, default: Optional[type] = np.float32) -> DTypeLike:
    mapping: Dict[DTypeLike, type] = {
        np.dtype(np.complex64): np.float32,
        np.dtype(np.complex128): np.float64,
        np.dtype(complex): np.dtype(float).type,
    }

    # If we're given a real type already, return it
    dt = np.dtype(d)
    if dt.kind == "f":
        return dt

    # Otherwise, try to map the dtype.
    # If no match is found, return the default.
    return np.dtype(mapping.get(dt, default))
