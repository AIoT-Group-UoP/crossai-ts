# The functionalities in this implementation are basically derived from
# librosa v0.10.1:
# https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py
# https://github.com/librosa/librosa/blob/main/librosa/util/utils.py
from typing import Optional, Union
import numpy as np

_Real = Union[float, "np.integer[Any]", "np.floating[Any]"]
_Number = Union[complex, "np.number[Any]"]


def phase_vocoder(
    D: np.ndarray,
    *,
    rate: float,
    hop_length: Optional[int] = None,
    n_fft: Optional[int] = None,
) -> np.ndarray:
    if n_fft is None:
        n_fft = 2 * (D.shape[-2] - 1)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)

    # Create an empty output array
    shape = list(D.shape)
    shape[-1] = len(time_steps)
    d_stretch = np.zeros_like(D, shape=shape)

    # Expected phase advance in each bin
    phi_advance = np.linspace(0, np.pi * hop_length, D.shape[-2])

    # Phase accumulator; initialize to the first sample
    phase_acc = np.angle(D[..., 0])

    # Pad 0 columns to simplify boundary logic
    padding = [(0, 0) for _ in D.shape]
    padding[-1] = (0, 2)
    D = np.pad(D, padding, mode="constant")

    for t, step in enumerate(time_steps):
        columns = D[..., int(step) : int(step + 2)]

        # Weighting for linear magnitude interpolation
        alpha = np.mod(step, 1.0)
        mag = (1.0 - alpha) * np.abs(columns[..., 0]) + alpha * np.abs(columns[..., 1])

        # Store to output array
        d_stretch[..., t] = phasor(phase_acc, mag=mag)

        # Compute phase advance
        dphase = np.angle(columns[..., 1]) - np.angle(columns[..., 0]) - phi_advance

        # Wrap to -pi:pi range
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))

        # Accumulate phase
        phase_acc += phi_advance + dphase

    return d_stretch


def _phasor_angles(x) -> np.complex_:  # pragma: no cover
    return np.cos(x) + 1j * np.sin(x)  # type: ignore


def phasor(
    angles: Union[np.ndarray, _Real],
    *,
    mag: Optional[Union[np.ndarray, _Number]] = None,
) -> Union[np.ndarray, np.complex_]:

    z = _phasor_angles(angles)

    if mag is not None:
        z *= mag

    return z
