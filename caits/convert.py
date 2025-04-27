# The functionality in this implementation is basically derived from
# librosa v0.10.1:
# https://github.com/librosa/librosa/blob/main/librosa/core/convert.py
import numpy as np
from typing import Optional, Union, Any
from caits.core._core_typing import _ScalarOrSequence, _IntLike_co

def times_like(
    X: Union[np.ndarray, float],
    *,
    sr: float = 22050,
    hop_length: int = 512,
    n_fft: Optional[int] = None,
    axis: int = -1,
) -> np.ndarray:
    """Returns an array of time values to match the time axis from a feature
    matrix.

    Args:
        X: A numpy array or scalar
            - If ndarray, X is a feature matrix, e.g., STFT, chromagram, or mel
            spectrogram.
            - If scalar, X represents the number of frames.
        sr: A number (scalar) > 0 that indicates the sampling rate of the
            signal.
        hop_length: An integer (scalar) > 0 corresponding to the number of
            samples between successive frames
        n_fft: Can be None or integer (scalar) > 0
            Optional: length of the FFT window.
            If given, time conversion will include an offset of ``n_fft // 2``
            to counteract windowing effects when using a non-centered STFT.
        axis: An integer (scalar) which indicates the axis representing the time
            axis of ``X``. By default, the last axis ``(-1)`` is taken.

    Returns:
        A numpy array in [shape=(n,)] that corresponds to the times (in
        seconds) corresponding to each frame of X.
    """
    samples = samples_like(X, hop_length=hop_length, n_fft=n_fft, axis=axis)
    time: np.ndarray = samples_to_time(samples, sr=sr)
    return time



def samples_like(
    X: Union[np.ndarray, float],
    *,
    hop_length: int = 512,
    n_fft: Optional[int] = None,
    axis: int = -1,
) -> np.ndarray:
    """Returns an array of sample indices to match the time axis from a
    feature matrix.

    Args:
        X: A numpy array or scalar
        - If ndarray, X is a feature matrix, e.g., STFT, chromagram, or mel
            spectrogram.
        - If scalar, X represents the number of frames.
        hop_length: An integer (scalar) > 0 with the number of samples between
            successive frames.
        n_fft: Can be None or integer (scalar) > 0
            Optional: length of the FFT window.
            If given, time conversion will include an offset of ``n_fft // 2``
            to counteract windowing effects when using a non-centered STFT.
        axis: An integer (scalar) which indicates the axis representing the
            time axis of ``X``. By default, the last axis ``(-1)`` is taken.

    Returns:
        A numpy array in [shape=(n,)] with the sample indices corresponding to
        each frame of ``X``.

    """
    # suppress type checks because mypy does not understand isscalar
    if np.isscalar(X):
        frames = np.arange(X)  # type: ignore
    else:
        frames = np.arange(X.shape[axis])  # type: ignore
    return frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)


def samples_to_time(
    samples: _ScalarOrSequence[_IntLike_co], *, sr: float = 22050
) -> Union[np.floating[Any], np.ndarray]:
    """Convert sample indices to time (in seconds).

    Parameters
    ----------
    samples : np.ndarray
        Sample index or array of sample indices
    sr : number > 0
        Sampling rate

    Returns
    -------
    times : np.ndarray [shape=samples.shape]
        Time values corresponding to ``samples`` (in seconds)

    """
    return np.asanyarray(samples) / float(sr)


def frames_to_samples(
    frames: _ScalarOrSequence[_IntLike_co],
    *,
    hop_length: int = 512,
    n_fft: Optional[int] = None,
) -> Union[np.integer[Any], np.ndarray]:
    """Convert frame indices to audio sample indices.

    Parameters
    ----------
    frames : number or np.ndarray [shape=(n,)]
        frame index or vector of frame indices
    hop_length : int > 0 [scalar]
        number of samples between successive frames
    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``n_fft // 2``
        to counteract windowing effects when using a non-centered STFT.

    Returns
    -------
    times : number or np.ndarray
        time (in samples) of each given frame number::

            times[i] = frames[i] * hop_length

    """
    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    return (np.asanyarray(frames) * hop_length + offset).astype(int)
