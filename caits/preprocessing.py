import numpy as np


def normalize_signal(
        sig: np.ndarray
) -> np.ndarray:
    """Normalizes a signal to the proper range.

    Args:
        sig: An array-like input signal.

    Returns:
        array-like: The normalized signal.
    """
    if np.all(sig == 0):
        return sig
    else:  
        try:
            intinfo = np.iinfo(sig.dtype)
            return sig / max(intinfo.max, -intinfo.min)

        except ValueError:  # array is not integer dtype
            return sig / max(sig.max(), -sig.min())


def resample_signal(
        sig: np.ndarray,
        native_sr: int,
        target_sr: int,
        d_type: np.dtype = np.float32
) -> np.ndarray:
    """Resamples an input audio buffer to the goal sampling rate. Linear
    resampling using numpy is significantly faster than Librosa's default
    technique.

    Args:
        sig: The input signal as a numpy.ndarray.
        native_sr: The native sampling rate of the input signal as integer.
        target_sr: The goal sampling rate as integer.
        d_type: The data type of the resampled audio buffer.

    Returns:
        np.ndarray: The resampled signal.
    """
    duration = len(sig) / native_sr
    n_target_samples = int(duration * g_sr)
    time_x_source = np.linspace(0, duration, len(sig),
                                dtype=d_type)
    time_x = np.linspace(0, duration, n_target_samples,
                         dtype=d_type)
    resampled_buffer = np.interp(time_x, time_x_source, sig)
    return resampled_buffer


# TODO: Make it also work for (n_samples, ) arrays for robustness
def resample_2d(
        audio_data: np.ndarray,
        native_sr: int,
        target_sr: int
) -> np.ndarray:
    """Resamples 2D audio data (multi-channel) to a target sampling rate.

    Args:
        audio_data: The input audio data as a 2D numpy (n_samples, n_channels).
        native_sr: The native sampling rate of the input audio data.
        target_sr: The target sampling rate as integer.

    Returns:
        np.ndarray: The resampled audio data as a 2D numpy.ndarray.
    """

    # Initialize a list to hold resampled channels
    resampled_channels = []

    # Iterate through each channel in the audio data
    for i in range(audio_data.shape[1]):
        channel_data = audio_data[:, i]
        resampled_channel_data = resample_signal(channel_data,
                                                 native_sr,
                                                 target_sr)

        resampled_channels.append(resampled_channel_data)

    # Stack the resampled channels back into a 2D array
    resampled_audio_data = np.stack(resampled_channels, axis=1)

    return resampled_audio_data


def trim_signal(
        array: np.ndarray,
        axis=0,
        epsilon: float = 1e-5
) -> np.ndarray:
    """Trims the noise from beginning and end of a signal.

    Args:
      array: The input signal.
      axis: The axis to trim.
      epsilon: The max value to be considered as noise.
    Returns:
      np.ndarray: A signal of start and stop with shape `[..., 2, ...]`.
    """
    shape = array.shape
    length = shape[axis]

    nonzero = np.greater(array, epsilon)
    check = np.any(nonzero, axis=axis)

    forward = np.array(nonzero, np.int64)
    reverse = forward[::-1]

    start = np.where(check, np.argmax(forward, axis=axis), length)
    stop = np.where(check, np.argmax(reverse, axis=axis),
                    np.array(0, np.int64))
    stop = length - stop

    return array[start:stop]
