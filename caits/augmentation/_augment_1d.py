from typing import Optional, Union, Tuple, List, Any
import numpy as np
from caits.base import fix_length, is_positive_int, resample
from caits.fe._spectrum import stft, istft
from caits.base import phase_vocoder


def add_white_noise(
        array: np.ndarray,
        noise_factor: float
) -> np.ndarray:
    """Adds white noise to a signal.

    Args:
        array (ndarray): Input signal.
        noise_factor (float): Noise factor.

    Returns:
        ndarray: Noisy signal.
    """
    noise = np.random.normal(0, array.std(), array.size)
    return array + noise_factor * noise


def random_gain(
        array: np.ndarray,
        min_factor: float = 0.1,
        max_factor: float = 0.12
) -> np.ndarray:
    """Applies random gain to a signal.

    Args:
        array: The input signal.
        min_factor: The minimum gain factor.
        max_factor: The maximum gain factor.

    Returns:
        ndarray: The signal with random gain applied.
    """
    gain_rate = np.random.uniform(min_factor, max_factor)
    return array * gain_rate


def polarity_inversion(
        array: np.ndarray,
) -> np.ndarray:
    """Inverts the polarity of a signal.

    Args:
        array: The input signal.

    Returns:
        ndarray: The signal with inverted polarity.
    """
    return array * -1


def add_noise_ts(
        array: np.ndarray,
        loc: Union[float, Tuple[float, float], List[float]] = 0.0,
        scale: Union[float, Tuple[float, float], List[float]] = 0.1,
        distribution: str = "gaussian",
        kind: str = "additive",
        per_channel: bool = True,
        normalize: bool = True,
        repeats: int = 1,
        prob: float = 1.0,
        seed: Optional[int] = None,
        export_as_list: bool = False
) -> Union[np.ndarray, list]:
    """Adds noise to a time series. The length of the time series is preserved
    and the noise is added to each channel independently if `per_channel` is
    True.

    Args:
        array: The input time series (can be also multidimensional).
        loc: Mean of the noise distribution. If a float, the mean is
            constant (all noise values are the sampled to the same mean).
            If a list, the noise added to a series (to each channel
            independently if `per_channel` is True) is sampled from a
            distribution with a mean value that is randomly selected from the
            list. If tuple with two elements, the noise added to a series (to
            each channel independently if `per_channel` is True) is sampled
            from a distribution with a mean value that is randomly selected
            from the interval. The default is 0.0.
        scale: Standard deviation of the noise distribution. If a float, the
            standard deviation is constant (all noise values are the sampled
            to the same standard deviation). If a list, the noise added to a
            series (to each channel independently if `per_channel` is True) is
            sampled from a distribution with a standard deviation that is
            randomly selected from the list. If tuple with two elements, the
            noise added to a series (to each channel independently if
            `per_channel` is True) is sampled from a distribution with a
            standard deviation that is randomly selected from the interval.
            The default is 0.1.
        distribution: The noise distribution. Can be "gaussian", "uniform", or
            "laplace". The default is "gaussian".
        kind: The way the noise is added to the series. Can be "additive" or
            "multiplicative". The default is "additive".
        per_channel: Whether to add noise independently to each channel. The
            default is True.
        normalize: Whether to normalize the noise. If True, each channel of
            the noise is normalized to [0, 1]. The default is True.
        repeats: The number of times to apply the augmentation. The default is
            1.
        prob: The probability of applying the augmentation. The default is 1.0.
        seed: The seed to use for the random number generator. The default is
            None.
        export_as_list: Whether to export the augmented data as a list. The
            default is False.

    Returns:
        np.ndarray or list: The time series data with added noise in np.ndarray
        format or as a list. The list is returned if `export_as_list` is True.
        The np.ndarray has the shape of (n_repeats, window_size (n_samples),
        n_channels).
    """
    from tsaug import AddNoise

    arr = AddNoise(loc=loc, scale=scale, distr=distribution,
                   kind=kind, per_channel=per_channel,
                   normalize=normalize, repeats=repeats,
                   prob=prob, seed=seed).augment(array)

    if repeats > 1 and array.ndim > 1:
        length = array.shape[0]
        arr = arr_splitter(arr, length, repeats)

    if export_as_list:
        return return_listed_augmentations(arr, repeats)
    else:
        return arr


def convolve_ts(
        array: np.ndarray,
        window: Union[str, Tuple, List[Union[str, Tuple]]] = "hann",
        kernel: Union[int, Tuple[int, int], List[int]] = 7,
        per_channel: bool = False,
        repeats: int = 1,
        prob: float = 1.0,
        seed: Optional[int] = None,
        export_as_list: bool = False
) -> Union[np.ndarray, list]:
    """Convolves a time series with a kernel. The length of the time series is
    preserved and the convolution is applied to each channel independently if
    `per_channel` is True.

    Args:
        array: The input time series data in np.ndarray.
        window: The window to use for the convolution. Can be a string
            representing the window type (e.g., "hann") or a tuple or list
            of strings representing the window type for each channel. The
            default is "hann".
        kernel: The size of the kernel. Can be an integer representing the
            kernel size (e.g., 7) or a tuple or list of integers representing
            the kernel size for each channel. The default is 7.
        per_channel: Whether to convolve each channel independently. The
            default is False.
        repeats: The number of times to apply the augmentation. The default is
            1.
        prob: The probability of applying the augmentation. The default is 1.0.
        seed: The seed to use for the random number generator. The default is
            None.
        export_as_list: Whether to export the augmented data as a list. The
            default is False.

    Returns:
        np.ndarray or list: the convolved time series data in np.ndarray
        format or as a list. The list is returned if `export_as_list` is True.
    """
    from tsaug import Convolve
    arr = Convolve(window=window, size=kernel, per_channel=per_channel,
                   repeats=repeats, prob=prob, seed=seed).augment(array)

    if repeats > 1 and array.ndim > 1:
        length = array.shape[0]
        arr = arr_splitter(arr, length, repeats)

    if export_as_list:
        return return_listed_augmentations(arr, repeats)
    else:
        return arr


def crop_ts(
        array: np.ndarray,
        size: Union[int, Tuple[int, int], List[int]],
        resize: Optional[int] = None,
        repeats: int = 1,
        prob: float = 1.0,
        seed: Optional[int] = None,
        export_as_list: bool = False
) -> Union[np.ndarray, list]:
    """Crops a time series. The length of the time series is NOT preserved and
    the cropping is applied to all channels simultaneously.

    Args:
        array: The input time series data in np.ndarray.
        size: The size of the crop. Can be an integer representing the size of
            the crop (e.g., 100) or a tuple or list of integers representing
            the size of the crop for each channel. If the size is an integer,
            the same size is used for all channels. The default is 100.
        resize: The size to resize the crop to. If None, the crop is not
            resized. The default is None.
        repeats: The number of times to apply the augmentation. The default is
            1.
        prob: The probability of applying the augmentation. The default is 1.0.
        seed: The seed to use for the random number generator. The default is
            None.
        export_as_list: Whether to export the augmented data as a list. The
            default is False.

    Returns:
        np.ndarray or list: The cropped time series data in np.ndarray format
        or as a list. The list is returned if `export_as_list` is True.
    """
    from tsaug import Crop
    arr = Crop(size=size, resize=resize, repeats=repeats,
               prob=prob, seed=seed).augment(array)

    if repeats > 1 and array.ndim > 1:
        length = array.shape[0]
        arr = arr_splitter(arr, length, repeats)

    if export_as_list:
        return return_listed_augmentations(arr, repeats)
    else:
        return arr


def drift_ts(
        array: np.ndarray,
        max_drift: Union[float, Tuple[float, float]] = 0.5,
        n_drift_points: Union[int, List[int]] = 3,
        kind: str = "additive",
        per_channel: bool = True,
        normalize: bool = True,
        repeats: int = 1,
        prob: float = 1.0,
        seed: Optional[int] = None,
        export_as_list: bool = False
) -> Union[np.ndarray, list]:
    """Drifts a time series. The length of the time series is preserved and the
    drift is applied to each channel independently if `per_channel` is True.

    Args:
        array: The input time series data in np.ndarray.
        max_drift: The maximum drift. Can be a float representing the maximum
            drift (e.g., 0.5) or a tuple representing the maximum drift for
            each channel. The default is 0.5.
        n_drift_points: The number of drift points. Can be an integer
            representing the number of drift points (e.g., 3) or a list of
            integers representing the number of drift points for each channel.
            The default is 3.
        kind: The kind of drift. Can be "additive" or "multiplicative". The
            default is "additive".
        per_channel: Whether to drift each channel independently. The default is
            True.
        normalize: Whether to normalize the drift. If True, each channel of
            the drift is normalized to [0, 1]. The default is True.
        repeats: The number of times to apply the augmentation. The default is
            1.
        prob: The probability of applying the augmentation. The default is 1.0.
        seed: The seed to use for the random number generator. The default is
            None.
        export_as_list: Whether to export the augmented data as a list. The
            default is False.

    Returns:
        np.ndarray or list: The time series data with added drift in np.ndarray
        format or as a list. The list is returned if `export_as_list` is True.
    """
    from tsaug import Drift
    arr = Drift(max_drift=max_drift, n_drift_points=n_drift_points,
                kind=kind, per_channel=per_channel, normalize=normalize,
                repeats=repeats, prob=prob, seed=seed).augment(array)

    if repeats > 1 and array.ndim > 1:
        length = array.shape[0]
        arr = arr_splitter(arr, length, repeats)

    if export_as_list:
        return return_listed_augmentations(arr, repeats)
    else:
        return arr


def dropouts_ts(
        array: np.ndarray,
        p: Union[float, Tuple[float, float], List[float]] = 0.05,
        size: Union[int, Tuple[int, int], List[int]] = 1,
        fill: Union[str, float] = "ffill",
        per_channel: bool = False,
        repeats: int = 1,
        prob: float = 1.0,
        seed: Optional[int] = None,
        export_as_list: bool = False
) -> Union[np.ndarray, list]:
    """Adds dropouts to a time series. The length of the time series is
    preserved and the dropouts are added to each channel independently if
    `per_channel` is True.

    Args:
        array: The input time series data in np.ndarray.
        p: Probability of the value of a time point to be dropped out. Can be
            a float representing the probability (e.g., 0.05) or a tuple
            representing the probability for each channel. The default is 0.05.
        size: The size of the dropped out units. Can be an integer representing
            the size of the units to be dropped out or a tuple representing
            the size for each channel. The default is 1.
        fill: The way to fill the dropped out units. Can be a) "ffill" (forward
            fill): fill with the last previous value that is not
            dropped, b) "bfill" (backward fill): fill with the first
            next value that is not dropped, c) mean: fill with the mean value
            of this channel in this series, or d) a float representing the
            value to fill the dropped out units with. The default is "ffill".
        per_channel: Whether to add dropouts independently to each channel. The
            default is False.
        repeats: The number of times to apply the augmentation. The default is
            1.
        prob: The probability of applying the augmentation. The default is 1.0.
        seed: The seed to use for the random number generator. The default is
            None.
        export_as_list: Whether to export the augmented data as a list. The
            default is False.

    Returns:
        np.ndarray or list: The time series data with dropouts in np.ndarray
        format or as a list. The list is returned if `export_as_list` is True.
    """
    from tsaug import Dropout
    arr = Dropout(p=p, size=size, fill=fill, per_channel=per_channel,
                  repeats=repeats, prob=prob, seed=seed).augment(array)

    if repeats > 1 and array.ndim > 1:
        length = array.shape[0]
        arr = arr_splitter(arr, length, repeats)

    if export_as_list:
        return return_listed_augmentations(arr, repeats)
    else:
        return arr


def pool_ts(
        array: np.ndarray,
        kind: str = "ave",
        size: Union[int, Tuple[int, int], List[int]] = 2,
        per_channel: bool = False,
        repeats: int = 1,
        prob: float = 1.0,
        seed: Optional[int] = None,
        export_as_list: bool = False
) -> Union[np.ndarray, list]:
    """Reduces the temporal resolution of a time series without changing the
    length. The length of the time series is preserved and the pooling is
    applied to each channel independently if `per_channel` is True.

    Args:
        array: The input time series data in np.ndarray.
        kind: The kind of pooling. Can be "ave" (average pooling), "max"(max
            pooling), or "min" (min pooling). The default is "ave".
        size: The size of the pooling. Can be an integer representing the size
            of the pooling (e.g., 2), a list of integers representing a series
            (for each channel if `per_channel` is True) which is
            pooled with a pooling size sampled from this list of integers
            randomly, or a tuple representing a series that is pooled with a
            size based on the defined interval randomly. The default is 2.
        per_channel: Whether to pool each channel independently. The default is
            False.
        repeats: The number of times to apply the augmentation. The default is
            1.
        prob: The probability of applying the augmentation. The default is 1.0.
        seed: The seed to use for the random number generator. The default is
            None.
        export_as_list: Whether to export the augmented data as a list. The
            default is False.

    Returns:
        np.ndarray or list: The time series data with the reduced temporal
        resolution in np.ndarray format or as a list. The list is returned if
        `export_as_list` is True.
    """
    from tsaug import Pool
    arr = Pool(kind=kind, size=size, per_channel=per_channel,
               repeats=repeats, prob=prob, seed=seed).augment(array)

    if repeats > 1 and array.ndim > 1:
        length = array.shape[0]
        arr = arr_splitter(arr, length, repeats)

    if export_as_list:
        return return_listed_augmentations(arr, repeats)
    else:
        return arr


def quantize_ts(
        array: np.ndarray,
        n_levels: Union[int, Tuple[int, int], List[int]] = 10,
        how: str = "uniform",
        per_channel: bool = False,
        repeats: int = 1,
        prob: float = 1.0,
        seed: Optional[int] = None,
        export_as_list: bool = False
) -> Union[np.ndarray, list]:
    """Quantizes the time series to a level set. The values in a time series
    are rounded to the nearest level in the level set. The length of the time
    series is preserved and the quantization is applied to each channel if
    `per_channel` is True.

    Args:
        array: The input time series data in np.ndarray.
        n_levels: The number of quantization levels. Can be an integer
            representing the number of levels (e.g., 10), a list of integers
            representing a series (for each channel if `per_channel` is True)
            which is quantized with a number of levels sampled from this list
            of integers randomly, or a tuple representing a series that is
            quantized with a number of levels based on the defined interval
            randomly. The default is 10.
        how: The way to quantize the time series. Can be "uniform" (uniform
            quantization) or "quantile" (quantile quantization), or "kmeans"
            (kmeans quantization). The default is "uniform".
        per_channel: Whether to quantize each channel independently. The default
            is False.
        repeats: The number of times to apply the augmentation. The default is
            1.
        prob: The probability of applying the augmentation. The default is 1.0.
        seed: The seed to use for the random number generator. The default is
            None.
        export_as_list: Whether to export the augmented data as a list. The
            default is False.

    Returns:
        np.ndarray or list: The quantized time series data in np.ndarray format
        or as a list. The list is returned if `export_as_list` is True.
    """
    from tsaug import Quantize
    arr = Quantize(n_levels=n_levels, how=how, per_channel=per_channel,
                   repeats=repeats, prob=prob, seed=seed).augment(array)

    if repeats > 1 and array.ndim > 1:
        length = array.shape[0]
        arr = arr_splitter(arr, length, repeats)

    if export_as_list:
        return return_listed_augmentations(arr, repeats)
    else:
        return arr


def resize_ts(
        array: np.ndarray,
        size: int,
        repeats: int = 1,
        prob: float = 1.0,
        seed: Optional[int] = None,
        export_as_list: bool = False
) -> Union[np.ndarray, list]:
    """Changes the temporal resolution of time series. The length of the time
    series is NOT preserved. The resized time series is obtained by linear
    interpolation of the original time series.

    Args:
        array: The input time series data in np.ndarray.
        size: The size target of the resized time series.
        repeats: The number of times to apply the augmentation. The default is
            1.
        prob: The probability of applying the augmentation. The default is 1.0.
        seed: The seed to use for the random number generator. The default is
            None.
        export_as_list: Whether to export the augmented data as a list. The
            default is False.

    Returns:
        np.ndarray or list: The resized time series data in np.ndarray format
        or as a list. The list is returned if `export_as_list` is True.
    """
    from tsaug import Resize
    arr = Resize(size=size, repeats=repeats, prob=prob, seed=seed).augment(
        array)

    if repeats > 1 and array.ndim > 1:
        length = array.shape[0]
        arr = arr_splitter(arr, length, repeats)

    if export_as_list:
        return return_listed_augmentations(arr, repeats)
    else:
        return arr


def reverse_ts(
        array: np.ndarray,
        repeats: int = 1,
        prob: float = 1.0,
        seed: Optional[int] = None,
        export_as_list: bool = False
) -> Union[np.ndarray, list]:
    """Reverses the time series.

    Args:
        array: The input time series data in np.ndarray.
        repeats: The number of times to apply the augmentation. The default is
            1.
        prob: The probability of applying the augmentation. The default is 1.0.
        seed: The seed to use for the random number generator. The default is
            None.
        export_as_list: Whether to export the augmented data as a list. The
            default is False.

    Returns:
        np.ndarray or list: The reversed time series data in np.ndarray format
        or as a list. The list is returned if `export_as_list` is True.
    """
    from tsaug import Reverse
    arr = Reverse(repeats=repeats, prob=prob, seed=seed).augment(array)

    if repeats > 1 and array.ndim > 1:
        length = array.shape[0]
        arr = arr_splitter(arr, length, repeats)

    if export_as_list:
        return return_listed_augmentations(arr, repeats)
    else:
        return arr


def time_warp_ts(
        array: np.ndarray,
        n_speed_change: int = 3,
        max_speed_ratio: Union[float, Tuple[float, float], List[float]] = 3.0,
        repeats: int = 1,
        prob: float = 1.0,
        seed: Optional[int] = None,
        export_as_list: bool = False
) -> Union[np.ndarray, list]:
    """ Changes the speed of the time series. The length of the time series is
    preserved. The time warping is controlled by the number of speed changes
    and the maximal ratio of max/min speed.

    Args:
        array: The input time series data in np.ndarray.
        n_speed_change: The number of speed changes. The default is 3.
        max_speed_ratio: The maximal ratio of max/min speed. Can be a float
            representing the maximal ratio (e.g., 3.0) or a list of floats
            representing a series which is time warped with a maximal ratio
            sampled from this list or a tuple where each series is warpped with
            a ratio that is randomly sampled from the interval. The default is
            3.0.
        repeats: The number of times to apply the augmentation. The default is
            1.
        prob: The probability of applying the augmentation. The default is 1.0.
        seed: The seed to use for the random number generator. The default is
            None.
        export_as_list: Whether to export the augmented data as a list. The
            default is False.

    Returns:
        np.ndarray or list: The time warped time series data in np.ndarray
        format or as a list. The list is returned if `export_as_list` is True.
    """
    from tsaug import TimeWarp
    arr = TimeWarp(n_speed_change=n_speed_change,
                   max_speed_ratio=max_speed_ratio,
                   repeats=repeats, prob=prob, seed=seed).augment(array)

    if repeats > 1 and array.ndim > 1:
        length = array.shape[0]
        arr = arr_splitter(arr, length, repeats)

    if export_as_list:
        return return_listed_augmentations(arr, repeats)
    else:
        return arr


def arr_splitter(
        array: np.ndarray,
        instance_samples_length: int,
        repeats: int = None
) -> np.ndarray:
    """Unpacks the array into a list of arrays that occurred from
     the augmentation process, and stacks them together in the form
     of a 3D array (instances, samples, axes)

    Args:
        array: The array with the augmented instances to be split.
        instance_samples_length: The length of the original array.
        repeats: The number of times the augmentation process
            took place.

    Returns:
        np.ndarray: The augmented instances in the form of a 3D array:
            (instances, samples, axes).
    """
    index_start = 0
    arrays_list = []
    for i in range(repeats):
        arrays_list.append(
            array[index_start: index_start + instance_samples_length])
        index_start += instance_samples_length
    return np.stack(arrays_list)


def return_listed_augmentations(
        array: np.ndarray,
        repeats: int,
) -> list:
    """Returns the augmented time series data in a list.

    Args:
        array: The input time series data in np.ndarray.
        repeats: The number or new augmented instances.

    Returns:
        list: A list where each item includes the augmented time series data
        in np.ndarray format.
    """
    augmented_data = []

    if repeats > 1:
        augmented_data = [row for row in array]
    else:
        augmented_data.append(augmented_data)

    return augmented_data


def time_stretch_ts(
        y: np.ndarray,
        *,
        rate: float,
        **kwargs: Any
) -> np.ndarray:
    # The functionality in this implementation are basically derived from
    # librosa v0.10.1:
    # https://github.com/librosa/librosa/blob/main/librosa/effects.py

    if rate <= 0:
        raise ValueError("rate must be a positive number")

    # Construct the short-term Fourier transform (STFT)
    stft_item = stft(y, **kwargs)

    # Stretch by phase vocoding
    stft_stretch = phase_vocoder(
        stft_item,
        rate=rate,
        hop_length=kwargs.get("hop_length", None),
        n_fft=kwargs.get("n_fft", None),
    )

    # Predict the length of y_stretch
    len_stretch = int(round(y.shape[-1] / rate))

    # Invert the STFT
    y_stretch = istft(stft_stretch, dtype=y.dtype, length=len_stretch,
                      **kwargs)

    return y_stretch


def pitch_shift_ts(
    y: np.ndarray,
    *,
    sr: float,
    n_steps: float,
    bins_per_octave: int = 12,
    res_type: str = "soxr_hq",
    scale: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    # The functionality in this implementation are basically derived from
    # librosa v0.10.1:
    # https://github.com/librosa/librosa/blob/main/librosa/effects.py

    if not is_positive_int(bins_per_octave):
        raise ValueError(
            f"bins_per_octave={bins_per_octave} must be a positive integer."
        )

    rate = 2.0 ** (-float(n_steps) / bins_per_octave)

    # Stretch in time, then resample
    y_shift = resample(
        time_stretch_ts(y, rate=rate, **kwargs),
        orig_sr=float(sr) / rate,
        target_sr=sr,
        res_type=res_type,
        scale=scale,
    )

    # Crop to the same dimension as the input
    return fix_length(y_shift, size=y.shape[-1])
