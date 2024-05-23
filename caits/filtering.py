from typing import List, Union

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import butter, filtfilt, medfilt, sosfilt, sosfilt_zi, sosfiltfilt


def filter_median_simple(array: np.ndarray, kernel_size: int = None) -> np.ndarray:
    """Performs a median filter on an N-dimensional array.

    Args:
        array (array_like): The input signal to be filtered.
        kernel_size (int): The size of the median filter kernel.

    Returns:
        array_like: The filtered signal.

    Examples:
        >>> import numpy as np
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> filtered_signal = filter_median_simple(array, kernel_size=3)
    """
    filtered_signal = medfilt(array, kernel_size)
    return filtered_signal


def filter_median_gen(array, window_size, output=None, mode="reflect", cval=0.0, origin=0) -> np.ndarray:
    """Calculates a multidimensional median filter. This is more general
        function than median_simple, and thus, has a more efficient
        implementation of a median filter and therefore runs much faster.

    Args:
        array (array_like): The input signal to be filtered.
        window_size (int or tuple of ints): The size of the sliding window for
            median calculation.
        output: The array in which to place the output, or the dtype of the
            returned array.
        mode: The mode parameter determines how the input array is extended
            when the filter overlaps a border.
        cval: Value to fill past edges of input if mode is 'constant'. Default
            is 0.0.
        origin: The origin parameter controls the placement of the filter.
            Default is 0.

    Returns:
        array_like: The filtered signal.

    Examples:
        >>> import numpy as np
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> filtered_signal = filter_median_gen(array, window_size=3)
    """
    filtered_signal = median_filter(array, size=window_size, output=output, mode=mode, cval=cval, origin=origin)
    return filtered_signal


def filter_butterworth(
    array: np.ndarray,
    fs: float,
    filter_type: str = "lowpass",
    cutoff_freq: Union[float, tuple, None] = None,
    order: int = 5,
    analog: bool = False,
    method: str = "filtfilt",
    zi_enable: bool = False,
    axis: int = 0,
    **kwargs,
) -> np.ndarray:
    """Applies a Butterworth filter to a signal.

        Notes: For now, 'ba' and 'sos' types of output are supported: ‘ba’ is
        for backwards compatibility (filtfilt), but ‘sos’ should be used for
        general-purpose filtering, which is applied by default with the
        sosfiltfilt and sosfilt methods.

        Implementation of the Butterworth filter has been applied by using
        the scipy.signal.butter and the following methods:
            * scipy.signal.filtfilt
            * scipy.signal.sosfilt
            * scipy.signal.sosfiltfilt
        For further information and argument modifications based on the filter
        methods, please refer to the documentation of the Scipy and use the
        **kwargs to pass the arguments.

    Args:
        array (array_like): Input signal.
        fs (float): Sampling frequency of the signal.
        filter_type (str, optional): Type of the filter ('lowpass', 'highpass',
            'bandpass', 'bandstop'). Defaults to 'lowpass'.
        cutoff_freq (float or tuple): Cutoff frequency(-ies) of the filter.
            For lowpass and highpass filters, it should be a single frequency.
            For bandpass and bandstop filters, it should be a tuple of two
            frequencies.
        order (int, optional): Order of the Butterworth filter. Defaults to 5.
        analog (bool, optional): If True, return the filter coefficients for an
            analog filter. Defaults to False.
        method (str, optional): Method of applying the filter ('filtfilt',
            'sosfilt', 'sosfiltfilt'). Defaults to 'filtfilt'.
        zi_enable (bool, optional): If True, return the initial conditions for
            the filter. Defaults to False. It is used only for 'sosfilt'
            method.
        axis (int, optional): The axis along which to apply the filter.
            Defaults to 0 for the CrossAI-TS data handling in the transformers,
            as each column of a Dataframe or NumPy array is in the form of
            (window_size, 1).

    Returns:
        array_like: Filtered signal.
    """
    # Check if filter type is valid
    if filter_type not in ["lowpass", "highpass", "bandpass", "bandstop"]:
        raise ValueError(
            "Invalid filter type provided. Please choose from " "'lowpass', 'highpass', 'bandpass', or 'bandstop'."
        )

    nyquist_freq = 0.5 * fs
    # Normalize cutoff frequency(-ies)
    if isinstance(cutoff_freq, tuple):
        normalized_cutoff_freq = (cutoff_freq[0] / nyquist_freq, cutoff_freq[1] / nyquist_freq)
    else:
        normalized_cutoff_freq = cutoff_freq / nyquist_freq

    # Create Butterworth filter coefficients
    if method == "filtfilt":
        b, a = butter(order, normalized_cutoff_freq, btype=filter_type, analog=analog, output="ba")
        return filtfilt(b, a, array, axis=axis, **kwargs)
    elif method == "sosfilt" or method == "sosfiltfilt":
        sos = butter(order, normalized_cutoff_freq, btype=filter_type, analog=False, output="sos")
        if method == "sosfilt":
            if zi_enable:
                tmp = sosfilt_zi(sos)
                print("initial shape", tmp.shape)
                if len(tmp.shape) < 3:
                    zi = tmp[:, :, np.newaxis]
                    print("if shape < 3, transform to: ", zi.shape)
                else:
                    zi = tmp
                    print("else zi shape", tmp.shape)
                # return the filtered signal and the final filter delay
                return sosfilt(sos, array, axis=axis, zi=zi)[0]
            else:
                return sosfilt(sos, array, axis=axis, **kwargs)
        elif method == "sosfiltfilt":
            return sosfiltfilt(sos, array, axis=axis, **kwargs)
        else:
            raise ValueError("Invalid method provided. Please choose from " "'sosfilt' or 'sosfiltfilt'.")
    else:
        raise ValueError("Invalid method provided. Please choose from " "'filtfilt', 'sosfilt', or 'sosfiltfilt'.")


def filter_gaussian(
    array: np.ndarray,
    sigma: Union[float, List[float]] = 1,
    order: Union[int, List[int]] = 0,
    output=None,
    mode="reflect",
    cval=0.0,
    truncate=4.0,
) -> np.ndarray:
    """Applies a Gaussian filter to a signal using SciPy.

    Args:
        array (array_like): The input signal to be filtered.
        sigma (scalar or sequence of scalars, optional): The standard
            deviation(s) of the Gaussian filter along each axis. Default is 1.
        order (int or sequence of ints, optional): The order of the filter
            along each axis. Default is 0.
        output (array, optional): The array in which to place the output.
            Default is None.
        mode ({'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional):
            The mode parameter determines how the input array is extended
            beyond its boundaries. Default is 'reflect'.
        cval (scalar, optional): Value to fill past edges of input if mode is
            'constant'. Default is 0.0.
        truncate (scalar, optional): Truncate the filter at this many standard
            deviations. Default is 4.0.

    Returns:
        array_like: The filtered signal.

    Examples:
        >>> import numpy as np
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> filtered_signal = filter_gaussian(signal, sigma=1)
    """
    return gaussian_filter(array, sigma=sigma, order=order, output=output, mode=mode, cval=cval, truncate=truncate)
