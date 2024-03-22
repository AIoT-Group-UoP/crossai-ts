import numpy as np


def get_non_overlap_probas(
        probabilities: np.ndarray,
        overlap_percentage: float
) -> np.ndarray:
    """Extracts probabilities corresponding to the non-overlapping part of
    each window (of the time series instances), based on the specified overlap
    percentage. This function is useful in scenarios where predictions are made
    on overlapping window segments of a time series, and there is a need to
    distill probabilities to represent non-overlapping segments for analysis
    or visualization.

    The function reduces the input matrix by selecting every N-th row, where N
    is determined by the overlap percentage. For example, with a 50% overlap,
    every second row is selected, effectively halving the dataset to represent
    only the non-overlapping segments.

    Args:
        probabilities: A 2D numpy array of prediction probabilities, where each
                       row represents an instance (or window) and each column
                       represents a class probability.
        overlap_percentage: The percentage of overlap between consecutive
                            windows, expressed as a decimal between 0 and 1.
                            For instance, a 50% overlap is represented as 0.5.

    Returns:
        np.ndarray: A 2D numpy array containing the probabilities for
                    non-overlapping parts of the windows. The shape of the
                    output array is (n_non_overlap_instances, n_classes),
                    where `n_non_overlap_instances` is less than or equal
                    to `n_instances` from the input, depending on the overlap
                    percentage.

    Example:
        If `probabilities` is an array with 100 rows (instances) and there is
        a 50% overlap between windows, the function will return a new array
        with 50 rows, each representing the probability of a class for
        non-overlapping segments.
    """
    # Calculate the selection interval (step) based on the overlap percentage
    step = int(1 / (1 - overlap_percentage))

    # Select rows from the probability matrix at intervals defined by the step
    non_overlap_probs = probabilities[::step]

    return non_overlap_probs


def apply_probability_threshold(
        interpolated_probs: np.ndarray,
        threshold: float
) -> np.ndarray:
    """Applies a probability threshold to the interpolated probabilities
    matrix. Points exceeding the threshold are left unchanged, while
    others are set to 0.

    Args:
        interpolated_probs: 2D array of interpolated probabilities, where each
                            column represents a class.
        threshold: Probability threshold. Points below this value are set to 0.

    Returns:
        numpy.ndarray: The modified interpolated probabilities matrix with the
                       same shape.
    """
    # Apply thresholding
    modified_probs = np.where(
        interpolated_probs > threshold,
        interpolated_probs, 0
    )

    return modified_probs


def apply_duration_threshold(
        interpolated_probs: np.ndarray,
        sampling_rate: int,
        duration_threshold: float
) -> np.ndarray:
    """Applies a duration threshold to the interpolated probabilities.
    Any continuous segments below the duration threshold are set to 0.

    Args:
        interpolated_probs: Interpolated probabilities, where rows correspond
                            to time steps.
        sampling_rate: The number of samples per second in the time series.
        duration_threshold: The duration threshold in seconds. Continuous
                            segments below this duration will be zeroed out.

    Returns:
        np.ndarray: The modified interpolated probabilities after applying
                    the duration threshold.
    """
    # Convert duration from seconds to samples
    duration_samples = int(sampling_rate * duration_threshold)
    n_instances, n_classes = interpolated_probs.shape
    modified_probs = np.zeros_like(interpolated_probs)

    # Iterate over each class
    for i in range(n_classes):
        class_probs = interpolated_probs[:, i]
        # Create a boolean array indicating where the
        # class probability is above zero
        is_above_zero = class_probs > 0
        # Find the indices where the above-zero segments start and end
        above_zero_diff = np.diff(is_above_zero.astype(int))
        # +1 to correct the diff offset
        segment_starts = np.where(above_zero_diff == 1)[0] + 1
        segment_ends = np.where(above_zero_diff == -1)[0] + 1

        if is_above_zero[0]:
            segment_starts = np.insert(segment_starts, 0, 0)
        if is_above_zero[-1]:
            segment_ends = np.append(segment_ends, n_instances)

        # Filter segments by duration and only keep those
        # that meet the duration threshold
        for start, end in zip(segment_starts, segment_ends):
            if end - start >= duration_samples:
                modified_probs[start:end, i] = class_probs[start:end]

    return modified_probs


def get_continuous_events(threshold_probas: np.ndarray) -> list[tuple]:

    significant_segments = []

    # Iterate over each class to find significant segments
    for class_idx in range(threshold_probas.shape[1]):
        class_probs = threshold_probas[:, class_idx]
        is_above_zero = (class_probs > 0).astype(int)
        above_zero_diff = np.diff(is_above_zero, prepend=0, append=0)
        segment_starts = np.where(above_zero_diff == 1)[0]
        segment_ends = np.where(above_zero_diff == -1)[0]

        # Identify segments that meet the duration threshold
        for start, end in zip(segment_starts, segment_ends):
            # -1 to make end index inclusive
            significant_segments.append((start, end - 1, class_idx))

    return significant_segments
