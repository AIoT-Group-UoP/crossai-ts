import numpy as np


def extract_non_overlap_probas(
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
