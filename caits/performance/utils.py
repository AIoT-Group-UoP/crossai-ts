from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator
import tensorflow as tf
from tensorflow.keras import Model


def generate_probabilities(
    model: Union[BaseEstimator, Model],
    X: Union[np.ndarray, tf.Tensor],
    repeats: int = 1
) -> np.ndarray:
    """Executes inference using a TensorFlow or scikit-learn model on provided
    data, optionally repeating the process multiple times. This function is
    designed to accommodate models with different prediction interfaces,
    attempting to use `predict_proba` for probabilistic outcomes and falling
    back to `predict` for direct predictions if the former is not available.

    Args:
        model: The machine learning model to be used for predictions. This can
               be either a TensorFlow model or a scikit-learn model. The
               function attempts to use `predict_proba` for probabilistic
               outcomes first; if not available, it falls back to `predict`
               for direct predictions.
        X: The dataset on which predictions are to be made. Expected to be
              formatted appropriately for the model's input requirements.
        repeats: The number of times the prediction process should be repeated.
                 This is useful for assessing model consistency and uncertainty
                 in predictions.

    Returns:
        An array of predictions made by the model. For repeated predictions,
        the results are stacked along a new dimension, allowing for further
        analysis of prediction consistency or uncertainty.
    """
    if hasattr(model, "predict_proba"):
        # Attempt to use predict_proba for probabilistic outcomes
        predictions = [model.predict_proba(X) for s in range(repeats)]
    else:
        # Fallback to using predict for direct predictions
        predictions = [model(X) for s in range(repeats)]

    # Stack predictions along a new dimension for repeated predictions
    all_predictions = np.stack(predictions, axis=0)

    return all_predictions


def interpolate_probas(
    probabilities: np.ndarray,
    n_points: int,
    with_overlaps: bool = False,
    overlap_percentage: Optional[float] = None,
    kind: Optional[Literal["cubic", "linear"]] = "cubic",
    clamp: bool = True,
) -> np.ndarray:
    """Interpolates prediction probabilities for smoother visualization.

    Args:
        probabilities: A 2D array of shape (n_instances, n_classes) containing
                       class probabilities for each instance/window.
        n_points: The desired number of points in the interpolated output.
        with_overlaps: If True, performs interpolation for each channel to the 
                       given probability matrix. If False, keeps only the instances
                       predicted with no overlap. Defaults to False.
        overlap_percentage: The percentage of overlap between windows, required
                       when `with_overlaps` is True.
        kind: The type of interpolation to use (e.g., "linear", "cubic").
                       Defaults to "cubic".
        clamp: Whether to clamp the interpolated values between 0 and 1.
                       Defaults to True.

    Returns:
        A 2D array of shape (n_points, n_classes) containing the interpolated
        probabilities for each class.

    Raises:
        ValueError: If `with_overlaps` is True and `overlap_percentage` is None.
    """

    if with_overlaps:
        n_instances, n_classes = probabilities.shape
    else:
        if overlap_percentage is None:
            raise ValueError("Overlap percentage must be provided.")
        probabilities = _get_non_overlap_probas(probabilities, overlap_percentage)
        n_instances, n_classes = probabilities.shape

    interpolated_probabilities = np.zeros((n_points, n_classes))

    # Prepare the original and target x-values for interpolation
    x_original = np.arange(n_instances)
    x_interpolated = np.linspace(0, n_instances - 1, n_points)

    # Perform cubic interpolation for each class
    for i in range(n_classes):
        # Create the interpolator function for the current class
        interpolator = interp1d(x_original, probabilities[:, i], kind=kind, fill_value="extrapolate")

        # Apply interpolation and store the results
        interpolated_probs = interpolator(x_interpolated)

        if clamp:
            # Clamp the interpolated values to [0, 1]
            interpolated_probs = np.clip(interpolated_probs, 0, 1)

        # Apply interpolation and store the results
        interpolated_probabilities[:, i] = interpolated_probs

    return interpolated_probabilities


def _get_non_overlap_probas(
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


def get_gt_events_from_dict(
    events: Dict[str, Any],
    class_names: List[str],
    sr: Optional[int] = None,
) -> Dict[str, List[Tuple[int, int, int]]]:
    """Extracts and optionally converts start and end intervals from a given
    JSON structure to samples. The output is a dictionary keyed by the original
    keys from `events`, with values being lists of tuples. Each tuple
    represents an interval, including start and end points, and the label.

    Args:
        events: The JSON data containing the intervals.
        sr: Sampling rate to convert start and end from time to samples.
            If None, the values are used as is.

    Returns:
        dict: A dictionary where each key corresponds to the original keys
              in `events`, and each value is a list of tuples. Each tuple
              contains the start, end, and the label of an interval.
    """
    intervals = {
        key: [
            (
                int(item["start"] * sr) if item.get("type") == "time" else item["start"],
                int(item["end"] * sr) if item.get("type") == "time" else item["end"],
                list(class_names).index(item["label"]),
            )
            for item in items
        ]
        for key, items in events.items()
    }

    return intervals
