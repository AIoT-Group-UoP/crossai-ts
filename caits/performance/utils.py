from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.interpolate as spi
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


def interpolate_probabilities(
    probabilities: np.ndarray,
    sr: int,
    ws: float,
    overlap_percentage: float
) -> np.ndarray:
    """Interpolates each column of the prediction probability
    matrix using cubic spline.

    Args:
        probabilities: Prediction probability matrix (windows x classes).
        sr: Sampling rate.
        ws: Window size in seconds.
        overlap_percentage (float): Overlap percentage between windows.

    Returns:
        np.ndarray: Interpolated probabilities matrix.
    """
    # Window size in samples
    ws_samples = int(ws * sr)
    # Overlap size in samples
    op_samples = int(ws_samples * overlap_percentage)
    # Non-overlapping segment in samples
    non_op_step = ws_samples - op_samples
    # Number of instances, classes
    n_instances, num_classes = probabilities.shape

    # Original indices
    start_idx = np.arange(n_instances) * non_op_step
    end_idx = start_idx + non_op_step

    # Calculate the end of the last window probability
    final_end_idx = end_idx[-1]

    # Create an array for interpolated indices
    interp_indices = np.arange(start_idx[0], final_end_idx)

    # Interpolated probability matrix
    interpolated_probabilities = np.zeros((final_end_idx, num_classes))

    for i in range(num_classes):
        # Create a cubic spline interpolator with clamped boundary conditions
        spline = spi.CubicSpline(start_idx, probabilities[:, i], bc_type="natural")
        # Interpolate data points
        interpolated_probabilities[:, i] = spline(interp_indices)

    return interpolated_probabilities


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
