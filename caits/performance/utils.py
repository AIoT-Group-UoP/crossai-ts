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
    back to Keras `predict`, then to direct callable invocation for TF
    SavedModel wrapper functions.

    Args:
        model: The machine learning model to be used for predictions. This can
               be either a TensorFlow model (Keras or SavedModel), or a
               scikit-learn model. The function attempts to use `predict_proba`
               for probabilistic outcomes first; if not available, it falls
               back to Keras `predict`, then to direct callable invocation
               for TF SavedModel wrapper functions.
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
    
    def _extract_probas(output) -> np.ndarray:
        """Normalizes a model output into a NumPy probability matrix regardless
        of the raw return type (dict, tuple/list, Tensor, or ndarray).

        Args:
            output: Raw prediction output from a model call.

        Returns:
            A NumPy ndarray of prediction probabilities.

        Raises:
            ValueError: If a dict output contains more than one key and the
                        correct one cannot be determined automatically.
        """
        if isinstance(output, dict):
            if len(output) == 1:
                output = next(iter(output.values()))
            else:
                raise ValueError(
                    f"Model returned a dict with multiple keys {list(output.keys())}. "
                    f"Cannot automatically determine which contains probabilities."
                )

        if isinstance(output, (list, tuple)):
            output = output[0]

        if isinstance(output, tf.Tensor):
            return output.numpy()

        if isinstance(output, np.ndarray):
            return output

        raise TypeError(
            f"Unexpected prediction output type: {type(output)}. "
            f"Expected tf.Tensor, np.ndarray, dict, or tuple."
        )

    predictions = []

    for _ in range(repeats):
        if hasattr(model, "predict_proba"):
            # scikit-learn estimators exposing probabilistic outputs
            preds = model.predict_proba(X)

        elif hasattr(model, "predict") and callable(getattr(model, "predict")):
            # Classic Keras model loaded via tf.keras.models.load_model()
            preds = model.predict(X)

        elif callable(model):
            # TF SavedModel wrapper function loaded via tf.saved_model.load()
            # type: tensorflow.python.saved_model.load._WrapperFunction
            X_tensor = tf.cast(
                tf.convert_to_tensor(X), dtype=tf.float32
            ) if not isinstance(X, tf.Tensor) else tf.cast(X, dtype=tf.float32)
            preds = model(X_tensor)

        else:
            raise TypeError(
                f"Unsupported model type: {type(model)}. Expected a scikit-learn "
                f"estimator, a Keras Model, or a TF SavedModel callable."
            )

        preds = _extract_probas(preds)
        predictions.append(preds)

    all_predictions = np.stack(predictions, axis=0)

    return all_predictions


def interpolate_probabilities(
    probabilities: np.ndarray,
    sr: int,
    ws: float,
    overlap_percentage: float,
    interp_choice: int = 2,
) -> np.ndarray:
    """Interpolates each column of the prediction probability
    matrix using cubic spline.

    Args:
        probabilities: Prediction probability matrix (windows x classes).
        sr: Sampling rate.
        ws: Window size in seconds.
        overlap_percentage: Overlap percentage between windows.
        interp_choice: Choice for interpolation points (1: start, 2: middle, 3: end).

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

    # Calculate starting and ending indices for each non-overlapping segment
    start_idx = np.arange(n_instances) * non_op_step
    end_idx = start_idx + non_op_step

    # Calculate interpolation points based on choice
    if interp_choice == 1:
        interp_idx = start_idx
    elif interp_choice == 2:
        interp_idx = (start_idx + end_idx) // 2
    elif interp_choice == 3:
        interp_idx = end_idx
    else:
        raise ValueError("Invalid interp_choice. Choose 1 (start), 2 (middle), or 3 (end).")

    # Create an array for interpolated indices
    final_end_idx = end_idx[-1]
    interp_indices = np.arange(0, final_end_idx)

    # Interpolated probability matrix
    interpolated_probabilities = np.zeros((final_end_idx, num_classes))

    for i in range(num_classes):
        # Create a cubic spline interpolator
        spline = spi.CubicSpline(interp_idx, probabilities[:, i])
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
