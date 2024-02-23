import numpy as np
from scipy.interpolate import interp1d
from typing import Optional
from tensorflow.types.experimental import TensorLike


def gen_pred_probs(
    model,
    X: TensorLike,
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

    try:
        # Attempt to use predict_proba for probabilistic outcomes
        predictions = [model.predict_proba(X) for s in range(repeats)]
    except AttributeError:
        # Fallback to using predict for direct predictions
        predictions = [model.predict(X, verbose=0) for s in range(repeats)]

    # Stack predictions along a new dimension for repeated predictions
    all_predictions = np.stack(predictions, axis=0)

    return all_predictions


def compute_predict_trust_metrics(
        probabilities: TensorLike,
        compute: str = "all"
) -> dict:
    """Analyzes prediction probabilities to assess model trustworthiness
    and training adequacy. This function computes statistics from prediction
    probabilities, assuming probabilities have the shape
    (n_repeats, n_instances, n_classes). The analysis includes class
    predictions, raw probabilities, and metrics like mean, standard deviation,
    variance, and entropy across multiple prediction repeats. High variability
    (e.g., high standard deviation) in the prediction probabilities suggests
    less reliability in the model's predictions, indicating areas where the
    model may require further training or adjustment.

    Args:
        probabilities: A 3D array of shape (n_repeats, n_instances, n_classes)
                       containing prediction probabilities.
        compute: Specifies the types of results to compute. Options include:
                 "class" for class predictions, "probas" for returning the raw
                 probabilities, "mean_pred" for the  mean probabilities across
                 repeats, "std" for standard deviation, "variance" for
                 variance, "entropy" for entropy, and "all" for all metrics.

    Returns:
        A dictionary containing the computed results according to the `compute`
        option.
    """
    results = {}

    # Compute class predictions based on the mean probabilities across repeats
    if compute in ["class", "all"]:
        mean_probs = np.mean(probabilities, axis=0)
        results["class"] = np.argmax(mean_probs, axis=1)

    # Include raw probabilities if requested
    if compute in ["probas", "all"]:
        results["probas"] = probabilities

    # Compute mean prediction probabilities across repeats for each class
    if compute in ["mean_pred", "all"]:
        results["mean_pred"] = np.mean(probabilities, axis=0)

    # Compute standard deviation across repeats for each class
    if compute in ["std", "all"]:
        results["std"] = np.std(probabilities, axis=0)

    # Compute variance across repeats for each class
    if compute in ["variance", "all"]:
        results["variance"] = np.var(probabilities, axis=0)

    # Compute entropy across repeats for each class
    if compute in ["entropy", "all"]:
        # Ensuring numerical stability by adding
        # epsilon to probabilities before log
        epsilon = np.finfo(float).eps
        entropy = -np.sum(
            probabilities * np.log(probabilities + epsilon), axis=2
            ) / np.log(2)
        results["entropy"] = np.mean(entropy, axis=0)

    return results


def interpolate_probas(
        probabilities: np.ndarray,
        sampling_rate: int,
        Ws: float,
        n_points: int = None,
        kind: Optional[str] = "cubic",
        clamp: Optional[bool] = True
) -> np.ndarray:
    """Interpolates prediction probabilities for a smoother representation
    over time or samples.

    This function applies cubic interpolation to a matrix of class
    probabilities associated with instances or window segments in a
    time series, allowing for the visualization of smoothed probability
    transitions across classes.

    Args:
        probabilities: A 2D array where each row represents an
            instance (or window) and each column a class probability.
        sampling_rate: The sampling rate of the time series data, used to
            scale the interpolation points in relation to the original data.
        Ws: The window size in seconds, defining the temporal span of each
            instance or window in the original time series.
        n_points: The number of points for the interpolated output.
            If None, calculated as `n_instances * sampling_rate * Ws`.
        kind: The type of interpolation. Deafults to `linear`.
        clamp: Flag to clamp interpolated values between range [0, 1].
               Defaults to False.

    Returns:
        np.ndarray: A 2D array of shape `(n_points, n_classes)` containing the
            interpolated probabilities for each class. This array offers a
            continuous and smooth representation of class probabilities
            over time.
    """
    # Determine the shape of the input probability matrix
    n_instances, n_classes = probabilities.shape

    # Automatically calculate the number of interpolation points
    # if not provided. Goal here is the number of points to match the
    # original length of the time series instance
    if n_points is None:
        n_points = int(n_instances * sampling_rate * Ws)

    # Initialize the output array for interpolated probabilities
    interpolated_probabilities = np.zeros((n_points, n_classes))

    # Prepare the original and target x-values for interpolation
    x_original = np.arange(n_instances)
    x_interpolated = np.linspace(0, n_instances - 1, n_points)

    # Perform cubic interpolation for each class
    for i in range(n_classes):
        # Create the interpolator function for the current class
        interpolator = interp1d(x_original, probabilities[:, i],
                                kind=kind, fill_value='extrapolate')

        # Apply interpolation and store the results
        interpolated_probs = interpolator(x_interpolated)

        if clamp:
            # Clamp the interpolated values to [0, 1]
            interpolated_probs = np.clip(interpolated_probs, 0, 1)

        # Apply interpolation and store the results
        interpolated_probabilities[:, i] = interpolated_probs

    return interpolated_probabilities


def intersection_over_union(
        predicted_segment: tuple,
        ground_truth_segment: tuple
) -> float:
    """Calculates the Intersection over Union (IoU) for a single pair
    of predicted and ground truth segments.

    The IoU is a measure of how much the predicted segment overlaps with the
    ground truth segment. It is defined as the size of the intersection
    divided by the size of the union of the two segments.

    Args:
        predicted_segment: A tuple (start, end) representing the start and end
                           indices of the predicted segment.
        ground_truth_segment: A tuple (start, end) representing the start and
                              end indices of the ground truth segment.

    Returns:
        float: The IoU value, which is a float between 0 and 1.
               A higher IoU indicates a greater degree of overlap.
    """
    start_pred, end_pred = predicted_segment
    start_gt, end_gt = ground_truth_segment

    # Calculate intersection
    intersection_start = max(start_pred, start_gt)
    intersection_end = min(end_pred, end_gt)
    intersection = max(0, intersection_end - intersection_start)

    # Calculate union
    union = (end_pred - start_pred) + (end_gt - start_gt) - intersection

    # Compute IoU
    iou = intersection / union if union > 0 else 0
    return iou
