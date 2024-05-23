from typing import Any, List, Optional, Tuple, Union

import numpy as np

TensorLike = Union[np.ndarray, Any]


def compute_class(predictions: TensorLike) -> np.ndarray:
    return np.argmax(np.mean(predictions, axis=0), axis=1)


def compute_entropy(predictions: TensorLike) -> np.ndarray:
    epsilon: np.floating = np.finfo(float).eps
    return np.mean(-np.sum(predictions * np.log(predictions + epsilon), axis=2) / np.log(2), axis=0)


# Global dictionary of metric functions
_stats_functions = {
    "class": compute_class,
    "probas": lambda x: x,
    "mean_pred": lambda x: np.mean(x, axis=0),
    "std": lambda x: np.std(x, axis=0),
    "variance": lambda x: np.var(x, axis=0),
    "entropy": compute_entropy,
}


def prediction_statistics(probabilities: TensorLike, stats: Optional[Union[List[str], str]] = "all") -> dict:
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
        stats: Specifies the types of results to compute. Options include:
                 "class" for class predictions, "probas" for returning the raw
                 probabilities, "mean_pred" for the  mean probabilities across
                 repeats, "std" for standard deviation, "variance" for
                 variance, "entropy" for entropy, and "all" for all metrics.

    Returns:
        A dictionary containing the computed statistics from the prediction
        probalities according to the `stats` option.
    """
    metrics = ""
    if stats == "all":
        metrics = list(_stats_functions.keys())

    if isinstance(metrics, str):
        metrics = [metrics]

    results = {}
    for metric in metrics:
        if metric in _stats_functions:
            results[metric] = _stats_functions[metric](probabilities)
        else:
            print(f"Metric '{metric}' not recognized.")

    return results


def intersection_over_union(predicted_segment: Tuple[int, int], ground_truth_segment: Tuple[int, int]) -> float:
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


def detection_ratio(C: int, D: int, S: int) -> float:
    """Calculates the Detection Ratio for event recognition.

    Args:
        C: Number of correctly identified events.
        D: Number of deletions (missed events).
        S: Number of substitutions (misclassified events).

    Returns:
        float: The Detection Ratio, a measure of how well the
        system recognizes events.
    """
    return C / (D + C + S) if (D + C + S) > 0 else 0


def reliability(C: int, IN: int) -> float:
    """Calculates the Reliability for event recognition.

    Args:
        C: Number of correctly identified events.
        I: Number of insertions (false positives).

    Returns:
        float: The Reliability, a measure of the system's ability
        to detect events without false positives.
    """
    return C / (C + IN) if (C + IN) > 0 else 0


def erer(D: int, IN: int, S: int, C: int) -> float:
    """Calculates the Event Recognition Error Rate (ERER)
    for event recognition.

    Args:
        D: Number of deletions (missed events).
        I: Number of insertions (false positives).
        S: Number of substitutions (misclassified events).
        C: Number of correctly identified events.

    Returns:
        float: The ERER, indicating the overall error rate of the
        system in recognizing events.
    """
    return (D + IN + S) / (D + C + S) if (D + C + S) > 0 else 0
