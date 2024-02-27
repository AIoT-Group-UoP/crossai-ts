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
