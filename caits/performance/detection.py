from typing import List, Tuple

import numpy as np

from .metrics import intersection_over_union


def apply_probability_threshold(interpolated_probs: np.ndarray, threshold: float) -> np.ndarray:
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
    modified_probs = np.where(interpolated_probs > threshold, interpolated_probs, 0)

    return modified_probs


def apply_duration_threshold(
    interpolated_probs: np.ndarray,
    sampling_rate: int,
    duration_threshold: float,
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


def get_continuous_events(threshold_probas: np.ndarray) -> List[Tuple[int, int, int]]:
    significant_segments = []

    # Iterate over each class to find significant segments
    for class_idx in range(threshold_probas.shape[1]):
        class_probs = threshold_probas[:, class_idx]
        is_above_zero: np.ndarray = (class_probs > 0).astype(int)
        above_zero_diff = np.diff(is_above_zero, prepend=0, append=0)
        segment_starts = np.where(above_zero_diff == 1)[0]
        segment_ends = np.where(above_zero_diff == -1)[0]

        # Identify segments that meet the duration threshold
        for start, end in zip(segment_starts, segment_ends):
            # -1 to make end index inclusive
            significant_segments.append((start, end - 1, class_idx))

    return significant_segments


def classify_events(
    predicted_events: List[Tuple[int, int, int]],
    ground_truth_events: List[Tuple[float, float, int]],
    IoU_th: float,
) -> Tuple[int, int, int, int]:
    """Classifies predicted events into Insertions, Correct identifications,
    Substitutions, and Deletions based on IoU score, class labels.

    - Insertions are predicted events with no overlap with ground
      truth (IoU_th == 0).
    - Correct identifications are predicted events with sufficient
      overlap (IoU >= IoU_th) and correctly predicted class labels.
    - Substitutions are predicted events with sufficient overlap but
      incorrectly predicted class labels.
    - Deletions are predicted events that are either too short
      (below dur_thresh) or have insufficient overlap (IoU < IoU_th).

    Args:
        predicted_events: A list where each tuple contains the start and end
                          indices of a predicted event and the predicted
                          class label.
        ground_truth_events: A list where each tuple contains the start and
                             end times in seconds of a ground truth event and
                             the actual class label.
        IoU_th: The IoU threshold for determining if an event is considered
                correctly identified.

    Returns:
        tuple: Counts of each event classification type
              (insertions, corrects, substitutions, deletions).
    """
    insertions = corrects = substitutions = deletions = 0

    for predicted_event in predicted_events:
        predicted_label = predicted_event[2]

        # Calculate IoU for the predicted event with
        # all ground truth events and check labels
        # [(iou, label), ....., (iou, label)]
        matches = [
            (intersection_over_union((predicted_event[0], predicted_event[1]), (gt_event[0], gt_event[1])), gt_event[2])
            for gt_event in ground_truth_events
        ]

        # Check if matches is non-empty and find the max IoU and best label
        best_match = max(matches, key=lambda x: x[0]) if matches else (0, None)
        max_iou, best_label = best_match

        # Classify the event based on IoU, duration, and class label
        if max_iou == 0:
            insertions += 1
        elif max_iou < IoU_th:  # Duration threhold satisfaction is implied
            deletions += 1
        elif predicted_label == best_label:  # IoU > IoU_th
            corrects += 1
        else:  # IoU > IoU_th && misclassified event
            substitutions += 1

    return insertions, corrects, substitutions, deletions
