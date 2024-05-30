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
    potential_events: List[Tuple[int, int, int]],
    sr: int,
    duration_threshold: float,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Applies a duration threshold to filter desired events.

    This function filters out events in the `potential_events` list that have a
    duration below the specified threshold. The duration threshold is converted 
    from seconds to samples using the provided sampling rate. Events that meet the 
    threshold are retained, and the corresponding segments in `interpolated_probs`
    are preserved.

    Args:
        interpolated_probs: A 2D array of interpolated probabilities (n_timesteps, n_classes).
        potential_events: A list of tuples (start, end, class) representing potential events.
                          Start and end are in samples.
        sr: Sampling rate of the time series.
        duration_threshold: Duration threshold in seconds.

    Returns:
        Tuple:
            - np.ndarray: Modified interpolated probabilities, where segments below the threshold are zeroed.
            - List[Tuple[int, int, int]]: Filtered events that meet the duration threshold.
    """
    
    # Convert duration to samples.
    min_duration_samples = int(sr * duration_threshold)
    
    # Filter events based on duration.
    events = [(start, end, cls) for start, end, cls in potential_events 
              if end - start >= min_duration_samples]
    
    # Zero out segments below the threshold.
    modified_probs = np.zeros_like(interpolated_probs)
    for start, end, cls in events:
        modified_probs[start:end, cls] = interpolated_probs[start:end, cls]

    return modified_probs, events


def get_continuous_events(probabilities: np.ndarray) -> List[Tuple[int, int, int]]:
    """Identifies continuous segments where probabilities
    are above zero and returns them as events.

    Args:
        probabilities: A 2D numpy array (n_instances, n_classes)
                       containing probabilities for each class.

    Returns:
        List[Tuple[int, int, int]]: A list of tuples (start, end, class) 
                                    representing the continuous events. 
                                    Start and end indices are inclusive.
    """
    events = []
    for class_idx in range(probabilities.shape[1]):  # Iterate through classes
        class_probs = probabilities[:, class_idx]

        # Efficiently find segment transitions using np.diff and np.where
        is_above_zero = (class_probs > 0).astype(int)
        above_zero_diff = np.diff(is_above_zero, prepend=0, append=0)
        start_indices = np.where(above_zero_diff == 1)[0]
        end_indices = np.where(above_zero_diff == -1)[0] # not substracting 1 on purpose

        events.extend(zip(start_indices, end_indices, [class_idx] * len(start_indices)))

    return events


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
