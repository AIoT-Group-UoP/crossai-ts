from .metrics import intersection_over_union


def classify_events(
        predicted_events: list[tuple],
        ground_truth_events: list[tuple],
        IoU_th: float,
) -> dict:
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
        dict: Counts of each event classification type
              (insertions, corrects, substitutions, deletions).
    """
    insertions = corrects = substitutions = deletions = 0

    for predicted_event in predicted_events:

        predicted_label = predicted_event[2]

        # Calculate IoU for the predicted event with
        # all ground truth events and check labels
        # [(iou, label), ....., (iou, label)]
        matches = [
            (
                intersection_over_union(
                    (predicted_event[0], predicted_event[1]),
                    (gt_event[0], gt_event[1])
                ),
                gt_event[2]
            )
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

    return {
        "insertions": insertions,
        "corrects": corrects,
        "substitutions": substitutions,
        "deletions": deletions
    }
