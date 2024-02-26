from typing import Union
from sklearn.pipeline import Pipeline
from tensorflow.keras import Model
from sklearn.base import BaseEstimator
from .metrics import intersection_over_union
from caits.transformers import Dataset

_OPTIONS = [
    'transformed_data', 'prediction_probas', 'trust_metrics',
    'non_overlapping_probas', 'interpolated_probas', 'smoothed_probas',
    'thresholded_probas', 'predicted_events', 'ICSD', 'figures'
]


def classify_events(
        predicted_events: list[tuple],
        ground_truth_events: list[tuple],
        IoU_th: float,
) -> tuple:
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

    return insertions, corrects, substitutions, deletions


def evaluate_instance(
        pipeline: Pipeline,
        model: Union[BaseEstimator, Model],
        instance: Dataset,
        sample_rate: int,
        ws: float,
        perc_overlap: float,
        ground_truths: list[tuple],
        repeats: int = 5,
        metrics: str = "all",
        prob_th: float = 0.7,
        duration_th: float = 1.,
        iou_th: float = 0.5,
        display: bool = False
) -> dict:
    """Performs the evaluation of the model on the pilot data, optionally
    returning figures and allowing selective inclusion of results.

    Args:
        pipeline: Fitted Sklearn-pipeline.
        model: Sklearn or Tensorflow Model to be evaluated.
        instance: Dataset object with a single instance.
        sample_rate: The sampling rate of the data.
        ws: Window size for processing.
        perc_overlap: The percentage overlap used when segmenting
                      the data for predictions.
        ground_truths: A list of tuples representing the ground truth events.
        repeats: The number of times to repeat the prediction
                 process for generating trust metrics.
        metrics: Specifies which trust metrics to compute for the
                 prediction probabilities.
        prob_th: The probability threshold above which a prediction is
                 considered positive.
        duration_th: The minimum duration threshold for an event to be
                     considered valid.
        iou_th: The Intersection over Union (IoU) threshold used to classify
                the accuracy of the predicted events against ground truth.
        display: If True, various plots will be displayed during the
                 valuation process.
        append_options: A list of strings indicating which parts of the
                        evaluation to include in the results dictionary.
        Options include: 'transformed_data', 'prediction_probas', 'figures',
                         'non_overlapping_probas', 'interpolated_probas',
                        'smoothed_probas', 'thresholded_probas', 'ICSD',
                        'ICSD', 'trust_metrics'.

    Returns:
        dict: A dictionary containing selected computed items based on
              `append_options`.
    """
    # Fit the pilot instance data to the processing pipeline
    transformed_cai_instance = pipeline.transform(instance)

    # Convert CAI data object to numpy array
    X_pilot, y_pilot, file_pilot = transformed_cai_instance.to_numpy()  # probably needs to return single instance from y_pilot and file_pilot
    pilot_instance_filename = file_pilot[0]
    print("Pilot instance: ", pilot_instance_filename)
    print("With label: ", y_pilot[0])

    # Generate prediction probabilities of the model
    prediction_probas = gen_pred_probs(model=model, X=X_pilot, repeats=repeats)

    # compute stats metrics for prediciton probabilty tensor
    trust_metircs = compute_predict_trust_metrics(prediction_probas, compute=metrics)

    # get mean predicitons
    mean_pred_probas = trust_metircs["mean_pred"]
    print(f"Shape of mean predictions: {mean_pred_probas.shape}")

    if display:
        plot_prediction_probas(mean_predictions, sample_rate, ws, perc_overlap)

    non_overlap_probas = extract_non_overlap_probas(mean_pred_probas, perc_overlap)
    print(f"Shape of non-overlapping predictions: {non_overlap_probas.shape}")

    interpolated_probas = interpolate_probas(non_overlap_probas, sampling_rate=sample_rate,
                                             Ws=ws, kind="cubic", clamp=True)
    print(f"Shape of interpolated predictions probabilities: {interpolated_probas.shape}")

    if display:
        plot_interpolated_probas(interpolated_probas)

    # Apply Moving Average Filter
    smoothed_probas = np.array([
        moving_average_filter(pred_probas, window_size=50)
        for pred_probas in interpolated_probas.T
    ]).T

    # Apply a probability threshold to the interpolated probabilities
    # and a `at least event time` duration
    threshold_probas = apply_probability_threshold(smoothed_probas, prob_th)
    threshold_probas = apply_duration_threshold(threshold_probas, sample_rate,
                                                duration_th)

    # Plot the modified interpolated probabilities after thresholding
    if display:
        plot_interpolated_probas(threshold_probas)

    # Extract event segments after applying the rules
    predicted_events = get_continuous_events(threshold_probas)
    print(f"Predicted Events: {predicted_events}")
    print(f"Ground truth Events: {ground_truths}")

    insertions, corrects, substitutions, deletions = \
        classify_events(predicted_events, ground_truths, IoU_th=iou_th)

    return
