from typing import Union, Optional
import os
from numpy import ndarray
from numpy import array
from sklearn.pipeline import Pipeline
from tensorflow.keras import Model
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from caits.dataset import Dataset
from caits.performance.metrics import intersection_over_union
from caits.performance.utils import generate_pred_probas, \
    compute_predict_trust_metrics, interpolate_probas, extract_intervals
from caits.visualization import plot_prediction_probas, \
    plot_interpolated_probas
from caits.performance.detection import extract_non_overlap_probas
from caits.performance.detection import apply_duration_threshold, \
    apply_probability_threshold, get_continuous_events
from caits.filtering import filter_butterworth

_OPTIONS = [
    "transformed_data", "prediction_probas", "trust_metrics",
    "non_overlapping_probas", "interpolated_probas", "smoothed_probas",
    "thresholded_probas", "predicted_events", "ICSD", "figures"
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
        cutoff: float,
        ground_truths: list[tuple],
        repeats: int = 5,
        metrics: str = "all",
        prob_th: float = 0.7,
        duration_th: float = 1.,
        iou_th: float = 0.5,
        append_options: Optional[list[str]] = None,
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
        cutoff: The cut-off frequency of the low pass filter for
                interpolated signals smoothening.
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

        Options include: "transformed_data", "prediction_probas", "figures",
                         "non_overlapping_probas", "interpolated_probas",
                        "smoothed_probas", "thresholded_probas", "ICSD",
                        "ICSD", "trust_metrics".

    Returns:
        dict: A dictionary containing selected computed items based on
              `append_options`.
    """
    # Dictionary to append any desired calculated
    # information based on `append_options`
    results = {}

    if append_options is None:
        append_options = _OPTIONS

    # Fit the pilot instance data to the processing pipeline
    transformed_cai_instance = pipeline.transform(instance)
    if "transformed_data" in append_options: # maybe numpy arrays more suitable
        results["transformed_data"] = transformed_cai_instance

    if isinstance(transformed_cai_instance, ndarray):
        X_pilot = transformed_cai_instance
    else:
        # Convert CAI data object to numpy array
        # TODO: # Return single instance from y_pilot and file_pilot
        X_pilot, y_pilot, file_pilot = transformed_cai_instance.to_numpy()
        pilot_instance_filename = file_pilot[0]
        print("Pilot instance: ", pilot_instance_filename)
        print("With label: ", y_pilot[0])

    # Generate prediction probabilities of the model
    prediction_probas = generate_pred_probas(model, X_pilot, repeats)
    # Append prediction probabilities
    if "prediction_probas" in append_options:
        results["prediction_probas"] = prediction_probas

    # compute stats metrics for prediciton probabilty tensor
    trust_metircs = compute_predict_trust_metrics(prediction_probas, metrics)
    # Append trust metrics
    if "trust_metrics" in append_options:
        results["trust_metrics"] = trust_metircs

    # Get mean predicitons
    mean_pred_probas = trust_metircs["mean_pred"]
    print(f"Shape of mean predictions: {mean_pred_probas.shape}")
    # Create figure for probabilities plot
    pred_probas_fig = plot_prediction_probas(mean_pred_probas, sample_rate,
                                             ws, perc_overlap)

    # Bring back to shape before sliding window
    non_overlap_probas = extract_non_overlap_probas(mean_pred_probas,
                                                    perc_overlap)
    print(f"Shape of non-overlapping predictions: {non_overlap_probas.shape}")
    # Append non-overlapping probabilities
    if "non_overlapping_probas" in append_options:
        results["non_overlapping_probas"] = non_overlap_probas

    # Express it as a spline
    interpolated_probas = interpolate_probas(non_overlap_probas,
                                             sampling_rate=sample_rate,
                                             Ws=ws, kind="cubic", clamp=True)
    print(f"Shape of interpolated probabilities: {interpolated_probas.shape}")
    # Append interpolated probabilities
    if "interpolated_probas" in append_options:
        results["interpolated_probas"] = interpolated_probas
    # Create figure plot for splines
    interp_probas_fig = plot_interpolated_probas(interpolated_probas)

    # Apply a low pass butterworth filter
    smoothed_probas = array([
        filter_butterworth(cls_probas, sample_rate, cutoff_freq=cutoff)
        for cls_probas in interpolated_probas.T
    ]).T
    # Append smoothed probabilities
    if "smoothed_probas" in append_options:
        results["smoothed_probas"] = smoothed_probas

    # Apply a probability threshold to the interpolated probabilities
    # and a `at least event time` duration
    threshold_probas = apply_probability_threshold(smoothed_probas, prob_th)
    threshold_probas = apply_duration_threshold(threshold_probas, sample_rate,
                                                duration_th)
    # Append thresholded probabilities
    if "thresholded_probas" in append_options:
        results["thresholded_probas"] = threshold_probas

    # Plot the modified interpolated probabilities after thresholding
    thresh_probas_fig = plot_interpolated_probas(threshold_probas)
    # Append Figure Objects
    if "figures" in append_options:
        results["figures"] = {
            "pred_probas_fig": pred_probas_fig,
            "interp_probas_fig": interp_probas_fig,
            "thresh_probas_fig": thresh_probas_fig
        }

    # Extract event segments after applying the rules
    predicted_events = get_continuous_events(threshold_probas)
    print(f"Predicted Events: {predicted_events}")
    print(f"Ground truth Events: {ground_truths}")

    insertions, corrects, substitutions, deletions = \
        classify_events(predicted_events, ground_truths, IoU_th=iou_th)

    # Append classified Events
    if "ICSD" in append_options:
        results["Insertions"] = insertions
        results["corrects"] = corrects
        results["substitutions"] = substitutions
        results["deletions"] = deletions

    return results


def evaluate_batch(
        pipeline: Pipeline,
        model: Union[BaseEstimator, Model],
        dataset: Dataset,
        events: dict,
        label_encoder: LabelEncoder,
        sample_rate: int,
        ws: float,
        perc_overlap: float,
        cutoff: float,
        repeats: int = 5,
        metrics: str = "all",
        prob_th: float = 0.7,
        duration_th: float = 1.,
        iou_th: float = 0.5,
        append_options: Optional[list[str]] = None,
) -> dict:

    results = {}

    # Extracts ground truths for whole pilot dataset
    ground_truths_dict = extract_intervals(events, label_encoder)

    for i in range(len(dataset)):
        # Take advantage of slicing dunder to return the object
        pilot_instance = dataset[i:i+1]

        # Extract filename to serve as key in global results dict
        file_path = pilot_instance[0][-1]
        # Remove extension
        pilot_instance_filename = os.path.splitext(os.path.basename(file_path))[0]

        # Extract ground truths for specific pilot file
        ground_truths_instance = ground_truths_dict[pilot_instance_filename]

        # Evaluate single instance
        instance_results = evaluate_instance(
            pipeline=pipeline,
            model=model,
            instance=pilot_instance,
            cutoff=cutoff,
            sample_rate=sample_rate,
            ws=ws,
            perc_overlap=perc_overlap,
            ground_truths=ground_truths_instance,
            repeats=repeats,
            metrics=metrics,
            prob_th=prob_th,
            duration_th=duration_th,
            iou_th=iou_th,
            append_options=append_options,
        )

        results[pilot_instance_filename] = instance_results

    return results
