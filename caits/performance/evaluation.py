from typing import Union, Optional
import os
import numpy as np
from numpy import array
from sklearn.pipeline import Pipeline
from tensorflow.keras import Model
from sklearn.base import BaseEstimator
from caits.dataset import Dataset
from caits.performance.utils import generate_pred_probas, interpolate_probas, \
        get_intervals_from_events
from caits.performance.metrics import prediction_statistics
from caits.visualization import plot_prediction_probas, \
    plot_interpolated_probas
from caits.performance.detection import get_non_overlap_probas, \
    apply_duration_threshold, apply_probability_threshold, \
        get_continuous_events, classify_events
from caits.filtering import filter_butterworth
from caits.performance.metrics import detection_ratio, reliability, erer


_OPTIONS = [
    "transformed_data", "prediction_probas", "trust_metrics",
    "non_overlapping_probas", "interpolated_probas", "smoothed_probas",
    "thresholded_probas", "predicted_events", "ICSD", "figures"
]


def model_robustness_instance(
        model: Union[BaseEstimator, Model],
        instance: np.ndarray,
        class_names: list[str],
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
        figsize=(14, 6),
        append_options: Optional[list[str]] = None,
) -> dict:
    """Performs the evaluation of the model on the pilot data, optionally
    returning figures and allowing selective inclusion of results.

    Args:
        model: Sklearn or Tensorflow Model to be evaluated.
        instance: Pilot instance as numpy array.
        class_names: A list of the unique class names, used to
                      interpret the model's predictions. The order of
                      the labels should match the order of the model's output.
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
                        "ICSD", "pred_stats", "trust_metrics".

    Returns:
        dict: A dictionary containing selected computed items based on
              `append_options`.
    """
    # Dictionary to append any desired calculated
    # information based on `append_options`
    results = {}

    if append_options is None:
        append_options = _OPTIONS

    # if "transformed_data" in append_options:
    #     results["transformed_data"] = instance

    # Generate prediction probabilities of the model
    prediction_probas = generate_pred_probas(model, instance, repeats)
    # Append prediction probabilities
    if "prediction_probas" in append_options:
        results["prediction_probas"] = prediction_probas

    # compute stats metrics for prediciton probabilty tensor
    pred_stats = prediction_statistics(prediction_probas, metrics)
    # Append trust metrics
    if "pred_stats" in append_options:
        results["pred_stats"] = pred_stats

    # Get mean predicitons
    mean_pred_probas = pred_stats["mean_pred"]
    print(f"Shape of mean predictions: {mean_pred_probas.shape}")
    # Create figure for probabilities plot
    pred_probas_fig = plot_prediction_probas(
        mean_pred_probas, sample_rate, ws, perc_overlap, class_names, figsize
    )

    # Bring back to shape before sliding window
    non_overlap_probas = get_non_overlap_probas(mean_pred_probas, perc_overlap)
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
    interp_probas_fig = plot_interpolated_probas(
        interpolated_probas, class_names, figsize
    )

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
    thresh_probas_fig = plot_interpolated_probas(
        threshold_probas, class_names, figsize
    )
    # Append Figure Objects
    if "figures" in append_options:
        results["figures"] = {
            # "pilot_signal": pilot_signal,
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

    if "trust_metrics" in append_options:
        results["DR"] = detection_ratio(corrects, deletions, substitutions)
        results["Reliability"] = reliability(corrects, insertions)
        results["ERER"] = erer(deletions, insertions, substitutions, corrects)

    return results


def model_robustness(
        pipeline: Pipeline,
        model: Union[BaseEstimator, Model],
        dataset: Dataset,
        events: dict,
        class_names: list[str],
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
        figsize: tuple = (14, 6),
) -> dict:

    results = {}

    # Extracts ground truths for whole pilot dataset
    ground_truths_dict = get_intervals_from_events(
        events, class_names, sample_rate
    )

    for i in range(len(dataset)):
        # Take advantage of slicing dunder to return the object
        pilot_dataset_instance = dataset[i:i+1]
        pilot_instance_transformed = pipeline.transform(pilot_dataset_instance)
        if isinstance(pilot_instance_transformed, Dataset):
            X_pilot, _, _ = pilot_instance_transformed.to_numpy()
        else:
            X_pilot = pilot_instance_transformed

        # Extract filename to serve as key in global results dict
        file_path = pilot_dataset_instance[0][-1]
        # Remove extension
        pilot_instance_filename = os.path.splitext(os.path.basename(file_path))[0]

        # Extract ground truths for specific pilot file
        ground_truths_instance = ground_truths_dict[pilot_instance_filename]

        # Evaluate single instance
        instance_results = model_robustness_instance(
            model=model,
            instance=X_pilot,
            class_names=class_names,
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
            figsize=figsize
        )

        results[pilot_instance_filename] = instance_results

    return results
