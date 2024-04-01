from typing import Union, Optional
import numpy as np
from numpy import array
from tensorflow.keras import Model
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from caits.dataset import Dataset
from caits.performance.utils import generate_pred_probas, interpolate_probas, \
        get_gt_events_from_dict
from caits.performance.metrics import prediction_statistics
from caits.visualization import plot_prediction_probas, \
    plot_interpolated_probas
from caits.performance.detection import get_non_overlap_probas, \
    apply_duration_threshold, apply_probability_threshold, \
        get_continuous_events, classify_events
from caits.filtering import filter_butterworth
from caits.performance.metrics import detection_ratio, reliability, erer
from caits.visualization import plot_signal


_OPTIONS = [
    "transformed_data", "prediction_probas", "trust_metrics",
    "non_overlapping_probas", "interpolated_probas", "smoothed_probas",
    "thresholded_probas", "predicted_events", "ICSD", "figures"
]


def robustness_analysis(
        model: Union[BaseEstimator, Model],
        input_data: np.ndarray,
        class_names: list[str],
        sample_rate: int,
        ws: float,
        perc_overlap: float,
        ground_truths: list[tuple],
        cutoff: float,
        repeats: int = 5,
        metrics: str = "all",
        prob_th: float = 0.7,
        duration_th: float = 1.,
        iou_th: float = 0.5,
        figsize=(14, 6),
        options_to_include: Optional[list[str]] = None,
) -> dict:
    """Evaluates the model's robustness in event detection tasks using
    time-series data, providing detailed metrics and optional visualizations.

    Args:
        model: The model to be evaluated, compatible with Scikit-learn
               or TensorFlow.
        input_data: The data to be passed to the model for inference.
                    Must be at least 2-dimensional. Each instance should
                    represent the window while the second dimension should
                    represent the features.
        class_names: List of unique class names corresponding to the model's
                     outputs.
        sample_rate: Sampling rate of the input data.
        ws: Window size for segmenting the data.
        perc_overlap: Percentage of overlap between consecutive data segments.
        ground_truths: Ground truth events for comparison with model
                       predictions.
        cutoff: Cut-off frequency for the low-pass filter applied to the data.
        repeats: Number of times the prediction process is repeated.
        metrics: Specifies which metrics to compute; 'all' computes all
                 available metrics.
        prob_th: Probability threshold for considering a prediction positive.
        duration_th: Minimum duration for an event to be considered valid.
        iou_th: Intersection over Union threshold for event accuracy
                classification.
        figsize: Figure size for any generated plots.
        append_options: Additional result components to include in the output.

        Options include: "transformed_data", "prediction_probas", "figures",
                         "non_overlapping_probas", "interpolated_probas",
                         "smoothed_probas", "thresholded_probas", "ICSD",
                         "pred_stats", "trust_metrics".

    Returns:
        A dictionary containing selected computed items based on
        `append_options`.
    """
    # Ensure input_data is at least 2D
    if input_data.ndim < 2:
        raise ValueError("`input_data` must be at least 2D.")

    # Dictionary to append any desired calculated
    # information based on `options_to_append`
    results = {}
    options_to_include = options_to_include or _OPTIONS

    if "transformed_data" in options_to_include:
        results["transformed_data"] = input_data

    # Generate prediction probabilities of the model
    prediction_probas = generate_pred_probas(model, input_data, repeats)
    # Append prediction probabilities
    if "prediction_probas" in options_to_include:
        results["prediction_probas"] = prediction_probas

    # compute stats metrics for prediciton probabilty tensor
    pred_stats = prediction_statistics(prediction_probas, metrics)
    # Append trust metrics
    if "pred_stats" in options_to_include:
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
    if "non_overlapping_probas" in options_to_include:
        results["non_overlapping_probas"] = non_overlap_probas

    # Express it as a spline
    interpolated_probas = interpolate_probas(non_overlap_probas,
                                             sampling_rate=sample_rate,
                                             Ws=ws, kind="cubic", clamp=True)
    print(f"Shape of interpolated probabilities: {interpolated_probas.shape}")
    # Append interpolated probabilities
    if "interpolated_probas" in options_to_include:
        results["interpolated_probas"] = interpolated_probas
    # Create figure plot for splines
    interp_probas_fig = plot_interpolated_probas(
        interpolated_probas, class_names, figsize
    )

    # Apply a low pass butterworth filter
    smoothed_probas = array([
        filter_butterworth(
            array=cls_probas,
            fs=sample_rate,
            filter_type="lowpass",
            cutoff_freq=cutoff,
            order=3
        )
        for cls_probas in interpolated_probas.T
    ]).T

    # Append smoothed probabilities
    if "smoothed_probas" in options_to_include:
        results["smoothed_probas"] = smoothed_probas

    # Apply a probability threshold to the interpolated probabilities
    # and a `at least event time` duration
    threshold_probas = apply_probability_threshold(smoothed_probas, prob_th)
    threshold_probas = apply_duration_threshold(threshold_probas, sample_rate,
                                                duration_th)
    # Append thresholded probabilities
    if "thresholded_probas" in options_to_include:
        results["thresholded_probas"] = threshold_probas

    # Plot the modified interpolated probabilities after thresholding
    thresh_probas_fig = plot_interpolated_probas(
        threshold_probas, class_names, figsize
    )
    # Append Figure Objects
    if "figures" in options_to_include:
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
    if "ICSD" in options_to_include:
        results["Insertions"] = insertions
        results["corrects"] = corrects
        results["substitutions"] = substitutions
        results["deletions"] = deletions

    if "trust_metrics" in options_to_include:
        results["DR"] = detection_ratio(corrects, deletions, substitutions)
        results["Reliability"] = reliability(corrects, insertions)
        results["ERER"] = erer(deletions, insertions, substitutions, corrects)

    return results


def robustness_analysis_many(
        model: Union[BaseEstimator, Model],
        X: list[np.ndarray],
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
        options_to_include: Optional[list[str]] = None,
        figsize: tuple = (14, 6),
) -> dict:

    results = {}

    # Extracts ground truths for whole pilot dataset
    ground_truths_dict = get_gt_events_from_dict(
        events, class_names, sample_rate
    )

    for i, (filename, gt_events) in enumerate(ground_truths_dict.items()):
        # Get instance
        ts_input_data = X[i]
        # Evaluate single instance
        ts_instance_results = robustness_analysis(
            model=model,
            input_data=ts_input_data,
            class_names=class_names,
            cutoff=cutoff,
            sample_rate=sample_rate,
            ws=ws,
            perc_overlap=perc_overlap,
            ground_truths=gt_events,
            repeats=repeats,
            metrics=metrics,
            prob_th=prob_th,
            duration_th=duration_th,
            iou_th=iou_th,
            options_to_include=options_to_include,
            figsize=figsize
        )

        results[filename] = ts_instance_results

    return results


def robustness_analysis_batch(
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
        options_to_include: Optional[list[str]] = None,
        figsize: tuple = (14, 6),
) -> dict:

    results = {}
    options_to_include = options_to_include or _OPTIONS

    # Extracts ground truths for whole pilot dataset
    ground_truths_dict = get_gt_events_from_dict(
        events, class_names, sample_rate
    )

    for i, (filename, gt_events) in enumerate(ground_truths_dict.items()):
        # Take advantage of slicing dunder to return the object
        # if single index used, it will return a tuple
        dataset_instance = dataset[i:i+1]

        # define the label name for the instance
        label = dataset_instance.y[0]
        if isinstance(label, int):
            label = class_names[label]

        # Append Figure Objects
        if "figures" in options_to_include:
            # Since `dataset_instance` is the raw time series instance
            # we can plot it and store it for logging purposes
            pilot_signal = plot_signal(
                dataset_instance.X[0].values.flatten(), sr=sample_rate,
                name="Pilot Signal", mode="samples", channels=label,
                figsize=figsize
            )  # TODO: Modify function to control the x axis mode (samples vs time)

            results["figures"] = {
                "pilot_signal": pilot_signal,
            }

        # transform the data using the pipeline
        input_data = pipeline.transform(dataset_instance)

        # Evaluate single instance
        instance_results = robustness_analysis(
            model=model,
            input_data=input_data,
            class_names=class_names,
            cutoff=cutoff,
            sample_rate=sample_rate,
            ws=ws,
            perc_overlap=perc_overlap,
            ground_truths=gt_events,
            repeats=repeats,
            metrics=metrics,
            prob_th=prob_th,
            duration_th=duration_th,
            iou_th=iou_th,
            options_to_include=options_to_include,
            figsize=figsize
        )

        results[filename] = instance_results

    return results
