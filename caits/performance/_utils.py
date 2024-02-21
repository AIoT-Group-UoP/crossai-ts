import numpy as np
from tensorflow.types.experimental import TensorLike


def gen_pred_probs(
    model,
    data: TensorLike,
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
        data: The dataset on which predictions are to be made. Expected to be
              formatted appropriately for the model's input requirements.
        repeats: The number of times the prediction process should be repeated.
                 This is useful for assessing model consistency and uncertainty
                 in predictions.

    Returns:
        An array of predictions made by the model. For repeated predictions, 
        the results are stacked along a new dimension, allowing for further
        analysis of prediction consistency or uncertainty.
    """

    # Initialize a list to hold all predictions
    all_predictions = []

    for _ in range(repeats):
        try:
            # Attempt to use predict_proba for probabilistic outcomes
            predictions = model.predict_proba(data)
        except AttributeError:
            # Fallback to using predict for direct predictions
            predictions = model.predict(data, verbose=0)

        all_predictions.append(predictions)

    # Stack predictions along a new dimension for repeated predictions
    all_predictions = np.stack(all_predictions, axis=0)

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
