import numpy as np
from tensorflow.types.experimental import TensorLike


def gen_pred_probs(
    model,
    data: TensorLike,
    repeats: int = 1
):
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
