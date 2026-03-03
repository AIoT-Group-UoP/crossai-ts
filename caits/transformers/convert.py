from typing import Union
from joblib import dump, load
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


def sklearn_to_pkl(model: Union[BaseEstimator, Pipeline], filename: str) -> None:
    """Saves a scikit-learn model or pipeline to a .pkl file using joblib.

    This function uses the joblib library to serialize and save scikit-learn
    models or pipelines. It is efficient for models that include large numpy
    arrays.

    Args:
        model: The scikit-learn model or pipeline to be saved. Must be an
               instance of a scikit-learn BaseEstimator or Pipeline.
        filename: The name of the file to save the model to. If the filename
                  does not end with '.pkl', it will be appended automatically.
                  It is recommended to use a meaningful filename that reflects
                  the model type and training characteristics.

    Returns:
        None: This function does not return a value. It writes
              the model to a file.

    Example:
    ```python
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Load data
    X, y = load_iris(return_X_y=True)

    # Create and fit a model
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X, y)

    # Save the model to a file
    save_sklearn_model_to_pkl(model, 'iris_model.pkl')
    ```

    Raises:
        ValueError: If the provided model is not an instance of a scikit-learn
                    BaseEstimator or Pipeline.
    """
    if not isinstance(model, (BaseEstimator, Pipeline)):
        raise ValueError(
            "The model must be a scikit-learn \
                         BaseEstimator or Pipeline instance."
        )

    # Ensure the filename ends with '.pkl'
    if not filename.endswith(".pkl"):
        filename += ".pkl"

    # Save the model to the specified file
    dump(model, filename)
    print(f"Model saved to {filename}")


def load_sklearn_model_from_pkl(filename: str) -> Union[BaseEstimator, Pipeline]:
    """Loads a scikit-learn model or pipeline from a .pkl file using joblib.

    This function uses the joblib library to deserialize and load scikit-learn
    models or pipelines that were previously saved using the `joblib.dump` function.
    It is efficient for models that include large numpy arrays.

    Args:
        filename: The name of the file to load the model from. Must be a .pkl file.

    Returns:
        The loaded scikit-learn model or pipeline.

    Example:
    ```python
    # Load the model from a file
    model = load_sklearn_model_from_pkl('iris_model.pkl')

    # Use the model for prediction
    X_new = ...  # New data for prediction
    predictions = model.predict(X_new)
    ```

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    try:
        model = load(filename)
        print(f"Model loaded from {filename}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filename} does not exist.")
