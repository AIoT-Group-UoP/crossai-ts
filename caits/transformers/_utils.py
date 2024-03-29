from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from joblib import dump
from typing import Union


class ToSklearn(BaseEstimator, TransformerMixin):
    def __init__(self):
        """Initializes the ToSklearnTransformer."""
        super().__init__()

    def fit(self, X, y=None):
        """Fit method for the transformer.

        Since this transformer doesn't need to learn anything,
        we simply return self.

        Args:
            X: The input Dataset object.
            y: Unused.

        Returns:
            self.
        """
        return self

    def transform(self, X):
        """Transforms the Dataset object into a 2D numpy array
        suitable for scikit-learn.

        Args:
            X: The Dataset object to be transformed.

        Returns:
            numpy.ndarray: A 2D array where each row is a concatenated
                           representation of the DataFrames in the Dataset.
        """
        X, _, _ = X.to_numpy()
        return X.squeeze(axis=-1)


def sklearn_to_pkl(
        model: Union[BaseEstimator, Pipeline], filename: str) -> None:
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
        raise ValueError("The model must be a scikit-learn \
                         BaseEstimator or Pipeline instance.")

    # Ensure the filename ends with '.pkl'
    if not filename.endswith('.pkl'):
        filename += '.pkl'

    # Save the model to the specified file
    dump(model, filename)
    print(f"Model saved to {filename}")
