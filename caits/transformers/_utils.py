from sklearn.base import BaseEstimator, TransformerMixin


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
