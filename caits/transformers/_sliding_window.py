from sklearn.base import BaseEstimator, TransformerMixin
from caits.windowing import sliding_window_df
from ._data_object import CAI


class SlidingWindow(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=10, overlap=1):
        """Initializes the sliding window transformer.

        Args:
            window_size (int): The number of time steps in each window.
            overlap (int): The number of time steps to overlap adjacent
                           windows.
        """
        self.window_size = window_size
        self.overlap = overlap

    def fit(self, X, y=None):
        """Fit does nothing in this case, but is required to be
        present for compatibility with scikit-learn's Transformer API.

        Args:
            X: The input data.
            y: The target variables (not used).
        """
        return self

    def transform(self, X: CAI) -> CAI:
        """Apply the sliding window transformation to the input data.

        Args:
            X (CAI): The input data object containing a list of DataFrames.

        Returns:
            CAI: A new CAI object with transformed data.
        """
        transformed_X = []
        new_y = []
        new_id = []

        for df, label, id_ in zip(X.X, X.y, X._id):
            windows = sliding_window_df(
                df=df, ws=self.window_size, overlap=self.overlap
            )
            transformed_X.extend(windows)
            new_y.extend([label] * len(windows))
            new_id.extend([id_] * len(windows))

        return CAI(transformed_X, new_y, new_id)
