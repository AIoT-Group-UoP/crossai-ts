from sklearn.base import BaseEstimator, TransformerMixin

from ..dataset import Dataset
from ..windowing import sliding_window_df


class SlidingWindow(BaseEstimator, TransformerMixin):
    def __init__(self, window_size: int = 10, overlap: int = 1):
        """Initializes the sliding window transformer.

        Args:
            window_size: The number of time steps in each window.
            overlap: The number of time steps to overlap adjacent
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

    def transform(self, X: Dataset) -> Dataset:
        """Apply the sliding window transformation to the input data.

        Args:
            X: The input data object containing a list of DataFrames.

        Returns:
            Dataset: A new Dataset object with transformed data.
        """
        transformed_X = []
        new_y = []
        new_id = []

        for df, label, id_ in X:
            windowed_dfs = sliding_window_df(df, self.window_size, self.overlap)
            transformed_X.extend(windowed_dfs)
            new_y.extend([label] * len(windowed_dfs))
            new_id.extend([id_] * len(windowed_dfs))

        return Dataset(transformed_X, new_y, new_id)
