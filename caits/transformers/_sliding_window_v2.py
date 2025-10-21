from sklearn.base import BaseEstimator, TransformerMixin

from caits.dataset._datasetList import Dataset
from ..windowing import sliding_window_arr


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
        self.fitted_ = True
        return self

    def transform(self, data: Dataset) -> Dataset:
        """Apply the sliding window transformation to the input data.

        Args:
            data: The input data object containing a list of DataFrames.

        Returns:
            Dataset: A new Dataset object with transformed data.
        """
        tmp = data.apply(sliding_window_arr, window_size=self.window_size, overlap=self.overlap)
        return data.stack(tmp)
