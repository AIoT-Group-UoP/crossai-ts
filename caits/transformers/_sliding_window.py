from sklearn.base import BaseEstimator, TransformerMixin
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

        if self.overlap >= self.window_size:
            raise ValueError("Overlap must be smaller than window size.")

        step_size = self.window_size - self.overlap

        for df, label, id_ in zip(X.X, X.y, X._id):
            num_rows = df.shape[0]
            for start in range(0, num_rows - self.window_size + 1, step_size):
                end = start + self.window_size
                windowed_df = df.iloc[start:end]
                transformed_X.append(windowed_df)
                new_y.append(label)
                new_id.append(id_)

        return CAI(transformed_X, new_y, new_id)
