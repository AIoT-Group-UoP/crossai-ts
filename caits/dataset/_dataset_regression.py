from typing import Optional, List

import numpy as np
import pandas as pd
from pandas import DataFrame


class RegressionDataset:
    def __init__(self, X: DataFrame, y: DataFrame) -> None:
        # Check if X and y are DataFrames.
        if not all(isinstance(data, DataFrame) for data in [X, y]):
            raise TypeError("X and y must be DataFrames.")

        # Check that all inputs have the same length
        if not (len(X) == len(y)):
            raise ValueError(f"All input lists must have the same length. (len(X) = {len(X)}, len(y) = {len(y)})")

        self.X = X
        self.y = y

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Allows for dataset indexing/slicing to get a specific data point."""
        if isinstance(idx, slice):
            # Handle slicing
            return RegressionDataset(self.X.loc[idx], self.y.loc[idx])
        elif isinstance(idx, int):
            # Handle single item selection
            return self.X.loc[idx], self.y.loc[idx]
        else:
            raise TypeError("Invalid argument type.")

    def __iter__(self):
        """Allows for iterating over the dataset."""
        self._current = 0
        return self

    def __next__(self):
        """Returns the next item from the dataset."""
        if self._current < len(self):
            result = (self.X.loc[self._current], self.y.loc[self._current])
            self._current += 1
            return result
        else:
            raise StopIteration

    def __repr__(self) -> str:
        """Provide a string representation of the CAI object."""
        return f"RegressionDataset with {len(self)} instances"


    def to_numpy(self, dtype=None):
        """Converts data to NumPy arrays, ensuring X_np has shape (k, n, m)."""

        # Convert to list of 2D arrays
        X_np = self.X.to_numpy(dtype=self.X.dtypes)
        y_np = self.y.to_numpy(dtype=self.y.dtypes)

        return X_np, y_np

    def to_dict(self):
        """Converts data to Dictionary."""
        return {"X": self.X, "y": self.y}

    def unify(self):
        """Converts data to Pandas DataFrames."""
        return pd.concat([self.X, self.y])

    def train_test_split(self, test_size=0.2, as_numpy=False):
        """Splits the dataset into training and testing subsets,
        with an option to stratify the split.
        """

        Nx = int( len(self.X) * (1 - test_size) )
        X_train, X_test = self.X.loc[:(Nx-1)], self.X.loc[Nx:]
        y_train, y_test = self.y.loc[:(Nx-1)], self.y.loc[Nx:]

        # Convert back to Dataset objects if requested
        train_dataset = RegressionDataset(X_train, y_train)
        test_dataset = RegressionDataset(X_test, y_test)

        if as_numpy:
            try:
                # Return as numpy arrays
                train_dataset = train_dataset.to_numpy()
                test_dataset = test_dataset.to_numpy()
            except ValueError:
                print("Cannot convert to numpy arrays due to length inconsistency.")
                print("Returning `Dataset` objects instead.")

        return train_dataset, test_dataset


def TrainTestArraysToRegressionDataset(
        X: np.ndarray,
        y: np.ndarray,
        X_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
) -> RegressionDataset:
    """Converts a NumPy array, in which each row is a DataFrame, to a
    CrossAI Dataset object. The features and labels are in
    the form (n_samples, features), (n_samples, output_columns).

    Args:
        X: np.ndarray of timeseries of X.
        y: np.ndarray of timeseries of y.
        X_labels: column names of timeseries of X.
        y_labels: column names of timeseries of y.

    Returns:
        Dataset: The CrossAI RegressionDataset object.
    """

    if X_labels is not None and X.shape[1] != len(X_labels):
        raise ValueError("Number of columns in `X` and `X_labels` do not match.")
    if y_labels is not None and y.shape[1] != len(y_labels):
        raise ValueError("Number of columns in `y` and `y_labels` do not match.")

    return RegressionDataset(
        X=DataFrame(X, columns=X_labels),
        y=DataFrame(y, columns=y_labels)
    )


def DataFrameToRegressionDataset(
        df: DataFrame,
        X_cols: Optional[List[str]] = None,
        y_cols: Optional[List[str]] = None,
) -> RegressionDataset:
    """Converts a NumPy array, in which each row is a DataFrame, to a
    CrossAI Dataset object. The features and labels are in
    the form (n_samples, features), (n_samples, output_columns).

    Args:
        X: np.ndarray of timeseries of X.
        y: np.ndarray of timeseries of y.
        X_labels: column names of timeseries of X.
        y_labels: column names of timeseries of y.

    Returns:
        Dataset: The CrossAI RegressionDataset object.
    """
    if X_cols is None:
        X_cols = df.columns
    if y_cols is None:
        y_cols = df.columns[-1]
    X_cols = list(set(X_cols).difference(set(y_cols)))

    X = df[X_cols]
    y = df[y_cols]

    return RegressionDataset(X=X, y=y)


def ListToRegressionDataset(
        X,
        y,
        X_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None
) -> RegressionDataset:
    """Converts a list of Lists to a CrossAI Dataset object.

    Args:
        X: list of X timeseries.
        y: list of y timeseries.
        X_labels: columns names of X timeseries.
        y_labels: columns names of y timeseries.

    Returns:
        Dataset: The CrossAI RegressionDataset object.
    """

    return RegressionDataset(
        X=pd.DataFrame(list(zip(*X)), columns=X_labels),
        y=pd.DataFrame(list(zip(*y)), columns=y_labels)
    )


def ArrayToRegressionDataset(
        array: np.ndarray,
        X_idxs: Optional[List[int]] = None,
        y_idxs: Optional[List[int]] = None,
        X_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
) -> RegressionDataset:

    if X_idxs is None and y_idxs is None:
        X_idxs = [i for i in range(array.shape[1] - 1)]
        y_idxs = [array.shape[1] - 1]
    elif X_idxs is None and y_idxs is not None:
        X_idxs = list(set(range(array.shape[1])).difference(set(y_idxs)))
    elif X_idxs is not None and y_idxs is None:
        y_idxs = list(set(range(array.shape[1])).difference(X_idxs))[-1]

    X_array = array[:, X_idxs]
    y_array = array[:, y_idxs]

    return TrainTestArraysToRegressionDataset(X=X_array, y=y_array, X_labels=X_labels, y_labels=y_labels)


