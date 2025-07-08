from typing import Optional, List

import numpy as np
import pandas as pd
from pandas import DataFrame


class RegressionDataset:
    def __init__(self, X: List[DataFrame], y: List[DataFrame]) -> None:
        # Check if X and y are DataFrames.
        if not all(isinstance(data, List) for data in [X, y]):
            raise TypeError("X and y must be DataFrames.")

        # Check that all inputs have the same length
        # if len(set([len(tmp) for tmp in X] + [len(tmp) for tmp in y])) != 1:
        #     raise ValueError("X and y must have the same number of columns.")
        if not len(X) == len(y):
            raise ValueError("X and y must have the same length.")

        self.X = X
        self.y = y

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X[0])

    def __getitem__(self, idx):
        """Allows for dataset indexing/slicing to get a specific data point."""
        if isinstance(idx, slice):
            # Handle slicing
            return RegressionDataset(X=[tmp.loc[idx] for tmp in self.X], y=[tmp.loc[idx] for tmp in self.y])
        elif isinstance(idx, int):
            # Handle single item selection
            return [tmp.loc[idx] for tmp in self.X], [tmp.loc[idx] for tmp in self.y]
        else:
            raise TypeError("Invalid argument type.")

    def __iter__(self):
        """Allows for iterating over the dataset."""
        self._current = 0
        return self

    def __next__(self):
        """Returns the next item from the dataset."""
        if self._current < len(self):
            result = (tuple([tmp.loc[self._current] for tmp in self.X]), tuple([tmp.loc[self._current] for tmp in self.y]))
            self._current += 1
            return result
        else:
            raise StopIteration

    def __repr__(self) -> str:
        """Provide a string representation of the CAI object."""
        return f"RegressionDataset with {len(self.X[0])} instances"


    def to_numpy(self, dtype=None):
        """Converts data to NumPy arrays, ensuring X_np has shape (k, n, m)."""

        # Convert to list of 2D arrays
        X_arrays = [tmp.to_numpy(dtype=tmp.dtypes)[:, 0] for tmp in self.X]
        y_arrays = [tmp.to_numpy(dtype=tmp.dtypes)[:, 0] for tmp in self.y]

        X_np = np.stack(X_arrays, axis=1)
        y_np = np.stack(y_arrays, axis=1)

        return X_np, y_np

    def to_dict(self):
        """Converts data to Dictionary."""
        return {"X": self.X, "y": self.y}

    def unify(self, other):
        return RegressionDataset(X=[self.X[i] + other.X[i] for i in range(len(self.X))],
                                 y=[self.y[i] + other.y[i] for i in range(len(self.y))])

    def train_test_split(self, test_size=0.2, as_numpy=False, random_state=None):
        """Splits the dataset into training and testing subsets,
        with an option to stratify the split.
        """
        Nx = int( len(self.X[0]) * (1 - test_size) )

        all_idxs = np.arange(len(self.X[0]))

        if random_state is not None:
            train_idxs = np.random.RandomState(random_state).choice(all_idxs, size=Nx, replace=False)
        else:
            train_idxs = np.arange(Nx)

        test_idxs = np.array(list(set(np.arange(len(self.X[0]))).difference(set(train_idxs))))

        X_train = [tmp.loc[train_idxs] for tmp in self.X]
        X_test = [tmp.loc[test_idxs] for tmp in self.X]
        y_train = [tmp.loc[train_idxs] for tmp in self.y]
        y_test = [tmp.loc[test_idxs] for tmp in self.y]

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


def ArrayToRegressionDataset(
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
        X=[pd.DataFrame(X[:, i], columns=[X_labels[i]]) for i in range(X.shape[1])],
        y=[pd.DataFrame(y[:, i], columns=[y_labels[i]]) for i in range(y.shape[1])],
    )


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
        X=[pd.DataFrame(X[i], columns=[X_labels[i]]) for i in range(len(X))],
        y=[pd.DataFrame(y[i], columns=[y_labels[i]]) for i in range(len(y))]
    )
