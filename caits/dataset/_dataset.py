from typing import List, Optional

import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split as sklearn_tts


class Dataset:
    def __init__(self, X: List[DataFrame], y: List[str], id: List[str]) -> None:
        # Check if X, y, and id are lists
        if not all(isinstance(data, list) for data in [X, y, id]):
            raise TypeError("X, y, and id must be lists.")

        # Check that all inputs have the same length
        if not (len(X) == len(y) == len(id)):
            raise ValueError("All input lists must have the same length.")

        self.X = X
        self.y = y
        self._id = id

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Allows for dataset indexing/slicing to get a specific data point."""
        if isinstance(idx, slice):
            # Handle slicing
            return Dataset(self.X[idx], self.y[idx], self._id[idx])
        elif isinstance(idx, int):
            # Handle single item selection
            return self.X[idx], self.y[idx], self._id[idx]
        else:
            raise TypeError("Invalid argument type.")

    def __iter__(self):
        """Allows for iterating over the dataset."""
        self._current = 0
        return self

    def __next__(self):
        """Returns the next item from the dataset."""
        if self._current < len(self):
            result = (self.X[self._current], self.y[self._current], self._id[self._current])
            self._current += 1
            return result
        else:
            raise StopIteration

    def __repr__(self) -> str:
        """Provide a string representation of the CAI object."""
        return f"Dataset with {len(self)} instances"

    def batch(self, batch_size=1):
        """Yields data instances or batches from the dataset."""
        for i in range(0, len(self), batch_size):
            X_batch = self.X[i : i + batch_size]
            y_batch = self.y[i : i + batch_size]
            id_batch = self._id[i : i + batch_size]

            yield X_batch, y_batch, id_batch

    def unify(self, other: "Dataset") -> "Dataset":
        """Concatenates two Dataset objects by appending their rows."""

        return Dataset(self.X + other.X, self.y + other.y, self._id + other._id)

    def to_numpy(self, dtype=None):
        """Converts data to NumPy arrays, ensuring X_np has shape (k, n, m)."""

        # Handle dtype inference
        if dtype is None:
            dtypes = [df.dtypes[0] for df in self.X]
            if len(set(dtypes)) > 1:
                raise ValueError("Inconsistent dtypes across DataFrames.")
            dtype = dtypes[0]

        # Convert to list of 2D arrays
        X_arrays = [df.to_numpy(dtype=dtype) for df in self.X]

        # Create the arrays
        X_np = np.stack(X_arrays)  # Shape: (k, n, m)
        y_np = np.array(self.y)
        id_np = np.array(self._id)

        return X_np, y_np, id_np

    def to_dict(self):
        """Converts data to Dictionary."""
        return {"X": self.X, "y": self.y, "id": self._id}

    def to_df(self):
        """Converts data to Pandas DataFrames."""
        return DataFrame({"X": self.X, "y": self.y, "id": self._id})

    def train_test_split(self, test_size=0.2, stratify=None, random_state=None, shuffle=True, as_numpy=False):
        """Splits the dataset into training and testing subsets,
        with an option to stratify the split.
        """

        # Apply stratification if requested
        stratify_labels = self.y if stratify else None

        # Use sklearn's train_test_split
        X_train, X_test, y_train, y_test, id_train, id_test = sklearn_tts(
            self.X,
            self.y,
            self._id,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=random_state,
            shuffle=shuffle,
        )

        # Convert back to Dataset objects if requested
        train_dataset = Dataset(X_train, y_train, id_train)
        test_dataset = Dataset(X_test, y_test, id_test)

        if as_numpy:
            try:
                # Return as numpy arrays
                train_dataset = train_dataset.to_numpy()
                test_dataset = test_dataset.to_numpy()
            except ValueError:
                print("Cannot convert to numpy arrays due to length inconsistency.")
                print("Returning `Dataset` objects instead.")

        return train_dataset, test_dataset


def ArrayToDataset(X: np.ndarray, y: np.ndarray, _id: Optional[np.ndarray] = None) -> Dataset:
    """Converts a 1D NumPy array, in which each row is a DataFrame, to a
    CrossAI Dataset object. The features, labels, and instance IDs are in
    the form (features,), (labels,) and (instance IDs,).

    Args:
        X: np.ndarray of DataFrames.
        y: np.ndarray of labels.
        _id: np.ndarray of instance IDs.

    Returns:
        Dataset: The CrossAI Dataset object.
    """

    if _id is None:
        _id = []
        for i in range(len(X)):
            _id.append("No info available")
    else:
        _id = np.ndarray.tolist(_id)

    return Dataset(X=np.ndarray.tolist(X), y=np.ndarray.tolist(y), id=_id)


def ListToDataset(X, y, _id=None) -> Dataset:
    """Converts a list of DataFrames to a CrossAI Dataset object.

    Args:
        X: list of DataFrames.
        y: list of labels.
        _id: list of instance IDs.

    Returns:
        Dataset: The CrossAI Dataset object.
    """

    if _id is None:
        _id = []
        for i in range(len(X)):
            _id.append("No info available")

    return Dataset(X=X, y=y, id=_id)
