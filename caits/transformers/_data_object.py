from typing import List
from pandas import DataFrame
import numpy as np


class Dataset:
    def __init__(
            self,
            X: List[DataFrame],
            y: List[str],
            id: List[str]
    ) -> None:
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
            return Dataset([self.X[idx]], [self.y[idx]], [self._id[idx]])
        else:
            raise TypeError("Invalid argument type.")

    def __iter__(self):
        """Allows for iterating over the dataset."""
        self._current = 0
        return self

    def __next__(self):
        """Returns the next item from the dataset."""
        if self._current < len(self):
            result = (
                self.X[self._current],
                self.y[self._current],
                self._id[self._current]
            )
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
            X_batch = self.X[i:i+batch_size]
            y_batch = self.y[i:i+batch_size]
            id_batch = self._id[i:i+batch_size]

            yield X_batch, y_batch, id_batch

    def to_numpy(self, dtype=np.float32):
        """Converts data to NumPy arrays."""
        X_np = np.array(self.X, dtype=dtype)
        y_np = np.array(self.y)
        id_np = np.array(self._id)
        return X_np, y_np, id_np

    def train_test_split(self, test_size=0.2):
        """Splits the dataset into training and testing subsets."""
        total_samples = len(self)
        test_samples = int(total_samples * test_size)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        test_indices = indices[:test_samples]
        train_indices = indices[test_samples:]

        X_train = [self.X[i] for i in train_indices]
        y_train = [self.y[i] for i in train_indices]
        id_train = [self._id[i] for i in train_indices]

        X_test = [self.X[i] for i in test_indices]
        y_test = [self.y[i] for i in test_indices]
        id_test = [self._id[i] for i in test_indices]

        return Dataset(X_train, y_train, id_train), \
            Dataset(X_test, y_test, id_test)
