from typing import List
from pandas import DataFrame


class CAI:
    def __init__(
            self, X: List[DataFrame],
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
            return CAI(self.X[idx], self.y[idx], self._id[idx])
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
        return f"CAI(Dataset with {len(self)} samples)"

    # FIX: Error when batch_size not divisible with CAI len
    def batch(self, batch_size=1):
        """Yields data instances or batches from the dataset."""
        for i in range(0, len(self), batch_size):
            X_batch = self.X[i:i+batch_size]
            y_batch = self.y[i:i+batch_size]
            id_batch = self._id[i:i+batch_size]

            yield X_batch, y_batch, id_batch
