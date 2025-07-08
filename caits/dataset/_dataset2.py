from typing import List, Dict, TypeVar, Tuple, Union
import numpy as np

T = TypeVar('T', bound="Dataset")

class Dataset2:
    def __init__(self, X: List[Dict[str, np.ndarray]]) -> None:
        if not isinstance(X, list):
            raise TypeError("X must be a list.")
        if not all(isinstance(data, dict) for data in X):
            raise TypeError("Items in X must be dicts.")
        if not all(isinstance(key, str) for data in X for key in data.keys()):
            raise TypeError("Each key of each dictionary must be a string.")
        if not all(isinstance(arr, np.ndarray) for data in X for arr in data.values()):
            raise TypeError("Each value of each dictionary in X must be a numpy.array.")

        # Check that all inputs have the same length
        for d in X:
            if len({len(v) for v in d.values()}) != 1:
                raise ValueError("Arrays must have the same length in each dictionary.\t")

        self.X = X

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Allows for dataset indexing/slicing to get a specific data point."""
        data = self.to_dict()

        if isinstance(idx, slice):
            # Handle slicing
            tmp = {k: v[idx] for k, v in data.items()}
            return self.__class__(**tmp)
        elif isinstance(idx, int):
            # Handle single item selection
            tmp = tuple(data[k][idx] for k in data.keys())
            return tmp
        else:
            raise TypeError("Invalid argument type.")

    def __iter__(self):
        """Allows for iterating over the dataset."""
        self._current = 0
        return self

    def __next__(self):
        """Returns the next item from the dataset."""
        data = self.to_dict()
        if self._current < len(self):
            # result = self.X[self._current]
            result = {k: v[self._current] for k, v in data.items()}
            self._current += 1
            return result
        else:
            raise StopIteration

    def __repr__(self) -> str:
        """Provide a string representation of the CAI object."""
        return f"Dataset with {len(self)} instances"

    def to_dict(self) -> Dict[str, List[Dict[str, np.ndarray]]]:
        """Return a dictionary representation of the CAI object."""
        return {"X": self.X}

    def batch(self, batch_size=1):
        """Yields data instances or batches from the dataset."""
        data = self.to_dict()

        for i in range(0, len(self), batch_size):
            batch = {k: v[i : i + batch_size] for k, v in data.items()}
            yield batch

    def unify(self: T, other: T) -> T:
        """Concatenates two Dataset objects by appending their rows."""
        data1 = self.to_dict()
        data2 = other.to_dict()

        if data1.keys() != data2.keys():
            raise ValueError("Datasets must have the same keys.")

        data = {k: data1[k] + data2[k] for k in data1.keys()}

        return self.__class__(**data)

    def train_test_split(self, test_size=0.2, random_state=None) -> Tuple[T, T]:
        """Splits the dataset into training and testing subsets,
        with an option to stratify the split.
        """

        Nx = int(len(self.X) * (1 - test_size))
        idxs = np.arange(len(self.X))

        if random_state is None:
            train_idxs = idxs[:Nx]
        else:
            train_idxs = np.random.RandomState(random_state).choice(idxs, Nx, replace=False)
        test_idxs = np.array(list(set(idxs).difference(set(train_idxs))))

        train, test = {}, {}

        for k, v in self.to_dict().items():
            train[k] = [v[idx] for idx in train_idxs]
            test[k] = [v[idx] for idx in test_idxs]

        train_dataset = self.__class__(**train)
        test_dataset = self.__class__(**test)

        return train_dataset, test_dataset


class DatasetCLF(Dataset2):
    def __init__(self, X: List[Dict[str, np.ndarray]], y: List[Union[str, int]], id: List[str]) -> None:
        super().__init__(X)

        if not isinstance(y, list):
            raise TypeError("y must be a list.")
        if not all(isinstance(data, int) or isinstance(data, str) for data in y):
            raise TypeError("Items in y must be dicts.")
        if not isinstance(id, list):
            raise TypeError("y must be a list.")
        if not all(isinstance(data, str) for data in y):
            raise TypeError("Items in y must be dicts.")

        # Check that all inputs have the same length
        for d in y:
            if len({len(v) for v in d.values()}) != 1:
                raise ValueError("Arrays must have the same length in each dictionary.\t")

        self.y = y



class DatasetRGR(Dataset2):
    def __init__(self, X: List[Dict[str, np.ndarray]], y: List[Dict[str, np.ndarray]]) -> None:
        super().__init__(X)

        if not isinstance(y, list):
            raise TypeError("y must be a list.")
        if not all(isinstance(data, dict) for data in y):
            raise TypeError("Items in y must be dicts.")
        if not all(isinstance(key, str) for data in y for key in data.keys()):
            raise TypeError("Each key of each dictionary must be a string.")
        if not all(isinstance(arr, np.ndarray) for data in y for arr in data.values()):
            raise TypeError("Each value of each dictionary in y must be a numpy.array.")

        # Check that all inputs have the same length
        for d in y:
            if len({len(v) for v in d.values()}) != 1:
                raise ValueError("Arrays must have the same length in each dictionary.\t")

        self.y = y

    def to_dict(self) -> Dict[str, List[Dict[str, np.ndarray]]]:
        """Return a dictionary representation of the CAI object."""
        return {"X": self.X, "y": self.y}


# def ArrayToDataset(X: np.ndarray, y: np.ndarray, _id: Optional[np.ndarray] = None) -> Dataset:
#     """Converts a 1D NumPy array, in which each row is a DataFrame, to a
#     CrossAI Dataset object. The features, labels, and instance IDs are in
#     the form (features,), (labels,) and (instance IDs,).
#
#     Args:
#         X: np.ndarray of DataFrames.
#         y: np.ndarray of labels.
#         _id: np.ndarray of instance IDs.
#
#     Returns:
#         Dataset: The CrossAI Dataset object.
#     """
#
#     if _id is None:
#         _id = []
#         for i in range(len(X)):
#             _id.append("No info available")
#     else:
#         _id = np.ndarray.tolist(_id)
#
#     return Dataset(X=np.ndarray.tolist(X), y=np.ndarray.tolist(y), id=_id)
#
#
# def ListToDataset(X, y, _id=None) -> Dataset:
#     """Converts a list of DataFrames to a CrossAI Dataset object.
#
#     Args:
#         X: list of DataFrames.
#         y: list of labels.
#         _id: list of instance IDs.
#
#     Returns:
#         Dataset: The CrossAI Dataset object.
#     """
#
#     if _id is None:
#         _id = []
#         for i in range(len(X)):
#             _id.append("No info available")
#
#     return Dataset(X=X, y=y, id=_id)
