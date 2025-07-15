from typing import List, Dict, TypeVar, Tuple, Union
import numpy as np
from pandas import DataFrame

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

        self._data = {"X": X}
        self.X = self._data["X"]

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Allows for dataset indexing/slicing to get a specific data point."""
        if isinstance(idx, slice):
            # Handle slicing
            tmp = {k: v[idx] for k, v in self._data.items()}
            return self.__class__(**tmp)
        elif isinstance(idx, int):
            # Handle single item selection
            tmp = tuple(self._data[k][idx] for k in self._data.keys())
            return tmp
        else:
            raise TypeError("Invalid argument type.")

    def __iter__(self):
        """Allows for iterating over the dataset."""
        self._current = 0
        return self

    def __next__(self):
        """Returns the next item from the dataset."""
        if self._current < len(self):
            # result = self.X[self._current]
            result = {k: v[self._current] for k, v in self._data.items()}
            self._current += 1
            return result
        else:
            raise StopIteration

    def __repr__(self) -> str:
        """Provide a string representation of the CAI object."""
        return f"Dataset with {len(self)} instances"

    def to_dict(self) -> Dict[str, List[Dict[str, np.ndarray]]]:
        """Return a dictionary representation of the CAI object."""
        return self._data

    def batch(self, batch_size=1):
        """Yields data instances or batches from the dataset."""
        for i in range(0, len(self), batch_size):
            batch = {k: v[i : i + batch_size] for k, v in self._data.items()}
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

        self._data["y"] = y
        self._data["id"] = id
        self.y = self._data["y"]
        self.id = self._data["id"]



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

        self._data["y"] = y
        self.y = self._data["y"]


class DatasetCLS(Dataset2):
    def __init__(self, X: List[Dict[str, np.ndarray]]) -> None:
        super().__init__(X)

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

class DatasetArray:
    def __init__(self, X, y=None, cols_X=None, cols_y=None) -> None:
        if y is None:
            self.y = np.array([[None] for _ in range(X.shape[0])])
        elif y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same length.")
        else:
            self.y = y

        self.X = X

        if cols_X is None:
            self.cols_X = {f"X_Channel_{i}": i for i in range(X.shape[1])}
        else:
            if len(cols_X) == X.shape[1]:
                self.cols_X = {cols_X[i]: i for i in range(X.shape[1])}
            else:
                raise ValueError("X Columns must have the same length.")

        if cols_y is None:
            self.cols_y = {f"y_Channel_{i}": i for i in range(self.y.shape[1])}
        else:
            if len(cols_y) == y.shape[1]:
                self.cols_y = {cols_y[i]: i for i in range(self.y.shape[1])}
            else:
                raise ValueError("y Columns must have the same length.")


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: Union[
        str,
        int,
        slice,
        Tuple[int, Union[int, slice, str, List[str]]],
        Tuple[slice, Union[int, slice, str, List[str]]],
    ]):
        if isinstance(idx, str):
            if idx in self.cols_X:
                return self.X[:, self.cols_X[idx]]
            elif idx in self.cols_y:
                return self.y[:, self.cols_y[idx]]
            else:
                raise ValueError(f"Column '{idx}' is not a valid index.")
        elif isinstance(idx, int):
            return self.X[idx, :], self.y[idx, :]
        elif isinstance(idx, slice):
            return self.X[idx], self.y[idx, :]

        elif isinstance(idx, tuple):
            if not (isinstance(idx[1], str) or (isinstance(idx[1], list) and all(isinstance(i, str) for i in idx[1]))):
                return self.X[idx]
            else:
                if isinstance(idx[1], str):
                    if idx[1] in self.cols_X:
                        return self.X[idx[0], self.cols_X[idx[1]]]
                    elif idx[1] in self.cols_y:
                        return self.y[idx[0], self.cols_y[idx[1]]]
                    else:
                        raise ValueError(f"Column '{idx[1]}' is not a valid index.")
                elif isinstance(idx[1], list):
                    if all(isinstance(i, str) for i in idx[1]):
                        if isinstance(idx[0], list):
                            idxs_X, idxs_y, cols_X, cols_y = [], [], [], []
                            for i, col in enumerate(idx[1]):
                                if col in self.cols_X.keys():
                                    idxs_X.append(i)
                                    cols_X.append(self.cols_X[col])
                                elif col in self.cols_y.keys():
                                    idxs_y.append(i)
                                    cols_y.append(self.cols_y[col])
                                else:
                                    raise ValueError(f"'{col}' is not a valid column.")

                            return (
                                self.X[idxs_X, cols_X] if cols_X else np.array([]),
                                self.y[idxs_y, cols_y] if cols_y else np.array([]),
                            )

                        elif isinstance(idx[0], int):
                            cols_X = [self.cols_X[i] for i in set(idx[1]).intersection(set(self.cols_X.keys()))]
                            cols_y = [self.cols_y[i] for i in set(idx[1]).intersection(set(self.cols_y.keys()))]

                            return (
                                self.X[[idx[0] for _ in range(len(cols_X))], cols_X] if cols_X else np.array([]),
                                self.y[[idx[0] for _ in range(len(cols_y))], cols_y] if cols_y else np.array([]),
                            )

                    elif all(isinstance(i, int) for i in idx[1]):
                        return self.X[idx]
                    else:
                        raise TypeError("Invalid argument type.")
                else:
                    raise TypeError("Invalid argument type.")

    def __iter__(self):
        self._current = 0
        return self

    def __next__(self):
        if self._current < self.X.shape[0]:
            result = self.X[self._current, :], self.y[self._current, :]
            self._current += 1
            return result
        else:
            raise StopIteration

    def __repr__(self) -> str:
        return f"Dataset object with {len(self.X)} items."

    def __add__(self, other):
        return self.unify(other)

    def cols_X_list(self):
        return [key for key, _ in sorted(self.cols_X.items(), key=lambda item: item[1])]

    def cols_y_list(self):
        return [key for key, _ in sorted(self.cols_y.items(), key=lambda item: item[1])]

    def to_dict(self):
        return {
            "X": {k: self.X[:, i] for i, k in enumerate(self.cols_X_list())},
            "y": {k: self.y[:, i] for i, k in enumerate(self.cols_y_list())},
        }

    def to_df(self):
        return {
            "X": DataFrame(self.X, columns=self.cols_X_list()),
            "y": DataFrame(self.y, columns=self.cols_y_list()),
        }

    def unify(self, other):
        if self.cols_X == other.cols_X and self.cols_y == other.cols_y:
            return DatasetArray(
                X=np.concatenate((self.X, other.X), axis=0),
                y=np.concatenate((self.y, other.y), axis=0),
                cols_X=self.cols_X_list(),
                cols_y=self.cols_y_list()
            )
        else:
            raise ValueError("DatasetArray objects must have the same column arrangement.")

    def batch(self, batch_size=1):
        for i in range(0, len(self), batch_size):
            yield self.X[i : i + batch_size, :], self.y[i : i + batch_size, :]

    def train_test_split(self, test_size: float, random_state=None):
        all_idxs = np.arange(len(self.X))
        Nx = int(self.X.shape[0] * (1-test_size))

        if random_state is None:
            train_idxs = all_idxs[:Nx]
            test_idxs = all_idxs[Nx:]
        else:
            train_idxs = np.random.RandomState(random_state).choice(all_idxs, Nx, replace=False)
            test_idxs = np.array(list(set(all_idxs).difference(set(train_idxs))))

        train_X = self.X[train_idxs, :]
        test_X = self.X[test_idxs, :]
        train_y = self.y[train_idxs, :]
        test_y = self.y[test_idxs, :]

        return (
            DatasetArray(
                X=train_X,
                y=train_y,
                cols_X=self.cols_X_list(),
                cols_y=self.cols_y_list()
            ),
            DatasetArray(
                X=test_X,
                y=test_y,
                cols_X=self.cols_X_list(),
                cols_y=self.cols_y_list()
            )
        )






