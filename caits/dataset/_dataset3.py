from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from typing import Optional, Union, List, Dict
import copy


class CaitsArray:
    class _iLocIndexer:
        def __init__(self, parent) -> None:
            self.parent = parent

        def __getitem__(self, index):
            if len(index) != self.parent.ndim:
                raise ValueError(f'Index must be {self.parent.ndim} dimensional')
            else:
                return self.parent.values[index]

    class _LocIndexer:
        def __init__(self, parent, indexer):
            self.parent = parent
            self.indexer = indexer

        def __getitem__(self, index):
            if len(index) != self.parent.ndim:
                raise ValueError(f'Index must be {self.parent.parent.values.ndim} dimensional')
            else:
                idxs = []
                for i, t in enumerate(index):
                    if isinstance(t, str):
                        idxs.append(self.parent.axis_names[f"axis_{i}"][t])
                    elif isinstance(t, list):
                        idxs.append([self.parent.axis_names[f"axis_{i}"][j] for j in t])
                    elif isinstance(t, slice):
                        idxs.append(
                            slice(
                                self.parent.axis_names[f"axis_{i}"][t.start] if t.start is not None else None,
                                self.parent.axis_names[f"axis_{i}"][t.stop] if t.stop is not None else None,
                                t.step
                            )
                        )
                    else:
                        raise IndexError("Unsupported index type")

                return self.parent.values[*idxs]


    def __init__(self, values: np.ndarray, axis_names=Optional[Dict]):
        self.values = values
        self.axis_names = {f"axis_{i}": {} for i in range(values.ndim)}
        self.shape = values.shape
        self.ndim = values.ndim

        if len(axis_names) > values.ndim:
            raise ValueError("Axis names must not exceed number of dimensions")
        for i, axis in enumerate([f"axis_{j}" for j in range(values.ndim)]):
            if axis in axis_names.keys():
                if len(axis_names[axis]) != values.shape[i]:
                    raise ValueError(
                        f"Shapes {[len(axis) for axis in axis_names.values()]} "
                        f"and {list(values.shape)} don't match.")
                else:
                    self.axis_names[axis] = copy.deepcopy(axis_names[axis])
            else:
                self.axis_names[axis] = {j: j for j in range(values.shape[i])}

        self.loc = self._LocIndexer(self, self.axis_names)
        self.iloc = self._iLocIndexer(self)

    def __len__(self):
        return self.values.shape[0]


class Dataset3(ABC):
    def __init__(self, X: Union[CaitsArray, List[CaitsArray]], y: Optional[Union[CaitsArray, List]] = None):
        if y is None:
            self.y = CaitsArray(np.array([[None] for _ in range(X.shape[0])]), axis_names={"axis_1": "y_Channel_0"})
        self.X = X
        self.y = y

    @abstractmethod
    def __len__(self):
        return len(self.X)

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    @abstractmethod
    def __iter__(self):
        self._current = 0
        return self

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def batch(self, batch_size: int):
        pass

    @abstractmethod
    def unify(self, other):
        pass

    @abstractmethod
    def to_numpy(self):
        pass

    @abstractmethod
    def to_df(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def to_list(self):
        pass

    @abstractmethod
    def numpy_to_dataset(self, X):
        pass

    @abstractmethod
    def dict_to_dataset(self, X):
        pass

    @abstractmethod
    def train_test_split(self):
        pass

    @abstractmethod
    def apply(self, func, *args, **kwargs):
        pass


class DatasetArray(Dataset3):
    def __init__(self, X: CaitsArray, y: Optional[CaitsArray] = None):
        super().__init__(X, y)

    def __len__(self):
        return self.X.shape[0]

    def __iter__(self):
        self._current = 0
        return self

    def __getitem__(self, idx: int):
        return self.X.iloc[idx, :]

    def __next__(self):
        if self._current < len(self):
            res = self.X.iloc[self._current, :], self.y.iloc[self._current, :]
            self._current += 1
            return res
        else:
            raise StopIteration

    def __repr__(self):
        return f"DatasetArray object with {len(self.X)} instances."

    def __add__(self, other):
        return self.unify(other)

    def batch(self, batch_size: int):
        for i in range(0, self.X.shape[0], batch_size):
            yield self.X.iloc[i : i + batch_size, :], self.y.iloc[i : i + batch_size, :]

    def unify(self, other):
        if self.X.shape[1] == other.X.shape[1] and self.y.shape[1] == other.y.shape[1]:
            axis_0_names = {
                i: i for i in range(
                    len(self.X.axis_names["axis_0"]) + len(other.X.axis_names["axis_0"])
                )
            }
            axis_names_X = copy.deepcopy(self.X.axis_names)
            axis_names_y = copy.deepcopy(self.y.axis_names)
            axis_names_X["axis_0"] = axis_0_names
            axis_names_y["axis_0"] = axis_0_names

            X = CaitsArray(np.concatenate([self.X.values, other.X.values], axis=0), axis_names=axis_names_X)
            y = CaitsArray(np.concatenate([self.y.values, other.y.values], axis=0), axis_names=axis_names_y)

            return self.__class__(X=X, y=y)
        elif self.X.shape[1] != other.X.shape[1]:
            raise ValueError("self.X and other.X must have the same number of columns.")
        else:
            raise ValueError("self.X and other.y must have the same number of columns.")

    def to_numpy(self):
        return self.X.values, self.y.values

    def to_df(self):
        pass

    def to_dict(self):
        return {"X": self.X, "y": self.y}

    def to_list(self):
        pass

    def numpy_to_dataset(self, X):
        dfX = CaitsArray(X, axis_names=self.X.axis_names)
        return DatasetArray(X=dfX, y=self.y)

    def dict_to_dataset(self, X):
        vals = np.stack([row for row in X.values()])
        axis_names = copy.deepcopy(self.X.axis_names)
        axis_names["axis_0"] = {key: i for i, key in enumerate(X.keys())}
        dfX = CaitsArray(vals, axis_names=axis_names)
        return dfX

    def train_test_split(self, random_state: Optional[int]=None, test_size: float=0.2):
        all_idxs = np.arange(self.X.shape[0])
        Nx = int(self.X.shape[0] * (1 - test_size))

        if random_state is not None:
            train_idxs = np.random.RandomState(random_state).choice(all_idxs, Nx, replace=False)
            test_idxs = np.setdiff1d(all_idxs, train_idxs)
        else:
            train_idxs = all_idxs[:Nx]
            test_idxs = all_idxs[Nx:]

        train_axis_names_X = copy.deepcopy(self.X.axis_names)
        train_axis_names_X["axis_0"] = {j: i for i, j in enumerate(train_idxs)}
        test_axis_names_X = copy.deepcopy(self.X.axis_names)
        test_axis_names_X["axis_0"] = {j: i for i, j in enumerate(test_idxs)}

        train_axis_names_y = copy.deepcopy(self.y.axis_names)
        train_axis_names_y["axis_0"] = {i: i for i in train_idxs}
        test_axis_names_y = copy.deepcopy(self.y.axis_names)
        test_axis_names_y["axis_0"] = {i: i for i in test_idxs}

        train_X = CaitsArray(self.X.iloc[train_idxs, :], axis_names=train_axis_names_X)
        test_X = CaitsArray(self.X.iloc[test_idxs, :], axis_names=test_axis_names_X)
        train_y = CaitsArray(self.y.iloc[train_idxs, :], axis_names=train_axis_names_y)
        test_y = CaitsArray(self.y.iloc[test_idxs, :], axis_names=test_axis_names_y)

        return self.__class__(train_X, train_y), self.__class__(test_X, test_y)

    def apply(self, func, *args, **kwargs):
        return func(self.X.values, *args, **kwargs)


class DatasetList(Dataset3):
    def __init__(self, X: List[CaitsArray], y=List[Union[str, int]], id=List[str]) -> None:
        super().__init__(X, y)
        self._id = id

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self._id[idx]

    def __iter__(self):
        self._current = 0
        return self

    def __next__(self):
        if self._current < len(self):
            res = self.X[self._current], self.y[self._current], self._id[self._current]
            self._current += 1
            return res
        else:
            raise StopIteration

    def __repr__(self):
        return f"DatasetList object with {len(self.X)} instances."

    def __add__(self, other):
        return self.unify(other)

    def batch(self, batch_size: int):
        for i in range(0, len(self.X), batch_size):
            yield self.X[i : i + batch_size], self.y[i : i + batch_size], self._id[i : i + batch_size]

    # TODO: Add check for columns
    def unify(self, other):
        return self.__class__(
            X=self.X + other.X,
            y=self.y + other.y,
            id=self._id + other._id,
            )

    def to_numpy(self):
        pass

    def to_df(self):
        pass

    def to_dict(self):
        return {
            "X": self.X,
            "y": self.y,
            "_id": self._id
        }

    def to_list(self):
        pass

    def numpy_to_dataset(self, X):
        axis_names = copy.deepcopy(self.X[0].axis_names)
        del axis_names["axis_0"]
        listDfX = [CaitsArray(x, axis_names=axis_names) for x in X]
        return DatasetList(X=listDfX, y=self.y, id=self._id)

    def dict_to_dataset(self, X):
        vals = [np.stack([X[k][i] for k in X.keys()]) for i in range(len(X[list(X.keys())[0]]))]
        axis_names = copy.deepcopy(self.X[0].axis_names)
        axis_names["axis_0"] = {col: i for i, col in enumerate(X.keys())}
        listDfX = [CaitsArray(x, axis_names=axis_names) for x in vals]
        return DatasetList(X=listDfX, y=self.y, id=self._id)

    def train_test_split(self, random_state: Optional[int]=None, test_size: float=0.2):
        all_idxs = np.arange(len(self.X))
        Nx = int(len(self.X) * (1 - test_size))
        if random_state is not None:
            train_idxs = np.random.RandomState(random_state).choice(all_idxs, Nx, replace=False)
            test_idxs = np.setdiff1d(all_idxs, train_idxs)
        else:
            train_idxs = all_idxs[:Nx]
            test_idxs = all_idxs[Nx:]

        train_X, test_X, train_y, test_y, train_id, test_id = [], [], [], [], [], []

        for idx in train_idxs:
            train_X.append(self.X[idx])
            train_y.append(self.y[idx])
            train_id.append(self._id[idx])

        for idx in test_idxs:
            test_X.append(self.X[idx])
            test_y.append(self.y[idx])
            test_id.append(self._id[idx])

        return self.__class__(train_X, train_y, train_id), self.__class__(test_X, test_y, test_id)

    def apply(self, func, *args, **kwargs):
        return [func(df.values, *args, **kwargs) for df in self.X]

