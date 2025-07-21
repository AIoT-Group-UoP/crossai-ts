from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pandas import DataFrame, concat
from typing import Optional, Union, List, Dict


class Dataset3(ABC):
    def __init__(self, X: Union[DataFrame, List[DataFrame]], y: Optional[DataFrame] = None):
        if y is None:
            self.y = DataFrame([[None] for _ in range(X.shape[0])], columns=['y_Channel_0'])
        elif len(X) != len(y):
            raise ValueError('X and y must have same number of rows')
        else:
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
    def to_dataset(self, X):
        pass

    @abstractmethod
    def train_test_split(self):
        pass

    @abstractmethod
    def apply(self, func, *args, **kwargs):
        pass


class DatasetArray(Dataset3):
    def __init__(self, X: DataFrame, y: Optional[DataFrame] = None):
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
            return self.__class__(X=concat([self.X, other.X], axis=0), y=concat([self.y, other.y], axis=0))
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

    def to_dataset(self, X):
        dfX = pd.DataFrame(X, columns=self.X.columns)
        return DatasetArray(X=dfX, y=self.y)


    def train_test_split(self, random_state: Optional[int]=None, test_size: float=0.2):
        all_idxs = np.arange(self.X.shape[0])
        Nx = int(self.X.shape[0] * (1 - test_size))

        if random_state is not None:
            train_idxs = np.random.RandomState(random_state).choice(all_idxs, Nx, replace=False)
            test_idxs = np.setdiff1d(all_idxs, train_idxs)
        else:
            train_idxs = all_idxs[:Nx]
            test_idxs = all_idxs[Nx:]

        train_X, test_X = self.X.iloc[train_idxs, :], self.X.iloc[test_idxs, :]
        train_y, test_y = self.y.iloc[train_idxs, :], self.y.iloc[test_idxs, :]

        return self.__class__(train_X, train_y), self.__class__(test_X, test_y)

    def apply(self, func, *args, **kwargs):
        return func(self.X.values, *args, **kwargs)


class DatasetList(Dataset3):
    def __init__(self, X: List[DataFrame], y=List[Union[str, int]], id=List[str]) -> None:
        super().__init__(X, y)
        if len(X) != len(id):
            raise ValueError('X and id must have same number of rows')
        else:
            self.id = id

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.id[idx]

    def __iter__(self):
        self._current = 0
        return self

    def __next__(self):
        if self._current < len(self):
            res = self.X[self._current], self.y[self._current], self.id[self._current]
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
            yield self.X[i : i + batch_size], self.y[i : i + batch_size], self.id[i : i + batch_size]

    # TODO: Add check for columns
    def unify(self, other):
        return self.__class__(
            X=self.X + other.X,
            y=self.y + other.y,
            id=self.id + other.id,
            )

    def to_numpy(self):
        pass

    def to_df(self):
        pass

    def to_dict(self):
        return {
            "X": self.X,
            "y": self.y,
            "id": self.id
        }

    def to_list(self):
        pass

    def to_dataset(self, X):
        listDfX = [pd.DataFrame(x, columns=self.X[0].columns) for x in X]
        return DatasetList(X=listDfX, y=self.y, id=self.id)

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
            train_id.append(self.id[idx])

        for idx in test_idxs:
            test_X.append(self.X[idx])
            test_y.append(self.y[idx])
            test_id.append(self.id[idx])

        return self.__class__(train_X, train_y, train_id), self.__class__(test_X, test_y, test_id)

    def apply(self, func, *args, **kwargs):
        return [func(df.values, *args, **kwargs) for df in self.X]

