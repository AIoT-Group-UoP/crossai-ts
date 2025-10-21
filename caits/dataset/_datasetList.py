from typing import Tuple
from copy import deepcopy

import numpy as np
from typing import Optional, Union, List, Dict

import pandas as pd
from math import ceil

from _dataset import Dataset
from _coreArray import CoreArray
from _datasetArray import DatasetArray

class DatasetList(Dataset):
    def __init__(
            self,
            X: List[CoreArray],
            y: Optional[List[Union[str, int]]]=None,
            id: Optional[List[str]]=None
    ) -> None:
        if y is None:
            _y = [None for _ in range(len(X))]
        else:
            _y = y
        if id is None:
            _id = [None for _ in range(len(X))]
        else:
            _id = id

        super().__init__(X, _y)
        self._id = _id

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: Union[int, slice, List, Tuple]):
        if not isinstance(idx, tuple):
            return DatasetList(*self.__get_single(idx))
        else:
            ret = self.__get_single(idx[0])

            if isinstance(idx[1], int):
                return DatasetList(
                    X=[x.iloc[:, idx[1]] for x in ret[0]],
                    y=ret[1],
                    id=ret[2]
                )
            elif isinstance(idx[1], slice):
                if idx[1].start is None and idx[1].stop is None:
                    return DatasetList(
                        X=[x.iloc[:, idx[1]] for x in ret[0]],
                        y=ret[1],
                        id=ret[2]
                    )
                elif isinstance(idx[1].start, str) or isinstance(idx[1].stop, str):
                    return DatasetList(
                        X=[x.loc[:, idx[1]] for x in ret[0]],
                        y=ret[1],
                        id=ret[2]
                    )
                elif isinstance(idx[1].start, int) or isinstance(idx[1].stop, int):
                    return DatasetList(
                        X=[x.iloc[:, idx[1]] for x in ret[0]],
                        y=ret[1],
                        id=ret[2]
                    )

            elif isinstance(idx[1], list):
                if all([isinstance(k, int) for k in idx[1]]):
                    return DatasetList(
                        X=[x.iloc[:, idx[1]] for x in ret[0]],
                        y=ret[1],
                        id=ret[2]
                    )
                elif all([isinstance(k, str) for k in idx[1]]):
                    return DatasetList(
                        X=[x.loc[:, idx[1]] for x in ret[0]],
                        y=ret[1],
                        id=ret[2]
                    )
                else:
                    raise ValueError
            elif isinstance(idx[1], str):
                return DatasetList(
                    X=[x.loc[:, idx[1]] for x in ret[0]],
                    y=ret[1],
                    id=ret[2]
                )

    def __get_single(self, idx: Union[int, slice, List]):
        if isinstance(idx, int):
            return [self.X[idx]], [self.y[idx]], [self._id[idx]]

        elif isinstance(idx, slice):
            return self.X[idx], self.y[idx], self._id[idx]

        elif isinstance(idx, list) and all([isinstance(k, int) for k in idx]):
            return [self.X[i] for i in idx], [self.y[i] for i in idx], [self._id[i] for i in idx]

        else:
            raise ValueError

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
        return self.unify([other])

    def batch(self, batch_size: int):
        for i in range(0, len(self.X), batch_size):
            yield self.X[i : i + batch_size], self.y[i : i + batch_size], self._id[i : i + batch_size]

    def unify(self, others, axis_names: Optional = None, axis: int=0):
        if axis == 0:
            return self.__class__(
                X=self.X + sum([o.X for o in others], []),
                y=self.y + sum([o.y for o in others], []),
                id=self._id + sum([o._id for o in others], []),
                )
        elif axis == 1:
            caitsX = []
            for i in range(len(self.X)):
                if axis_names is None:
                    _axis_names = deepcopy(self.X[i].axis_names)
                    axis_1 = list(_axis_names["axis_1"].keys())
                    axis_1 += sum([list(d.X[i].axis_names["axis_1"].keys()) for d in others], [])
                    _axis_names["axis_1"] = {name: i for i, name in enumerate(axis_1)}
                else:
                    _axis_names = axis_names

                values = np.concatenate([self.X[i].values] + [d.X[i].values for d in others], axis=1)

                caitsX.append(
                    CoreArray(
                        values=values,
                        axis_names=_axis_names
                    )
                )

            caitsY = self.y
            caitsId = self._id
            return DatasetList(X=caitsX, y=caitsY, id=caitsId)
        else:
            raise ValueError("Invalid axis argument.")

    def replace(self, other):
        if len(self.X) != len(other.X):
            raise ValueError("self.X and other.X must have same length.")
        if len(self.y) != len(other.y):
            raise ValueError("self.y and other.y must have same length.")
        if len(self._id) != len(other._id):
            raise ValueError("self.id and other.id must have same length.")
        if len(set(self.X[0].axis_names[f"axis_1"].keys()).intersection(other.X[0].axis_names[f"axis_1"].keys())) == 0:
            raise ValueError("self.X[0] and other.X[0] must have same axis_name.")

        idxs = [self.X[0].axis_names["axis_1"][o] for o in other.X[0].axis_names["axis_1"].keys()]

        for i in range(len(self.X)):
            self.X[i].values[:, idxs] = other.X[i].values


    def to_numpy(self, flatten=False):
        if flatten:
            return np.stack([x.values.flatten() for x in self.X]), np.array(self.y), np.array(self._id)
        else:
            return np.stack([x.values for x in self.X]), np.array(self.y), np.array(self._id)

    def to_df(self):
        return {
            "X": {
                [pd.DataFrame(x, columns=list(self.X[0].axis_names["axis_1"].keys())) for x in self.X]
            },
            "y": pd.DataFrame(self.y),
            "id": pd.DataFrame(self._id)
        }

    def to_dict(self):
        return {
            "X": self.X,
            "y": self.y,
            "id": self._id
        }

    def to_list(self):
        pass

    def get_axis_names_X(self):
        return deepcopy(self.X[0].axis_names)

    def numpy_to_dataset(
            self,
            X,
            axis_names: Optional[Dict[str, Dict[Union[str, int], int]]] = None,
            split: bool = True
    ):
        if split:
            _X = [CoreArray(x, axis_names=axis_names) for x in X]
            return DatasetList(X=_X, y=self.y, id=self._id)
        else:
            _X = CoreArray(X)
            return DatasetArray(X=_X, y=self.y)


    def features_dict_to_dataset(self, features, axis_names, axis):
        features_tmp = {}
        for feat, values in features.items():
            if values[0].ndim == 1:
                features_tmp[feat] = values
            else:
                for i in range(values[0].shape[0]):
                    features_tmp[f"{feat}_{i}"] = [values[j][i, ...] for j in range(len(values))]

        values = [
            np.stack(
                [feat[i] for feat in features_tmp.values()],
                axis=axis,
            ) for i in range(len(list(features_tmp.values())[0]))
        ]

        axis_names[f"axis_{axis}"] = {name: i for i, name in enumerate(features_tmp.keys())}

        return self.numpy_to_dataset(values, axis_names)

    def dict_to_dataset(self, X):
        return DatasetList(**X)

    def train_test_split(self, random_state: Optional[int]=None, test_size: float=0.2, stratified: bool = False):

        if not stratified:
            all_idxs = [np.arange(len(self.X))]
        else:
            tmp = {}

            for i, y in enumerate(self.y):
                if y not in tmp.keys():
                    tmp[y] = []
                else:
                    tmp[y].append(i)

            all_idxs = tmp.values()

        train_idxs = []
        test_idxs = []
        for idxs in all_idxs:
            Nx = ceil(len(idxs) * (1 - test_size))

            if random_state is not None:
                train_idxs_part = np.random.RandomState(random_state).choice(idxs, Nx, replace=False)
                test_idxs_part = np.setdiff1d(idxs, train_idxs_part)
            else:
                train_idxs_part = idxs[:Nx]
                test_idxs_part = idxs[Nx:]
            train_idxs.append(train_idxs_part)
            test_idxs.append(test_idxs_part)

        train_idxs = np.concatenate(train_idxs, axis=0)
        test_idxs = np.concatenate(test_idxs, axis=0)

        train_X = []
        train_y = []
        train_id = []
        test_X = []
        test_y = []
        test_id = []

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

    def stack(self, data):
        X = sum(data, [])
        y = [self.y[i] for i, x in enumerate(data) for _ in x]
        id = [self._id[i] for i, x in enumerate(data) for _ in x]
        caitsX = [CoreArray(values=x, axis_names={"axis_1": self.X[0].axis_names["axis_1"]}) for x in X]
        return DatasetList(X=caitsX, y=y, id=id)

    def flatten(self, axis_names_sep=","):
        axis_names_0 = list(self.X[0].axis_names["axis_0"].keys())
        axis_names_1 = list(self.X[0].axis_names["axis_1"].keys())
        axis_names = {"axis_1": {name: i for i, name in enumerate([f"{s0}{axis_names_sep}{s1}" for s0 in axis_names_0 for s1 in axis_names_1])}}

        # return DatasetList(
        #     X=CaitsArray(np.stack([x.values.flatten() for x in self.X], axis=0), axis_names=axis_names),
        #     y=self.y,
        #     id=self._id
        # )

        return DatasetArray(
            X=CoreArray(np.stack([x.values.flatten() for x in self.X], axis=0), axis_names=axis_names),
            y=self.y,
        )

    def shuffle(self, seed: int=42):
        idxs = np.arange(len(self.X))
        np.random.RandomState(seed).shuffle(idxs)
        return DatasetList(
            X=[self.X[i] for i in idxs],
            y=[self.y[i] for i in idxs],
            id=[self._id[i] for i in idxs]
        )

