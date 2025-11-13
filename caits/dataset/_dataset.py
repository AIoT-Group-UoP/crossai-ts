from typing import Tuple
from copy import deepcopy
import numpy as np
from typing import Optional, Union, List, Dict
import pandas as pd
from math import ceil, floor
from ._coreArray import CoreArray
from ._datasetBase import DatasetBase
import itertools


class DatasetArray(DatasetBase):
    def __init__(self, X: CoreArray, y: Optional[CoreArray] = None):
        if y is None:
            _y = CoreArray(np.array([[None] for _ in range(len(X))]))
        else:
            _y = y
        super().__init__(X, _y)

    def __len__(self):
        return self.X.shape[0]

    def __iter__(self):
        self._current = 0
        return self

    def __getitem__(self, idx: Union[int, slice, list, Tuple]):
        if isinstance(idx, int):
            return DatasetArray(self.X.iloc[idx, ...], self.y.iloc[idx, ...])
        elif isinstance(idx, slice):
            return DatasetArray(self.X.iloc[idx, ...], self.y.iloc[idx, ...])
        elif isinstance(idx, list):
            return DatasetArray(self.X.iloc[idx, ...], self.y.iloc[idx, ...])
        elif isinstance(idx, tuple):
            if isinstance(idx[0], int) and isinstance(idx[1], int):
                return (self.X.iloc[idx], self.y.iloc[idx] if idx[1] < self.y.shape[1] else None)
            elif isinstance(idx[0], slice) and isinstance(idx[1], int):
                return DatasetArray(
                    self.X.iloc[idx],
                    self.y.iloc[idx] if idx[1] < self.y.shape[1] else None
                )
            elif isinstance(idx[0], list) and isinstance(idx[1], int):
                return DatasetArray(
                    self.X.iloc[idx],
                    self.y.iloc[idx] if idx[1] < self.y.shape[1] else None
                )
            elif (isinstance(idx[0], int) or isinstance(idx[0], slice) or isinstance(idx[0], list)) and isinstance(idx[1], slice):
                if isinstance(idx[1].start, str):
                    if idx[1].start in self.X.axis_names["axis_1"].keys():
                        return DatasetArray(self.X.loc[idx], self.y)
                    elif idx[1].start in self.y.axis_names["axis_1"].keys():
                        return DatasetArray(self.X, self.y.loc[idx])
                    else:
                        raise IndexError
                elif isinstance(idx[1].stop, str):
                    if idx[1].stop in self.X.axis_names["axis_1"].keys():
                        return DatasetArray(self.X.loc[idx], self.y)
                    elif idx[1].stop in self.y.axis_names["axis_1"].keys():
                        return DatasetArray(self.X, self.y.loc[idx])
                    else:
                        raise IndexError
                else:
                    return DatasetArray(self.X.iloc[idx], self.y.iloc[idx])
            elif isinstance(idx[1], list):
                if all([isinstance(i, str) for i in idx[1]]):
                    cols_map = {col: i for i, col in enumerate(idx[1])}
                    cols = set(idx[1])
                    colsX = list(cols.intersection(set(self.X.axis_names["axis_1"].keys())))
                    colsY = list(cols.intersection(set(self.y.axis_names["axis_1"].keys())))
                    if isinstance(idx[0], list):
                        rowsX = [idx[0][cols_map[col]] for col in colsX]
                        rowsY = [idx[0][cols_map[col]] for col in colsY]
                        return DatasetArray(self.X.loc[(rowsX, colsX)], self.y.loc[(rowsY, colsY)])
                    elif isinstance(idx[0], slice):
                        return DatasetArray(self.X.loc[(idx[0], colsX)], self.y.loc[(idx[0], colsY)])
                    else:
                        raise IndexError
                elif all([isinstance(i, int) for i in idx[1]]):
                    return DatasetArray(self.X.iloc[idx], self.y.iloc[idx])
                else:
                    raise IndexError
            elif isinstance(idx[1], str):
                if idx[1] in self.X.axis_names["axis_1"].keys():
                    return DatasetArray(self.X.loc[idx], self.y)
                elif idx[1] in self.y.axis_names["axis_1"].keys():
                    return DatasetArray(self.X, self.y.loc[idx])
                else:
                    raise IndexError(f"Column name {idx[1]} not found in X or y.")
            else:
                raise IndexError(f"Column name {idx[1]} not found in X or y.")
        else:
            raise IndexError(f"Column name {idx} not found in X or y.")

    def __next__(self):
        if self._current < len(self):
            res = self.X.iloc[self._current, ...], self.y.iloc[self._current, ...]
            self._current += 1
            return res
        else:
            raise StopIteration

    def __repr__(self):
        return f"DatasetArray object with {len(self.X)} instances."

    def __add__(self, other):
        return self.unify([other])

    def batch(self, batch_size: int):
        for i in range(0, self.X.shape[0], batch_size):
            yield self.X.iloc[i : i + batch_size, ...], self.y.iloc[i : i + batch_size, ...]

    def unify(self, others, axis_names: Optional = None, axis: int=0):
        _axis = 1 if axis == 0 else 0

        if all(
                [
                    self.X.shape[_axis] == o.X.shape[_axis] and
                    self.X.axis_names[f"axis_{_axis}"] == o.X.axis_names[f"axis_{_axis}"]
                    for o in others
                ]
        ):
            tmp_axis_names_X = self.X.keys()[f"axis_{axis}"] + sum([o.X.keys()[f"axis_{axis}"] for o in others], [])

            if axis_names is None:
                axis_names_X = self.X.keys()
                axis_names_X[f"axis_{axis}"] = tmp_axis_names_X
            else:
                axis_names_X = axis_names

            X = CoreArray(
                np.concatenate(
                    [self.X.values] + [o.X.values for o in others],
                    axis=axis
                ),
                axis_names=axis_names_X
            )

            if axis == 0:
                y = CoreArray(
                    np.concatenate(
                        [self.y.values] + [o.y.values for o in others],
                        axis=0
                    ),
                    axis_names={"axis_1": self.y.keys()["axis_1"]}
                )
            else:
                y = self.y

            return self.__class__(X=X, y=y)

        else:
            raise ValueError("self.X and other.X must have the same number of columns.")

    def replace(self, other):
        if len(set(self.X.keys()["axis_1"]).intersection(other.X.keys()["axis_1"])) == 0:
            raise IndexError(f"Column names {other.X.axis_names['axis_names'].keys()} not found in X or y.")

        column_names = other.X.keys()["axis_1"]
        idxs = [self.X.axis_names["axis_1"][col] for col in column_names]

        self.X.values[:, idxs] = other.X.values

    # TODO: Adjust
    def to_numpy(self, flatten=False):
        if flatten:
            return self.X.values.flatten(), self.y.values
        else:
            return self.X.values, self.y.values

    def to_df(self):
        if self.X.ndim == 2:
            return {
                "X": pd.DataFrame(
                    self.X.values,
                    columns=self.X.keys()["axis_1"],
                    index=self.X.keys()["axis_0"]
                ),
                "y": pd.DataFrame(
                    self.y.values,
                    columns=self.y.keys()["axis_1"],
                    index=self.y.keys()["axis_0"]
                )
            }
        else:
            raise NotImplementedError("Not implemented for ndim != 2 yet.")

    def to_dict(self):
        return {"X": self.X, "y": self.y}

    def to_list(self):
        pass

    def get_axis_names_X(self):
        return deepcopy(self.X.axis_names)

    def numpy_to_dataset(
            self,
            X,
            axis_names: Optional[Dict[str, Dict[Union[str, int], int]]] = None,
            split: bool = False
    ):
        if not split:
            dfX = CoreArray(X, axis_names=axis_names)
            return DatasetArray(X=dfX, y=self.y)
        else:
            dfX = [CoreArray(x, axis_names=axis_names) for x in X]
            return DatasetList(X=dfX, y=self.y)

    def features_dict_to_dataset(self, features, axis_names, axis):
        features_tmp = {}
        for feat, vals in features.items():
            if vals.ndim == 1:
                features_tmp[feat] = vals
            else:
                for i in range(vals.shape[0]):
                    features_tmp[f"{feat}_{i}"] = vals[i, ...]

        features_X = np.stack([feat for feat in features_tmp.values()], axis=axis)

        axis_names[f"axis_{axis}"] = list(features_tmp.keys())

        _axis = 1 if axis == 0 else 0

        axis_names_X = axis_names | {f"axis_{_axis}": self.X.keys()[f"axis_{_axis}"]}

        return DatasetArray(
            X=CoreArray(features_X, axis_names=axis_names_X),
            y=self.y
        )

    # TODO: something is wrong
    def dict_to_dataset(self, X):
        vals = np.stack([row for row in X.values()])
        dfX = CoreArray(
            vals,
            axis_names={
                axis: names for axis, names in X.keys().items() if axis != "axis_0"
            }
        )
        return dfX

    def train_test_split(
            self,
            random_state: Optional[int]=None,
            test_size: float=0.2,
            stratified=False
    ):
        if not stratified:
            all_idxs = [np.arange(self.X.shape[0])]
        else:
            y_unique_vals = list({tuple(row) for row in self.y.values})
            all_idxs = []
            for t in y_unique_vals:
                all_idxs.append(np.where((self.y.values == t).all(axis=1))[0])

        train_idxs = []
        test_idxs = []
        for idxs in all_idxs:
            Nx = floor(len(idxs) * (1 - test_size))

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

        train_X = self.X.iloc[train_idxs, ...]
        test_X = self.X.iloc[test_idxs, ...]
        train_y = self.y.iloc[train_idxs, ...]
        test_y = self.y.iloc[test_idxs, ...]

        return self.__class__(train_X, train_y), self.__class__(test_X, test_y)

    def apply(self, func, *args, **kwargs):
        X = func(self.X.values, *args, **kwargs)
        return X

    def stack(self, X: List[np.ndarray]):
        return DatasetList(
            X=[
                CoreArray(values=x, axis_names={"axis_1": self.X.keys()["axis_1"]})
                for x in X
            ],
            y=self.y
        )

    # TODO: Adjust
    def flatten(self, axis_names_sep=","):
        axis_names_0 = self.X.keys()["axis_0"]
        axis_names_1 = self.X.keys()["axis_1"]
        axis_names = {
            "axis_0": [
                f"{s0}{axis_names_sep}{s1}"
                for s0 in axis_names_0
                for s1 in axis_names_1
            ]
        }

        return DatasetArray(CoreArray(self.X.values.flatten(), axis_names=axis_names), self.y)

    def shuffle(self, seed: int=42):
        idxs = np.arange(len(self.X))
        np.random.shuffle(idxs)
        return DatasetArray(
            self.X.iloc[idxs, ...],
            self.y.iloc[idxs, ...],
        )



class DatasetList(DatasetBase):
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
                    _axis_names = self.X[i].keys()
                    axis_1 = _axis_names["axis_1"]
                    axis_1 += sum([list(d.X[i].keys()["axis_1"]) for d in others], [])
                    _axis_names["axis_1"] = axis_1
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
                [pd.DataFrame(x, columns=list(self.X[0].keys()["axis_1"])) for x in self.X]
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

        axis_names[f"axis_{axis}"] = list(features_tmp.keys())

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
        axis_names = self.X[0].keys()
        axis_names_list = []
        for i in range(len(axis_names)):
            axis_names_list.append(axis_names[f"axis_{i}"])

        axis_names_combs = []
        for comb in itertools.product(*axis_names_list):
            axis_names_combs.append(axis_names_sep.join([str(x) for x in comb]))

        axis_names = {
            "axis_1": axis_names_combs,
        }

        flattened_values = np.stack([x.values.flatten() for x in self.X], axis=0)

        return DatasetArray(
            X=CoreArray(
                flattened_values,
                axis_names=axis_names
            ),
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

