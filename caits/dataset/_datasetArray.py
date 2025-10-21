import numpy as np
from typing import Optional, Union, Tuple, Dict, List
from _coreArray import CoreArray
from _dataset import Dataset
import copy
import pandas as pd
from math import floor


class DatasetArray(Dataset):
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
            tmp_axis_names_X = {
                (name if axis == 1 else i): i for i, name in enumerate(
                    list(self.X.axis_names[f"axis_{axis}"].keys()) +
                    sum([list(o.X.axis_names[f"axis_{axis}"].keys()) for o in others], [])
                )
            }

            if axis_names is None:
                axis_names_X = copy.deepcopy(self.X.axis_names)
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
                    axis_names={"axis_1": self.y.axis_names["axis_1"]}
                )
            else:
                y = self.y

            return self.__class__(X=X, y=y)

        else:
            raise ValueError("self.X and other.X must have the same number of columns.")

    def replace(self, other):
        if len(set(self.X.axis_names["axis_1"].keys()).intersection(other.X.axis_names["axis_1"].keys())) == 0:
            raise IndexError(f"Column names {other.X.axis_names['axis_names'].keys()} not found in X or y.")

        column_names = list(other.X.axis_names["axis_1"].keys())
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
                    columns=list(self.X.axis_names["axis_1"].keys()),
                    index=list(self.X.axis_names["axis_0"].keys())
                ),
                "y": pd.DataFrame(
                    self.y.values,
                    columns=list(self.y.axis_names["axis_1"].keys()),
                    index=list(self.y.axis_names["axis_0"].keys())
                )
            }
        else:
            raise NotImplementedError("Not implemented for ndim != 2 yet.")

    def to_dict(self):
        return {"X": self.X, "y": self.y}

    def to_list(self):
        pass

    def get_axis_names_X(self):
        return copy.deepcopy(self.X.axis_names)

    def numpy_to_dataset(
            self,
            X,
            axis_names: Optional[Dict[str, Dict[Union[str, int], int]]] = None,
            split: bool = True
    ):
        dfX = CoreArray(X, axis_names=axis_names)
        return DatasetArray(X=dfX, y=self.y)

    def features_dict_to_dataset(self, features, axis_names, axis):
        features_tmp = {}
        for feat, vals in features.items():
            if vals.ndim == 1:
                features_tmp[feat] = vals
            else:
                for i in range(vals.shape[0]):
                    features_tmp[f"{feat}_{i}"] = vals[i, ...]

        features_X = np.stack([feat for feat in features_tmp.values()], axis=axis)

        axis_names[f"axis_{axis}"] = {name: i for i, name in enumerate(features_tmp.keys())}

        _axis = 1 if axis == 0 else 0

        axis_names_X = axis_names | {f"axis_{_axis}": self.X.axis_names[f"axis_{_axis}"]}

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
                axis: names for axis, names in X.axis_names.items() if axis != "axis_0"
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
            X=[CoreArray(values=x, axis_names={"axis_1": self.X.axis_names["axis_1"]}) for x in X],
            y=self.y
        )

    # TODO: Adjust
    def flatten(self, axis_names_sep=","):
        axis_names_0 = list(self.X.axis_names["axis_0"].keys())
        axis_names_1 = list(self.X.axis_names["axis_1"].keys())
        axis_names = {
            "axis_0": {
                name: i for i, name in enumerate(
                    [f"{s0}{axis_names_sep}{s1}" for s0 in axis_names_0 for s1 in axis_names_1]
                )
            }
        }

        return DatasetArray(CoreArray(self.X.values.flatten(), axis_names=axis_names), self.y)

    def shuffle(self, seed: int=42):
        idxs = np.arange(len(self.X))
        np.random.shuffle(idxs)
        return DatasetArray(
            self.X.iloc[idxs, ...],
            self.y.iloc[idxs, ...],
        )
