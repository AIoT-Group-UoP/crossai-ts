from .._core import CoreArray, CoreDataset
from typing import Tuple, Union, Optional, Dict
import numpy as np
from copy import deepcopy
import pandas as pd
from math import floor


class DatasetArray(CoreDataset):
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
            return deepcopy(DatasetArray(self.X.iloc[idx, ...], self.y.iloc[idx, ...]))
        elif isinstance(idx, slice):
            return deepcopy(DatasetArray(self.X.iloc[idx, ...], self.y.iloc[idx, ...]))
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
        return f"DatasetArray object with {len(self.X)} instances"

    def __add__(self, other):
        return self.unify([other])

    def __copy__(self):
        return deepcopy(self)

    def batch(self, batch_size: int):
        for i in range(0, self.X.shape[0], batch_size):
            yield self.X.iloc[i : i + batch_size, ...], self.y.iloc[i : i + batch_size, ...]

    def unify(
            self,
            others,
            axis_names: Optional = None,
            axis: int=0,
            to_X: bool = True,
            to_y: bool = False
    ):
        init_data = {
            "X": self.X,
            "y": self.y
        }
        other_data = {}

        if to_X:
            other_data["X"] = [o.X for o in others]

        if to_y:
            other_data["y"] = [o.y for o in others]

        new_data = init_data.copy()

        for applies, data in other_data.items():
            init_shape = init_data[applies].shape
            shape = [init_shape[i] for i in range(len(init_shape)) if i != axis]

            _axis_names = {k: v for k, v in init_data[applies].keys().items() if k != f"axis_{axis}"}

            for o in data:
                tmp_shape = [o.shape[i] for i in range(len(o.shape)) if i != axis]
                tmp_axis_names = {k: v for k, v in o.keys().items() if k != f"axis_{axis}"}

                if shape != tmp_shape:
                    raise ValueError(f"All DatasetArray.{applies} objects must have equal dimensions except the one "
                                     f"they will be appended on.")

                if _axis_names != tmp_axis_names:
                    raise ValueError(f"All DatasetArray.{applies} objects must the same axis names on all dimensions "
                                     f"except the one they will be appended.")

            if axis_names is None or applies not in axis_names.keys():
                new_axis_names = init_data[applies].keys()
                to_append = sum([o.keys()[f"axis_{axis}"] for o in data], [])
                new_axis_names[f"axis_{axis}"].extend(to_append)
            else:
                new_axis_names = axis_names[applies]

            new_data[applies] = CoreArray(
                np.concatenate(
                    [init_data[applies].values] + [o.values for o in data],
                    axis=axis
                ),
                axis_names=new_axis_names
            )

        return DatasetArray(**new_data)


    def replace(
            self,
            other,
            axis=1
    ):
        ret_data = self[:]

        column_names_X = other.X.keys()[f"axis_{axis}"]
        column_names_y = other.y.keys()[f"axis_{axis}"]

        if len(column_names_X) != 0 and (set(self.X.keys()[f"axis_{axis}"]).intersection(column_names_X)) == 0:
            raise IndexError(f"Column names {other.X.keys()[f'axis_{axis}']} not found in X.")

        if len(column_names_y) != 0 and len(set(self.y.keys()[f"axis_{axis}"]).intersection(column_names_y)) == 0:
            raise IndexError(f"Column names {other.y.keys()[f'axis_{axis}']} not found in y.")

        idxs_X = [self.X.axis_names[f"axis_{axis}"][col] for col in column_names_X]
        idxs_y = [self.y.axis_names[f"axis_{axis}"][col] for col in column_names_y]

        if len(idxs_X) != 0:
            ret_data.X.values[:, idxs_X] = other.X.values

        if len(idxs_y) != 0:
            ret_data.y.values[:, idxs_y] = other.y.values

        return ret_data

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

    def get_axis_names_y(self):
        return deepcopy(self.y.axis_names)

    @staticmethod
    def from_numpy(
            X,
            y,
            axis_names_X: Optional[Dict[str, Dict[Union[str, int], int]]] = None,
            axis_names_y = None,
    ):
        dfy = CoreArray(y, axis_names=axis_names_y)
        dfX = CoreArray(X, axis_names=axis_names_X)

        return DatasetArray(X=dfX, y=dfy)

    @staticmethod
    def features_dict_to_dataset(
            features: dict,
            axis_names,
            axis,
            to_X = True,
            to_y = False
    ):
        features_arrs = {}

        if not to_X:
            features_arrs["X"] = features.pop("X")

        if not to_y:
            features_arrs["y"] = features.pop("y")

        for part, feats in features.items():
            features_tmp = {}

            for feat, vals in feats.items():
                if vals.ndim == 1:
                    features_tmp[feat] = vals
                else:
                    for i in range(vals.shape[0]):
                        features_tmp[f"{feat}_{i}"] = vals[i, ...]

            features_stacked = np.stack([feat for feat in features_tmp.values()], axis=axis)
            axis_names[part][f"axis_{axis}"] = list(features_tmp.keys())

            features_arrs[part] = CoreArray(features_stacked, axis_names=axis_names[part])

        return DatasetArray(**features_arrs)

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

    def apply(
            self,
            func,
            to_X=True,
            to_y=False,
            export_to=None,
            *args,
            **kwargs
    ):
        if to_X:
            X_tr = func(self.X.values, *args, **kwargs)
        else:
            X_tr = self.X.values

        if to_y:
            y_tr = func(self.y.values, *args, **kwargs)
        else:
            y_tr = self.y.values

        if export_to is None or export_to == "tuple":
            return X_tr, y_tr
        elif export_to == "dict":
            return {
                "X": X_tr,
                "y": y_tr
            }
        else:
            raise Exception(f"export_to {export_to} is not supported.")


    def apply_windowing(self, func, *args, **kwargs):
        X_vals, y_vals = self.apply(func, to_X=True, to_y=True, *args, **kwargs)

        axis_names = {
            "X": deepcopy(self.X.axis_names),
            "y": deepcopy(self.y.axis_names)
        }

        X = np.stack(X_vals)
        y = np.stack(y_vals)

        del axis_names["X"]["axis_0"]
        del axis_names["y"]["axis_0"]

        axis_names_X = {
            "axis_0": [f"# Window_{i}" for i in range(len(y_vals))],
            "axis_2": axis_names["X"]["axis_1"]
        }

        axis_names_y = {
            "axis_0": [f"Window_{i}" for i in range(len(y_vals))],
            "axis_2": axis_names["y"]["axis_1"]
        }

        X_core = CoreArray(X, axis_names_X)
        y_core = CoreArray(y, axis_names_y)

        return DatasetArray(
            X=X_core,
            y=y_core
        )


    # TODO: Adjust
    def flatten(
            self,
            to_X=True,
            to_y=False,
            axis_names_sep=","
    ):
        init_axis_names = {}
        axis_names = {}
        parts = {
            "X": self.X,
            "y": self.y
        }

        if to_X:
            init_axis_names["X"] = self.X.keys()
            axis_names["X"] = {}
        if to_y:
            init_axis_names["y"] = self.y.keys()
            axis_names["y"] = {}

        for part in axis_names.keys():
            axis_names_0 = init_axis_names[part]["axis_0"]
            axis_names_1 = init_axis_names[part]["axis_1"]
            axis_names_2 = init_axis_names[part]["axis_2"]

            tmp_axis_names = {
                "axis_0": axis_names_0,
                "axis_1": [
                    f"{s1}{axis_names_sep}{s2}"
                    for s1 in axis_names_1
                    for s2 in axis_names_2
                ]
            }

            axis_names[part] = tmp_axis_names

            vals = parts[part].values
            parts[part] = CoreArray(
                vals.reshape(vals.shape[0], -1),
                axis_names=axis_names[part]
            )

        return DatasetArray(**parts)


    def reshape(
            self,
            shape_X,
            shape_y,
            axis_names_X,
            axis_names_y
    ):
        if shape_y is not None:
            _y = self.y.values.reshape(shape_y)
        else:
            _y = self.y.values

        if shape_X is not None:
            _X = self.X.values.reshape(shape_X)
        else:
            _X = self.X.values

        return DatasetArray.from_numpy(
            _X,
            _y,
            axis_names_X=axis_names_X,
            axis_names_y=axis_names_y
        )


    def shuffle(self, seed: int=42):
        idxs = np.arange(len(self.X))
        np.random.shuffle(idxs)
        return DatasetArray(
            self.X.iloc[idxs, ...],
            self.y.iloc[idxs, ...],
        )

    def rename(self, renamings: dict):
        new_dataset_arr = deepcopy(self)

        for part in renamings.keys():
            for axis, contents in renamings[part].items():
                for old_name, new_name in contents.items():
                    if part == "X":
                        data_part = new_dataset_arr.X
                    else:
                        data_part = new_dataset_arr.y

                    if new_name in data_part.axis_names[axis]:
                        raise ValueError(f"renamings[{part}][{axis}][{new_name}] already exists.")

                    value = data_part.axis_names[axis].pop(old_name)
                    data_part.axis_names[axis][new_name] = value

        return new_dataset_arr


