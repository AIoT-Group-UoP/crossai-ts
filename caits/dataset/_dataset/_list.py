from .._core import CoreArray, CoreDataset
from typing import List, Optional, Tuple, Union, Dict
from copy import deepcopy
import numpy as np
import pandas as pd
from math import ceil
import itertools


class DatasetList(CoreDataset):
    def __init__(
            self,
            X: List[CoreArray],
            y: Optional[CoreArray]=None,
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
            return [self.X[idx]], self.y.iloc[[idx], ...], [self._id[idx]]

        elif isinstance(idx, slice):
            return self.X[idx], self.y.iloc[idx, ...], self._id[idx]

        elif isinstance(idx, list) and all([isinstance(k, int) for k in idx]):
            return [self.X[i] for i in idx], [self.y.iloc[i, ...] for i in idx], [self._id[i] for i in idx]

        else:
            raise ValueError

    def __iter__(self):
        self._current = 0
        return self

    def __next__(self):
        if self._current < len(self):
            res = self.X[self._current], self.y.iloc[self._current, ...], self._id[self._current]
            self._current += 1
            return res
        else:
            raise StopIteration

    def __repr__(self):
        return f"DatasetList object with {len(self.X)} instances"

    def __add__(self, other):
        return DatasetList.concat([self, other])

    def batch(self, batch_size: int):
        for i in range(0, len(self.X), batch_size):
            yield self.X[i : i + batch_size], self.y.iloc[i : i + batch_size, ...], self._id[i : i + batch_size]

    @staticmethod
    def concat(
            init_data,
            axis_names: Optional = None,
            axis: int=0,
            to_X: bool = True,
            to_y: bool = True
    ):
        if axis == 0:
            y_vals = np.concatenate([o.y.values for o in init_data])

            return DatasetList(
                X=sum([o.X for o in init_data], []),
                y=CoreArray(values=y_vals, axis_names={"axis_1": init_data[0].y.keys()["axis_1"]}),
                id=sum([o._id for o in init_data], []),
            )
        elif axis == 1:
            caitsX = []
            for i in range(len(init_data[0].X)):
                if axis_names is None:
                    _axis_names = init_data[0].X[i].keys()
                    axis_1 = _axis_names["axis_1"]
                    axis_1 += sum([list(d.X[i].keys()["axis_1"]) for d in init_data[1:]], [])
                    _axis_names["axis_1"] = axis_1
                else:
                    _axis_names = axis_names

                values = np.concatenate([d.X[i].values for d in init_data], axis=1)

                caitsX.append(
                    CoreArray(
                        values=values,
                        axis_names=_axis_names
                    )
                )

            # TODO: implement for y
            caitsY = init_data[0].y
            caitsId = init_data[0]._id
            return DatasetList(X=caitsX, y=caitsY, id=caitsId)
        else:
            raise ValueError("Invalid axis argument.")

    def replace(self, other):
        new_data = deepcopy(self)

        if len(self.X) != len(other.X):
            raise ValueError("self.X and other.X must have same length.")
        if len(self.y) != len(other.y):
            raise ValueError("self.y and other.y must have same length.")
        if len(self._id) != len(other._id):
            raise ValueError("self.id and other.id must have same length.")
        if len(set(self.X[0].axis_names[f"axis_1"].keys()).intersection(other.X[0].axis_names[f"axis_1"].keys())) == 0:
            raise ValueError("self.X[0] and other.X[0] must have same axis_name.")

        idxs = [new_data.X[0].axis_names["axis_1"][o] for o in other.X[0].axis_names["axis_1"].keys()]

        for i in range(len(self.X)):
            new_data.X[i].values[:, idxs] = other.X[i].values

        return new_data

    def to_numpy(self, flatten=False):
        if flatten:
            return np.stack([x.values.flatten() for x in self.X]), self.y.values, np.array(self._id)
        else:
            return np.stack([x.values for x in self.X]), self.y.values, np.array(self._id)

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

    def get_axis_names_y(self):
        return deepcopy(self.y.axis_names)

    @staticmethod
    def from_numpy(
            X,
            y,
            id,
            axis_names_X: Optional[Dict[str, Dict[Union[str, int], int]]] = None,
            axis_names_y = None,
    ):
        _y = CoreArray(y, axis_names=axis_names_y)
        _X = [CoreArray(x, axis_names=axis_names_X) for x in X]

        return DatasetList(X=_X, y=_y, id=id)


    @staticmethod
    def features_dict_to_dataset(
            features,
            axis_names,
            axis,
            to_X = True,
            to_y = False
    ):

        ret_values = {}
        ret_axis_names = deepcopy(axis_names)

        # Processing X
        if to_X:
            features_tmp = {}
            for feat, values in features["X"].items():
                if values[0].ndim == 1:
                    features_tmp[feat] = values
                else:
                    for i in range(values[0].shape[0]):
                        features_tmp[f"{feat}_{i}"] = [values[j][i, ...] for j in range(len(values))]

            tmp = [
                np.stack(
                    [feat[i] for feat in features_tmp.values()],
                    axis=axis,
                ) for i in range(len(list(features_tmp.values())[0]))
            ]

            ret_axis_names["X"][f"axis_{axis}"] = list(features_tmp.keys())

            ret_values["X"] = [CoreArray(x, ret_axis_names["X"]) for x in tmp]
        else:
            ret_values["X"] = features["X"]

        if to_y:
            features_tmp = {}
            # Processing y
            for feat, values in features["y"].items():
                if values.ndim == 1:
                    features_tmp[feat] = values
                else:
                    for i in range(values.shape[0]):
                        features_tmp[f"{feat}_{i}"] = values[i, ...]

            tmp = np.stack([feat for feat in features_tmp.values()], axis=axis)

            ret_axis_names["y"][f"axis_{axis}"] = list(features_tmp.keys())

            ret_values["y"] = CoreArray(tmp, ret_axis_names["y"])
        else:
            ret_values["y"] = features["y"]

        ret_values["id"] = features["id"]

        return DatasetList(**ret_values)

    def dict_to_dataset(self, X):
        return DatasetList(**X)

    def train_test_split(self, random_state: Optional[int]=None, test_size: float=0.2, stratified: bool = False):

        if not stratified:
            all_idxs = [np.arange(len(self.X))]
        else:
            tmp = {}

            for i, y in enumerate(self.y.values[:, 0]):
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
        train_id = []
        test_X = []
        test_id = []

        for idx in train_idxs:
            train_X.append(self.X[idx])
            train_id.append(self._id[idx])

        train_y = self.y.iloc[train_idxs, ...]

        for idx in test_idxs:
            test_X.append(self.X[idx])
            test_id.append(self._id[idx])

        test_y = self.y.iloc[test_idxs, ...]

        return self.__class__(train_X, train_y, train_id), self.__class__(test_X, test_y, test_id)


    def apply(
        self,
        func,
        to_X = True,
        to_y = False,
        export_to=None,
        *args,
        **kwargs
    ):
        if to_X:
            X_tr = [func(df.values, *args, **kwargs) for df in self.X]
        else:
            X_tr = self.X.values

        if to_y:
            y_tr = func(self.y.values, *args, **kwargs),
        else:
            y_tr = self.y.values

        if export_to is None or export_to == "tuple":
            return X_tr, y_tr, self._id
        elif export_to == "dict":
            return {
                "X": X_tr,
                "y": y_tr,
                "id": self._id
            }
        else:
            raise Exception(f"export_to {export_to} is not supported.")

    def apply_windowing(self, func, *args, **kwargs):
        windowed_data = self.apply(func, to_y=False, *args, **kwargs)

        X = sum(windowed_data[0], [])
        y = CoreArray(
            values=np.array(
                [self.y.values[i, ...] for i, x in enumerate(windowed_data[0]) for _ in x]
            ),
            axis_names={"axis_1": self.y.keys()["axis_1"]}
        )
        id = [self._id[i] for i, x in enumerate(windowed_data[0]) for _ in x]

        axis_names_X = {"axis_1": self.get_axis_names_X()["axis_1"]}

        caitsX = [CoreArray(values=x, axis_names=axis_names_X) for x in X]

        return DatasetList(X=caitsX, y=y, id=id)


    # TODO: Adjust for y
    def flatten(
            self,
            to_X = True,
            to_y = False,
            axis_names_sep=","
    ):
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

        flattened_values = [x.values.flatten() for x in self.X]

        return DatasetList(
            X=[
                CoreArray(x, axis_names=axis_names)
                for x in flattened_values
            ],
            y=self.y,
            id=self._id
        )

        # flattened_values = np.stack([x.values.flatten() for x in self.X], axis=0)
        # return DatasetArray(
        #     X=CoreArray(
        #         flattened_values,
        #         axis_names=axis_names
        #     ),
        #     y=self.y,
        # )


    def reshape(
            self,
            shape_X,
            shape_y,
            axis_names_X,
            axis_names_y
    ):
        pass


    def shuffle(self, seed: int=42):
        idxs = np.arange(len(self.X))
        np.random.RandomState(seed).shuffle(idxs)
        return DatasetList(
            X=[self.X[i] for i in idxs],
            y=self.y.iloc[idxs, ...],
            id=[self._id[i] for i in idxs]
        )


    def rename(self, renamings):
        new_dataset_list = deepcopy(self)

        for part in renamings.keys():
            for axis, contents in renamings[part].items():
                for old_name, new_name in contents.items():
                    if part == "X":
                        data_part = new_dataset_list.X

                        if new_name in new_dataset_list.get_axis_names_X()[axis]:
                            raise ValueError(f"renamings[{part}][{axis}][{new_name}] already exists.")

                        for i, x in enumerate(self.X):
                            value = data_part[i].axis_names[axis].pop(old_name)
                            data_part[i].axis_names[axis][new_name] = value

                    else:
                        data_part = new_dataset_list.y

                        if new_name in new_dataset_list.get_axis_names_y()[axis]:
                            raise ValueError(f"renamings[{part}][{axis}][{new_name}] already exists.")

                        value = data_part.axis_names[axis].pop(old_name)
                        data_part.axis_names[axis][new_name] = value

        return new_dataset_list

