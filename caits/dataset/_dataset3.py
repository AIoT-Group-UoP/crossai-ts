from abc import ABC, abstractmethod
from typing import List, Tuple
from copy import deepcopy

import numpy as np
from typing import Optional, Union, List, Dict
import copy

import pandas as pd

DISPLAY_NUM_ROWS = 5
DISPLAY_NUM_COLS = 6
DISPLAY_VECTOR_NUM_ROWS = 60

class CaitsArray:
    class _iLocIndexer:
        def __init__(self, parent) -> None:
            self.parent = parent

        def __getitem__(self, index):
            if len(index) != self.parent.ndim:
                raise ValueError(f'Index must be {self.parent.ndim} dimensional')
            else:
                vals = self.parent.values[index]
                if not isinstance(vals, np.ndarray):
                    return vals
                else:
                    axis_names_arr = {axis: np.array(list(v.keys())) for axis, v in self.parent.axis_names.items()}
                    axis_names_arr_indexed = {axis: v[index[i]] for i, (axis, v) in enumerate(axis_names_arr.items())}

                    if all([isinstance(i, list) for i in index]):
                        combs = axis_names_arr_indexed.values()
                        axis_names = {
                            "axis_0": {
                                t: i for i, t in enumerate(zip(*combs))
                            }
                        }
                    else:
                        axis_names = {
                            axis: {v: i for i, v in enumerate(names)}
                            for j, (axis, names) in enumerate(axis_names_arr_indexed.items())
                            if not isinstance(index[j], int)
                        }
                        axis_names = {f"axis_{i}": names for i, names in enumerate(axis_names.values())}

                    return CaitsArray(vals, axis_names=axis_names)


    class _LocIndexer:
        def __init__(self, parent, indexer):
            self.parent = parent
            self.indexer = indexer

        def __getitem__(self, index):
            if len(index) != self.parent.ndim:
                raise ValueError(f'Index must be {self.parent.parent.values.ndim} dimensional')
            else:
                idxs = []
                axis_names = {axis: np.array(list(names.keys())) for axis, names in self.parent.axis_names.items()}

                for i, t in enumerate(index):
                    if isinstance(t, str) or isinstance(t, int):
                        idxs.append(self.parent.axis_names[f"axis_{i}"][t])
                    elif isinstance(t, list):
                        idxs.append([self.parent.axis_names[f"axis_{i}"][j] for j in t])
                    elif isinstance(t, slice):
                        idxs.append(
                            slice(
                                self.parent.axis_names[f"axis_{i}"][t.start] if t.start is not None else None,
                                (self.parent.axis_names[f"axis_{i}"][t.stop] + 1) if t.stop is not None else None,
                                t.step
                            )
                        )
                    else:
                        raise IndexError(f"Unsupported index type {index}")

                vals = self.parent.values[*idxs]
                if not isinstance(vals, np.ndarray):
                    return vals
                else:
                    if all([isinstance(i, list) for i in idxs]):
                        axis_names = {
                            "axis_0": {
                                t: i for i, t in enumerate(zip(*index))
                            }
                        }
                    else:
                        axis_names = {
                            axis: {n: i for i, n in enumerate(names[idxs[i]])}
                            for i, (axis, names) in enumerate(axis_names.items())
                            if not isinstance(idxs[i], int)
                        }
                        axis_names = {f"axis_{i}": names for i, names in enumerate(axis_names.values())}

                    return CaitsArray(self.parent.values[*idxs], axis_names=axis_names)


    def __init__(self, values: np.ndarray, axis_names: Optional[Dict]=None):
        self.values = values
        self.axis_names = {f"axis_{i}": {} for i in range(values.ndim)}
        self.shape = values.shape
        self.ndim = values.ndim

        if axis_names is None:
            axis_names = {}

        if len(axis_names) > values.ndim:
            print(axis_names)
            print(values)
            raise ValueError("Axis names must not exceed number of dimensions")
        for i, axis in enumerate([f"axis_{j}" for j in range(values.ndim)]):
            if axis in axis_names.keys():
                if len(axis_names[axis]) != values.shape[i]:
                    print(values)
                    print(axis_names[axis])
                    raise ValueError(
                        f"Shapes {[len(axis) for axis in axis_names.values()]} "
                        f"and {list(values.shape)} don't match.")
                else:
                    self.axis_names[axis] = copy.deepcopy(axis_names[axis])
            else:
                self.axis_names[axis] = {j: j for j in range(values.shape[i])}

        self.loc = self._LocIndexer(self, self.axis_names)
        self.iloc = self._iLocIndexer(self)
        self.dtypes = self.values.dtype

    def __len__(self):
        return self.values.shape[0]

    def __repr__(self):
        res = ""
        if self.ndim > 2:
            for idx in np.ndindex(*self.shape[:-2]):
                res += f"----------------------------- FOLD {idx} -----------------------------\n"
                res += self.__repr_single_frame(idx)
        elif self.ndim == 2:
            res += self.__repr_single_frame()
        else:
            if len(self.values) <= DISPLAY_VECTOR_NUM_ROWS:
                column_names = [str(s) for s in list(self.axis_names["axis_0"].keys())]
                value_strs = [str(x) for x in self.values]
            else:
                column_names = [str(s) for s in list(self.axis_names["axis_0"].keys())[:DISPLAY_NUM_ROWS]]
                column_names.append("...")
                column_names.extend([str(s) for s in list(self.axis_names["axis_0"].keys())[(-DISPLAY_NUM_ROWS):]])
                value_strs = [str(s) for s in list(self.values)[:DISPLAY_NUM_ROWS]]
                value_strs.append("...")
                value_strs.extend([str(s) for s in list(self.values)[-DISPLAY_NUM_ROWS:]])

            widths = [max([len(s) for s in column_names]), max([len(s) for s in value_strs])]
            names = [f"{col:>{widths[0]}}  " for col in column_names]
            values = [f"{val:>{widths[1]}}" for val in value_strs]
            res += "\n".join([n+v for n, v in zip(names, values)]) + "\n\n"

        res += f"CaitsArray with shape {self.shape}\n"
        return res

    def __repr_single_frame(self, comb=None):
        column_names = self.axis_names[list(self.axis_names.keys())[-1]]
        row_names = self.axis_names[list(self.axis_names.keys())[-2]]
        col_widths = [0 for _ in range(self.shape[-1])]
        num_rows = min(11, self.shape[-2])
        if num_rows > 2 * DISPLAY_NUM_ROWS:
            tmp = [i for i in range(DISPLAY_NUM_ROWS)]
            row_idxs = tmp + [-(i+1) for i in tmp[::-1]]
        else:
            row_idxs = [i for i in range(num_rows)]

        for i, col in enumerate(column_names):
            if comb is not None:
                all_col_strs = [str(col)] + [str(x) for x in self.values[*comb, row_idxs, i]]
            else:
                all_col_strs = [str(col)] + [str(x) for x in self.values[row_idxs, i]]
            width = max([len(s) for s in all_col_strs])
            col_widths[i] = width

        header = [f"{col:>{col_widths[i]}}  " for i, col in enumerate(column_names)]

        ret = []
        for row in row_idxs:
            row_idxs_dict = row_names
            row_idx = row_idxs_dict[list(row_idxs_dict.keys())[row]]

            if comb is not None:
                row_str = [f"{self.values[*comb, row_idx, i]:>{col_widths[i]}}  " for i in range(self.shape[-1])]
            else:
                row_str = [f"{self.values[row_idx, i]:>{col_widths[i]}}  " for i in range(self.shape[-1])]
            ret.append(row_str)

        final_ret = []

        if num_rows > 2 * DISPLAY_NUM_ROWS:
            sep = [f"{'...':>{width}}  " for width in col_widths]

        index_width = max([len(str(i)) for i in row_names])
        index = [str(i) for i in row_names]
        if num_rows > 2 * DISPLAY_NUM_ROWS:
            index = ([" " * (index_width + 2)] +
                     [f"{i:>{index_width}}  " for i in index[:DISPLAY_NUM_ROWS]] +
                     [f"{'...':>{index_width}}  "] +
                     [f"{i:>{index_width}}  " for i in index[-DISPLAY_NUM_ROWS:]])
        else:
            index = [" " * (index_width + 2)] + [f"{i:>{index_width}}  " for i in index]

        if len(col_widths) > DISPLAY_NUM_COLS:
            for i in range(0, len(col_widths), DISPLAY_NUM_COLS):
                tmp_header = header[i:i+DISPLAY_NUM_COLS]
                tmp = [tmp_header] + [ret[j][i:i+DISPLAY_NUM_COLS] for j in range(len(ret))]
                if num_rows > 2 * DISPLAY_NUM_ROWS:
                    tmp = tmp[:(DISPLAY_NUM_ROWS+1)] + [sep[i:i+DISPLAY_NUM_COLS]] + tmp[(DISPLAY_NUM_ROWS+1):]

                final_ret.append(tmp)

            for i in range(len(final_ret) - 1):
                final_ret[i][0].append("\\")

        else:
            tmp = [header] + ret
            if num_rows > 2 * DISPLAY_NUM_ROWS:
                tmp = tmp[:(DISPLAY_NUM_ROWS + 1)] + [sep[:len(tmp[0])]] + tmp[(DISPLAY_NUM_ROWS + 1):]
            final_ret = [tmp]

        result = ""
        for part_idx in range(len(final_ret)):
            tmp = []
            for row_idx in range(len(final_ret[part_idx])):
                tmp.append(index[row_idx] + "".join(final_ret[part_idx][row_idx]) + "\n")

            result += "".join(tmp) + "\n"

        return result


class Dataset3(ABC):
    def __init__(
            self,
            X: Union[CaitsArray, List[CaitsArray]],
            y: Union[CaitsArray, List]
    ):
        self.X = X
        self.y = y

    @abstractmethod
    def __len__(self):
        return len(self.X)

    @abstractmethod
    def __getitem__(self, idx: Union[int, slice, tuple, List]):
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
    def unify(self, others, axis_names: Optional = None, axis: int=0):
        pass

    @abstractmethod
    def replace(self, other):
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
    def get_axis_names_X(self):
        pass

    @abstractmethod
    def numpy_to_dataset(
            self,
            X,
            axis_names: Optional[Dict[str, Dict[Union[str, int], int]]] = None,
            split: bool=True
    ):
        pass

    @abstractmethod
    def features_dict_to_dataset(self, features, axis_names, axis):
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

    @abstractmethod
    def stack(self, data: List):
        pass

    @abstractmethod
    def flatten(self):
        pass

    @abstractmethod
    def shuffle(self, seed: int=42):
        pass


class DatasetArray(Dataset3):
    def __init__(self, X: CaitsArray, y: Optional[CaitsArray] = None):
        if y is None:
            _y = CaitsArray(np.array([[None] for _ in range(len(X))]))
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

    # TODO: y
    def unify(self, others, axis_names: Optional = None, axis: int=0):
        _axis = 1 if axis == 0 else 0
        if all(
                [
                    self.X.shape[_axis] == o.X.shape[_axis] and
                    self.X.axis_names[f"axis_{axis}"] != o.X.axis_names[f"axis_{axis}"]
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

            X = CaitsArray(np.concatenate([self.X.values] + [o.X.values for o in others], axis=axis), axis_names=axis_names_X)
            return self.__class__(X=X, y=self.y)

        elif all([self.X.shape[_axis] != o.X.shape[_axis] for o in others]):
            raise ValueError("self.X and other.X must have the same number of columns.")
        else:
            raise ValueError("self.X and other.y must have the same number of columns.")

    def replace(self, other):
        if len(set(self.X.axis_names["axis_1"].keys()).intersection(other.X.axis_names["axis_1"].keys())) == 0:
            raise IndexError(f"Column names {other.X.axis_names['axis_names'].keys()} not found in X or y.")

        column_names = list(other.X.axis_names["axis_1"].keys())
        idxs = [self.X.axis_names["axis_1"][col] for col in column_names]

        self.X.values[:, idxs] = other.X.values


    def to_numpy(self):
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
        return self.X.axis_names

    # TODO: y
    def numpy_to_dataset(
            self,
            X,
            axis_names: Optional[Dict[str, Dict[Union[str, int], int]]] = None,
            split: bool = True
    ):
        if axis_names is None:
            _axis_names_X = self.X.axis_names
            _axis_names_y = self.y.axis_names
        elif all([isinstance(v, str) for v in axis_names.values()]):
            _axis_names_X = {
                to_axis: self.X.axis_names[from_axis] for to_axis, from_axis in axis_names.items()
            }
            _axis_names_y = {
                to_axis: self.y.axis_names[from_axis] for to_axis, from_axis in axis_names.items()
            }
        elif all([isinstance(v, dict) for v in axis_names.values()]):
            _axis_names_X = axis_names["X"]
            _axis_names_y = axis_names["y"]

        if split:
            dfX = CaitsArray(X[0], axis_names=(_axis_names_X))
            dfY = CaitsArray(X[1], axis_names=(_axis_names_y))
        else:
            dfX = CaitsArray(X, axis_names=(_axis_names_X))
            dfY = self.y
        return DatasetArray(X=dfX, y=dfY)

    # TODO: y
    def features_dict_to_dataset(self, features, axis_names, axis):
        features_X = np.stack([feat[0] for feat in features.values()], axis=axis)
        features_y = np.stack([feat[1] for feat in features.values()], axis=axis)

        _axis = 1 if axis == 0 else 0

        axis_names_X = axis_names | {f"axis_{_axis}": self.X.axis_names[f"axis_{_axis}"]}
        axis_names_y = axis_names | {f"axis_{_axis}": self.y.axis_names[f"axis_{_axis}"]}

        return DatasetArray(
            X=CaitsArray(features_X, axis_names=axis_names_X),
            y=CaitsArray(features_y, axis_names=axis_names_y)
        )

    # TODO: something is wrong
    def dict_to_dataset(self, X):
        vals = np.stack([row for row in X.values()])
        dfX = CaitsArray(
            vals,
            axis_names={
                axis: names for axis, names in X.axis_names.items() if axis != "axis_0"
            }
        )
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

        train_X = self.X.iloc[train_idxs, ...]
        test_X = self.X.iloc[test_idxs, ...]
        train_y = self.y.iloc[train_idxs, ...]
        test_y = self.y.iloc[test_idxs, ...]

        return self.__class__(train_X, train_y), self.__class__(test_X, test_y)

    # TODO: y
    def apply(self, func, *args, **kwargs):
        X = func(self.X.values, *args, **kwargs)
        y = func(self.y.values, *args, **kwargs)
        return X, y

    # TODO: y
    def stack(self, X: List[np.ndarray]):
        return DatasetList(
            X=[CaitsArray(values=x, axis_names={"axis_1": self.X.axis_names["axis_1"]}) for x in X[0]],
            y=[CaitsArray(values=y, axis_names={"axis_1": self.y.axis_names["axis_1"]}) for y in X[1]]
        )

    # TODO: y
    def flatten(self):
        return DatasetArray(CaitsArray(self.X.values.flatten()), self.y)

    def shuffle(self, seed: int=42):
        idxs = np.arange(len(self.X))
        np.random.shuffle(idxs)
        return DatasetArray(
            self.X.iloc[idxs, ...],
            self.y.iloc[idxs, ...],
        )



class DatasetList(Dataset3):
    def __init__(
            self,
            X: List[CaitsArray],
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
                    CaitsArray(
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


    def to_numpy(self):
        return [np.array(x.values) for x in self.X], np.array(self.y), np.array(self._id)

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
        return self.X[0].axis_names

    def get_axis_names_y(self):
        return {}

    def numpy_to_dataset(
            self,
            X,
            axis_names: Optional[Dict[str, Dict[Union[str, int], int]]] = None,
            split: bool = True
    ):
        if split:
            if axis_names is not None:
                _axis_names = {}
                for k, v in axis_names.items():
                    if isinstance(v, str):
                        _axis_names[k] = self.X[0].axis_names[v]
                    elif isinstance(v, dict):
                        _axis_names[k] = v
                    else:
                        raise ValueError
            else:
                _axis_names = None

            _X = [CaitsArray(x, axis_names=_axis_names,) for x in X]
        else:
            _X = CaitsArray(X)

        return DatasetList(X=_X, y=self.y, id=self._id)

    def features_dict_to_dataset(self, features, axis_names, axis):
        values = [
            np.stack(
                [feat[i] for feat in features.values()],
                axis=axis,
            ) for i in range(len(list(features.values())[0]))
        ]

        print(values[0].shape)

        return self.numpy_to_dataset(values, axis_names)

    def dict_to_dataset(self, X):
        return DatasetList(**X)

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

    def stack(self, data):
        X = sum(data, [])
        y = [self.y[i] for i, x in enumerate(data) for _ in x]
        id = [self._id[i] for i, x in enumerate(data) for _ in x]
        caitsX = [CaitsArray(values=x, axis_names={"axis_1": self.X[0].axis_names["axis_1"]}) for x in X]
        return DatasetList(X=caitsX, y=y, id=id)

    def flatten(self):
        return DatasetList(
            X=CaitsArray(np.stack([x.values.flatten() for x in self.X], axis=0)),
            y=self.y,
            id=self._id
        )

    def shuffle(self, seed: int=42):
        idxs = np.arange(len(self.X))
        np.random.RandomState(seed).shuffle(idxs)
        return DatasetList(
            X=[self.X[i] for i in idxs],
            y=[self.y[i] for i in idxs],
            id=[self._id[i] for i in idxs]
        )

