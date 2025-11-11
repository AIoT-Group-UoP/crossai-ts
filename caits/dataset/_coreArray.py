import numpy as np
from typing import Optional, Dict, Iterable

DISPLAY_NUM_ROWS = 5
DISPLAY_NUM_COLS = 6
DISPLAY_VECTOR_NUM_ROWS = 60

class CoreArray:
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

                    return CoreArray(vals, axis_names=axis_names)


    class _LocIndexer:
        def __init__(self, parent, indexer):
            self.parent = parent

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

                    return CoreArray(self.parent.values[*idxs], axis_names=axis_names)


    def __init__(
            self,
            values: np.ndarray,
            axis_names: Optional[Dict[str, Iterable]]=None
    ):
        self.values = values
        self.axis_names = {f"axis_{i}": {} for i in range(values.ndim)}
        self.shape = values.shape
        self.ndim = values.ndim

        if axis_names is None:
            axis_names = {}

        if len(axis_names) > values.ndim:
            raise ValueError(
                f"Number of axis names ({len(axis_names)},) must not exceed number of dimensions ({values.ndim},)."
            )
        for i, axis in enumerate([f"axis_{j}" for j in range(values.ndim)]):
            if axis in axis_names.keys():
                if len(axis_names[axis]) != values.shape[i]:
                    raise ValueError(
                        f"Number of axis names ({len(axis_names[axis])},) "
                        f"and number of values ({list(values[axis])},) in axis={axis} don't match.")
                else:
                    self.axis_names[axis] = {name: j for j, name in enumerate(axis_names[axis])}
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

        res += f"CoreArray with shape {self.shape}\n"
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


    def keys(self):
        return {
            k: list(v.keys())
            for k, v in self.axis_names.items()
        }