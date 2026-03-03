from ._dataset import DatasetArray, DatasetList
from ._core import CoreArray
import numpy as np
import pandas as pd
from typing import Literal
from copy import deepcopy


def from_numpy(
        X,
        y,
        _id=None,
        axis_names_X=None,
        axis_names_y=None,
        to: Literal["DatasetList", "DatasetArray"]="DatasetArray"
):
    _y = CoreArray(np.array(y), axis_names_y)

    if to == "DatasetList":
        _X = [CoreArray(x, axis_names_X) for x in X]
        return DatasetList(_X, _y, _id)
    elif to == "DatasetArray":
        _X = CoreArray(X, axis_names_X)
        return DatasetArray(_X, _y)
    else:
        raise ValueError(f"{to} is not a supported type")


def stack(data):
    _X = np.stack([arr.values for arr in data.X])
    axis_names = data.get_axis_names_X()
    axis_names = {
        f"axis_{int(axis.split('_')[-1]) + 1}": vals
        for axis, vals in axis_names.items()
    }




    return DatasetArray(
        X = CoreArray(_X, axis_names=axis_names),
        y = data.y
    )


def reshape(
        data,
        shape_X,
        shape_y,
        axis_names_X,
        axis_names_y,
        export_to
):
    reshaped_data = data.reshape(shape_X, shape_y)

    if data.__class__.__name__ == export_to:
        return reshaped_data
    elif data.__class__.__name__ == "DatasetList" and export_to == "DatasetArray":
        return stack(reshaped_data)
    elif data.__class__.__name__ == "DatasetArray" and export_to == "DatasetList":
        values = [reshaped_data.X.values[i, ...] for i in range(reshaped_data.X.values.shape[0])]
        return DatasetList(
            X = [CoreArray(v, axis_names=axis_names_X) for v in values],
            y = reshaped_data.y
        )
    else:
        raise ValueError(f"{export_to} is not a supported type")


def flatten(
        data,
        to_X=True,
        to_y=False,
        axis_names_sep=",",
        export_to="DatasetArray"
):
    flattened_data = data.flatten(to_X, to_y, axis_names_sep)

    if data.__class__.__name__ == export_to:
        return flattened_data
    elif data.__class__.__name__ == "DatasetList" and export_to == "DatasetArray":
        return stack(flattened_data)
    elif data.__class__.__name__ == "DatasetArray" and export_to == "DatasetList":
        _values = [flattened_data.X.values[i, ...] for i in range(flattened_data.X.values.shape[0])]
        _axis_names = flattened_data.X.keys()
        del _axis_names["axis_0"]

        return DatasetList(
            X = [CoreArray(v, axis_names=_axis_names) for v in _values],
            y = flattened_data.y
        )
    else:
        raise ValueError(f"{export_to} is not a supported type")



def from_dict():
    pass

def from_list():
    pass

def from_pandas():
    pass