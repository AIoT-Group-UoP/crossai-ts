from ._dataset import DatasetArray, DatasetList
from ._core import CoreArray
import numpy as np
import pandas as pd
from typing import Literal
from copy import deepcopy


def from_numpy(to: Literal["DatasetList", "DatasetArray"], *args, **kwargs):
    if to == "DatasetList":
        return DatasetList(*args, **kwargs)
    elif to == "DatasetArray":
        return DatasetArray(*args, **kwargs)
    else:
        raise ValueError(f"{to} is not a supported type")


def stack(data):
    _X = np.stack([arr.values for arr in data])
    axis_names = deepcopy(data.keys())
    del axis_names["axis_0"]

    return DatasetArray(
        X = CoreArray(_X, axis_names=axis_names),
        y = data.y
    )

def from_dict():
    pass

def from_list():
    pass

def from_pandas():
    pass