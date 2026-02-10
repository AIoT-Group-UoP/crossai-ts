import numpy as np
from typing import Tuple, Optional, List, Literal, Union
from ..dataset import CoreArray, DatasetArray, DatasetList
from ._audio import wav_loader
from ._csv_read import csv_loader_single_file
import os
import glob
from tqdm import tqdm

def load_file(
        file_path: str,
        type: Literal["csv", "wav"],
        X_cols: Optional[list[str]] = None,
        y_cols: Optional[list[str]] = None,
        export_to: Literal["datasetArray", "dict", "tuple"] = "dict",
        *args,
        **kwargs
) -> Union[dict|DatasetArray|tuple]:
    if type == "csv":
        single_file_loader_fun = csv_loader_single_file
    elif type == "wav":
        single_file_loader_fun = wav_loader
    else:
        raise ValueError(f"Type {type} is not supported.")

    data, sr = single_file_loader_fun(file_path, *args, **kwargs)

    ret = {
        "data": {},
        "sr": sr
    }

    if y_cols is not None:
        ret["data"]["y"] = data.loc[:, y_cols]
        data = data.drop({"axis_1":y_cols})

    ret["data"]["X"] = data.loc[:, X_cols] if X_cols is not None else data
    ret["sr"] = sr

    if export_to == "dict":
        return ret
    elif export_to == "datasetArray":
        return DatasetArray(**ret["data"]), ret["sr"]
    elif export_to == "tuple":
        return (
            (ret["data"]["X"],) if "y" not in ret.keys() else (ret["data"]["X"], ret["data"]["y"]),
            ret["sr"]
        )
    else:
        raise ValueError("export_to must be either 'dict' or 'datasetArray'")


def load_dir(
        dir_path: str,
        type: Literal["csv", "wav"],
        classes: Optional[list[str]] = None,
        X_cols: Optional[list[str]] = None,
        *args,
        **kwargs
):
    search_pattern = os.path.join(dir_path, "**", f"*.{type}")
    file_paths = glob.glob(search_pattern, recursive=True)

    X = []
    y = []
    all_id = []

    for file_path in tqdm(file_paths, desc="Loading data files"):
        subdir = os.path.basename(os.path.dirname(file_path))

        # check if desired
        if classes is None or subdir in classes:
            file = os.path.basename(file_path)
            try:
                single_file_data, sr = load_file(
                    file_path=file_path,
                    type=type,
                    X_cols=X_cols,
                    y_cols=None,
                    export_to="tuple",
                    *args,
                    **kwargs
                )

                X.append(single_file_data)
                # todo: add sample rate, sample width to the dictionary?
                y.append(subdir)
                all_id.append(file)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    return DatasetList(
        X=X,
        y=CoreArray(values=np.array(y)[:, np.newaxis], axis_names={"axis_1": ["Label"]}),
        id=all_id
    )