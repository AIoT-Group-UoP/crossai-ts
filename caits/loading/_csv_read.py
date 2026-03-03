import glob
import os
from typing import Dict, List, Literal, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from ..dataset import CoreArray, DatasetArray


def csv_loader(
    dataset_path: str,
    header: Union[None, int, str] = "infer",
    channels: Union[List[str], None] = None,
    export: Literal["df", "dict"] = "dict",
    classes: Optional[List[str]] = None,
) -> Union[pd.DataFrame, Dict[str, List]]:
    """Loads CSV files from a directory into a DataFrame or dictionary.

    Args:
        dataset_path: Path to the dataset directory containing CSV files.
        header: Specifies the row(s) to use as the column names.
                Defaults to "infer".
        channels: List of column names to use. If None, all columns are used.
        export: Format to export the loaded data, "dict" or "df" for DataFrame.
        classes: Optional list of directory names to include;
                 if None, all directories are included.

    Returns:
        pd.DataFrame or dict: Loaded CSV data.
    """
    all_features = []
    all_y = []
    all_id = []

    # Generate a search pattern to find CSV files in the dataset directory
    search_pattern = os.path.join(dataset_path, "**", "*.csv")
    file_paths = glob.glob(search_pattern, recursive=True)

    for file_path in tqdm(file_paths, desc="Loading CSV files"):
        subdir = os.path.basename(os.path.dirname(file_path))

        # check if desired
        if classes is None or subdir in classes:
            file = os.path.basename(file_path)
            try:
                # Load the CSV file, specifying header
                # and column names if provided
                read_csv_kwargs = {"header": header}
                if channels is not None:
                    read_csv_kwargs["usecols"] = channels

                df = pd.read_csv(file_path, **read_csv_kwargs)

                all_features.append(df)
                all_y.append(subdir)
                all_id.append(file)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    # Export the loaded data as a DataFrame or
    # dictionary based on the 'export' argument
    if export == "df":
        return pd.DataFrame({"X": all_features, "y": all_y, "id": all_id})
    elif export == "dict":
        return {"X": all_features, "y": all_y, "id": all_id}


def csv_loader_single_file(
        file_path: str,
        axis_names: Optional[Dict] = None,
        **kwargs
) -> Tuple:
    """Reads a CSV file into a pandas DataFrame with custom header logic.

    This function checks if the CSV has a header. If not, it assigns default
    names 'ch-1', 'ch-2', etc. It also allows the user to override any
    existing or default header with a custom list of column names.

    Args:
        file_path (str): The full path to the CSV file.
        delimiter (str, optional): The CSV delimiter. Defaults to ",".
        axis_names (list of str, optional): A list of strings to be used as
            column names. If provided, this will override any existing header
            or default naming. Defaults to None.

    Returns:
        pandas.DataFrame: The CSV data as a DataFrame with the correct headers.

    Raises:
        FileNotFoundError: If the file_path does not point to an existing file.
        ValueError: If the number of columns in the custom 'channels' list
                    does not match the number of columns in the CSV file.

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Error: The file '{file_path}' was not found.")

    axis_0_flag = axis_names is not None and "axis_0" in axis_names.keys()
    axis_1_flag = axis_names is not None and "axis_1" in axis_names.keys()

    if axis_0_flag:
        kwargs["index_col"] = None
    if axis_1_flag:
        kwargs["header"] = None
    else:
        kwargs["header"] = "infer"

    df = pd.read_csv(file_path, **kwargs)

    if kwargs["header"] == "infer":
        if axis_names is None:
            axis_names = {}
        axis_names["axis_1"] = df.columns

    if (
            (axis_0_flag and len(axis_names["axis_0"]) != df.nrows)
            or
            (axis_1_flag and len(axis_names["axis_1"]) != len(df.columns))
    ):
        axis_0 = len(axis_names["axis_0"]) if axis_0_flag else None
        axis_1 = len(axis_names["axis_1"]) if axis_1_flag else None
        raise ValueError(
            f"Mismatch: The provided axis_names dict has "
            f"({axis_0}, {axis_1}) names, "
            f"but the CSV file has shape ({df.nrows}, {len(df.columns)})."
        )

    return (
        CoreArray(
            values=df.values,
            axis_names=axis_names
        ), None
    )
