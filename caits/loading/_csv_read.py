import glob
import os
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


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


def csv_loader_regression(
        dataset_path: str,
        X_cols: Union[List[str], None] = None,
        y_cols: Union[List[str], None] = None,
        header: Union[None, int, str] = "infer",
        export: Literal["df", "dict"] = "dict",
        **kwargs
) -> Union[pd.DataFrame, Dict[str, List]]:
    """Loads CSV files from a directory into a DataFrame or dictionary.

    Args:
        dataset_path: Path to the dataset directory containing CSV files.
        X_cols: List of column names to use as input. If None, all columns
            except those defined by y_cols are used.
        y_cols: List of column names to use as output. If None, the last
            column is used.
        header: Specifies the row(s) to use as the column names.
                Defaults to "infer".
        export: Format to export the loaded data, "dict" or "df" for
            DataFrame. Defaults to "dict".

    Returns:
        pd.DataFrame or dict: Loaded CSV data.
    """
    X = pd.DataFrame()
    y = pd.DataFrame()

    try:
        if X_cols is not None:
            kwargs["usecols"] = X_cols
            if y_cols is not None:
                kwargs["usecols"].extend(y_cols)

        X = pd.read_csv(dataset_path, header=header, **kwargs)

        if y_cols is None:
            y_cols = [X.columns[-1]]

        y = X[y_cols]
        X = X.drop(columns=y_cols)

        X = [X[[col]] for col in X.columns]
        y = [y[[col]] for col in y.columns]

    except Exception as e:
        print(f"Error loading file {dataset_path}: {e}")

    if export == "df":
        return pd.DataFrame({"X": X, "y": y})
    elif export == "dict":
        return {"X": X, "y": y}
    else:
        raise ValueError(f"Provide a valid method for exporting the data. "
                         f"Valid methods: 'df', 'dict'")


def csv_loader_single_file(
        file_path: str,
        channels: Optional[List[str]] = None
) -> pd.DataFrame:
    """Reads a CSV file into a pandas DataFrame with custom header logic.

    This function checks if the CSV has a header. If not, it assigns default
    names 'ch-1', 'ch-2', etc. It also allows the user to override any
    existing or default header with a custom list of column names.

    Args:
        file_path (str): The full path to the CSV file.
        channels (list of str, optional): A list of strings to be used as
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

    if channels:
        df = pd.read_csv(file_path, header=None)
        if len(channels) != len(df.columns):
            raise ValueError(
                f"Mismatch: The provided list 'channels' has "
                f"{len(channels)} names, "
                f"but the CSV file has {len(df.columns)} columns."
            )
        df.columns = channels

        return df
    else:
        df = pd.read_csv(file_path)

        is_default_header = all(isinstance(col, int) for col in df.columns)

        if is_default_header:
            num_columns = len(df.columns)
            new_column_names = [f'ch_{i + 1}' for i in range(num_columns)]
            df.columns = new_column_names

        return df