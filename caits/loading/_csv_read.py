import glob
import os
from typing import Dict, List, Literal, Optional, Union

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


def csv_loader2(
        dataset_path: str,
        header: Union[None, int, str] = "infer",
        input_channels: Union[List[str], None] = None,
        output_channels: Union[List[str], None] = None,
        export: Literal["df", "dict"] = "dict",
) -> Union[pd.DataFrame, Dict[str, List]]:
    """Loads CSV files from a directory into a DataFrame or dictionary.

    Args:
        dataset_path: Path to the dataset directory containing CSV files.
        channels: List of column names to use. If None, all columns are used.
        export: Format to export the loaded data, "dict" or "df" for DataFrame.
        classes: Optional list of directory names to include;
                 if None, all directories are included.

    Returns:
        pd.DataFrame or dict: Loaded CSV data.
    """
    all_features = []
    all_y = []

    try:
        read_csv_kwargs = {"header": header}
        if input_channels is not None:
            read_csv_kwargs["usecols"] = input_channels + output_channels

        df = pd.read_csv(dataset_path, **read_csv_kwargs)

        all_features.append(df[input_channels])
        all_y.append(df[output_channels])

    except Exception as e:
        print(f"Error loading file {dataset_path}: {e}")

    # Export the loaded data as a DataFrame or
    # dictionary based on the 'export' argument
    if export == "df":
        return pd.DataFrame({"X": all_features, "y": all_y})
    elif export == "dict":
        return {"X": all_features, "y": all_y}
