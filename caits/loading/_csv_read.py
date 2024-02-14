import pandas as pd
from typing import Union


def csv_loader(
        dataset_path: str,
        header: Union[None, int, str] = "infer",
        channels: Union[list, None] = None,
        export: str = "dict"
) -> Union[pd.DataFrame, dict]:
    """Loads CSV files from a directory into a DataFrame.

    Args:
        dataset_path: Path to the directory containing subdirectories with the
            CSV files.
        header:
        channels: List of column names to be used in the DataFrame. Default is
            None, which will use the column names from the CSV file.
        export: A boolean that defines the export format. Default is "dict".
            It can be "dict" or "df".

    Returns:

    """
    import os

    all_features = []
    all_y = []
    all_id = []

    for subdir in os.listdir(dataset_path):
        if subdir != ".DS_Store":
            print("loading files in dir:", subdir)
            label_path = os.path.join(dataset_path, subdir)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    file_path = os.path.join(label_path, file)
                    try:
                        df = pd.read_csv(file_path, header=header,
                                         names=channels)
                        all_features.append(df)
                        all_y.append(subdir)
                        all_id.append(file)
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")

    if export == "df":
        return pd.DataFrame({
            "X": all_features,
            "y": all_y,
            "id": all_id
        })
    elif export == "dict":
        return {
            "X": all_features,
            "y": all_y,
            "id": all_id
        }
