from typing import List, Optional, Union, Literal, Dict

import boto3
import pandas as pd
from tqdm import tqdm


def s3_csv_loader(
    bucket: str,
    prefix: str,
    endpoint_url: str,
    header: Union[None, int, str] = "infer",
    channels: Union[List[str], None] = None,
    export: Literal["df", "dict"] = "dict",
    classes: Optional[List[str]] = None,
) -> Union[pd.DataFrame, Dict[str, List]]:
    """Loads CSV files from an S3 bucket into a DataFrame or dictionary.

    Args:
        bucket: Name of the S3 bucket.
        prefix: Prefix of the S3 bucket.
        endpoint_url: URL of the S3 bucket.
        header: Specifies the row(s) to use as the column names.
                Defaults to "infer".
        channels: List of column names to use. If None, all columns are used.
        export: Format to export the loaded data, "dict" or "df" for DataFrame.
        classes: Optional list of directory names to include;
                 if None, all directories are included.

    Returns:
        pd.DataFrame or dict: Loaded CSV data.
    """

    s3 = boto3.client("s3", endpoint_url=endpoint_url)

    all_features = []
    all_y = []
    all_id = []

    # Collect all file paths ending with .csv
    paginator = s3.get_paginator("list_objects_v2")
    file_paths = []
    for result in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in result:
            for key in result["Contents"]:
                file_path = key["Key"]
                if file_path.endswith(".csv"):
                    file_paths.append(file_path)

    for file_path in tqdm(file_paths, desc="Loading CSV files"):
        parts = file_path.strip('/').split('/')
        if len(parts) < 2:
            continue  # Skip files not in the expected format
        subdir = parts[-2]

        # Check if desired
        if classes is None or subdir in classes:
            file = parts[-1]
            try:
                obj = s3.get_object(Bucket=bucket, Key=file_path)
                read_csv_kwargs = {"header": header}
                if channels is not None:
                    read_csv_kwargs["usecols"] = channels

                df = pd.read_csv(obj["Body"], **read_csv_kwargs)

                all_features.append(df)
                all_y.append(subdir)
                all_id.append(file)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    if export == "df":
        return pd.DataFrame({"X": all_features, "y": all_y, "id": all_id})
    elif export == "dict":
        return {"X": all_features, "y": all_y, "id": all_id}
