import pandas as pd
from typing import Union
import boto3


def s3_csv_loader(
        bucket: str,
        prefix: str,
        endpoint_url: str,
        header: Union[None, int, str] = "infer",
        channels: Union[list, None] = None,
        export: str = "dict"
) -> Union[pd.DataFrame, dict]:
    """Loads CSV files from a directory into a DataFrame.

    Args:
        bucket: Name of the S3 bucket.
        prefix: Prefix of the S3 bucket.
        endpoint_url: URL of the S3 bucket.
        header:
        channels: List of column names to be used in the DataFrame. Default is
            None, which will use the column names from the CSV file.
        export: A boolean that defines the export format. Default is "dict".
            It can be "dict" or "df".

    Returns:
        A DataFrame containing the CSV data.
    """

    s3 = boto3.client("s3", endpoint_url=endpoint_url)

    all_features = []
    all_y = []
    all_id = []

    paginator = s3.get_paginator("list_objects_v2")
    for result in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in result:
            for key in result["Contents"]:
                file_path = key["Key"]
                if file_path.endswith(".csv"):
                    try:
                        obj = s3.get_object(Bucket=bucket, Key=file_path)
                        df = pd.read_csv(obj["Body"], header=header, names=channels)
                        all_features.append(df)
                        all_y.append(file_path.split("/")[-2])
                        all_id.append(file_path.split("/")[-1])
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
