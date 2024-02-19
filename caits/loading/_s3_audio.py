import pandas as pd
import numpy as np
import soundfile as sf
from typing import Union, List
import boto3
import io


def _s3_wav_loader(
        file_content: bytes,
        channels: List[str] = ["channel_1"]
) -> pd.DataFrame:
    """Loads an audio file into a DataFrame.

    Args:
        mode: "scipy" | "pydub" | "soundfile"
        file_path: Path to the audio file.
        channels: List of channel names.

    Returns:
        A DataFrame containing the audio data.
    """
    
    # Create a file object from bytes
    file_obj = io.BytesIO(file_content)
    
    audio_data, sample_rate = sf.read(file_obj)

    df = pd.DataFrame(audio_data, columns=channels)

    return df


def s3_audio_loader(
        bucket: str,
        prefix: str,
        endpoint_url: str,
        mode: str = "soundfile",
        format: str = "wav",
        channels: List[str] = ["channel_1"],
        export: str = "dict"
) -> Union[pd.DataFrame, dict]:
    """Loads audio files from a directory into a DataFrame.

    Args:
        bucket: Name of the S3 bucket.
        prefix: Prefix of the S3 bucket.
        endpoint_url: URL of the S3 bucket.
        mode:
        format:
        channels:
        export: "dict" | "df"

    Returns:

    """
    s3 = boto3.client('s3', endpoint_url=endpoint_url)

    all_features = []
    all_y = []
    all_id = []

    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in result:
            for key in result["Contents"]:
                file_path = key["Key"]
                if file_path.endswith(f".{format}"):
                    try:
                        obj = s3.get_object(Bucket=bucket, Key=file_path)
                        audio_data = _s3_wav_loader(obj['Body'].read(), channels)
                        all_features.append(audio_data)
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
