import io
import boto3
import pandas as pd
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from typing import List, Optional, Tuple, Union, Literal
from tqdm import tqdm

from ..preprocessing import resample_2d


def s3_wav_loader(
    file_content: bytes,
    mode: str = "soundfile",
    target_sr: Optional[int] = None,
    dtype: str = "float64",
    channels: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, int]:
    """Loads and optionally resamples a mono or multichannel audio
    file from bytes into a DataFrame.

    Args:
        file_content: The audio file as bytes.
        mode: Loading mode ("soundfile", "scipy").
        target_sr: Optional target sampling rate for resampling.
        dtype: The desired data type for the audio data. Supported options:
            - "float64" (default): Double-precision floating-point, normalized to [-1, 1]
            - "float32": Single-precision floating-point, normalized to [-1, 1]
            - "int16": 16-bit signed integer, no normalization
            - "int32": 32-bit signed integer, no normalization
        channels: List of channel names for the DataFrame.

    Returns:
        pd.DataFrame: Loaded and optionally resampled audio data in 2D shape.
        int: Sample rate of the audio file.
    """
    file_obj = io.BytesIO(file_content)

    if mode == "soundfile":
        audio_data, sample_rate = sf.read(file_obj, always_2d=True, dtype=dtype)
    elif mode == "scipy":
        sample_rate, audio_data = wavfile.read(file_obj)
        if audio_data.dtype != dtype and dtype in ["float32", "float64"]:
            # Normalize to [-1, 1] for float types
            audio_data = audio_data / np.iinfo(audio_data.dtype).max
            audio_data = audio_data.astype(dtype)  # Convert to specified type
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if target_sr is not None and target_sr != sample_rate:
        # Resamples audio to target_sr per channel
        audio_data = resample_2d(audio_data, sample_rate, target_sr, dtype)
        sample_rate = target_sr

    if channels is None or len(channels) != audio_data.shape[1]:
        channels = [f"ch_{i + 1}" for i in range(audio_data.shape[1])]

    return pd.DataFrame(audio_data, columns=channels), sample_rate


def s3_audio_loader(
    bucket: str,
    prefix: str,
    endpoint_url: str,
    mode: str = "soundfile",
    format: str = "wav",
    dtype: str = "float64",
    target_sr: Optional[int] = None,
    classes: Optional[List[str]] = None,
    channels: Optional[List[str]] = None,
    export: Literal["df", "dict"] = "dict",
) -> Union[pd.DataFrame, dict]:
    """Loads audio files from an S3 bucket into a DataFrame or dictionary with optional resampling.

    Args:
        bucket: Name of the S3 bucket.
        prefix: Prefix of the S3 bucket.
        endpoint_url: URL of the S3 bucket.
        mode: Loading mode, supports "soundfile" or "scipy".
        format: Audio file format, defaults to "wav".
        dtype: The desired data type for the audio data. Supported options:
            - "float64" (default): Double-precision floating-point, normalized to [-1, 1]
            - "float32": Single-precision floating-point, normalized to [-1, 1]
            - "int16": 16-bit signed integer, no normalization
            - "int32": 32-bit signed integer, no normalization
        target_sr: Optional target sampling rate for resampling.
        classes: Optional list of directory names to include; if None, all directories are included.
        channels: List of channel names for the DataFrame.
        export: Format to export the loaded data, "dict" or "df" for DataFrame.

    Returns:
        pd.DataFrame or dict: Loaded and optionally resampled audio data with progress displayed.
    """
    s3 = boto3.client("s3", endpoint_url=endpoint_url)

    all_features = []
    all_y = []
    all_id = []

    paginator = s3.get_paginator("list_objects_v2")
    file_paths = []
    for result in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in result:
            for key in result["Contents"]:
                file_path = key["Key"]
                if file_path.endswith(f".{format}"):
                    file_paths.append(file_path)

    for file_path in tqdm(file_paths, desc="Loading audio files"):
        parts = file_path.split('/')
        if len(parts) < 2:
            continue  # Skip files not in the expected format
        subdir = parts[-2]

        # Check if the subdir (class) is desired
        if classes is None or subdir in classes:
            file = parts[-1]
            try:
                obj = s3.get_object(Bucket=bucket, Key=file_path)
                file_content = obj["Body"].read()
                df, _ = s3_wav_loader(
                    file_content,
                    mode=mode,
                    target_sr=target_sr,
                    dtype=dtype,
                    channels=channels,
                )
                all_features.append(df)
                all_y.append(subdir)
                all_id.append(file)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    if export == "df":
        return pd.DataFrame({"X": all_features, "y": all_y, "id": all_id})
    elif export == "dict":
        return {"X": all_features, "y": all_y, "id": all_id}
