import glob
import os
import wave
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.io import wavfile
from tqdm import tqdm

from ..preprocessing import resample_2d


def wav_loader(
    file_path: str,
    mode: str = "soundfile",
    target_sr: Optional[int] = None,
    dtype: str = "float64",
    channels: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, int]:
    """Loads and optionally resamples a mono or multichannel audio
    file into a DataFrame.

    Args:
        file_path: Path to the audio file.
        mode: Loading mode ("soundfile", "scipy").
        target_sr: Optional target sampling rate for resampling. In case of
                   more than one channel, the resampling is done per channel.
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

    if mode == "soundfile":
        audio_data, sample_rate = sf.read(file_path, always_2d=True, dtype=dtype)
    elif mode == "scipy":
        sample_rate, audio_data = wavfile.read(file_path)
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
        audio_data = resample_2d(audio_data, sample_rate, target_sr)
    else:
        target_sr = sample_rate

    if channels is None or len(channels) != audio_data.shape[1]:
        channels = [f"ch_{i + 1}" for i in range(audio_data.shape[1])]

    return pd.DataFrame(audio_data, columns=channels), target_sr


def audio_loader(
    dataset_path: str,
    mode: str = "soundfile",
    format: str = "wav",
    dtype: str = "float64",
    target_sr: Optional[List[int]] = None,
    classes: Optional[List[str]] = None,
    channels: List[str] = ["Ch_1"],
    export: Literal["df", "dict"] = "dict",
) -> Union[pd.DataFrame, Dict[str, List]]:
    """Loads audio files from a directory into a DataFrame
    or dictionary with optional resampling.

    Args:
        dataset_path: Path to the dataset directory.
        mode: Loading mode, supports "soundfile" or "scipy".
        format: Audio file format, defaults to "wav".
        dtype: The desired data type for the audio data. Supported options:
            - "float64" (default): Double-precision floating-point, normalized to [-1, 1]
            - "float32": Single-precision floating-point, normalized to [-1, 1]
            - "int16": 16-bit signed integer, no normalization
            - "int32": 32-bit signed integer, no normalization
        target_sr: Optional target sampling rate for resampling.
        classes: Optional list of directory names to include;
                 if None, all directories are included.
        channels: List of channel names for the DataFrame.
        export: Format to export the loaded data, "dict" or "df" for DataFrame.

    Returns:
        pd.DataFrame or dict: Loaded and optionally resampled audio
                              data with progress displayed using tqdm.
    """
    all_features = []
    all_y = []
    all_id = []

    search_pattern = os.path.join(dataset_path, "**", f"*.{format}")
    file_paths = glob.glob(search_pattern, recursive=True)

    for file_path in tqdm(file_paths, desc="Loading audio files"):
        subdir = os.path.basename(os.path.dirname(file_path))

        # check if desired
        if classes is None or subdir in classes:
            file = os.path.basename(file_path)
            try:
                df, _ = wav_loader(file_path, mode, target_sr, dtype, channels)
                all_features.append(df)
                # todo: add sample rate, sample width to the dictionary?
                all_y.append(subdir)
                all_id.append(file)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    if export == "df":
        return pd.DataFrame({"X": all_features, "y": all_y, "id": all_id})
    elif export == "dict":
        return {"X": all_features, "y": all_y, "id": all_id}


def wav_specs_check(wav_file_path: str, print_base: bool = False) -> Dict:
    """Checks the specifications of a WAV file.

    It returns the sample rate, the number of channels and other information
    regarding the wav file.

    Args:
        print_base: If True, prints the sample rate and number of channels.
        wav_file_path: Path to the WAV file.

    Returns:
        A dictionary containing the specifications of the WAV file.
    """
    with wave.open(wav_file_path, "rb") as wf:
        num_channels = wf.getnchannels()
        sr = wf.getframerate()
        if print_base:
            print(f"Sample rate: {sr} Hz")

        if num_channels == 1 and print_base:
            print("Mono")
        elif num_channels == 2 and print_base:
            print("Stereo")
        elif num_channels > 2 and print_base:
            print(f"Multi-channel: ({num_channels} channels)")

    return wf.getparams()._asdict()
