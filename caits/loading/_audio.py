import os
import wave
import pandas as pd
import numpy as np
from typing import Union, List, Optional
from tqdm import tqdm
import glob
from caits.preprocessing import resample_2d


def wav_loader(
        mode: str = "soundfile",
        file_path: str = None,
        channels: List[str] = None,
        target_sr: int = None
) -> pd.DataFrame:
    """Loads and optionally resamples a mono or multichannel audio
    file into a DataFrame, ensuring the output is always 2D.

    Args:
        mode: Loading mode ("soundfile", "scipy", "pydub").
        file_path: Path to the audio file.
        channels: List of channel names for the DataFrame.
                  Defaults None.
        target_sr: Optional target sampling rate for resampling.

    Returns:
        pd.DataFrame: Loaded and optionally resampled audio data in 2D shape.
    """
    # Load audio data
    if mode == "scipy":
        from scipy.io import wavfile
        sample_rate, audio_data = wavfile.read(file_path)
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)
    elif mode == "pydub":
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(file_path)
        sample_rate = audio.frame_rate
        audio_data = np.array(audio.get_array_of_samples())
        audio_data = audio_data.reshape((-1, audio.channels))
    elif mode == "soundfile":
        import soundfile as sf
        audio_data, sample_rate = sf.read(file_path, always_2d=True)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Resample if a target sample rate is provided
    if target_sr is not None and target_sr != sample_rate:
        audio_data = resample_2d(audio_data, sample_rate, target_sr)

    # Define channel names if not provided
    if channels is None:
        channels = [f"Ch_{i+1}" for i in range(audio_data.shape[1])]

    # Create DataFrame from the audio data
    df = pd.DataFrame(audio_data, columns=channels)

    return df


def audio_loader(
        dataset_path: str,
        mode: str = "soundfile",
        format: str = "wav",
        channels: list = ["Ch_1"],
        export: str = "dict",
        target_sr: int = None,
        classes: Optional[List[str]] = None
) -> Union[pd.DataFrame, dict]:
    """Loads audio files from a directory into a DataFrame
    or dictionary with optional resampling.

    Args:
        dataset_path: Path to the dataset directory.
        mode: Loading mode, supports "soundfile", "scipy", or "pydub".
        format: Audio file format, defaults to "wav".
        channels: List of channel names for the DataFrame.
        export: Format to export the loaded data, "dict" or "df" for DataFrame.
        target_sr: Optional target sampling rate for resampling.
        classes: Optional list of directory names to include;
                 if None, all directories are included.

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
                df = wav_loader(mode, file_path, channels,
                                target_sr=target_sr)
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


def wav_specs_check(
    wav_file_path: str,
    print_base: bool = False
) -> dict:
    """Checks the specifications of a WAV file.

    It returns the sample rate, the number of channels and other information
    regarding the wav file.

    Args:
        wav_file_path: Path to the WAV file.

    Returns:
        A dictionary containing the specifications of the WAV file.
    """
    with wave.open(wav_file_path, 'rb') as wf:
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
