import os
import wave
import pandas as pd
import numpy as np
from typing import Union, List


def _wav_loader(
        mode: str = "soundfile",
        file_path: str = None,
        channels: List[str] = ["channel_1"]
) -> Union[pd.DataFrame, dict]:
    """Loads an audio file into a DataFrame.

    Args:
        mode:
        file_path:
        channels:

    Returns:

    """
    if mode == "scipy":
        from scipy.io import wavfile
        sample_rate, audio_data = wavfile.read(file_path)
    elif mode == "pydub":
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(file_path)
        audio_data = np.array(audio.get_array_of_samples())
    elif mode == "soundfile":
        import soundfile as sf
        audio_data, sample_rate = sf.read(file_path)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    df = pd.DataFrame(audio_data, columns=channels)

    return df


def audio_loader(
        dataset_path: str,
        mode: str = "soundfile",
        format: str = "wav",
        channels: list = ["channel_1"],
        export: str = "dict"
) -> pd.DataFrame | dict:
    """Loads audio files from a directory into a DataFrame.

    Args:
        dataset_path:
        mode:
        format:
        channels:
        export: "dict" | "df"

    Returns:

    """
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
                        if format == "wav":
                            df = _wav_loader(mode, file_path,
                                             channels=channels)
                        else:
                            raise ValueError(f"Unsupported format: {format}")

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


def wav_specs_check(wav_file_path) -> dict:
    """Check the specifications of a WAV file.

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
        print(f"Sample rate: {sr} Hz")
        if num_channels == 1:
            print("Mono")
        elif num_channels == 2:
            print("Stereo")
        else:
            print(f"Multi-channel ({num_channels} channels)")

    return wf.getparams()._asdict()
