def load_wav_instance():

    from caits.loading import load_yaml_config
    from caits.loading import wav_specs_check
    from caits.loading import wav_loader

    config = load_yaml_config("config.yml")
    audio_params = wav_specs_check("data/yes.wav")
    audio_load = wav_loader("data/yes.wav")

    return {
        "signal": audio_load[0].to_numpy(),
        "params": audio_params,
        "sr": audio_params["framerate"],
        "shape": audio_load[0].shape
    }


def load_csv_instance():
    import pandas as pd
    from caits.loading import load_yaml_config

    config = load_yaml_config("config.yml")
    df = pd.read_csv("data/scratching_eye.csv")
    

    return {
        "signal": df.to_numpy(),
        "sr": config["sampling_rate"],
        "shape": df.shape,
        "channels": df.columns.tolist()
    }
