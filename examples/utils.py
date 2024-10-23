def load_wav_instance(path, return_vector=True):

    from caits.loading import wav_specs_check
    from caits.loading import wav_loader

    audio_params = wav_specs_check(path)
    audio_load = wav_loader(path)

    if audio_params["nchannels"] == 1 and return_vector:
        ch_name = list(audio_load[0].columns)[0]
        sig = audio_load[0][ch_name].values
    elif audio_params["nchannels"] == 1 and not return_vector:
        ch_name = list(audio_load[0].columns)
        sig = audio_load[0].values

    return {
        "signal": sig,
        "params": audio_params,
        "channels": ch_name,
        "sr": audio_params["framerate"],
        "shape": sig.shape
    }


def load_csv_instance(path):
    import pandas as pd
    df = pd.read_csv(path)

    return {
        "signal": df.to_numpy(),
        "shape": df.shape,
        "channels": df.columns.tolist()
    }
