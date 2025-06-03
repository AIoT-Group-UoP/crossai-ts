import pandas as pd
import numpy as np
import caits.fe._spectrum as spc1
import dev.fe._spectrum as spc2

def init_dataset(uni_dim=False):

    data = pd.read_csv("examples/data/AirQuality/AirQuality.csv", sep=";")

    if uni_dim:
        data = data[["CO(GT)"]]

    else:
        data = data.drop(columns=['Date', 'Time'])

    data = data.applymap(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
    data = data.fillna(0)

    return data.values.T


def test_spectrogram():
    data = init_dataset(uni_dim=True)
    
    spec1 = spc2.spectrogram(
        y=data,
        n_fft=2048,
        win_length=10,
        hop_length=512,
        axis=1
    )

    spec2 = spc2.spectrogram(
        y=data.T,
        n_fft=2048,
        win_length=10,
        hop_length=512,
        axis=0
    )

    assert np.array_equal(spec1[0], spec2[0])
    assert spec1[1] == spec2[1]
    

def test_stft():
    data = init_dataset(uni_dim=True)

    stft1 = spc2.stft(
        y=data,
        n_fft=2048,
        hop_length=512,
        win_length=10,
        axis=0
    )

    stft2 = spc2.stft(
        y=data.T,
        n_fft=2048,
        hop_length=512,
        win_length=10,
        axis=1
    )

    assert np.all(stft1 == stft2)


def test_istft():
    data = init_dataset(uni_dim=True)

    stft1 = spc1.stft(
        y=data,
        n_fft=2048,
        hop_length=512,
        win_length=10,
    )

    stft2 = spc2.stft(
        y=data.T,
        n_fft=2048,
        hop_length=512,
        win_length=10,
        axis=0
    )

    print()
    print(stft1.shape)
    print(stft2.shape)

    istft1 = spc1.istft(
        stft_matrix=stft1[0],
        hop_length=512,
        n_fft=2048,
        win_length=10
    )

    istft2 = spc1.istft(
        stft_matrix=stft2[0],
        hop_length=512,
        n_fft=2048,
        win_length=10
    )

    print()
    print(istft1.shape)
    print(istft2.shape)

    assert np.array_equal(istft1, istft2)