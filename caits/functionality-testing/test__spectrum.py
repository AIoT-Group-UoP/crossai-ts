import numpy as np
import caits.fe._spectrum as spc1
import dev.fe._spectrum as spc2
from _utils import init_dataset

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
    data = init_dataset(uni_dim=False)

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
        axis=1
    )

    istft1 = spc1.istft(
        stft_matrix=stft1,
        hop_length=512,
        n_fft=2048,
        win_length=10,
        axis=0,
    )

    istft2 = spc1.istft(
        stft_matrix=stft2,
        hop_length=512,
        n_fft=2048,
        win_length=10,
        axis=1,
    )

    assert np.array_equal(istft1, istft2.T)