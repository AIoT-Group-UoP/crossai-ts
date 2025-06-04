import numpy as np
import pytest
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


def test_mfcc_stats_export():
    # Generate a simple sine wave as test audio
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Test export as array
    arr = spc1.mfcc_stats(y=y, sr=sr, export="array")
    assert isinstance(arr, np.ndarray)
    # Should be 4 blocks of n_mfcc=13 statistics, shape (13*4, 1) or (52, 1)
    assert arr.shape[0] == 13 * 4 or arr.shape[0] == 13 * 4
    # Test export as dict
    stats = spc1.mfcc_stats(y=y, sr=sr, export="dict")
    assert isinstance(stats, dict)
    assert set(stats.keys()) == {"mfcc_mean", "mfcc_std", "delta_mean", "delta2_mean"}
    for v in stats.values():
        assert isinstance(v, np.ndarray)
        assert v.shape[0] == 13


def test_delta_computation():
    # Create a ramp signal
    data = np.arange(20, dtype=float)
    # First order delta should be all ones (except at the edges)
    delta1 = spc1.delta(data, width=5, order=1)
    # Central region should be close to 1
    assert np.allclose(delta1[2:-2], 1, atol=1e-6)
    # Second order delta should be close to zero for a ramp
    delta2 = spc1.delta(data, width=5, order=2)
    assert np.allclose(delta2[2:-2], 0, atol=1e-6)


def test_db_amplitude_power_invertibility():
    # Generate positive amplitude and power arrays
    S = np.abs(np.random.randn(10, 10)) + 1e-3
    # Amplitude <-> dB
    S_db = spc1.amplitude_to_db(S)
    S_rec = spc1.db_to_amplitude(S_db)
    assert np.allclose(S, S_rec, rtol=1e-4, atol=1e-4)
    # Power <-> dB
    P = S ** 2
    P_db = spc1.power_to_db(P)
    P_rec = spc1.db_to_power(P_db)
    assert np.allclose(P, P_rec, rtol=1e-4, atol=1e-4)


def test_stft_invalid_pad_mode():
    y = np.random.randn(1024).astype(np.float32)
    # Use an unsupported pad_mode
    with pytest.raises(ValueError, match="pad_mode='wrap' is not supported"):
        spc1.stft(y, pad_mode="wrap")


def test_delta_invalid_parameters():
    data = np.arange(10, dtype=float)
    # width < 3
    with pytest.raises(ValueError, match="width must be an odd integer >= 3"):
        spc1.delta(data, width=1)
    # width even
    with pytest.raises(ValueError, match="width must be an odd integer >= 3"):
        spc1.delta(data, width=4)
    # order <= 0
    with pytest.raises(ValueError, match="order must be a positive integer"):
        spc1.delta(data, width=5, order=0)
    # order not integer
    with pytest.raises(ValueError, match="order must be a positive integer"):
        spc1.delta(data, width=5, order=1.5)


def test_power_to_db_invalid_parameters():
    S = np.abs(np.random.randn(5, 5)) + 1e-3
    # amin <= 0
    with pytest.raises(ValueError, match="amin must be strictly positive"):
        spc1.power_to_db(S, amin=0)
    # top_db < 0
    with pytest.raises(ValueError, match="top_db must be non-negative"):
        spc1.power_to_db(S, top_db=-1)
