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


def test_empty_signal():
    # Test with empty signal
    y = np.array([])
    with pytest.raises(ValueError):
        spc1.stft(y)
    
    with pytest.raises(ValueError):
        spc1.melspectrogram(y=y)


def test_stft_n_fft_too_large():
    # Test when n_fft is too large for the signal
    y = np.random.randn(100)
    
    # With center=True, should give warning
    with pytest.warns(UserWarning, match="n_fft=.*is too large for input signal"):
        spc1.stft(y, n_fft=200, center=True)
    
    # With center=False, should raise ValueError
    with pytest.raises(ValueError, match="n_fft=.*is too large for uncentered analysis"):
        spc1.stft(y, n_fft=200, center=False)


def test_mfcc_invalid_lifter():
    # Test with negative lifter value
    y = np.random.randn(1000)
    with pytest.raises(ValueError, match="MFCC lifter=.*must be a non-negative number"):
        spc1.mfcc(y=y, lifter=-1)


def test_complex_input_warnings():
    # Test warnings for complex input to power_to_db and amplitude_to_db
    S_complex = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
    
    with pytest.warns(UserWarning, match="power_to_db was called on complex input"):
        spc1.power_to_db(S_complex)
    
    with pytest.warns(UserWarning, match="amplitude_to_db was called on complex input"):
        spc1.amplitude_to_db(S_complex)


def test_delta_width_exceeds_data():
    # Test when width exceeds data length in delta function
    data = np.random.randn(5)
    with pytest.raises(ValueError, match="width=.*cannot exceed data.shape"):
        spc1.delta(data, width=7, mode="interp")


def test_spectrogram_missing_parameters():
    # Test spectrogram with missing parameters
    with pytest.raises(ValueError, match="Input signal must be provided"):
        spc1.spectrogram(y=None, S=None)
    
    with pytest.raises(ValueError, match="Unable to compute spectrogram"):
        spc1.spectrogram(y=np.random.randn(100), n_fft=None)


def test_istft_shape_mismatch():
    # Test istft with mismatched output shape
    stft_matrix = np.random.randn(1025, 10) + 1j * np.random.randn(1025, 10)
    out = np.zeros((100,), dtype=np.float32)  # Wrong shape
    
    with pytest.raises(ValueError, match="Shape mismatch for provided output array"):
        spc1.istft(stft_matrix, out=out)


def test_istft_non_complex_input():
    # Test istft with non-complex input
    stft_matrix = np.random.randn(1025, 10)  # Real input, not complex
    
    # Should work with real input
    result = spc1.istft(stft_matrix)
    assert isinstance(result, np.ndarray)


def test_mfcc_stats_invalid_export():
    # Test mfcc_stats with invalid export parameter
    y = np.random.randn(1000)
    with pytest.raises(ValueError, match="Unsupported export="):
        spc1.mfcc_stats(y=y, export="invalid")


def test_stft_istft_round_trip():
    # Generate a test signal (sine wave)
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    # Compute STFT
    S = spc1.stft(y, n_fft=1024, hop_length=256, win_length=1024)
    # Invert with ISTFT
    y_rec = spc1.istft(S, hop_length=256, win_length=1024, length=len(y))
    # The reconstructed signal should match the original within numerical precision
    assert y_rec.shape == y.shape
    # Allow for a small tolerance due to windowing and overlap-add
    assert np.allclose(y, y_rec, rtol=1e-4, atol=1e-4)


def test_stft_n_fft_too_large_uncentered():
    y = np.random.randn(100).astype(np.float32)
    n_fft = 200
    # Should raise ValueError when center=False and n_fft > len(y)
    with pytest.raises(ValueError, match="n_fft=.*is too large for uncentered analysis"):
        spc1.stft(y, n_fft=n_fft, center=False)


def test_melspectrogram_output():
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    n_fft = 1024
    hop_length = 256
    n_mels = 40
    melspec = spc1.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # Output should be 2D: (n_mels, n_frames)
    assert melspec.ndim == 2
    assert melspec.shape[0] == n_mels
    # Number of frames should match STFT frame count
    S, _ = spc1.spectrogram(y=y, n_fft=n_fft, hop_length=hop_length)
    expected_frames = S.shape[1]
    assert melspec.shape[1] == expected_frames
    # All values should be non-negative
    assert np.all(melspec >= 0)


def test_mfcc_negative_lifter():
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    # Negative lifter should raise ValueError
    with pytest.raises(ValueError, match="MFCC lifter=.*must be a non-negative number"):
        spc1.mfcc(y=y, sr=sr, lifter=-5)

