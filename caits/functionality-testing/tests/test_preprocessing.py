import numpy as np
import pytest
from caits.preprocessing import normalize_signal, resample_signal, resample_2d, trim_signal


def test_normalize_signal_shape():
    arr = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    out = normalize_signal(arr)
    assert out.shape == arr.shape
    arr_int = np.array([0, 100, 200, 300], dtype=np.int16)
    out_int = normalize_signal(arr_int)
    assert out_int.shape == arr_int.shape
    arr_zeros = np.zeros(10)
    out_zeros = normalize_signal(arr_zeros)
    assert out_zeros.shape == arr_zeros.shape


def test_resample_signal_shape():
    arr = np.linspace(0, 1, 100)
    out = resample_signal(arr, native_sr=100, target_sr=50)
    assert out.shape == (50,)
    out2 = resample_signal(arr, native_sr=100, target_sr=200)
    assert out2.shape == (200,)


def test_resample_2d_shape():
    arr = np.random.randn(100, 2)
    out = resample_2d(arr, native_sr=100, target_sr=50)
    assert out.shape == (50, 2)
    arr_1d = np.random.randn(100)
    out_1d = resample_2d(arr_1d, native_sr=100, target_sr=25)
    assert out_1d.shape == (25, 1)
    arr_3d = np.random.randn(10, 10, 10)
    with pytest.raises(ValueError):
        resample_2d(arr_3d, native_sr=10, target_sr=5)


def test_trim_signal_shape():
    arr = np.zeros(100)
    arr[10:90] = 1
    trimmed = trim_signal(arr)
    assert trimmed.ndim == 1
    assert trimmed.shape[0] == 80
    arr2d = np.zeros((100, 2))
    arr2d[20:80, :] = 2
    trimmed2d = trim_signal(arr2d[:, 0])
    assert trimmed2d.ndim == 1
    assert trimmed2d.shape[0] == 60
