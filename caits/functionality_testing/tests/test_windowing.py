import numpy as np
import pandas as pd
import pytest
import caits.windowing as wnd


def test_rolling_window_df_shape():
    df = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
    windows = wnd.rolling_window_df(df, ws=20, overlap=10)
    assert isinstance(windows, list)
    assert all(isinstance(w, pd.DataFrame) for w in windows)
    assert all(w.shape[0] == 20 for w in windows)
    # Test with 1D DataFrame
    df1d = pd.DataFrame(np.random.randn(100, 1), columns=['a'])
    windows1d = wnd.rolling_window_df(df1d, ws=20, overlap=10)
    assert all(w.shape[0] == 20 for w in windows1d)


def test_sliding_window_df_shape():
    df = pd.DataFrame(np.random.randn(100, 2), columns=['a', 'b'])
    windows = wnd.sliding_window_df(df, window_size=20, overlap=5)
    assert isinstance(windows, list)
    assert all(isinstance(w, pd.DataFrame) for w in windows)
    assert all(w.shape[0] == 20 for w in windows)
    # Test with 1D DataFrame
    df1d = pd.DataFrame(np.random.randn(100, 1), columns=['a'])
    windows1d = wnd.sliding_window_df(df1d, window_size=20, overlap=5)
    assert all(w.shape[0] == 20 for w in windows1d)


def test_windowing_df_shape():
    # Create a DataFrame with 'X' as a DataFrame and 'y' as a label
    X = pd.DataFrame(np.random.randn(100, 2), columns=['a', 'b'])
    df = pd.DataFrame({'X': [X], 'y': [1]})
    out = wnd.windowing_df(df, ws=20, overlap=10, mode='dict')
    assert isinstance(out, dict)
    assert 'X' in out and 'y' in out
    assert all(isinstance(w, pd.DataFrame) for w in out['X'])
    assert all(w.shape[0] == 20 for w in out['X'])
    out_df = wnd.windowing_df(df, ws=20, overlap=10, mode='df')
    assert isinstance(out_df, pd.DataFrame)
    assert set(out_df.columns) == {'X', 'y'}
    assert all(isinstance(w, pd.DataFrame) for w in out_df['X'])
    assert all(w.shape[0] == 20 for w in out_df['X'])


def test_frame_signal_shape():
    x = np.random.randn(100)
    frame_length = 20
    hop_length = 10
    frames = wnd.frame_signal(x, frame_length, hop_length)
    n_frames = 1 + int(np.floor((len(x) - frame_length) / hop_length))
    assert frames.shape == (frame_length, n_frames)
    # Test with 2D input (n_samples, n_channels)
    x2d = np.random.randn(100, 2)
    frames2d = wnd.frame_signal(x2d[:, 0], frame_length, hop_length)
    assert frames2d.shape == (frame_length, n_frames)


def test_create_chunks_shape():
    x = np.arange(100)
    chunk_length = 20
    chunks = wnd.create_chunks(x, chunk_length)
    assert isinstance(chunks, list)
    assert all(isinstance(c, np.ndarray) for c in chunks)
    assert all(len(c) <= chunk_length for c in chunks)
    # Test with 2D input (n_samples, n_channels)
    x2d = np.random.randn(100, 2)
    chunks2d = wnd.create_chunks(x2d, chunk_length)
    assert all(isinstance(c, np.ndarray) for c in chunks2d)
    assert all(c.shape[0] <= chunk_length for c in chunks2d)
    assert all(c.shape[1] == 2 for c in chunks2d)
