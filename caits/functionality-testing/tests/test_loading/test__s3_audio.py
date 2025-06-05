import io
import numpy as np
import pandas as pd
import pytest
import wave
from unittest.mock import patch, MagicMock
from caits.loading._s3_audio import s3_wav_loader, s3_audio_loader, s3_wav_specs_check


def generate_wav_bytes(duration_sec=1, sr=8000, n_channels=1, dtype=np.int16):
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(dtype)
    if n_channels > 1:
        data = np.tile(data[:, None], (1, n_channels))
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(np.dtype(dtype).itemsize)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    return buf.getvalue()


def test_s3_wav_loader_shape():
    wav_bytes = generate_wav_bytes(duration_sec=0.1, sr=8000, n_channels=2)
    df, sr = s3_wav_loader(wav_bytes, mode='soundfile', dtype='float64')
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert df.shape[1] == 2
    assert isinstance(sr, int)
    # Test mono
    wav_bytes_mono = generate_wav_bytes(duration_sec=0.1, sr=8000, n_channels=1)
    df_mono, sr_mono = s3_wav_loader(wav_bytes_mono, mode='soundfile', dtype='float64')
    assert df_mono.shape[1] == 1


def test_s3_wav_loader_channels():
    wav_bytes = generate_wav_bytes(duration_sec=0.1, sr=8000, n_channels=3)
    custom_channels = ['L', 'R', 'C']
    df, sr = s3_wav_loader(wav_bytes, mode='soundfile', dtype='float64', channels=custom_channels)
    assert list(df.columns) == custom_channels


def test_s3_wav_loader_resample():
    wav_bytes = generate_wav_bytes(duration_sec=0.1, sr=8000, n_channels=1)
    target_sr = 4000
    df, sr = s3_wav_loader(wav_bytes, mode='soundfile', dtype='float64', target_sr=target_sr)
    assert sr == target_sr
    assert isinstance(df, pd.DataFrame)


def test_s3_wav_specs_check():
    wav_bytes = generate_wav_bytes(duration_sec=0.1, sr=8000, n_channels=2)
    specs = s3_wav_specs_check(wav_bytes)
    assert isinstance(specs, dict)
    assert specs['nchannels'] == 2
    assert specs['framerate'] == 8000
    assert specs['sampwidth'] == 2
    assert specs['nframes'] > 0


@pytest.fixture
def mock_s3_client():
    mock_client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {'Contents': [
            {'Key': 'class1/file1.wav'},
            {'Key': 'class2/file2.wav'}
        ]}
    ]
    mock_client.get_paginator.return_value = paginator
    # Return a valid wav file for both files
    wav_bytes = generate_wav_bytes(duration_sec=0.1, sr=8000, n_channels=1)
    mock_client.get_object.side_effect = lambda Bucket, Key: {'Body': io.BytesIO(wav_bytes)}
    return mock_client


def test_s3_audio_loader_dict_shape(monkeypatch, mock_s3_client):
    monkeypatch.setattr('boto3.client', lambda *args, **kwargs: mock_s3_client)
    result = s3_audio_loader(
        bucket='test-bucket',
        prefix='',
        endpoint_url='http://mock',
        export='dict',
        format='wav',
    )
    assert isinstance(result, dict)
    assert set(result.keys()) == {'X', 'y', 'id'}
    assert len(result['X']) == 2
    assert len(result['y']) == 2
    assert len(result['id']) == 2
    for df in result['X']:
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0
        assert df.shape[1] == 1
    assert set(result['y']) == {'class1', 'class2'}
    assert set(result['id']) == {'file1.wav', 'file2.wav'}


def test_s3_audio_loader_df_shape(monkeypatch, mock_s3_client):
    monkeypatch.setattr('boto3.client', lambda *args, **kwargs: mock_s3_client)
    result = s3_audio_loader(
        bucket='test-bucket',
        prefix='',
        endpoint_url='http://mock',
        export='df',
        format='wav',
    )
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'X', 'y', 'id'}
    assert result.shape[0] == 2
    for i, row in result.iterrows():
        assert isinstance(row['X'], pd.DataFrame)
        assert row['X'].shape[0] > 0
        assert row['X'].shape[1] == 1
        assert row['y'] in {'class1', 'class2'}
        assert row['id'] in {'file1.wav', 'file2.wav'}
