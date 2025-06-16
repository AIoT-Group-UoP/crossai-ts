import os
import pandas as pd
from caits.loading import wav_loader, audio_loader, wav_specs_check

def test_wav_loader_shape():
    # Use a known mono or stereo file from examples/data
    test_file = os.path.join(os.path.dirname(__file__), '../../../../examples/data/yes.wav')
    df, sr = wav_loader(test_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0  # n_samples
    assert df.shape[1] in (1, 2)  # mono or stereo
    assert isinstance(sr, int)

def test_audio_loader_shape():
    # Use the examples/data directory
    test_dir = os.path.join(os.path.dirname(__file__), '../../../../examples/data')
    result = audio_loader(test_dir, export='dict')
    assert isinstance(result, dict)
    assert set(result.keys()) == {'X', 'y', 'id'}
    assert isinstance(result['X'], list)
    assert isinstance(result['y'], list)
    assert isinstance(result['id'], list)
    if result['X']:
        assert isinstance(result['X'][0], pd.DataFrame)
        assert result['X'][0].shape[0] > 0

def test_wav_specs_check_shape():
    test_file = os.path.join(os.path.dirname(__file__), '../../../../examples/data/yes.wav')
    specs = wav_specs_check(test_file)
    assert isinstance(specs, dict)
    assert 'nchannels' in specs
    assert 'framerate' in specs
    assert 'nframes' in specs
