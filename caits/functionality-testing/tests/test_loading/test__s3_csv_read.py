import io
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from caits.loading._s3_csv_read import s3_csv_loader

CSV_CONTENT = "col1,col2\n1,2\n3,4\n"
CSV_CONTENT2 = "col1,col2\n5,6\n7,8\n"

@pytest.fixture
def mock_s3_client():
    mock_client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {'Contents': [
            {'Key': 'classA/fileA.csv'},
            {'Key': 'classB/fileB.csv'}
        ]}
    ]
    mock_client.get_paginator.return_value = paginator
    # Return different CSV content for each file
    def get_object_side_effect(Bucket, Key):
        if Key == 'classA/fileA.csv':
            return {'Body': io.StringIO(CSV_CONTENT)}
        elif Key == 'classB/fileB.csv':
            return {'Body': io.StringIO(CSV_CONTENT2)}
        else:
            raise FileNotFoundError
    mock_client.get_object.side_effect = get_object_side_effect
    return mock_client


def test_s3_csv_loader_dict_shape(monkeypatch, mock_s3_client):
    monkeypatch.setattr('boto3.client', lambda *args, **kwargs: mock_s3_client)
    result = s3_csv_loader(
        bucket='test-bucket',
        prefix='',
        endpoint_url='http://mock',
        export='dict',
    )
    assert isinstance(result, dict)
    assert set(result.keys()) == {'X', 'y', 'id'}
    assert len(result['X']) == 2
    assert len(result['y']) == 2
    assert len(result['id']) == 2
    for df in result['X']:
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
    assert set(result['y']) == {'classA', 'classB'}
    assert set(result['id']) == {'fileA.csv', 'fileB.csv'}


def test_s3_csv_loader_df_shape(monkeypatch, mock_s3_client):
    monkeypatch.setattr('boto3.client', lambda *args, **kwargs: mock_s3_client)
    result = s3_csv_loader(
        bucket='test-bucket',
        prefix='',
        endpoint_url='http://mock',
        export='df',
    )
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'X', 'y', 'id'}
    assert result.shape[0] == 2
    for i, row in result.iterrows():
        assert isinstance(row['X'], pd.DataFrame)
        assert row['X'].shape == (2, 2)
        assert row['y'] in {'classA', 'classB'}
        assert row['id'] in {'fileA.csv', 'fileB.csv'}


def test_s3_csv_loader_channels(monkeypatch, mock_s3_client):
    monkeypatch.setattr('boto3.client', lambda *args, **kwargs: mock_s3_client)
    result = s3_csv_loader(
        bucket='test-bucket',
        prefix='',
        endpoint_url='http://mock',
        export='dict',
        channels=['col1']
    )
    for df in result['X']:
        assert list(df.columns) == ['col1']
        assert df.shape == (2, 1)


def test_s3_csv_loader_classes(monkeypatch, mock_s3_client):
    monkeypatch.setattr('boto3.client', lambda *args, **kwargs: mock_s3_client)
    result = s3_csv_loader(
        bucket='test-bucket',
        prefix='',
        endpoint_url='http://mock',
        export='dict',
        classes=['classA']
    )
    assert len(result['X']) == 1
    assert result['y'] == ['classA']
    assert result['id'] == ['fileA.csv']
    assert result['X'][0].shape == (2, 2)
