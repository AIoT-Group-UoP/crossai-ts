import os
import pytest
import pandas as pd
from caits.loading import csv_loader

EXAMPLES_DATA = os.path.join(os.path.dirname(__file__), '../../../../examples/data')


def test_csv_loader_dict_airquality():
    dataset_path = os.path.join(EXAMPLES_DATA, 'AirQuality')
    result = csv_loader(dataset_path, export='dict')
    # Should be a dict with keys X, y, id
    assert isinstance(result, dict)
    assert set(result.keys()) == {'X', 'y', 'id'}
    # Should have one file loaded
    assert len(result['X']) == 1
    assert len(result['y']) == 1
    assert len(result['id']) == 1
    # X[0] should be a DataFrame
    assert isinstance(result['X'][0], pd.DataFrame)
    # y[0] should be the subdir name
    assert result['y'][0] == 'AirQuality'
    # id[0] should be the filename
    assert result['id'][0] == 'AirQuality.csv'
    # DataFrame shape: rows > 0, columns > 0
    assert result['X'][0].shape[0] > 0
    assert result['X'][0].shape[1] > 0


def test_csv_loader_df_airquality():
    dataset_path = os.path.join(EXAMPLES_DATA, 'AirQuality')
    result = csv_loader(dataset_path, export='df')
    # Should be a DataFrame with columns X, y, id
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'X', 'y', 'id'}
    # Should have one row
    assert result.shape[0] == 1
    # X column should contain a DataFrame
    assert isinstance(result.iloc[0]['X'], pd.DataFrame)
    # y column should be the subdir name
    assert result.iloc[0]['y'] == 'AirQuality'
    # id column should be the filename
    assert result.iloc[0]['id'] == 'AirQuality.csv'
    # DataFrame shape: rows > 0, columns > 0
    assert result.iloc[0]['X'].shape[0] > 0
    assert result.iloc[0]['X'].shape[1] > 0


def test_csv_loader_dict_fitabase():
    dataset_path = os.path.join(EXAMPLES_DATA, 'Fitabase_Data_3.12.16-4.11.16')
    result = csv_loader(dataset_path, export='dict')
    # Should be a dict with keys X, y, id
    assert isinstance(result, dict)
    assert set(result.keys()) == {'X', 'y', 'id'}
    # Should have as many files as there are CSVs in the directory
    expected_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    assert len(result['X']) == len(expected_files)
    assert len(result['y']) == len(expected_files)
    assert len(result['id']) == len(expected_files)
    # All X should be DataFrames
    for df in result['X']:
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    # All y should be the subdir name
    for y in result['y']:
        assert y == 'Fitabase_Data_3.12.16-4.11.16'
    # All id should be in expected_files
    for id_ in result['id']:
        assert id_ in expected_files


def test_csv_loader_df_fitabase():
    dataset_path = os.path.join(EXAMPLES_DATA, 'Fitabase_Data_3.12.16-4.11.16')
    result = csv_loader(dataset_path, export='df')
    expected_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    # Should be a DataFrame with as many rows as CSVs
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'X', 'y', 'id'}
    assert result.shape[0] == len(expected_files)
    for i, row in result.iterrows():
        assert isinstance(row['X'], pd.DataFrame)
        assert row['X'].shape[0] > 0
        assert row['X'].shape[1] > 0
        assert row['y'] == 'Fitabase_Data_3.12.16-4.11.16'
        assert row['id'] in expected_files


def test_csv_loader_dict_flatfile():
    # Test a flat CSV file (not in a subdir)
    dataset_path = EXAMPLES_DATA
    result = csv_loader(dataset_path, export='dict')
    # Should include scratching_eye.csv and scratching_nose.csv
    found = False
    for id_ in result['id']:
        if id_ in {'scratching_eye.csv', 'scratching_nose.csv'}:
            found = True
    assert found
    # All X should be DataFrames
    for df in result['X']:
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0
        assert df.shape[1] > 0


def test_csv_loader_df_flatfile():
    dataset_path = EXAMPLES_DATA
    result = csv_loader(dataset_path, export='df')
    # Should be a DataFrame
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'X', 'y', 'id'}
    # At least one row for the flat CSVs
    found = False
    for i, row in result.iterrows():
        if row['id'] in {'scratching_eye.csv', 'scratching_nose.csv'}:
            found = True
        assert isinstance(row['X'], pd.DataFrame)
        assert row['X'].shape[0] > 0
        assert row['X'].shape[1] > 0
    assert found
