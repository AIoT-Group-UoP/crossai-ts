import os
import tempfile
import shutil
import json
import yaml
import pytest
from caits.loading import load_yaml_config, json_loader


def test_load_yaml_config_success():
    data = {'a': 1, 'b': {'c': 2}}
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        loaded = load_yaml_config(path)
        assert loaded == data
    finally:
        os.remove(path)


def test_load_yaml_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_yaml_config('nonexistent_file.yaml')


def test_load_yaml_config_invalid_yaml():
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        f.write('not: [valid: [yaml')
        path = f.name
    try:
        with pytest.raises(yaml.YAMLError):
            load_yaml_config(path)
    finally:
        os.remove(path)


def test_json_loader_basic():
    temp_dir = tempfile.mkdtemp()
    try:
        # Create two JSON files in different subdirs
        os.makedirs(os.path.join(temp_dir, 'classA'))
        os.makedirs(os.path.join(temp_dir, 'classB'))
        dataA = {'foo': 1}
        dataB = {'bar': 2}
        fileA = os.path.join(temp_dir, 'classA', 'fileA.json')
        fileB = os.path.join(temp_dir, 'classB', 'fileB.json')
        with open(fileA, 'w') as f:
            json.dump(dataA, f)
        with open(fileB, 'w') as f:
            json.dump(dataB, f)
        result = json_loader(temp_dir)
        assert set(result.keys()) == {'fileA', 'fileB'}
        assert result['fileA'] == dataA
        assert result['fileB'] == dataB
    finally:
        shutil.rmtree(temp_dir)


def test_json_loader_classes():
    temp_dir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(temp_dir, 'classA'))
        os.makedirs(os.path.join(temp_dir, 'classB'))
        dataA = {'foo': 1}
        dataB = {'bar': 2}
        fileA = os.path.join(temp_dir, 'classA', 'fileA.json')
        fileB = os.path.join(temp_dir, 'classB', 'fileB.json')
        with open(fileA, 'w') as f:
            json.dump(dataA, f)
        with open(fileB, 'w') as f:
            json.dump(dataB, f)
        result = json_loader(temp_dir, classes=['classA'])
        assert set(result.keys()) == {'fileA'}
        assert result['fileA'] == dataA
    finally:
        shutil.rmtree(temp_dir)


def test_json_loader_invalid_json():
    temp_dir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(temp_dir, 'classA'))
        fileA = os.path.join(temp_dir, 'classA', 'fileA.json')
        with open(fileA, 'w') as f:
            f.write('{not valid json')
        # Should print error and skip
        result = json_loader(temp_dir)
        assert 'fileA' not in result
    finally:
        shutil.rmtree(temp_dir)
