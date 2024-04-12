import yaml
import os
import glob
import json
from tqdm import tqdm
from typing import List, Optional


def load_yaml_config(config_path: str) -> dict:
    """Loads a YAML configuration file from a specified path.

    This function attempts to open and parse a YAML file, raising exceptions if
    the file cannot be found or if there is an error parsing the YAML content.

    Args:
        config_path: The filesystem path to the YAML configuration file.

    Returns:
        dict: The configuration loaded from the YAML file.

    Raises:
        FileNotFoundError: If the YAML configuration file cannot be found at
                           the specified path.
        yaml.YAMLError: If an error occurs during parsing of the YAML content.
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file \
                                not found: {config_path}") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML \
                             configuration: {config_path}") from e


def json_loader(dataset_path: str, classes: Optional[List[str]] = None) -> dict:
    """Loads JSON files from a directory, ensuring keys do
    not include file extensions. Each JSON file's contents are stored
    as a dictionary under the corresponding key.

    Args:
        dataset_path: Path to the dataset directory containing JSON files.

    Returns:
        dict: Dictionary with filenames (without extensions) as
              keys and JSON contents as values.
    """
    json_data = {}

    # Generate a search pattern to find JSON files in the dataset directory
    search_pattern = os.path.join(dataset_path, "**", "*.json")
    file_paths = glob.glob(search_pattern, recursive=True)

    for file_path in tqdm(file_paths, desc="Loading JSON files"):
        
        subdir = os.path.basename(os.path.dirname(file_path))
        # check if desired
        if classes is None or subdir in classes:
            # Extract filename without extension
            filename = os.path.splitext(os.path.basename(file_path))[0]
            try:
                with open(file_path, 'r') as f:
                    # Assuming the top-level JSON structure is
                    # an object (i.e., a dictionary)
                    data = json.load(f)
                json_data[filename] = data
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    return json_data
