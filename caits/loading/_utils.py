import yaml


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
