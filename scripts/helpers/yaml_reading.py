from pathlib import Path
import yaml

def load_params(path: str) -> dict:
    """
    Reads YAML file and returns its contents as a dictionary.
    :param path: str name of path to YAML file
    :return: dictionary with contents of YAML file
    """
    with open(path, "r") as file:
        return yaml.safe_load(file)