# Imports

from pathlib import Path
import sys
import ast
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.helpers import temp_help_name as hp
from scripts.helpers import yaml_reading as yr

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXO_DIR = PROCESSED_DIR / "exo_data"
EB_DIR = PROCESSED_DIR / "eb_data"
REAL_MODEL_DIR = PROJECT_ROOT / "data" / "real_model"

yaml_file = PROJECT_ROOT / "configs" / "data_process_params.yaml"

# Functions
def files_to_df(file_list: List[str]) -> pd.DataFrame:
    """
    Takes in list of files and concatenates them together
    :param file_list: list of file paths
    :return: data frame with all data read and concatenated
    :raises ValueError: If any file path does not have a `.csv` suffix.
    """
    dfs = []

    for file in file_list:
        df = pd.read_csv(file).drop_duplicates()
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def create_data_classification(df: pd.DataFrame, label: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates data classification real_model
    :param df: data frame with "flux" columns to extract
    :param label: Generates number classification
    :return: Stacked data set
    """
    stacked = np.stack(df["flux"].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32)))
    return stacked, np.ones(len(stacked)) * label


def shuffle(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly shuffle X and y while keeping matching input/output pairs together.

    :param X: Input data.
    :param y: Output labels.
    :param seed: Random seed for reproducible shuffling.
    :return: Randomly shuffled X and y arrays.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    return X[indices], y[indices]

def normalize(X: np.ndarray) -> np.ndarray:
    """
    Normalizes input data using the median of the data.
    :param X: input data
    :return: normalized data
    """
    X = (X - np.median(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)
    X = X[..., np.newaxis]
    return X


def main(seed, train_ratio, val_test_ratio) -> None:


    # Pull .csv file paths
    exo_files = sorted(EXO_DIR.glob("kepler_exo_flux_*.csv"))
    eb_files = sorted(EB_DIR.glob("kepler_eb_flux_*.csv"))

    # Download files into DataFrames
    kepler_exo_df = files_to_df(exo_files)
    kepler_eb_df = files_to_df(eb_files)

    # Remove any overlaps
    exo_ids = set(kepler_exo_df["kepid"])
    eb_ids = set(kepler_eb_df["kepid"])

    overlap_ids = exo_ids & eb_ids

    kepler_exo_df = kepler_exo_df[~kepler_exo_df["kepid"].isin(overlap_ids)].copy()
    kepler_eb_df = kepler_eb_df[~kepler_eb_df["kepid"].isin(overlap_ids)].copy()

    # Limit dataframes to same size
    kepler_exo_df, kepler_eb_df = hp.same_size(kepler_exo_df, kepler_eb_df)

    exo_training, exo_validation, exo_test = hp.split_by_kepid(kepler_exo_df, train_ratio=train_ratio,
                                                               val_test_ratio=val_test_ratio, seed=seed)

    eb_training, eb_validation, eb_test = hp.split_by_kepid(kepler_eb_df, train_ratio=train_ratio,
                                                            val_test_ratio=val_test_ratio, seed=seed)

    # Label and Classify data sets
    X_exo_training, y_exo_training = create_data_classification(exo_training, 0)
    X_exo_val, y_exo_val = create_data_classification(exo_validation, 0)
    X_exo_test, y_exo_test = create_data_classification(exo_test, 0)
    X_eb_training, y_eb_training = create_data_classification(eb_training, 1)
    X_eb_val, y_eb_val = create_data_classification(eb_validation, 1)
    X_eb_test, y_eb_test = create_data_classification(eb_test, 1)

    # Combine exo and eb data sets
    X_train = np.concatenate([X_exo_training, X_eb_training], axis=0)
    y_train = np.concatenate([y_exo_training, y_eb_training], axis=0)

    X_val = np.concatenate([X_exo_val, X_eb_val], axis=0)
    y_val = np.concatenate([y_exo_val, y_eb_val], axis=0)

    X_test = np.concatenate([X_exo_test, X_eb_test], axis=0)
    y_test = np.concatenate([y_exo_test, y_eb_test], axis=0)

    # Shuffle data
    X_train, y_real_train = shuffle(X_train, y_train, seed)
    X_val, y_real_val = shuffle(X_val, y_val, seed+1)
    X_test, y_real_test = shuffle(X_test, y_test, seed+2)

    # Normalize data
    X_real_train = normalize(X_train)
    X_real_val = normalize(X_val)
    X_real_test = normalize(X_test)

    # Fix axes for model
    X_real_train = np.squeeze(X_real_train, axis=-1)
    X_real_val = np.squeeze(X_real_val, axis=-1)
    X_real_test = np.squeeze(X_real_test, axis=-1)

    # Save data sets for next stage
    output_dir = REAL_MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_dir / "train.npz", X=X_real_train, y=y_real_train)
    np.savez_compressed(output_dir / "val.npz", X=X_real_val, y=y_real_val)
    np.savez_compressed(output_dir / "test.npz", X=X_real_test, y=y_real_test)

if __name__ == "__main__":
    params = yr.load_params(yaml_file)
    seed = params["organize_seed"]
    train_ratio = params["organize_train_ratio"]
    val_test_ratio = params["organize_val_test_ratio"]
    main(seed, train_ratio, val_test_ratio)