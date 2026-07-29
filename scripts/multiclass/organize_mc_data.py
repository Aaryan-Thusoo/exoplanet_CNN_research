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
MODEL_DIR = PROJECT_ROOT / "data" / "multi_model"

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
    Creates model inputs and class labels from a dataframe of flux sequences.
    :param df: Dataframe with a "flux" column to extract.
    :param label: Class label assigned to every row in the dataframe.
    :return: Stacked input data and matching class labels.
    """
    stacked = np.stack(df["flux"].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32)))
    return stacked, np.ones(len(stacked)) * label


def shuffle(
    X: np.ndarray,
    y: np.ndarray,
    kepids: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly shuffle X and y while keeping matching input/output pairs together.

    :param X: Input data.
    :param y: Output labels.
    :param seed: Random seed for reproducible shuffling.
    :return: Randomly shuffled X and y arrays.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    return X[indices], y[indices], kepids[indices]

def normalize(X: np.ndarray) -> np.ndarray:
    """
    Normalizes input data using the median of the data.
    :param X: input data
    :return: normalized data
    """
    X = (X - np.median(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)
    X = X[..., np.newaxis]
    return X


def balance_by_kepid(df_list: List[pd.DataFrame], seed: int) -> List[pd.DataFrame]:
    """
    Balance multiple dataframes by selecting the same number of unique KepIDs.
    All rows/chunks from each selected KepID are kept together.
    """
    rng = np.random.default_rng(seed)

    kepid_lists = [df["kepid"].drop_duplicates().to_numpy() for df in df_list]
    min_kepids = min(len(kepids) for kepids in kepid_lists)

    balanced_dfs = []

    for df, kepids in zip(df_list, kepid_lists):
        selected_kepids = rng.choice(kepids, size=min_kepids, replace=False)
        balanced_df = df[df["kepid"].isin(selected_kepids)].copy()
        balanced_dfs.append(balanced_df)

    return balanced_dfs


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

    kepler_seb_df = kepler_eb_df[kepler_eb_df["deep"] == False]
    kepler_deb_df = kepler_eb_df[kepler_eb_df["deep"] == True]

    # Balance classes by KepID while keeping all chunks from each selected KepID together
    kepler_exo_df, kepler_seb_df, kepler_deb_df = balance_by_kepid(
        [kepler_exo_df, kepler_seb_df, kepler_deb_df],
        seed=seed,
    )

    exo_training, exo_validation, exo_test = hp.split_by_kepid(kepler_exo_df, train_ratio=train_ratio,
                                                               val_test_ratio=val_test_ratio, seed=seed)

    seb_training, seb_validation, seb_test = hp.split_by_kepid(kepler_seb_df, train_ratio=train_ratio,
                                                            val_test_ratio=val_test_ratio, seed=seed)

    deb_training, deb_validation, deb_test = hp.split_by_kepid(kepler_deb_df, train_ratio=train_ratio,
                                                               val_test_ratio=val_test_ratio, seed=seed)

    # Label and Classify data sets
    X_exo_training, y_exo_training = create_data_classification(exo_training, 0)
    X_exo_val, y_exo_val = create_data_classification(exo_validation, 0)
    X_exo_test, y_exo_test = create_data_classification(exo_test, 0)
    X_seb_training, y_seb_training = create_data_classification(seb_training, 1)
    X_seb_val, y_seb_val = create_data_classification(seb_validation, 1)
    X_seb_test, y_seb_test = create_data_classification(seb_test, 1)
    X_deb_training, y_deb_training = create_data_classification(deb_training, 2)
    X_deb_val, y_deb_val = create_data_classification(deb_validation, 2)
    X_deb_test, y_deb_test = create_data_classification(deb_test, 2)

    # Save kepids of each section
    kepid_exo_training = exo_training["kepid"].to_numpy()
    kepid_exo_val = exo_validation["kepid"].to_numpy()
    kepid_exo_test = exo_test["kepid"].to_numpy()
    kepid_seb_training = seb_training["kepid"].to_numpy()
    kepid_seb_val = seb_validation["kepid"].to_numpy()
    kepid_seb_test = seb_test["kepid"].to_numpy()
    kepid_deb_training = deb_training["kepid"].to_numpy()
    kepid_deb_val = deb_validation["kepid"].to_numpy()
    kepid_deb_test = deb_test["kepid"].to_numpy()

    # Combine exo and eb data sets
    X_train = np.concatenate([X_exo_training, X_seb_training, X_deb_training], axis=0)
    y_train = np.concatenate([y_exo_training, y_seb_training, y_deb_training], axis=0)
    kepid_train = np.concatenate([kepid_exo_training, kepid_seb_training, kepid_deb_training], axis=0)

    X_val = np.concatenate([X_exo_val, X_seb_val, X_deb_val], axis=0)
    y_val = np.concatenate([y_exo_val, y_seb_val, y_deb_val], axis=0)
    kepid_val = np.concatenate([kepid_exo_val, kepid_seb_val, kepid_deb_val], axis=0)

    X_test = np.concatenate([X_exo_test, X_seb_test, X_deb_test], axis=0)
    y_test = np.concatenate([y_exo_test, y_seb_test, y_deb_test], axis=0)
    kepid_test = np.concatenate([kepid_exo_test, kepid_seb_test, kepid_deb_test], axis=0)

    # Shuffle data
    X_train, y_train, kepid_train = shuffle(X_train, y_train, kepid_train, seed)
    X_val, y_val, kepid_val = shuffle(X_val, y_val, kepid_val, seed+1)
    X_test, y_test, kepid_test = shuffle(X_test, y_test, kepid_test, seed+2)

    # Normalize data
    X_train = normalize(X_train)
    X_val = normalize(X_val)
    X_test = normalize(X_test)

    # Fix axes for model
    X_train = np.squeeze(X_train, axis=-1)
    X_val = np.squeeze(X_val, axis=-1)
    X_test = np.squeeze(X_test, axis=-1)

    # Save data sets for next stage
    output_dir = MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_dir / "train.npz", X=X_train, y=y_train, kepid=kepid_train)
    np.savez_compressed(output_dir / "val.npz", X=X_val, y=y_val, kepid=kepid_val)
    np.savez_compressed(output_dir / "test.npz", X=X_test, y=y_test, kepid=kepid_test)

if __name__ == "__main__":
    params = yr.load_params(yaml_file)
    seed = params["organize_seed"]
    train_ratio = params["organize_train_ratio"]
    val_test_ratio = params["organize_val_test_ratio"]
    main(seed, train_ratio, val_test_ratio)
