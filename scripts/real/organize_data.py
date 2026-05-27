# Imports
from pathlib import Path

# Functions
def files_to_df(file_list: List[str]) -> pd.DataFrame:
    """
    Takes in list of files and concatenates them together
    Every file must be a .csv file.

    :param file_list: list of file paths
    :return: data frame with all data read and concatenated
    """
    dfs = []

    for file in file_list:
        file_name = f"data/Kepler_Data/{file}"
        df = pd.read_csv(file_name).drop_duplicates()
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def create_data_classification(df, label):
    """
    Creates data classification model
    :param df: data frame with "flux" columns to extract
    :param label: Generates number classification
    :return: Stacked data set
    """
    stacked = np.stack(df["flux"].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32)))
    return stacked, np.ones(len(stacked)) * label


def shuffle(X, y):
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    return X, y

def normalize(X):
    X = (X - np.median(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)
    X = X[..., np.newaxis]
    return X


def main() -> None:
    folder = Path("data/Kepler_data/")
    files = [f.name for f in folder.iterdir() if f.is_file()]

    Exo_files = [file for file in files if "kepler_flux" in file]
    EB_files = [file for file in files if "keplerEB" in file]

    kepler_exo_df = files_to_df(Exo_files)
    kepler_eb_df = files_to_df(EB_files)

    exo_ids = set(kepler_exo_df["kepid"])
    eb_ids = set(kepler_eb_df["kepid"])

    exo_eb_overlap = exo_ids & eb_ids

    kepler_exo_df = kepler_exo_df[~kepler_exo_df["kepid"].isin(overlap_ids)].copy()
    kepler_eb_df = kepler_eb_df[~kepler_eb_df["kepid"].isin(overlap_ids)].copy()

    exo_validation = kepler_exo_df.loc[7000:7350]
    exo_test = kepler_exo_df.loc[7351:]
    exo_training = kepler_exo_df.loc[:7000]

    eb_validation = kepler_eb_df.loc[6999:7350]
    eb_test = kepler_eb_df.loc[7351:]
    eb_training = kepler_eb_df.loc[:6998]

    # Label and Classify data sets
    X_exo_training, y_exo_training = create_data_classification(exo_training, 0)
    X_exo_val, y_exo_val = create_data_classification(exo_validation, 0)
    X_exo_test, y_exo_test = create_data_classification(exo_test, 0)
    X_eb_training, y_eb_training = create_data_classification(eb_training, 1)
    X_eb_val, y_eb_val = create_data_classification(eb_validation, 1)
    X_eb_test, y_eb_test = create_data_classification(eb_test, 1)

    """kepid_exo_train = exo_training["kepid"].to_numpy()
    kepid_exo_val = exo_validation["kepid"].to_numpy()
    kepid_exo_test = exo_test["kepid"].to_numpy()

    kepid_eb_train = eb_training["kepid"].to_numpy()
    kepid_eb_val = eb_validation["kepid"].to_numpy()
    kepid_eb_test = eb_test["kepid"].to_numpy()"""

    X_train = np.concatenate([X_exo_training, X_eb_training], axis=0)
    y_train = np.concatenate([y_exo_training, y_eb_training], axis=0)

    X_val = np.concatenate([X_exo_val, X_eb_val], axis=0)
    y_val = np.concatenate([y_exo_val, y_eb_val], axis=0)

    X_test = np.concatenate([X_exo_test, X_eb_test], axis=0)
    y_test = np.concatenate([y_exo_test, y_eb_test], axis=0)

    X_train, y_real_train = shuffle(X_train, y_train)
    X_val, y_real_val = shuffle(X_val, y_val)
    X_test, y_real_test = shuffle(X_test, y_test)

    X_real_train = normalize(X_train)
    X_real_val = normalize(X_val)
    X_real_test = normalize(X_test)

    X_real_train = np.squeeze(X_real_train, axis=-1)
    X_real_val = np.squeeze(X_real_val, axis=-1)
    X_real_test = np.squeeze(X_real_test, axis=-1)

    

