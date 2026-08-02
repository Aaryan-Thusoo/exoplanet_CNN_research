import lightkurve as lk
import numpy as np
import pandas as pd


def download_lcs_exo(kic, n=4894):
    """
    Download a Kepler light curve for a given KIC, preprocess it,
    and split the flux into equal chunks of length n.

    Parameters
    ----------
    kic : int or str
        Kepler ID.
    n : int
        Chunk length.

    Returns
    -------
    chunks : list of np.ndarray
        List of flux chunks, each of length n.
    """

    search_result = lk.search_lightcurve(f"KIC {kic}", author="Kepler")

    lcc = search_result[-5:-1].download_all()

    lc = lcc.stitch().remove_nans().remove_outliers(sigma=5).normalize()
    lc_flat, _ = lc.flatten(window_length=301, return_trend=True)

    flux = np.array(lc_flat.flux, dtype=np.float32)

    chunks = [flux[i:i+n] for i in range(0, len(flux), n) if len(flux[i:i+n]) == n]

    return chunks

def download_lcs_eb(kic, n=4894, threshold=-1):
    """
    Download a Kepler light curve for a given KIC, preprocess it,
    and split the flux into equal chunks of length n.

    Parameters
    ----------
    kic : int or str
        Kepler ID.
    n : int
        Chunk length.
    threshold : int
        Minimum depth threshold for EB to be considered deep

    Returns
    -------
    chunks : list of np.ndarray
        List of flux chunks, each of length n.
    depth : float
        Median of 100 lowest points in light curve
    deep: boolean
        True if light curve determined to be of depth greater than the set threshold
    """

    search_result = lk.search_lightcurve(f"KIC {kic}", author="Kepler")

    lcc = search_result[-5:-1].download_all()

    lc = lcc.stitch().remove_nans().remove_outliers(sigma=5).normalize()
    lc_flat, _ = lc.flatten(window_length=301, return_trend=True)

    flux = np.array(lc_flat.flux, dtype=np.float32)

    lowest_values = np.sort(flux)[:100]
    depth  = np.median(lowest_values)
    deep = depth < threshold

    chunks = [flux[i:i+n] for i in range(0, len(flux), n) if len(flux[i:i+n]) == n]

    return chunks, depth, deep



def split_df(df, split_ratio):

    split_index = int(len(df) * split_ratio)

    def check(df, idx):
        return df.iloc[idx]["kepid"]

    i = split_index
    j = split_index

    original_index = check(df, split_index)

    while check(df, i) == original_index and check(df, j) == original_index:
        i += 1
        j -= 1

    if check(df, i) != check(df, split_index):
        split_point = i
    elif check(df, j) != check(df, split_index):
        split_point = j + 1
    else:
        split_point = -1

    df_left = df[:split_point]
    df_right = df[split_point:]

    return df_left, df_right

def same_size(df1, df2):
    min_len = min(len(df1), len(df2))
    return df1.iloc[:min_len].copy(), df2.iloc[:min_len].copy()

def split_by_kepid(
    df: pd.DataFrame,
    train_ratio: float,
    val_test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Shuffle and split a dataframe by kepid groups.

    All rows with the same kepid stay in the same split.

    :param df: Dataframe containing a "kepid" column.
    :param train_ratio: Fraction of unique kepids assigned to training.
    :param val_test_ratio: Fraction of unique kepids assigned to validation.
    :param seed: Random seed for reproducible shuffling.
    :return: Train, validation, and test dataframes.
    """
    rng = np.random.default_rng(seed)

    unique_kepids = df["kepid"].drop_duplicates().to_numpy()
    rng.shuffle(unique_kepids)

    n_total = len(unique_kepids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_test_ratio)

    train_ids = set(unique_kepids[:n_train])
    val_ids = set(unique_kepids[n_train:n_train + n_val])
    test_ids = set(unique_kepids[n_train + n_val:])

    train_df = df[df["kepid"].isin(train_ids)].copy()
    val_df = df[df["kepid"].isin(val_ids)].copy()
    test_df = df[df["kepid"].isin(test_ids)].copy()

    return train_df, val_df, test_df