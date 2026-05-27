def download_lcs(kic, n=4894):
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