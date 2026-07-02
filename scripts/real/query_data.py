import pandas as pd
import requests
import certifi
from io import StringIO
from urllib.parse import urlencode

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.helpers.yaml_reading import load_params

PARAMS_FILE = PROJECT_ROOT / "configs" / "data_process_params.yaml"
SAVE_FILE = PROJECT_ROOT / "data" / "raw"

BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
EB_URL = "https://archive.stsci.edu/kepler/eclipsing_binaries.html"

def query_exoplanet_archive(query: str) -> pd.DataFrame:
    params = {
        "query": query,
        "format": "csv",
    }

    response = requests.get(
        BASE_URL,
        params=params,
        verify=certifi.where(),
        timeout=60,
    )
    response.raise_for_status()

    return pd.read_csv(StringIO(response.text))

def eb_html_query() -> pd.DataFrame:
    response = requests.get(EB_URL, verify=certifi.where(), timeout=60)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    eb_df = tables[0]  # adjust if needed after inspecting tables

    eb_df.columns = (
        eb_df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    eb_df = eb_df.rename(columns={"kepler_id": "kepid"})

    return eb_df.iloc[:-4].copy()


def chunks(values, size):
    for i in range(0, len(values), size):
        yield values[i:i + size]

def main():
    params = load_params(PARAMS_FILE)
    kep_query_cols_list = params["kep_query_cols"]
    kep_query_cols = ", ".join(kep_query_cols_list)

    eb_table_cols = params["eb_table_cols"]

    transit_query = f"""
                    SELECT {kep_query_cols}
                    FROM cumulative
                    WHERE koi_disposition = 'CONFIRMED' """

    transit_df = query_exoplanet_archive(transit_query)

    eb_df = eb_html_query()

    eb_kepids = eb_df["kepid"].dropna().astype(int).unique().tolist()

    all_results = []

    for kepid_chunk in chunks(eb_kepids, 200):
        kepid_string = ",".join(str(kepid) for kepid in kepid_chunk)

        query = f"""
        SELECT {kep_query_cols}
        FROM cumulative
        WHERE kepid IN ({kepid_string})
        """

        chunk_df = query_exoplanet_archive(query)
        all_results.append(chunk_df)

    eb_snr_df = pd.concat(all_results, ignore_index=True)

    eb_snr_max = (
        eb_snr_df
        .groupby("kepid", as_index=False)
        .agg(koi_model_snr=("koi_model_snr", "max"))
    )

    eb_df["kepid"] = pd.to_numeric(eb_df["kepid"], errors="coerce")
    eb_snr_max["kepid"] = pd.to_numeric(eb_snr_max["kepid"], errors="coerce")

    eb_df = eb_df.dropna(subset=["kepid"]).copy()
    eb_snr_max = eb_snr_max.dropna(subset=["kepid"]).copy()

    eb_df["kepid"] = eb_df["kepid"].astype(int)
    eb_snr_max["kepid"] = eb_snr_max["kepid"].astype(int)

    eb_with_snr = eb_df.merge(
        eb_snr_max,
        on="kepid",
        how="left"
    )

    eb_with_snr = eb_with_snr[eb_table_cols].fillna("")

    transit_df.to_csv(SAVE_FILE / "Kepler_Confirmed_ExoPlanets.csv", index=False)
    eb_with_snr.to_csv(SAVE_FILE / "Kepler_Confirmed_EB.csv", index=False)

if __name__ == "__main__":
    main()