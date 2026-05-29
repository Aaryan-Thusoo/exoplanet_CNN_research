from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.helpers.yaml_reading import load_params
from scripts.helpers.temp_help_name import download_lcs

import numpy as np
import pandas as pd

PARAMS_FILE = PROJECT_ROOT / "configs" / "data_process_params.yaml"
CHUNK_SIZE = 200
LIGHT_CURVE_LENGTH = 4894


def main() -> None:
    params = load_params(PARAMS_FILE)

    input_file = Path(params["exo_input_file"])
    skipped_rows = params["exo_skipped_rows"]
    output_dir = Path(params["exo_output_folder"])
    output_dir.mkdir(parents=True, exist_ok=True)

    kic_df = pd.read_csv(input_file, skiprows=skipped_rows)
    kic_df = kic_df.drop_duplicates(subset="kepid", keep="last").dropna(subset=["kepid"])

    failed_kepids = []

    for start in range(0, len(kic_df), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(kic_df))

        kepid_list = []
        exo_list = []

        print(f"Started exo chunk: KIC #{start} to KIC #{end}")

        for i, kepid in enumerate(kic_df.iloc[start:end]["kepid"]):
            try:
                kic_lcs = download_lcs(kepid, n=LIGHT_CURVE_LENGTH)
            except Exception as error:
                print(f"\tSkipping {kepid}: {error}")
                failed_kepids.append({"kepid": kepid, "error": str(error)})
                continue

            if not kic_lcs:
                print(f"\tSkipping {kepid}: no light curves returned")
                failed_kepids.append({"kepid": kepid, "error": "no light curves returned"})
                continue

            for chunk in kic_lcs:
                kepid_list.append(kepid)
                exo_list.append(chunk)

            print(f"\tCompleted section {i}")

        final_exo_df = pd.DataFrame(
            {
                "kepid": kepid_list,
                "flux": [np.asarray(arr).tolist() for arr in exo_list],
            }
        )

        output_file = output_dir / f"kepler_exo_flux_{start}_{end}.csv"
        final_exo_df.to_csv(output_file, index=False)

        print(f"Completed exo chunk: {output_file}")

    if failed_kepids:
        failed_df = pd.DataFrame(failed_kepids)
        failed_df.to_csv(output_dir / "failed_exo_downloads.csv", index=False)

if __name__ == "__main__":
    main()