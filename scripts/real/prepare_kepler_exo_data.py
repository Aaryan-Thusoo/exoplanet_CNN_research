from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.helpers.yaml_reading import load_params
from scripts.helpers.temp_help_name import download_lcs

import numpy as np
import pandas as pd


yaml_file = PROJECT_ROOT / "configs" / "data_process_params.yaml"


def main() -> None:
    params = load_params(yaml_file)

    input_file = params["exo_input_file"]
    skipped_rows = params["exo_skipped_rows"]
    output_dir = Path(params["exo_output_folder"])
    output_dir.mkdir(parents=True, exist_ok=True)

    KIC_df = pd.read_csv(input_file, skiprows=skipped_rows)
    KIC_df.drop_duplicates(subset="kepid", keep="last").dropna()

    split = np.arange(0, len(KIC_df), 200)

    for j in range(len(split) - 1):
        kepid_list = []
        exo_list = []

        print(f"Started Split {j}: KIC #{split[j]} to KIC #{split[j + 1]}")

        for i, kepid in enumerate(KIC_df.iloc[split[j]:split[j + 1]]["kepid"]):
            kic_lcs = download_lcs(kepid, n=4894)  # returns a list of equal-length chunks

            # Add one row per chunk
            for chunk in kic_lcs:
                kepid_list.append(kepid)
                exo_list.append(chunk)

            print(f"\tCompleted section {i}")

        print(f"Completed {j}")

        final_exo_df = pd.DataFrame({
            "kepid": kepid_list,
            "flux": [arr.tolist() for arr in exo_list]
        })

        final_exo_df.to_csv(
            output_dir / f"kepler_exo_flux_{split[j]}_{split[j + 1]}.csv",
            index=False,
        )

if __name__ == "__main__":
    main()