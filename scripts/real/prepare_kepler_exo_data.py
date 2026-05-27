# region Imports
from pathlib import Path
import sys
from scripts.helpers.yaml_reading import load_params
from scripts.helpers.temp_help_name import download_lcs

import pandas as pd


# Parameters file path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "configs"))
yaml_file = PROJECT_ROOT / "configs"/ "data_process_params.yaml"


def main() -> None:
    params = load_params(yaml_file)

    input_file = params["exo_input_file"]
    skipped_rows = params["exo_skipped_rows"]
    output_dir = params["output_dir"]

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
            f"{output_dir}/kepler_exo_flux_{split[j]}_{split[j + 1]}.csv",
            index=False,
        )

if __name__ == "__main__":
    main()