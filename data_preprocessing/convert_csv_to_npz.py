import os
import argparse

import numpy as np
import pandas as pd


def convert_csv_to_npz(csv_path: str, out_dir: str) -> None:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 4:
        raise ValueError(f"CSV file {csv_path} must have at least 4 columns (id, lat, lon, time...).")

    node_ids = df.iloc[:, 0].to_numpy()
    lat = df.iloc[:, 1].to_numpy(dtype=np.float32)
    lon = df.iloc[:, 2].to_numpy(dtype=np.float32)

    values = df.iloc[:, 3:]
    data = values.to_numpy(dtype=np.float32).T  # (T, N)
    mask_orig = ~np.isnan(data)

    try:
        time_idx = np.array(values.columns, dtype=np.int32)
    except ValueError:
        time_idx = np.arange(data.shape[0], dtype=np.int32)

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    dataset_name = base_name.split("_")[0]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, dataset_name + ".npz")

    np.savez_compressed(
        out_path,
        data=data,
        mask_orig=mask_orig,
        lat=lat.astype(np.float32),
        lon=lon.astype(np.float32),
        time_idx=time_idx,
        node_ids=node_ids,
        dataset_name=np.array(dataset_name),
    )

    print(f"Saved {dataset_name} to {out_path} with data shape {data.shape} (T, N)")


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(base_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=os.path.join(repo_root, "Mydata"))
    parser.add_argument("--output-dir", default=os.path.join(repo_root, "data", "processed_npz"))
    args = parser.parse_args()

    csv_files = [
        "bohai_300.csv",
        "nanhai_265.csv",
    ]

    for fname in csv_files:
        csv_path = os.path.join(args.input_dir, fname)
        if os.path.exists(csv_path):
            convert_csv_to_npz(csv_path, args.output_dir)
        else:
            print(f"Warning: {csv_path} not found, skipped.")


if __name__ == "__main__":
    main()
