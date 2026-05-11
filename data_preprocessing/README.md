# Data preprocessing helpers

This directory contains helper scripts for preparing reusable data artifacts from the processed CSV files.

## Convert CSV to NPZ

```bash
python data_preprocessing/convert_csv_to_npz.py
```

Default behavior:

- Input directory: `Mydata/`
- Output directory: `data/processed_npz/`

Custom paths:

```bash
python data_preprocessing/convert_csv_to_npz.py --input-dir Mydata --output-dir data/processed_npz
```

The NPZ files contain:

- `data`: chlorophyll-a matrix with shape `(time, node)`.
- `mask_orig`: finite-value mask from the processed CSV.
- `lat`, `lon`: node coordinates.
- `time_idx`: temporal column indices.
- `node_ids`: node identifiers.
