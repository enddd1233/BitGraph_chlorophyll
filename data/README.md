# Data loading

`GenerateDataset.py` implements the data pipeline used by the training scripts.

Main components:

- Dataset loading.
- Controlled missing-mask generation.
- Sliding-window sample construction.
- Chronological data splitting.
- Dataloader and scaler construction.

Supported marine datasets:

- `Bohai`
- `Nanhai`

The dataset names used by the scripts should match the processed files provided in `Mydata/`.
