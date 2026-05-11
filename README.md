# BiTGraph for Marine Chlorophyll-a Forecasting

This repository contains the code and processed datasets for the paper **Mask-Aware Biased Graph Learning for Marine Chlorophyll-a Spatiotemporal Forecasting under High Missing Rates**.

The repository is organized for model training, evaluation, and data loading. Paper-figure plotting scripts, intermediate experimental outputs, and runtime artifacts are not included.

## Repository contents

```text
github_BitGraph_chlorophyll/
 README.md
 requirements.txt
 main.py
 test_forecasting.py
 run_all_experiments.py
 batch_evaluate_masked.py
 measurement_utils.py
 Mydata/
    bohai_300.csv
    nanhai_265.csv
    README.md
 data/
    GenerateDataset.py
    __init__.py
    README.md
 data_preprocessing/
    convert_csv_to_npz.py
    README.md
 models/
    __init__.py
    BiaTCGNet/
        __init__.py
        BiaTCGNet.py
        BiaTCGNet_layer.py
 results/
    README.md
 checkpoints/
    README.md
 CITATION.cff
 .gitignore
```

## Main files

- `main.py`: main entry point for model training and evaluation.
- `test_forecasting.py`: evaluation code for forecasting metrics.
- `run_all_experiments.py`: batch experiment entry point.
- `batch_evaluate_masked.py`: batch evaluation entry point for saved checkpoints.
- `measurement_utils.py`: timing and memory measurement utilities.
- `data/GenerateDataset.py`: data loading, missing-mask generation, data splitting, and dataloader construction.
- `models/BiaTCGNet/BiaTCGNet.py`: BiTGraph model definition.
- `models/BiaTCGNet/BiaTCGNet_layer.py`: model layers and graph/temporal modules.

## Environment setup

Install the required Python packages from the repository root:

```bash
pip install -r requirements.txt
```

A GPU-enabled PyTorch environment is recommended because the current code uses CUDA during model execution.

## Data

Processed datasets are stored in `Mydata/`:

- `bohai_300.csv`
- `nanhai_265.csv`

More details are provided in `Mydata/README.md` and `data/README.md`.

## Entry commands

Start a single training/evaluation run:

```bash
python main.py
```

Start batch experiments:

```bash
python run_all_experiments.py
```

Start evaluation for saved checkpoints:

```bash
python batch_evaluate_masked.py
```

Start the standalone forecasting evaluation script:

```bash
python test_forecasting.py
```

Run the CSV-to-NPZ data conversion helper:

```bash
python data_preprocessing/convert_csv_to_npz.py
```

Command-line arguments are defined in the corresponding scripts and can be set as needed for a specific experiment.

## Outputs

Runtime outputs, logs, checkpoints, metrics, and generated artifacts are ignored by Git. Lightweight result summaries can be placed in `results/`, and optional pretrained weights can be placed in `checkpoints/` or released separately.

## Citation

If you use this code or data, please cite the associated paper. Update `CITATION.cff` with the final DOI and GitHub URL after publication.

