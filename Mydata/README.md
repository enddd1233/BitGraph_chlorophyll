# Processed chlorophyll-a datasets

This directory contains the processed datasets used by BiTGraph.

## Files

- `bohai_300.csv`: Bohai Sea chlorophyll-a dataset with 300 spatial nodes.
- `nanhai_265.csv`: South China Sea chlorophyll-a dataset with 265 spatial nodes.

## Format

The CSV files use one row per spatial node. The first columns store spatial metadata, including node/date identifier information and latitude/longitude coordinates. The remaining columns store time-indexed chlorophyll-a values.

`data/GenerateDataset.py` reads these CSV files, drops latitude/longitude columns for model input, sorts the temporal columns by integer time index, transposes the data to `(time, node)`, and constructs sliding-window forecasting samples.

## Missingness simulation

Controlled missing observations are generated inside `data/GenerateDataset.py` during data loading.

## Data source note

The processed files are derived from satellite ocean-color chlorophyll-a products used in the associated paper. If the raw source products cannot be redistributed through GitHub, provide their download instructions or archive link in the public release.
