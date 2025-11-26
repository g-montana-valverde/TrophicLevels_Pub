# Trophic Levels + Graph Metrics Relation Analysis Pipeline

This repository contains a full processing pipeline for:
- Loading trophic levels, and effective connectivity matrices (GECs)
- Performing ComBat harmonization (neuroHarmonize)
- Computing GEC metrics
- Performing regression analyses
- Categorizing brain regions into propagators / mediators / receivers
- Plotting histogramns and graph-metrics correlations

## Repository Structure

CharacterizationOfTrophicLevelsInAMyloidNegative/

├── src/functions_pipeline/ # all used functions

├── src/main_pipeline.py # main analysis workflow



## Installation

Required versions:
- Python 3.10+
- numpy, scipy, pandas
- nibabel
- matplotlib
- seaborn
- statsmodels
- bctpy
- neuroHarmonize
- scikit-learn
- tqdm

## Running the Pipeline

python src/main_pipeline.py --data-dir /path/to/data/ --out-dir /path/to/outputs/


