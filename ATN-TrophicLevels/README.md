# Amyloid-Tau-Neurodegeneration (ATN) relation with Trophic Levels Pipeline

This repository contains a full processing pipeline for:
- Loading trophic levels and ATN biomarkers
- Performing ComBat harmonization (neuroHarmonize)
- Computing Network Level Trophic Levels and ATN from Regions
- Computing Linear Mixed Effects Models with Ridge Regularization + Plotting Results

## Repository Structure

ATN-TrophicLevels/

├── src/functions_pipeline/ # all used functions

├── src/data/ # region labels and region-to-network correspondance

├── src/main_pipeline.py # main analysis workflow


## Installation

Required versions:
- Python 3.10+
- numpy, scipy, pandas
- matplotlib
- seaborn
- statsmodels
- neuroHarmonize
- scikit-learn

## Running the Pipeline

python src/main_pipeline.py --data-dir /path/to/data/ --deriv-dir /path/to/derivatives/ --out-dir /path/to/outputs/


