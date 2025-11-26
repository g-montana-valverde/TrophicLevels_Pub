# Trophic Levels + Directedness Groups Comparisons Pipeline

This repository contains a full processing pipeline for:
- Loading trophic levels and directedness
- Performing ComBat harmonization (neuroHarmonize)
- Computing Network Level Trophic Levels from Regions
- Performing Group Comparisons Statistics for Multiple Linear Regression
- Plotting Results

## Repository Structure

CharacterizationOfTrophicLevelsInAMyloidNegative/

├── src/functions_pipeline/ # all used functions

├── src/data/ # region labels and region-to-network correspondance

├── src/main_pipeline.py # main analysis workflow


## Installation

Required versions:
- Python 3.10+
- numpy, scipy, pandas
- matplotlib
- statsmodels
- neuroHarmonize
- tqdm
- ptitprince

## Running the Pipeline

python src/main_pipeline.py --data-dir /path/to/data/ --deriv-dir /path/to/derivatives/ --out-dir /path/to/outputs/


