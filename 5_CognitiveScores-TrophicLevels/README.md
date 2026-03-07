# Trophic Levels relation with Cognitive Assessments Pipeline

This repository contains a full processing pipeline for:
- Loading trophic levels
- Filtering Assessments from Original ADNI files
- Performing ComBat harmonization (neuroHarmonize)
- Computing Network Level Trophic Levels from Regions
- Computing Multiple Linear Regression Models + Plotting Results

## Repository Structure

CognitiveScores-TrophicLevels/

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


