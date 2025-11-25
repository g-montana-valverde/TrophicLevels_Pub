# Trophic Levels + GEC Analysis Pipeline

This repository contains a full processing pipeline for:
- Importing trophic levels, perturbability maps, and effective connectivity matrices
- Cleaning demographics and site information
- Removing single-site subjects
- Performing ComBat harmonization (neuroHarmonize)
- Computing GEC metrics
- Performing regression analyses
- Plotting cortical & subcortical maps
- Categorizing brain regions into propagators / mediators / receivers
- Saving outputs

## Repository Structure

trophic-levels-project/
├── src/TrophicLevels/ # all reusable functions
├── src/main_pipeline.py # main analysis workflow
├── data/ # raw & processed data


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

python src/main_pipeline.py --dataset ADNI3 --group HCn

Outputs will be stored in: data/processed/<dataset>/<group>/


## Notes

- The atlas must be placed in `data/atlas/`
- Input `.mat` files for TL and GEC must be in `data/raw/`
- The TrophicLevels package can be imported as:
    from TrophicLevels.plotting import plot_subcort
    from TrophicLevels.metrics import run_GEC_metrics
