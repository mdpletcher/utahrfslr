# Machine learning models for predicting snow-to-liquid ratio

## Intro
This repository is the code associated with the WAF manuscript titled "Snow-to-liquid ratio prediction over the contiguous United States using machine learning" written by Pletcher, M.D., P. G. Veals, R. J. Chase, S. Hilberg, N. Newman, A. A. Rosenow, and W. James Steenburgh, _in review_.

## About
Snow-to-liquid ratio (SLR), or the ratio of freshly fallen snow to liquid precipitation equivalent, is used operationally to forecast snowfall amount and diagnose avalanche hazards during winter storms. Often, current operational SLR prediction methods focus on specific locations or regions, which may introduce bias when applied to other areas. Thus, we have developed several machine learning (ML) models (primarily using random forests) to predict SLR more accurately across the contiguous United States (CONUS) using a CONUS-wide training dataset [[see the Pletcher et al. (2024) repo for our first efforts in this area]](https://github.com/mdpletcher/SLR_random_forest_pletcher). These ML models can be applied to any weather modeling system and outperform existing SLR prediction methods used by the National Weather Service (NWS). 

In this repository, you’ll find the code used to build the ML models and their datasets, code for predicting SLR using the current NWS prediction methods, and Jupyter notebook examples on how to forecast SLR with the machine learning models and NWS National Blends of Models (NBM) methods using forecast profiles and 3-d model grids.

Funding for this research was provided by the NOAA Weather Program Office and the NWS CSTAR Program.

## Overview of python scripts
- `hrrr_config.py`: Configuration file for HRRR scripts. Can be modified for user needs.
- `nbm_config.py`: Configuration file for NBM SLR methods. These should not be modified.
- `hrrr_funcs.py`: Functions to read, process, and save HRRR data, specifically extracting individual HRRR profiles
- `era5_funcs.py`: Functions to read, process, and save extracted ERA5 1-d and 2-d variables.
- `slr_grid_funcs`: Calculate SLR for a 2-d model grid.
- `nbm_slr_funcs`: Functions for calculating SLR with NBM methods.
- `ptype_funcs.py`: Functions for calculating variables for determining precipitation type. We found these methods to produce worse SLR forecasts, so they were not included in the paper results.
- `postprocess.py`: Functions for postprocessing gridded model data. Most compatible with HRRR data.
- `train.py`: Train and evaluate ML models
- `roebber_ens_members.py`: Weights and biases used for Roebber SLR method

## Overview of Jupyter Notebook examples
- `predict_slr.ipynb`: Examples on how to predict SLR with a random forest using 1-d HRRR profiles and the HRRR's 3-d model grid
- `train_model.ipynb`: Examples on training/validating linear regression and random forest SLR models
