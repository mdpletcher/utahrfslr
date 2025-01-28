# Machine learning models for predicting snow-to-liquid ratio
## About
Snow-to-liquid ratio (SLR), or the ratio of freshly fallen snow to liquid precipitation equivalent, is used operationally to forecast snowfall amount and diagnose avalanche hazards during winter storms. Often, current operational SLR prediction methods focus on specific locations or regions, which may introduce bias when applied to other areas. Thus, we have developed several machine learning (ML) models (primarily using random forests) to skillfully predict SLR across the contiguous United States (CONUS) using a CONUS-wide training dataset. These ML models can be applied to any weather modeling system and outperform existing SLR prediction methods used by the National Weather Service (NWS). 

In this repository, you’ll find the code used to build the ML models and their datasets, code for predicting SLR using the current NWS prediction methods, and Jupyter notebook examples on how to forecast SLR using a forecast profile and 3-d model grids.

Funding for this research was provided by the NOAA Weather Program Office and the NWS CSTAR Program.
