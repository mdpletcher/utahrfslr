"""
Michael Pletcher
Created: 11/14/2024
Edited: 01/27/2025

Acknowledgments: The author thanks Kevin Birk for providing
code for helping diagnose environments that require adjusting
SLR as well as helping troubleshoot Python code. We also 
acknowledge Bourgouin (2000) and Birk et al. (2021) for their 
precipitation-type work that influenced some of our code.

##### Summary #####
.py script containing functions used to calculate
2-d gridded SLR analyses from the HRRR. All 

########## Function List ##########
    calc_gridded_agl_vars() - Function to interpolate variables
                              needed to predict SLR from the 
                              random forest.

    calc_needed_vars() - Function to calculate needed
                         variables for SLR prediction

    calc_slr() - Function to calculate SLR using input features
                    from the calc_gridded_agl_vars() as well as 
                    input features from the input gridded dataset.

    calc_grids() - Function to create 2-d grids from the random
                   forest SLR

    
    


                              
"""



# Imports
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as netcdf

from metpy.units import units

# Globals
MODEL_TYPE = 'RF' # options are RF or LR
VERSION_NUM = '1.1' # options are 1.0 and 1.1 

# SLR model components
if MODEL_TYPE == 'RF':
    model = np.load(
        '../models/rf/UUtah_rf_slr_modelv%s.pickle' % VERSION_NUM, 
        allow_pickle = True
    )
    model_keys = np.load(
        '../models/rf/UUtah_rf_slr_modelv%s_keys.npy' % VERSION_NUM, 
        allow_pickle = True
    )
elif MODEL_TYPE == 'LR':
    model = np.load(
        '../models/lr/UUtah_lr_slr_modelv%s.pickle' % VERSION_NUM, 
        allow_pickle = True
    )
    model_keys = np.load(
        '../models/lr/UUtah_lr_slr_modelv%s_keys.npy' % VERSION_NUM, 
        allow_pickle = True
    )



def calc_gridded_agl_vars(ds, agl_levs):
    """
    Calculate temperature, wind speed, and relative humidity interpolated to user-specified 
    above-ground levels (AGL). These interpolated values are needed to predict SLR from 
    the random forest.

    Parameters:
    ds : xarray.Dataset
        Input dataset containing `gh` (geopotential height), `orog` (surface elevation), 
        `t` (temperature), `spd` (wind speed), and `r` (relative humidity).
    agl_levs : list
        Ground levels for interpolation.

    Returns:
    tuple of dict
        Dictionaries for temperature, wind speed, and relative humidity, keyed by AGL levels.
    """
    
    def interp_vals(val, zc, zdn, val_dn, vals):
        """Interpolate values to AGL levels."""
        return xr.where(
            (zc > 0) & (zdn < 0), 
            (zc / (zc - zdn)) * (val_dn - val) + val, 
            vals
        )

    TAGL, SPDAGL, RHAGL = {}, {}, {}

    for agl_lev in agl_levs:
        gh_agl = (ds.gh - (ds.orog + agl_lev)).compute()
        # Where gh_agl is zero, set to 1
        gh_agl = xr.where(gh_agl == 0, 1, gh_agl)

        # If the 1000mb height is > 0, use the 1000 mb 
        # value to start. Otherwise assign 0
        tvals = xr.where(gh_agl.sel(isobaricInhPa = 1000) > 0, ds.t.sel(isobaricInhPa = 1000), 0) # Temp
        spdvals = xr.where(gh_agl.sel(isobaricInhPa = 1000) > 0, ds.spd.sel(isobaricInhPa = 1000), 0) # Wind speed
        rhvals = xr.where(gh_agl.sel(isobaricInhPa = 1000) > 0, ds.r.sel(isobaricInhPa = 1000), 0) # RH

        for j in range(ds.dims['isobaricInhPa'] - 1, -1, -1):
            zc = gh_agl.isel(isobaricInhPa = j)
            zdn = gh_agl.isel(isobaricInhPa = j - 1)

            tvals = interp_vals(ds.t.isel(isobaricInhPa = j), zc, zdn, ds.t.isel(isobaricInhPa = j - 1), tvals)
            spdvals = interp_vals(ds.spd.isel(isobaricInhPa = j), zc, zdn, ds.spd.isel(isobaricInhPa = j - 1), spdvals)
            rhvals = interp_vals(ds.r.isel(isobaricInhPa = j), zc, zdn, ds.r.isel(isobaricInhPa = j - 1), rhvals)

        TAGL[agl_lev] = tvals
        SPDAGL[agl_lev] = spdvals
        RHAGL[agl_lev] = rhvals

    return TAGL, SPDAGL, RHAGL

def calc_slr(
    features,
    model, 
    model_keys,
):
    """
    Predict SLR using trained random forest model. This function pre-processes the data
    so that the feature column names match the required input feature names that are fed
    into the random forest.

    Parameters:
    features : dict
        Dictionary containing input features for the model.
    model : sklearn.ensemble.RandomForestRegressor
        Trained random forest model used for predicting SLR.
    model_keys : list
        List of feature names expected by the model.

    Returns:
    numpy.ndarray
        The predicted SLR values reshaped to the HRRR's 2D grid shape.
    """

    df = pd.DataFrame()
    # Loop through the features dictionary to create columns
    # that match the model_keys input
    for feature_key, feature_value in features.items():
        if isinstance(feature_value, dict):
            # If the feature is a dictionary, create columns for each sub-feature
            for sub_key, sub_value in feature_value.items():
                clean_key = feature_key.replace("AGL", "")  # Remove 'AGL' from the key
                column_name = "%s%02dK" % (clean_key, sub_key // 100)
                df[column_name] = pd.Series(sub_value.to_numpy().flatten())
        else:
            # For other features that are not dictionaries
            # (e.g., lat, lon, elev), add them directly
            df[key] = pd.Series(feature_value.to_numpy().flatten())

    # Select only the columns that match model_keys
    df = df.loc[:, model_keys]
    
    # Predict SLR and reshape to HRRR 2d grid using the latitude RF feature
    df['slr'] = model.predict(df)
    initslr = df['slr'].to_numpy().reshape(features['lat'].shape)

    return slr

def calc_grids(ds, fpath, fsave = False):
    
    # AGL levels to interpolate to
    agl_levs = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]
    TAGL, SPDAGL, RHAGL = calc_gridded_agl_vars(ds, agl_levs)

    # Random forest input features
    features = {
        'TAGL': TAGL, 
        'SPDAGL': SPDAGL, 
        'RAGL': RHAGL,
        'lat': ds.latitude,
        'lon': ds.longitude,
        'elev': ds.orog
    }

    # Add random forest SLR/QSF to Dataset
    ds['slr'] = calc_slr(features, slr_model, slr_model_keys)
    ds['qsf'] = ds.slr * ds.qsf

    if fsave:
        ds.to_netcdf(fpath, engine = 'h5netcdf')
    ds.close()
    del ds

    return ds

def calc_needed_vars(ds):

    # Calculate wind speed
    ds['spd'] = np.sqrt((ds.u ** 2) + (ds.v ** 2))

    # Broadcast pressure levels
    p_arr = np.ones(ds.t.shape)
    p_arr = np.array(
        [p_arr[i, :, :] * ds.isobaricInhPa[i].values for i in range(ds.isobaricInhPa.size)]
    ).transpose(0, 1, 2)
    # Add Dataset attributes to new pressure variable
    # after broadcasting pressure levels
    p = ds.t.copy().rename('p') 
    p.values = p_arr
    ds['p'] = p_arr

    return ds