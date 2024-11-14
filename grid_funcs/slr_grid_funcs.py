"""
Michael Pletcher
Created: 11/14/2024
Edited: 

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

    calc_wbz() - Function to calculate the wet-bulb 0.5 degrees
                (Celsius) level, which is a proxy for snow level.

    calc_rf_slr() - Function to calculate SLR using input features
                    from the calc_gridded_agl_vars() as well as 
                    input features from the input gridded dataset.

    calc_layer_melting_energy() - Function to calculate melting 
                                  energy in a layer.

    calc_final_layer_melting_energy() - Function that sets layer
                                        energy to zero if below
                                        ground. Also accounts for
                                        terrain if the layer
                                        contains surface elevation

    calc_total_melting_energy() - Function to calculate the total
                                  melting energy in a column

    calc_grids() - Function to create 2-d grids from desired 
                   variable

    
    


                              
"""

# Imports
import pandas as pd
import numpy as np
import config

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

def calc_rf_slr(
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
            df[key] = pd.Series(value.to_numpy().flatten())

    # Select only the columns that match model_keys
    df = df.loc[:, model_keys]
    
    # Predict SLR and reshape to HRRR 2d grid using the latitude RF feature
    df['rf_slr'] = model.predict(df)
    initslr = df['rf_slr'].to_numpy().reshape(features['lat'].shape)

    return slr

def calc_grids(ds, fpath):
    
    # AGL levels to interpolate to
    agl_levs = [300, 600, 900, 1200, 1500, 1800, 2100, 2400]
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
    rf_slr = calc_rf_slr(features, slr_model, slr_model_keys)

    # Reshape grids from 3d to 2d for faster processing
    gh_2d = ds.gh.to_numpy().reshape(ds.gh.shape[0], -1)
    t_2d = ds.t.to_numpy().reshape(ds.t.shape[0], -1)
    p_2d = ds.p.to_numpy().reshape(ds.p.shape[0], -1)
    rh_2d = ds.r.to_numpy().reshape(ds.r.shape[0], -1)
    w_2d = ds.w.to_numpy().reshape(ds.w.shape[0], -1)

    # Fill DataFrame with variables needed to calculate
    # SLR for NBM methods. A DataFrame is used here 
    # because it is a lot faster for processing compared
    # looping through the Dataset and calculating SLR
    # for a 2d DataArray.
    df = pd.DataFrame(
        {
            'z_prof': [gh_2d[:, i] for i in range(gh_2d.shape[1])],
            'temp_prof': [t_2d[:, i] for i in range(t_2d.shape[1])],
            'pres_prof': [p_2d[:, i] for i in range(p_2d.shape[1])],
            'rh_prof': [rh_2d[:, i] for i in range(rh_2d.shape[1])],
            'w_prof': [w_2d[:, i] for i in range(w_2d.shape[1])],
            'orog': ds.orog.to_numpy().reshape(-1),
        }
    )

    # Calculate Cobb SLR at each grid point
    df_slr['slr_cobb'], df_slr['w_cloudy_profile_mean'], df_slr['w_sq_profile_mean'] = zip(
        *df_slr.apply(
            lambda grid_point: slr_funcs.calc_cobb_slr(
                grid_point.z_prof,
                grid_point.temp_prof,
                grid_point.rh_prof,
                grid_point.w_prof,
                grid_point.pres_prof,
                grid_point.orog,
                grid_point.wbz,
                grid_point.mlthick
            ), axis = 1
        )
    )
    