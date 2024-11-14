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
    Calculate temperature, wind speed, and relative humidity interpolated to specific 
    above-ground levels (AGL).

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

def calc_gridded_agl_vars(ds, agl_levs):

    '''
    :param ds: input dataset, xarray Dataset-like
    :param agl_levs: input ground levels to
    interpolate, list
    '''

    # Define empty dictionaries for each AGL variable
    TAGL, SPDAGL, RHAGL, = {}, {}, {}

    # Loop through each AGL level
    for agl_lev in agl_levs:

        # Geo Height - Surface Elev + a distance AGL. 
        # Gives Geo Heights ABOVE GROUND LEVEL + 
        # a buffer defined by agl_lev
        gh_agl = (ds.gh - (ds.orog + agl_lev)).compute()

        # Where this is zero, set to 1
        gh_agl = xr.where(gh_agl == 0, 1, gh_agl)

        # If the 1000mb height is > 0, use the 1000 mb 
        # temperature to start. Otherwise assign 0

        # Temperature 
        tvals = xr.where(
            gh_agl.sel(isobaricInhPa = 1000) > 0,
            ds.t.sel(isobaricInhPa = 1000),
            0
        )
        # Wind speed
        spdvals = xr.where(
            gh_agl.sel(isobaricInhPa = 1000) > 0,
            ds.spd.sel(isobaricInhPa = 1000),
            0
        )
        # Relative humidity
        rhvals = xr.where(
            gh_agl.sel(isobaricInhPa = 1000) > 0,
            ds.r.sel(isobaricInhPa = 1000),
            0
        )

        # Loop through all pressure levels in profile
        for j in range(ds.dims['isobaricInhPa'] - 1, -1, -1):

            # Current level variables
            zc = gh_agl.isel(isobaricInhPa = j)
            tc = ds.t.isel(isobaricInhPa = j)
            spdc = ds.spd.isel(isobaricInhPa = j)
            rhc = ds.r.isel(isobaricInhPa = j)

            # Level below current below
            zdn = gh_agl.isel(isobaricInhPa = j - 1)
            tdn = ds.t.isel(isobaricInhPa = j - 1)
            spddn = ds.spd.isel(isobaricInhPa = j - 1)
            rhdn = ds.r.isel(isobaricInhPa = j - 1)

            # Interpolate to AGL levels
            tvals = xr.where(
                (zc > 0) & (zdn < 0), 
                (zc / (zc - zdn)) * (tdn - tc) + tc, 
                tvals
            )
            spdvals = xr.where(
                (zc > 0) & (zdn < 0), 
                (zc / (zc - zdn)) * (spddn - spdc) + spdc, 
                spdvals
            )
            rhvals = xr.where(
                (zc > 0) & (zdn < 0), 
                (zc / (zc - zdn)) * (rhdn - rhc) + rhc, 
                rhvals
            )

        # Assign values to dictionaries
        TAGL[agl_lev] = tvals
        SPDAGL[agl_lev] = spdvals
        RHAGL[agl_lev] = rhvals

    return TAGL, SPDAGL, RHAGL