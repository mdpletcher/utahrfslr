"""
Michael Pletcher
Created: 11/14/2024
Edited: 

##### Summary #####
.py script containing functions used to calculate
2-d gridded SLR analyses. 


"""

# Imports
import pandas as pd
import numpy as np
import config

def calc_gridded_agl_vars(ds, agl_levs):

    # Define empty dictionaries for each AGL variable
    #TAGL, SPDAGL, QAGL, RHAGL, WAGL, = {}, {}, {}, {}, {}
    TAGL, SPDAGL, RHAGL, = {}, {}, {}

    # Loop through each AGL level
    for agl_lev in agl_levs:
        # Geo Height - Surface Elev + a distance AGL. 
        # Gives Geo Heights ABOVE GROUND LEVEL + a buffer defined by agl_lev
        gh_agl = (ds.gh - (ds.orog + agl_lev)).compute()

        # Where this is zero, set to 1.0
        gh_agl = xr.where(gh_agl == 0, 1, gh_agl)

        # # If the 1000mb height is > 0, use the 1000 mb temperature to start. Otherwise assign 0
        tvals = xr.where(
            gh_agl.sel(isobaricInhPa = 1000) > 0,
            ds.t.sel(isobaricInhPa = 1000),
            0
        )
        spdvals = xr.where(
            gh_agl.sel(isobaricInhPa = 1000) > 0,
            ds.spd.sel(isobaricInhPa = 1000),
            0
        )
        rhvals = xr.where(
            gh_agl.sel(isobaricInhPa = 1000) > 0,
            ds.r.sel(isobaricInhPa = 1000),
            0
        )
        '''
        qvals = xr.where(
            gh_agl.sel(isobaricInhPa = 1000) > 0,
            ds.q.sel(isobaricInhPa = 1000),
            0
        )

        wvals = xr.where(
            gh_agl.sel(isobaricInhPa = 1000) > 0,
            ds.w.sel(isobaricInhPa = 1000),
            0
        )
        '''

        # Loop through all vertical levels
        for j in range(ds.dims['isobaricInhPa'] - 1, -1, -1):

            # Current level variables
            zc = gh_agl.isel(isobaricInhPa = j)
            tc = ds.t.isel(isobaricInhPa = j)
            spdc = ds.spd.isel(isobaricInhPa = j)
            rhc = ds.r.isel(isobaricInhPa = j)
            #qc = ds.q.isel(isobaricInhPa = j)
            #wc = ds.w.isel(isobaricInhPa = j)

            # Level below current below
            zdn = gh_agl.isel(isobaricInhPa = j - 1)
            tdn = ds.t.isel(isobaricInhPa = j - 1)
            spddn = ds.spd.isel(isobaricInhPa = j - 1)
            rhdn = ds.r.isel(isobaricInhPa = j - 1)
            #qdn = ds.q.isel(isobaricInhPa = j - 1)
            #wdn = ds.w.isel(isobaricInhPa = j - 1)

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
            '''
            qvals = xr.where(
                (zc > 0) & (zdn < 0), 
                (zc / (zc - zdn)) * (qdn - qc) + qc, 
                qvals
            )
            wvals = xr.where(
                (zc > 0) & (zdn < 0), 
                (zc / (zc - zdn)) * (wdn - wc) + wc, 
                wvals
            )
            '''

        # Assign values to dictionaries
        TAGL[agl_lev] = tvals
        SPDAGL[agl_lev] = spdvals
        RHAGL[agl_lev] = rhvals
        #QAGL[agl_lev] = qvals
        #WAGL[agl_lev] = wvals

    return TAGL, SPDAGL, RHAGL
    #return TAGL, SPDAGL, RHAGL, QAGL, WAGL