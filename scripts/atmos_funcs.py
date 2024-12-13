"""
Michael Pletcher
Created: 12/13/2024
Edited: 

##### Summary #####
.py script containg function to calculate various
atmospheric variables

########### Function List ###########
    calc_wet_bulb_temp() - Calculate wet-bulb temperature based on 
    NWS adaptation (based on https://github.com/Unidata/MetPy/issues/409)
    calc_total_melting_energy() - 
    calc_final_layer_melting_energy() - 
    calc_layer_melting_energy() - 



"""

# Imports
import numpy as np

def calc_total_melting_energy(z_up, z_dn, tw_up, tw_dn):
    """
    Compute melting energy in an atmospheric layer. Adapted from code provided
    by Kevin Birk of the NWS. This code is modified such that it accounts for melting
    energy within a layer where the wet-bulb temperature crosses the melting point. Works
    for decreasing or increasing (i.e., inverted) wet-bulb profiles

    Parameters:
    z_up : 2d np.array or DataArray
        Geopotential height of the upper portion of the layer (meter)
    z_dn : 2d np.array or DataArray
        Geopotential height of the lower portion of the layer (meter)
    tw_up : 2d np.array or DataArray
        Wet-bulb temperature of the upper portion of the layer (Kelvin)
    tw_up : 2d np.array or DataArray
        Wet-bulb temperature of the lower portion of the layer (Kelvin)

    Returns:
    2d array of float
        Total melting energy across input 2-d grid (J/kg)
    """

    tmelt = 273.15 
    g = 9.81

    # Calculate inital layer energy
    layer_energy = np.where(
        (tw_up > tmelt) & (tw_dn > tmelt), 
        g * ((z_up - z_dn) * (((tw_up + tw_dn) / 2) - tmelt) / tmelt), 
        np.nan
    )
    # If the wet-bulb temperatures are less than freezing,
    # set the melting energy to zero
    layer_energy = np.where(
        (tw_up <= tmelt) & (tw_dn <= tmelt),
        0,
        layer_energy
    )
    # Find where the wet-bulb profile crosses zero
    # degrees (if any)
    zero_cross_height = np.where(
        (tw_up - tmelt) * (tw_dn - tmelt) < 0,
        z_dn + (tmelt - tw_dn) * (z_up - z_dn) / (tw_up - tw_dn),
        np.nan
    )
    # Adjust layer energy if there is a wet-bulb profile
    # crosses zero degrees
    layer_energy = np.where(
        ((tw_up - tmelt) * (tw_dn - tmelt) < 0) & (tw_dn > tmelt), 
        g * ((zero_cross_height - z_dn) * (((tmelt + tw_dn) / 2) - tmelt) / tmelt), 
        layer_energy
    )
    layerenergy = np.where(
        ((tw_up - tmelt) * (tw_dn - tmelt) < 0) & (tw_up > tmelt), 
        g * ((z_up - zero_cross_height) * (((tmelt + tw_up) / 2) - tmelt) / tmelt), 
        layerenergy
    )
    return layer_energy