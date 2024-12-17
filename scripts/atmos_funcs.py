"""
Michael Pletcher
Created: 12/13/2024
Edited: 12/17/2024

##### Summary #####
.py script containg function to calculate various
atmospheric variables

########### Function List ###########
    calc_wet_bulb_temp() - Calculate wet-bulb temperature based on 
    NWS adaptation (based on https://github.com/Unidata/MetPy/issues/409)
    calc_total_melting_energy() - 
    calc_final_layer_melting_energy() - 
    calc_layer_melting_energy() - 
    calc_total_melting_energy_1d() - 
    calc_final_layer_melting_energy_1d() - 
    calc_layer_melting_energy_1d() - 



"""

# Imports
import numpy as np

# Globals
BOUR_USE_TW2M = True
TMELT = 273.15
G = 9.81



### Melting energy functions ###
def calc_layer_melting_energy(z_up, z_dn, tw_up, tw_dn):
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

    # Calculate inital layer energy
    layer_energy = np.where(
        (tw_up > TMELT) & (tw_dn > TMELT), 
        G * ((z_up - z_dn) * (((tw_up + tw_dn) / 2) - TMELT) / TMELT), 
        np.nan
    )
    # If the wet-bulb temperatures are less than freezing,
    # set the melting energy to zero
    layer_energy = np.where(
        (tw_up <= TMELT) & (tw_dn <= TMELT),
        0,
        layer_energy
    )
    # Find where the wet-bulb profile crosses zero
    # degrees (if any)
    zero_cross_height = np.where(
        (tw_up - TMELT) * (tw_dn - TMELT) < 0,
        z_dn + (TMELT - tw_dn) * (z_up - z_dn) / (tw_up - tw_dn),
        np.nan
    )
    # Adjust layer energy if there is a wet-bulb profile
    # crosses zero degrees
    layer_energy = np.where(
        ((tw_up - TMELT) * (tw_dn - TMELT) < 0) & (tw_dn > TMELT), 
        G * ((zero_cross_height - z_dn) * (((TMELT + tw_dn) / 2) - TMELT) / TMELT), 
        layer_energy
    )
    layerenergy = np.where(
        ((tw_up - TMELT) * (tw_dn - TMELT) < 0) & (tw_up > TMELT), 
        G * ((z_up - zero_cross_height) * (((TMELT + tw_up) / 2) - TMELT) / TMELT), 
        layerenergy
    )
    return layer_energy

def calc_final_melting_energy(
    layer_energy, 
    z_up,
    z_dn, 
    orog, 
    BOUR_USE_TW2M
):
    """
    Sets layer energy to zero if the layer is below ground or accounts for 
    terrain if the layer contains surface elevation. Adapted from code given
    by Kevin Birk. Designed to use 3-d arrays of profiles and 2-d arrays of 
    layer energy / model topography as input.

    Parameters:
    layer_energy : 2d np.array or DataArray
        Energy within an atmospheric layer (J/kg)
     z_up : 2d np.array or DataArray
        Geopotential height of the upper portion of the layer (meter)
    z_dn : 2d np.array or DataArray
        Geopotential height of the lower portion of the layer (meter)
    orog : 2d np.array or DataArray
        Model elevation (meter)
    BOUR_USE_TW2M : boolean
        True or False

    Returns:
    2d array of float
        Adjusted melting energy in an atmospheric layer
    """
    #Only use layer if it is above the first pressure level above ground.
    # This is done by setting layer energy to 0 if the lowest level of 
    # the layer is below ground
    if BOUR_USE_TW2M:
        layer_energy = np.where(z_dn <= orog, 0, layer_energy)

    # Otherwise, use layers that include the topography.
    # In this case, we have calculated the layer energy 
    # using an interpolated surface value and the tw from 
    # the first pressure level above ground
    else:
        layer_energy = np.where(
            ((orog > z_dn) * (orog <= z_up)) | (z_up > orog),
            layer_energy,
            0
        )
    layer_energy[layer_energy < 0] = 0
    return layer_energy

def calc_total_melting_energy(orog, z_prof_grid, tw_2m, tw_prof_grid):
    """
    Calculate total melting energy in an atmospheric column. Adapted
    from code given by Kevin Birk. Designed to use 3-d arrays of profiles 
    and 2-d arrays of layer energy / model topography as input.

    Parameters:
    orog : 2d np.array or DataArray
        Model elevation (meter)
    z_prof_grid : 3d np.array or DataArray
        Geopotential height profiles in grid (meter)
    tw_2m : 2d np.array or DataArray
        2-meter wet-bulb temperature (Kelvin)
    tw_prof_grid : 3d np.array or DataArray
        Wet-bulb temperature profiles in grid (Kelvin)

    Returns:
    2d array of float
        Total melting energy in atmospheric profile
    """
    total_melting_energy, above_top, sfc_melting_energy = (np.zeros_like(orog) for array in range(3))

    # Create mask (sets everything to false since twsurf is never > 1000)
    above_top = (tw_2m > 1000)
    
    for lev in range(z_prof_grid.shape[0] - 1):

        # Create mask of values below ground
        was_below_ground = np.logical_not(above_top)
        above_bottom = (z_prof_grid[lev, :, :] > orog) & (was_below_ground)
        mask = (above_bottom)

        # Calculate surface energy (only used if BOUR_USE_TW2M = True)
        sfc_melting_energy = np.where(
            mask,
            calc_layer_melting_energy(z_prof_grid[lev, :, :], orog, tw_prof_grid[lev, :, :], tw_2m),
            sfc_melting_energy
        )

        # If BOUR_USE_TW2M = False, set sfc_melting_energy to zero so it's not used
        # and instead calculate the surface wet-bulb temperature based on interpolating
        # between the first above and below ground pressure levels
        if not BOUR_USE_TW2M:
            sfc_melting_energy[:, :] = 0
            tw_2m_interp = np.where(
                (orog > z_prof_grid[lev, :, :]) & (orog < z_prof_grid[lev + 1, :, :]),
                tw_prof_grid[lev, :, :] + ((orog - z_prof_grid[lev, :, :]) / (z_prof_grid[lev + 1, :, :] \
                - z_prof_grid[lev, :, :])) * (tw_prof_grid[lev + 1, :, :] - tw_prof_grid[lev, :, :]), # If True, linearly interpolate
                tw_prof_grid[lev, :, :]
            ) 

        # Calculate layer energy; this includes subterranean values *if* 
        # one or both layers is below ground
        layer_energy = calc_layer_melting_energy(
            z_prof_grid[lev + 1, :, :], 
            z_prof_grid[lev, :, :], 
            tw_prof_grid[lev + 1, :, :], 
            tw_prof_grid[lev, :, :]
        )

        # If bouruse2mtw = False, overwrite layer energy based on layer energy calculated 
        # with interpolated surface wet-bulb temperature and surface elevatio for the 
        # bottom of the layer
        if not BOUR_USE_TW2M:
            layer_energy = np.where(
                (orog > z_prof_grid[lev, :, :]) & (orog < z_prof_grid[lev + 1, :, :]),
                calc_layer_melting_energy(z_prof_grid[lev + 1, :, :], orog, tw_prof_grid[lev + 1, :, :], tw_2m_interp),
                layer_energy
            )

        # Modify mask as profile is looped through
        above_top |= above_bottom
        sfc_melting_energy[sfc_melting_energy < 0] = 0

        # Add layer melting energys and total
        total_melting_energy += calc_final_melting_energy(
            layer_energy,
            z_prof_grid[lev, :, :],
            z_prof_grid[lev + 1, :, :],
            orog,
            BOUR_USE_TW2M
        )
    
    # Adjust total melting energy. This will add zero if BOUR_USE_TW2M = False
    # since sfc_melting_energy is set to zero
    total_melting_energy += sfc_melting_energy
    total_melting_energy[total_melting_energy > 500] = 500
    total_melting_energy[total_melting_energy < 0] = 0
    return total_melting_energy

def calc_layer_melting_energy_1d(z_up, z_dn, tw_up, tw_dn):
    """
    Compute melting energy in an atmospheric layer. Adapted from code provided
    by Kevin Birk of the NWS. This code is modified such that it accounts for melting
    energy within a layer where the wet-bulb temperature crosses the melting point. Works
    for decreasing or increasing (i.e., inverted) wet-bulb profiles

    Parameters:
    z_up : float
        Geopotential height of the upper portion of the layer (meter)
    z_dn : float
        Geopotential height of the lower portion of the layer (meter)
    tw_up : float
        Wet-bulb temperature of the upper portion of the layer (Kelvin)
    tw_up : float
        Wet-bulb temperature of the lower portion of the layer (Kelvin)

    Returns:
    float
        Total melting energy (J/kg)
    """
    layer_energy = 0
    # Calculate inital layer energy
    if tw_up > TMELT and tw_dn > TMELT:
        layer_energy = G * ((z_up - z_dn) * (((tw_up + tw_dn) / 2) - TMELT) / TMELT)
    # If the wet-bulb temperatures are less than freezing,
    # set the melting energy to zero
    elif tw_up <= TMELT and tw_dn <= TMELT:
        layer_energy = 0
    # Find where the wet-bulb profile crosses zero
    # degrees (if any)
    elif (tw_up - TMELT) * (tw_dn - TMELT) < 0:
        zero_cross_height = z_dn + (TMELT - tw_dn) * (z_up - z_dn) / (tw_up - tw_dn)
        if tw_dn > TMELT:  # Crossing from above freezing to below
            layer_energy = G * ((zero_cross_height - z_dn) * (((TMELT + tw_dn) / 2) - TMELT) / TMELT)
        elif tw_up > TMELT:  # Crossing from below freezing to above
            layer_energy = G * ((z_up - zero_cross_height) * (((TMELT + tw_up) / 2) - TMELT) / TMELT)

    return max(layer_energy, 0)

def calc_final_melting_energy_1d(
    layer_energy, 
    z_up,
    z_dn, 
    orog, 
    BOUR_USE_TW2M
):
    """
    Sets layer energy to zero if the layer is below ground or accounts for 
    terrain if the layer contains surface elevation. Adapted from code given
    by Kevin Birk. 

    Parameters:
    layer_energy : float
        Energy within an atmospheric layer (J/kg)
     z_up : float
        Geopotential height of the upper portion of the layer (meter)
    z_dn : float
        Geopotential height of the lower portion of the layer (meter)
    orog : float
        Model elevation (meter)
    BOUR_USE_TW2M : boolean
        True or False

    Returns:
    float
        Adjusted melting energy in an atmospheric layer
    """
    #Only use layer if it is above the first pressure level above ground.
    # This is done by setting layer energy to 0 if the lowest level of 
    # the layer is below ground
    if BOUR_USE_TW2M:
        layer_energy = 0 if z_dn <= orog else layer_energy

    # Otherwise, use layers that include the topography.
    # In this case, we have calculated the layer energy 
    # using an interpolated surface value and the tw from 
    # the first pressure level above ground
    else:
        if (orog > z_dn and orog <= z_up) or (z_up > orog):
            layer_energy = max(layer_energy, 0)
        else:
            layer_energy = 0
    return layer_energy

def calc_total_melting_energy_1d(orog, z_prof, tw_2m, tw_prof):
    """
    Calculate total melting energy in an atmospheric column. Adapted
    from code given by Kevin Birk.

    Parameters:
    orog : float
        Model elevation (meter)
    z_prof : 1d np.array
        Geopotential height profile (meter)
    tw_2m : float
        2-meter wet-bulb temperature (Kelvin)
    tw_prof : 1d np.array
        Wet-bulb temperature profile (Kelvin)

    Returns:
    float
        Total melting energy in atmospheric profile (J/kg)
    """
    total_melting_energy, sfc_melting_energy = 0, 0

    # Create mask (sets everything to false since twsurf is never > 1000)
    above_top = False
    
    for lev in range(len(z_prof) - 1):

        # Create mask of values below ground
        was_below_ground = not above_top
        above_bottom = (z_prof[lev] > orog) and was_below_ground
        mask = above_bottom

        # Calculate surface energy (only used if BOUR_USE_TW2M = True)
        sfc_melting_energy = np.where(
            mask,
            calc_layer_melting_energy(z_prof[lev], orog, tw_prof[lev], tw_2m),
            sfc_melting_energy
        )

        # If BOUR_USE_TW2M = False, set sfc_melting_energy to zero so it's not used
        # and instead calculate the surface wet-bulb temperature based on interpolating
        # between the first above and below ground pressure levels
        if not BOUR_USE_TW2M:
            sfc_melting_energy = 0
            tw_2m_interp = np.where(
                (orog > z_prof[lev]) & (orog < z_prof[lev + 1]),
                tw_prof[lev] + ((orog - z_prof[lev]) / (z_prof[lev + 1] \
                - z_prof[lev])) * (tw_prof[lev + 1] - tw_prof[lev]), # If True, linearly interpolate
                tw_prof[lev]
            ) 

        # Calculate layer energy; this includes subterranean values *if* 
        # one or both layers is below ground
        layer_energy = calc_layer_melting_energy(
            z_prof[lev + 1], 
            z_prof[lev], 
            tw_prof[lev + 1], 
            tw_prof[lev]
        )

        # If bouruse2mtw = False, overwrite layer energy based on layer energy calculated 
        # with interpolated surface wet-bulb temperature and surface elevatio for the 
        # bottom of the layer
        if not BOUR_USE_TW2M:
            layer_energy = np.where(
                (orog > z_prof[lev]) & (orog < z_prof[lev + 1]),
                calc_layer_melting_energy(z_prof[lev + 1], orog, tw_prof[lev + 1], tw_2m_interp),
                layer_energy
            )

        # Modify mask as profile is looped through
        above_top |= above_bottom
        if sfc_melting_energy < 0:
            sfc_melting_energy = 0

        # Add layer melting energys and total
        total_melting_energy += calc_final_melting_energy(
            layer_energy,
            z_prof[lev],
            z_prof[lev + 1],
            orog,
            BOUR_USE_TW2M
        )
    
    # Adjust total melting energy. This will add zero if BOUR_USE_TW2M = False
    # since sfc_melting_energy is set to zero
    total_melting_energy += sfc_melting_energy
    if total_melting_energy > 500:
        total_melting_energy = 500
    elif total_melting_energy < 0:
        total_melting_energy = 0
    return total_melting_energy


