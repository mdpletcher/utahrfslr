"""
Michael Pletcher
Created: 12/13/2024
Edited: 03/05/2025

##### Summary #####
.py script containg function to calculate variables
used for adjusting SLR based on several precipitation
type methods
"""

# Imports
import numpy as np
import xarray as xr

# Globals
BOUR_USE_TW2M = True
TMELT = 273.15
G = 9.81
ME_TOP = 2
ME_BOTTOM = 9
WBZ_PARAM = 0.5
MELT_DEPTH = 200



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
    layer_energy = np.where(
        ((tw_up - TMELT) * (tw_dn - TMELT) < 0) & (tw_up > TMELT), 
        G * ((z_up - zero_cross_height) * (((TMELT + tw_up) / 2) - TMELT) / TMELT), 
        layer_energy
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

    return layer_energy

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
        if mask:
            sfc_melting_energy = calc_layer_melting_energy_1d(
                z_prof[lev], 
                orog, 
                tw_prof[lev], 
                tw_2m
            )

        # If BOUR_USE_TW2M = False, set sfc_melting_energy to zero so it's not used
        # and instead calculate the surface wet-bulb temperature based on interpolating
        # between the first above and below ground pressure levels
        if not BOUR_USE_TW2M:
            sfc_melting_energy = 0
            if orog > z_prof[lev] and orog < z_prof[lev + 1]:
                tw_2m_interp = tw_prof[lev] + ((orog - z_prof[lev]) / (z_prof[lev + 1] - z_prof[lev])) * (tw_prof[lev + 1] - tw_prof[lev])
            else:
                tw_2m_interp = tw_prof[lev]

        # Calculate layer energy; this includes subterranean values *if* 
        # one or both layers is below ground
        layer_energy = calc_layer_melting_energy_1d(
            z_prof[lev + 1], 
            z_prof[lev], 
            tw_prof[lev + 1], 
            tw_prof[lev]
        )

        # If bouruse2mtw = False, overwrite layer energy based on layer energy calculated 
        # with interpolated surface wet-bulb temperature and surface elevatio for the 
        # bottom of the layer
        if not BOUR_USE_TW2M:
            if orog > z_prof[lev] and orog < z_prof[lev + 1]:
                layer_energy = calc_layer_melting_energy_1d(
                    z_prof[lev + 1], 
                    orog, 
                    tw_prof[lev + 1], 
                    tw_2m_interp
                )
   
        # Modify mask as profile is looped through
        above_top |= above_bottom
        if sfc_melting_energy < 0:
            sfc_melting_energy = 0

        # Add layer melting energys and total
        total_melting_energy += calc_final_melting_energy_1d(
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

def adjust_melt_slr(
    initslr, 
    method, 
    ndim,
    total_melting_energy = None,
    wbz = None, 
    orog = None
):
    """
    Adjust snow-to-liquid ratio based on two schemes
        1) 'BOUR': A hybrid approach based on Bourgouin (2000) and Birk et al. (2021). The global variables
        ME_TOP and ME_BOTTOM were chosen based on these studies, where ME_TOP = 2 J/kg is consistent 
        with the "melt energy aloft" used by Bourgouin (2000) and results in a deeper transition
        zone, and ME_BOTTOM = 9 J/kg for SLR = 0 is based on Bourgouin (2000) Fig. 3 and Eq. 5 
        and Birk et al. 2021 Fig. 8 and Eq. 9. Assuming a wet-bulb lapse rate of 6.5C/km, ME_BOTTOM = 9 J/kg
        is very close to 200-m below the 0.5C wet-bulb temperature.
        Bourgouin (2000): https://journals.ametsoc.org/view/journals/wefo/15/5/1520-0434_2000_015_0583_amtdpt_2_0_co_2.xml
        Birk et al. (2021): https://journals.ametsoc.org/view/journals/wefo/36/2/WAF-D-20-0118.1.xml

        2) 'WBZ': Adjusts SLR based on the height of the highest wet-bulb 0.5 degC temperature. Linearly
        interpolates SLR to zero from this height to 200 meters (MELT_DEPTH global) below. 
        See https://www.weather.gov/media/wrh/online_publications/TAs/TA1901.pdf for more info.

    Params:
    initslr : float or 2d np.array / xarray.DataArray
        Input initial snow-to-liquid ratio
    method : str
        SLR adjust method. Options are 'BOUR' or 'WBZ'.
    ndim : int
        Number of dimensions in the input data. Options are 1 or 2.
    total_melting_energy (optional) : float or 2d np.array / xarray.DataArray
        Total melting energy returned from calc_total_melting_energy() (J/kg).
        Only used when 'BOUR' is the method.
    wbz (optional) : float or 2d np.array / xarray.DataArray
        Input height of the highest wet-bulb 0.5 C level (meter AMSL)
    orog (optional) : float or 2d np.array / xarray.DataArray
        Station or model elevation (meter)

    Returns:
    float or 2d np.array
        Adjusted snow-to-liquid ratio based on either the Bourgouin or WBZ0.5 approaches.
    """
    if method == 'WBZ' and wbz is None and orog is None:
        raise ValueError("The 'wbz' and 'orog' arguments are required when method is set to 'WBZ'.")
    elif method == 'BOUR' and total_melting_energy is None:
        raise ValueError("The 'total_melting_energy' argument is required when method is set to 'BOUR'.")
    assert method in ['WBZ', 'BOUR'], "Invalid method: '%s'. Please use either 'WBZ' or 'BOUR" % method
    
    if ndim == 1:

        # Bourgouin (2000) / Birk et al. (2021) hybrid approach
        if method == 'BOUR':
            # Conditions used to adjust SLR based on total
            # melting energy thresholds
            cond1 = total_melting_energy >= ME_BOTTOM
            cond2 = ME_TOP <= total_melting_energy < ME_BOTTOM
            # Adjust SLR based on where the above conditons are met
            if cond1 or initslr < 0:
                slr = 0
            elif cond2:
                slr = initslr * (1 - ((total_melting_energy - ME_TOP) / (ME_BOTTOM - ME_TOP)))
            else:
                slr = initslr

        # Van Cleave (2019) approach
        elif method == 'WBZ':
            snowlevel = wbz
            if snowlevel < 0:
                snowlevel = 0
        
            if orog >= snowlevel:
                slr = initslr
            elif orog < snowlevel and orog > snowlevel - MELT_DEPTH:
                slr = initslr * (orog - (snowlevel - MELT_DEPTH)) / MELT_DEPTH
            
            if slr < 0:
                slr = 0

    elif ndim == 2:

        if method == 'BOUR':
            # Total melting energy adjustments
            cond1 = total_melting_energy >= ME_BOTTOM
            cond2 = (total_melting_energy >= ME_TOP) & (total_melting_energy < ME_BOTTOM)
            
            # Adjust SLR
            slr = xr.where(cond1, 0, initslr)
            slr = xr.where(cond2, initslr * (1 - ((total_melting_energy - MEtop) / (MEbottom-MEtop))), slr)
            slr = xr.where(slr < 0, 0, slr)

        elif method == 'WBZ':
            # Set snow level to height of maximum WB0.5 deg height
            snowlevel = wbz
            snowlevel = xr.where(snowlevel < 0, 0, snowlevel)

            # Adjust SLR
            slr = xr.where(orog >= snowlevel, initslr, 0)
            slr = xr.where(
                ((orog < snowlevel) & (orog > (snowlevel - MELT_DEPTH))),
                (initslr * (orog - (snowlevel - MELT_DEPTH)) / MELT_DEPTH),
                slr
            )
            slr = xr.where(slr < 0, 0, slr)

    return slr