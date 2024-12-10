"""
Michael Pletcher, Jim Steenburgh
Created: 10/07/2024
Edited: 12/10/2024

Acknowledgments: The author thanks Robert James of the Meteorological
Development Lab (MDL) for providing Fortran code for the Cobb, MaxTAloft,
and 850-700 mb Thickness methods, Jesse Meng of the Environmental Modeling
Center (EMC) for his help troubleshooting the Roebber method, and finally
the Unified Post Processor team for open-source code that helped convert 
the Roebber method from Fortran to Python.

##### Summary #####
.py script containing functions for each snow-to-liquid (SLR) method
used in the Cobb, MaxTAloft, 850 - 700mb Thickness,
and Roebber snow-to-liquid ratio methods. Each function 
calculates SLR at a specified point.

Each method is currently configured for NBM v4.2 
(see https://vlab.noaa.gov/web/mdl/nbm-versions#version-4-2 for more info on v4.2).
If access is available, The COMET Program also contains more info on 
each SLR method (see https://www.meted.ucar.edu/nwp/NBM40_snow/index.htm).

########### Function List ###########
    calc_layer_slr() - 
    calc_layer_vars() - 
    calc_cobb_slr() - 
    calc_maxtaloft_slr() - 
    calc_thickness_slr() - 
    calc_roebber_components() -
    mlp_1_hidden_layer() - 
    mlp_2_hidden_layers() - 
    calc_roebber_slr() - 
    adjust_nbm_slr() - 




"""

# Imports
import pandas as pd
import numpy as np
import nbm_config
import roebber_breadboards

from math import isnan
from glob import glob
from datetime import datetime



### Cobb Functions ###
def calc_layer_slr(temp):
    """
    Calculate layer snow ratio from layer temperature.

    Parameters: 
    temp : float
        Input temperature on atmospheric layer (K)

    Returns:
    float
        layer SLR based on input temperature

    """
    if temp < config.COBB_TTHRESH[0]:
        return config.C1_COBB[0]
    elif temp >= config.COBB_TTHRESH[-1]:
        return 0 if temp > 3 else np.nan
    for i in range(len(config.COBB_TTHRESH) - 1):
        if config.COBB_TTHRESH[i] <= temp < config.COBB_TTHRESH[i + 1]:
            tdiff = temp - config.COBB_TTHRESH[i]
            return config.C1_COBB[i] + config.C2_COBB[i] * tdiff + config.C3_COBB[i] * tdiff**2 + config.C4_COBB[i] * tdiff**3
        
# Calculate layer weighting factors for Cobb method
def calc_layer_vars(
        z, 
        temp, 
        rh, 
        w, 
        pres, 
        elev, 
        w_units = 'Pa/s'
):
    """
    Calculate layer variables for Cobb SLR function.

    Parameters:
    z : float
        Input geopotential height on atmospheric layer (meter)
    temp : float
        Input temperature on atmospheric layer (K)
    rh : float
        Input relative humidity on atmospheric layer (%)
    w : float
        Input vertical velocity on atmospheric layer (Pa/s or cm/s)
    pres : float
        Input pressure on atmospheric layer (mb or hPa) 
    elev : float
        Surface elevation (meter)
    w_units : str
        Vertical velocity units, Pa/s and cm/s accepted

    Returns:
    float
        Layer variables used to calculate SLR in calc_cobb_slr()
    """    

    # return None when layer geopotential height is below
    # surface elevation or layer pressure is less than 300 hPa
    # since Cobb only between 925 and 300 hPa 
    if z <= elev or pres < 300 or pres > 925:
        return None, None, None

    # Convert to cm/s from user-specified vertical
    # velocity units
    if w_units != None:
        # Pascals per sec conversion
        if w_units == 'Pa/s':
            flux = w / 9.8
            dens = pres * 100 / (287.058 * temp)
            w = -100 * (flux / dens)
    if w < 0:
        w = 1

    # Convert to Celsius, is 274.15 and not 273.15 because of 
    # NBM v4.2 calculation. 
    temp = temp - 274.15 

    # Calculate layer variables
    w = np.sqrt(w)
    layer_mean_slr = np.nanmean([calc_layer_slr(temp + k) for k in range(3)])
    w_cloudy = np.where(rh >= 80, w, w * ((rh ** 2) / 6400)) # Vertical velocity in 'cloudy' regions, where RH >= 80%

    return layer_mean_slr, w_cloudy, w ** 2

# Calculate Cobb SLR at single point and the average 
# vertical velocity and weighting factors in the profile
def calc_cobb_slr(
    gh_prof, 
    temp_prof, 
    rh_prof, 
    w_prof, 
    pres_prof,
    elev, 
    snowlev = None, 
    mlthick = None
):
    """
    Calculate Cobb SLR at single point and the average vertical velocity and 
    weighting factors in the profile using the methods from Cobb and Waldstreicher (2005) 
    (https://ams.confex.com/ams/WAFNWP34BC/techprogram/paper_94815.htm) Adjustments have 
    been made to the Cobb SLR method in the NBM, including NBM v4.0 and v4.2 
    (see https://vlab.noaa.gov/web/mdl/nbm-versions#version-4-2)

    Parameters:
    gh_prof : np.array
        Input geopotential height profile array (meter)
    temp_prof : np.array
        Input temperature profile array (K)
    rh_prof : np.array
        Input relative humidity profile array (%)
    w_prof : np.array
        Input vertical velocity profile array (Pa/s or cm/s)
    pres_prof : np.array
        Input pressure profile (mb or hPa)
    elev: float
        Surface elevation (meter)
    
    Returns:
    tuple of float
        Floats of Cobb SLR, average profile vertical velocity, and average weighting factor
    """
    layer_slr_sum = 0
    w_sum = 0
    weight_sum = 0

    # Convert to meters
    gh_prof = gh_prof / 10

    # Calculate layer variables
    for i in range(len(gh_prof)):
        layer_slr, w_cloudy, w_sq = calc_layer_vars(
            gh_prof[i], 
            temp_prof[i], 
            rh_prof[i], 
            w_prof[i], 
            pres_prof[i], 
            elev, 
            w_units = 'Pa/s'
        )
        if layer_slr is not None and not np.isnan(layer_slr) and not np.isnan(w_cloudy) and not np.isnan(w_sq):
            layer_slr_sum += layer_slr * w_cloudy
            w_sum += w_cloudy
            weight_sum += w_sq

    # Calculate SLR
    w_sum = w_sum or 0.0001
    slr = layer_slr_sum / w_sum
    slr = np.nan if slr > 50 or slr <= 0 else slr

    # Adjust SLR if desired
    if snowlev is not None:
        if elev < snowlev < elev + mlthick:
            slr *= (elev + mlthick - snowlev) / mlthick
        if snowlev > elev + mlthick or slr < 3 or slr > 50:
            slr = np.nan
    
    w_prof_avg = w_sum / len(gh_prof)
    w_sq_prof_avg = weight_sum / len(gh_prof)
    
    return slr, w_prof_avg, w_sq_prof_avg



### MaxTAloft Functions ###
# Calculate snow-to-liquid ratio by finding maximum temperature
# between 2000 ft AGL (609.6 m AGL) and 400 hPa. Converted to python
# from NBM v4.1 Fortran code. See more on MaxTAloft
# from the COMET Program's NBM v4.0 SLR Methods module 
# (https://www.meted.ucar.edu/nwp/NBM40_snow/navmenu.php?tab=1&page=2-3-0&type=flash).
def calc_maxtaloft_slr(
        gh_prof, 
        temp_prof, 
        pres_prof, 
        elev, 
):
    """
    Calculate snow-to-liquid ratio by finding maximum temperature between 2000 ft AGL (609.6 m AGL) 
    and 400 hPa. Converted to python from NBM v4.1 Fortran code. See more on MaxTAloft from the COMET 
    Program's NBM v4.0 SLR Methods module (https://www.meted.ucar.edu/nwp/NBM40_snow/navmenu.php?tab=1&page=2-3-0&type=flash).
    
    Parameters:
    gh_prof : np.array
        Input geopotential height profile array (meter)
    temp_prof : np.array
        Input temperature profile array (K)
    pres_prof : np.array
        Input pressure profile array (mb or hPa)
    elev : float
        Surface elevation (meter)
    """

    # Convert to arrays for processing
    gh_prof, pres_prof, temp_prof = np.array(gh_prof), np.array(pres_prof), np.array(temp_prof)

    # Ensure profile is above ground
    gh_prof = np.where(gh_prof > elev, gh_prof, np.nan)

    # Find maximum temperature aloft [between 2000 ft AGL 
    # (609.6 m AGL) and 400 hPa] and convert from Celsius to 
    # Kelvin. 2000 ft AGL can be either above a physical site 
    # elevation or above model topography, depending on use
    MaxT = np.nanmax(
        np.where(
            (gh_prof > elev + 609.6) & 
            (pres_prof >= 400), 
            temp_prof, 
            np.nan
        )
    ) - 273.15

    # Calculate SLR
    if MaxT < -40:
        slr = 8
    elif MaxT > 5:
        slr = np.nan
    else:
        slr = (
            config.COEFS_MAXT[0] * (MaxT**5) + 
            config.COEFS_MAXT[1] * (MaxT**4) + 
            config.COEFS_MAXT[2] * (MaxT**3) +
            config.COEFS_MAXT[3] * (MaxT**2) - 
            config.COEFS_MAXT[4] * MaxT + 
            config.COEFS_MAXT[5] # From NBM v4.1 Fortran code
        )
    
    # Remove spurious SLRs
    if slr < 2 or slr > 50:
        slr = np.nan
    
    return slr, MaxT



### Thickness Functions ###
def calc_thickness_slr(
    gh_prof,
    pres_prof,
    elev,
):
    '''
    :param: gh_prof: atmospheric geopotential height profile, meter
    :param pres_prof: atmospheric geopotential height profile, meter
    :param elev: surface elevation, meter

    '''
    
    # Ensure profile is above ground
    gh_prof = np.where(gh_prof > elev, gh_prof, np.nan)
    
    # Find geopotential heights at 850 and 700 hPa
    # and the 850 - 700 hPa thickness
    z_850, z_700 = gh_prof[np.array(pres_prof) == 850], gh_prof[np.array(pres_prof) == 700]
    thck_850_700 = z_700 - z_850

    # Calculate SLR using equation from NBM v4.2 Fortran code
    slr = np.nan if thck_850_700 == np.nan else -(0.16559 * thck_850_700) + 263.35
    slr = np.nan if slr < 0 else slr
        
    return slr



### Roebber Functions ###
# This code was converted from Fortran
# to python using code from NOAA EMC's
# Unifed Post-Processor Github page
# (see https://github.com/NOAA-EMC/UPP/blob/develop/sorc/ncep_post.fd/UPP_PHYSICS.f
# for more info).
# Compute each component 
def calc_roebber_components(
        coefs, 
        qpf, 
        spd10m, 
        temp, 
        rh, 
        i
):
    return (
        coefs[0] + 
        coefs[1] * qpf[i] + 
        coefs[2] * spd10m[i] + 
        sum(coefs[j + 3] * temp[j] for j in range(14)) + 
        sum(coefs[j + 17] * rh[j] for j in range(13))
    )

# Artificial neural network (ANN) that computes probabilities for each
# snow class (light, moderate, heavy) for the first 5 members in 
# the 10-member ANN using a multilayer perceptron framework with a
# single hidden layer.
def mlp_1_hidden_layer(
    inputFile, 
    hidden1Axon, 
    hidden1Synapse, 
    outputSynapse, 
    month_index, 
    f1, 
    f2, 
    f3, 
    f4, 
    f5, 
    f6
):

    # Define empty arrays for network
    inputAxon = np.zeros(7)
    fgrid1 = np.zeros(40)
    fgrid2 = np.zeros(3)
    outputAxon = np.zeros(3)
    f = [month_index, f1, f2, f3, f4, f5, f6]

    # Calculate input axons based on calculated factors (f[k])
    for j in range(7):
        inputAxon[j] = inputFile[0, j] * f[j] + inputFile[1, j]

    # Calculate operation results of layers 1 and 2
    for j in range(40):
        for i in range(7):
            fgrid1[j] += hidden1Synapse[i, j] * inputAxon[i]
        fgrid1[j] += hidden1Axon[j]
        fgrid1[j] = (np.exp(fgrid1[j]) - np.exp(-fgrid1[j])) / (np.exp(fgrid1[j]) + np.exp(-fgrid1[j]))
    
    fgridsum = 0
    for j in range(3):
        for i in range(40):
            fgrid2[j] += outputSynapse[i, j] * fgrid1[i]
        fgrid2[j] += outputAxon[j]
        fgrid2[j] = np.exp(fgrid2[j])
        fgridsum += fgrid2[j]

    for j in range(3):
        fgrid2[j] /= fgridsum

    # Calculate probabilities 
    p1, p2, p3 = fgrid2[0], fgrid2[1], fgrid2[2]

    return p1, p2, p3

# Artificial neural network (ANN) that computes probabilities for each
# snow class (light, moderate, heavy) for the first 5 members in 
# the 10-member ANN using a multilayer perceptron framework with two
# hidden layers.
def mlp_2_hidden_layers(
    inputFile, 
    hidden1Axon, 
    hidden2Axon, 
    hidden1Synapse, 
    hidden2Synapse, 
    outputSynapse, 
    month_index, 
    f1, 
    f2, 
    f3, 
    f4, 
    f5, 
    f6
):

    # Define empty arrays for network
    inputAxon = np.zeros(7)
    fgrid1 = np.zeros(7)
    fgrid2 = np.zeros(4)
    fgrid3 = np.zeros(3)
    outputAxon = np.zeros(3)
    f = [month_index, f1, f2, f3, f4, f5, f6]

    # --- Start of neural network --- 
    # Calculate input axons based on calculated factors (f[k])
    for k in range(7):
        inputAxon[k] = inputFile[0, k] * f[k] + inputFile[1, k]

    # Calculate operation results of layers 1 and 2
    for k in range(7):
        for l in range(7):
            fgrid1[k] += hidden1Synapse[l, k]*inputAxon[l]
        fgrid1[k] += hidden1Axon[k]
        fgrid1[k] = (np.exp(fgrid1[k]) - np.exp(-fgrid1[k])) / (np.exp(fgrid1[k]) + np.exp(-fgrid1[k]))

    # Calculate operation results of layers 2 and 3
    for k in range(4):
        for l in range(7):
            fgrid2[k] += hidden2Synapse[l, k] * fgrid1[l]
        fgrid2[k] += hidden2Axon[k]
        fgrid2[k] = (np.exp(fgrid2[k]) - np.exp(-fgrid2[k])) / (np.exp(fgrid2[k]) + np.exp(-fgrid2[k]))

    fgridsum = 0
    for k in range(3):
        for l in range(4):
            fgrid3[k] += outputSynapse[l, k] * fgrid2[l]
        fgrid3[k] += outputAxon[k]
        fgrid3[k] = np.exp(fgrid3[k])
        fgridsum += fgrid3[k]

    for k in range(3):
        fgrid3[k] /= fgridsum

    # Calculate probabilities 
    p1, p2, p3 = fgrid3[0], fgrid3[1], fgrid3[2]

    return p1, p2, p3

# Create global from two the two ANNs
ANN_FUNCS = [mlp_1_hidden_layer] * 5 + [mlp_2_hidden_layers] * 5
BREADBOARDS = [
    roebber_breadboards.Breadboard1,
    roebber_breadboards.Breadboard2,
    roebber_breadboards.Breadboard3,
    roebber_breadboards.Breadboard4,
    roebber_breadboards.Breadboard5,
    roebber_breadboards.Breadboard6,
    roebber_breadboards.Breadboard7,
    roebber_breadboards.Breadboard8,
    roebber_breadboards.Breadboard9,
    roebber_breadboards.Breadboard10
]

def calc_roebber_slr(
    pres_prof, 
    temp_prof, 
    rh_prof,
    gh_prof,
    mslp,
    psfc,
    qpf,
    spd10m,
    t2m,
    r2m,
    ob_collect_time,
    elev,
):
    """
    Calculate snow-to-liquid ratio at a grid point using the Roebber et al. (2003)
    methodology. The original code was written in C++ but was converted to Fortran by NOAA EMC's
    UPP group and adapted to calculate an explicit SLR which the original Roebber et al. (2003)
    method did not do. Thus, users should be cautious when interpreting results from this algorithm.

    Parameters:
        pres_prof : np.array
            Input pressure profile (Pa)
        temp_prof : np.array
            Input temperature profile (K)
        rh_prof : np.array
            Input relative humidity profile (%)
        gh_prof : np.array
            Input geopotential height profile (meter)
        mslp : float, np.array
            Input mean sea level pressure (Pa)
        psfc : float, np.array
            Input surface pressure (Pa)
        qpf : float, np.array
            Input QPF (inch)
        spd10m : float, np.array
            Input 10-meter wind speed (m/s)
        t2m : float, np.array
            Input 2-meter temperature (K)
        r2m : float, np.array
            Input 2-meter relative humidity (%)
        ob_collect_time : datetime
            Observation collection time, formatted YYYY-MM-DD MM:SS:00
        elev : float, np.array
            Input elevation (meter). This can be either a real elevation
            value or a model grid point elevation
    """

    

    # Calculate sigma levels and set first
    # level to 1
    pres_prof = [lev * 100 for lev in pres_prof]
    sigma_levels = pres_prof / psfc
    sigma_levels[0] = 1
    
    # Empty arrays for temperature and humidity data interpolated to 
    # 14 sigma levels
    rhms = np.zeros(14)
    tms = np.zeros(14)

    # Pressure pertubation calculation
    prp = mslp / psfc
    prp = prp * 100000 / psfc

    # Use profiles above ground
    #temp_agl = np.where(pres_prof <= )

    # SLRS
    slrs = []

    # Empty arrays for temperature and humidity data interpolated to 
    # 14 sigma levels from Roebber et al. (2003)
    rhms = np.zeros(14)
    tms  = np.zeros(14)

    # Pressure pertubation calculation
    prp = mslp / psfc  # 
    prp = prp * 100000 / psfc #  
    p = prp

    # Use levels above ground
    t_agl = np.where(gh_prof > elev, temp_prof, np.nan)
    rh_agl = np.where(gh_prof > elev, rh_prof, np.nan)
    sigma_agl = np.where(gh_prof > elev, sigma_levels, np.nan)
    display(sigma_agl)

    below_mask = ~np.isnan(t_agl)

    t_agl     = t_agl[below_mask]
    rh_agl    = rh_agl[below_mask]
    sigma_agl = sigma_agl[below_mask]

    t_agl[0]     = t2m
    rh_agl[0]    = r2m
    sigma_agl[0] = 1
    
    # Loop through each sigma level        
    for ks in range(len(SIGMA_LEVS)):

        # Surface pressure
        # psfc is surface pressure
        # pres, psurf are surface mean sea level pressure

        # Sigma level in loop
        sgw = SIGMA_LEVS[ks]

        # Loop through each level in each HRRR vertical profile above ground
        for j in range(t_agl.shape[0] - 1):
            if j == 0:
                sg1 = 1
            else:
                sg1 = sigma_agl[j]
            # Above sigma level
            sg2 = sigma_agl[j + 1]
            if sg1 == sgw:
                tms[ks]  = t_agl[j] # current sigma level temperature
                rhms[ks] = rh_agl[j] # current sigma level RH
            elif sg2 == sgw:
                tms[ks]  = t_agl[j + 1] # Above sigma level temperature
                rhms[ks] = rh_agl[j + 1] # Above sigma level RH
            elif sgw < sg1 and sgw > sg2:
                # Difference in temperature between layers in profile loop and
                # temperature at each sigma level (14 sigma levels)
                dtds = (t_agl[j + 1] - t_agl[j]) / (sg2 - sg1)
                tms[ks] = ((sgw - sg1) * dtds) + t_agl[j]
                # Finds difference in RH between layers in profile loop and
                # RH at each sigma level (14 sigma levels)
                rhds = (rh_agl[j + 1] - rh_agl[j]) / (sg2 - sg1)
                rhms[ks] = ((sgw - sg1) * rhds) + rh_agl[j]

    # Set first sigma level value to 2-meter temperature/RH
    tms[0]  = t2m
    rhms[0] = r2m

    f = []
    # Compute all 6 components
    for c in ROEBBER_COEFS:
        f.append(calc_roebber_components(c, qpf, spd10m, tms, rhms))
    
    ob_collect_time = datetime.datetime.strptime(str(ob_collect_time), '%Y-%m-%d %H:%M:%S+00:00')
    if ob_collect_time.strftime("%m") == '01':
        month_idx = MONTH_IDXS[0]
    elif ob_collect_time.strftime("%m") == '03':
        month_idx = MONTH_IDXS[10]
    elif ob_collect_time.strftime("%m") == '04':
        month_idx = MONTH_IDXS[9]
    else:
        month_idx = MONTH_IDXS[11]

    hprob_tot = 0
    mprob_tot = 0
    lprob_tot = 0

    # Loop through each neural network 
    for k in range(10):
        breadboard = BREADBOARDS[k]()
        if k < 5:
            probs = ANN_FUNCS[k](
                breadboard[0], breadboard[1], breadboard[2], breadboard[3],
                month_idx, f[0], f[1], f[2], f[3], f[4], f[5]
            )
        else:
            probs = ANN_FUNCS[k](
        breadboard[0], breadboard[1], breadboard[2], breadboard[3],
        breadboard[4], breadboard[5], month_idx, f[0], f[1], f[2], f[3], f[4], f[5]
            )

        # Total probabilities for each snowfall class (heavy, medium, light)
        hprob_tot += probs[0]
        mprob_tot += probs[1]
        lprob_tot += probs[2]

    # Convert probabilites to fractions
    hprob = hprob_tot / 10
    mprob = mprob_tot / 10
    lprob = lprob_tot / 10

    # Multiplication factor for light snow class
    lprob_factor = 18 if lprob < 0.67 else 27

    # Convert probabilities to explicit SLR
    slr = (hprob * 8 + mprob * 13 + lprob * lprob_factor) * p / (hprob + mprob + lprob)

    return slr



### Global NBM Functions ###
def adjust_nbm_slr(slr, tw2m, qpf):
    """
    Adjust snow-to-liquid ratio based on melting scheme developed by 
    Daniel Cobb for NBM v4.2 (see https://vlab.noaa.gov/documents/6609493/7858320/Cobb+Melting+SLR+Method.pdf
    for more info). 
    
    Params:
    slr : float
        Input snow-to-liquid ratio 
    tw2m : float
        2-meter wet-bulb temperature, Celsius
    qpf : float
        1-h quantitative precipitation forecast (QPF), inches
    
    Returns:
    float
        Adjusted snow-to-liquid ratio based on wet-bulb temperature and QPF
    """
    
    if 31 < round(tw2m) < 33:
        slr = slr * ((qpf - 0.01) / qpf)
    elif 32 < round(tw2m) < 34:
        slr = slr * ((qpf - 0.03) / qpf)
    elif 33 < round(tw2m) < 35:
        slr = slr * ((qpf - 0.05) / qpf)
    elif 34 < round(tw2m) < 36:
        slr = slr * ((qpf - 0.07) / qpf)
    elif 35 < round(tw2m) < 37:
        slr = slr * ((qpf - 0.09) / qpf)
    elif 36 < round(tw2m) < 38:
        slr = slr * ((qpf - 0.12) / qpf)
    elif 37 < round(tw2m) < 39:
        slr = slr * ((qpf - 0.15) / qpf)
    elif 38 < round(tw2m) < 40:
        slr = slr * ((qpf - 0.18) / qpf)
    elif round(tw2m) >= 40:
        slr = slr * ((qpf - 0.21) / qpf)
        
    slr = 0 if slr < 0
        
    return slr