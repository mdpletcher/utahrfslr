"""
Michael Pletcher, Jim Steenburgh
Created: 10/07/2024
Edited: 06/12/2024

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



"""

# Imports
import pandas as pd
import numpy as np
import sys

from math import isnan
from glob import glob
from datetime import datetime

sys.path.append('../configs/')
import nbm_config, roebber_ens_members

# Create global for weights and biases from each
# member of the 10-member ANN ensemble
MEMBERS = [
    roebber_ens_members.Member1,
    roebber_ens_members.Member2,
    roebber_ens_members.Member3,
    roebber_ens_members.Member4,
    roebber_ens_members.Member5,
    roebber_ens_members.Member6,
    roebber_ens_members.Member7,
    roebber_ens_members.Member8,
    roebber_ens_members.Member9,
    roebber_ens_members.Member10
]



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
    if temp < nbm_config.COBB_TTHRESH[0]:
        return nbm_config.C1_COBB[0]
    elif temp >= nbm_config.COBB_TTHRESH[-1]:
        return 0 if temp > 3 else np.nan
    for i in range(len(nbm_config.COBB_TTHRESH) - 1):
        if nbm_config.COBB_TTHRESH[i] <= temp < nbm_config.COBB_TTHRESH[i + 1]:
            tdiff = temp - nbm_config.COBB_TTHRESH[i]
            return nbm_config.C1_COBB[i] + nbm_config.C2_COBB[i] * tdiff + nbm_config.C3_COBB[i] * tdiff**2 + nbm_config.C4_COBB[i] * tdiff**3
        
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

    Returns:
    tuple of float
        Floats of MaxTAloft SLR, maximum temperature between 2000 ft AGL and 400 hPa  
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
            nbm_config.COEFS_MAXT[0] * (MaxT**5) + 
            nbm_config.COEFS_MAXT[1] * (MaxT**4) + 
            nbm_config.COEFS_MAXT[2] * (MaxT**3) +
            nbm_config.COEFS_MAXT[3] * (MaxT**2) - 
            nbm_config.COEFS_MAXT[4] * MaxT + 
            nbm_config.COEFS_MAXT[5] # From NBM v4.1 Fortran code
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
    """
    Calculate snow-to-liquid ratio by finding the 850 - 700 mb thickness See more on the
    850-700 mb Thickness method from the COMET Program's NBM v4.0 SLR Methods module 
    (https://www.meted.ucar.edu/nwp/NBM40_snow/navmenu.php?tab=1&page=2-3-0&type=flash).
    
    Parameters:
    gh_prof : np.array
        Input geopotential height profile array (meter)
    pres_prof : np.array
        Input pressure profile array (mb or hPa)
    elev : float
        Surface elevation (meter)

    Returns:
    float
        850-700 mb Thickness SLR
    """
    # Ensure profile is above ground
    gh_prof = np.where(gh_prof > elev, gh_prof, np.nan)
    
    # Find geopotential heights at 850 and 700 hPa
    # and the 850 - 700 hPa thickness
    z_850, z_700 = gh_prof[np.array(pres_prof) == 850], gh_prof[np.array(pres_prof) == 700]
    thck_850_700 = z_700 - z_850

    # Calculate SLR using equation from NBM v4.2 Fortran code
    slr = np.nan if thck_850_700 == np.nan else -(nbm_config.THICK_COEFS[0] * thck_850_700) + nbm_config.THICK_COEFS[1]
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
    temp_sigma, 
    rh_sigma, 
):
    """
    Computes each of the six principal components used in calc_roebber_slr()

    
    Parameters:
    coefs : list
        List of 29 Roebber ANN coefficients
    qpf : float
        Input model QPF (mm)
    spd10m : np.array
        10-meter wind speed (m/s)
    temp_sigma : np.array
        Temperature profile interpolated to the 14 sigma levels (Celsius)
    rh_sigma
        Relative humidity profile interpolated to the 14 sigma levels (%)

    Returns:
    float
        Computed principal component value
    """
    # Base components from coefficients
    base_component = (coefs[0] + coefs[1] * qpf[i] + coefs[2] * spd10m[i])
    temp_contributions = sum(coefs[j + 3] * temp[j] for j in range(14)) # Temperature contributions (14 coefficients)
    rh_contributions = sum(coefs[j + 17] * rh[j] for j in range(13))  # Relative humidity contributions (13 coefficients)
    principal_component = base_component + temp_contributions + rh_contributions

    return principal_component

# Artificial neural network (ANN) that computes probabilities for each
# snow class (light, moderate, heavy) for the first 5 members in 
# the 10-member ANN using a multilayer perceptron framework with a
# single hidden layer.
def mlp_1_hidden_layer(
    inputFile, 
    hidden1Axon, 
    hidden1Synapse, 
    outputSynapse, 
    month_idx, 
    f1, 
    f2, 
    f3, 
    f4, 
    f5, 
    f6
):
    """
    Computes probabilities for each snow class using the weights 
    and biases from the first 5 ensemble members in the 10-member
    ensemble as well as the required inputs.

    Parameters:
    inputFile: : 2d np.array
        Weights and biases for the 7 inputs
    hidden1Axon : 1d np.array
        Biases for each of the 40 neurons in the hidden layer
    hidden1Synapse : 2d np.array
        Weights connecting 7 inputs to the 40 neurons in the first
        hidden layer
    outputSynapse : 2d np.array
        Weights connecting the 40 neurons in the hidden layer
        to the three snowfall classes (light, moderate, heavy)
    month_idx : float
        Weighting value based on month
    f1, f2, f3, f4, f5, f6 : float
        6 input principal components calculated in calc_roebber_components()

    Returns:
    tuple of float
        Floats of each snow class probability
    """
    # Define empty arrays for network
    inputAxon = np.zeros(7)
    fgrid1 = np.zeros(40)
    fgrid2 = np.zeros(3)
    outputAxon = np.zeros(3)
    f = [month_idx, f1, f2, f3, f4, f5, f6]

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
    month_idx, 
    f1, 
    f2, 
    f3, 
    f4, 
    f5, 
    f6
):
 """
    Computes probabilities for each snow class using the weights 
    and biases from the first 5 ensemble members in the 10-member
    ensemble as well as the required inputs.

    Parameters:
    inputFile: : 2d np.array
        Weights and biases for the 7 inputs
    hidden1Axon : 1d np.array
        Biases for each of the 7 neurons in the first hidden layer
    hidden2Axon : 1d np.array
        Biases for each of the 4 neurons in the second hidden layer
    hidden1Synapse : 2d np.array
        Weights connecting the 7 inputs to the 7 neurons in the first
        hidden layer
    hidden2Synapse : 2d np.array
        Weights connecting the 7 neurons in the first hidden layer
        to the 4 neurons in the second hidden layer 
    outputSynapse : 2d np.array
        Weights connecting the 4 neurons in the second hidden layer
        to the three snowfall classes (light, moderate, heavy)
    month_idx : float
        Weighting value based on month
    f1, f2, f3, f4, f5, f6 : float
        6 input principal components calculated in calc_roebber_components()

    Returns:
    tuple of float
        Floats of each snow class probability
    """
    # Define empty arrays for network
    inputAxon = np.zeros(7)
    fgrid1 = np.zeros(7)
    fgrid2 = np.zeros(4)
    fgrid3 = np.zeros(3)
    outputAxon = np.zeros(3)
    f = [month_idx, f1, f2, f3, f4, f5, f6]

    # --- Start of neural network --- 
    # Calculate input axons based on calculated factors (f[k])
    for k in range(7):
        inputAxon[k] = inputFile[0, k] * f[k] + inputFile[1, k]

    # Calculate operation results of layers 1 and 2
    for k in range(7):
        for l in range(7):
            fgrid1[k] += hidden1Synapse[l, k] * inputAxon[l]
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

ANN_FUNCS = [mlp_1_hidden_layer] * 5 + [mlp_2_hidden_layers] * 5

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
    method did not do. This code was then converted to Python from Fortran using the UPP's UPP_PHYSICS.f
    code on Github (https://github.com/NOAA-EMC/UPP/blob/develop/sorc/ncep_post.fd/UPP_PHYSICS.f).
    Thus, users should be cautious when interpreting results from this algorithm.
    See Fig. 3 in Roebber et al. (2003) for confirmation on units for each variable.

    Parameters:
        pres_prof : np.array
            Input pressure profile (Pa)
        temp_prof : np.array
            Input temperature profile (Celsius)
        rh_prof : np.array
            Input relative humidity profile (%)
        gh_prof : np.array
            Input geopotential height profile (meter)
        mslp : float, np.array
            Input mean sea level pressure (Pa)
        psfc : float, np.array
            Input surface pressure (Pa)
        qpf : float, np.array
            Input QPF (mm)
        spd10m : float, np.array
            Input 10-meter wind speed (m/s)
        t2m : float, np.array
            Input 2-meter temperature (Celsius)
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
    rhms, tms = np.zeros(14), np.zeros(14)

    # Pressure pertubation calculation
    prp = mslp / psfc
    prp = prp * 100000 / psfc

    # Use levels above ground
    above_sfc = gh_prof > elev
    t_agl = np.where(above_sfc, temp_prof, np.nan)
    rh_agl = np.where(above_sfc, rh_prof, np.nan)
    sigma_agl = np.where(above_sfc, sigma_levels, np.nan)

    # Remove NaNs which are values below ground
    remove_below_vals = ~np.isnan(t_agl)
    t_agl = t_agl[remove_below_vals]
    rh_agl = rh_agl[remove_below_vals]
    sigma_agl = sigma_agl[remove_below_vals]

    # Set 2-meter values as first level
    t_agl[0], rh_agl[0], sigma_agl[0] = t2m, r2m, 1
  
    # Loop through each sigma level        
    for ks, sgw in enumerate(nbm_config.SIGMA_LEVS):
        # Loop through each level in each HRRR vertical profile above ground
        for j in range(len(t_agl) - 1):
            sg1, sg2 = sigma_agl[j], sigma_agl[j + 1]
            if sg1 == sgw:
                tms[ks], rhms[ks] = t_agl[j], rh_agl[j]
            elif sg2 == sgw:
                tms[ks], rhms[ks] = t_agl[j + 1], rh_agl[j + 1]
            # Interpolate between levels
            elif sgw < sg1 and sgw > sg2:
                dist = (sgw - sg1) / (sg2 - sg1)
                tms[ks] = t_agl[j] + dist * (t_agl[j + 1] - t_agl[j])
                rhms[ks] = rh_agl[j] + dist * (rh_agl[j + 1] - rh_agl[j])

    # Set first sigma level value to 2-meter temperature/RH
    tms[0], rhms[0] = t2m, r2m

    # Compute all 6 principal components
    components = [calc_roebber_components(c, qpf, spd10m, tms, rhms) for c in nbm_config.ROEBBER_COEFS]
    
    ob_collect_time = datetime.datetime.strptime(str(ob_collect_time), '%Y-%m-%d %H:%M:%S+00:00')
    if ob_collect_time.strftime("%m") == '01':
        month_idx = nbm_config.MONTH_IDXS[0]
    elif ob_collect_time.strftime("%m") == '03':
        month_idx = nbm_config.MONTH_IDXS[10]
    elif ob_collect_time.strftime("%m") == '04':
        month_idx = nbm_config.MONTH_IDXS[9]
    else:
        month_idx = nbm_config.MONTH_IDXS[11]

    hprob_tot, mprob_tot, lprob_tot = 0, 0, 0

    # Loop through each ANN member in the 10-member ensemble
    # and calculate member snow classes probabilities
    for k, member in enumerate(MEMBERS):
        probs = ANN_FUNCS[k](*member(), month_idx, *components)

        # Total probabilities for each snowfall class (heavy, medium, light)
        hprob_tot += probs[0]
        mprob_tot += probs[1]
        lprob_tot += probs[2]

    # Convert probabilites to fractions
    hfrac, mfrac, lfrac = hprob_tot / 10, mprob_tot / 10, lprob_tot / 10

    # Multiplication factor for light snow class
    lfrac_factor = 18 if lfrac < 0.67 else 27

    # Convert probabilities to explicit SLR
    slr = (hfrac * 8 + mfrac * 13 + lfrac * lfrac_factor) * prp / (hfrac + mfrac + lfrac)

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