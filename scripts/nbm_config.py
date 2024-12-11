"""
Michael Pletcher
Created: 10/07/2024
Edited: 

##### Summary #####
Config file containing coefficients
used in the Cobb, MaxTAloft, 850 - 700mb Thickness,
and Roebber snow-to-liquid ratio methods. Each 
method is currently configured for NBM v4.2 
(see https://vlab.noaa.gov/web/mdl/nbm-versions#version-4-2 for more info on v4.2).
If access is available, The COMET Program also contains more info on 
each SLR method (see https://www.meted.ucar.edu/nwp/NBM40_snow/index.htm)


"""


# Globals for the Cobb method [see Cobb and Waldstreicher (2005)
# for more info]
COBB_TTHRESH = [
    -24, -21, -19, 
    -16, -12, -10, 
    -8, -7, -5, 
    -3, 0
] # Temperature thresholds
C1_COBB = [
    8, 12, 21,
    30, 19, 9, 
    8, 9, 13, 
    6, 2
]
C2_COBB = [
    -0.0017, 3.5034, 4.9065, 
    -0.4650, -4.7608, -2.0799, 
    0.3122, 2.0127, -0.7004, 
    -3.7110, 0
]
C3_COBB = [
    0, 1.1684, -0.4668, 
    -1.0679, -0.1594, 1.2318, 
    0.3630, 1.3375, -2.6941, 
    1.1888, 0
]
C4_COBB = [
    0.1298, -0.2725, -0.0573, 
    0.0865, 0.1855, -0.1931, 
    0.3249, -0.6719, 0.6472, 
    -0.1321, 0
]
# List of each Cobb coefficient
COEFS_COBB = [C1_COBB, C2_COBB, C3_COBB, C4_COBB]



# Globals for MaxTAloft 
COEFS_MAXT = [
    0.0000045, 0.0004432, 0.0130903,
    0.0585968, 1.8150809, 5.9805722
]



# Globals for Roebber ANN method
# Sigma levels
SIGMA_LEVS = [
    1.0, 0.975, 0.95, 
    0.925, 0.9, 0.875, 
    0.85, 0.8, 0.75, 
    0.7, 0.65, 0.6, 
    0.5, 0.4
]
# Monthly indexes
MONTH_IDXS = [
    1.0, 0.67, 0.33,
    0.0, -0.33, -0.67, 
    -1.00, -0.67, -0.33, 
    0.0, 0.33, 0.67
]
# Neural network coefficents
# First coefficient list
C1_ROEBBER = [
    -0.2926, 0.0070, -0.0099, 
    0.0358, 0.0356, 0.0353, 
    0.0333, 0.0291, 0.0235, 
    0.0169, 0.0060, -0.0009, 
    -0.0052, -0.0079, -0.0093,
    -0.0116, -0.0137, 0.0030, 
    0.0033, -0.0005, -0.0024, 
    -0.0023, -0.0021, -0.0007, 
    0.0013, 0.0023, 0.0024, 
    0.0012, 0.0002, -0.0010
]
# Second coefficient list
C2_ROEBBER = [
    -9.7961, 0.0099, -0.0222, 
    -0.0036, -0.0012, 0.0010, 
    0.0018, 0.0018, 0.0011, 
    -0.0001, -0.0016, -0.0026, 
    -0.0021, -0.0015, -0.0010,
    -0.0008, -0.0017, 0.0238, 
    0.0213, 0.0253, 0.0232, 
    0.0183, 0.0127, 0.0041, 
    -0.0063, -0.0088, -0.0062, 
    -0.0029, 0.0002, 0.0019
]
# Third coefficient list
C3_ROEBBER = [
    5.0037, -0.0097, -0.0130,
    -0.0170, -0.0158, -0.0141, 
    -0.0097, -0.0034, 0.0032, 
    0.0104, 0.0200, 0.0248, 
    0.0273, 0.0280, 0.0276,
    0.0285, 0.0308, -0.0036, 
    -0.0042, -0.0013, 0.0011, 
    0.0014, 0.0023, 0.0011, 
    -0.0004, -0.0022, -0.0030, 
    -0.0033, -0.0031, -0.0019
]
# Fourth coefficient list
C4_ROEBBER = [
    -5.0141, 0.0172, -0.0267, 
    0.0015, 0.0026, 0.0033, 
    0.0015, -0.0007, -0.0030, 
    -0.0063, -0.0079, -0.0074, 
    -0.0055, -0.0035, -0.0015,
    -0.0038, -0.0093, 0.0052, 
    0.0059, 0.0019, -0.0022, 
    -0.0077, -0.0102, -0.0109, 
    -0.0077, 0.0014, 0.0160, 
    0.0217, 0.0219, 0.0190
]
# Fifth coefficient list
C5_ROEBBER = [
    -5.2807, -0.0240, 0.0228, 
    0.0067, 0.0019, -0.0010, 
    -0.0003, 0.0012, 0.0027, 
    0.0056, 0.0067, 0.0067, 
    0.0034, 0.0005, -0.0026, 
    -0.0039, -0.0033, -0.0225, 
    -0.0152, -0.0157, -0.0094, 
    0.0049, 0.0138, 0.0269, 
    0.0388, 0.0334, 0.0147, 
    0.0018, -0.0066, -0.0112
]
# Sixth coefficient list
C6_ROEBBER = [
    -2.2663, 0.0983, 0.3666, 
    0.0100, 0.0062, 0.0020, 
    -0.0008, -0.0036, -0.0052, 
    -0.0074, -0.0086, -0.0072, 
    -0.0057, -0.0040, -0.0011,
    0.0006, 0.0014, 0.0012, 
    -0.0005, -0.0019, 0.0003, 
    -0.0007, -0.0008, 0.0022, 
    0.0005, -0.0016, -0.0052, 
    -0.0024, 0.0008, 0.0037
]
# List of each Roebber coefficent
ROEBBER_COEFS = [
    C1_ROEBBER, C2_ROEBBER, C3_ROEBBER, 
    C4_ROEBBER, C5_ROEBBER, C6_ROEBBER
]



# Globals for 850-700mb thickness
# 850-700mb Thickness coefficients
# These numbers were retrieved from
# Robert James of MDL in the Fortran code
THICK_COEFS = [0.16559, 263.35]