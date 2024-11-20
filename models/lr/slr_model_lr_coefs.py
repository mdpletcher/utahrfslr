# This file contains the linear regression coefficients
# used to run the SLR model developed by Steenburgh 
# group in the Department of Atmospheric Sciences at
# the University of Utah. 

# Created 11/13/2024 by Michael Pletcher

# This iteration is the first to be sent to EMC 
# after the meeting with Jesse Meng and Hui-Ya Chuang
# on Nov 13, 2024. Addtional iterations will likely
# follow based on requests from EMC.

# Units for each input feature
'''
Wind speed units: m/s
Temperature units: 
RH units: %
Latitude units: degrees
Longitude units: degrees (be aware that
we subtracted 360 from our longitude values,
be sure to do the same)
Elevation units: meters
'''

# Each feature is named in the below dictionary
# as it would be when used to predict SLR. For example,
# 2.1-km AGL relative humidity is abbreviated R21K, 
# 300 m AGL temperature is abbreviated T03K, latitude
# is abbreviated lat, etc. 

# Dictionary containing the linear regression
# coefficients and the intercept
lr_coefs = {
    "SPD03K": 0.2561487739186439,
    "SPD06K": -0.34144077110587795,
    "SPD09K": 0.01996173942380777,
    "SPD12K": 0.11973617589553293,
    "SPD15K": -0.2293664677626013,
    "SPD18K": 0.3574318535292301,
    "SPD21K": -0.5474236558935975,
    "SPD24K": 0.3394467555198417,
    "T03K": 0.16284421565307805,
    "T06K": -0.38372066460038723,
    "T09K": 0.25782534617989483,
    "T12K": -0.03433753041663118,
    "T15K": -0.020999148104604428,
    "T18K": 0.2931240591479839,
    "T21K": -0.5107554450495457,
    "T24K": -0.0846096410101448,
    "R03K": -0.0903241932688199,
    "R06K": 0.07233551130886726,
    "R09K": -0.018165220741486333,
    "R12K": -0.010248879017236093,
    "R15K": 0.025090729666029845,
    "R18K": 0.022627155290128792,
    "R21K": -0.03024236012809826,
    "R24K": -0.023962673629607655,
    "lat": 0.1297888748855706,
    "lon": 0.022325615084860806,
    "elev": 0.00026349443366296965,
    "intercept": 96.85075137
}