# This file contains the linear regression coefficients
# used to run the SLR model developed by Steenburgh 
# group in the Department of Atmospheric Sciences at
# the University of Utah. 

# Created 11/20/2024 by Michael Pletcher

# This iteration is the same as the file
# 'slr_model_lr_coefs.py' but does not use lat/lon
# and elevation as predictors. This was done so that 
# the model can be reasonably applied to global models 
# like the GFS since it is trained only using data from
# the continental US.

# Units for each input feature
'''
Wind speed units: m/s
Temperature units: 
RH units: %
'''

# Each feature is named in the below dictionary
# as it would be when used to predict SLR. For example,
# 2.1-km AGL relative humidity is abbreviated R21K, 
# 300 m AGL temperature is abbreviated T03K, etc.

# Dictionary containing the linear regression
# coefficients and the intercept
lr_coefs = {
    "SPD03K": 0.2113589753880838,
    "SPD06K": -0.3113780353218734,
    "SPD09K": 0.030295727788329747,
    "SPD12K": 0.14200126274780872,
    "SPD15K": -0.3036948150474089,
    "SPD18K": 0.36742135429588796,
    "SPD21K": -0.45316009735021756,
    "SPD24K": 0.2732018488504477,
    "T03K": 0.08908223593334653,
    "T06K": -0.24948847161912707,
    "T09K": 0.14521457107694088,
    "T12K": 0.17265963006356744,
    "T15K": -0.3741056734263027,
    "T18K": 0.39704205782424823,
    "T21K": -0.36577798019766355,
    "T24K": -0.12603742209070648,
    "R03K": -0.08523012915185951,
    "R06K": 0.0879493556495648,
    "R09K": -0.04508491900731953,
    "R12K": 0.0347032913014311,
    "R15K": -0.031872141634061976,
    "R18K": 0.05199814866971972,
    "R21K": -0.02739515218481534,
    "R24K": -0.0338838765912164,
    "intercept": 97.96209163
}