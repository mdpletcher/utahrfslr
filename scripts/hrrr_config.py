"""
Michael Pletcher
Created: 03/06/2025
Edited:

config file for HRRR scripts
"""

# Levels needed for the SLR model (both ERA5 and HRRR)
VERTLEVS = [
    1000, 
    975, 
    950,
    925, 
    900, 
    875,
    850, 
    825, 
    800,
    775, 
    750, 
    725,
    700, 
    675, 
    650,
    625, 
    600, 
    575,
    550, 
    525, 
    500,
    475, 
    450, 
    425,
    400
]

# dict needed for selecting HRRR variables
# on pressure levels
ISOBARIC_SEL = {
    'typeOfLevel' : 'isobaricInhPa',
    'stepType' : 'instant'
}
# dict needed for selecting surface-based
# HRRR variables
SFC_SEL = {'stepType' : 'instant'}
# Selected HRRR variables
HRRR_VARS = [
    't', 
    'u', 
    'v', 
    'q', 
    'w', 
    'r', 
    'gh', 
    'tp', 
    '2t', 
    '2r', 
    '2d', 
    '10u', 
    '10v', 
    'orog', 
    'mslma', 
    'sp'
]
# Mapping for each HRRR variable when reading the .grib2 raw HRRR files
HRRR_KEYS = {
    'u': {'filter_by_keys': {**ISOBARIC_SEL, 'shortName': 'u'}, 'sel': VERTLEVS},
    'v': {'filter_by_keys': {**ISOBARIC_SEL, 'shortName': 'v'}, 'sel': VERTLEVS},
    'w': {'filter_by_keys': {**ISOBARIC_SEL, 'shortName': 'w'}, 'sel': VERTLEVS},
    'q': {'filter_by_keys': {**ISOBARIC_SEL, 'shortName': 'q'}, 'sel': VERTLEVS},
    'gh': {'filter_by_keys': {**ISOBARIC_SEL, 'shortName': 'gh'}, 'sel': VERTLEVS},
    'r': {'filter_by_keys': {**ISOBARIC_SEL, 'shortName': 'r'}, 'sel': VERTLEVS},
    't': {'filter_by_keys': {**ISOBARIC_SEL, 'shortName': 't'}, 'sel': VERTLEVS},
    'tp': {},
    '2t': {'filter_by_keys': {**SFC_SEL, 'shortName': '2t'}, 'sel': None},
    '2r': {'filter_by_keys': {**SFC_SEL, 'shortName': '2r'}, 'sel': None},
    '2d': {'filter_by_keys': {**SFC_SEL, 'shortName': '2d'}, 'sel': None},
    '10u': {'filter_by_keys': {**SFC_SEL, 'shortName': '10u'}, 'sel': None},
    '10v': {'filter_by_keys': {**SFC_SEL, 'shortName': '10v'}, 'sel': None},
    'orog': {'filter_by_keys': {**SFC_SEL, 'shortName': 'orog', 'typeOfLevel': 'surface'}, 'sel': None},
    'mslma': {'filter_by_keys': {**SFC_SEL, 'shortName': 'mslma'}, 'sel': None},
    'sp': {'filter_by_keys': {**SFC_SEL, 'typeOfLevel': 'surface', 'shortName': 'sp'}, 'sel': None},
}
# Mapping for surface-based HRRR variables
SFC_VAR_MAP = {
    '2t': 't2m',
    '2r': 'r2',
    '2d': 'd2m',
    '10u': 'u10',
    '10v': 'v10',
    'orog': 'orog',
    'mslma': 'mslma',
}

# HRRR isobaric keys
ISO_KEYS = [
    'gh', 
    'q', 
    'r', 
    't', 
    'u', 
    'v', 
    'w'
]
# HRRR surface keys
SFC_KEYS = [
    'tp', 
    '10u', 
    '10v', 
    'sp', 
    'mslma', 
    '2t', 
    '2r'
]

# Add units with modified keys
ISO_KEYS_UNITS = {
    'geopotential_height_gh': 'gpm (geopotential meters)', 
    'specific_humidity_q': 'kg/kg',
    'relative_humidity_r': '%',
    'temperature_t': 'K',
    'u_component_of_wind_u': 'm/s',
    'v_component_of_wind_v': 'm/s',
    'vertical_velocity_w': 'Pa/s' 
}
SFC_KEYS_UNITS = {
    '1h_precipitation_tp': 'kg/m^2',
    '10m_u_component_of_wind_10u': 'm/s',
    '10m_v_component_of_wind_10v': 'm/s',
    'surface_pressure_sp': 'Pa',
    'mean_sea_level_maps_system_reduction_mslma': 'Pa',
    '2m_temperature_2t': 'K',
    '2m_relative_humidity_2r': '%',
}