"""
Michael Pletcher
Created: 06/12/2025
Edited: 

.py script containing tools for post-processing
model data
"""


import numpy as np

def get_profile_arrays(i, levs, var_arrays, input_vars):
    profile_data = {var: [] for var in input_vars}

    for lev in levs:
        for var in input_vars:
            profile_data[var].append(var_arrays[f'{var}{lev}'][i])

    # Convert lists to numpy arrays
    for var in profile_data:
        profile_data[var] = np.array(profile_data[var])

    # Compute derived wind speed if U and V are available
    if 'U' in profile_data and 'V' in profile_data:
        profile_data['SPD'] = np.sqrt(profile_data['U'] ** 2 + profile_data['V'] ** 2)

    return profile_data

def interpolate_linear(val1, val2, frac):
    return val2 + (val1 - val2) * frac

def pressure_levs_to_agl(df, input_vars, pressure_levs, agl_levs):
    # Initialize structures
    pressure_var_list = [
        '%s%s' % (var, pressure_lev) for var in input_vars for pressure_lev in pressure_levs
    ]
    pressure_var_arrays = {
        var: np.array(df[var][:]) for var in pressure_var_list
    }
    agl_var_dict = {
        f"{var}{key}" : np.zeros(len(df)) for var in input_vars for key in keys_levs
    }
    keys_levs = ["%02dK" % (agl_lev // 100) for agl_lev in agl_levs[1:]]

    # Loop through all records
    for i in range(len(df)):
        try:
            profile_data = get_profile_arrays(
                i,
                pressure_levs,
                pressure_var_arrays,
                input_vars
            )
            for agl_lev, key_lev in zip(agl_levs, keys_levs):
                if 'GH' in profile_data.keys():
                    profile_data['Z'] = profile_data.pop('GH')

                # Get nearest profile levels based on site elevation
                z_prof = profile_data['Z']
                hgt_target = df.site_elev[i] + agl_lev
                hgt_diff = np.abs(z_prof - hgt_target)
                z1, z2 = np.argsort(hgt_diff)[:2]

                # Compute height fraction
                if z_prof[z2] == z_prof[z1]:
                    hgt_frac = 0
                else:
                    hgt_frac = hgt_diff[z2] / (z_prof[z2] - z_prof[z1])

                # Create dict of interpolated variables 
                interp_vals = {
                    key: interpolate_linear(profile_data[key][z1], profile_data[key][z2], hgt_frac) for key in profile_data
                }

                # Add data to dict
                for key, val in interp_vals.items():
                    full_key = f"{key}{key_lev}"
                    if full_key in agl_var_dict:
                        agl_var_dict[full_key][i] = val

        except Exception as e:
            print(f"Skipping row {i} due to error: {e}")
    
    # Add data to original dataframe
    for key, array in agl_var_dict.items():
        df[key] = array
    
    return df