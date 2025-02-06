"""
Michael Pletcher
Created: 01/24/2025
Edited: 01/28/2025

##### Summary #####
.py script containg function to 

########### Function List ###########




"""

# Imports
import numpy as np
import pandas as pd
import xarray as xr
import os
import cfgrib
import netCDF4 as netcdf
import warnings
import metpy.calc as mpc

from datetime import datetime, timedelta
from scipy.spatial import KDTree
from multiprocessing import Pool

warnings.filterwarnings('ignore')

# Globals
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
ISOBARIC_SEL = {
    'typeOfLevel' : 'isobaricInhPa',
    'stepType' : 'instant'
}
SFC_SEL = {'stepType' : 'instant'}
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
SFC_VAR_MAP = {
    '2t': 't2m',
    '2r': 'r2',
    '2d': 'd2m',
    '10u': 'u10',
    '10v': 'v10',
    'orog': 'orog',
    'mslma': 'mslma',
}



def load_hrrr_var(fdir, init_time, fhr, key):
    f = fdir + init_time.strftime('%Y%m%d%H') + 'F' + str(fhr).zfill(2) + 'hrrr.grib2'
    if key == 'tp':
        #fhr = int(os.path.basename(f)[-12:10])
        filt_by_key = {'stepRange': str(fhr - 1) + '-' + str(fhr), 'shortName':'tp'}
    else:
        filt_by_key = HRRR_KEYS.get(key, {}).get('filter_by_keys', {})

    try:
        if key == 'tp':
            ds = xr.open_dataset(
                f,
                engine = 'cfgrib',
                decode_coords = 'all',
                filter_by_keys = filt_by_key
            )
        else:
            ds = xr.open_dataset(
                f,
                engine = 'cfgrib',
                decode_coords = 'all',
                filter_by_keys = filt_by_key
            )
    except Exception as e:
        print('Cannot load HRRR variable, %s' % e)

    if 'step' in ds.variables or 'step' in ds.dims or 'step' in ds.coords:
        if key in SFC_VAR_MAP:
            ds = ds[SFC_VAR_MAP[key]].drop_vars(['step'])
        elif key != 'tp':
            ds - ds.drop_vars(['step'])
    ds['longitude'] -= 360

    return ds

def load_all_hrrr_vars(fdir, init_time, fhr):
    datasets = [load_hrrr_var(fdir, init_time, fhr, var) for var in HRRR_VARS]
    merged_ds = xr.merge(datasets, compat = 'override')
    return merged_ds

def calc_needed_vars(ds, ptype = False):
    # Wind speed
    ds['spd'] = np.sqrt((ds.u ** 2) + (ds.v ** 2))

    # Broadcoasted pressure levels
    p = ds.isobaricInhPa
    _p = np.ones(ds.t.shape)
    _p = np.array([_p[i, :, :] * p[i].values for i in range(p.size)]).transpose(0, 1, 2)
    p = ds.t.copy().rename('p')
    p.values = _p
    ds['p'] = p

    '''
    if ptype:
        # Pressure units for metpy
        p_unit = p.values * units.millibar
        # Temperature units for metpy
        t_unit = (ds.t.values - 273.15) * units.degC
        # Metpy mixing ratio calculations
        qv = ds.t.copy().rename('qv')
        qv.values = np.array(
            mpc.mixing_ratio_from_relative_humidity(
                pw_unit,
                t_unit,
                ds.r.values / 100
            )
        )

        # Repair dimensions after metpy calc
        qv['time'], qv['isobaricInhPa'], qv['latitude'], qv['longitude'] = ds.time, ds.isobaricInhPa, ds.latitude, ds.longitude
        # Wet-bulb temperature calculation
        tw = wrf.wetbulb(p * 100, ds.t, qv, units = 'degK')
        ds['tw'] = tw

        # Surface calculations
        qvsurf = mpc.specific_humidity_from_dewpoint(
            ds.sp * units.Pa,
            ds.d2m * units.degK
        )
        twsurf = wrf.wetbulb(
            ds.sp.values.flatten(),
            ds.t2m.values.flatten(),
            qvsurf.values.flatten()
        ).reshape(ds.sp.shape)

    '''
    return ds

def get_datetime(file):
    match = re.search(r'(\d{4})(\d{2})(\d{2})(\d{2})F(\d{2})', file)
    if not match:
        raise ValueError('Filename %s does not match the expected pattern' % file)
    yr, mn, dy, init, fhr = map(int, match.groups())
    # Make sure the forecast hour is between 13 and 24
    if 12 < fhr <= 24:
        return datetime(yr, mn, dy, init) + timedelta(hours = fhr)

def add_time_dim(ds, file):
    valid_dt = get_datetime(file)
    ds = ds.expand_dims(valid_time = [valid_dt])
    return ds

def load_sample_hrrr_kdtree(file):

    print('Using %s sample .nc file' % file)
    sample = xr.open_dataset(file)
    lats, lons = sample.latitude.values, sample.longitude.values
    sample['longitude'] -= 360
    print('Building KDTree')

    # Create projection
    proj = ccrs.LambertConformal(
        central_longitude = -97.5,
        central_latitude = 38.5,
        standard_parallels = [38.5]
    )
    # Create transform
    transform = np.vectorize(
        lambda x, y: proj.transform_point(x, y, ccrs.PlateCarree())
    )
    print('Transforming grid...')

    # Transform HRRR lat/lon grid
    grid_x, grid_y = sample.isel(x = 0), sample.isel(y = 0)
    _, proj_y = transform(grid_y.longitude, grid_y.latitude)
    proj_x, _ = transform(grid_x.longitude, grid_x.latitude)
    sample['x'], sample['y'] = proj_x, proj_y
    proj_lons, proj_lats = transform(lons, lats)

    # Create KDTree
    return KDTree(list(zip(proj_lons.ravel(), proj_lats.ravel())))

def select_grid_points(df, kdtree, sample):
    if df is not None and kdtree is not None:
        selected_grid_points = {}
        for idx, row in df.iterrows():
            lat, lon = (row.lat, row.lon) if 'lat' in row and 'lon' in row else (row.latitude, row.longitude)
            # Transform site lat/lon to match transformed HRRR grid
            proj = ccrs.LambertConformal(
                central_longitude = -97.5,
                central_latitude = 38.5,
                standard_parallels = [38.5]
            )
            transform = np.vectorize(
                lambda x, y: proj.transform_point(x, y, ccrs.PlateCarree())
            )
            transformed_lon, transformed_lat = transform(lon, lat)
            dist, idx = kdtree.query(
                np.array([transformed_lon, transformed_lat])
            )
            # Select grid points
            yi, xi = np.unravel_index(idx, sample.latitude.shape)
            grid_lat, grid_lon = sample.latitude.isel(y = yi, x = xi).values, sample.longitude.isel(y = yi, x = xi).values
            selected_grid_points[idx] = (yi, xi, grid_lat, grid_lon)

    return selected_grid_points

def extract_profile(file, key, selected_grid_points):
    ds = load_hrrr_var(file, key)
    ds = add_time_dim(ds, fname)
    ds = ds.chunk({'valid_time' : 1}).load()
    profiles = []
    for yi, xi, grid_lat, grid_lon in selected_grid_points.values():
        if key in SFC_VAR_MAP.keys():
            prof = ds.isel(y = yi, x = xi)
        else:
            prof = ds[key].isel(y = yi, x = xi)
        prof.attrs = {}
        profiles.append((prof, grid_lat, grid_lon))
        prof.close()
        del prof
    ds.close()
    del ds
    gc.collect()

    return profiles

def concat_profiles(init_time, profiles, key, savedir):
    prof_dict = defaultdict(list)
    for prof, lat, lon in profiles:
        lat_lon_tuple = (
            lat.item() if isinstance(lat, np.ndarray) else lat,
            lon.item() if isinstance(lon, np.ndarray) else lon
        )
        prof_dict[lat_lon_tuple].append((prof))
    date = init_time[:8]
    os.makedirs(os.path.join(savedir, date), exist_ok = True)
    for (lat, lon), prof_list in prof_dict.items():
        fname = 'hrrrprof_{:.3f}N_{:.3f}W.{:s}.{:s}.nc'.format(
            lat, 
            abs(lon),
            key,
            init_time,
        )
        if os.path.isfile(savedir + date + '/' + fname):
            print(f'{os.path.basename(fname)} exists, skipping')
            pass
        else:
            try:
                prof_list = [prof for prof in prof_list if prof is not None]
                concat_prof = xr.concat(
                    prof_list,
                    dim = 'valid_time'
                ).rename({'time': 'init_time'})
                concat_prof.load().to_netcdf(
                    savedir + date + '/' + fname, engine = 'h5netcdf'
                )
                concat_prof.close()
                del concat_prof
                gc.collect()
            except:
                raise

def save_var(key, flist, df):
    try:
        inits = {}
        for f in flist:
            init_time = os.path.basename(f)[:10]
            if init_time not in inits:
                inits[init_time] = []
            inits[init_time].append(f)
        for init_time, files in inits.items():
            profs = []
            for f in files:
                fhr = int(os.path.basename(f)[-12:-10])
                # Only use forecast hours between 13 and 24
                if fhr > 12 and fhr <= 24:
                    prof = extract_profile(f, df, key)
                    profs.extend(prof)
                    del prof
            concat_profiles(init_time, profs, key)
            print('Saved all %s %s files' % (key, init_time))
    except:
        raise

def run_parallel(flist, df, keys):
    # Create list of items for parallel processing
    # Number of processes of equal to number of keys
    items, n_processes = [(key, flist, df) for key in keys], len(keys)
    print(
        'Extracting profiles for these HRRR variables: %s\nRunning in parallel with %s processes' % (keys, n_processes)
    )
    with get_context('fork').Pool(processes = n_processes) as p:
        p.starmap(save_var, items)
        p.close()
        p.join()






    

    
