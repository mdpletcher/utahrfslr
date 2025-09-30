"""
Michael Pletcher
Created: 01/24/2025
Edited: 03/05/2025

##### Summary #####
.py script containg functions to read, process,
and save HRRR data, specifically extracting individual
HRRR profiles from the HRRR's grid
"""

# Imports
import numpy as np
import pandas as pd
import xarray as xr
import re
import os
import sys
import cfgrib
import netCDF4 as netcdf
import warnings
import metpy.calc as mpc
import hrrr_config
import cartopy.crs as ccrs
import gc

from datetime import datetime, timedelta
from scipy.spatial import KDTree
from multiprocessing import get_context

sys.path.append('../configs/')
import hrrr_config

warnings.filterwarnings('ignore')

def load_hrrr_var(fdir, init_time, fhr, key):
    """
    Load HRRR variable from input .grib2 file based on initialization time,
    forecast hour, and variable key

    Parameters:
    fdir : str
        Input file directory that contains .grib2 files
    init_time : str
        Initialization time of forecast
    fhr : str
        Forecast hour
    key : str
        HRRR .grib2 key

    Returns:
    xarray.DataArray
        Requested HRRR variable in xarray.DataArray format

    """
    f = fdir + init_time.strftime('%Y%m%d%H') + 'F' + str(fhr).zfill(2) + 'hrrr.grib2'
    if key == 'tp':
        #fhr = int(os.path.basename(f)[-12:10])
        filt_by_key = {'stepRange': str(fhr - 1) + '-' + str(fhr), 'shortName':'tp'}
    else:
        filt_by_key = hrrr_config.HRRR_KEYS.get(key, {}).get('filter_by_keys', {})

    try:
        ds = xr.open_dataset(
            f,
            engine = 'cfgrib',
            decode_coords = 'all',
            filter_by_keys = filt_by_key
        )
    except Exception as e:
        print('Cannot load HRRR variable, %s' % e)

    if 'step' in ds.variables or 'step' in ds.dims or 'step' in ds.coords:
        if key in hrrr_config.SFC_VAR_MAP:
            ds = ds[hrrr_config.SFC_VAR_MAP[key]].drop_vars(['step'])
        elif key != 'tp':
            ds = ds.drop_vars(['step'])
    ds['longitude'] -= 360

    return ds

def load_all_hrrr_vars(fdir, init_time, fhr):
    """
    Load all HRRR variables listed in hrrr_config.py. The list
    of these variables can be user modified.

    Params:
    fdir : str
        Input file directory that contains .grib2 files
    init_time : str
        Initialization time of forecast
    fhr : str
        Forecast hour

    Returns:
    xarray.Dataset
        Merged xarray.Dataset containing all requested HRRR
        variables
    """

    datasets = [load_hrrr_var(fdir, init_time, fhr, var) for var in hrrr_config.HRRR_VARS]
    merged_ds = xr.merge(datasets, compat = 'override')
    return merged_ds

def calc_needed_vars(ds, ptype = False):

    """
    Calculated the variables required to predict SLR
    using machine learning models

    Params:
    ds : xarray.Dataset
        Input xarray.Dataset containing HRRR variables
    ptype : boolean, optional
        Calculate extra variables for adjusting SLR

    Returns:
    xarray.Dataset
        Modified xarray.Dataset with newly calculated
        variables
    """

    # Wind speed
    ds['spd'] = np.sqrt((ds.u ** 2) + (ds.v ** 2))

    # Broadcoasted pressure levels
    p = ds.isobaricInhPa
    _p = np.ones(ds.t.shape)
    _p = np.array([_p[i, :, :] * p[i].values for i in range(p.size)]).transpose(0, 1, 2)
    p = ds.t.copy().rename('p')
    p.values = _p
    ds['p'] = p

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
        ds['qvsurf'], ds['twsurf'] = qvsurf, twsurf

    return ds

def get_datetime(fname):
    """
    Get the datetime from the HRRR file name

    Params:
    file : str
        Input file name containing a timestamp in the format YYYYMMDDHHFHH

    Returns:
    datetime.datetime
        datetime object representing the HRRR valid forecast time
        if the forecast hour is between 13 and 24
    """

    match = re.search(r'(\d{4})(\d{2})(\d{2})(\d{2})F(\d{2})', fname)
    if not match:
        raise ValueError('Filename %s does not match the expected pattern' % fname)
    yr, mn, dy, init, fhr = map(int, match.groups())
    # Make sure the forecast hour is between 13 and 24
    if 12 < fhr <= 24:
        return datetime(yr, mn, dy, init) + timedelta(hours = fhr)

def add_time_dim(ds, fname):
    """
    Add the time dimension to a 1-d HRRR profile

    Params:
    ds: xarray.DataArray
        Input xarray.DataArray HRRR profile
    file : str
        Input file name

    Returns:
    xarray.DataArray
        xarray.DataArray with added 'valid_time' dimension
        corresponding to xtracted datetime from the file name
    """

    valid_dt = get_datetime(fname)
    ds = ds.expand_dims(valid_time = [valid_dt])
    return ds

def load_sample_hrrr_kdtree(file):

    """
    Load a sample .nc HRRR file and create KDTree of 
    projected HRRR lat/lon grid points

    Params:
    file : str
        Path to the sample HRRR .nc file
    
    Returns:
    scipy.spatial.KDTree
        KDTree of transformed HRRR grid coordinates in a 
        Lambert Conformal projection
    """

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
    grid_y, grid_x = sample.isel(x = 0), sample.isel(y = 0)
    _, proj_y = transform(grid_y.longitude, grid_y.latitude)
    proj_x, _ = transform(grid_x.longitude, grid_x.latitude)
    sample['x'], sample['y'] = proj_x, proj_y
    proj_lons, proj_lats = transform(lons, lats)

    # Create KDTree
    return KDTree(list(zip(proj_lons.ravel(), proj_lats.ravel())))

def select_grid_points(df, kdtree, sample_file):
    """
    Select the nearest HRRR grid points for given lat/lon
    coordinates from pandas DataFrame using a KDTree

    Params:
    df : pd.DataFrame
        pandas DataFrame containing site locations with lat/lon
        information
    kdtree : scipy.spatial.KDTree
        Preprocessed KDTree with HRRR transformed grid coordinates
    sample_file : xarray.Dataset
        Sample HRRR dataset

    Returns:
    dict
        Dictionary mapping index values to tuples containing
        (yi, xi, grid_lat, grid_lon), where
        - yi, xi are indices of the selected HRRR grid point
        - grid_lat, grid_lon are corresponding lat/lon of the 
        selected HRRR grid point
    """

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
            yi, xi = np.unravel_index(idx, sample_file.latitude.shape)
            grid_lat, grid_lon = sample_file.latitude.isel(y = yi, x = xi).values, sample_file.longitude.isel(y = yi, x = xi).values
            selected_grid_points[idx] = (yi, xi, grid_lat, grid_lon)

    return selected_grid_points

def extract_profile(
    key, 
    selected_grid_points,
    init_time,
    fhr,
    fdir = None,
    ds = None
):
    """
    Extract 1-h profile HRRR at selected grid points

    Params:
    file : str
        Path to input HRRR .grib2 file
    key : str
        Specified HRRR variable
    selected_grid_points : dict
        Dictionary mapping index values to tuples containing
        (yi, xi, grid_lat, grid_lon), where
        - yi, xi are indices of the selected HRRR grid point
        - grid_lat, grid_lon are corresponding lat/lon of the 
        selected HRRR grid point

    Returns:
    list of tuples
        List of extracted profiles, where each tuple contains
        - xarray.DataArray: Extracted HRRR variable profile at
        the selected grid point
        - float: Latitude of the selected HRRR grid point
        - float: Longiude of the selected HRRR grid point
    """
    if fdir is not None:
        ds = load_hrrr_var(fdir, init_time, fhr, key)
    fname = init_time.strftime('%Y%m%d%H') + 'F' + str(fhr).zfill(2) + 'hrrr.grib2'
    ds = add_time_dim(ds, fname)
    ds = ds.chunk({'valid_time' : 1}).load()
    profiles = []
    for yi, xi, grid_lat, grid_lon in selected_grid_points.values():
        if key in hrrr_config.SFC_VAR_MAP.keys():
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
    """
    Concatenates extracted 1-h HRRR profiles along valid_time 
    dimension for unique initialization time

    Params:
    init_time : str
        Initialization time of forecast
    profiles : list of tuples
        List of extracted profiles, where each tuple contains
        - xarray.DataArray: Extracted HRRR variable profile at
        the selected grid point
        - float: Latitude of the selected HRRR grid point
        - float: Longiude of the selected HRRR grid point
    key : str
        Specified HRRR variable
    savedir : str
        Directory where concatenated profiles are saved

    Returns:
    None
        Saves .nc profiles for each lat/lon location
    """

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

def process_var(key, flist, selected_grid_points, savedir):
    """
    Processes HRRR variable by grouping by initialization
    time, extracting 1-h forecasts for each initialization
    time, concatenating along the forecast valid times, and
    then saving them

    Params:
    key : str
        Specified HRRR variable
    flist : list of str
        List of HRRR file paths
    selected_grid_points : dict
        Dictionary mapping index values to tuples containing
        (yi, xi, grid_lat, grid_lon), where
        - yi, xi are indices of the selected HRRR grid point
        - grid_lat, grid_lon are corresponding lat/lon of the 
        selected HRRR grid point
    savedir : str
        Directory path where processed .nc files are saved

    Returns:
    None
        Saves concatenated NetCDF files for each unique initialization 
        time
    """

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
                    prof = extract_profile(f, key, selected_grid_points)
                    profs.extend(prof)
                    del prof
            concat_profiles(init_time, profs, key, savedir)
            print('Saved all %s %s files' % (key, init_time))
    except:
        raise

def run_parallel(flist, df, keys, sample_file, savedir):
    """
    Runs the process_var() function by parallelizing by 
    each HRRR variable

    Params:
    flist : list of str
        List of HRRR file paths
    df : pandas.DataFrame
        DataFrame containing lat/lon points used to find
        nearest HRRR profile to be processed
    keys : list of str
        List of HRRR variables to process
    sample_file : str
        Sample HRRR dataset
    savedir : 
        Directory path where processed .nc files are saved

    Returns:
    None
    """
    kdtree = load_sample_hrrr_kdtree(sample_file)
    selected_grid_points = select_grid_points(df, kdtree, sample_file)

    # Create list of items for parallel processing
    # Number of processes of equal to number of keys
    items, n_processes = [(key, flist, selected_grid_points, savedir) for key in keys], len(keys)
    print(
        'Extracting profiles for these HRRR variables: %s\n \
         Running in parallel with %s processes' % (keys, n_processes)
    )
    with get_context('fork').Pool(processes = n_processes) as p:
        p.starmap(process_var, items)
        p.close()
        p.join()

def concat_profiles_by_time(directory, lat, lon, key, year):
    """
    Concatenate HRRR profiles by time given lat, lon, variable,
    and year. 

    Params:
    directory : str
        Path to directory containing subdirectories (e.g., a directory named
        20190226 contains hourly profiles for each specified variable (key)
        and each specified lat/lon)
    lat : float
        HRRR grid point latitude value
    lon : float
        HRRR grid point longitude value
    key : str
        Specified HRRR variable to process
    year : int
        Specified year to process
    
    Returns:
    xarray.Dataset
        Concatenated HRRR profiles along 'valid_time' dimension
    """

    dirs = sorted([d for d in os.listdir(directory) if str(year) in d and os.path.isdir(os.path.join(directory, d))])
    iso_profs, sfc_vars = [], []
    for dir_ in dirs:
        path = os.path.join(directory, dir_)
        fstr = '*%.3f*%.3f*%s.%s*.nc' % (lat, abs(lon), key, year)
        nc_files = sorted(glob(os.path.join(path, fstr)))
        for f in nc_files:
            d = xr.open_dataset(f).load()
            if 'isobaricInhPa' in list(d.dims):
                iso_profs.append(d.drop_vars(['latitude', 'longitude']))
            else:
                sfc_vars.append(d.drop_vars(['latitude', 'longitude']))
    
    # Return based on whether var is 1-d or 2-d
    if 'isobaricInhPa' in list(d.dims):
        return xr.concat(iso_profs, dim = 'valid_time')
    else:
        return xr.concat(sfc_vars, dim = 'valid_time')

def merge_profiles(lat, lon, year, fsavedir):
    """
    Merges all specified HRRR variables into
    a single .nc file

    Params:
    lat : float
        HRRR grid point latitude value
    lon : float
        HRRR grid point longitude value
    year : int
        Specified year to process
    fsavedir : str

    Returns:
    None
        Saves merged .nc file as xarray.Dataset
    """
    savestr = 'hrrrprof.%.4fN_%.4fW.2021.nc' % (lat, abs(lon))
    if os.path.isfile(outdir + savestr):
        print(f'{os.path.basename(savestr)} exists, skipping')
        pass
    else:
        iso, sfc = [], []
        print('Merging these variables: %s, %s' % (hrrr_config.ISO_KEYS, hrrr_config.SFC_KEYS))
        for key in np.append(hrrr_config.ISO_KEYS, hrrr_config.SFC_KEYS):
            if key in hrrr_config.ISO_KEYS:
                iso.append(concat_profiles_by_time(_dir, lat, lon, key))
            else:
                sfc.append(concat_profiles_by_time(_dir, lat, lon, key))
        merged = xr.merge(
            [xr.merge(iso), xr.merge(sfc, compat = 'override')]
        )
        del iso, sfc

        # Add attributes
        merged.attrs['history'] = 'Created {}'.format(datetime.utcnow())
        for key, unit in {**hrrr_config.ISO_KEYS_UNITS, **hrrr_config.SFC_KEYS_UNITS}.items():
            merged.attrs[key + ' units'] = unit
        merged.attrs['latitude'], merged.attrs['longitude'] = lat, lon

        print('Saving: %s'%savestr)
        merged.to_netcdf(fsavedir + savestr, engine = 'h5netcdf')
        del merged
        gc.collect()