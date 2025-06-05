"""
Michael Pletcher, Michael Wessler
Created: 06/04/2025
Edited:

.py script for reading ERA5 fields, extracting
variables (parallelized by year), and then
saving the variables.
"""

import numpy as np
import xarray as xr
import pandas as pd
import gc
import sys
import os
import warnings

from glob import glob
from multiprocessing import get_context, Pool, cpu_count
from functools import partial
from scipy.spatial import KDTree

os.environ['OMP_NUM_THREADS'] = '1'

# Directories
isodir = '/uufs/chpc.utah.edu/common/home/steenburgh-group8/era5/iso/'
sfcdir = '/uufs/chpc.utah.edu/common/home/steenburgh-group8/era5/sfc/'
profdir = '/uufs/chpc.utah.edu/common/home/steenburgh-group10/mpletcher/SLR_CONUS/data/era5/profiles/disagg_2000_2024/'
scriptdir = '/uufs/chpc.utah.edu/common/home/u1070830/code/model-tools/era5/'
sampledir = '/uufs/chpc.utah.edu/common/home/steenburgh-group10/mpletcher/SLR_CONUS/data/era5/' 

def mkdir_p(path):
    import errno    
    import os

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    
    return path

def preprocess(ds):
    # Rename 'valid_time' to 'time' and 'pressure_level' to 'level' if present
    rename_dict = {}
    if 'valid_time' in list(ds.dims):
        rename_dict['valid_time'] = 'time'
    if 'pressure_level' in list(ds.dims):
        rename_dict['pressure_level'] = 'level'
    ds = ds.drop_vars(['expver', 'number'], errors='ignore')
    ds = ds.rename(rename_dict)
    return ds

def get_year(year, key, levelset, xi, yi):
    
    print('Working on: %s %04d' % (key, year))
        
    year_data = []
    for month in [11, 12, 1, 2, 3, 4, 5]:
        datadir = isodir #if levelset == 'iso' else sfcdir
        date_dir = datadir + '%04d%02d'%(year, month)
        flist = sorted(glob(date_dir + '/*_%s.*' % key))
        try:
            # Open data at monthly intervals and select nearest ERA5 gridpoint
            month_data = xr.open_mfdataset(
                flist, 
                concat_dim = 'time', 
                drop_variables = ['utc_date'], 
                parallel = False,
                decode_cf = True, 
                decode_times = True, 
                decode_coords = False, 
                combine = 'nested',
                preprocess = preprocess
            ).isel(latitude = yi, longitude = xi).drop_vars(['latitude', 'longitude'])
        except Exception as e:
           print(f'Failed: %04d %02d, Error: {e}' % (year, month))
        else:
            # Profiles 
            if levelset == 'iso':
                month_data = month_data.chunk(
                    {
                        'time': month_data[key].shape[0] * 1,
                        'level': month_data[key].shape[1] * 1,
                    }
                ).load()
                for var_name, variable in month_data.variables.items():
                    if var_name not in ['time', 'level']: 
                        month_data[var_name] = month_data[var_name].astype('float32')
            # Surface variables
            else:
                if key in ['2d', '2t', '10v', '10u']:
                    var = 'd2m' if key == '2d' else 't2m' if key == '2t' else 'v10' if key == '10v' else 'u10'
                    month_data = month_data.chunk({'time' : month_data[var].shape[0] * 1}).load()
                else:
                    month_data = month_data.chunk({'time' : month_data[key].shape[0] * 1}).load()
            month_data.attrs = {}
            year_data.append(month_data)
    try:
        year_data = xr.concat(year_data, dim = 'time')
    except:
        return None
    else:
        return year_data
    
if __name__ == '__main__':

    csv, start, end = sys.argv[1:]
    start, end = int(start), int(end)

    isokeys = ['q', 't', 'u', 'v', 'w', 'z', 'r'] # ['t', 'u', 'v', 'z'] 
    sfckeys = ['tp', '2d', '2t', '10v', '10u', 'msl'] # Options: ['100u', '100v', '10u', '10v', '2d', '2t', 'blh', 'cape', 'msl', 'sp', 'tp'] ['tp']

    df = pd.read_csv(csv)

    sample = xr.open_dataset(sampledir + 'era5_sample_conus.nc')
    lats, lons = sample.latitude, sample.longitude

    # Build lat/lon array for KDTree creation
    lat_lon = np.array(
                [
                    [lat, lon] for lat in lats.values.ravel() \
                    for lon in lons.values.ravel()
                ]
             )

    # Build KDTree
    tree = KDTree(lat_lon)

    for idx, row in df.iterrows():
        lat, lon = row.lat, row.lon

        print('Creating ERA5 Profile: {}, {}, {}, {}'.format(
            lat, lon, start, end))
        
        dist, idx = tree.query(np.array([lat, lon]))
        yi, xi = np.unravel_index(idx, (lats.size, lons.size))

        # Find lat/lon
        lat = sample.isel(latitude = yi, longitude = xi)['latitude']
        lon = sample.isel(latitude = yi, longitude = xi)['longitude']

        print('ERA5 profile at gridpoint: %.2f, %.2f'%(lat, lon))
        
        for key in np.append(isokeys, sfckeys):
            try:
                filepath = mkdir_p(profdir) + 'era5prof_{:.2f}N_{:.2f}W.{:s}.{:04d}_{:04d}.nc'.format(
                    lat.values, 
                    abs(lon.values), 
                    key.upper(), 
                    start, 
                    end
                )
                if os.path.isfile(filepath):
                    print(filepath, 'exists, skipping')
                    pass
                else:
                    levelset = 'iso' if key in isokeys else 'sfc'
                    mpfunc = partial(
                        get_year, 
                        key = key, 
                        levelset = levelset, 
                        xi = xi, 
                        yi = yi
                    ) #result = get_year(2024, key, levelset, xi, yi)
                    with get_context('fork').Pool(len(np.arange(start, end + 1))) as p:
                        result = p.map(
                            mpfunc, 
                            np.arange(start, end + 1), 
                            chunksize = 1
                        )
                        p.close()
                        p.join()

                    result = [r for r in result if r is not None]
                    result = xr.concat(result, dim = 'time')
                    result.to_netcdf(
                        filepath, 
                        engine = 'h5netcdf', 
                        encoding = {var: {'zlib': True, 'complevel': 5} for var in result.data_vars}, 
                        compute = True
                    )
                    print('Saved: ', filepath)

                    result.close()
                    del result
                    gc.collect() 
            except ValueError as exc:
                print(exc)
    exit()

def agg_profiles(lats, lons, profdir):
    for lat, lon in zip(lats, lons):
        flist = glob(profdir +'disagg_2000_2024/' + '*%s*%s*2004_2024.nc'%(float(lat), abs(float(lon))))

        # Define the output file path
        savestr = 'era5prof_%.2fN_%.2fW_2004_2024.nc' % (lat, abs(lon))
        save_path = os.path.join(profdir, 'agg_2000-2024', savestr)
        print(save_path)

        if os.path.exists(save_path):
            print(f"File {save_path} already exists. Skipping.") 
            pass
        else:
            iso, sfc = [], []
            for f in flist:
                d = xr.open_dataset(f)
                if 'level' in list(d.dims):
                    print('iso variable')
                    iso.append(d.to_dataframe().reset_index().set_index(['time', 'level']).sort_index())
                    print(d.to_dataframe().reset_index().set_index(['time', 'level']).sort_index())
                else:
                    sfc.append(d.sel(expver = 1).drop('expver').to_dataframe())
                    print('sfc variable')

            isomerge = None
            for i in range(1, len(iso)):
                print('Merging isobaric variable %d/%d' % (i, len(iso) - 1))
                if isomerge is not None:
                    isomerge = isomerge.merge(iso[i], on = ['time', 'level'])
                else:
                    isomerge = iso[i - 1].merge(iso[i], on = ['time', 'level'])
            isomerge = isomerge[~isomerge.index.duplicated()]

            merge = xr.merge([isomerge.to_xarray()])
            print('merged structure = ', merge)
            print('Saving: %s' % savestr)
            merge.to_netcdf(profdir + 'agg/' + savestr)
