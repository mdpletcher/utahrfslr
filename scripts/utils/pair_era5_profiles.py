"""
Michael Pletcher
Created: 09/03/2024
Edited: 06/11/2025

.py script for pairing ERA5 profiles with
observations for training SLR ML models

"""

import numpy as np
import xarray as xr
import pandas as pd
import os
import warnings
import pytz
import argparse

from glob import glob
from datetime import datetime, timedelta, timezone
from scipy.spatial import cKDTree
from timezonefinder import TimezoneFinder

warnings.filterwarnings('ignore')

def local_to_utc(row):
    local_timezone = pytz.timezone(row['time_zone'])
    local_time = local_timezone.localize(row['time'])
    return local_time.astimezone(pytz.utc)

def build_era5_kdtree(era5_sample):
    lats, lons = era5_sample.latitude, era5_sample.longitude
    lat_lon = np.array(
        [
            [lat, lon] for lat in lats.values.ravel() for lon in lons.values.ravel()
        ]
    )
    return cKDTree(lat_lon)

def load_site_data(site, base_dir):
    path = glob(os.path.join(base_dir, "%s*.pd" % site))
    if not path:
        raise FileNotFoundError(f"No observation file found for site {site}")
    return pd.read_pickle(path[0])

def query_era5_grid_point(
    site_lat, 
    site_lon, 
    era5_sample,
    era5_latlon_kdtree
):
    dist, idx = era5_latlon_kdtree.query(np.array([site_lat, site_lon]))
    yi, xi = np.unravel_index(idx, (era5_sample.latitude.size, era5_sample.longitude.size))

    queried_era5_lat = era5_sample.isel(latitude = yi, longitude = xi)['latitude'].values
    queried_era5_lon = era5_sample.isel(latitude = yi, longitude = xi)['longitude'].values

    return queried_era5_lat, queried_era5_lon

def add_time_data(site_data):

    tf = TimezoneFinder()
    site_data['time_zone'] = site_data.apply(
        lambda row: tf.timezone_at(lat = row.lat, lng = row.lon), axis = 1
    )
    site_data['time_utc'] = site_data.apply(local_to_utc, axis = 1)
    site_data.insert(1, 'time_utc', site_data.pop('time_utc'))
    site_data.insert(2, 'time_zone', site_data.pop('time_zone'))
    site_data = site_data.reset_index().set_index('time_utc').sort_index()

    return site_data

def rename_era5_variables(era5_prof, era5_df):
    for var in era5_prof.data_vars:
        if 'level' in era5_prof[var].coords:
            for level in era5_prof.level:
                if level >= 350:
                    newname = '%s%s'%(var.upper(), int(level.values))
                    era5_df[newname] = era5_prof[var].sel(level = level)
        else:
            newname = var.upper()
            era5_df[newname] = era5_prof[var]

    return era5_df

def get_era5_interval_data(
    site_data, 
    era5_df, 
    tp_df, 
    interval
):
    # Empty lists for appending interval data
    mean_df_matched, sum_df_matched = [], []
    # Loop through all obs for a site
    for t in site_data.index:

        # Start of observing period
        t0 = t - timedelta(hours = interval - 1)

        # 24-h time window for ERA5 data
        time_window = (era5_df.index >= t0) & (era5_df.index <= t)

        # For irregular observing period lengths (e.g., for
        # an ob collected at 0815 UTC)
        if len(time_window) == 23:
            missing_idx = time_window.index[0] - timedelta(hours = 1)
            missing_row = era5_df.loc[era5_df.index == missing_idx]
            era5_df = pd.concat[missing_row, era5_df]

        # Take mean of ERA5 variable and sum TP over 24-h time window
        df_mean, df_tp_sum = era5_df.loc[time_window].mean(), tp_df.loc[time_window].sum()

        # Add obs collect time (t)
        df_mean['obs_collect_time_utc'] = t
        df_tp_sum['obs_collect_time_utc'] = t

        # Append matched data to empty lists
        mean_df_matched.append(pd.DataFrame(df_mean).T)
        sum_df_matched.append(pd.DataFrame(df_tp_sum).T)
    
    return mean_df_matched, sum_df_matched

def qc_data(data):

    # Cutoffs for QC
    min_slr, max_slr = 2, 50
    swe_cutoff, snow_cutoff = 2.794, 50.8 # mm for both

    data = data[
        (data.slr != 10) & # Remove 10 as they are sometimes placeholders
        (data.slr > min_slr) &
        (data.slr < max_slr) &
        (data.swe_mm >= swe_cutoff) &
        (data.snow_mm >= 50.8)
    ]
    data = data.dropna()

    return data

def process_site(
    site,
    metadata,
    site_data_path,
    era5_sample,
    era5_latlon_kdtree,
    datadir,
    interval
):
    # Find site metadata
    site_metadata = metadata.loc[metadata.index == site, :]
    site_elev = site_metadata.elev.values[0] 
    site_lat = site_metadata.lat.values[0]
    site_lon = site_metadata.lon.values[0]

    # Skip sites with no elevation data
    if np.isnan(site_elev) or site_elev < 0:
        print("Skipping %s due to missing or invalid elevation data" % site)
        return None
    
    # Query ERA5 grid point
    era5_lat, era5_lon = query_era5_grid_point(site_lat, site_lon, era5_sample, era5_latlon_kdtree)

    # Load site data
    base_dir = os.path.dirname(site_data_path)
    site_df = load_site_data(site, base_dir)
    site_df = add_time_data(site_df)

    site_df = site_df[[k for k in site_df.columns if str(interval) in k]].dropna(how='all')
    site_df = site_df[~site_df.index.duplicated()]
    site_df.index = site_df.index.tz_convert('UTC')

    try:
        era5_prof = xr.open_dataset(
            datadir + "era5/profiles/agg_2000-2024/" + 'era5prof_%.2fN_%.2fW_2000_2024.nc' % (era5_lat, abs(era5_lon))
        ).load()
    except Exception as e:
        print('Unable to open ERA5 profile, missing data: ', e)

    # Prepare ERA5 dataframe
    era5_df = rename_era5_variables(era5_prof, pd.DataFrame())
    era5_df['time'] = era5_prof.time
    era5_df = era5_df.reset_index().set_index('time').sort_index().dropna()
    era5_df.index = era5_df.index.tz_localize('UTC')

    # Create separate dataframe for total precipitation
    tp_df = pd.DataFrame(era5_df.pop('TP'))
    tp_df.rename(columns = {'TP' : 'swe_mm_model'}, inplace = True)

    # Only include site obs for when ERA5 data is available
    site_df = site_df.iloc[np.where((site_df.index > era5_df.index[0]) & (site_df.index < era5_df.index[-1]))]

    # Get hourly ERA5 data for each observing period based on interval
    era5_matched, tp_matched = get_era5_interval_data(site_df, era5_df, tp_df, interval = 24)

    # Combine data
    concat_era5 = pd.concat(era5_matched).set_index('obs_collect_time_utc').sort_index()
    concat_tp = pd.concat(tp_matched).set_index('obs_collect_time_utc').sort_index()
    matched_df = pd.concat(
        [concat_tp, concat_era5],
        axis = 1,
        join = 'inner'
    ).rename(columns = {"QPF" : "swe_mm_model"})

    # Time match if needed
    matched_df = matched_df[np.isin(matched_df.index, site_df.index)]
    site_df = site_df[np.isin(site_df.index, matched_df.index)]

    # Add obs to ERA5 dataframe
    for k in site_df.keys():
        matched_df.insert(0, k.replace('%d' % interval, ''), site_df[k])

    # Add site metadata
    matched_df.insert(0, 'site', site)
    matched_df['site_elev'], matched_df['interval'] = site_elev, interval
    matched_df['era5_lat'], matched_df['era5_lon'], = era5_lat, era5_lon
    matched_df['site_lat'], matched_df['site_lon'] = site_lat, site_lon

    #print(matched_df)

    return matched_df

def main():

    datadir = "/uufs/chpc.utah.edu/common/home/steenburgh-group10/mpletcher/SLR_CONUS/data/"

    # Load ERA5 sample data
    era5_sample = xr.open_dataset(datadir + 'era5/era5_orog_sample_conus.nc')

    # Add parser for site metadata and observations
    parser = argparse.ArgumentParser(description = ".csv file with metadata and .pickle files with site obs")
    parser.add_argument("csv", help = "Path to .csv file with site metadata")
    parser.add_argument("site_data", help = "Path to .pickle file(s) with site obs")
    args = parser.parse_args()
    csv = args.csv
    site_data_path = args.site_data

    # Create list of sites based on available data
    site_data_paths = glob(site_data_path + '/*.pd')
    site_list = [f.split('/')[-1].split('_')[0] for f in site_data_paths]
    
    # Read in metadata
    metadata = pd.read_csv(csv)
    metadata = metadata[np.isin(metadata.code, site_list)].rename(columns = {'code': 'site'}).set_index('site')

    # Create KDTree for finding nearest ERA5 grid point to site
    era5_latlon_kdtree = build_era5_kdtree(era5_sample)

    interval = 24 # Observation interval
    all_data = []

    # Loop through all sites
    for site in site_list:
        result = process_site(
            site, metadata, args.site_data, era5_sample, era5_latlon_kdtree, datadir, interval
        )
        if result is not None:
            all_data.append(result)

    full_df = pd.concat(all_data).reset_index()
    # Apply QC
    full_df = qc_data(full_df)

    return full_df

if __name__ == '__main__':
    main()