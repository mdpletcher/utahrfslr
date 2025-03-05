"""
Michael Pletcher, Jim Steenburgh
Created: 02/07/2025
Edited:

##### Summary #####
.py script containing functions to postprocess HRRR profile
data. The original HRRR data was preprocessed from .grib2
to .csv and .pickle using xarray, pandas, and parallel 
computing packages.

########## Function List ##########



















"""

# Imports
import numpy as np
import pandas as pd
import xarray as xr

from scipy.ndimage import label, find_objects
from scipy.spatial import KDTree
from scipy.interplate import griddata

def fill_small_areas_with_surrounding(data, category, max_area_size):
    """
    Smooths areas with spurious data using surrounding data

    Parameters:
    data (xr.DataArray) : 2-d array containing category numbers for each
    snow climate
    category (int) : Value representing snow climate
    max_area_size: (int) : Maximum number of grid points that is allowed
    for an area to be filled

    Returns:
    xr.DataArray
        2-d array containing modified category numbers for each snow climate
    """

    mask = data == category
    labeled_array, num_features = label(mask)

    for i in range(1, num_features + 1):
        slices = find_objects(labeled_array == i)
        if not slices:
            continue
        
        slice_x, slice_y = slices[0]
        roi = labeled_array[slice_x, slice_y]

        # If the region of interest is smaller than the max_area_size,
        # fill with the most frequent surrounding value
        if np.sum(roi == i) < max_area_size:
            data_roi = data[slice_x, slice_y]
            component_mask = roi == i

            # Find surrounding values
            surrounding_values = []
            for x, y in zip(*np.where(component_mask)):
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < data_roi.shape[0] and 0 <= ny < data_roi.shape[1]:
                        if not component_mask[nx, ny]:
                            surrounding_values.append(data_roi[nx, ny])
            # Determine the most frequent surrounding value
            if surrounding_values:
                fill_val = max(set(surrounding_values), key = surrounding_values.count)
                data_roi[component_mask] = fill_value
    
    return data

def points_vals(lons, lats, vals):
    points, values = np.array([lons.flatten(), lats.flatten()]).T, vals.values.flatten()
    return points, values

def mesh_grid(lon, lat):
    
    lon_dim = [dim for dim in lon.dims if 'lon' in dim or 'x' in dim][0]
    lat_dim = [dim for dim in lat.dims if 'lat' in dim or 'y' in dim][0]

    lon_mesh, lat_mesh = np.meshgrid(
        np.linspace(lon.min(), lon.max(), lon.sizes[lon_dim]),
        np.linspace(lat.min(), lat.max(), lat.sizes[lat_dim])
    )
    
    return lon_mesh, lat_mesh

def interp_2d_grid(
    low_res_xy,
    low_res_data,
    high_res_xy,
    interp_method = 'griddata'
):
    """
    Interpolate lower resolution 2-d gridded data to a higher
    resolution 2-d grid

    Parameters:
    low_res_xy : np.ndarray (Nx2)
        List of lower resolution longtide-latitude coordinate pairs, 
        where each row represents a (lon, lat) point
    low_res_data : np.ndarray (N,)
        List of flattened lower resolution data to be interpolated
    high_res_xy : tuple ((M, K), (M, K))
        2-d mesh grid of high resolution longitude and latitude values

    Returns:
    xr.DataArray
        2-d grid of interpolated higher resolution values
    """

    high_res_data = griddata(low_res_xy, low_res_data, high_res_xy, method = 'linear')
    interp_data = xr.DataArray(
        high_res_data,
        coords = {
            'lat' : (('y', 'x'), high_res_xy[1]),
            'lon' : (('y', 'x'), high_res_xy[0])
        },
        dims = ['y', 'x']
    )
    return interp_data

def create_climate_kdtree(climate_data, nohrsc_data, baxter_data):
    """
    Create kd-tree from processed climate data, NOHRSC snowfall
    data, and Baxter SLR data. The kd-tree is used to assign
    snow climates to each CoCoRAHS site using a lat/lon query

    Parameters:
    climate_data : xr.DataArray
        Processed snow climate categories
    nohrsc_data : xr.DataArray
        Processed NOHRSC mean annual snowfall
    baxter_data : xr.DataArray
        Processed Baxter mean SLR
    """
    climate_df = pd.DataFrame(
        {
            'climate' : climate_data.values.flatten(),
            'lat' : climate_data.values.flatten(),
            'lon' : climate_data.values.flatten(),
            'mean_annual_nohrsc_snow' : mean_snow.values.flatten(),
            'mean_baxter_slr' : baxter_data.values.flatten()
        }
    )
    climate_df = climate_df.dropna()
    kdtree = KDTree(climate_df[['lat', 'lon']])

    return kdtree



    
    
