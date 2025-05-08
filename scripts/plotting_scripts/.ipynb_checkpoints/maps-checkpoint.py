"""
Michael Pletcher
Created: 03/05/2025
Edited:

Various functions used to plot data on maps
"""

import sys
import matplotlib.pyplot as plt
import cartopy.feature as cfeature

from cartopy.io.img_tiles import GoogleTiles
from metpy.plots import USCOUNTIES



class ShadedReliefESRI(GoogleTiles):
    """
    Creates topography overlay for the contiguous United States
    """

    def _image_url(self, tile):

        """
        Creates URL based on desired map background terrain color
        """

        map_background = 'hillshade'  
        x, y, z = tile
        if map_background == 'topo':
            url = (
                'https://a.tile.opentopomap.org/{z}/{x}/{y}.png'
            ).format(z = z, y = y, x = x)
        elif map_background == 'hillshade':
            url = (
                'https://services.arcgisonline.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}.jpg'
            ).format(z = z, y = y, x = x)
        elif map_background == 'hillshade-dark':
            url = (
                'https://services.arcgisonline.com/arcgis/rest/services/Elevation/World_Hillshade_Dark/MapServer/tile/{z}/{y}/{x}.jpg'
            ).format(z = z, y = y, x = x)
        else: 
            sys.exit()
        return url

def conus_map(
    counties_overlay = False,
    topo_overlay = False,
    map_detail = 6,
    ax = None,
    extent = None
):
    """
    Creates a map of the contiguous United States with
    state borders. Can be modified to make a map of anywhere
    else using the arg extent.

    Params:
    counties_overlay : bool
        Switch for adding county borders
    topo_overlay : bool
        Switch for adding terrain overlay
    map_detail : int
        Integer (1 - 12) for determining terrain detail. Higher
        is finer resolution
    ax : NoneType
    extent : NoneType or list of float
        optional arg that defines extent of the map

    Returns:
    tuple of fig, ax
        fig (matplotlib.figure.Figure) and ax (matplotlib.axes._subplots.AxesSubplot)
    """

    if ax is None:
        fig, ax = plt.figure(figsize = (20, 20)), plt.axes(projection = ShadedReliefESRI().crs)
    else:
        fig = ax.figure
    
    if extent is not None:
        ax.set_extext(extent)
    else:
        ax.set_extent([-125.25, -66.5, 24.75, 45.5])
    
    ax.add_feature(
        cfeature.STATES.with_scale('10m'),
        linewidth = 0.75,
        edgecolor = 'w',
        zorder = 2
    )
    ax.add_feature(
        cfeature.LAKES.with_scale('10m'),
        edgecolor = 'none',
        facecolor = 'silver',
        zorder = 2
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            'physical', 
            'ocean', 
            scale = '10m', 
            edgecolor = 'none', 
            facecolor = 'silver'
        ),
        zorder = 0
    )

    if counties_overlay:
        ax.add_feature(
            USCOUNTIES.with_scale('500k'), 
            linewidth = 0.5, 
            edgecolor = 'black', 
            zorder = 4
        )
    if topo_overlay:
        ax.add_image(ShadedReliefESRI(), map_detail)

    return fig, ax