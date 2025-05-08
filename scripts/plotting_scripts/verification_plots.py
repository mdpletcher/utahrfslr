"""
Michael Pletcher
Created: 03/07/2025
Edited: 05/01/2025

Plots used for verifying models and forecasts
"""

import numpy as np

from copy import copy
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
from sklearn.metrics import r2_score


def hist_2d(
    predicted_vals, 
    observed_vals, 
    metrics,
    max_value = 40,
    bin_width = 1,
    levels = None,
    title = None, 
    cmap = 'plasma',
    plot_medians = False,
    cbar = False
):

    """
    2-d histogram plot for verification

    Params:
    predicted_vals : pd.Series (1-d)
        Predicted values to be plotted
    observed_vals : pd.Series (1-d)
        Observed values to be plotted
    metrics : dict of np.array
        Performance metrics obtained from train-test
        iterations
    max_value : int, optional
        Integer used to determine axis properties
    bin_width : int or float, optional
        Width of histogram bins
    levels : list, optional
        Levels used for histogram colormap normalization
    title : str, optional
        Title of plot
    cmap : str, optional
        Colormap used for histogram heatmap
    plot_medians : bool, optional
        If True, plots bin medians for observed and forecast values
    cbar : bool, optional
        If True, displays color bar that indicates the number
        of events per bin
    
    Returns:
        None

    """

    # Verificaton metrics (mean absolute error,
    # coefficient of determination, root mean 
    # squared error)
    mae = np.nanmean(abs(predicted_vals - observed_vals))
    r_squared = r2_score(observed_vals, predicted_vals)
    rmse = np.sqrt(np.nanmean((observed_vals - predicted_vals) ** 2))

    fig, ax = plt.subplots(1, 1, figsize = (7.5, 7.5))
    bins = np.arange(0, max_value, bin_width)
    heatmap, xedges, yedges = np.histogram2d(observed_vals, predicted_vals, bins = bins)

    cmap = copy(plt.get_cmap(cmap))
    cmap.set_under('w', 1)

    levels = [0, 1, 5, 10, 20, 30, 50, 75, 100]
    if levels is not None:
        levels = levels
    levels[0] = 1e-5
    norm = BoundaryNorm(levels, ncolors = cmap.N, clip = True)

    # Plot bin medians for predicted and observed
    if plot_medians:
        fcst_label_bool = False
        obs_label_bool = False
        for i, rowbin in enumerate(bins[1:]):
            fc_med = np.nanmedian(
                np.where(
                    (observed_vals > rowbin - 2.5) & (observed_vals <= rowbin), 
                    predicted_vals, 
                    np.nan
                ).astype(float)
            )
            ob_med = np.nanmedian(
                np.where(
                    (predicted_vals > rowbin - 2.5) & (predicted_vals <= rowbin), 
                    observed_vals, 
                    np.nan
                    ).astype(float)
            )
            
            no_obs = np.where(
                (observed_vals > rowbin - 2.5) & (observed_vals <= rowbin), 
                predicted_vals, 
                np.nan
            ).astype(float)
            no_obs = no_obs[~np.isnan(no_obs)]
            
            if len(no_obs) >= 10:
                if ~np.isnan(fc_med):
                    ax.scatter(
                        np.searchsorted(bins, fc_med) - 0.5, 
                        i + 0.5, 
                        zorder = 100, 
                        c = 'white', 
                        s = 22.5,
                        edgecolor = 'k',
                        marker = 'o',
                        label = 'Forecast Median' if not fcst_label_bool else ""
                    )
                    fcst_label_bool = True
                if ~np.isnan(ob_med):
                    ax.scatter(
                        i + 0.5, 
                        np.searchsorted(bins, ob_med) + 0.5, 
                        zorder = 100, 
                        c = 'gray', 
                        s = 22.5, 
                        edgecolor = 'k',
                        marker = 'o',
                        label = 'Observed Median' if not obs_label_bool else ""
                    )
                    obs_label_bool = True
    
    # Plot 2-d histogram
    heatmap = np.where(heatmap > 0, heatmap, np.nan)
    hist_2d = ax.pcolormesh(heatmap, cmap = cmap, norm = norm)

    # Plot settings
    ax.set_xticks(np.arange(0, max_value + 1, 5))
    ax.set_yticks(np.arange(0, max_value + 1, 5))
    ax.axline((0, 0), (1, 1), linewidth = 0.5, color = 'k', zorder = 1)
    ax.grid(True, which = 'major', linestyle = '-', color = 'k', alpha = 0.5)
    ax.grid(True, which = 'minor', color = 'k', linestyle = '-', alpha = 0.5)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18, pad = 5)
    ax.set_aspect('equal', adjustable = 'box')
    ax.set_ylabel('Observed SLR', fontsize = 24)
    ax.set_xlabel('Predicted SLR', fontsize = 24)
    ax.set_yticklabels(np.arange(0, max_value + 1, 5))
    ax.set_xticklabels(np.arange(0, max_value + 1, 5))
    ax.set_xlim(0, max_value)
    ax.set_ylim(0, max_value)

    # Legend 
    leg = ax.legend(
        loc = 'upper right',
        prop = {'size' : 15}
    )
    leg.get_frame().set_linewidth(0.75)
    leg.get_frame().set_edgecolor("k")
    leg.get_frame().set_boxstyle('Square')

    # Plot verification statistics
    text = ax.text(
        0.665, 
        0.0325,
        '$\mathregular{R^{2}}$: %.2f\nMAE: %.2f\nRMSE: %.2f\nMBE: %.2f\nn = %.0f' % (metrics['Agg-R2'], metrics['Agg-MAE'], metrics['Agg-RMSE'], metrics['Agg-MBE'], len(predicted_vals)),
        fontsize = 24,
        transform = ax.transAxes
    )
    text.set_bbox(dict(facecolor = 'white', alpha = 0.6, pad = 4.0))

    if title is not None:
        ax.set_title(title, fontsize = 26)
    if cbar:
        cbar_ax = fig.add_axes([0.085, -0.0425, 0.95, 0.05]) # (left, bottom, width, height)
        '''
        fig.subplots_adjust(
            bottom = 0.135, 
            hspace = 0.124, 
            wspace = 0.005, 
            top = 0.977, 
            left = 0.125, 
            right = 0.935
        )
        '''
        cbar_ = fig.colorbar(
            hist_2d,
            cax = cbar_ax,
            fraction = 0.046,
            pad = 0.04,
            orientation = 'horizontal',
            extend = 'max'
        )
        cbar_.set_label(label = '# of events', size = 24)
        cbar_.ax.tick_params(labelsize = 18)

    plt.tight_layout()

'''
def performance_diagram():
'''