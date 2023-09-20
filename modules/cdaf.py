import numpy as np
from scipy.signal import argrelextrema, argrelmin, argrelmax
import matplotlib.pyplot as plt

def plot_cdaf(lmc0, *lmcs, bins=None, proportion=False, xticks=None, window=None, rotate_xticks=0, round_to=2, absol=False, show_vline=False, title=None, return_plot_values=False, ax=None):
    assert len(lmcs) == 1    
    lmc_list = []
    if not absol:
        lmc0 = np.abs(lmc0.flatten())
    lmc_list.append(lmc0)
    for lmc in lmcs:
        if not absol:
            lmc = np.abs(lmc.flatten())
        lmc_list.append(lmc)
    length = len(lmc0)

    lmc_concat = np.sort(np.concatenate(lmc_list))
    if bins is None:
        bins = np.linspace(0, lmc_concat.max()*1.01, length)

    lmc_counts = []
    for lmc in lmc_list:
        lmc_count, _ = np.histogram(lmc, bins=bins)
        lmc_counts.append(lmc_count)
    
    # fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4))
        
    # cd_diff = lmc_counts[0].cumsum() - lmc_counts[1].cumsum()
    cd_diff = (lmc_counts[0] - lmc_counts[1]).cumsum()
    cd_diff = np.concatenate([cd_diff[0:1], cd_diff])

    if proportion:
        if len(cd_diff) != length:
            cd_diff = cd_diff/len(cd_diff) * (len(cd_diff)/length)
        else:
            cd_diff = cd_diff/len(cd_diff)   
        
    ax.step(x=bins, y=cd_diff, where="post", lw=1.5, color="#348ABD")
    ax.set_xlim(0)
    ax.tick_params(axis='both', which='major', labelsize=9)  # Adjust the size of the ticks
    
    if xticks is None:
        if window is not None:
            # Find the indices of local minima and maxima
            minima = argrelmin(cd_diff, order=window)
            maxima = argrelmax(cd_diff, order=window)
            
            # Get the corresponding x-values
            xticks = np.round(np.concatenate([bins[minima], bins[maxima]]),round_to)
        else:
            xticks = ax.get_xticks()
    
    # Set the xticklabels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=rotate_xticks)
    
    ylabel = "Proportion"
    if not proportion:
        yticks = np.int64(ax.get_yticks())
        ax.set_yticks(yticks)
        ylabel = "Count"
    
    if show_vline == True:
        # Get the y-values of the step plot at the positions specified by xticks
        y_values = np.interp(xticks, bins, cd_diff)
        # Add vertical lines at the positions specified by xticks
        ax.vlines(xticks, ymin=ax.get_ylim()[0], ymax=y_values, colors='r', linestyles='dotted', lw=.75, alpha=.5)
    elif isinstance(show_vline, list):
        y_values = np.interp(show_vline, bins, cd_diff)
        ax.vlines(show_vline, ymin=ax.get_ylim()[0], ymax=y_values, colors='r', linestyles='dotted', lw=.75, alpha=.65)
    
    # ax.set_ylim(ax.get_ylim())
    
    _, xmax = ax.get_xlim()
    ax.hlines(0, 0, xmax, ls="--", colors="black")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("LMC value", fontsize=9)
    ax.set_title(title, fontsize=10)
    
    if return_plot_values:
        return bins, lmc_counts, np.sort(xticks)
