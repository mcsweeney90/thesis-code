#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of STG results.
"""

import pathlib, dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.patches as mpl_patches
from math import sqrt
from itertools import product
from scipy.stats import skew, kurtosis, kstest, ks_2samp

####################################################################################################

# Set some parameters for plots.
# See here: http://www.futurile.net/2016/02/27/matplotlib-beautiful-plots-with-style/
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'bold' 
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['axes.titlepad'] = 0
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['lines.markersize'] = 3
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 12
#plt.rcParams["figure.figsize"] = (9.6,4)
plt.ioff() # Don't show plots.
# print(plt.rcParams['axes.prop_cycle'].by_key()['color'])

####################################################################################################

df = pd.read_csv('stg.csv')

summary_path = "summaries/rpm"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "plots/rpm"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

covs = [0.01, 0.03, 0.1, 0.3]
str_covs = ["001", "003", "01", "03"]
methods = ["CPM", "K LOWER", "K UPPER", "SCULLI", "CORLCA", "MC10", "SIM10", "DOM10", "MC100", "SIM100", "DOM100"] 

####################################################################################################

# =============================================================================
# Summaries.
# =============================================================================

def summarize(data, name):
    with open("{}/rpm_{}.txt".format(summary_path, name), "w") as dest:
        print("SUMMARY OF CLT AND RPM HEURISTICS.", file=dest)  
        for stat in ["MU", "VAR"]:                            
            print("\n\n\n------------------------------------------------------------------", file=dest)
            print("% RELATIVE ERROR IN {} (AVG, MAX)".format(stat), file=dest)
            print("------------------------------------------------------------------", file=dest)            
            for method in methods:
                if method == "CPM" and stat == "VAR":
                    continue               
                err = 100*abs(data["{} {}".format(method, stat)] - data["REF {}".format(stat)]) / data["REF {}".format(stat)]
                print("{}: ({}, {})".format(method, err.mean(), err.max()), file=dest)      
                
        print("\n\n\n------------------------------------------------------------------", file=dest)
        print("KS STATISTICS (AVG, MAX)", file=dest)
        print("------------------------------------------------------------------", file=dest)   
        for method in methods[3:]:
            print("{}: ({}, {})".format(method, data["{} KS".format(method)].mean(), data["{} KS".format(method)].max()), file=dest)
        
summarize(df, name="all")
for cov in covs:
    sdf = df.loc[(df['COV'] == cov)] 
    summarize(sdf, name="cov{}".format(cov))

# =============================================================================
# Plots.
# =============================================================================
    
# KS statistics.
def mean_hist(data, name, col, ylabel=None):
    means = {method: data["{} {}".format(method, col)].mean() for method in methods[4:]}
    x = np.arange(len(means))      
    # Sort methods to identify three best.
    # sort = sorted(means, key=means.get)[:3]    
    
    colors = ['#348ABD', '#988ED5', '#E24A33', '#8EBA42', '#FBC15E', '#E24A33', '#8EBA42'] 
    fig, ax = plt.subplots(dpi=400)
    rects = ax.bar(x, list(means.values()), color=colors)
    ax.bar_label(rects, labels=list(round(p, 3) for p in means.values()), label_type='edge')
    
    plt.minorticks_on()
    plt.grid(True, linestyle='-', axis='y', which='major')
    plt.grid(True, linestyle=':', axis='y', which='minor')
    plt.grid(False, axis='x')  
    
    # # Get the three best and label them.
    # handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 3
    # labels = list('{} = {}'.format(m, round(means[m], 3)) for m in sort)    
    # ax.legend(handles, labels, handlelength=0, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white', framealpha=1.0) 
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=5)
    ax.set_xticks(x)
    xlabels = methods[4:]
    ax.set_xticklabels(methods[4:], rotation=90)     
    ax.tick_params(axis='x', which='minor', bottom=False) 
    fig.tight_layout()    
    plt.savefig('{}/stg_rpm_{}_{}'.format(plot_path, col, name), bbox_inches='tight') 
    plt.close(fig)  
    

# mean_hist(data=df, name="all", col="KS", ylabel="MEAN KS STATISTIC")
# for cov, name in zip(covs, str_covs):
#     sdf = df.loc[(df['COV'] == cov)] 
#     y = "MEAN KS STATISTIC" if cov in [0.01, 0.1] else None
#     mean_hist(sdf, name="cov{}".format(name), col="KS", ylabel=y)
# mean_hist(data=df, name="all", col="TIME", ylabel="MEAN TIME (SECONDS)")
