#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate summaries and plots of autopsy method data for STG graphs.
"""

import pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import pandas as pd
import numpy as np
from itertools import product

####################################################################################################

# Set some parameters for plots.
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
plt.ioff() # Don't show plots.

####################################################################################################

size = 1000
df = pd.read_csv('results{}.csv'.format(size))

summary_path = "summaries/n{}".format(size)
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "plots/n{}".format(size)
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

ccrs = [0.01, 0.1, 1.0, 10.0]
r = 32
ngpus = [1, 4]

# =============================================================================
# Human readable summaries.
# =============================================================================       

def summarize(data, save_dest):
    with open(save_dest, "w") as dest:
        print("EVALUATION OF AUTOPSY METHOD FOR HEFT SCHEDULES.", file=dest) 
        
        reds = 100*(data["HEFT"] - data["AUT"])/data["HEFT"]
        print("\nREDUCTION VS ORIGINAL HEFT SCHEDULE", file=dest)
        print("--------------", file=dest)
        print("MEAN = {}%".format(reds.mean()), file=dest)
        print("BEST = {}%".format(reds.max()), file=dest)
        print("WORST = {}%".format(reds.min()), file=dest) 
        better = (reds.values > 0.0).sum()
        print("#BETTER: {}/{}".format(better, data.shape[0]), file=dest)
        print("%BETTER: {}".format(100 * better/data.shape[0]), file=dest)
        alag = (reds.values >= 0.0).sum()
        print("#AT LEAST AS GOOD: {}/{}".format(alag, data.shape[0]), file=dest)
        print("%AT LEAST AS GOOD: {}".format(100 * alag/data.shape[0]), file=dest)
        
        print("\nFAILURES", file=dest)
        print("--------------", file=dest)
        heft_speedups = data["MST"] / data["HEFT"]
        heft_failures = (heft_speedups.values < 1.0).sum()
        print("% HEFT FAILURES: {}".format(100 * heft_failures/data.shape[0]), file=dest)
        aut_speedups = data["MST"] / data["AUT"]
        aut_failures = (aut_speedups.values < 1.0).sum()
        print("% AUTOPSY FAILURES: {}".format(100 * aut_failures/data.shape[0]), file=dest)
        print("% IMPROVEMENT: {}".format(100 * heft_failures/data.shape[0] - 100 * aut_failures/data.shape[0]), file=dest)
        
        print("\nADDITIONAL RUNTIME (% OF HEFT RUNTIME)", file=dest)
        print("--------------", file=dest)
        print("MEAN = {} %".format(data["AUT PTI"].mean()), file=dest)
        print("BEST = {} %".format(data["AUT PTI"].min()), file=dest)
        print("WORST = {} %".format(data["AUT PTI"].max()), file=dest) 

# # All data.
loc = "{}/all.txt".format(summary_path)
summarize(data=df, save_dest=loc)

# By platform.
for s in ngpus:
    sdf = df.loc[(df['s'] == s)]  
    loc = "{}/s{}.txt".format(summary_path, s)
    summarize(data=sdf, save_dest=loc)
    
# By CCR.
for b in ccrs:
    sdf = df.loc[(df['CCR'] == b)]   
    loc = "{}/b{}.txt".format(summary_path, b)
    summarize(data=sdf, save_dest=loc)

# By platform and CCR.  
for s, b in product(ngpus, ccrs):
    sdf = df.loc[(df['s'] == s) & (df['CCR'] == b)]   
    loc = "{}/s{}_b{}.txt".format(summary_path, s, b)
    summarize(data=sdf, save_dest=loc)
        
# =============================================================================
# Plots.
# =============================================================================

format_ccr = {0.01 : "001", 0.1 : "01", 1.0 : "1", 10.0 : "10"} # won't take e.g., ccr0.01 as name b/c of decimal point.

def plot_reductions(data, name, ylabel=True, dotcolor='#988ED5'):
    reds = 100*(data["HEFT"] - data["AUT"])/data["HEFT"]
    better = (reds.values > 0.0).sum()
    fig, ax = plt.subplots(dpi=400)
    plt.scatter(range(len(data)), reds, marker='o', color=dotcolor)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
    
    plt.grid(True, linestyle='-', axis='y', which='major')
    ax.set_xticks([])
    ax.tick_params(axis='y', which='minor', left=False)
    
    if ylabel:
        ax.set_ylabel("MAKESPAN REDUCTION (%)", labelpad=5)
    
    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 2
    labels = []
    labels.append('Mean : {}'.format(round(np.mean(reds), 1)))
    labels.append('Better : {}%'.format(round(100 * better/data.shape[0], 1)))
    ax.legend(handles, labels, handlelength=0, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white', framealpha=1.0)
    
    plt.savefig('{}/aut_stg_{}'.format(plot_path, name), bbox_inches='tight') 
    plt.close(fig)     

# Plot makespan reduction by number of GPUs and CCR.
for s, b in product(ngpus, ccrs):
    sdf = df.loc[(df['s'] == s) & (df['CCR'] == b)]  
    y = True if s == 1 else False
    col = '#988ED5' if s == 1 else '#8EBA42'
    plot_reductions(data=sdf, name="s{}_b{}".format(s, format_ccr[b]), ylabel=y, dotcolor=col)
    


    
