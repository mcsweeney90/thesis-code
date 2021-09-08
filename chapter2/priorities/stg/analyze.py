#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summaries and plots for the STG data.
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
plt.rcParams['legend.fontsize'] = 10
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
runs = 10
u_avgs = ["M", "MD", "B", "SB", "W", "SW", "HM", "SHM", "GM", "SGM", "R", "D", "NC", "SD"]
o_avgs = ["M", "MD", "B", "W", "HM", "GM", "R", "D", "NC", "SD"]
all_avgs = ["{}-U".format(avg) for avg in u_avgs] + ["{}-O".format(avg) for avg in o_avgs]

# =============================================================================
# Human readable summaries.
# =============================================================================       

def summarize(data, save_dest):
    with open(save_dest, "w") as dest:
        print("COMPARISON OF RANKING PHASES IN PRIORITY-BASED FRAMEWORK.", file=dest) 
        print("SELECTION PHASE EFT/GREEDY IN ALL CASES.", file=dest) 
        
        bests = data.loc[:, all_avgs].min(axis=1)            
        for avg in all_avgs:
            print("\n\n\n---------------------------------", file=dest)
            print("AVERAGE TYPE : {}".format(avg), file=dest)
            print("---------------------------------", file=dest)
            
            slrs = data[avg] / data["MLB"]   
            print("\nSLR", file=dest)
            print("--------------", file=dest)
            print("MEAN = {}".format(slrs.mean()), file=dest)
            print("BEST = {}".format(slrs.min()), file=dest)
            print("WORST = {}".format(slrs.max()), file=dest)
            optimal = (abs(slrs.values - 1.0) < 1e-6).sum()
            print("#OPTIMAL: {}/{}".format(optimal, data.shape[0]), file=dest)
            print("%OPTIMAL: {}".format(100 * optimal/data.shape[0]), file=dest)  
            
            speedups = data["MST"] / data[avg]   
            print("\nSPEEDUP", file=dest)
            print("--------------", file=dest)
            print("MEAN = {}".format(speedups.mean()), file=dest)
            print("BEST = {}".format(speedups.max()), file=dest)
            print("WORST = {}".format(speedups.min()), file=dest)
            failures = (speedups.values < 1.0).sum()
            print("#FAILURES: {}/{}".format(failures, data.shape[0]), file=dest)
            print("%FAILURES: {}".format(100 * failures/data.shape[0]), file=dest)
            
            imps = 100*(data["RND"] - data[avg])/data["RND"] 
            print("\nIMPROVEMENT VS RANDOM", file=dest)
            print("--------------", file=dest)
            print("MEAN = {}%".format(imps.mean()), file=dest)
            print("BEST = {}%".format(imps.max()), file=dest)
            print("WORST = {}%".format(imps.min()), file=dest)   
            worse = (imps.values < 0.0).sum()
            print("#WORSE: {}/{}".format(worse, data.shape[0]), file=dest)
            print("%WORSE: {}".format(100 * worse/data.shape[0]), file=dest)             
            
            pds = 100*(data[avg] - bests)/bests
            print("\nPERCENTAGE DEGRADATION (PD)", file=dest)
            print("--------------", file=dest)        
            print("MEAN = {}".format(pds.mean()), file=dest)
            print("WORST = {}".format(pds.max()), file=dest)
            best_occs = (pds.values == 0.0).sum()
            print("#BESTS: {}/{}".format(best_occs, data.shape[0]), file=dest)
            print("%BESTS: {}".format(100 * best_occs/data.shape[0]), file=dest)
            
            if avg == "M-U":
                continue
            reds = 100*(data["M-U"] - data[avg])/data["M-U"]
            print("\nREDUCTION VS DEFAULT M-U RANKING", file=dest)
            print("--------------", file=dest)
            print("MEAN = {}%".format(reds.mean()), file=dest)
            print("BEST = {}%".format(reds.max()), file=dest)
            print("WORST = {}%".format(reds.min()), file=dest) 
            alag = (reds.values >= 0.0).sum()
            print("#AT LEAST AS GOOD: {}/{}".format(alag, data.shape[0]), file=dest)
            print("%AT LEAST AS GOOD: {}".format(100 * alag/data.shape[0]), file=dest)

# All data.
loc = "{}/all.txt".format(summary_path)
summarize(data=df, save_dest=loc)

# By platform.
for s in ngpus:
    sdf = df.loc[(df['s'] == s)]  
    loc = "{}/s{}.txt".format(summary_path, s)
    summarize(data=sdf, save_dest=loc)
    
# By CCR.
for ccr in ccrs:
    sdf = df.loc[(df['CCR'] == ccr)]   
    loc = "{}/ccr{}.txt".format(summary_path, ccr)
    summarize(data=sdf, save_dest=loc)

# By platform and CCR.  
for s in ngpus:
    for ccr in ccrs:
        sdf = df.loc[(df['s'] == s) & (df['CCR'] == ccr)]   
        loc = "{}/s{}_b{}.txt".format(summary_path, s, ccr)
        summarize(data=sdf, save_dest=loc)
        
# =============================================================================
# Plots.
# =============================================================================

format_ccr = {0.01 : "001", 0.1 : "01", 1.0 : "1", 10.0 : "10"} # won't take e.g., ccr0.01 as name b/c of decimal point.

def add_patch(legend, mpd, sort):
    ax = legend.axes

    handles, labels = ax.get_legend_handles_labels()
    handles += [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 3
    labels += list('{} = {}'.format(m, round(mpd[m], 3)) for m in sort) 

    legend._legend_box = None
    legend._init_legend_box(handles, labels)
    legend._set_loc(legend._loc)
    legend.set_title(legend.get_title().get_text())

def plot_mpd(data, plot_name, legend=True, ylabel=True):
    """
    Plot the MPD.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    plot_name : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    mpd_plot_path = "{}/mpd/".format(plot_path)
    pathlib.Path(mpd_plot_path).mkdir(parents=True, exist_ok=True)
    
    bests = data.loc[:, all_avgs].min(axis=1) 
    u_data = list((100*(data["{}-U".format(avg)] - bests)/bests).mean() for avg in u_avgs)
    check = lambda avg : (100*(data["{}-O".format(avg)] - bests)/bests).mean() if avg not in ["SW", "SB", "SHM", "SGM"] else 0.0
    o_data = list(check(avg) for avg in u_avgs)   
    
    mpd = {method : (100*(data[method] - bests)/bests).mean() for method in all_avgs}   
    # Sort methods to identify three best.
    sort = sorted(mpd, key=mpd.get)[:3]
          
    x = np.arange(len(u_avgs))
    width = 0.4
    
    fig, ax = plt.subplots(dpi=400)
    rects1 = ax.bar(x - width/2, u_data, width, label='UPWARD RANK', color='#E24A33')
    rects2 = ax.bar(x + width/2, o_data, width, label='OPTIMISTIC COSTS', color='#348ABD')
    
    plt.minorticks_on()
    plt.grid(True, linestyle='-', axis='y', which='major')
    plt.grid(True, linestyle=':', axis='y', which='minor')
    plt.grid(False, axis='x')
    
    # Get the three best and label them.
    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 3
    labels = list('{} = {}'.format(m, round(mpd[m], 3)) for m in sort)    
    ax.legend(handles, labels, handlelength=0, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white', framealpha=1.0) 
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    if ylabel:
        ax.set_ylabel("MEAN PERCENTAGE DEGRADATION", labelpad=5)
    ax.set_xticks(x)
    ax.set_xticklabels(u_avgs)
    ax.tick_params(axis='x', which='minor', bottom=False)
    if legend:
        lgd = ax.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white')
        add_patch(lgd, mpd=mpd, sort=sort)
                
    fig.tight_layout()    
    plt.savefig('{}/prio_stg_mpd_{}'.format(mpd_plot_path, plot_name), bbox_inches='tight') 
    plt.close(fig) 
    
def plot_superior(data, plot_name, legend=True, ylabel=True):
    """
    Comparison with U-M ranking.
    """
    sup_plot_path = "{}/superior/".format(plot_path)
    pathlib.Path(sup_plot_path).mkdir(parents=True, exist_ok=True)
    
    u_data = [0.0]
    u_data += list(100*((data["M-U"] - data["{}-U".format(avg)]).values > 0.0).sum()/data.shape[0] for avg in u_avgs[1:])
    check = lambda avg : 100*((data["M-U"] - data["{}-O".format(avg)]).values > 0.0).sum()/data.shape[0] if avg not in ["SW", "SB", "SHM", "SGM"] else 0.0
    o_data = list(check(avg) for avg in u_avgs)      
             
    x = np.arange(len(u_avgs))
    width = 0.4
    
    fig, ax = plt.subplots(dpi=400)
    ax.axhline(y=50, color='black', linestyle='-', linewidth=1.0)
    rects1 = ax.bar(x - width/2, u_data, width, label='UPWARD RANK', color='#E24A33')
    rects2 = ax.bar(x + width/2, o_data, width, label='OPTIMISTIC COSTS', color='#348ABD')    
    
    plt.minorticks_on()
    plt.grid(True, linestyle='-', axis='y', which='major')
    plt.grid(True, linestyle=':', axis='y', which='minor')
    plt.grid(False, axis='x')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    if ylabel:
        ax.set_ylabel("BETTER THAN M-U (%)", labelpad=5)
    ax.set_xticks(x)
    ax.set_xticklabels(u_avgs)
    ax.tick_params(axis='x', which='minor', bottom=False)
    if legend:
        ax.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white', framealpha=1.0)
                
    fig.tight_layout()    
    plt.savefig('{}/prio_stg_sup_{}'.format(sup_plot_path, plot_name), bbox_inches='tight') 
    plt.close(fig)     

# All data.
plot_mpd(data=df, plot_name="all")
plot_superior(data=df, plot_name="all")

# By platform.
for s in ngpus:
    sdf = df.loc[(df['s'] == s)]  
    leg = True if s == 1 else False
    plot_mpd(data=sdf, plot_name="s{}".format(s), legend=leg)
    plot_superior(data=sdf, plot_name="s{}".format(s), legend=leg)
        
# By CCR.
for b in ccrs:
    sdf = df.loc[(df['CCR'] == b)]   
    leg = True if b == 0.01 else False
    plot_superior(data=sdf, plot_name="b{}".format(format_ccr[b]), legend=leg)
    plot_mpd(data=sdf, plot_name="b{}".format(format_ccr[b]), legend=leg)

# By platform and tile size.  
for s, b in product(ngpus, ccrs):
    sdf = df.loc[(df['s'] == s) & (df['CCR'] == b)]  
    leg = True if (b == 0.01 and s == 1) else False
    y = True if s == 1 else False
    plot_mpd(data=sdf, plot_name="s{}_b{}".format(s, format_ccr[b]), legend=leg, ylabel=y) 
    plot_superior(data=sdf, plot_name="s{}_b{}".format(s, format_ccr[b]), legend=leg, ylabel=y) 

