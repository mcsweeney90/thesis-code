#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze task prioritization data.
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

summary_path = "summaries/"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "plots/"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

df = pd.read_csv('../results.csv')

nprocessors = [2, 4, 8]
ccrs = [0.01, 0.1, 1.0, 2.0]
Rs = [0.1, 0.5, 0.9]
Vs = [0.2, 1.0]
runs = 3

avgs = ["M", "MD", "B", "SB", "W", "SW", "HM", "SHM", "GM", "SGM", "R", "D", "NC", "SD"]
bounds = ["LB", "UB"]
mc = ["EV-A", "EV-H", "CR-A", "CR-H"]
all_h = [avg + "-H" for avg in avgs] + [b + "-H" for b in bounds] + mc

# =============================================================================
# Human readable summaries.
# =============================================================================      

def summarize(data, save_dest):
    with open(save_dest, "w") as dest:
        print("COMPARISON OF CRITICAL PATH APPROXIMATION METHODS FOR TASK PRIORITIES IN HEFT.", file=dest) 
        print("STG SET.", file=dest) 
        
        bests = data.loc[:, all_h].min(axis=1)            
        for method in all_h:
            print("\n\n\n---------------------------------", file=dest)
            print("METHOD : {}".format(method), file=dest)
            print("---------------------------------", file=dest)
            
            slrs = data[method] / data["MLB"]   
            print("\nSLR", file=dest)
            print("----------------------------", file=dest)
            print("MEAN = {}".format(slrs.mean()), file=dest)
            print("BEST = {}".format(slrs.min()), file=dest)
            print("WORST = {}".format(slrs.max()), file=dest)
            optimal = (abs(slrs.values - 1.0) < 1e-6).sum()
            print("#OPTIMAL: {}/{}".format(optimal, data.shape[0]), file=dest)
            print("%OPTIMAL: {}".format(100 * optimal/data.shape[0]), file=dest) 
            
            speedups = data["MST"] / data[method]   
            print("\nSPEEDUP", file=dest)
            print("----------------------------", file=dest)
            print("MEAN = {}".format(speedups.mean()), file=dest)
            print("BEST = {}".format(speedups.max()), file=dest)
            print("WORST = {}".format(speedups.min()), file=dest)
            failures = (speedups.values < 1.0).sum()
            print("#FAILURES: {}/{}".format(failures, data.shape[0]), file=dest)
            print("%FAILURES: {}".format(100 * failures/data.shape[0]), file=dest)  
            
            imps = 100*(data["RND-H"] - data[method])/data["RND-H"] 
            print("\n% IMPROVEMENT VS RANDOM", file=dest)
            print("----------------------------", file=dest)
            print("MEAN = {}".format(imps.mean()), file=dest)
            print("BEST = {}".format(imps.max()), file=dest)
            print("WORST = {}".format(imps.min()), file=dest) 
            worse = (imps.values < 0.0).sum()
            print("#WORSE: {}/{}".format(worse, data.shape[0]), file=dest)
            print("%WORSE: {}".format(100 * worse/data.shape[0]), file=dest)              
            
            pds = 100*(data[method] - bests)/bests
            print("\nPERCENTAGE DEGRADATION (PD)", file=dest)
            print("----------------------------", file=dest)        
            print("MEAN = {}".format(pds.mean()), file=dest)
            print("WORST = {}".format(pds.max()), file=dest)
            best_occs = (pds.values == 0.0).sum()
            print("#BESTS: {}/{}".format(best_occs, data.shape[0]), file=dest)
            print("%BESTS: {}".format(100 * best_occs/data.shape[0]), file=dest)
            
            if method == "M-H":
                continue
            reds = 100*(data["M-H"] - data[method])/data["M-H"]
            print("\nREDUCTION VS DEFAULT MEAN METHOD", file=dest)
            print("----------------------------", file=dest)
            print("MEAN = {}%".format(reds.mean()), file=dest)
            print("BEST = {}%".format(reds.max()), file=dest)
            print("WORST = {}%".format(reds.min()), file=dest) 
            better = (reds.values > 0.0).sum()
            print("%BETTER: {}".format(100 * better/data.shape[0]), file=dest) 
            same = (reds.values == 0.0).sum()
            print("%SAME: {}".format(100 * same/data.shape[0]), file=dest) 
            worse = (reds.values < 0.0).sum()
            print("%WORSE: {}".format(100 * worse/data.shape[0]), file=dest) 
                            

# All data.
loc = "{}/all.txt".format(summary_path)
summarize(data=df, save_dest=loc)

# By number of processors.
for q in nprocessors:
    sdf = df.loc[(df['q'] == q)]  
    loc = "{}/q{}.txt".format(summary_path, q)
    summarize(data=sdf, save_dest=loc)

# By CCR.
for ccr in ccrs:
    sdf = df.loc[(df['mu_ccr'] == ccr)]  
    loc = "{}/ccr{}.txt".format(summary_path, ccr)
    summarize(data=sdf, save_dest=loc)
    
# By rtask.
for rtask in Rs:
    sdf = df.loc[(df['rtask'] == rtask)]  
    loc = "{}/rtask{}.txt".format(summary_path, rtask)
    summarize(data=sdf, save_dest=loc)
    
# By rmach.
for rmach in Rs:
    sdf = df.loc[(df['rmach'] == rmach)]  
    loc = "{}/rmach{}.txt".format(summary_path, rmach)
    summarize(data=sdf, save_dest=loc)
    
# By V.
for v in Vs:
    sdf = df.loc[(df['V'] == v)]  
    loc = "{}/V{}.txt".format(summary_path, v)
    summarize(data=sdf, save_dest=loc)
    
# =============================================================================
# Plots.
# =============================================================================

clean = {0.01 : "001", 0.1 : "01", 1.0 : "1", 10.0 : "10", 0.2 : "02", 0.5 : "05", 0.9 : "09", 2.0 : "2"}
remove = lambda s : s[:-2] if s not in mc else s

####################################################################
# 1. MPD/best occurrences.
####################################################################

def plot_mpd(data, name, ylabel=True):
    """Plot MPD instances."""
    mpd_plot_path = "{}/mpd/".format(plot_path)
    pathlib.Path(mpd_plot_path).mkdir(parents=True, exist_ok=True)
    
    bests = data.loc[:, all_h].min(axis=1)     
    mpd = {method : (100*(data[method] - bests)/bests).mean() for method in all_h}   
    # Sort methods to identify three best.
    sort = sorted(mpd, key=mpd.get)[:3]
    
    x = np.arange(len(all_h))  
    colors = len(avgs) * ['#E24A33'] + 2 * ['#FBC15E'] + len(mc) * ['#348ABD']  
    fig, ax = plt.subplots(dpi=400)
    rects = ax.bar(x, list(mpd.values()), color=colors)    
    plt.minorticks_on()
    plt.grid(True, linestyle='-', axis='y', which='major')
    plt.grid(True, linestyle=':', axis='y', which='minor')
    plt.grid(False, axis='x')  
    if ylabel:
        ax.set_ylabel("MEAN PERCENTAGE DEGRADATION", labelpad=5)
    
    # Get the three best and label them.
    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 3
    labels = list('{} = {}'.format(remove(m), round(mpd[m], 3)) for m in sort)    
    ax.legend(handles, labels, handlelength=0, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white', framealpha=1.0)    
    
    ax.set_xticks(x)
    xlabels = avgs + bounds + mc
    ax.set_xticklabels(xlabels, rotation=90)
    ax.tick_params(axis='x', which='minor', bottom=False)       
    fig.tight_layout()    
    plt.savefig('{}/prio_stg_mpd_{}'.format(mpd_plot_path, name), bbox_inches='tight') 
    plt.close(fig)  
    
# All data.
plot_mpd(data=df, name="all")

# By CCR.
for b in ccrs:
    sdf = df.loc[(df['mu_ccr'] == b)]  
    plot_mpd(data=sdf, name="b{}".format(clean[b]))
    
# By V.
for V in Vs:
    sdf = df.loc[(df['V'] == V) ]  
    plot_mpd(data=sdf, name="V{}".format(clean[V]))

for V, b in product(Vs, ccrs):
    sdf = df.loc[(df['mu_ccr'] == b) & (df['V'] == V)] 
    y = True if V == 0.2 else False
    plot_mpd(data=sdf, name="V{}_b{}".format(clean[V], clean[b]), ylabel=y)

#By q.
for q in nprocessors: 
    plot_mpd(data=df.loc[(df['q'] == q)], name="q{}".format(q))  

# By rtask.
for rtask in Rs:
    sdf = df.loc[(df['rtask'] == rtask)]  
    plot_mpd(data=sdf, name="rtask{}".format(clean[rtask]))
    
# By rmach.
for rmach in Rs:
    sdf = df.loc[(df['rmach'] == rmach)]  
    plot_mpd(data=sdf, name="rmach{}".format(clean[rmach]))



