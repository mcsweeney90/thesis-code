#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Critical path approximation methods for path identification in CPOP.
TODO.
"""

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpl_patches
from itertools import product

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
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 12
#plt.rcParams["figure.figsize"] = (9.6,4)
plt.ioff() # Don't show plots.
# print(plt.rcParams['axes.prop_cycle'].by_key()['color'])

####################################################################################################

summary_path = "summaries/"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "plots/"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

df = pd.read_csv('../chol.csv')

ntiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
ntasks = {5:35, 10:220, 15:680, 20:1540, 25:2925, 30:4960, 35:7770, 40:11480, 45:16215, 50:22100}
nprocessors = 20
Vs = [0.2, 1.0]
ccrs = [0.01, 0.1, 1.0, 10.0] 

avgs = ["M", "MD", "B", "SB", "W", "SW", "HM", "SHM", "GM", "SGM", "R", "D", "NC", "SD"]
bounds = ["LB", "UB"]
mc = ["MCP-A", "MCP-H"]
all_c = ["M-H", "RND-C"] + [avg + "-C" for avg in avgs] + [b + "-C" for b in bounds] + mc

# =============================================================================
# Human readable summaries.
# =============================================================================      

def summarize(data, save_dest):
    with open(save_dest, "w") as dest:
        print("COMPARISON OF CRITICAL PATH APPROXIMATION METHODS FOR PATH ASSIGNMENT.", file=dest) 
        print("CHOLESKY DAG SET.", file=dest) 
        print("STANDARD HEFT UPWARD RANKS USED FOR PRIORITIES IN ALL CASES.", file=dest) 
        
        bests = data.loc[:, all_c].min(axis=1)            
        for method in all_c:
            print("\n\n\n---------------------------------", file=dest)
            print("METHOD : {}".format(method), file=dest)
            print("---------------------------------", file=dest)
            
            if method in mc:                
                print("\nNUMBER OF OBSERVED CRITICAL PATHS", file=dest)
                print("----------------------------", file=dest)
                paths = data["{} PATHS".format(method[:-2])]
                print("MEAN: {}".format(paths.mean()), file=dest)  
                print("MOST = {}".format(paths.max()), file=dest)
                print("LEAST = {}".format(paths.min()), file=dest)
            
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
                       
            pds = 100*(data[method] - bests)/bests
            print("\nPERCENTAGE DEGRADATION (PD)", file=dest)
            print("----------------------------", file=dest)        
            print("MEAN = {}".format(pds.mean()), file=dest)
            print("WORST = {}".format(pds.max()), file=dest)
            best_occs = (pds.values == 0.0).sum()
            print("#BESTS: {}/{}".format(best_occs, data.shape[0]), file=dest)
            print("%BESTS: {}".format(100 * best_occs/data.shape[0]), file=dest)
            
            if method != "RND-C":
                imps = 100*(data["RND-C"] - data[method])/data["RND-C"] 
                print("\n% IMPROVEMENT VS RANDOM PATH ASSIGNMENT", file=dest)
                print("----------------------------", file=dest)
                print("MEAN = {}".format(imps.mean()), file=dest)
                print("BEST = {}".format(imps.max()), file=dest)
                print("WORST = {}".format(imps.min()), file=dest) 
                worse = (imps.values < 0.0).sum()
                print("#WORSE: {}/{}".format(worse, data.shape[0]), file=dest)
                print("%WORSE: {}".format(100 * worse/data.shape[0]), file=dest) 
            
            if method != "M-H":
                imps = 100*(data["M-H"] - data[method])/data["M-H"] 
                print("\n% IMPROVEMENT VS NO PATH ASSIGNMENT", file=dest)
                print("----------------------------", file=dest)
                print("MEAN = {}".format(imps.mean()), file=dest)
                print("BEST = {}".format(imps.max()), file=dest)
                print("WORST = {}".format(imps.min()), file=dest) 
                worse = (imps.values < 0.0).sum()
                print("#WORSE: {}/{}".format(worse, data.shape[0]), file=dest)
                print("%WORSE: {}".format(100 * worse/data.shape[0]), file=dest) 
            
            if method != "M-C":
                reds = 100*(data["M-C"] - data[method])/data["M-C"]
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
# loc = "{}/all.txt".format(summary_path)
# summarize(data=df, save_dest=loc)

# # By CCR.
# for b in ccrs:
#     sdf = df.loc[(df['CCR'] == b)]  
#     loc = "{}/b{}.txt".format(summary_path, b)
#     summarize(data=sdf, save_dest=loc)
    
# # By V.
# for V in Vs:
#     sdf = df.loc[(df['vproc'] == V) & (df['vrel'] == V) & (df['vband'] == V) ]  
#     loc = "{}/V{}.txt".format(summary_path, V)
#     summarize(data=sdf, save_dest=loc)

# for V, b in product(Vs, ccrs):
#     sdf = df.loc[(df['CCR'] == b) & (df['vproc'] == V) & (df['vrel'] == V) & (df['vband'] == V) ] 
#     loc = "{}/V{}_b{}.txt".format(summary_path, V, b)
#     summarize(data=sdf, save_dest=loc)    

# # By number of tiles.
# for N in ntiles:
#     sdf = df.loc[(df['nt'] == N)]  
#     loc = "{}/N{}.txt".format(summary_path, N)
#     summarize(data=sdf, save_dest=loc)
    
# =============================================================================
# Plots.
# =============================================================================

clean = {0.01 : "001", 0.1 : "01", 1.0 : "1", 10.0 : "10", 0.2 : "02", 0.5 : "05", 0.9 : "09"}
remove = lambda s : s[:-2] if s not in mc+["EFT"] else s
namechange = lambda s : "EFT" if s == "M-H" else s

####################################################################
# 1. MPD/best occurrences.
####################################################################

def plot_mpd(data, name, ylabel=True):
    """Plot MPD instances."""
    mpd_plot_path = "{}/mpd/".format(plot_path)
    pathlib.Path(mpd_plot_path).mkdir(parents=True, exist_ok=True)
    
    bests = data.loc[:, all_c].min(axis=1)     
    mpd = {namechange(method) : (100*(data[method] - bests)/bests).mean() for method in all_c}  
    # Sort methods to identify three best.
    sort = sorted(mpd, key=mpd.get)[:3]
    
    x = np.arange(len(all_c))  
    colors = ['#988ED5'] + ['#8EBA42'] + len(avgs) * ['#E24A33'] + 2 * ['#FBC15E'] + len(mc) * ['#348ABD']  
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
    xlabels = ["EFT", "RND"] + avgs + bounds + mc
    ax.set_xticklabels(xlabels, rotation=90)
    ax.tick_params(axis='x', which='minor', bottom=False)       
    fig.tight_layout()    
    plt.savefig('{}/assign_chol_mpd_{}'.format(mpd_plot_path, name), bbox_inches='tight') 
    plt.close(fig)      

# All data.
# plot_mpd(data=df, name="all")

# # By CCR.
# for b in ccrs:
#     sdf = df.loc[(df['CCR'] == b)]  
#     plot_mpd(data=sdf, name="b{}".format(clean[b]))
    
# # By V.
# for V in Vs:
#     sdf = df.loc[(df['vproc'] == V) & (df['vrel'] == V) & (df['vband'] == V) ]  
#     plot_mpd(data=sdf, name="V{}".format(clean[V]))

for V, b in product(Vs, ccrs):
    sdf = df.loc[(df['CCR'] == b) & (df['vproc'] == V) & (df['vrel'] == V) & (df['vband'] == V) ] 
    y = True if V == 0.2 else False
    plot_mpd(data=sdf, name="V{}_b{}".format(clean[V], clean[b]), ylabel=y)

# By number of tiles.
# for N in ntiles:
#     sdf = df.loc[(df['nt'] == N)]  
#     plot_mpd(data=sdf, name="N{}".format(N))

