#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of processor selection rules for STG.
"""

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product
from matplotlib.patches import Patch

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
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.ioff() # Don't show plots.
# print(plt.rcParams['axes.prop_cycle'].by_key()['color'])

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
all_rules = ["EFT", "NC", "OL-I", "OL-II", "GCP", "HAL"]

# =============================================================================
# Human readable summaries.
# =============================================================================       

def summarize(data, save_dest):
    with open(save_dest, "w") as dest:
        print("COMPARISON OF SELECTION RULES IN PRIORITY-BASED FRAMEWORK.", file=dest) 
        print("PRIORITIZATION PHASE M-U (AS IN HEFT) IN ALL CASES.", file=dest) 
        
        bests = data.loc[:, all_rules].min(axis=1)            
        for rule in all_rules:
            print("\n\n\n---------------------------------", file=dest)
            print("SELECTION RULE : {}".format(rule), file=dest)
            print("---------------------------------", file=dest)
            
            slrs = data[rule] / data["MLB"]   
            print("\nSLR", file=dest)
            print("--------------", file=dest)
            print("MEAN = {}".format(slrs.mean()), file=dest)
            print("BEST = {}".format(slrs.min()), file=dest)
            print("WORST = {}".format(slrs.max()), file=dest)
            optimal = (abs(slrs.values - 1.0) < 1e-6).sum()
            print("#OPTIMAL: {}/{}".format(optimal, data.shape[0]), file=dest)
            print("%OPTIMAL: {}".format(100 * optimal/data.shape[0]), file=dest)  
            
            speedups = data["MST"] / data[rule]   
            print("\nSPEEDUP", file=dest)
            print("--------------", file=dest)
            print("MEAN = {}".format(speedups.mean()), file=dest)
            print("BEST = {}".format(speedups.max()), file=dest)
            print("WORST = {}".format(speedups.min()), file=dest)
            failures = (speedups.values < 1.0).sum()
            print("#FAILURES: {}/{}".format(failures, data.shape[0]), file=dest)
            print("%FAILURES: {}".format(100 * failures/data.shape[0]), file=dest)           
            
            pds = 100*(data[rule] - bests)/bests
            print("\nPERCENTAGE DEGRADATION (PD)", file=dest)
            print("--------------", file=dest)        
            print("MEAN = {}".format(pds.mean()), file=dest)
            print("WORST = {}".format(pds.max()), file=dest)
            best_occs = (pds.values == 0.0).sum()
            print("#BESTS: {}/{}".format(best_occs, data.shape[0]), file=dest)
            print("%BESTS: {}".format(100 * best_occs/data.shape[0]), file=dest)
            
            if rule == "EFT":
                continue
            reds = 100*(data["EFT"] - data[rule])/data["EFT"]
            print("\nREDUCTION VS DEFAULT EFT SELECTION", file=dest)
            print("--------------", file=dest)
            print("MEAN = {}%".format(reds.mean()), file=dest)
            print("BEST = {}%".format(reds.max()), file=dest)
            print("WORST = {}%".format(reds.min()), file=dest) 
            better = (reds.values > 0.0).sum()
            print("#BETTER: {}/{}".format(better, data.shape[0]), file=dest)
            print("%BETTER: {}".format(100 * better/data.shape[0]), file=dest)
            equal = (reds.values == 0.0).sum()
            print("#EQUAL: {}/{}".format(equal, data.shape[0]), file=dest)
            print("%EQUAL: {}".format(100 * equal/data.shape[0]), file=dest)
            alag = (reds.values >= 0.0).sum()
            print("#AT LEAST AS GOOD: {}/{}".format(alag, data.shape[0]), file=dest)
            print("%AT LEAST AS GOOD: {}".format(100 * alag/data.shape[0]), file=dest)

# # All data.
# loc = "{}/all.txt".format(summary_path)
# summarize(data=df, save_dest=loc)

# # By platform.
# for s in ngpus:
#     sdf = df.loc[(df['s'] == s)]  
#     loc = "{}/s{}.txt".format(summary_path, s)
#     summarize(data=sdf, save_dest=loc)
    
# # By CCR.
# for ccr in ccrs:
#     sdf = df.loc[(df['CCR'] == ccr)]   
#     loc = "{}/ccr{}.txt".format(summary_path, ccr)
#     summarize(data=sdf, save_dest=loc)

# # By platform and tile size.  
# for s in ngpus:
#     for ccr in ccrs:
#         sdf = df.loc[(df['s'] == s) & (df['CCR'] == ccr)]   
#         loc = "{}/s{}_ccr{}.txt".format(summary_path, s, ccr)
#         summarize(data=sdf, save_dest=loc)
        
# =============================================================================
# Plots.
# =============================================================================

format_ccr = {0.01 : "001", 0.1 : "01", 1.0 : "1", 10.0 : "10"} # TODO: won't take e.g., ccr0.01 as name b/c of decimal point.

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
    
    bests = data.loc[:, all_rules].min(axis=1) 
    mpd = {}
    for rule in all_rules:
        pds = 100*(data[rule] - bests)/bests
        mpd[rule] = pds.mean()     
          
    x = np.arange(len(all_rules))  
    fig, ax = plt.subplots(dpi=400)
    
    colors = ['#E24A33', '#348ABD', '#988ED5', '#FBC15E', '#8EBA42', '#FFB5B8']
    
    rects = ax.bar(x, list(mpd.values()), color=colors)    
    ax.bar_label(rects, labels=list(round(p, 1) for p in mpd.values()), label_type='edge')
    
    plt.minorticks_on()
    plt.grid(True, linestyle='-', axis='y', which='major')
    plt.grid(True, linestyle=':', axis='y', which='minor')
    plt.grid(False, axis='x')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    if ylabel:
        ax.set_ylabel("MEAN PERCENTAGE DEGRADATION", labelpad=5)
    ax.set_xticks(x)
    ax.set_xticklabels(all_rules)
    ax.tick_params(axis='x', which='minor', bottom=False)
    if legend:
        # map names to colors
        cmap = dict(zip(all_rules, colors))        
        # create the rectangles for the legend
        patches = [Patch(color=v, label=k) for k, v in cmap.items()]
        ax.legend(labels=all_rules, handles=patches, handlelength=3, handletextpad=0.4, ncol=2, loc='best', fancybox=True, facecolor='white')
                
    fig.tight_layout()    
    plt.savefig('{}/select_stg_mpd_{}'.format(mpd_plot_path, plot_name), bbox_inches='tight') 
    plt.close(fig) 
    
def plot_better(data, plot_name, legend=True, ylabel=True):
    
    # Better/equal.
    bet_plot_path = "{}/better/".format(plot_path)
    pathlib.Path(bet_plot_path).mkdir(parents=True, exist_ok=True)
    
    better = [round(100*(data["EFT"] - data[rule] > 0.0).sum()/data.shape[0]) for rule in all_rules[1:]]
    equal = [round(100*(data["EFT"] - data[rule] == 0.0).sum()/data.shape[0]) for rule in all_rules[1:]]
    width = 0.4
    fig, ax = plt.subplots(dpi=400)
    rects1 = ax.bar(all_rules[1:], better, width, label="BETTER THAN EFT", color='#E24A33')
    rects2 = ax.bar(all_rules[1:], equal, width, bottom=better, label="SAME AS EFT", color='#348ABD')
    if ylabel:
        ax.set_ylabel("%", labelpad=5)
    ax.set_xticklabels(all_rules[1:])
    check = lambda x : x if x > 3 else ""
    ax.bar_label(rects1, labels=list(check(x) for x in better), label_type='center')
    ax.bar_label(rects2, labels=list(check(x) for x in equal), label_type='center')
    if legend:
        ax.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white') 
            
    fig.tight_layout()    
    plt.savefig('{}/select_stg_bet_{}'.format(bet_plot_path, plot_name), bbox_inches='tight') 
    plt.close(fig)   
   
# All data.
# plot_better(data=df, plot_name="all")
plot_mpd(data=df, plot_name="all")

# By platform and CCR.
for s, b in product(ngpus, ccrs):
    sdf = df.loc[(df['s'] == s) & (df['CCR'] == b)]  
    leg = True if s == 1 and b == 0.01 else False
    ylabel = True if s == 1 else False
    plot_mpd(data=sdf, plot_name="s{}_b{}".format(s, format_ccr[b]), legend=leg, ylabel=ylabel) 



            
