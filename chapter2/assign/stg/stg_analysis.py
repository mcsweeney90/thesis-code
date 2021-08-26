#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of critical assignment methods for STG.
"""

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

summary_path = "summaries/"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "plots/"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

df = pd.read_csv('stg2.csv')
ccrs = [10.0]#, 0.1, 1.0, 10.0]
r = 32
ngpus = [1, 4]
all_rules = ["EFT", "M-C", "M-A", "HM-C", "HM-A", "O-A", "O-C"]

# =============================================================================
# Human readable summaries.
# =============================================================================       

def summarize(data, save_dest):
    with open(save_dest, "w") as dest:
        print("COMPARISON OF CRITICAL ASSIGNMENT SELECTION RULES IN PRIORITY-BASED FRAMEWORK.", file=dest) 
        print("PRIORITIZATION PHASE AS IN HEFT IN ALL CASES.", file=dest) 
        
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
            alag = (reds.values >= 0.0).sum()
            print("#AT LEAST AS GOOD: {}/{}".format(alag, data.shape[0]), file=dest)
            print("%AT LEAST AS GOOD: {}".format(100 * alag/data.shape[0]), file=dest)

# All data.
loc = "{}/all.txt".format(summary_path)
summarize(data=df, save_dest=loc)

# By platform.
for s in ngpus:
    sdf = df.loc[(df['r'] == r) & (df['s'] == s)]  
    loc = "{}/s{}.txt".format(summary_path, s)
    summarize(data=sdf, save_dest=loc)
    
# By CCR.
for ccr in ccrs:
    sdf = df.loc[(df['CCR'] == ccr)]   
    loc = "{}/ccr{}.txt".format(summary_path, ccr)
    summarize(data=sdf, save_dest=loc)

# By platform and tile size.  
for s in ngpus:
    for ccr in ccrs:
        sdf = df.loc[(df['s'] == s) & (df['CCR'] == ccr)]   
        loc = "{}/s{}_ccr{}.txt".format(summary_path, s, ccr)
        summarize(data=sdf, save_dest=loc)