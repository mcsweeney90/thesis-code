#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summaries and plots for the Cholesky data.
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
df = pd.read_csv('results.csv')

ntiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
ntasks = {5:35, 10:220, 15:680, 20:1540, 25:2925, 30:4960, 35:7770, 40:11480, 45:16215, 50:22100}
nbs = [128, 1024]
r = 32
ngpus = [1, 4]
rules = ["EFT", "NC", "BL", "OL-I", "OL-II", "GCP", "HAL"]

# =============================================================================
# Human readable summaries.
# =============================================================================       

# def summarize(data, save_dest):
#     with open(save_dest, "w") as dest:
#         print("COMPARISON OF LOOKHEAD-BASED SELECTION RULES IN PRIORITY-BASED FRAMEWORK.", file=dest) 
#         print("PRIORITIZATION PHASE UPWARD RANK WITH ARITHMETIC MEAN IN ALL CASES.", file=dest) 
        
#         bests = data.loc[:, rules].min(axis=1)            
#         for rule in rules:
#             print("\n\n\n---------------------------------", file=dest)
#             print("SELECTION RULE : {}".format(rule), file=dest)
#             print("---------------------------------", file=dest)
            
#             slrs = data[rule] / data["MLB"]   
#             print("\nSLR", file=dest)
#             print("--------------", file=dest)
#             print("MEAN = {}".format(slrs.mean()), file=dest)
#             print("BEST = {}".format(slrs.min()), file=dest)
#             print("WORST = {}".format(slrs.max()), file=dest)
#             optimal = (abs(slrs.values - 1.0) < 1e-6).sum()
#             print("#OPTIMAL: {}/{}".format(optimal, data.shape[0]), file=dest)
#             print("%OPTIMAL: {}".format(100 * optimal/data.shape[0]), file=dest) 
            
#             speedups = data["MST"] / data[rule]   
#             print("\nSPEEDUP", file=dest)
#             print("--------------", file=dest)
#             print("MEAN = {}".format(speedups.mean()), file=dest)
#             print("BEST = {}".format(speedups.max()), file=dest)
#             print("WORST = {}".format(speedups.min()), file=dest)
#             failures = (speedups.values < 1.0).sum()
#             print("#FAILURES: {}/{}".format(failures, data.shape[0]), file=dest)
#             print("%FAILURES: {}".format(100 * failures/data.shape[0]), file=dest)               
            
#             pds = 100*(data[rule] - bests)/bests
#             print("\nPERCENTAGE DEGRADATION (PD)", file=dest)
#             print("--------------", file=dest)        
#             print("MEAN = {}".format(pds.mean()), file=dest)
#             print("WORST = {}".format(pds.max()), file=dest)
#             best_occs = (pds.values == 0.0).sum()
#             print("#BESTS: {}/{}".format(best_occs, data.shape[0]), file=dest)
#             print("%BESTS: {}".format(100 * best_occs/data.shape[0]), file=dest)
            
#             if rule == "EFT":
#                 continue
#             reds = 100*(data["EFT"] - data[rule])/data["EFT"]
#             print("\nREDUCTION VS DEFAULT EFT RULE", file=dest)
#             print("--------------", file=dest)
#             print("MEAN = {}%".format(reds.mean()), file=dest)
#             print("BEST = {}%".format(reds.max()), file=dest)
#             print("WORST = {}%".format(reds.min()), file=dest) 
#             alag = (reds.values >= 0.0).sum()
#             print("#AT LEAST AS GOOD: {}/{}".format(alag, data.shape[0]), file=dest)
#             print("%AT LEAST AS GOOD: {}".format(100 * alag/data.shape[0]), file=dest)
            
#             print("\nRUNTIME INCREASE VS DEFAULT EFT RULE", file=dest)
#             print("--------------", file=dest)
#             print("MEAN = x {}".format(data["{} TIME".format(rule)].mean()), file=dest)
#             print("BEST = x {}".format(data["{} TIME".format(rule)].min()), file=dest)
#             print("WORST = x {}".format(data["{} TIME".format(rule)].max()), file=dest) 
         

# # All data.
# loc = "{}/all.txt".format(summary_path)
# summarize(data=df, save_dest=loc)

# # By platform.
# for s in ngpus:
#     sdf = df.loc[(df['s'] == s)]  
#     loc = "{}/s{}.txt".format(summary_path, s)
#     summarize(data=sdf, save_dest=loc)
    
# # By tile size.
# for nb in nbs:
#     sdf = df.loc[(df['NB'] == nb)]   
#     loc = "{}/nb{}.txt".format(summary_path, nb)
#     summarize(data=sdf, save_dest=loc)

# # By platform and tile size.  
# for s in ngpus:
#     for nb in nbs:
#         sdf = df.loc[(df['s'] == s) & (df['NB'] == nb)]   
#         loc = "{}/s{}_nb{}.txt".format(summary_path, s, nb)
#         summarize(data=sdf, save_dest=loc)
        
# =============================================================================
# Plots.
# =============================================================================

for s in ngpus:
    for nb in nbs:
        sdf = df.loc[(df['s'] == s) & (df['NB'] == nb)]  
        fig = plt.figure(dpi=400)
        ax1 = fig.add_subplot(111)
        ax1.plot(ntiles, sdf["EFT"]/sdf["MLB"] , color='#E24A33', marker='o', label="EFT")
        ax1.plot(ntiles, sdf["NC"]/sdf["MLB"], color='#348ABD', marker='o', label="NC")
        ax1.plot(ntiles, sdf["BL"]/sdf["MLB"], color='#777777', marker='o', label="BL")
        ax1.plot(ntiles, sdf["OL-I"]/sdf["MLB"], color='#988ED5', marker='o', label="OL-I")
        ax1.plot(ntiles, sdf["OL-II"]/sdf["MLB"], color='#FBC15E', marker='o', label="OL-II")    
        ax1.plot(ntiles, sdf["GCP"]/sdf["MLB"], color='#8EBA42', marker='o', label="GCP") 
        ax1.plot(ntiles, sdf["HAL"]/sdf["MLB"], color='#FFB5B8', marker='o', label="HAL") 
        
        plt.minorticks_on()
        plt.grid(True, linestyle='-', axis='y', which='major')
        plt.grid(True, linestyle=':', axis='y', which='minor')
        plt.grid(True, linestyle='-', axis='x', which='major')
        plt.grid(True, linestyle=':', axis='x', which='minor')
        
        # plt.yscale('log')
        if s == 4:
            ax1.set_xlabel("N", labelpad=5)
        ax1.set_ylabel("SCHEDULE LENGTH RATIO", labelpad=5)
        if s == 1:
            ax1.legend(handlelength=3, handletextpad=0.4, ncol=2, loc='best', fancybox=True, facecolor='white') 
        plt.savefig('{}/lk_chol_slr_s{}_nb{}'.format(plot_path, s, nb), bbox_inches='tight') 
        plt.close(fig) 

