#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of STG results wrt CLT-based heuristics.
"""

import pathlib, dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
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
plt.ioff() # Don't show plots.
# print(plt.rcParams['axes.prop_cycle'].by_key()['color'])

####################################################################################################

df = pd.read_csv('stg.csv')

summary_path = "summaries/clt/"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "plots/clt/"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

dists = ["normal", "gamma", "uniform"]
covs = [0.01, 0.03, 0.1, 0.3]

####################################################################################################

def summarize(data, name):
    with open("{}/clt_{}.txt".format(summary_path, name), "w") as dest:
        print("SUMMARY OF CLT-BASED HEURISTICS.", file=dest)  
        for stat in ["MU", "VAR"]:                            
            print("\n\n\n------------------------------------------------------------------", file=dest)
            print("% RELATIVE ERROR IN {} (AVG, MAX)".format(stat), file=dest)
            print("------------------------------------------------------------------", file=dest)
            
            for method in ["CPM", "K LOWER", "K UPPER", "SCULLI", "CORLCA", "MC10", "MC100"]:
                if method == "CPM" and stat == "VAR":
                    continue               
                err = 100*abs(data["{} {}".format(method, stat)] - data["REF {}".format(stat)]) / data["REF {}".format(stat)]
                print("{}: ({}, {})".format(method, err.mean(), err.max()), file=dest)                
        print("\n\n\n------------------------------------------------------------------", file=dest)
        print("KS STATISTICS (AVG, MAX)", file=dest)
        print("------------------------------------------------------------------", file=dest)   
        for method in ["SCULLI", "CORLCA", "MC10", "MC100"]:
            print("{}: ({}, {})".format(method, data["{} KS".format(method)].mean(), data["{} KS".format(method)].max()), file=dest)
        
        print("\n\n\n------------------------------------------------------------------", file=dest)
        print("KAMBUROWSKI VIOLATIONS", file=dest)
        print("------------------------------------------------------------------", file=dest)  
        
        mu_lower_viols = (data["REF MU"] - data["K LOWER MU"] < 0.0).sum()
        print("% TIMES MU LOWER VIOLATED: {}".format(100*mu_lower_viols/len(data)), file=dest)        
        mu_upper_viols = (data["K UPPER MU"] - data["REF MU"] < 0.0).sum()
        print("% TIMES MU UPPER VIOLATED: {}".format(100*mu_upper_viols/len(data)), file=dest)
        var_lower_viols = (data["REF VAR"] - data["K LOWER VAR"] < 0.0).sum()
        print("% TIMES VAR LOWER VIOLATED: {}".format(100*var_lower_viols/len(data)), file=dest)        
        var_upper_viols = (data["K UPPER VAR"] - data["REF VAR"] < 0.0).sum()
        print("% TIMES VAR UPPER VIOLATED: {}".format(100*var_upper_viols/len(data)), file=dest)
        
# Entire data set 
# summarize(df, name="all")
# # By cov.
# for cov in covs:
#     sdf = df.loc[(df['COV'] == cov)] 
#     summarize(sdf, name="cov{}".format(cov))
    

# =============================================================================
# Plots.
# =============================================================================

# # Relative error in variance (average).
# methods = ["K LOWER", "K UPPER", "SCULLI", "CORLCA", "MC10", "MC100"]
# means = {method : [] for method in methods}
# for cov in covs:
#     data = df.loc[(df['COV'] == cov)] 
#     for method in methods:
#         err = 100*abs(data["{} VAR".format(method)] - data["REF VAR"]) / data["REF VAR"]
#         means[method].append(err.mean())
# x = np.arange(len(covs))
# width = 0.16   
# fig, ax = plt.subplots(dpi=400)
# rects1 = ax.bar(x - 2*width, means["SCULLI"], width, label='SCULLI', color='#E24A33')
# rects2 = ax.bar(x - width, means["CORLCA"], width, label='CORLCA', color='#348ABD')
# rects3 = ax.bar(x, means["K UPPER"], width, label='K. UPPER', color='#8EBA42')
# rects4 = ax.bar(x + width, means["MC10"], width, label='MC10', color='#988ED5')
# rects5 = ax.bar(x + 2*width, means["MC100"], width, label='MC100', color='#FBC15E')

# plt.minorticks_on()
# plt.grid(True, linestyle='-', axis='y', which='major')
# plt.grid(True, linestyle=':', axis='y', which='minor')
# plt.grid(False, axis='x')  

# ax.set_ylabel("MEAN % ERROR IN VARIANCE", labelpad=5)
# ax.set_xticks(x)
# ax.set_xticklabels(covs)
# ax.set_xlabel("MEAN COEFFICIENT OF VARIATION ($\mu_v$)", labelpad=5)
# ax.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white')
# ax.tick_params(axis='x', which='minor', bottom=False) 
# fig.tight_layout()    
# plt.savefig('{}/stg_clt_mean_var'.format(plot_path), bbox_inches='tight') 
# plt.close(fig) 


# # KS statistics (mean or max).
# methods = ["SCULLI", "CORLCA", "MC10", "MC100"]
# means = {method : [] for method in methods}
# # maxs = {method : [] for method in methods}
# for cov in covs:
#     data = df.loc[(df['COV'] == cov)] 
#     for method in methods:
#         err = 100*abs(data["{} VAR".format(method)] - data["REF VAR"]) / data["REF VAR"]
#         means[method].append(data['{} KS'.format(method)].mean())
#         # maxs[method].append(data['{} KS'.format(method)].max())
# x = np.arange(len(covs))
# width = 0.2  
# fig, ax = plt.subplots(dpi=400)
# rects1 = ax.bar(x - 1.5*width, means["SCULLI"], width, label='SCULLI', color='#E24A33')
# rects2 = ax.bar(x - 0.5*width, means["CORLCA"], width, label='CORLCA', color='#348ABD')
# rects3 = ax.bar(x + 0.5*width, means["MC10"], width, label='MC10', color='#988ED5')
# rects4 = ax.bar(x + 1.5*width, means["MC100"], width, label='MC100', color='#FBC15E')

# plt.minorticks_on()
# plt.grid(True, linestyle='-', axis='y', which='major')
# plt.grid(True, linestyle=':', axis='y', which='minor')
# plt.grid(False, axis='x')  

# ax.set_ylabel("MEAN KS STATISTIC", labelpad=5)
# ax.set_xticks(x)
# ax.set_xticklabels(covs)
# ax.set_xlabel("MEAN COEFFICIENT OF VARIATION ($\mu_v$)", labelpad=5)
# ax.legend(handlelength=3, handletextpad=0.4, ncol=2, loc='best', fancybox=True, facecolor='white')
# ax.tick_params(axis='x', which='minor', bottom=False) 
# fig.tight_layout()    
# plt.savefig('{}/stg_clt_ks'.format(plot_path), bbox_inches='tight') 
# plt.close(fig) 
