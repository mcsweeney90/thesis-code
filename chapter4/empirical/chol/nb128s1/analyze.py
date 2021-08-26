#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of empirical (i.e., Monte Carlo) longest path distributions.
"""

import pathlib, dill
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
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

summary_path = "summaries/"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "plots/"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

runs = 100000
ntiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
ntasks = {5:35, 10:220, 15:680, 20:1540, 25:2925, 30:4960, 35:7770, 40:11480, 45:16215, 50:22100}
dists = ["normal", "gamma", "uniform"]
upper_dists = {"normal" : "NORMAL", "gamma" : "GAMMA", "uniform" : "UNIFORM"}

# =============================================================================
# Summaries.
# =============================================================================
   
# for nt in ntiles:
#     with open("{}/{}.txt".format(summary_path, nt), "w") as dest:
#         print("SUMMARY OF EMPIRICAL LONGEST PATH DISTRIBUTION.", file=dest) 
#         print("NUMBER OF TILES: {}".format(nt), file=dest)
#         print("NUMBER OF TASKS: {}".format(ntasks[nt]), file=dest)
#         print("EMPIRICAL DISTRIBUTIONS GENERATED THROUGH {} REALIZATIONS OF GRAPH WEIGHTS.".format(runs), file=dest)  
#         # Get the data.
#         data = {}
#         for dist in dists:
#             with open('data/{}/{}.dill'.format(dist, nt), 'rb') as file:
#                 D = dill.load(file)
#             data[dist] = D
#             print("\n\n\n---------------------------------", file=dest)
#             print("{} WEIGHTS".format(upper_dists[dist]), file=dest)
#             print("---------------------------------", file=dest)
            
#             # Summary statistics.
#             print("\nSUMMARY STATISTICS", file=dest)
#             print("------------------------------", file=dest)
#             mu = np.mean(D)
#             print("MEAN: {}".format(mu), file=dest)
#             sigma = np.std(D)
#             var = sigma**2
#             print("VARIANCE/STD: {} / {}".format(var, sigma), file=dest)
#             skw = skew(D)
#             print("SKEWNESS: {}".format(skw), file=dest)
#             kur = kurtosis(D)
#             print("EXCESS KURTOSIS: {}".format(kur), file=dest)
#             med = np.median(D)
#             print("MEDIAN: {}".format(med), file=dest)
#             mx = max(D)
#             print("MAXIMUM: {}".format(mx), file=dest)
#             mn = min(D)
#             print("MINIMUM: {}".format(mn), file=dest)
            
#             # Relative error in mean and sd as number of samples increases.
#             print("\nRELATIVE ERROR (MU, SIGMA)", file=dest)
#             print("------------------------------", file=dest)
#             for samps in [10, 100, 1000, 10000]:
#                 R = D[:samps]
#                 rmu = np.mean(R)
#                 rsigma = np.std(R)
#                 print("{} SAMPLES: ({}, {})".format(samps, 100*abs(rmu - mu)/mu, 100*abs(rsigma - sigma)/sigma), file=dest)            
                            
#         print("\n\n\n---------------------------------", file=dest)
#         print("SIMILARITY COMPARISONS (TWO-SIDED KS TESTS)", file=dest)
#         print("---------------------------------", file=dest)        
#         N, G, U = data.values()
#         ks, p = ks_2samp(N, G)
#         print("NORMAL-GAMMA: ({}, {})".format(ks, p), file=dest)
#         ks, p = ks_2samp(N, U)
#         print("NORMAL-UNIFORM: ({}, {})".format(ks, p), file=dest)
#         ks, p = ks_2samp(G, U)
#         print("GAMMA-UNIFORM: ({}, {})".format(ks, p), file=dest)
        
            
# =============================================================================
# Plots.
# =============================================================================

colors = {"normal":'#E24A33', "gamma":'#348ABD', "uniform":'#8EBA42'}

# Histrograms.
fig, axs = plt.subplots(nrows=len(ntiles), ncols=len(dists), figsize=(9, 9), subplot_kw={'xticks': [], 'yticks': []}, dpi=400)
for i, nt in enumerate(ntiles):
    for j, dist in enumerate(dists):
        # Get the data.
        with open('data/{}/{}.dill'.format(dist, nt), 'rb') as file:
            D = dill.load(file)
        # Add subplot.
        axs[i, j].hist(D, color=colors[dist], density=True, bins='auto') 
        axs[i, j].axvline(np.mean(D), color='k', linestyle='solid', linewidth=1)
        if j == 0:
            axs[i, j].set_ylabel("N = {}".format(nt), fontsize=6)
        if nt == 50:
            axs[i, j].set_xlabel("{}".format(upper_dists[dist]), fontsize=10)
plt.subplots_adjust(wspace=0) 
plt.savefig('{}/nb128s1histograms'.format(plot_path), bbox_inches='tight') 
plt.close(fig) 

# # Progression of MC solutions.
# samps = [10, 100, 1000, 10000, 100000]
# for dist in dists:
#     fig, axs = plt.subplots(nrows=len(ntiles), ncols=len(samps), figsize=(6, 9), subplot_kw={'xticks': [], 'yticks': []}, dpi=400)
#     for i, nt in enumerate(ntiles):
#         with open('data/{}/{}.dill'.format(dist, nt), 'rb') as file:
#             D = dill.load(file)
#         for j, samp in enumerate(samps):
#             # Get the data.
#             R = D[:samp] #np.random.choice(D, samp)
#             # Add subplot.
#             axs[i, j].hist(R, color=colors[dist], density=True, bins='auto') 
#             # axs[i, j].axvline(np.mean(D), color='k', linestyle='solid', linewidth=1)
#             if j == 0:
#                 axs[i, j].set_ylabel("nt = {}".format(nt), fontsize=6)
#             if nt == 50:
#                 axs[i, j].set_xlabel("{}".format(samp), fontsize=10)
#     plt.subplots_adjust(wspace=0) 
#     plt.savefig('{}/{}_prog'.format(plot_path, dist), bbox_inches='tight') 
#     plt.close(fig) 

            

