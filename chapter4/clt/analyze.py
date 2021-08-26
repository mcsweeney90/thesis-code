#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of CLT-based heuristics (and related bounds).
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
#plt.rcParams["figure.figsize"] = (9.6,4)
plt.ioff() # Don't show plots.
# print(plt.rcParams['axes.prop_cycle'].by_key()['color'])

####################################################################################################

df = pd.read_csv('clt.csv')
mc_df = pd.read_csv('../rpm/rpm.csv')
time = pd.read_csv('timing.csv')

summary_path = "summaries/"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "plots/"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

ntiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
ntasks = {5:35, 10:220, 15:680, 20:1540, 25:2925, 30:4960, 35:7770, 40:11480, 45:16215, 50:22100}
dists = ["normal", "gamma", "uniform"]
upper_dists = {"normal" : "NORMAL", "gamma" : "GAMMA", "uniform" : "UNIFORM"}

# =============================================================================
# Summaries.
# =============================================================================

# for nb, s in product([128, 1024], [1, 4]):
#     new_summary_path = "{}/nb{}s{}/".format(summary_path, nb, s)
#     pathlib.Path(new_summary_path).mkdir(parents=True, exist_ok=True) 
#     for nt in ntiles:
#         row = df.loc[(df['nb'] == nb) & (df['s'] == s) & (df['nt'] == nt)].to_dict(orient='records')[0]        
#         data = {}
#         for dist in dists:
#             with open('../empirical/chol/nb{}s{}/data/{}/{}.dill'.format(nb, s, dist, nt), 'rb') as file:
#                 D = dill.load(file)
#             data[dist] = D
#         with open("{}/{}.txt".format(new_summary_path, nt), "w") as dest:
#             print("COMPARISON OF CLT-BASED HEURISTICS WITH EMPIRICAL DISTRIBUTIONS.", file=dest) 
#             print("NUMBER OF TILES: {}".format(nt), file=dest)
#             print("NUMBER OF TASKS: {}".format(ntasks[nt]), file=dest)            
            
#             # Relative error in the mean.
#             print("\n\n\n------------------------------------------------------------------", file=dest)
#             print("RELATIVE ERROR IN MEAN (%)", file=dest)
#             print("------------------------------------------------------------------", file=dest)
            
#             for dist in dists:
#                 D = data[dist]
#                 mu = np.mean(D)
#                 print("\n------------------------------", file=dest)
#                 print("{} WEIGHTS".format(upper_dists[dist]), file=dest)
#                 print("REFERENCE MEAN = {}".format(mu), file=dest)
#                 print("------------------------------", file=dest)
                
#                 cpm = row['CPM']
#                 rel = 100*abs(cpm - mu)/mu
#                 print("CPM = {}".format(rel), file=dest)
                
#                 kl = row['K LOWER MU']
#                 rel = 100*abs(kl - mu)/mu
#                 print("KAMB. LOWER = {}".format(rel), file=dest)
                
#                 ku = row['K UPPER MU']
#                 rel = 100*abs(ku - mu)/mu
#                 print("KAMB. UPPER = {}".format(rel), file=dest)
                
#                 sc = row['SCULLI MU']
#                 rel = 100*abs(sc - mu)/mu
#                 print("SCULLI = {}".format(rel), file=dest)
                
#                 cor = row['CORLCA MU']
#                 rel = 100*abs(cor - mu)/mu
#                 print("CORLCA = {}".format(rel), file=dest)
                
#                 # MC10. 
#                 mc10 = np.mean(data["uniform"][:10])
#                 rel = 100*abs(mc10 - mu)/mu
#                 print("MC10 = {}".format(rel), file=dest)    
                
#                 # MC100. 
#                 mc100 = np.mean(data["uniform"][:100])
#                 rel = 100*abs(mc100 - mu)/mu
#                 print("MC100 = {}".format(rel), file=dest) 
            
#             # Relative error in the variance.
#             print("\n\n\n------------------------------------------------------------------", file=dest)
#             print("RELATIVE ERROR IN VARIANCE (%)", file=dest)
#             print("------------------------------------------------------------------", file=dest)
            
#             for dist in dists:
#                 D = data[dist]
#                 var = np.var(D)
#                 print("\n------------------------------", file=dest)
#                 print("{} WEIGHTS".format(upper_dists[dist]), file=dest)
#                 print("REFERENCE VARIANCE = {}".format(var), file=dest)
#                 print("------------------------------", file=dest)
                                
#                 kl = row['K LOWER VAR']
#                 rel = 100*abs(kl - var)/var
#                 print("KAMB. LOWER = {}".format(rel), file=dest)
                
#                 ku = row['K UPPER VAR']
#                 rel = 100*abs(ku - var)/var
#                 print("KAMB. UPPER = {}".format(rel), file=dest)
                
#                 sc = row['SCULLI VAR']
#                 rel = 100*abs(sc - var)/var
#                 print("SCULLI = {}".format(rel), file=dest)
                
#                 cor = row['CORLCA VAR']
#                 rel = 100*abs(cor - var)/var
#                 print("CORLCA = {}".format(rel), file=dest)
                
#                 # MC10. 
#                 mc10 = np.var(data["uniform"][:10])
#                 rel = 100*abs(mc10 - var)/var
#                 print("MC10 = {}".format(rel), file=dest)
                
#                 # MC100. 
#                 mc100 = np.var(data["uniform"][:100])
#                 rel = 100*abs(mc100 - var)/var
#                 print("MC100 = {}".format(rel), file=dest)
            
#             # Relative error in the variance.
#             print("\n\n\n------------------------------------------------------------------", file=dest)
#             print("KS STATISTICS - NORMAL DISTS WITH COMPUTED MOMENTS VS EMPIRICAL DISTS.", file=dest)
#             print("------------------------------------------------------------------", file=dest)
            
#             for dist in dists:
#                 D = data[dist]
#                 print("\n------------------------------", file=dest)
#                 print("{} WEIGHTS".format(upper_dists[dist]), file=dest)
#                 print("------------------------------", file=dest)
                
#                 kl_mu, kl_sd = row['K LOWER MU'], sqrt(row['K LOWER VAR'])
#                 ks, p = kstest(D, cdf='norm', args=(kl_mu, kl_sd))
#                 print("KAMB. LOWER = ({}, {})".format(ks, p), file=dest)
                
#                 ku_mu, ku_sd = row['K UPPER MU'], sqrt(row['K UPPER VAR'])
#                 ks, p = kstest(D, cdf='norm', args=(ku_mu, ku_sd))
#                 print("KAMB. UPPER = ({}, {})".format(ks, p), file=dest)
                
#                 sc_mu, sc_sd = row['SCULLI MU'], sqrt(row['SCULLI VAR'])
#                 ks, p = kstest(D, cdf='norm', args=(sc_mu, sc_sd))
#                 print("SCULLI = ({}, {})".format(ks, p), file=dest)
                
#                 cor_mu, cor_sd = row['CORLCA MU'], sqrt(row['CORLCA VAR'])
#                 ks, p = kstest(D, cdf='norm', args=(cor_mu, cor_sd))
#                 print("CORLCA = ({}, {})".format(ks, p), file=dest)
                
#                 # MC10. 
#                 mc10 = data["uniform"][:10]
#                 ks, p = ks_2samp(D, mc10)
#                 print("MC10 = ({}, {})".format(ks, p), file=dest)   
                
#                 # MC10. 
#                 mc100 = data["uniform"][:100]
#                 ks, p = ks_2samp(D, mc100)
#                 print("MC100 = ({}, {})".format(ks, p), file=dest) 
            
            
# =============================================================================
# Plots.
# =============================================================================                
            
# # Variance. 
# for nb, s in product([128, 1024], [1, 4]):
#     data = df.loc[(df['nb'] == nb) & (df['s'] == s)] 
#     mc_data = mc_df.loc[(mc_df['nb'] == nb) & (mc_df['s'] == s)] 
#     ref_vars = []
#     for nt in ntiles:
#         with open('../empirical/chol/nb{}s{}/data/gamma/{}.dill'.format(nb, s, nt), 'rb') as file:
#             D = dill.load(file)
#         ref_vars.append(np.var(D))
#     # Make the plot. 
#     fig = plt.figure(dpi=400)
#     ax = fig.add_subplot(111)
#     ax.plot(ntiles, ref_vars, color='k', label="ACTUAL")
#     ax.plot(ntiles, list(mc_data["MC10 VAR"]), color='#988ED5', label="MC10")
#     ax.plot(ntiles, list(mc_data["MC100 VAR"]), color='#FBC15E', label="MC100")
#     ax.plot(ntiles, data["SCULLI VAR"], color='#E24A33', label="SCULLI") 
#     ax.plot(ntiles, data["CORLCA VAR"], color='#348ABD', label="CORLCA")
#     ax.fill_between(ntiles, data["K LOWER VAR"], data["K UPPER VAR"], color='#8EBA42', alpha=0.3, label="KAMB.") 
#     plt.yscale('log')
#     # plt.xticks(ntiles, list(ntasks.values()))
#     if nb == 1024:
#         ax.set_xlabel("N", labelpad=5)
#     if s == 1:
#         ax.set_ylabel("VARIANCE", labelpad=5)
#     if s == 1 and nb == 128:
#         ax.legend(handlelength=3, handletextpad=0.4, ncol=2, loc='best', fancybox=True, facecolor='white') 
#     plt.savefig('{}/clt_nb{}s{}variance'.format(plot_path, nb, s), bbox_inches='tight') 
#     plt.close(fig) 

# # KS stats for Sculli, CorLCA and MC10.
# for nb, s in product([128, 1024], [1, 4]):
#     data = df.loc[(df['nb'] == nb) & (df['s'] == s)] 
#     mc_data = mc_df.loc[(mc_df['nb'] == nb) & (mc_df['s'] == s)] 
#     sculli, corlca = [], []
#     for nt in ntiles:        
#         with open('../empirical/chol/nb{}s{}/data/gamma/{}.dill'.format(nb, s, nt), 'rb') as file:
#             D = dill.load(file)        
#         row = data.loc[(data['nt'] == nt)].to_dict(orient='records')[0] 
#         # Sculli.
#         sc_mu, sc_sd = row['SCULLI MU'], sqrt(row['SCULLI VAR'])
#         ks, _ = kstest(D, cdf='norm', args=(sc_mu, sc_sd))
#         sculli.append(ks)
#         # CorLCA.
#         cor_mu, cor_sd = row['CORLCA MU'], sqrt(row['CORLCA VAR'])
#         ks, _ = kstest(D, cdf='norm', args=(cor_mu, cor_sd))
#         corlca.append(ks)       
#     fig, ax = plt.subplots(dpi=400)
#     ax.plot(ntiles, sculli, label='SCULLI', color='#E24A33', marker='s')
#     ax.plot(ntiles, corlca, label='CORLCA', color='#348ABD', marker='o')
#     ax.plot(ntiles, list(mc_data["MC10 KS"]), label='MC10', color='#988ED5', marker='v')
#     ax.plot(ntiles, list(mc_data["MC100 KS"]), label='MC100', color='#FBC15E', marker='^')   
    
#     plt.minorticks_on()
#     plt.grid(True, linestyle='-', axis='y', which='major')
#     plt.grid(True, linestyle=':', axis='y', which='minor')
#     plt.grid(True, linestyle='-', axis='x', which='major')
#     plt.grid(True, linestyle=':', axis='x', which='minor')    
    
#     if s == 1:
#         ax.set_ylabel("KS STATISTIC", labelpad=5)
#     if nb == 1024:
#         ax.set_xlabel("N", labelpad=5)
#     if s == 1 and nb == 128:
#         ax.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white')    
#     ax.tick_params(axis='x', which='minor', bottom=False)
#     ax.tick_params(axis='y', which='minor', left=False)    
    
#     fig.tight_layout()    
#     plt.savefig('{}/clt_nb{}s{}KS'.format(plot_path, nb, s), bbox_inches='tight') 
#     plt.close(fig) 
        
# Time relative to CPM method.
for nb, s in product([128, 1024], [1, 4]):
    data = time.loc[(time['nb'] == nb) & (time['s'] == s)] 
    sculli = data["SCULLI"] / data["CPM"]
    corlca = data["CORLCA"] / data["CPM"]
    kamb = data["K"] / data["CPM"]
    mc10 = data["MC10"] / data["CPM"]
    mc100 = data["MC100"] / data["CPM"]
            
    fig, ax = plt.subplots(dpi=400)
    ax.plot(ntiles, sculli, label='SCULLI', color='#E24A33', marker='s')
    ax.plot(ntiles, corlca, label='CORLCA', color='#348ABD', marker='o')
    ax.plot(ntiles, kamb, label='KAMB.', color='#8EBA42', marker='D')
    ax.plot(ntiles, mc10, label='MC10', color='#988ED5', marker='v')
    ax.plot(ntiles, mc100, label='MC100', color='#FBC15E', marker='^')
    
    plt.minorticks_on()
    plt.grid(True, linestyle='-', axis='y', which='major')
    plt.grid(True, linestyle=':', axis='y', which='minor')
    plt.grid(True, linestyle='-', axis='x', which='major')
    plt.grid(True, linestyle=':', axis='x', which='minor')    
    
    if s == 1:
        ax.set_ylabel("TIME (NORMALIZED)", labelpad=5)
    if nb == 1024:
        ax.set_xlabel("N", labelpad=5)
    if s==1 and nb == 128:
        ax.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white')
    
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='y', which='minor', left=False)    
    
    fig.tight_layout()    
    plt.savefig('{}/clt_nb{}s{}_time'.format(plot_path, nb, s), bbox_inches='tight') 
    plt.close(fig) 
        
            
            
            