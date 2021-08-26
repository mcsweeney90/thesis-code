#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze results.
"""

import pathlib, dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from math import sqrt
from itertools import product
import matplotlib.patches as mpl_patches

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
plt.rcParams['lines.markersize'] = 6 # Changed from 3.
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
#plt.rcParams["figure.figsize"] = (9.6,4)
plt.ioff() # Don't show plots.
# print(plt.rcParams['axes.prop_cycle'].by_key()['color'])

####################################################################################################

df = pd.read_csv('scalar.csv')

summary_path = "summaries/"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "plots/"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

covs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]

####################################################################################################

# data = df
# reds = 100*(data["HEFT MU"] - data["MCS100 MU"]) / data["HEFT MU"]
# print(reds.mean())

# # Summary of SHEFT.
# means = []
# for cov in covs:
#     data = df.loc[(df['COV'] == cov)] 
#     sheft_reds = 100*(data["SHEFT MU"] - data["HEFT MU"]) / data["HEFT MU"]
#     means.append(sheft_reds.mean())
# print(means)

# =============================================================================
# Plots.
# =============================================================================

################################################
# Probability of better schedule (histogram).
################################################

probs = {"MC" : [], "UCB" : [], "SHEFT" : [], "SDLS" : []} 
for cov in covs:
    data = df.loc[(df['COV'] == cov)] 
    probs["MC"].append(data["MCS100 PROB"].mean())
    probs["UCB"].append(data["UCB100 PROB"].mean())   
    probs["SHEFT"].append(data["SHEFT PROB"].mean()) 
    # probs["SDLS"].append(data["SDLS PROB"].mean())         
x = np.arange(len(covs))
width = 0.25
fig, ax = plt.subplots(dpi=400)
rects1 = ax.bar(x - width, probs["MC"], width, label='MC', color='#348ABD')
rects2 = ax.bar(x, probs["UCB"], width, label='UCB', color='#E24A33')
rects3 = ax.bar(x + width, probs["SHEFT"], width, label='SHEFT', color='#8EBA42')

plt.minorticks_on()
plt.grid(True, linestyle='-', axis='y', which='major')
plt.grid(True, linestyle=':', axis='y', which='minor')
plt.grid(False, axis='x')  

# ax.axhline(y=0.5, color='black', linestyle='-', linewidth=1.0)
ax.set_ylabel("MEAN PROB. SHORTER THAN HEFT ($\\beta$)", labelpad=5)
ax.set_xticks(x)
ax.set_xticklabels(covs)
ax.set_xlabel("MEAN COEFFICIENT OF VARIATION ($\mu_v$)", labelpad=5) # TODO: or nu?

ax.tick_params(axis='x', which='minor', bottom=False) 

ax.legend(handlelength=3, handletextpad=0.4, ncol=3, loc='best', fancybox=True, facecolor='white')
fig.tight_layout()    
plt.savefig('{}/prob'.format(plot_path), bbox_inches='tight') 
plt.close(fig) 

################################################
# Expected values.
################################################

# means = {"SAM" : [], "UCB" : [], "SHEFT" : [], "HEFT" : []} 
# for cov in covs:
#     data = df.loc[(df['COV'] == cov)] 
#     means["SAM"].append(data["MCS100 MU"].mean())
#     means["UCB"].append(data["UCB100 MU"].mean())   
#     means["SHEFT"].append(data["SHEFT MU"].mean()) 
#     means["HEFT"].append(data["HEFT MU"].mean())         
    
# x = np.arange(len(covs))
# width = 0.2
# fig, ax = plt.subplots(dpi=400)
# rects1 = ax.bar(x - 1.5*width, means["SAM"], width, label='SAM', color='#E24A33')
# rects2 = ax.bar(x - 0.5*width, means["UCB"], width, label='UCB', color='#348ABD')
# rects3 = ax.bar(x + 0.5*width, means["SHEFT"], width, label='SHEFT', color='#8EBA42')
# rects4 = ax.bar(x + 1.5*width, means["HEFT"], width, label='HEFT', color='#8EBA42')

# plt.minorticks_on()
# plt.grid(True, linestyle='-', axis='y', which='major')
# plt.grid(True, linestyle=':', axis='y', which='minor')
# plt.grid(False, axis='x')  

# ax.set_ylabel("MEAN $\mu$", labelpad=5)
# ax.set_xticks(x)
# ax.set_xticklabels(covs)
# ax.set_xlabel("MEAN COEFFICIENT OF VARIATION ($\mu_v$)", labelpad=5) # TODO: or nu?

# ax.tick_params(axis='x', which='minor', bottom=False) 

# ax.legend(handlelength=3, handletextpad=0.4, ncol=3, loc='best', fancybox=True, facecolor='white')
# fig.tight_layout()    
# plt.savefig('{}/mu'.format(plot_path), bbox_inches='tight') 
# plt.close(fig) 

# =============================================================================
# Scatter plots.
# =============================================================================

clean = {0.05:"005", 0.1:"01", 0.15:"015", 0.2:"02", 0.25:"025", 0.3:"03", 0.5:"05"}

def plot_reductions(data, name, ylabel=True, colors=False):
    ucb_reds = 100*(data["HEFT MU"] - data["UCB100 MU"])/data["HEFT MU"]
    mc_reds = 100*(data["HEFT MU"] - data["MCS100 MU"])/data["HEFT MU"]   
    ucb_better = (ucb_reds.values > 0.0).sum()
    mc_better = (mc_reds.values > 0.0).sum()
    
    fig, ax = plt.subplots(dpi=400)
    plt.scatter(range(len(data)), mc_reds, marker='.', color='#348ABD', label="MC : {}".format(round(np.mean(mc_reds), 1)))
    plt.scatter(range(len(data)), ucb_reds, marker='.', color='#E24A33', label="UCB : {}".format(round(np.mean(ucb_reds), 1)))
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
    
    plt.grid(True, linestyle='-', axis='y', which='major')
    ax.set_xticks([])
    ax.tick_params(axis='y', which='minor', left=False)
    
    if ylabel:
        ax.set_ylabel("EXPECTED VALUE REDUCTION (%)", labelpad=5)
    
    ax.legend(handlelength=0, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white', framealpha=1.0)
    
    plt.savefig('{}/red_mu_stg_{}'.format(plot_path, name), bbox_inches='tight') 
    plt.close(fig) 

# plot_reductions(data=df, name="all", ylabel=False)
# for cov in covs:
#     data = df.loc[(df['COV'] == cov)] 
#     y = True if cov in [0.05, 0.15, 0.25, 0.5] else False
#     plot_reductions(data, name="cov{}".format(clean[cov]), ylabel=y)