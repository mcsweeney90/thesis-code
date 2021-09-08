#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze data and make plots.
"""

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
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

df = pd.read_csv('rpm.csv')

summary_path = "summaries/"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "plots/"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

ntiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
ntasks = {5:35, 10:220, 15:680, 20:1540, 25:2925, 30:4960, 35:7770, 40:11480, 45:16215, 50:22100}
          
####################################################################################################

# =============================================================================
# Plots.
# =============================================================================        

########################################
# KS stats.        
########################################

for nb, s in product([128, 1024], [1, 4]):
    data = df.loc[(df['nb'] == nb) & (df['s'] == s)]  
    fig, ax = plt.subplots(dpi=400)
    # MC10 and derivatives.
    ax.plot(ntiles, data["MC10 KS"], label='MC10', color='#988ED5', marker='s')
    ax.plot(ntiles, data["SIM10 KS"], label='SIM10', color='#E24A33', linestyle='-', marker='s')
    ax.plot(ntiles, data["DOM10 KS"], label='DOM10', color='#8EBA42', linestyle='-', marker='s')
    # MC100.
    ax.plot(ntiles, data["MC100 KS"], label='MC100', color='#FBC15E', marker='o')
    ax.plot(ntiles, data["SIM100 KS"], label='SIM100', color='#E24A33', linestyle='--', marker='o')
    ax.plot(ntiles, data["DOM100 KS"], label='DOM100', color='#8EBA42', linestyle='--', marker='o')
    
    plt.minorticks_on()
    plt.grid(True, linestyle='-', axis='y', which='major')
    plt.grid(True, linestyle=':', axis='y', which='minor')
    plt.grid(True, linestyle='-', axis='x', which='major')
    plt.grid(True, linestyle=':', axis='x', which='minor')   
    
    if s == 1:
        ax.set_ylabel("KS STATISTIC", labelpad=5)
    if nb == 1024:
        ax.set_xlabel("N", labelpad=5)
    if s == 1 and nb == 128:
        ax.legend(handlelength=2.5, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white')
        
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='y', which='minor', left=False)  
    
    fig.tight_layout()    
    plt.savefig('{}/chol_rpm_nb{}s{}KS'.format(plot_path, nb, s), bbox_inches='tight') 
    plt.close(fig) 
