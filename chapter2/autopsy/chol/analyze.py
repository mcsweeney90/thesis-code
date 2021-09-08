#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summaries and plots for the Cholesky data.
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

# =============================================================================
# Human readable summaries.
# =============================================================================       

def summarize(data, save_dest):
    with open(save_dest, "w") as dest:
        print("EVALUATION OF AUTOPSY METHOD FOR HEFT SCHEDULES.", file=dest) 
        
        reds = 100*(data["HEFT"] - data["AUT"])/data["HEFT"]
        print("\nREDUCTION VS ORIGINAL HEFT SCHEDULE", file=dest)
        print("--------------", file=dest)
        print("MEAN = {}%".format(reds.mean()), file=dest)
        print("BEST = {}%".format(reds.max()), file=dest)
        print("WORST = {}%".format(reds.min()), file=dest) 
        alag = (reds.values >= 0.0).sum()
        print("#AT LEAST AS GOOD: {}/{}".format(alag, data.shape[0]), file=dest)
        print("%AT LEAST AS GOOD: {}".format(100 * alag/data.shape[0]), file=dest)
        
        print("\nADDITIONAL RUNTIME (% OF HEFT RUNTIME)", file=dest)
        print("--------------", file=dest)
        print("MEAN = {} %".format(data["AUT PTI"].mean()), file=dest)
        print("BEST = {} %".format(data["AUT PTI"].min()), file=dest)
        print("WORST = {} %".format(data["AUT PTI"].max()), file=dest) 

# All data.
loc = "{}/all.txt".format(summary_path)
summarize(data=df, save_dest=loc)

# By platform.
for s in ngpus:
    sdf = df.loc[(df['s'] == s)]  
    loc = "{}/s{}.txt".format(summary_path, s)
    summarize(data=sdf, save_dest=loc)
    
# By tile size.
for nb in nbs:
    sdf = df.loc[(df['NB'] == nb)]   
    loc = "{}/nb{}.txt".format(summary_path, nb)
    summarize(data=sdf, save_dest=loc)

# By platform and tile size.  
for s, nb in product(ngpus, nbs):
    sdf = df.loc[(df['s'] == s) & (df['NB'] == nb)]   
    loc = "{}/s{}_nb{}.txt".format(summary_path, s, nb)
    summarize(data=sdf, save_dest=loc)

# =============================================================================
# Plots.
# =============================================================================
        
# Calculate percentage reductions in makespan.
reds = {}
for s, nb in product(ngpus, nbs):
    sdf = df.loc[(df['s'] == s) & (df['NB'] == nb)] 
    reds[(s, nb)] = 100*(sdf["HEFT"] - sdf["AUT"])/sdf["HEFT"]

fig = plt.figure(dpi=400)
ax1 = fig.add_subplot(111)
ax1.plot(ntiles, reds[(1, 128)], color='#988ED5', marker='o', label="s = 1, nb = 128", linewidth=1.0)
ax1.plot(ntiles, reds[(1, 1024)], color='#988ED5', marker='s', label="s = 1, nb = 1024", linewidth=1.0, linestyle='--')
ax1.plot(ntiles, reds[(4, 128)], color='#8EBA42', marker='o', label="s = 4, nb = 128", linewidth=1.0)
ax1.plot(ntiles, reds[(4, 1024)], color='#8EBA42', marker='s', linestyle='--',  label="s = 4, nb = 1024", linewidth=1.0)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.0)

plt.minorticks_on()
plt.grid(True, linestyle='-', axis='y', which='major')
plt.grid(True, linestyle=':', axis='y', which='minor')
plt.grid(True, linestyle='-', axis='x', which='major')
plt.grid(True, linestyle=':', axis='x', which='minor')

ax1.set_xlabel("N", labelpad=5)
ax1.set_ylabel("MAKESPAN REDUCTION (%)", labelpad=5)
ax1.tick_params(axis='x', which='minor', bottom=False)
ax1.tick_params(axis='y', which='minor', left=False)

ax1.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white') 
plt.savefig('{}/aut_red_all'.format(plot_path), bbox_inches='tight') 
plt.close(fig) 