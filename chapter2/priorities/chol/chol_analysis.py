#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summaries and plots for the Cholesky data.
"""

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpl_patches
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

df = pd.read_csv('chol.csv') 
ntiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
tilings = list("{}x{}".format(N, N) for N in ntiles)
ntasks = {5:35, 10:220, 15:680, 20:1540, 25:2925, 30:4960, 35:7770, 40:11480, 45:16215, 50:22100}
nbs = [128, 1024]
r = 32
ngpus = [1, 4]
u_avgs = ["M", "MD", "B", "SB", "W", "SW", "HM", "SHM", "GM", "SGM", "R", "D", "NC", "SD"]
o_avgs = ["M", "MD", "B", "W", "HM", "GM", "R", "D", "NC", "SD"]
all_avgs = ["{}-U".format(avg) for avg in u_avgs] + ["{}-O".format(avg) for avg in o_avgs]

# =============================================================================
# Human readable summaries.
# =============================================================================       

# def summarize(data, save_dest):
#     with open(save_dest, "w") as dest:
#         print("COMPARISON OF RANKING PHASES IN PRIORITY-BASED FRAMEWORK.", file=dest) 
#         print("SELECTION PHASE EFT/GREEDY IN ALL CASES.", file=dest) 
        
#         bests = data.loc[:, all_avgs].min(axis=1)            
#         for avg in all_avgs:
#             print("\n\n\n---------------------------------", file=dest)
#             print("AVERAGE TYPE : {}".format(avg), file=dest)
#             print("---------------------------------", file=dest)
            
#             slrs = data[avg] / data["MLB"]   
#             print("\nSLR", file=dest)
#             print("--------------", file=dest)
#             print("MEAN = {}".format(slrs.mean()), file=dest)
#             print("BEST = {}".format(slrs.min()), file=dest)
#             print("WORST = {}".format(slrs.max()), file=dest)
#             optimal = (abs(slrs.values - 1.0) < 1e-6).sum()
#             print("#OPTIMAL: {}/{}".format(optimal, data.shape[0]), file=dest)
#             print("%OPTIMAL: {}".format(100 * optimal/data.shape[0]), file=dest) 
            
#             speedups = data["MST"] / data[avg]   
#             print("\nSPEEDUP", file=dest)
#             print("--------------", file=dest)
#             print("MEAN = {}".format(speedups.mean()), file=dest)
#             print("BEST = {}".format(speedups.max()), file=dest)
#             print("WORST = {}".format(speedups.min()), file=dest)
#             failures = (speedups.values < 1.0).sum()
#             print("#FAILURES: {}/{}".format(failures, data.shape[0]), file=dest)
#             print("%FAILURES: {}".format(100 * failures/data.shape[0]), file=dest)  
            
#             imps = 100*(data["RND"] - data[avg])/data["RND"] 
#             print("\n% IMPROVEMENT VS RANDOM", file=dest)
#             print("--------------", file=dest)
#             print("MEAN = {}".format(imps.mean()), file=dest)
#             print("BEST = {}".format(imps.max()), file=dest)
#             print("WORST = {}".format(imps.min()), file=dest) 
#             worse = (imps.values < 0.0).sum()
#             print("#WORSE: {}/{}".format(worse, data.shape[0]), file=dest)
#             print("%WORSE: {}".format(100 * worse/data.shape[0]), file=dest)              
            
#             pds = 100*(data[avg] - bests)/bests
#             print("\nPERCENTAGE DEGRADATION (PD)", file=dest)
#             print("--------------", file=dest)        
#             print("MEAN = {}".format(pds.mean()), file=dest)
#             print("WORST = {}".format(pds.max()), file=dest)
#             best_occs = (pds.values == 0.0).sum()
#             print("#BESTS: {}/{}".format(best_occs, data.shape[0]), file=dest)
#             print("%BESTS: {}".format(100 * best_occs/data.shape[0]), file=dest)
            
#             if avg == "M-U":
#                 continue
#             reds = 100*(data["M-U"] - data[avg])/data["M-U"]
#             print("\nREDUCTION VS DEFAULT M-U RANKING", file=dest)
#             print("--------------", file=dest)
#             print("MEAN = {}%".format(reds.mean()), file=dest)
#             print("BEST = {}%".format(reds.max()), file=dest)
#             print("WORST = {}%".format(reds.min()), file=dest) 
#             alag = (reds.values >= 0.0).sum()
#             print("#AT LEAST AS GOOD: {}/{}".format(alag, data.shape[0]), file=dest)
#             print("%AT LEAST AS GOOD: {}".format(100 * alag/data.shape[0]), file=dest)         

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

####################################################################
# 1. How effective were the task prio phases compared to random?
####################################################################

# for nb in nbs:
#     for s in ngpus:
#         # Get specific data.
#         data = df.loc[(df['NB'] == nb) & (df['s'] == s)] 
        
#         # Best and worst upward ranking performances.
#         u_bests = data.loc[:, ["{}-U".format(avg) for avg in u_avgs]].min(axis=1) / data["MLB"]
#         u_worsts = data.loc[:, ["{}-U".format(avg) for avg in u_avgs]].max(axis=1) / data["MLB"]
        
#         # Best and worst optimistic cost performances.
#         o_bests = data.loc[:, ["{}-O".format(avg) for avg in o_avgs]].min(axis=1) / data["MLB"]
#         o_worsts = data.loc[:, ["{}-O".format(avg) for avg in o_avgs]].max(axis=1) / data["MLB"]              
        
#         fig = plt.figure(dpi=400)
#         ax1 = fig.add_subplot(111)
#         ax1.fill_between(ntiles, u_bests, u_worsts, color='#E24A33', alpha=0.3, label="UPWARD RANK")
#         ax1.fill_between(ntiles, o_bests, o_worsts, color='#348ABD', alpha=0.3, label="OPTIMISTIC COSTS")
#         ax1.plot(ntiles, data["RND"] / data["MLB"], color='black', label="RANDOM")
        
#         plt.minorticks_on()
#         plt.grid(True, linestyle='-', axis='y', which='major')
#         plt.grid(True, linestyle=':', axis='y', which='minor')
#         plt.grid(True, linestyle='-', axis='x', which='major')
#         plt.grid(True, linestyle=':', axis='x', which='minor')
                
#         # plt.yscale('log')
#         if nb == 1024:
#             ax1.set_xlabel("N", labelpad=5)
#         if s == 1:
#             ax1.set_ylabel("SCHEDULE LENGTH RATIO", labelpad=5)
#         if s == 1 and nb == 128:
#             ax1.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white') 
            
#         ax1.tick_params(axis='x', which='minor', bottom=False)
#         ax1.tick_params(axis='y', which='minor', left=False)
        
#         plt.savefig('{}/prio_chol_slr_s{}_nb{}'.format(plot_path, s, nb), bbox_inches='tight') 
#         plt.close(fig) 

#####################################################################
# 2. Mean PD.
#####################################################################

###############################################
# A. According to s and nb.
###############################################

def add_patch(legend):
    ax = legend.axes

    handles, labels = ax.get_legend_handles_labels()
    handles += [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 3
    labels += list('{} = {}'.format(m, round(mpd[m], 3)) for m in sort) 

    legend._legend_box = None
    legend._init_legend_box(handles, labels)
    legend._set_loc(legend._loc)
    legend.set_title(legend.get_title().get_text())

for nb in nbs:
    for s in ngpus:
        # Get specific data.
        data = df.loc[(df['NB'] == nb) & (df['s'] == s)] 
        bests = data.loc[:, all_avgs].min(axis=1) 
        u_data = list((100*(data["{}-U".format(avg)] - bests)/bests).mean() for avg in u_avgs)
        check = lambda avg : (100*(data["{}-O".format(avg)] - bests)/bests).mean() if avg not in ["SW", "SB", "SHM", "SGM"] else 0.0
        o_data = list(check(avg) for avg in u_avgs)   
        
        mpd = {method : (100*(data[method] - bests)/bests).mean() for method in all_avgs}   
        # Sort methods to identify three best.
        sort = sorted(mpd, key=mpd.get)[:3]
              
        x = np.arange(len(u_avgs))
        width = 0.4
        
        fig, ax = plt.subplots(dpi=400)
        rects1 = ax.bar(x - width/2, u_data, width, label='UPWARD RANK', color='#E24A33')
        rects2 = ax.bar(x + width/2, o_data, width, label='OPTIMISTIC COSTS', color='#348ABD')
        
        plt.minorticks_on()
        plt.grid(True, linestyle='-', axis='y', which='major')
        plt.grid(True, linestyle=':', axis='y', which='minor')
        plt.grid(False, axis='x')
        
        # Get the three best and label them.
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 3
        labels = list('{} = {}'.format(m, round(mpd[m], 3)) for m in sort)    
        ax.legend(handles, labels, handlelength=0, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white', framealpha=1.0)   
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        if s == 1:
            ax.set_ylabel("MEAN PERCENTAGE DEGRADATION", labelpad=5)
        ax.set_xticks(x)
        ax.set_xticklabels(u_avgs)
        ax.tick_params(axis='x', which='minor', bottom=False)
        if s == 1 and nb == 128:
            lgd = ax.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white')
            add_patch(lgd)
                        
        fig.tight_layout()    
        plt.savefig('{}/prio_chol_mpd_s{}_nb{}'.format(plot_path, s, nb), bbox_inches='tight') 
        plt.close(fig) 

###############################################
# B. Entire set.
###############################################

# data = df
# bests = data.loc[:, all_avgs].min(axis=1) 
# u_data = list((100*(data["{}-U".format(avg)] - bests)/bests).mean() for avg in u_avgs)
# check = lambda avg : (100*(data["{}-O".format(avg)] - bests)/bests).mean() if avg not in ["SW", "SB", "SHM", "SGM"] else 0.0
# o_data = list(check(avg) for avg in u_avgs)   
      
# x = np.arange(len(u_avgs))
# width = 0.4

# fig, ax = plt.subplots(dpi=400)
# rects1 = ax.bar(x - width/2, u_data, width, label='UPWARD RANK', color='#E24A33')
# rects2 = ax.bar(x + width/2, o_data, width, label='OPTIMISTIC COSTS', color='#348ABD')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel("MEAN PERCENTAGE DEGRADATION", labelpad=5)
# ax.set_xticks(x)
# ax.set_xticklabels(u_avgs)
# ax.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white')

# # ax.bar_label(rects1, label_type='edge', rotation=90, padding=3)
# # ax.bar_label(rects2, label_type='edge', rotation=90, padding=3)
        
# fig.tight_layout()    
# plt.savefig('{}/prio_chol_mpd'.format(plot_path), bbox_inches='tight') 
# plt.close(fig) 

#####################################################################
# 3. Best instances.
#####################################################################

###############################################
# A. Entire set.
###############################################

# data = df
# bests = data.loc[:, all_avgs].min(axis=1) 
# u_data = list( ((abs(bests - data["{}-U".format(avg)])).values == 0.0).sum() for avg in u_avgs) 
# check = lambda avg : ((abs(bests - data["{}-O".format(avg)])).values == 0.0).sum() if avg not in ["SW", "SB", "SHM", "SGM"] else 0.0
# o_data = list(check(avg) for avg in u_avgs)   
      
# x = np.arange(len(u_avgs))
# width = 0.4

# fig, ax = plt.subplots(dpi=400)
# rects1 = ax.bar(x - width/2, u_data, width, label='UPWARD RANK', color='#E24A33')
# rects2 = ax.bar(x + width/2, o_data, width, label='OPTIMISTIC COSTS', color='#348ABD')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel("# BESTS (OUT OF 40)", labelpad=5)
# ax.set_xticks(x)
# ax.set_xticklabels(u_avgs)
# # ax.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white')

# # ax.bar_label(rects1, label_type='edge', rotation=90, padding=3)
# # ax.bar_label(rects2, label_type='edge', rotation=90, padding=3)
        
# fig.tight_layout()    
# plt.savefig('{}/prio_chol_bests'.format(plot_path), bbox_inches='tight') 
# plt.close(fig)  


