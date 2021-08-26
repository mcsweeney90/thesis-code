#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPM variants for Cholesky DAGs.
"""

import dill
import pandas as pd
import numpy as np
from itertools import product
from timeit import default_timer as timer
from scipy.stats import ks_2samp

import sys
sys.path.append("../")


# =============================================================================
# Rename columns to more sensible labels.
# TODO: use this to modify further below.
# =============================================================================
df = pd.read_csv('rpm.csv')
d = {"M10-N" : "SIM10-I", "M10-C" : "SIM10", "K10-N" : "DOM10-I", "K10-C" : "DOM10",
     "M100-N" : "SIM100-I", "M100-C" : "SIM100", "K100-N" : "DOM100-I", "K100-C" : "DOM100"}
cols = {}
for m, v in d.items():
    for stat in ["MU", "VAR", "KS", "TIME"]:
        cols["{} {}".format(m, stat)] = "{} {}".format(v, stat)
cols["K10 PATH TIME"] = "DOM10 PATH TIME"
cols["K100 PATH TIME"] = "DOM100 PATH TIME"    
df.rename(columns=cols, inplace=True)
df.to_csv('rpm2.csv', encoding='utf-8', index=False)

# =============================================================================
# Get results.
# =============================================================================

# data = []
# ntasks = list(range(5, 51, 5))
# for nt, nb, s in product(ntasks, [128, 1024], [1, 4]): 
#     chol_load_path = '../chol_graphs/nb{}s{}'.format(nb, s)
#     with open('{}/{}.dill'.format(chol_load_path, nt), 'rb') as file:
#         G = dill.load(file)
        
#     with open('../empirical/chol/nb{}s{}/data/gamma/{}.dill'.format(nb, s, nt), 'rb') as file:
#         ref = dill.load(file)  
    
#     graph_data = {"n" : G.size, "nt" : nt, "nb" : nb, "s" : s}
    
#     for samples in [10, 100]:
        
#         start = timer()
#         paths, E = G.monte_carlo_paths(samples=samples, dist="U")      
#         mc_id_time = timer() - start
#         graph_data["MC{} PATH TIME".format(samples)] = mc_id_time
        
#         graph_data["MC{} NPATHS".format(samples)] = len(paths)
#         graph_data["MC{} MU".format(samples)] = np.mean(E)
#         graph_data["MC{} VAR".format(samples)] = np.var(E)
#         ks, _ = ks_2samp(ref, E)
#         graph_data["MC{} KS".format(samples)] = ks 
#         # Have timed elsewhere so no need for MC10/100.
        
#         start = timer()
#         PC = G.path_max(list(paths), samples=1000, correlations=True)
#         corr_max_time = timer() - start
        
#         start = timer()
#         PN = G.path_max(list(paths), samples=1000, correlations=False)
#         nc_max_time = timer() - start
        
#         # MC-based RPM.
#         for y, P, time in [("N", PN, nc_max_time), ("C", PC, corr_max_time)]:
#             graph_data["M{}-{} MU".format(samples, y)] = np.mean(P)
#             graph_data["M{}-{} VAR".format(samples, y)] = np.var(P)
#             ks, _ = ks_2samp(ref, P)
#             graph_data["M{}-{} KS".format(samples, y)] = ks
#             graph_data["M{}-{} TIME".format(samples, y)] = mc_id_time + time 
        
#         # K dominant paths.
#         start = timer()
#         kpaths = G.get_kdominant_paths(k=samples)
#         kpath_time = timer() - start
#         graph_data["K{} PATH TIME".format(samples)] = kpath_time 
        
#         start = timer()
#         PC = G.path_max(kpaths, samples=1000, correlations=True)
#         corr_max_time = timer() - start
        
#         start = timer()
#         PN = G.path_max(kpaths, samples=1000, correlations=False)
#         nc_max_time = timer() - start        
        
#         for y, P, time in [("N", PN, nc_max_time), ("C", PC, corr_max_time)]:
#             graph_data["K{}-{} MU".format(samples, y)] = np.mean(P)
#             graph_data["K{}-{} VAR".format(samples, y)] = np.var(P)
#             ks, _ = ks_2samp(ref, P)
#             graph_data["K{}-{} KS".format(samples, y)] = ks
#             graph_data["K{}-{} TIME".format(samples, y)] = kpath_time + time 
        
#     # Save the data.
#     data.append(graph_data)

# # Save the dataframe.
# df = pd.DataFrame(data)  
# df.to_csv('rpm.csv', encoding='utf-8', index=False)