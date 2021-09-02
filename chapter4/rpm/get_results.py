#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare RPM variants for Cholesky DAGs.
"""

import dill
import pandas as pd
import numpy as np
from itertools import product
from timeit import default_timer as timer
from scipy.stats import ks_2samp
# Import statement may be necessary (because objects saved by reference). 
import sys
sys.path.append("../")

# =============================================================================
# Get results.
# =============================================================================

data = []
ntiles = list(range(5, 51, 5))
for N, nb, s in product(ntiles, [128, 1024], [1, 4]): 
    chol_load_path = '../chol_graphs/nb{}s{}'.format(nb, s)
    with open('{}/{}.dill'.format(chol_load_path, N), 'rb') as file:
        G = dill.load(file)
        
    with open('../empirical/chol/nb{}s{}/data/gamma/{}.dill'.format(nb, s, N), 'rb') as file:
        ref = dill.load(file)  
    
    graph_data = {"n" : G.size, "N" : N, "nb" : nb, "s" : s}
    
    for samples in [10, 100]:
        
        start = timer()
        paths, E = G.monte_carlo_paths(samples=samples, dist="U")      
        mc_id_time = timer() - start
        graph_data["MC{} PATH TIME".format(samples)] = mc_id_time
        
        graph_data["MC{} NPATHS".format(samples)] = len(paths)
        graph_data["MC{} MU".format(samples)] = np.mean(E)
        graph_data["MC{} VAR".format(samples)] = np.var(E)
        ks, _ = ks_2samp(ref, E)
        graph_data["MC{} KS".format(samples)] = ks 
        # Have timed elsewhere so no need for MC10/100.
        
        # Independent max.
        start = timer()
        PN = G.path_max(list(paths), samples=1000, correlations=False)
        nc_max_time = timer() - start        
        graph_data["SIM{}-I MU".format(samples)] = np.mean(PN)
        graph_data["SIM{}-I VAR".format(samples)] = np.var(PN)
        ks, _ = ks_2samp(ref, PN)
        graph_data["SIM{}-I KS".format(samples)] = ks
        graph_data["SIM{}-I TIME".format(samples)] = mc_id_time + nc_max_time
        
        # Correlated max.
        start = timer()
        PC = G.path_max(list(paths), samples=1000, correlations=True)
        corr_max_time = timer() - start        
        graph_data["SIM{} MU".format(samples)] = np.mean(PC)
        graph_data["SIM{} VAR".format(samples)] = np.var(PC)
        ks, _ = ks_2samp(ref, PC)
        graph_data["SIM{} KS".format(samples)] = ks
        graph_data["SIM{} TIME".format(samples)] = mc_id_time + corr_max_time        
               
        # K dominant paths.
        start = timer()
        kpaths = G.get_kdominant_paths(k=samples)
        kpath_time = timer() - start
        graph_data["K{} PATH TIME".format(samples)] = kpath_time 
        
        # Independent max.        
        start = timer()
        PN = G.path_max(kpaths, samples=1000, correlations=False)
        nc_max_time = timer() - start    
        graph_data["DOM{}-I MU".format(samples)] = np.mean(PN)
        graph_data["DOM{}-I VAR".format(samples)] = np.var(PN)
        ks, _ = ks_2samp(ref, PN)
        graph_data["DOM{}-I KS".format(samples)] = ks
        graph_data["DOM{}-I TIME".format(samples)] = kpath_time + nc_max_time
        
        # Correlated max.
        start = timer()
        PC = G.path_max(kpaths, samples=1000, correlations=True)
        corr_max_time = timer() - start
        graph_data["DOM{} MU".format(samples)] = np.mean(PC)
        graph_data["DOM{} VAR".format(samples)] = np.var(PC)
        ks, _ = ks_2samp(ref, PC)
        graph_data["DOM{} KS".format(samples)] = ks
        graph_data["DOM{} TIME".format(samples)] = kpath_time + corr_max_time          
        
    # Save the data.
    data.append(graph_data)

# # Save the dataframe. (Commented out by default.)
# df = pd.DataFrame(data)  
# df.to_csv('rpm.csv', encoding='utf-8', index=False)