#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing...
"""

import dill
import numpy as np
import networkx as nx
from timeit import default_timer as timer
from src import RV, StochDAG, get_StochDAG
from scipy.stats import kstest, ks_2samp
from networkx.utils import pairwise

# nb = 1024
# s = 1
# chol_dag_path = 'chol_graphs/nb{}s{}'.format(nb, s)
# n_tasks = list(range(5, 51, 5))

# for nt in n_tasks:
#     with open('{}/{}.dill'.format(chol_dag_path, nt), 'rb') as file:
#         G = dill.load(file)      
#     print("\n\n\n------------------------------------------------------------------")
#     print("GRAPH SIZE: {}".format(G.size)) 
#     print("------------------------------------------------------------------") 
    
#     with open('empirical/chol/nb{}s{}/data/gamma/{}.dill'.format(nb, s, nt), 'rb') as file:
#         ref = dill.load(file) 
    
#     paths = G.get_kdominant_paths(k=10)
#     P = G.path_max(paths, samples=100000, correlations=False)
#     ks, _ = ks_2samp(ref, P)
#     print("RPM KS: {}".format(ks))
    
    
    
#     paths = G.get_dominating_paths(p=5)
#     print(len(paths))
    # for path in paths:
    #     print(G.path_length(path))
    
    
    
    # A = G.get_scalar_graph(aoa=True, johnson=False)  
    # paths = A.dag_klongest_paths(100)
    # for path in paths:
    #     print(G.path_length(path))
    
       
    # paths, E = G.monte_carlo_paths(samples=100000, dist="U")    
    # print(len(paths))
    # print(paths)
    # print("Number of paths: {}".format(len(paths)))
    
    # trad_mu = np.mean(E)
    # trad_var = np.var(E)
    # print("\nTrad. MC mean: {}".format(trad_mu))
    # print("Trad. MC var: {}".format(trad_var))   
    # ks, _ = ks_2samp(ref, E)
    # print("Trad. MC KS: {}".format(ks))
    
    # P = G.path_max(paths, samples=100000)
    # print("\nRPM mean: {}".format(np.mean(P)))
    # print("RPM var: {}".format(np.var(P)))
    # ks, _ = ks_2samp(ref, P)
    # print("RPM KS: {}".format(ks))
    
    

