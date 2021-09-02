#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get all results (CLT-based and MC/RPM) for STG set. 
Took about 11 hours on my machine.
"""

import dill, os
import pandas as pd
import numpy as np
from timeit import default_timer as timer
from math import sqrt
from scipy.stats import ks_2samp, kstest

import sys
sys.path.append("../")
from src import StochDAG

size = 100
dag_path = '../../graphs/STG/{}'.format(size)

runs = 10
covs = [0.01, 0.03, 0.1, 0.3]

# =============================================================================
# Get data.
# =============================================================================
    
data = [] 
for dname in os.listdir(dag_path):   
    
    # Load the DAG topology.
    with open('{}/{}'.format(dag_path, dname), 'rb') as file:
        T = dill.load(file)
    # Convert to DAG object.
    G = StochDAG(T)              
    
    for cov in covs:
        for run in range(runs): 

            # Set the weights.                  
            G.set_random_weights(exp_cov=cov)
            graph_data = {"DAG" : dname[:-5], "COV" : cov, "RUN" : run}  
                                            
            # Get the reference solution. 
            start = timer()
            ref = G.monte_carlo(samples=100000, dist="GAMMA")
            elapsed = timer() - start
            mu = np.mean(ref)
            graph_data["REF MU"] = mu
            var = np.var(ref)
            graph_data["REF VAR"] = var
            graph_data["REF TIME"] = elapsed
            
            # CPM.
            start = timer()
            cpm = G.CPM()
            elapsed = timer() - start
            graph_data["CPM MU"] = cpm
            graph_data["CPM TIME"] = elapsed
            
            # Kamburowski's bounds.
            start = timer()
            lm, um, ls, us = G.kamburowski()
            elapsed = timer() - start
            graph_data["K LOWER MU"] = lm
            graph_data["K UPPER MU"] = um
            graph_data["K LOWER VAR"] = ls
            graph_data["K UPPER VAR"] = us
            graph_data["K TIME"] = elapsed
            
            # Sculli's method.
            start = timer()
            S = G.sculli()
            elapsed = timer() - start
            s_mu, s_var = S.mu, S.var
            graph_data["SCULLI MU"] = s_mu
            graph_data["SCULLI VAR"] = s_var
            ks, _ = kstest(ref, cdf='norm', args=(s_mu, sqrt(s_var)))
            graph_data["SCULLI KS"] = ks
            graph_data["SCULLI TIME"] = elapsed
            
            # CorLCA.
            start = timer()
            C = G.corLCA()
            elapsed = timer() - start
            c_mu, c_var = C.mu, C.var
            graph_data["CORLCA MU"] = c_mu
            graph_data["CORLCA VAR"] = c_var
            ks, _ = kstest(ref, cdf='norm', args=(c_mu, sqrt(c_var)))
            graph_data["CORLCA KS"] = ks
            graph_data["CORLCA TIME"] = elapsed
            
            # RPM.
            for samples in [10, 100]:   
                
                start = timer()
                _ = G.monte_carlo(samples=samples, dist="U")
                elapsed = timer() - start
                graph_data["MC{} TIME".format(samples)] = elapsed
                
                start = timer()
                paths, E = G.monte_carlo_paths(samples=samples, dist="U")      
                mc_id_time = timer() - start
                graph_data["MC{} PATH TIME".format(samples)] = mc_id_time
                
                graph_data["MC{} NPATHS".format(samples)] = len(paths)
                graph_data["MC{} MU".format(samples)] = np.mean(E)
                graph_data["MC{} VAR".format(samples)] = np.var(E)
                ks, _ = ks_2samp(ref, E)
                graph_data["MC{} KS".format(samples)] = ks 
                
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
                        
            data.append(graph_data)  
            
# Save results. (Commented out by default.)
# df = pd.DataFrame(data)  
# df.to_csv('stg.csv', encoding='utf-8', index=False)
    