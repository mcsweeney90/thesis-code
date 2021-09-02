#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare similarity of empirical distributions for STG set.
Took about 18 hours to run on Matt's machine.
Decided against saving DAGs since > 200MB, which means empirical dists needed to be generated again by scripts for comparison. 
"""

import dill, os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

import sys
sys.path.append("../../")
from src import StochDAG

size = 100
dag_path = '../../../graphs/STG/{}'.format(size)

runs = 10
covs = [0.01, 0.03, 0.1, 0.3]
dists = ["NORMAL", "GAMMA", "UNIFORM"]

# =============================================================================
# Get data.
# =============================================================================    
    
data = [] 
for dname in os.listdir(dag_path):     
    # print(dname)
    
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
                                            
            # Get the empirical distributions.
            emps = {}
            for dist in dists:
                emps[dist] = G.monte_carlo(samples=100000, dist=dist)
            
            # Do the analysis.
            for dist in dists:
                mu = np.mean(emps[dist])
                graph_data["{} MU".format(dist)] = mu
                var = np.var(emps[dist])
                graph_data["{} VAR".format(dist)] = var   
                
            # Compare the distributions (KS statistics).
            ks, _ = ks_2samp(emps["NORMAL"], emps["GAMMA"])
            graph_data["N-G KS"] = ks
            
            ks, _ = ks_2samp(emps["NORMAL"], emps["UNIFORM"])
            graph_data["N-U KS"] = ks
            
            ks, _ = ks_2samp(emps["GAMMA"], emps["UNIFORM"])
            graph_data["G-U KS"] = ks
            
            data.append(graph_data)  

df = pd.DataFrame(data)  
df.to_csv('emp_stg.csv', encoding='utf-8', index=False)
    
    
    