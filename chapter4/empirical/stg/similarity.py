#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze similarity of empirical distributions for STG set.
NOTE: to avoid overwriting the data 'similarity.csv' that was used in thesis, have changed the name of save destination to 'new_similarity.csv'. 
ALSO:
    1. Took about 18 hours to run on my machine.
    2. Decided against saving DAGs since > 200MB, which means empirical dists needed to be generated again by scripts for comparison. 
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
            G.set_random_weights(mu_cov=cov)
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
# Save dataframe.
df.to_csv('new_similarity.csv', encoding='utf-8', index=False)

# =============================================================================
# Human-readable summary.
# =============================================================================
   
with open("summary.txt", "w") as dest:
    print("SIMILARITY OF EMPIRICAL LONGEST PATH DISTRIBUTIONS WITH DIFFERENT WEIGHT DISTRIBUTIONS.", file=dest) 
    print("EMPIRICAL DISTRIBUTIONS GENERATED THROUGH {} REALIZATIONS OF GRAPH WEIGHTS.".format(runs), file=dest)          
                        
    print("\n\n\n------------------------------------------------------------------", file=dest)
    print("% RELATIVE ERROR IN MEAN (AVG, MAX)", file=dest)
    print("------------------------------------------------------------------", file=dest)    
    ng = 100*abs(df["NORMAL MU"] - df["GAMMA MU"]) / df["GAMMA MU"]
    print("NORMAL-GAMMA: ({}, {})".format(ng.mean(), ng.max()), file=dest)
    nu = 100*abs(df["NORMAL MU"] - df["UNIFORM MU"]) / df["UNIFORM MU"]
    print("NORMAL-UNIFORM: ({}, {})".format(nu.mean(), nu.max()), file=dest)
    ug = 100*abs(df["UNIFORM MU"] - df["GAMMA MU"]) / df["GAMMA MU"]
    print("UNIFORM-GAMMA: ({}, {})".format(ug.mean(), ug.max()), file=dest)

    print("\n\n\n------------------------------------------------------------------", file=dest)
    print("% RELATIVE ERROR IN VARIANCE (AVG, MAX)", file=dest)
    print("------------------------------------------------------------------", file=dest)    
    ng = 100*abs(df["NORMAL VAR"] - df["GAMMA VAR"]) / df["GAMMA VAR"]
    print("NORMAL-GAMMA: ({}, {})".format(ng.mean(), ng.max()), file=dest)
    nu = 100*abs(df["NORMAL VAR"] - df["UNIFORM VAR"]) / df["UNIFORM VAR"]
    print("NORMAL-UNIFORM: ({}, {})".format(nu.mean(), nu.max()), file=dest)
    ug = 100*abs(df["UNIFORM VAR"] - df["GAMMA VAR"]) / df["GAMMA VAR"]
    print("UNIFORM-GAMMA: ({}, {})".format(ug.mean(), ug.max()), file=dest)
    
    print("\n\n\n------------------------------------------------------------------", file=dest)
    print("% RELATIVE ERROR IN STANDARD DEVIATION (AVG, MAX)", file=dest)
    print("------------------------------------------------------------------", file=dest)    
    ng = 100*abs(np.sqrt(df["NORMAL VAR"]) - np.sqrt(df["GAMMA VAR"])) / np.sqrt(df["GAMMA VAR"])
    print("NORMAL-GAMMA: ({}, {})".format(ng.mean(), ng.max()), file=dest)
    nu = 100*abs(np.sqrt(df["NORMAL VAR"]) - np.sqrt(df["UNIFORM VAR"])) / np.sqrt(df["UNIFORM VAR"])
    print("NORMAL-UNIFORM: ({}, {})".format(nu.mean(), nu.max()), file=dest)
    ug = 100*abs(np.sqrt(df["UNIFORM VAR"]) - np.sqrt(df["GAMMA VAR"])) / np.sqrt(df["GAMMA VAR"])
    print("UNIFORM-GAMMA: ({}, {})".format(ug.mean(), ug.max()), file=dest)
    
    print("\n\n\n------------------------------------------------------------------", file=dest)
    print("KS STATISTICS (AVG, MAX)", file=dest)
    print("------------------------------------------------------------------", file=dest)    
    print("NORMAL-GAMMA: ({}, {})".format(df["N-G KS"].mean(), df["N-G KS"].max()), file=dest)
    print("NORMAL-UNIFORM: ({}, {})".format(df["N-U KS"].mean(), df["N-U KS"].max()), file=dest)
    print("UNIFORM-GAMMA: ({}, {})".format(df["G-U KS"].mean(), df["G-U KS"].max()), file=dest)

    
    
    