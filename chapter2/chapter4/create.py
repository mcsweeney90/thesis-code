#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For Chapter 4.
Compute HEFT schedules for Cholesky graphs and save corresponding schedule graphs.
"""

import dill, pathlib
from statistics import variance
import sys
sys.path.append('../') 
from src import heft

####################################################################################################

dag_path = '../chol_graphs/'
ntiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
nbs = [128, 1024]
r = 32
ngpus = [1, 4]

timings_path = '../chol_graphs/timings/' 
with open('{}/timings.dill'.format(timings_path), 'rb') as file:
    timings = dill.load(file)
runs = len(timings[128]["G"]["c"]) 

####################################################################################################

# =============================================================================
# Create DAGs.
# =============================================================================

for nb in nbs:
    print("\nTile size: {}".format(nb))
    for nt in ntiles:  
        print("DAG: {}".format(nt))
        # Load the DAG.
        with open('{}/nb{}/{}.dill'.format(dag_path, nb, nt), 'rb') as file:
            G = dill.load(file)  
                    
        for s in ngpus:                        
            # Compute HEFT schedule.
            _, pi = heft(G, r, s, return_schedule=True)
            
            # Convert pi to a schedule graph.
            where_scheduled = {}
            for w, load in pi.items():
                for t in list(c[0] for c in load):
                    where_scheduled[t] = w 
            # Construct graph topology.
            S = G.graph.__class__()
            S.add_nodes_from(G.graph)
            S.add_edges_from(G.graph.edges)
            # Set the weights.
            for t in G.top_sort:
                proc_type = "c" if where_scheduled[t] < r else "g"
                task_type = t[0]
                mu = sum(timings[nb][task_type][proc_type])/runs
                var = variance(timings[nb][task_type][proc_type], xbar=mu)
                S.nodes[t]['weight'] = [mu, var]             
                
                for c in G.graph.successors(t):
                    if (where_scheduled[c] == where_scheduled[t]) or (where_scheduled[t] < r and where_scheduled[c] < r):
                        S[t][c]['weight'] = 0.0
                    else:       
                        child_type = c[0]
                        cmu = sum(timings[nb][child_type]["d"])/runs
                        cvar = variance(timings[nb][child_type]["d"], xbar=cmu)
                        S[t][c]['weight'] = [cmu, cvar] 
                    
                # Add disjunctive edge if necessary.
                idx = list(r[0] for r in pi[where_scheduled[t]]).index(t)
                if idx > 0:
                    d = pi[where_scheduled[t]][idx - 1][0]
                    if not S.has_edge(d, t):
                        S.add_edge(d, t)
                        S[d][t]['weight'] = 0.0
            # Save the graph.
            dag_save_path = '../../chapter4/chol_graphs/ORIGnb{}s{}/'.format(nb, s)
            pathlib.Path(dag_save_path).mkdir(parents=True, exist_ok=True)
            with open('{}/{}.dill'.format(dag_save_path, nt), 'wb') as handle:
                dill.dump(S, handle)
            
                        
            

