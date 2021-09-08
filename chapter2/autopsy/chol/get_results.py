#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autopsy method for Cholesky graphs.
NOTE: to avoid overwriting the data 'results.csv' that was used in thesis, have changed the name of save destination to 'new_results.csv'.    
"""

import dill
import pandas as pd
from timeit import default_timer as timer

import sys
sys.path.append('../../') 
from src import priority_scheduling

####################################################################################################

dag_path = '../../chol_graphs/'
ntiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
nbs = [128, 1024]
r = 32
ngpus = [1, 4]

# =============================================================================
# Generate results.
# =============================================================================

data = []
for nb in nbs:
    print("\nTile size: {}".format(nb))
    for N in ntiles:  
        print("DAG: {}".format(N))
        # Load the DAG.
        with open('{}/nb{}/{}.dill'.format(dag_path, nb, N), 'rb') as file:
            G = dill.load(file)  
            
        # Compute minimal serial time.
        mst = G.minimal_serial_time()
        
        for s in ngpus:
            
            graph_data = {"r" : r, "s" : s, "n" : G.size, "N" : N, "NB" : nb, "MST" : mst} 
                
            # Compute makespan lower bound.
            lb = G.makespan_lower_bound(r + s)
            graph_data["MLB"] = lb              
            
            # Compute standard upward ranks.
            U = G.get_upward_ranks(r, s, avg_type="M") 
            
            # HEFT.
            start = timer()
            heft, S = priority_scheduling(G, r, s, priorities=U, sel_policy="EFT", return_schedule=True)
            eft_time = timer() - start
            graph_data["HEFT"] = heft  
            
            # Autopsy method.
            start = timer()
            alpha = {}
            for worker, load in S.items():
                for block in load:
                    alpha[block[0]] = "c" if worker < r else "g"                    
            edge_delta = {"cc" : 0.0, "cg" : 1.0, "gc" : 1.0, "gg" : (s - 1)/s}
            ranks = {}
            backward_traversal = list(reversed(G.top_sort))
            for t in backward_traversal:
                a = alpha[t]
                ranks[t] = G.graph.nodes[t]['weight'][a]
                try:
                    ranks[t] += max(edge_delta[alpha[t] + alpha[c]]*G.graph[t][c]['weight'] + ranks[c] for c in G.graph.successors(t))
                except ValueError:
                    pass            
            aut = priority_scheduling(G, r, s, priorities=ranks, sel_policy="AMT", assignment=alpha)
            aut_time = timer() - start
            graph_data["AUT"] = aut
            graph_data["AUT PTI"] = 100*(aut_time/eft_time) # Percentage time increase.
        
            data.append(graph_data)

# Save the data. 
df = pd.DataFrame(data)  
df.to_csv('new_results.csv', encoding='utf-8', index=False) 