#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autopsy method for STG set.
"""

import dill, os
import pandas as pd
from timeit import default_timer as timer 

import sys
sys.path.append('../../') 
from src import DAG, priority_scheduling

####################################################################################################

size = 1000
dag_path = '../../../graphs/STG/{}'.format(size)
ccrs = [0.01, 0.1, 1.0, 10.0]
r = 32
ngpus = [1, 4]
runs = 10

# =============================================================================
# Generate results.
# =============================================================================

data = [] 
for dname in os.listdir(dag_path):  
    # Load the DAG topology.
    with open('{}/{}'.format(dag_path, dname), 'rb') as file:
        T = dill.load(file)
    # Convert to DAG object.
    G = DAG(T)
    
    for s in ngpus:
        for ccr in ccrs:
            for run in range(runs):  
                
                # Set the weights.    
                G.set_random_weights(r=r, s=s, ccr=ccr)    
                
                graph_data = {"n" : G.size, "r" : r, "s" : s, "CCR" : ccr, "DAG" : dname[:-5], "RUN" : run} 
                
                # Compute makespan lower bound.
                lb = G.makespan_lower_bound(r + s)
                graph_data["MLB"] = lb 
                
                # Minimal serial time.
                mst = G.minimal_serial_time()
                graph_data["MST"] = mst 
                
                # Compute standard upward ranks.
                U = G.get_upward_ranks(r, s, avg_type="M") 
                
                # HEFT.
                start = timer()
                heft, S = priority_scheduling(G, r, s, priorities=U, sel_policy="EFT", return_schedule=True)
                eft_time = timer() - start
                graph_data["HEFT"] = heft  
                
                # Autopsy.
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

# Save the dataframe. Commented out by default.
# df = pd.DataFrame(data)  
# df.to_csv('stg{}.csv'.format(size), encoding='utf-8', index=False)